# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real-weight multichunk waveform comparison against the merged baseline."""

import argparse
import copy
import json
from pathlib import Path

import torch
from safetensors import safe_open
from vllm.triton_utils import triton

from vllm_omni.model_executor.models.moss_tts import codec_gemm
from vllm_omni.model_executor.models.moss_tts.audio_tokenizer_v2 import (
    MossAudioTokenizerConfig,
    MossAudioTokenizerModel,
    MossAudioTokenizerMultiheadAttention,
)
from vllm_omni.model_executor.models.moss_tts.streaming_attention import _attention, masked_attention


def load_codec(path):
    config = MossAudioTokenizerConfig.from_pretrained(path)
    with torch.device("meta"):
        codec = MossAudioTokenizerModel(config)
    codec.encoder = torch.nn.ModuleList()
    assert not dict(codec.named_buffers()), "Initialize non-checkpoint buffers before using to_empty."
    codec.to_empty(device="cuda")
    params = dict(codec.named_parameters())
    loaded = set()
    mappings = [
        (".self_attn.in_proj.", ".self_attn.in_projs.0."),
        (".self_attn.out_proj.", ".self_attn.out_projs.0."),
        (".ffn.0.", ".linear1."),
        (".ffn.2.", ".linear2."),
        (".layer_scale_1.", ".ls1."),
        (".layer_scale_2.", ".ls2."),
        (".input_proj.", ".in_proj."),
        (".output_proj.", ".out_proj."),
    ]
    for shard in sorted(Path(path).glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as reader:
            for name in reader.keys():
                if name.startswith("encoder."):
                    continue
                target = name
                if target not in params:
                    for src, dst in mappings:
                        if src in name:
                            target = name.replace(src, dst)
                            break
                assert target in params, name
                tensor = reader.get_tensor(name)
                assert tensor.shape == params[target].shape, name
                params[target].copy_(tensor)
                loaded.add(target)
    assert loaded == set(params), set(params) - loaded
    codec.decoder.to(torch.bfloat16)
    codec.quantizer.build_decode_lut(12, dtype=torch.bfloat16)
    return codec.eval()


def baseline_attention(q, k, v, mask):
    b, h, t, d = q.shape
    out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    bm = 16 if t <= 32 else 64
    _attention[(triton.cdiv(t, bm), b * h)](
        q,
        k,
        v,
        mask,
        out,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        mask.stride(0),
        mask.stride(2),
        mask.stride(3),
        h,
        t,
        k.shape[2],
        d,
        bm,
        64,
        False,
        False,
        num_warps=8 if k.shape[2] <= 256 else 4,
        num_stages=2,
    )
    return out


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--chunks", type=int, default=10)
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()
    torch.manual_seed(17)
    base = load_codec(args.checkpoint)
    fast = copy.deepcopy(base)
    for model, attention in [(base, baseline_attention), (fast, masked_attention)]:
        model.initialize_decoder_state_pool(args.batch + 3)
        for module in model.modules():
            if isinstance(module, MossAudioTokenizerMultiheadAttention):
                module._streaming_attention = attention
    config = json.loads(Path(args.config).read_text())
    slots = torch.arange(1, args.batch + 1, device="cuda")
    valid = torch.ones(args.batch, device="cuda", dtype=torch.bool)
    for index in range(args.chunks):
        frames = 1 if index == 0 else 15
        codes = torch.randint(1024, (12, args.batch, frames), device="cuda")
        lengths = torch.full((args.batch,), frames, device="cuda", dtype=torch.long)
        codec_gemm.CONFIG = {}
        ref, rl = base.decode_streaming_tensors(codes, lengths, slots, valid)
        codec_gemm.CONFIG = config
        value, vl = fast.decode_streaming_tensors(codes, lengths, slots, valid)
        torch.testing.assert_close(rl, vl, atol=0, rtol=0)
        rel = ((ref.float() - value.float()).norm() / ref.float().norm()).item()
        print(
            json.dumps(
                dict(chunk=index, frames=frames, relative_rms=rel, max_absolute=(ref - value).abs().max().item())
            ),
            flush=True,
        )
        assert torch.isfinite(value).all() and rel < 0.03
        if index == 4:
            base.reset_decoder_state_slots(slots[:1])
            fast.reset_decoder_state_slots(slots[:1])


if __name__ == "__main__":
    main()
