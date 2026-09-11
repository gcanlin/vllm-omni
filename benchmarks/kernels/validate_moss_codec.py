# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Validate complete streaming codec waveforms with real weights and fixed codes.

The encoder is removed before materializing weights; no serving state is used.
Synthetic codes check numerical equivalence, not perceptual quality.
"""

import argparse
import copy
import json
from pathlib import Path

import torch
from safetensors import safe_open

from vllm_omni.model_executor.models.moss_tts.audio_tokenizer_v2 import (
    MossAudioTokenizerConfig,
    MossAudioTokenizerModel,
    MossAudioTokenizerMultiheadAttention,
)


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


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--mode", choices=["pack", "batching", "reference"], default="pack")
    parser.add_argument("--chunks", type=int, nargs="+", default=[1, 15, 15, 15, 15, 15, 15, 15, 15, 15])
    parser.add_argument("--dump", type=str, default=None, help="Optional path to save per-chunk fast outputs as .pt")
    args = parser.parse_args()
    torch.manual_seed(17)
    base = load_codec(args.checkpoint)
    fast = copy.deepcopy(base)
    for model in (base, fast):
        model.initialize_decoder_state_pool(args.batch + 2)
        for module in model.modules():
            if isinstance(module, MossAudioTokenizerMultiheadAttention):
                module._streaming_state.kv_cache._use_kv_pack = model is fast and args.mode == "pack"
    slots = torch.arange(args.batch, device="cuda") + 1
    valid = torch.ones(args.batch, device="cuda", dtype=torch.bool)
    dumped = []
    for index, frames in enumerate(args.chunks):
        codes = torch.randint(1024, (12, args.batch, frames), device="cuda")
        lengths = torch.full((args.batch,), frames, device="cuda", dtype=torch.long)
        if args.mode == "batching" and frames == 15:
            lengths[0] = 2
            codes[:, 0, 2:] = 0
            separate = [
                base.decode_streaming_tensors(
                    codes[:, row : row + 1], lengths[row : row + 1], slots[row : row + 1], valid[row : row + 1]
                )
                for row in range(args.batch)
            ]
            ref, ref_lengths = torch.cat([x[0] for x in separate]), torch.cat([x[1] for x in separate])
        else:
            ref, ref_lengths = base.decode_streaming_tensors(codes, lengths, slots, valid)
        out, out_lengths = fast.decode_streaming_tensors(codes, lengths, slots, valid)
        if args.dump:
            dumped.append(out.float().cpu())
        torch.testing.assert_close(ref_lengths, out_lengths)
        delta = out.float() - ref.float()
        rms = (delta.square().mean() / ref.float().square().mean().clamp_min(1e-12)).sqrt().item()
        maximum = delta.abs().max().item()
        print(json.dumps(dict(chunk=index, frames=frames, relative_rms=rms, max_absolute=maximum)), flush=True)
        assert torch.isfinite(out).all()
        # "batching" is an old-vs-old native batch-size sensitivity control,
        # not an altered arithmetic kernel. Report drift without claiming a
        # perceptual quality verdict from this waveform statistic alone.
        if args.mode != "batching":
            assert rms < 0.03
        torch.testing.assert_close(base._decoder_slot_offsets, fast._decoder_slot_offsets)
        if args.mode == "batching" and frames == 15:
            base.reset_decoder_state_slots(slots[:1])
            fast.reset_decoder_state_slots(slots[:1])
        if index == 4:
            # Exercise slot recycling with stale KV still resident.
            base.reset_decoder_state_slots(slots[:1])
            fast.reset_decoder_state_slots(slots[:1])
    if args.dump:
        torch.save(dumped, args.dump)


if __name__ == "__main__":
    main()
