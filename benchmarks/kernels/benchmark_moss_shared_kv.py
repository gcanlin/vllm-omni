# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real-weight waveform checks and codec-only CUDA Graph microbench.

Uses synthetic fixed codes, not TTS quality or end-to-end throughput. No server
is started. Run with CUDA_VISIBLE_DEVICES selecting an otherwise idle GPU.
"""

import argparse
import copy
import json
from pathlib import Path

import torch
from safetensors import safe_open

from vllm_omni.model_executor.models.moss_tts.audio_tokenizer_v2 import (
    MHAState,
    MossAudioTokenizerConfig,
    MossAudioTokenizerModel,
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


def rms(out, ref):
    return ((out.float() - ref.float()).square().mean() / ref.float().square().mean().clamp_min(1e-12)).sqrt().item()


def kv_bytes(codec):
    return sum(
        m._streaming_state.kv_cache.cache.nbytes
        for m in codec._streaming_modules
        if isinstance(m._streaming_state, MHAState)
    )


def validate(base, candidate, capacity, chunks):
    control = copy.deepcopy(base)
    control.shared_decoder_kv = False
    for model in (base, candidate, control):
        model.initialize_decoder_state_pool(capacity, capacity)
    print(
        json.dumps(
            dict(
                kind="memory",
                capacity=capacity,
                baseline_kv_bytes=kv_bytes(base),
                candidate_kv_bytes=kv_bytes(candidate),
                baseline_offsets=list(base._decoder_slot_offsets.shape),
                candidate_offsets=list(candidate._decoder_slot_offsets.shape),
            )
        ),
        flush=True,
    )
    slots = torch.tensor([1, 0, capacity, capacity + 1], device="cuda")
    valid = torch.tensor([True, True, False, False], device="cuda")
    measurements = []
    for i in range(chunks):
        frames = 1 if i == 0 else 15
        slots[:2] = torch.tensor([i % 2, 1 - i % 2], device="cuda")
        codes = torch.randint(1024, (12, 4, frames), device="cuda")
        lengths = torch.full((4,), frames, device="cuda", dtype=torch.long) * valid
        ref, ref_lengths = base.decode_streaming_tensors(codes, lengths, slots, valid)
        same, _ = control.decode_streaming_tensors(codes, lengths, slots, valid)
        out, out_lengths = candidate.decode_streaming_tensors(codes, lengths, slots, valid)
        torch.testing.assert_close(ref_lengths, out_lengths, rtol=0, atol=0)
        assert torch.isfinite(out).all()
        entry = dict(kind="waveform", chunk=i, candidate_rms=rms(out[:2], ref[:2]), control_rms=rms(same[:2], ref[:2]))
        print(json.dumps(entry), flush=True)
        measurements.append(entry)
        for model in (base, candidate, control):
            if i == 4:
                model.reset_decoder_state_slots(slots[:1])
        for module in candidate._streaming_modules:
            if isinstance(module._streaming_state, MHAState):
                assert module._streaming_state.kv_cache.cache[:, capacity].count_nonzero() == 0
        assert candidate._decoder_slot_offsets[:, capacity].count_nonzero() == 0
    for model in (base, candidate, control):
        model.close_decoder_state_pool()
    # Diagnostic threshold, not a perceptual-quality acceptance criterion.
    assert max(x["candidate_rms"] for x in measurements) < 0.03


def capture(codec, batch, frames, compiled):
    codes = torch.randint(1024, (12, batch, frames), device="cuda")
    lengths = torch.full((batch,), frames, device="cuda", dtype=torch.long)
    slots = torch.arange(batch, device="cuda")
    valid = torch.ones(batch, device="cuda", dtype=torch.bool)
    decode = (
        torch.compile(codec.decode_streaming_tensors, fullgraph=True) if compiled else codec.decode_streaming_tensors
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            decode(codes, lengths, slots, valid)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = decode(codes, lengths, slots, valid)
    # CUDA Graph records addresses, not Python ownership of external inputs.
    # Retain every input (and the compiled callable) until the graph is deleted.
    return graph, (output, codes, lengths, slots, valid, decode)


def time_graph(graph, repeats):
    for _ in range(5):
        graph.replay()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--capacity", type=int, default=32)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--frames", type=int, nargs="+", default=[1, 15])
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--chunks", type=int, default=10)
    parser.add_argument("--compiled", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.capacity < max(4, *args.batches) or args.chunks < 1 or args.repeats < 1:
        parser.error("capacity must cover all batches and validation; chunks/repeats must be positive")
    torch.manual_seed(17)
    base = load_codec(args.checkpoint)
    candidate = copy.deepcopy(base)
    base.shared_decoder_kv = False
    candidate.shared_decoder_kv = True
    print(
        json.dumps(
            dict(
                kind="environment",
                device=torch.cuda.get_device_name(),
                torch=torch.__version__,
                compiled=args.compiled,
                checkpoint=args.checkpoint,
            )
        ),
        flush=True,
    )
    validate(base, candidate, args.capacity, args.chunks)
    if args.validate_only:
        return
    for model in (base, candidate):
        model.initialize_decoder_state_pool(args.capacity, args.capacity)
    for batch in args.batches:
        for frames in args.frames:
            baseline_graph, baseline_out = capture(base, batch, frames, args.compiled)
            candidate_graph, candidate_out = capture(candidate, batch, frames, args.compiled)
            left, right = [], []
            for round_id in range(4):
                order = [(baseline_graph, left), (candidate_graph, right)]
                for graph, times in order if round_id % 2 == 0 else reversed(order):
                    times.append(time_graph(graph, args.repeats))
            print(
                json.dumps(
                    dict(
                        kind="microbench",
                        batch=batch,
                        frames=frames,
                        baseline_ms=left,
                        candidate_ms=right,
                        speedup=sum(left) / sum(right),
                    )
                ),
                flush=True,
            )
            del baseline_graph, candidate_graph, baseline_out, candidate_out
    for model in (base, candidate):
        model.close_decoder_state_pool()


if __name__ == "__main__":
    main()
