# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare MOSS Local frame generation with real weights and synthetic inputs.

Includes the local transformer, twelve codebook heads and sampling. Excludes
the backbone, codec and serving; these are not end-to-end throughput numbers.
Export the unmodified main local-depth module with git show and pass it as
--baseline-source. Both variants retain torch.compile and CUDA Graphs.
"""

import argparse
import importlib.util
import json
import statistics
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn
from transformers import GPT2Config

from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_local_depth import MossTTSLocalDepthTransformer
from vllm_omni.platforms import current_omni_platform


def load_reference(source: Path):
    spec = importlib.util.spec_from_file_location("moss_local_baseline", source)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load baseline source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MossTTSLocalDepthTransformer


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 64, 128])
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(*args.batch_sizes, args.iterations, args.repeats) <= 0:
        parser.error("Batch sizes, iterations and repeats must be positive")
    config = json.loads((args.checkpoint / "config.json").read_text())
    hidden_size = config["qwen3_config"]["hidden_size"]
    n_vq, vocab = config["n_vq"], config["audio_vocab_size"]
    dtype = torch.bfloat16
    torch.manual_seed(42)
    # Avoid Dynamo fallback penalizing a baseline specialized by depth/batch.
    torch._dynamo.config.recompile_limit = max(128, len(args.batch_sizes) * n_vq + 8)
    torch._dynamo.config.accumulated_recompile_limit = 2048
    depth_config = GPT2Config(**config["gpt2_config"])
    baseline = load_reference(args.baseline_source)(depth_config, hidden_size)
    optimized = MossTTSLocalDepthTransformer(depth_config, hidden_size)
    embeddings = nn.ModuleList([nn.Embedding(vocab, hidden_size) for _ in range(n_vq)])
    heads = nn.ModuleList([nn.Linear(hidden_size, vocab, bias=False) for _ in range(n_vq)])
    stop_head = nn.Linear(hidden_size, 2, bias=False)
    with safe_open(args.checkpoint / "model.safetensors", framework="pt") as weights:
        for prefix, module in (
            ("local_transformer", baseline),
            ("audio_embeddings", embeddings),
            ("audio_lm_heads", heads),
            ("local_text_lm_head", stop_head),
        ):
            module.load_state_dict({k: weights.get_tensor(f"{prefix}.{k}") for k in module.state_dict()})
    optimized.load_state_dict(baseline.state_dict())
    for module in (baseline, optimized, embeddings, heads, stop_head):
        module.to(device="cuda", dtype=dtype).eval()
    optimized.prepare_qkv_lookup(embeddings, n_vq)

    validation = []
    # Teacher forcing keeps identical prefixes and separates numeric error
    # from legitimate differences in sampling RNG and tie breaking.
    for batch in args.batch_sizes:
        prefix = torch.randn(batch, n_vq, hidden_size, device="cuda", dtype=dtype)
        attn = optimized.h[0].attn
        key = prefix.new_full((batch, attn.n_head, n_vq, attn.head_dim), float("nan"))
        value = torch.full_like(key, float("nan"))
        for position in range(n_vq):
            if position:
                tokens = torch.randint(vocab, (batch,), device="cuda")
                prefix[:, position] = embeddings[position - 1](tokens)
                qkv = optimized._qkv_lookup[position - 1][tokens]
            else:
                qkv = attn.c_attn(optimized.h[0].ln_1(prefix[:, 0]))
            actual = optimized._run_lookup_step(prefix[:, position], qkv, key, value, position)
            expected = baseline._forward_prefix(prefix[:, : position + 1])[:, -1]
            rms = (actual.float() - expected.float()).square().mean().sqrt()
            relative_rms = (rms / expected.float().square().mean().sqrt()).item()
            assert torch.isfinite(actual).all() and relative_rms < 0.03, (batch, position, relative_rms)
            validation.append(dict(batch=batch, position=position, hidden_relative_rms=relative_rms))
    print(
        json.dumps({"teacher_forcing_max_relative_rms": max(x["hidden_relative_rms"] for x in validation)}), flush=True
    )

    baseline.setup_compile()
    optimized.setup_compile()
    results = []
    for batch in args.batch_sizes:
        hidden = torch.randn(batch, hidden_size, device="cuda", dtype=dtype)
        row = {"batch": batch}
        captures = []
        for name, model in (("main", baseline), ("optimized", optimized)):

            def run():
                return model.generate_frame(
                    hidden,
                    heads,
                    embeddings,
                    stop_head,
                    n_vq=n_vq,
                    temperature=1.7,
                    top_k=25,
                    top_p=0.8,
                )

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(5):
                    run()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = run()
            for _ in range(10):
                graph.replay()
            captures.append((name, graph, output))
            row[f"{name}_samples_ms"] = []
        # Alternate timing order to reduce within-process temperature drift.
        for repeat in range(args.repeats):
            for name, graph, output in captures[:: 1 if repeat % 2 == 0 else -1]:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.iterations):
                    graph.replay()
                end.record()
                end.synchronize()
                row[f"{name}_samples_ms"].append(start.elapsed_time(end) / args.iterations)
                assert output[1].shape == (batch, n_vq)
        for name in ("main", "optimized"):
            samples = row[f"{name}_samples_ms"]
            row[f"{name}_median_ms"] = statistics.median(samples)
            row[f"{name}_range_ms"] = [min(samples), max(samples)]
        row["speedup"] = row["main_median_ms"] / row["optimized_median_ms"]
        results.append(row)
        print(json.dumps(row), flush=True)
        del captures, graph, output
    report = dict(
        gpu=current_omni_platform.get_device_name(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        dtype=str(dtype),
        checkpoint=str(args.checkpoint),
        baseline_source=str(args.baseline_source),
        warmup_calls=5,
        warmup_replays=10,
        iterations=args.iterations,
        repeats=args.repeats,
        qkv_table_bytes=optimized._qkv_lookup.numel() * optimized._qkv_lookup.element_size(),
        validation=validation,
        results=results,
    )
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
