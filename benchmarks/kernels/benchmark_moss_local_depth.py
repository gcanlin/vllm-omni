# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare prefix and QKV-lookup local decoding using actual checkpoint weights.

Run from the repository with CUDA_VISIBLE_DEVICES selecting an idle GPU:
    python benchmarks/kernels/benchmark_moss_local_depth.py --checkpoint /path/to/snapshot

Measures the local frame generator, including heads/sampling, with compile and
CUDA graphs. Backbone inputs are synthetic: results are not end-to-end TTS RTF.
"lookup" is the previous QKV-lookup baseline; all other optimized modes include
fused attention. "sampling" adds sampling fusion, "auto" also enables measured
small-batch GEMV winners. "linear" forces experimental projection paths;
"all" forces all remaining kernels regardless of measured speed.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn
from transformers import GPT2Config

from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_local_depth import MossTTSLocalDepthTransformer


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 64])
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--baseline-source", type=Path, help="Main local-depth source file for a fair cached-KV baseline"
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["lookup", "attention", "sampling", "auto"],
        choices=["main", "prefix", "lookup", "attention", "linear", "sampling", "all", "auto"],
    )
    args = parser.parse_args()
    config = json.loads((args.checkpoint / "config.json").read_text())
    hidden_size = config["qwen3_config"]["hidden_size"]
    n_vq, vocab = config["n_vq"], config["audio_vocab_size"]
    # Prefix execution specializes on both batch size and depth. Allow all
    # variants so a Dynamo fallback cannot inflate the baseline's latency.
    torch._dynamo.config.recompile_limit = max(128, len(args.batch_sizes) * n_vq + 8)
    torch._dynamo.config.accumulated_recompile_limit = 2048
    device, dtype = "cuda", torch.bfloat16
    torch.manual_seed(42)
    reference = MossTTSLocalDepthTransformer(GPT2Config(**config["gpt2_config"]), hidden_size)
    embeddings = nn.ModuleList([nn.Embedding(vocab, hidden_size) for _ in range(n_vq)])
    heads = nn.ModuleList([nn.Linear(hidden_size, vocab, bias=False) for _ in range(n_vq)])
    stop_head = nn.Linear(hidden_size, 2, bias=False)
    with safe_open(args.checkpoint / "model.safetensors", framework="pt") as weights:
        for prefix, module in (
            ("local_transformer", reference),
            ("audio_embeddings", embeddings),
            ("audio_lm_heads", heads),
            ("local_text_lm_head", stop_head),
        ):
            state = {k: weights.get_tensor(f"{prefix}.{k}") for k in module.state_dict()}
            module.load_state_dict(state)
            module.to(device=device, dtype=dtype).eval()
    import copy

    main_baseline = None
    if "main" in args.modes:
        if args.baseline_source is None:
            parser.error("--modes main requires --baseline-source")
        spec = importlib.util.spec_from_file_location("moss_main_depth_baseline", args.baseline_source)
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot load baseline source")
        baseline_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = baseline_module
        spec.loader.exec_module(baseline_module)
        main_baseline = baseline_module.MossTTSLocalDepthTransformer(GPT2Config(**config["gpt2_config"]), hidden_size)
        main_baseline.load_state_dict(reference.state_dict())
        main_baseline.to(device=device, dtype=dtype).eval()

    lookup = copy.deepcopy(reference)
    lookup.prepare_qkv_lookup(embeddings, n_vq)
    lookup._fused_attention = False
    lookup._fused_linear = False
    lookup._auto_fused_linear = False
    lookup._fused_sampling = False

    # Teacher forcing isolates numerical error from random sampling divergence.
    attn = lookup.h[0].attn
    prefix = torch.randn(8, n_vq, hidden_size, device=device, dtype=dtype)
    key = prefix.new_zeros((8, attn.n_head, n_vq, attn.head_dim))
    value = torch.zeros_like(key)
    errors = []
    reference_hidden = []
    for position in range(n_vq):
        if position == 0:
            qkv = attn.c_attn(lookup.h[0].ln_1(prefix[:, 0]))
        else:
            token = torch.randint(vocab, (8,), device=device)
            prefix[:, position] = embeddings[position - 1](token)
            qkv = lookup._qkv_lookup[position - 1][token]
        actual = lookup._run_lookup_step(prefix[:, position], qkv, key, value, position)
        expected = reference._forward_prefix(prefix[:, : position + 1])[:, -1]
        reference_hidden.append(expected.clone())
        error = (heads[position](actual).float() - heads[position](expected).float()).abs()
        errors.append(
            {"position": position, "logit_max_abs": error.max().item(), "logit_mean_abs": error.mean().item()}
        )
        # BF16 differences scale with activations; compare relative RMS too.
        relative_rms = (
            actual.float() - expected.float()
        ).square().mean().sqrt() / expected.float().square().mean().sqrt()
        assert torch.isfinite(actual).all() and relative_rms < 0.03, (position, relative_rms.item())
    print(json.dumps({"teacher_forcing_bf16": errors}), flush=True)
    variants = {}
    for mode in args.modes:
        model = main_baseline if mode == "main" else reference if mode == "prefix" else copy.deepcopy(lookup)
        assert model is not None
        model._fused_attention = mode not in ("prefix", "lookup")
        model._fused_linear = mode in ("linear", "all")
        model._fused_sampling = mode in ("sampling", "all")
        model._auto_fused_linear = mode == "auto"
        if mode == "auto":
            model._fused_sampling = True
        model.setup_compile()
        variants[mode] = model
        if mode not in ("prefix", "main"):
            validation_key, validation_value = torch.empty_like(key), torch.empty_like(value)
            max_relative_rms = 0.0
            for position in range(n_vq):
                if position == 0:
                    qkv = attn.c_attn(model.h[0].ln_1(prefix[:, 0]))
                else:
                    # Use the same embedded token as the prefix reference.
                    raw = attn.c_attn(model.h[0].ln_1(prefix[:, position]))
                    q, k, v = raw.split(hidden_size, -1)
                    q = q.reshape(8, attn.n_head, attn.head_dim)
                    k = k.reshape_as(q)
                    cos, sin = attn._rope_cos_cache[0, position], attn._rope_sin_cache[0, position]
                    q = q * cos + attn._rotate_half(q) * sin
                    k = k * cos + attn._rotate_half(k) * sin
                    qkv = torch.cat((q.flatten(1), k.flatten(1), v), -1)
                actual = model._run_lookup_step(prefix[:, position], qkv, validation_key, validation_value, position)
                expected = reference_hidden[position]
                relative_rms = (actual.float() - expected.float()).square().mean().sqrt()
                relative_rms /= expected.float().square().mean().sqrt()
                assert torch.isfinite(actual).all() and relative_rms < 0.03, (mode, position, relative_rms.item())
                max_relative_rms = max(max_relative_rms, relative_rms.item())
            print(json.dumps({"validation_mode": mode, "max_hidden_relative_rms": max_relative_rms}), flush=True)
    results = []
    for batch_size in args.batch_sizes:
        hidden = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
        row = {"batch_size": batch_size}
        for name, model in variants.items():

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
            samples = []
            for _ in range(5):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.iterations):
                    graph.replay()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) / args.iterations)
            row[f"{name}_ms"] = sorted(samples)[len(samples) // 2]
            assert output[1].shape == (batch_size, n_vq)
            del graph, output
        baseline = row[f"{args.modes[0]}_ms"]
        row["speedups"] = {name: baseline / row[f"{name}_ms"] for name in variants}
        results.append(row)
        print(json.dumps(row), flush=True)
    print(json.dumps({"gpu": torch.cuda.get_device_name(), "results": results}), flush=True)


if __name__ == "__main__":
    main()
