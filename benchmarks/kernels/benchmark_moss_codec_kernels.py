# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA graph ablations and offline FFN tuning on real codec matrix shapes."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from vllm.triton_utils import triton

from vllm_omni.model_executor.models.moss_tts import codec_gemm
from vllm_omni.model_executor.models.moss_tts.codec_gemm import ffn_gemm
from vllm_omni.model_executor.models.moss_tts.streaming_attention import _attention


def measure(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(8):
            fn()
    times = []
    for _ in range(5):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(20):
            graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / 160)
    return sorted(times)[2]


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--part", choices=["attention", "gemm"], required=True)
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 2, 4, 8, 15, 16, 30, 32, 60, 120, 240, 480])
    parser.add_argument("--verify-config", help="Retest a whitelist through the serving dispatcher")
    parser.add_argument(
        "--experimental-split-k",
        action="store_true",
        help="Include split-K candidates; requires separate waveform validation",
    )
    args = parser.parse_args()
    verify = json.loads(Path(args.verify_config).read_text()) if args.verify_config else None
    if verify is not None:
        codec_gemm.CONFIG = verify
    torch.manual_seed(17)
    torch._dynamo.config.recompile_limit = 256
    records, selected = [], {}
    if args.part == "attention":
        for b in [1, 4, 16]:
            for h, t, c in [(20, 1, 125), (20, 15, 125), (12, 32, 400), (12, 120, 400), (12, 480, 400)]:
                q = torch.randn(b, h, t, 64, device="cuda", dtype=torch.bfloat16)
                k, v = torch.randn(2, b, h, c, 64, device="cuda", dtype=torch.bfloat16)
                for filled in [min(t, c), c]:
                    mask = (
                        (torch.arange(c, device="cuda")[None, :] < filled)
                        .expand(t, c)[None, None]
                        .expand(b, 1, t, c)
                        .contiguous()
                    )
                    ref = None
                    for layout, skip in [(False, False), (True, False), (True, True)]:
                        out = (
                            torch.empty((b, t, h, 64), device="cuda", dtype=q.dtype).transpose(1, 2)
                            if layout
                            else torch.empty_like(q)
                        )

                        def run():
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
                                c,
                                64,
                                bm,
                                64,
                                layout,
                                skip,
                                num_warps=8 if c <= 256 else 4,
                                num_stages=2,
                            )
                            return out.transpose(1, 2).reshape(b, t, h * 64)

                        value = run().clone()
                        if ref is None:
                            ref = value
                        torch.testing.assert_close(value, ref, atol=0, rtol=0)
                        row = dict(b=b, h=h, t=t, c=c, filled=filled, layout=layout, skip=skip, us=measure(run))
                        print(json.dumps(row), flush=True)
                        records.append(row)
    else:
        for m in args.rows:
            for k, n in [(1280, 5120), (5120, 1280), (768, 3072), (3072, 768)]:
                x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                w = torch.randn(n, k, device="cuda", dtype=x.dtype) * k**-0.5
                scale = torch.randn(n, device="cuda", dtype=x.dtype) * 0.01
                residual = torch.randn(m, n, device="cuda", dtype=x.dtype)
                for mode in [0, 1 if n > k else 2]:
                    key = f"{m},{n},{k},{mode}"
                    if verify is not None and key not in verify:
                        continue

                    def baseline():
                        value = F.linear(x, w)
                        return F.gelu(value) if mode == 1 else (residual + value * scale if mode == 2 else value)

                    ref = baseline()
                    baseline_fn = torch.compile(baseline, fullgraph=True)
                    base = measure(baseline_fn)
                    best = base
                    winner = None
                    candidates = [
                        (16, 16, 64, 1),
                        (16, 32, 64, 1),
                        (16, 64, 64, 1),
                        (16, 64, 64, 4),
                        (16, 64, 64, 8),
                        (32, 64, 64, 1),
                        (32, 128, 64, 1),
                        (64, 64, 32, 1),
                    ]
                    for bm, bn, bk, split in [verify[key]] if verify is not None else candidates:

                        def candidate():
                            if verify is not None:
                                return codec_gemm.selected_linear(x, w, scale, residual, mode)
                            return ffn_gemm(x, w, scale, residual, mode, bm, bn, bk, split)

                        value = candidate()
                        error = (value.float() - ref.float()).norm() / ref.float().norm()
                        us = measure(candidate)
                        valid = error.item() < 0.004
                        if valid and us < best:
                            best, winner = us, [bm, bn, bk, split]
                        records.append(
                            dict(
                                m=m,
                                n=n,
                                k=k,
                                mode=mode,
                                config=[bm, bn, bk, split],
                                us=us,
                                base_us=base,
                                rel_l2=error.item(),
                            )
                        )
                    key = f"{m},{n},{k},{mode}"
                    # Recheck the winner against a fresh baseline; require a 5% margin.
                    if winner is not None:

                        def fast():
                            if verify is not None:
                                return codec_gemm.selected_linear(x, w, scale, residual, mode)
                            return ffn_gemm(x, w, scale, residual, mode, *winner)

                        base2, best2 = measure(baseline_fn), measure(fast)
                        records.append(dict(key=key, confirmation=True, base_us=base2, us=best2, speedup=base2 / best2))
                        if min(base / best, base2 / best2) > 1.05:
                            selected[key] = winner
                    print(
                        json.dumps(
                            dict(key=key, base_us=base, best_us=best, speedup=base / best, selected=key in selected)
                        ),
                        flush=True,
                    )
                    Path(args.output + ".config.json").write_text(json.dumps(selected, indent=2) + "\n")
    Path(args.output).write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(), records=records), indent=2) + "\n")


if __name__ == "__main__":
    main()
