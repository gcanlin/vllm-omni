# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""H200 local projection tile sweep (synthetic BF16 inputs, actual MOSS shapes).

CUDA graph timings have warm weights; confirm winners with the complete-frame
benchmark, where the three projections compete for L2 capacity.
"""

import argparse
import json

import torch
import torch.nn.functional as F

from vllm_omni.model_executor.models.moss_tts.local_kernels import fused_linear


def measure(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        # Amortize Python/driver replay submission cost for microsecond ops.
        for _ in range(32):
            out = fn()
    graph.replay()
    samples = []
    for _ in range(5):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(30):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / (30 * 32))
    return sorted(samples)[2], out


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8, 32, 64])
    args = parser.parse_args()
    torch._dynamo.config.recompile_limit = 128
    configs = [(16, 32, 128, 4), (16, 64, 64, 4), (16, 64, 128, 4), (32, 64, 64, 4), (32, 128, 64, 4)]
    for batch in args.batch_sizes:
        for name, k, n, activation in [
            ("proj", 2560, 2560, False),
            ("up", 2560, 9728, True),
            ("down", 9728, 2560, False),
        ]:
            x = torch.randn(batch, k, device="cuda", dtype=torch.bfloat16)
            weight = torch.randn(n, k, device="cuda", dtype=x.dtype) * 0.01
            bias = torch.randn(n, device="cuda", dtype=x.dtype)
            residual = None if activation else torch.randn(batch, n, device="cuda", dtype=x.dtype)

            def reference():
                out = F.linear(x, weight, bias)
                return F.silu(out) if activation else out + residual

            baseline, expected = measure(torch.compile(reference, options={"epilogue_fusion": False}))
            row = {"batch": batch, "op": name, "torch_us": baseline, "tiles_us": {}}
            selected_configs = configs + ([(0, 4, 512, 4), (0, 4, 1024, 4), (0, 8, 512, 4)] if batch <= 4 else [])
            for config in selected_configs:

                def run():
                    return fused_linear(x, weight, bias, residual, activation=activation, config=config)

                latency, actual = measure(run)
                torch.testing.assert_close(actual, expected, atol=0.04, rtol=0.03)
                row["tiles_us"][str(config)] = latency
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
