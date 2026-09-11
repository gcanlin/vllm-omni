# MOSS codec operator experiments

This change optimizes the V2 streaming codec independently of shared KV metadata
and the local-depth performance branch.

- Attention writes `[B,T,H,D]` storage and returns a `[B,H,T,D]` view, avoiding
  the copy before the output projection. Enabled by default.
- Optional skipping of wholly masked 64-key tiles preserves physical ring order
  and online-softmax block width. Disabled by default: sparse histories improve,
  but full windows regress.
- BF16 Tensor Core FFN kernels support exact-erf GELU or LayerScale + residual
  epilogues. Explicit intermediate BF16 rounding is retained; LayerNorm is unchanged.
- An offline tuner selects exact `(M,N,K,epilogue)` entries. Serving dispatch is
  inside an opaque custom op to preserve symbolic batch/frame dimensions. Missing
  entries use PyTorch GEMM. Configuration contents participate in the AOT cache key.

## Enable and reproduce

Set these in the server environment before starting the service:

```bash
export MOSS_CODEC_GEMM_CONFIG="$PWD/benchmarks/kernels/configs/moss_codec_h200.json"
```

The shipped H200 config selects GEMM+GELU for `N=3072,K=768`,
`M=2,4,8,15,16,30,32`, tile `(16,64,64)`, split=1. No GEMM-only or
GEMM+LayerScale+residual entry passed both performance and waveform gates.

Optional ablations (restart the server after changes):

- `MOSS_CODEC_ATTN_BTHD=0`: original contiguous attention output.
- `MOSS_CODEC_ATTN_SKIP_EMPTY=1`: enable experimental empty-tile skipping.
- Unset `MOSS_CODEC_GEMM_CONFIG`: original FFN path.
- `MOSS_CODEC_FFN_FUSION=0`: use GEMM-only whitelist entries. The shipped config
  contains none, so this setting falls back for all GEMMs.

From a configured development environment, select an idle GPU and run:

```bash
python benchmarks/kernels/benchmark_moss_codec_kernels.py \
  --part attention --output /tmp/moss-attention.json
python benchmarks/kernels/benchmark_moss_codec_kernels.py \
  --part gemm --verify-config benchmarks/kernels/configs/moss_codec_h200.json \
  --output /tmp/moss-gemm.json
python benchmarks/kernels/validate_moss_codec_kernels.py \
  --checkpoint /path/to/MOSS-Audio-Tokenizer-v2 \
  --config benchmarks/kernels/configs/moss_codec_h200.json --batch 1
python benchmarks/kernels/validate_moss_codec_kernels.py \
  --checkpoint /path/to/MOSS-Audio-Tokenizer-v2 \
  --config benchmarks/kernels/configs/moss_codec_h200.json --batch 4
```

Omit `--verify-config` to tune. Add `--experimental-split-k` only to investigate
split-K candidates: the original candidates failed long waveform checks and
are excluded from the default search. Validate any new configuration separately
before serving. The waveform script uses real weights and fixed random codes;
its generated outputs are numerical fixtures, not reference speech.

## Measurement scope

The microbenchmark uses three warmups followed by CUDA graph timings (eight
calls per graph, twenty replays per sample, median of five samples). GEMM
candidates are compared to fullgraph-compiled PyTorch GEMM+epilogue. Selection
requires relative L2 <0.004 and >5% speedup in both initial and confirmation
measurements. These reuse weights and are not end-to-end throughput estimates.

Original H200 measurements with torch 2.13.0+cu130 and vLLM 0.29.0:
attention output-layout speedup 1.23–1.57x at T>1 (B=1/4/16), T=1 neutral;
selected GEMM+GELU speedup 1.20–1.22x. Empty-tile skipping helps sparse histories
but regresses full windows and therefore remains off.

The shipped configuration passed ten streaming chunks at B=1 and B=4 on the
integration branch, including ring wrap and a slot reset. Maximum relative RMS
was 0.000508 and 0.008610 respectively (gate: <0.03). This is a numerical gate,
not a perceptual quality evaluation. Standalone PR validation is recorded in
the PR description; integration-branch serving numbers cannot be attributed
to these operators alone.
