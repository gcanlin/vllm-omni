# MOSS codec shared KV state experiment

## Source and scope

2026-09-09. Based on local main `677265b64b7870374705f18e81579c9d6e5ec8c7`.
The earlier local/codec experiments, tests, microbenchmarks and measurement
ledgers are preserved in branch `perf/moss-local-optimizations-20260909`,
commit `ca9e0a1d`. Unrelated untracked user files were not committed or changed.
This experiment does not include PR #7202 or the archived local kernels.

Enable before loading the model with `VLLM_OMNI_MOSS_CODEC_SHARED_KV=1`.
Default is **off**. Restart the process after changing the flag. The legacy
fixed-width streaming/reference path stays independent of the optimized pool.

## Implementation

- One metadata object per decoder transformer group: pre-step offsets,
  post-step offsets, scatter indices, physical ring positions and Boolean mask.
  All layers consume it before the group advances its offset once.
- Each group's MHA offset and ring end-offset alias its transformer offset.
  The actual v2 model reduces the offset bank from 190 rows to 6 rows.
  K/V values are **not** shared across layers or requests.
- Allocate `live_capacity + 1` state slots. All padding rows normalize to the
  final, non-leaseable, read-only null slot. Both KV and offset stores skip
  invalid rows; even identical writes to the null slot are avoided.
- Keep PyTorch's complete-ring gather, scatter and SDPA. A masked Triton
  writeback copies only ring positions touched by this chunk to persistent KV.
  For oversized chunks it commits the entire result of the original scatter;
  it does not impose a new last-writer policy on duplicate destinations.
- Give the optimized vLLM compile adapter a distinct identity to avoid reuse
  of old AOT artifacts with different mutation/aliasing behavior.

No change to sampling, precision, chunk size, scheduling, audio transport or
request admission. Slot-indirect CUTLASS attention and paged allocation are
**not implemented** in this first stage. Graph-padding compute/activation
memory also remains; only the persistent scratch state is eliminated.

## Memory accounting

For v2 revision `f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169`, decoder groups have
32/12/12/12/12/12 layers, widths 1280/768/768/768/768/768 and ring capacities
125/250/400/400/400/400. BF16 K+V require 84.5703125 MiB per persistent slot.

At live capacity 32 and maximum graph batch 32, the real model's allocated KV
tensor bytes are measured as 5,675,417,600 (legacy 64 slots) versus
2,926,387,200 (33 slots). Offset shapes are `[190,64]` versus `[6,33]`.

At C256 with graph bucket 256 the same shape formula predicts **42.2852 GiB
to 21.2252 GiB**, saving **21.0600 GiB** of persistent decoder KV. This is not
a measured whole-process memory reduction or an end-to-end throughput gain.

## Validation and reproduction

New regression suite covers CPU whole-codec exact parity, two resolutions,
wraparound, oversized chunks, slot reorder/recycling, all-padding execution,
read-only null state, CUDA Graph replay, and strict dynamic B/T compilation.
GPU writeback parity compares against the **same** reference scatter output,
including T=480/C=400; two independent duplicate-destination CUDA scatters
are not assumed deterministic. Actual graph-wrapper staging is exercised too.

```bash
CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m pytest tests/model_executor/models/moss_tts/test_shared_codec_kv.py \
  -o addopts= -q --run-level core_model

CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python benchmarks/kernels/benchmark_moss_shared_kv.py \
  --checkpoint /path/to/MOSS-Audio-Tokenizer-v2/snapshot \
  --capacity 32 --batches 1 4 16 --repeats 20
```

The benchmark loads actual decoder weights without instantiating the encoder.
It runs a ten-chunk fixed-code comparison with two live and two padded rows,
including slot reorder/reset, plus an independent legacy/legacy control.
It then times native CUDA Graph replays with four alternating-order rounds per
shape. External input tensors must remain owned for the graph's full lifetime.
`--compiled` additionally applies `torch.compile`; without it these are **plain
CUDA Graph** measurements, not the production Inductor path. Synthetic codes
are not a WER/SIM/UTMOS or perceptual-quality evaluation.

Initial results: 8 new tests passed; 33 existing Local Depth, per-request codes
and reference-encoder tests passed. Ruff and whitespace checks pass.

An initial microbench harness failed because it did not retain external graph
inputs. All timings from `/tmp/moss_shared_kv_20260909.log` are **discarded**.
The helper now retains codes, lengths, slots, validity and the decode callable
alongside the graph output. Production graph entries already retain inputs.

Timing and waveform results from the corrected harness are recorded below.
The requested E2E run is complete; results and limitations are recorded below.

## Corrected plain CUDA Graph results

H200 GPU 3, PyTorch 2.13.0+cu130, real v2 weights, native BF16 SDPA,
capacity 32, 20 measured replays per round and four alternating-order rounds.
The selected GPU had no other resident workload; the host/other GPUs were not
isolated. These are repeated microbench timings in one process, not independent
serving trials. Raw log: `/tmp/moss_shared_kv_20260909_owned_inputs.log`.

| Execution B | Base frames T | Legacy ms | Shared KV ms | Speedup | Time reduction |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 19.090 | 14.839 | 1.286x | 22.27% |
| 1 | 15 | 20.632 | 16.237 | 1.271x | 21.30% |
| 4 | 1 | 21.175 | 16.275 | 1.301x | 23.14% |
| 4 | 15 | 24.228 | 19.170 | 1.264x | 20.88% |
| 16 | 1 | 27.356 | 20.864 | 1.311x | 23.73% |
| 16 | 15 | 41.030 | 33.969 | 1.208x | 17.21% |

The corrected ten-chunk **eager** waveform comparison has maximum relative RMS
0.0047993, versus 0.0070909 in the legacy/legacy control. First chunk is exact.
The pre-timing eager comparison in the subsequent compiled-benchmark process
has maxima 0.0074871 and 0.0068062 respectively. The earlier diagnostic run
(before timing failed) had maxima 0.0137478 and 0.0059274. These observations
pass the script's diagnostic 0.03 ceiling but do **not** establish bit-exact
waveforms, perceptual equivalence or a universal noise-floor bound. In
particular the post-reset differences deserve further validation on real codes.

The `--compiled` switch currently affects the **timing** phase only. Its initial
waveform/control phase is eager; the compiled path has synthetic-model
regression coverage but not a full-checkpoint compiled-waveform quality gate.

## Inductor + CUDA Graph results

Same real model and capacity, with `--compiled --batches 1 4 --frames 15`.
Compilation/capture is excluded from timing. This uses plain `torch.compile`,
not a launched vLLM server; it is closer to the compiled execution path but
does not validate production AOT-cache reuse or serving throughput.
Raw log: `/tmp/moss_shared_kv_20260909_compiled.log`.

| Execution B | Base frames T | Legacy ms | Shared KV ms | Speedup |
|---:|---:|---:|---:|---:|
| 1 | 15 | 6.797 | 6.503 | 1.045x |

The smaller compiled-path gain is important: do not use the plain-graph
20–31% speedup range as a prediction for production throughput.

B=4 compilation was interrupted at the user's request to move directly to
serving benchmarks. No B=4 compiled timing is reported. The B=1 measurement
completed before interruption. The deploy config now uses 128 sequences per
stage and codec graph buckets up to 128 for the requested E2E run.

## Requested E2E run (complete)

Results: `results/moss_shared_kv_c128_20260909/`. Server PID 3790706,
port **8124** for both server and client (the user's client command named 8123,
which is occupied by an older API process and is deliberately not used).
Both stages use physical GPU 0 with capacity 128; codec buckets extend to 128.
`VLLM_OMNI_MOSS_CODEC_SHARED_KV=1` is verified in the codec worker environment.
Reference encoder placement was left unchanged on CPU, as requested.

Protocol: full 1088-prompt C128 warmup, excluded; then two 1088-prompt C128
measurements with `--output-len 256`, Seed-TTS EN voice clone, 48 kHz stereo,
and detailed result JSONs. Each CLI phase additionally performs its usual two
first-prompt warmups. Quality evaluation is not enabled. The new KV changes
are on main, without the archived Local lookup/fused-kernel optimizations;
historical throughput numbers are not a matched baseline for this run.

All three phases completed 1088 requests with zero failures and empty per-request
error arrays. Input lengths match in order across all phases; each has 143,858
input tokens. Cold warmup took 405.03 s and is excluded from the following table.

| Counted phase | Wall time s | Audio-s/s | Requests/s | Mean RTF | Mean TTFP ms | Mean E2EL ms |
|---|---:|---:|---:|---:|---:|---:|
| Round 2 | 25.7997 | 181.8349 | 42.1711 | 0.702422 | 786.11 | 2921.81 |
| Round 3 | 26.2267 | 180.0147 | 41.4844 | 0.701606 | 817.48 | 2928.87 |
| Arithmetic mean | — | **180.9248** | 41.8277 | 0.702014 | 801.80 | 2925.34 |

Pooled audio throughput is 180.9173 audio-s/s (9412.48 generated audio seconds
divided by 52.0264 wall seconds). The two-round audio-throughput span is about
1.01%. This is two runs on one hot server, not independent fresh-launch trials.

All 16 codec graphs were compiled and captured, with no fallback. A Local
sampling kernel JIT occurred during excluded cold warmup; no later JIT,
CUDA error, OOM or state-capacity error was found in the server log. Other GPUs
had changing external workloads; the host was not isolated. No matching
flag-off E2E comparison was performed, so this does not establish an E2E gain.
WER/SIM/UTMOS remain unevaluated.

Server initially retained on port 8124, API PID 3790706, AR PID 3792294, codec PID 3793700;
health returned HTTP 200 after measurement. GPU 0 used 107,309 MiB in the final
snapshot (whole device, not decoder KV alone). This server was later stopped
for the C64 deployment below. See the result directory's README and raw JSONs.

### C64 / 1088 follow-up, 2026-09-09

Same main + shared-KV source, GPU 0, memory fractions 0.60/0.30, CPU reference
encoder and output-len 256. Fresh deployment: both stage capacities 64, codec
capture buckets `[1,2,4,8,16,32,64]`, client concurrency 64. Thus this changes
server capacity as well as client concurrency versus the C128 run above.
Full 1088-prompt warmup excluded; two measured rounds:

| Phase | Completed / failed | Audio-s/s | Mean RTF | Mean TTFP ms | Mean E2EL ms |
|---|---:|---:|---:|---:|---:|
| Round 2 | 1088 / 0 | 144.1143 | 0.435502 | 340.90 | 1841.31 |
| Round 3 | 1088 / 0 | 146.3224 | 0.428525 | 341.61 | 1815.81 |
| Mean | — | **145.2184** | 0.432013 | 341.26 | 1828.56 |

Pooled throughput 145.2108 audio-s/s; round-to-round span 1.52%. Ordered input
lengths match across all phases; 143858 input tokens and zero nonempty errors
per phase. All 14 compiled codec graphs captured without fallback. No paired
flag-off baseline or quality evaluation: these results do not establish a gain.

Raw results and full setup: `results/moss_shared_kv_c64_20260909/README.md`.
Server retained on 8124: API 3932672, AR 3933193, codec 3934559; final health
HTTP 200. Final whole GPU 0 memory 100557 MiB. Deploy config now remains C64.
