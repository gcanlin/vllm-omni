# MOSS pure-main C64 baseline versus shared codec KV

2026-09-09. Baseline source: local main
`677265b64b7870374705f18e81579c9d6e5ec8c7`, with no tracked working-tree changes.
No fetch/pull was performed. Unrelated user-owned untracked files were preserved.

Shared-KV work was committed separately on
`perf/moss-codec-shared-kv-20260909` at `a2f90fd9` before returning to main.
That commit also archives the optimized C64/C128 reports, implementation notes,
tests and microbenchmarks. No push was performed.

## Matched setup

- Physical H200 GPU 0, both stages, max_num_seqs 64 each.
- AR/codec memory fractions 0.60/0.30; codec buckets 1,2,4,8,16,32,64.
- Chunk frames 1/15; CPU reference encoder; port 8124 for server and client.
- Seed-TTS EN voice_clone, 1088 prompts, client concurrency 64, output-len 256.
- 48 kHz stereo PCM accounting, no WER/SIM/UTMOS evaluation.
- Fresh process, one full 1088-prompt warmup excluded, then two measured rounds.
- Default two first-prompt CLI warmups also excluded from each phase's wall time.
- Same benchmark harness retrieved from commit a2f90fd9 and executed in memory;
  main model/server/benchmark source is unmodified.
- `VLLM_OMNI_MOSS_CODEC_SHARED_KV=0` is explicit; pure main has no shared-KV code.
- `deploy.yaml` matches the preceding optimized C64 snapshot exactly; relative
  to main's default deploy YAML it differs only in explanatory comments.

## Results

| Baseline phase | Completed / failed | Wall s | Audio-s/s | Requests/s | Mean RTF | Mean TTFP ms | Mean E2EL ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cold warmup (excluded) | 1088 / 0 | 393.8575 | 11.9073 | 2.7624 | 5.702123 | 21622.47 | 22120.86 |
| Round 2 | 1088 / 0 | 32.1009 | 144.1106 | 33.8931 | 0.435768 | 349.03 | 1809.31 |
| Round 3 | 1088 / 0 | 32.0542 | 146.5795 | 33.9426 | 0.430872 | 344.68 | 1817.02 |
| Counted mean | — | — | **145.3450** | 33.9178 | 0.433320 | 346.85 | 1813.17 |

Counted audio: 4626.08 + 4698.48 = 9324.56 s. Pooled baseline throughput:
**145.3441 audio-s/s**. Round-to-round span / mean: **1.70%**.
All three baseline phases completed 1088 requests, zero failures, and zero
nonempty error entries. Ordered input-length arrays match both optimized
measured rounds, with 143858 input tokens per phase.

### Matched C64 comparison

| Metric, two-round mean | Pure main | Shared KV | Shared KV relative to main |
| --- | ---: | ---: | ---: |
| Audio throughput (audio-s/s) | 145.3450 | 145.2184 | **-0.087%** |
| Requests/s | 33.9178 | 33.4492 | -1.382% |
| Mean RTF | 0.433320 | 0.432013 | -0.302% |
| Mean TTFP ms | 346.85 | 341.26 | -1.614% |
| Mean E2EL ms | 1813.17 | 1828.56 | +0.849% |

**No measurable C64 E2E throughput benefit in this comparison.** The 0.087%
mean difference is much smaller than either arm's round-to-round spread
(main 1.70%, shared KV 1.52%). This is not proof of exact equivalence or
statistical significance. Generated audio totals differ; requests/s and raw
completion latency should be interpreted with that in mind.

Optimized C64 rounds were 144.1143 / 146.3224 audio-s/s. Their raw JSONs remain in
`../moss_shared_kv_c64_20260909/`; their README is archived in a2f90fd9.

This is a sequential optimized-then-baseline comparison on a shared host,
not randomized/counterbalanced or statistically conclusive. No audio-quality
acceptance claim is made; generated lengths are stochastic.

## Validation and retained service

All 14 native main codec compiled graphs captured without fallback; codec
torch.compile took 376.39 s, excluded from measured wall time. No CUDA/OOM
failure was found. Health HTTP 200 after completion.

Repository remains on main with no tracked modifications. API PID 4071702,
AR PID 4072744, codec PID 4073595; service retained on port 8124 by the harness.
The previous shared-KV C64 service was stopped before switching branches.

Final whole GPU 0 memory: 105745 MiB baseline versus preceding optimized
snapshot 100557 MiB, a 5188 MiB lower snapshot for shared KV. This is not an
isolated KV allocation measurement; allocator/graph state may also differ.
Other GPUs had changing external workloads during these sequential runs.
