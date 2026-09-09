# MOSS shared codec KV — C64 / 1088 E2E

2026-09-09. Local main `677265b6` plus the shared-codec-KV working tree.
Archived Local lookup/fused kernels and PR #7202 are not included.

## Setup

- Both stages on physical H200 GPU 0, `max_num_seqs=64` each.
- Codec capture buckets `[1, 2, 4, 8, 16, 32, 64]`, chunk frames 1/15.
- AR/codec memory fractions 0.60/0.30; shared KV enabled.
- Reference encoder on CPU; port 8124 for both server and client.
- Seed-TTS EN voice_clone, 1088 prompts, concurrency 64, output-len 256.
- 48 kHz stereo PCM accounting; no quality-evaluation flag.
- Full 1088-prompt warmup excluded, followed by two measured rounds.
- Each CLI invocation also performs its default two first-prompt warmups.
- Configuration and command snapshots are alongside raw logs/results.

The prior C128 service was stopped for this fresh C64 deployment. This run
changes both server capacities/capture limit and client concurrency from C128;
it is not a client-only concurrency sweep on the same running service.

## Results

| Phase | Completed / failed | Wall s | Audio-s/s | Requests/s | Mean RTF | Mean TTFP ms | Mean E2EL ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cold warmup (excluded) | 1088 / 0 | 385.9527 | 12.0222 | 2.8190 | 5.606454 | 21272.58 | 21768.51 |
| Round 2 | 1088 / 0 | 32.7507 | 144.1143 | 33.2207 | 0.435502 | 340.90 | 1841.31 |
| Round 3 | 1088 / 0 | 32.3062 | 146.3224 | 33.6778 | 0.428525 | 341.61 | 1815.81 |
| Counted mean | — | — | **145.2184** | 33.4492 | 0.432013 | 341.26 | 1828.56 |

Counted audio: 4719.84 + 4727.12 = 9446.96 s. Pooled throughput:
**145.2108 audio-s/s**. Round-to-round span / mean: **1.52%**.
All three phases completed 1088 requests with zero failures and empty error
entries; ordered input-length arrays match, with 143858 input tokens per phase.
The codec captured all 14 compiled graphs without fallback. Its compilation
took 292.77 s, outside the measured phases. Startup optional-dependency and
auto-docstring config warnings did not prevent serving.

No matched shared-KV-off baseline is included.
The host is shared, generation is stochastic, and no E2E speedup or audio-quality
acceptance claim follows from this run alone.
The CLI's peak-concurrency values (111/108) use one-second buckets, not true
instantaneous concurrency; configured request concurrency is 64 throughout.

## Retained service

API PID 3932672, AR PID 3933193, codec PID 3934559; port 8124, health HTTP 200
after completion. The harness remains alive to retain the server. Final whole
GPU 0 memory snapshot: 100557 MiB (not a KV-only measurement).
