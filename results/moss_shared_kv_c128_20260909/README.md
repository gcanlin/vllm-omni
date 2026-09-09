# MOSS shared codec KV — C128 / 1088 E2E

Completed 2026-09-09. Source: local main `677265b6` plus the new shared-codec-KV
working-tree changes. Earlier Local lookup/fused kernels are archived separately
in `perf/moss-local-optimizations-20260909` at `ca9e0a1d`, not included here.

## Setup

- Model: OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5.
- Both AR and codec stages on physical H200 GPU 0; 128 sequences each.
- Codec capture buckets: 1, 2, 4, 8, 16, 32, 64, 128; chunk frames 1/15.
- `VLLM_OMNI_MOSS_CODEC_SHARED_KV=1`, verified in codec process environment.
- Reference encoder remains on CPU; no reference-device code change.
- Server **and client** use port 8124, avoiding the older process on 8123.
- Seed-TTS EN voice_clone, 1088 prompts, concurrency 128, output-len 256.
- Audio accounting: 48 kHz, stereo, PCM16. No quality-evaluation flag.
- Full 1088-prompt warmup excluded; two subsequent complete measured rounds.
  Each CLI invocation also executes its default two first-prompt warmups.
- `deploy.yaml`, `server_command.json` and per-phase command JSONs pin inputs.

## Results

| Phase | Completed / failed | Wall s | Audio-s/s | Requests/s | Mean RTF | Mean TTFP ms | Mean E2EL ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cold warmup (excluded) | 1088 / 0 | 405.0296 | 11.6375 | 2.6862 | 11.3684 | 43426.42 | 43926.98 |
| Round 2 | 1088 / 0 | 25.7997 | 181.8349 | 42.1711 | 0.702422 | 786.11 | 2921.81 |
| Round 3 | 1088 / 0 | 26.2267 | 180.0147 | 41.4844 | 0.701606 | 817.48 | 2928.87 |
| Counted mean | — | — | **180.9248** | 41.8277 | 0.702014 | 801.80 | 2925.34 |

Counted generated audio: 4691.28 + 4721.20 = 9412.48 s.
Pooled audio throughput: **180.9173 audio-s/s**. Round-to-round span: ~1.01%.
All phases have identical ordered input-length arrays, 143858 input tokens,
1088 completed requests, zero failures and empty detailed error arrays.

The server captured 16/16 compiled codec graphs in 293.44 s, without fallback.
One `_topk_topp_kernel` inference JIT occurred at 20:17:04 during cold warmup;
no later JIT warning or CUDA/OOM/state-capacity error was found. Compilation,
reference encoding warmup and client CLI warmups are not counted above.

No matched flag-off baseline was run. Do not compare these numbers directly
with archived measurements that included Local lookup/fused kernels, or with
PR #7202's different workload and timing protocol. No E2E speedup or quality
acceptance claim follows from this run alone. Other GPUs had changing external
workloads; the host was not isolated. The CLI's known one-second-bucket peak
concurrency statistic is not a true instantaneous in-flight count.

## Retained service

API PID 3790706, AR PID 3792294, codec PID 3793700. Health HTTP 200 after the run.
This service was subsequently stopped for the C64 run recorded in
`../moss_shared_kv_c64_20260909/README.md`; these PIDs are historical. Final GPU 0 memory
snapshot: 107309 MiB, whole device. Raw logs and detailed JSONs are in this
directory; implementation/validation notes: `docs/design/moss_codec_shared_kv.md`.
