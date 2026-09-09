# C128 serving memory: pure main versus shared codec KV

Measured 2026-09-09 on the same H200 GPU 0. Both stages have max_num_seqs=128;
codec capture buckets `[1,2,4,8,16,32,64,128]`, frames 1/15. Memory fractions
AR/codec 0.60/0.30. The two saved deploy YAML files are byte-identical.

Main: `677265b64b7870374705f18e81579c9d6e5ec8c7`.
Shared KV: branch `perf/moss-codec-shared-kv-20260909`, commit `a2f90fd9`.
Both worktrees had no tracked changes. The shared flag is explicitly 0/1.
Services were started serially on 8124; GPU 0 had no compute processes and
used 4 MiB before each launch. The old C64 service was stopped first.

## Matched post-capture idle snapshots

Each value below was identical across three samples, five seconds apart,
after HTTP health readiness and completion of all 16 codec graph captures.
These samples precede benchmark requests for both variants.

| Metric (MiB) | Pure main | Shared KV | Main minus shared KV |
| --- | ---: | ---: | ---: |
| Whole GPU used | 118859 | 107189 | **11670** |
| Whole GPU free, directly reported | 24298 | 35968 | -11670 |
| AR process | 83436 | 83436 | 0 |
| Codec process | 35406 | 23736 | **11670** |

Whole GPU used: **116.07 GiB -> 104.68 GiB**, saving **11.40 GiB**.
Actual free memory: **23.73 GiB -> 35.125 GiB**.
Codec process memory: **34.58 GiB -> 23.18 GiB**.

These are NVML/nvidia-smi service/process measurements, not allocator-level
KV tensor byte counts. The theoretical persistent KV tensor reduction is
10.49 GiB; graph pools, cached allocations and other state also contribute
to the measured codec-process difference. Main compiled fresh, whereas shared
KV reused an AOT artifact; do not attribute the extra difference exclusively
to permanent KV storage. No graph fallback was observed.

The GPU reports total memory 143771 MiB, with approximately 614 MiB not
included in used+free (driver-reserved accounting). Free memory is read
directly, not estimated as total-used. The earlier 35.62 GiB remaining-memory
estimate used subtraction and did not account for this distinction; this run
measures 35.125 GiB free in the post-capture optimized service.

## Scope change and raw data

The initial plan included full1088 warmup and a hot round. At the user's
request, main's in-progress warmup was stopped early and shared KV was run
with `--startup-only`. No complete throughput measurement or matched fully
warmed memory comparison is claimed here. Request cancellations in main's
log reflect the intentional client stop, not a spontaneous serving failure.
After the partial main warmup was stopped, GPU memory was 118861 MiB, only
2 MiB above its pre-request snapshot (AR +2 MiB, codec unchanged).

`main/memory.jsonl` and `shared_kv/memory.jsonl` contain five-second samples,
including startup. Sampled startup maxima were 120921 and 107189 MiB; these
are not exact peaks and are not a matched compile-cache-state comparison.
`provenance.json`, `deploy.yaml`, `process_tree.txt`, and `server.log` in each
case directory preserve commands, revisions, configurations and captures.

NVML returned host PIDs and `[Not Found]` process names in this environment.
AR/codec attribution follows their sequential allocation/startup: main
host PIDs 2087904/2096575 correspond to namespace PIDs 22387/23306; shared KV
host PIDs 2322114/2326301 correspond to namespace PIDs 85627/86535.

## Retained state

Main service stopped. Shared KV service retained on port 8124: API PID 84570,
AR PID 85627, codec PID 86535. Final health HTTP 200. Current branch is
`perf/moss-codec-shared-kv-20260909`; the running service uses its external
C128 deploy snapshot, not the branch's checked-in C64 default YAML.
