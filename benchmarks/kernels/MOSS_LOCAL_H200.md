# MOSS Local v1.5 H200 optimizations

This PR carries the retained model/runner changes from
`perf/moss-local-h200-20260910` (`6ec06883`) onto current main. It is independent
of shared codec metadata (#7332) and the newer codec GEMM/layout PR (#7407).

The final source branch retains torch.compile and CUDA Graph capture. An eager
codec capture experiment was reverted after low-concurrency throughput regressed;
removing compile is not part of this change.

## Retained changes

- Precompute normalized, RoPE-applied QKV for each local codebook embedding.
  This is valid for the one-layer local transformer because the projection's
  input embedding has no context dependence. Gather by sampled token ID and
  keep attention state private to the current frame.
- Fuse short local attention with KV insertion, deterministic top-k/top-p
  sampling, and selected BF16 tiny-batch projections with bias/residual.
- Return already-forced text tokens through the model sampler and avoid
  rebuilding token history that this sampler does not consume.
- Fuse codec QKV unpack/RoPE, causal-mask construction, and ring insertion plus
  compact KV gather. Keep physical ring order and the existing attention backend.
- Reuse device slot IDs and pinned staging for codec input assembly, batch
  token clamps, and combine terminal tails only when compatible captured graphs
  cover the state capacity. Preserve direct terminal-padding fallback semantics.
- Apply the vLLM compilation context while capturing local MTP graph buckets.
  Permit front-server media arguments in CLI ownership validation.

Rejected residual/LayerNorm fusion, direct ring attention, async D2H, fused-head
candidates and PDL experiments are excluded. Raw profiler dumps and historical
server scripts are not included.

## Reproduce local frame measurements

Select an idle H200 and use a configured development environment:

```bash
git show origin/main:vllm_omni/model_executor/models/moss_tts/modeling_moss_tts_local_depth.py \
  > /tmp/moss-main-depth-baseline.py
python benchmarks/kernels/benchmark_moss_local_depth.py \
  --checkpoint /path/to/MOSS-TTS-Local-Transformer-v1.5 \
  --baseline-source /tmp/moss-main-depth-baseline.py \
  --modes main auto --batch-sizes 1 64 --iterations 30
python -m pytest -o addopts='' tests/model_executor/models/moss_tts \
  tests/config/test_environment_variables.py \
  tests/core/sched/test_omni_sched_deferred_free_fence.py -q
python benchmarks/kernels/validate_moss_codec.py \
  --checkpoint /path/to/MOSS-Audio-Tokenizer-v2 --mode pack --batch 2
```

The main baseline uses upstream frame-local KV reuse. `prefix` instead denotes
historical full-prefix recomputation and must not be presented as today's main.
The frame benchmark loads real local weights/heads/embeddings, uses synthetic
backbone activations and sampling temperature=1.7, top_k=25, top_p=0.8. It includes
all twelve codebooks and sampling but excludes the backbone and codec. Five
warmups and ten graph warmup replays precede five timed samples; report the
median. Teacher forcing separately gates hidden-state relative RMS at <0.03.

The waveform script uses generated random code sequences and real codec weights;
its outputs are numerical fixtures, not reference speech or perceptual evidence.
The baseline and optimized paths share the branch's RoPE implementation, so the
`pack` mode isolates KV packing; separate RoPE unit tests compare to the unfused
formula. Perceptual evaluation remains pending.

## Historical end-to-end context

Source-branch records compared against main `a7c295d5`, one H200, the same
upstream H200 deployment configuration, 1088 SeedTTS English voice-clone prompts,
output length 256, two measured runs per concurrency (whole-run fill/drain included):

| Concurrency | Main audio-s/s | Source branch audio-s/s | Change |
|---|---:|---:|---:|
| 64 | 199.59 | 213.72 | +7.1% |
| 128 | 230.19 | 240.26 | +4.4% |
| 256 | 230.37 | 228.25 | -0.9% |

Both controls included the media-path CLI validation fix. These are historical
integration results, not a fresh end-to-end measurement of the rebased PR.
Fresh PR-branch tests and local-frame measurements are recorded in the PR body.
