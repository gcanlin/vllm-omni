# RFC: MOSS-TTS-Local-v1.5 Serving Optimization

## Background

MOSS-TTS-Local-v1.5 uses a Global Transformer + Local Transformer architecture. The Global Transformer is a Qwen3-style backbone that models text, reference-audio codes, and previous-frame audio feedback; the Local Transformer expands each global hidden state into one acoustic frame.

Each acoustic frame contains 12 RVQ codebooks, and codebooks inside the frame are generated sequentially. This means every audio frame requires not only one backbone step, but also a local continue/stop decision and 12 codebook sampling steps.

MOSS-Audio-Tokenizer-v2 handles both reference-audio encoding and waveform decoding, targets 48 kHz stereo output, and supports native incremental decoding. This gives MOSS Local high-quality audio and a natural streaming path, but it also makes the serving path heavier than many 24 kHz mono or lightweight-vocoder TTS models.

The current vLLM-Omni MOSS Local path is functionally usable, but it does not yet fully use batching, GPU-resident state, CUDA Graph, tensor payloads, or codec streaming. The goal is to reduce RTF, restore true low-TTFP streaming, and make high-concurrency behavior observable, recoverable, and tunable.

## Goals

- Reduce single-request and concurrent RTF by removing Python hot-path work, CPU syncs, small-kernel launches, and repeated tensor allocations.
- Let Stage 1 consume codec frames before Stage 0 finishes the full utterance, reducing TTFP.
- Preserve offline quality, stop semantics, and voice-clone behavior while adding batched, graphed, and streaming fast paths.
- Keep the optimization path compatible with vLLM-Omni multi-stage serving, benchmark, stats, and the OpenAI-compatible speech API.

## Proposed Optimizations

### Observability and Warmup

- [ ] Add MOSS-specific profiling for reference encoding, backbone decode, local frame decode, Stage0-to-Stage1 transfer, codec decode, and output formatting. Every RTF or TTFP change should be attributable to a specific stage instead of only showing up in end-to-end numbers.

- [ ] Record generated frames, real audio duration, sample rate, channel count, graph hit rate, and streaming chunk count in benchmarks and logs. MOSS Local outputs 48 kHz stereo audio, so incorrect duration accounting will directly corrupt RTF analysis.

- [ ] Load codec weights during service initialization and remove first-request codec cold start from serving latency. If CUDA Graph or streaming sessions are enabled, representative shapes and slots should also be warmed up during startup.

### Reference Audio Preprocessing

- [ ] Add a content-addressed cache for reference-audio codes to avoid repeatedly encoding the same speaker audio. Cache values should be stored on CPU in a compact dtype and returned as clones to avoid accidental mutation by downstream code.

- [ ] Add single-flight for the same uncached reference audio so concurrent requests trigger only one real encoding job. This prevents bursts of voice-clone traffic from overloading the codec encoder with duplicate work.

- [ ] Batch reference-audio encoding with a small wait window. The MOSS codec encoder is heavy enough that modest batching can improve GPU utilization, while per-item failures must still be isolated.

### Stage Transfer and Chunking

- [ ] Remove O(T^2) `torch.cat` from audio-code history accumulation. Stage 0 can use a growing buffer or a list of frame tensors and materialize the full code tensor only when needed.

- [ ] Emit delta frames from Stage 0 instead of emitting a full accumulated snapshot at every step and asking downstream code to deduplicate it. This reduces tensor copies, connector payload size, and Python object lifetime pressure.

- [ ] Keep Stage0-to-Stage1 audio codes as tensor payloads. Flattening `[T, n_vq]` or `[n_vq, T]` into `list[int]` should be a compatibility fallback, not the hot path.

- [ ] Restore true async chunking so Stage 0 sends frames to Stage 1 once the initial chunk threshold is reached. The first chunk can be smaller for lower TTFP, while later chunks can be larger for throughput.

- [ ] The chunk processor should track pending frames, emitted frame count, and finish state per request. On finish, it must flush remaining frames exactly once, and the Local Transformer's stop frame must not be sent to the codec as audio.

### AR Decode State and Local Transformer

- [ ] Introduce a GPU-resident decode state pool for feedback embeddings, current codes, sampling parameters, steps, seeds, stop flags, and related state. CPU-side state should only keep the request-id to row mapping and lifecycle management.

- [ ] The state pool must manage rows by request id, not by transient batch index. The vLLM scheduler may reorder, shrink, finish, or abort requests, so all exit paths must release the corresponding row.

- [ ] Use an input-embedding staging table for decode steps, where the previous frame's feedback embedding becomes the next backbone input. The Qwen3 backbone then sees a stable ordinary embedding lookup, while MOSS-specific multi-codebook fusion stays in the dedicated path.

- [ ] Batch local-transformer frame decode across active requests. The 12 codebooks still keep their within-frame sequential dependency, but each codebook step should run over the active batch.

- [ ] Reuse local-transformer KV within a frame to avoid re-prefilling the current prefix for every codebook. This must preserve MOSS Local's GPT-J/interleaved RoPE semantics and should be validated against the current eager path for logits and codes.

### Sampling and CUDA Graph

- [ ] Implement a GPU seeded sampler where the random stream is derived from request seed, frame step, and codebook index. This keeps sampling behavior stable for the same request across different batch compositions.

- [ ] Handle temperature, top-k, and top-p on GPU to avoid host-side control flow. Repetition penalty can initially fall back to eager and be moved into the graph after its history state becomes GPU-resident.

- [ ] Capture full local frame decode with CUDA Graph using batch-size buckets. The graphed region should include continue/stop, codebook logits, sampling, audio embedding lookup, and feedback embedding writeback.

- [ ] Frame graph must have explicit fallback behavior and metrics. Unsupported shapes, repetition penalty, and capture failures should record fallback reason and graph hit rate.

### Streaming Codec and Vocoder

- [ ] Use the native MOSS-Audio-Tokenizer-v2 streaming session. Each streaming request should acquire a slot, decode waveform incrementally by chunk, and release the slot on finish, error, or abort.

- [ ] Streaming slot exhaustion must not crash the request. The system should queue, fall back to offline decode, or explicitly reject new requests based on configuration, and the behavior must be observable.

- [ ] Add coalesced vocoder steps so when one request is due for decoding, other ready or near-ready slots can join the same codec call. All participating slots should use the same frame count plus an execution mask to amortize codec forward cost.

- [ ] Capture common streaming vocoder chunk sizes with CUDA Graph. The codec's internal streaming state must be updated in place so graph replay sees stable buffer addresses.

### Serving Configuration and Scheduling

- [ ] Tune Stage 1 capacity so it can cover the target streaming slots and batched codec decode. If Stage 1 can only process one sequence, streaming slots, coalescing, and codec batching cannot deliver their expected benefits.

- [ ] Split configuration modes into offline-fast, streaming-low-latency, and streaming-balanced. If a config declares streaming, the data path must actually send chunks instead of full-flushing only after request completion.

- [ ] Add a MOSS Local-specific GPU memory budget covering backbone KV cache, local decode state pool, codec weights, streaming codec state, and graph static buffers. Colocated single-GPU deployment needs explicit codec runtime headroom.

- [ ] Add backpressure between Stage 0 and Stage 1. When Stage 1 is busy, Stage 0 should not accumulate unbounded pending frames; it can increase steady chunk size or limit the pending queue to control memory and queuing.

### GPU-to-GPU Transfer

- [ ] Evaluate CUDA IPC for same-GPU cross-process tensor stream chunks. This should come after delta tensor payloads are implemented, otherwise the benefit for tiny frame rows may be hidden by Python and CPU-path overhead.

- [ ] Evaluate NCCL or another D2D relay for cross-GPU Stage0/Stage1 deployment. This should become a default path only when codec chunk tensors are large enough and CPU relay becomes a visible bottleneck.

## Validation

Correctness validation should cover offline/streaming audio duration, WER/CER, speaker similarity, chunk continuity, stop frames, max-frame handling, abort cleanup, and slot release. Performance validation should report RTF, TTFP, E2E latency, request throughput, audio throughput, graph hit rate, cache hit rate, and per-stage timing by warm state, concurrency level, and stream/non-stream mode.
