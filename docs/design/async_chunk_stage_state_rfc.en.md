# RFC: Async Chunk Stage State

## Status

Draft.

## Background

The `async_chunk` path allows an upstream stage to send intermediate outputs to a downstream stage before the upstream request has finished. Today, `OmniChunkTransferAdapter` passes itself to model-specific processors as `transfer_manager`:

```python
payload_data = self.custom_process_next_stage_input_func(
    transfer_manager=self,
    multimodal_output=multimodal_output,
    request=request,
    is_finished=is_segment_finished,
)
```

This gives processors access to the connector, chunk ids, config, and request lifecycle. In practice, processors also attach model-specific state directly to the adapter:

```python
transfer_manager.code_prompt_token_ids[request_id].append(frame)
transfer_manager.request_payload[request_id] = ref_code
transfer_manager._cached_ic[request_id] = initial_chunk_size
transfer_manager._moss_tts_raw_state[request_id] = {...}
```

These fields are spread across the framework and model processors. Their semantics are inconsistent, and the adapter must hard-code cleanup for fields it happens to know about.

## Problem

The current approach works, but it turns model-private state into dynamic attributes on `OmniChunkTransferAdapter`.

This has several issues:

1. Implicit API: processors can add arbitrary attributes to `transfer_manager`.
2. Incomplete cleanup: request finish, segment finish, abort, and preemption paths need to clean state, but the adapter does not know all model-private fields.
3. Name collisions: future framework fields or other models may reuse the same names.
4. Weak typing: fields such as `request_payload` and `code_prompt_token_ids` are reused by different models with different semantics.
5. Poor testability: state is scattered across dynamic attributes, making lifecycle assertions and leak detection harder.

MOSS-TTS-Local makes this visible because it needs a per-request cursor:

```text
talker emits an accumulated snapshot
async_chunk slices new frames using total_emitted
Stage1 codec streaming session decodes only the new frames
```

That requires state such as `total_emitted` and `prompt_emitted`. Storing it under `_moss_tts_raw_state` is functional, but it is not a clean framework interface.

## Goals

Introduce a formal async chunk state abstraction for per-request state used by model processors during chunk assembly.

Goals:

1. Provide namespaced per-request state for model processors.
2. Let `OmniChunkTransferAdapter` own state lifecycle.
3. Avoid dynamic model-private attributes on `transfer_manager`.
4. Support unified cleanup for request finish, segment finish, and abort.
5. Preserve model ownership of chunk semantics.

## Non-Goals

This RFC does not attempt to unify all model chunking strategies.

The following remain model-processor responsibilities:

1. How to validate a frame.
2. When to emit a chunk.
3. Whether to include left context.
4. Whether reference codes are prepended to `codes.audio` or sent as `codes.ref` side-band.
5. How to flatten codec codes.
6. Which meta fields to emit.
7. How to express terminal empty payloads.

This RFC also does not cover KV cache transfer, scheduler queue state, or Stage1 decoder runtime state.

## Design Overview

Add a `StageStateStore` owned by `OmniChunkTransferAdapter`:

```python
class OmniChunkTransferAdapter(...):
    def __init__(self, vllm_config):
        ...
        self.stage_state = StageStateStore()
```

Model processors should stop doing this:

```python
transfer_manager._moss_tts_raw_state[req_id]
```

and instead do this:

```python
state = transfer_manager.stage_state.get(
    namespace="moss_tts_raw",
    request_id=req_id,
    factory=MossTTSRawState,
)
```

The `namespace` isolates model or processor state. The `factory` creates the model-specific typed state object.

## Proposed API

### StageStateStore

```python
class StageStateStore:
    def get[T](self, namespace: str, request_id: str, factory: Callable[[], T]) -> T:
        ...

    def peek(self, namespace: str, request_id: str) -> object | None:
        ...

    def pop(self, namespace: str, request_id: str) -> object | None:
        ...

    def cleanup_request(self, request_id: str) -> None:
        ...

    def cleanup_segment(self, request_id: str) -> None:
        ...

    def clear_namespace(self, namespace: str) -> None:
        ...

    def request_count(self, namespace: str | None = None) -> int:
        ...
```

A minimal implementation can be:

```python
class StageStateStore:
    def __init__(self):
        self._states: dict[str, dict[str, object]] = defaultdict(dict)

    def get(self, namespace, request_id, factory):
        ns = self._states[namespace]
        state = ns.get(request_id)
        if state is None:
            state = factory()
            ns[request_id] = state
        return state

    def pop(self, namespace, request_id):
        return self._states.get(namespace, {}).pop(request_id, None)

    def cleanup_request(self, request_id):
        for ns in self._states.values():
            ns.pop(request_id, None)
```

### State Hook

To distinguish request finish from realtime segment finish, a state object may optionally implement lifecycle hooks:

```python
class AsyncChunkRequestState(Protocol):
    def on_segment_end(self) -> None: ...
    def on_request_end(self) -> None: ...
    def on_abort(self) -> None: ...
```

`StageStateStore.cleanup_segment()` calls `on_segment_end()`. `cleanup_request()` calls `on_request_end()` and then removes all state for that request.

Hooks are optional. Plain dataclasses remain valid state objects.

## Model State Examples

### MOSS-TTS-Local

```python
@dataclass
class MossTTSRawState:
    accumulated: torch.Tensor | None = None
    total_emitted: int = 0
    emitted_any: bool = False
    prompt: torch.Tensor | None = None
    prompt_emitted: bool = False

    def on_segment_end(self) -> None:
        self.accumulated = None
        self.total_emitted = 0
        self.emitted_any = False
        self.prompt = None
        self.prompt_emitted = False
```

Processor usage:

```python
state = transfer_manager.stage_state.get(
    "moss_tts_raw",
    req_id,
    MossTTSRawState,
)

state.accumulated = snapshot_cpu
pending = int(state.accumulated.shape[0]) - state.total_emitted
chunk_codes = state.accumulated[state.total_emitted : state.total_emitted + emit_frames]
state.total_emitted += emit_frames
```

### Qwen3-TTS

```python
@dataclass
class Qwen3TTSChunkState:
    frames: list[list[int]] = field(default_factory=list)
    ref_code: torch.Tensor | None = None
    cached_initial_chunk: int | None = None

    def on_request_end(self) -> None:
        self.frames.clear()
        self.ref_code = None
        self.cached_initial_chunk = None
```

Processor usage:

```python
state = transfer_manager.stage_state.get(
    "qwen3_tts_codec",
    request_id,
    Qwen3TTSChunkState,
)

state.frames.append(codec_codes)
length = len(state.frames)
window_frames = state.frames[-end_index:]
```

This can replace:

```python
transfer_manager.code_prompt_token_ids[request_id]
transfer_manager.request_payload[request_id]
transfer_manager._cached_ic[request_id]
```

## Transfer Manager Helpers

In addition to the state container, `OmniChunkTransferAdapter` can gradually expose common helpers.

### Config Access

Many processors repeat:

```python
connector = getattr(transfer_manager, "connector", None)
raw_cfg = getattr(connector, "config", {}) or {}
cfg = raw_cfg.get("extra", raw_cfg)
```

The adapter can provide:

```python
cfg = transfer_manager.chunk_config()
chunk_size = transfer_manager.get_extra_int("codec_chunk_frames", default=25)
```

### Chunk Counters

Processors currently read:

```python
transfer_manager.put_req_chunk[request_id]
```

The adapter can provide:

```python
chunk_id = transfer_manager.put_chunk_id(request_id)
sent = transfer_manager.sent_chunk_count(request_id)
```

### FrameBuffer Helper

For models such as Qwen3, Higgs, Mimo, and Voxtral that append frames incrementally:

```python
class FrameBuffer:
    def append(self, frame): ...
    def length(self) -> int: ...
    def window(self, context_frames: int, chunk_frames: int): ...
    def clear(self) -> None: ...
```

For accumulated snapshot models such as MOSS Local:

```python
class SnapshotCursor:
    total_emitted: int

    def pending(self, snapshot) -> int: ...
    def take(self, snapshot, n: int): ...
```

These helpers should remain optional utilities, not base classes that every processor must inherit.

## Lifecycle

### Request Finish

After a terminal payload is successfully sent, the sender side calls:

```python
self.stage_state.cleanup_request(external_req_id)
```

This should happen after connector `put()` succeeds, preserving the existing behavior that avoids cleanup/save-loop races.

### Segment Finish

For realtime or resumable requests, segment finish is not the same as request finish.

When `is_segment_finished=True`, the adapter can call:

```python
self.stage_state.cleanup_segment(external_req_id)
```

Whether all state is cleared is left to the state hook. Some realtime state may span segments, while codec chunk buffers may need to reset at segment boundaries.

### Abort / Scheduler Cleanup

When the scheduler calls adapter cleanup, the adapter should also clean:

```python
self.stage_state.cleanup_request(external_req_id)
```

If abort can be distinguished, the store can expose:

```python
self.stage_state.abort_request(external_req_id)
```

which calls `on_abort()` before removing state.

## Migration Plan

### Phase 1: Add the Store

1. Add `StageStateStore`.
2. Create `self.stage_state` in `OmniChunkTransferAdapter.__init__`.
3. Call store cleanup from `cleanup_sender()`, `cleanup()`, and segment cleanup paths.
4. Do not change existing model processor behavior yet.

### Phase 2: Migrate MOSS-TTS-Local

Replace:

```python
transfer_manager._moss_tts_raw_state
```

with:

```python
transfer_manager.stage_state.get("moss_tts_raw", req_id, MossTTSRawState)
```

This removes the most visible dynamic private field.

### Phase 3: Migrate Common TTS Processors

Gradually migrate:

```text
code_prompt_token_ids
request_payload
_cached_ic
higgs_v3_emitted_frames
_pending_streaming_prefills
```

`put_req_chunk`, `get_req_chunk`, and `finished_requests` should stay on the adapter because they are transfer protocol state, not model chunk assembly state.

### Phase 4: Add Helpers

After state migration is stable, add:

```text
chunk_config()
get_extra_int()
FrameBuffer
SnapshotCursor
```

This avoids starting with a heavy abstraction.

## Compatibility

The design can be backward compatible:

1. Existing processors still receive `transfer_manager`.
2. Existing fields remain temporarily supported.
3. New models should prefer `stage_state`.
4. Old processors can be migrated one model at a time.

## Alternatives

### Alternative A: Per-Model TransferManager Subclasses

Not recommended.

This tightly couples model logic to connector lifecycle. `OmniChunkTransferAdapter` already owns threads, connector calls, queues, chunk ids, and cleanup. Per-model subclasses would create complicated inheritance or registration problems and make processor reuse harder.

### Alternative B: Raw Dict Only

For example:

```python
transfer_manager.stage_state["moss_tts_raw"][req_id]
```

This is better than dynamic attributes, but it lacks lifecycle hooks, typed construction, and a cleanup API. It can be the internal representation, but should not be the public API.

### Alternative C: Model Processor Global State

Not recommended.

Global state can leak across adapters, stages, and workers. It is harder to clean and does not fit multi-instance deployments.

## Risks

1. Migrating too many processors at once can regress streaming behavior.
2. Over-designed hooks can force processors to implement unnecessary interfaces.
3. Cleaning state at the wrong time can break terminal chunk sending.

Mitigations:

1. Add the store first without behavior migration.
2. Keep hooks optional.
3. Keep sender request cleanup after successful connector `put()`.
4. Add async_chunk tests for first chunk, steady-state chunk, tail flush, abort, and cleanup for each migrated processor.

## Test Plan

1. `StageStateStore` unit tests:
   - Lazy creation through `get()`.
   - Namespace isolation.
   - `cleanup_request()` clears all namespaces.
   - Optional hooks are called.

2. Adapter lifecycle tests:
   - Sender state is cleaned after terminal chunk send.
   - Non-terminal chunk does not clean state.
   - Segment finish calls segment cleanup.
   - Abort cleanup clears state.

3. MOSS-TTS-Local processor tests:
   - Accumulated snapshot sends only delta.
   - `codes.ref` is sent only on the first chunk.
   - Finished flush sends remaining frames.
   - Cleanup prevents cursor reuse for the same request id.

4. Qwen3-TTS regression tests:
   - Prefill zero codes do not enter the frame buffer.
   - Decode frame append still works.
   - Left-context window behavior is unchanged.
   - First-chunk reference context behavior is unchanged.

## Conclusion

The value of `stage_state` is not to move model chunking logic into the framework. It is to formalize the implicit dynamic attributes currently attached to `transfer_manager`.

Recommended ownership:

```text
TransferManager owns lifecycle and namespaced state.
Model processor owns state schema and chunk semantics.
```

This fixes scattered state and incomplete cleanup while preserving per-model control over streaming chunk behavior.
