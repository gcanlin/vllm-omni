# MOSS-TTS Streaming `exec_mask` Optimization Note

## Context

MOSS-TTS Local v1.5 codec streaming decode now uses CUDA graph replay for the
main `_decode_frame` path. In profiling, steady chunks such as `T=15` are
captured by CUDA graph, but each chunk still has graph-external overhead before
`cudaGraphLaunch`.

One visible source is the current `exec_mask` update path:

```python
self.codec._set_streaming_exec_mask(exec_mask)
entry.static_codes.copy_(codes_step)
entry.graph.replay()
```

Here `T` is the codec step sequence length in audio frames. For the current
streaming config, steady state uses `T=15`, while first and tail chunks may use
smaller `T`.

## Current Behavior

`_set_streaming_exec_mask()` traverses the codec module tree and updates every
live streaming state:

```python
def _set_streaming_exec_mask(self, exec_mask):
    def _set(module):
        if isinstance(module, StreamingModule) and module._streaming_state is not None:
            module._streaming_state.set_exec_mask(exec_mask.to(module._streaming_state.device))

    self.apply(_set)
```

Each `StreamingState` then copies the mask into its own tensor:

```python
def set_exec_mask(self, exec_mask):
    self.exec_mask[:] = exec_mask
```

Although `exec_mask` is only shaped like `[codec_stream_slots]`, for example
`[4]`, it is copied once per streaming state. This creates many tiny device
copies / device ops before graph replay. These operations are outside the
captured graph, so CUDA graph cannot remove their launch overhead.

In a profiler timeline this can show up as:

- a block of small D2D operations before `cudaGraphLaunch`
- CPU gaps between tiny launches
- time attributed to the `set_exec_mask` Python stack

## Proposed Direction

Make `exec_mask` shared and persistent per streaming session.

Instead of giving every streaming state its own independent `exec_mask` tensor:

```python
state_0.exec_mask
state_1.exec_mask
state_2.exec_mask
...
```

create one session-level tensor:

```python
session_exec_mask: torch.BoolTensor[codec_stream_slots]
```

and make all streaming states reference it:

```python
state_0.exec_mask -> session_exec_mask
state_1.exec_mask -> session_exec_mask
state_2.exec_mask -> session_exec_mask
```

Then each streaming step only needs one mask update:

```python
session_exec_mask.zero_()
session_exec_mask[active_slots] = True
entry.static_codes.copy_(codes_step)
entry.graph.replay()
```

All graph-captured codec kernels will read the same shared mask tensor.

## Reset Semantics

The current `StreamingState.reset()` also writes `exec_mask`:

```python
self.exec_mask[:] = torch.where(reset_mask, torch.ones_like(self.exec_mask), self.exec_mask)
```

If `exec_mask` becomes shared, reset should not mutate it per state. A cleaner
contract is:

- `exec_mask` means: which stream slots are active in the current decode step
- reset means: clear KV / offsets / per-slot streaming state
- the next decode step is responsible for setting `exec_mask`

So reset should update codec state such as offsets and KV cache, but not restore
or modify the shared `exec_mask`.

## Expected Benefit

This should reduce graph-external overhead in steady streaming chunks by
removing repeated tiny mask copies across all streaming modules. It does not
change model math or codec output; it only changes how the per-step active-slot
mask is stored and updated.

This is orthogonal to CUDA graph capture sizes. Dense capture sizes such as
`1..codec_chunk_frames` solve tail-chunk graph misses, while shared `exec_mask`
targets the remaining pre-graph state-update overhead.
