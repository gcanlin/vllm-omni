# MOSS-TTS Local Last Hidden Runner Buffer Experiment

## Context

MOSS-TTS Local v1.5 runs a stateful local depth transformer in `talker_mtp`.
Every decode step needs the previous backbone hidden state as
`last_talker_hidden`.

The original path stores this value in per-request `model_intermediate_buffer`
via:

1. `model.postprocess(hidden_states)` returns `{"hidden_states": {"last": ...}}`.
2. `GPUModelRunner._update_intermediate_buffer()` stores it under the request.
3. Next decode step, `preprocess_decode_batch()` iterates over `req_infos`,
   reads `hidden_states.last` per request, applies `.to(device, dtype)`, then
   `torch.cat()`s rows into the batched `last_talker_hidden`.

In profiling, this path showed CPU overhead and CUDA synchronization around
`preprocess_decode_batch`, especially under concurrent decode.

## Experiment

The experiment changed MOSS Local to keep `last_hidden` directly in the
runner-owned contiguous GPU buffer `last_talker_hidden.gpu`.

The intended flow was:

1. MOSS Local declares capability flags:
   - `preprocess_decode_batch_accepts_runner_last_hidden = True`
   - `postprocess_last_hidden_to_talker_mtp_buffer = True`
2. Runner postprocess bypasses `model.postprocess()` for hidden-state updates.
3. Runner writes each request's last hidden directly to:
   - `self.last_talker_hidden.gpu[req_index:req_index + 1]`
4. Next decode step, `flush_decode_batch()` gathers the previous hidden batch
   from `self.last_talker_hidden.gpu` and passes it to
   `preprocess_decode_batch(last_talker_hidden=...)`.
5. MOSS `preprocess_decode_batch()` uses that tensor directly instead of
   iterating `req_infos` and calling `torch.cat(hidden_rows)`.

## Expected Benefit

The expected benefit was to remove the hot-path sequence:

```text
model_intermediate_buffer per request
  -> Python list of hidden rows
  -> per-row .to(device, dtype)
  -> torch.cat(hidden_rows)
  -> copy into runner buffer
```

For pure decode batches, this should reduce CPU overhead and some small GPU
copies before `talker_mtp`.

## Problems Observed

The first implementation hit an overlap bug in mixed prefill/decode batches:

```text
RuntimeError: unsupported operation: some elements of the input tensor and the
written-to tensor refer to a single memory location.
```

The reason was that decode requests can be in active request slots such as
`[1:5]`, while `talker_mtp` packs them into destination rows `[0:4]`. A direct
copy from `last_talker_hidden.gpu[1:5]` to `last_talker_hidden.gpu[0:4]`
partially overlaps. Cloning the source avoids the error, but adds a conditional
path and more complexity.

More importantly, the experiment made runner state semantics less obvious:

- `last_talker_hidden.gpu` became both a packed `talker_mtp` input buffer and a
  per-request persistent state buffer.
- The mapping depends on current `input_batch.req_id_to_index`, which can change
  as requests are added, finished, or reordered.
- Mixed prefill/decode batches require careful source/destination handling.
- The optimization interacts with vLLM async scheduling and
  `synchronize_input_prep()` timing, making performance attribution harder.

## Decision

The experiment is being reverted for now.

The safer current design is to keep `hidden_states.last` in
`model_intermediate_buffer`, while retaining the other independent MOSS Local
batch optimizations:

- batched decode preprocess,
- avoiding unnecessary `input_ids` int32 -> int64 conversion,
- avoiding empty per-request update merges,
- batch-local `talker_mtp` outputs for current audio codes and
  `should_continue`.

## Future Direction

If this optimization is revisited, the runner should separate the two concepts:

- a per-request persistent hidden-state store indexed by stable request slot,
- a packed `talker_mtp` input buffer indexed by current decode batch row.

The transfer from persistent store to packed buffer should be explicit and
overlap-safe. It should also be instrumented with NVTX ranges around:

- postprocess hidden-state write,
- persistent-to-packed gather,
- MOSS `preprocess_decode_batch`,
- `talker_mtp_forward`.

Only after the slot ownership and async-scheduling boundaries are clear should
this be reintroduced.
