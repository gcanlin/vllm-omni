# Omni Generation Runner Lifecycle and Payload Token Counts

This note records a design direction for vLLM-Omni generation/code2wav runners. The goal is to align with vLLM runner concepts instead of introducing a parallel `RequestInputSpan` abstraction.

## Problem

Stateful generation models such as streaming codec/vocoder stages need two contracts that are currently implicit:

1. When a request leaves the runner's active state, the model must release request-local resources such as stream slots, pending chunks, temporary generators, or deferred GPU buffers.
2. For code2wav stages, the model needs a per-request logical payload length for splitting flattened `input_ids`. This length is not always the scheduler token count and must not be the CUDA graph padded tensor length.

Today these concepts are scattered across:

- `SchedulerOutput.num_scheduled_tokens`
- `SchedulerOutput.finished_req_ids`
- `SchedulerOutput.preempted_req_ids`
- `scheduled_cached_reqs.resumed_req_ids`
- `query_start_loc` / `omni_query_start_loc`
- `request_token_spans`
- `seq_token_counts`
- `runtime_additional_information` / `model_intermediate_buffer`
- payload meta fields such as `code_flat_numel`, `left_context_size`, `stream_finished`, and `codec_chunk_frames`

## Align With vLLM

vLLM already defines the important runner-level concepts:

- `num_scheduled_tokens` describes the scheduler token work for each request in the current step.
- `query_start_loc` describes flat tensor row boundaries for attention/model execution.
- `batch_descriptor`, `ubatch_slices`, and padded token counts describe physical execution under CUDA graphs.
- `finished_req_ids` and `preempted_req_ids` describe requests that must leave the current runner state.
- `resumed_req_ids` describes requests returning after preemption.

vLLM v2 runner treats preempted requests as removed from runner state. Resume is handled by adding the request back later. vLLM-Omni generation runners should follow that lifecycle instead of treating preemption as a model-specific special case.

## Request Lifecycle Hook

The model hook should describe runner state removal, not only normal completion.

Use:

```python
model.on_requests_removed(req_ids)
```

Do not use:

```python
model.on_requests_finished(req_ids)
```

because the same cleanup path is needed for:

- normal finish
- client abort / disconnect
- engine-side cancellation
- scheduler preemption
- orphaned unscheduled requests left in the persistent batch

Models should interpret `on_requests_removed` as:

> These requests are no longer active in this runner state. Release or defer-release request-local resources owned by the model.

If a model must preserve resumable state across preemption, that should become an explicit future lifecycle contract. The current stateful codec/vocoder use cases need cleanup on removal.

## Payload Token Counts

Code2wav stages consume flattened payload tokens. They need a logical per-request payload length to split `input_ids`.

There are three different lengths:

| Length | Meaning |
| --- | --- |
| scheduler token count | transport/scheduler work for this step |
| physical tensor length | actual `input_ids` rows passed to forward, possibly CUDA graph padded |
| logical payload token count | model-semantic payload tokens, for example codec code tokens |

The MOSS local first packet demonstrates the difference:

```text
logical codec tokens: 12
physical input_ids rows under graph: 16
```

A finish sentinel demonstrates another difference:

```text
scheduler token count: 1
logical codec tokens: 0
```

Therefore code2wav models should split by `seq_token_counts`, and the runner should normalize `seq_token_counts` from payload metadata when available:

```text
payload logical token count -> seq_token_counts -> code2wav split
```

For backward compatibility with existing payloads, MOSS currently uses:

```text
meta.code_flat_numel
```

A cleaner framework field would be:

```text
meta.payload_token_count
```

Then the runner can resolve:

```text
payload_token_count ?? code_flat_numel ?? scheduler_count
```

The physical padded tensor length should only be used as an upper bound check, never as the logical count.

## Recommended Migration

1. Keep `seq_token_counts` as the model-facing split contract for code2wav stages.
2. Add a generic `MetaStruct.payload_token_count` field.
3. Teach generation runner to derive `seq_token_counts` from payload metadata:

```text
payload_token_count / code_flat_numel if present and valid
otherwise scheduler num_scheduled_tokens
```

4. Gradually remove model-side reads of `meta.code_flat_numel` for splitting.
5. Keep codec-specific chunk metadata in payload meta:

```text
left_context_size
right_holdback_size
stream_finished
codec_chunk_frames
codec_left_context_frames
```

These are payload semantics, not vLLM runner span semantics.

## Non-Goals

Do not introduce a parallel `RequestInputSpan` abstraction at this stage. It overlaps with vLLM's existing `query_start_loc`, `num_scheduled_tokens`, and physical graph padding concepts, and would likely confuse contributors.

The better direction is to:

- align lifecycle cleanup with vLLM runner removal semantics;
- standardize logical payload token counts as the source of `seq_token_counts`;
- keep vLLM execution spans and Omni payload semantics separate.
