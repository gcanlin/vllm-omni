# Talker MTP Timeline: MOSS-TTS-Local vs Qwen3-TTS

This note summarizes the decode-time data flow for MOSS-TTS-Local v1.5 and
Qwen3-TTS in vLLM-Omni.

## Key Difference

Both models use the previous backbone hidden state to produce the current audio
codes, but they differ in who owns the audio token generation.

```text
Qwen3-TTS:
  main decode token = layer0 audio code
  talker_mtp        = predicts residual codebooks

MOSS-TTS-Local:
  main decode token = control token, usually audio_assistant_slot / im_end
  talker_mtp        = predicts the full audio frame, all n_vq codebooks
```

This is why Qwen3-TTS mostly needs `last_hidden` and text conditioning, while
MOSS-TTS-Local also needs local generation control such as stop/emit/budget
state.

## MOSS-TTS-Local Timeline

### Prefill

```text
runner
  input_ids
  additional_information:
    codes.ref
    max_new_frames
    ref_offset
        |
        v
model.preprocess
  embeds = text_embedding(input_ids)
  if ref audio codes exist:
    embeds += audio_embedding(ref_codes)
  initialize:
    audio_state.step = 0
    audio_state.is_stopping = False
    audio_state.max_new_frames = request budget
    audio_codes.current = last ref row or pad
    audio_codes.emit = False
        |
        v
backbone forward
  Qwen backbone consumes prompt embeds
  outputs hidden_0
        |
        v
postprocess / runner hidden update
  last_hidden = hidden_0[-1]
```

### Decode Step `i`

```text
runner
  input_ids_i:
    control token from previous vLLM sample
    usually audio_assistant_slot, or im_end when stopping
  state_idx:
    request -> static state slot
  last_hidden_{i-1}:
    runner-owned tensor state
        |
        v
model.preprocess
  text_embed_i = text_embedding(control token)
  mtp_control[:, 0] = active flag
        |
        v
runner.talker_mtp_forward
  passes:
    input_ids_i
    text_embed_i
    last_hidden_{i-1}
    mtp_control
    talker_state_indices
        |
        v
model.talker_mtp
  active =
    mtp_control
    && not finished_prev
    && frame_step < max_new_frames

  should_continue_i, audio_codes_i =
    local_transformer.generate_frame(last_hidden_{i-1})

  emit_i = active && should_continue_i

  input_embed_i =
    text_embed_i + audio_embedding(audio_codes_i) * emit_i

  update static tensor state:
    frame_step += active
    finished = !emit_i
    emit = emit_i
        |
        v
runner
  replace current input embed with input_embed_i
  write audio_codes_i to audio_codes.current
        |
        v
backbone forward
  consumes input_embed_i
  outputs hidden_i
        |
        v
runner hidden update
  last_hidden[state_idx] = hidden_i
        |
        v
model.compute_logits
  does not decide stop from scratch
  only mirrors local stop state into the vLLM main token loop:

    if finished:
      force im_end_token_id
    else:
      force audio_assistant_slot_token_id
        |
        v
vLLM sampler
  samples the forced token
  token becomes next step's control token
        |
        v
model.make_omni_output
  emits audio_codes.current to downstream codec if this step produced audio
```

### MOSS Local Transformer: `generate_frame`

Inside `local_transformer.generate_frame()`:

```text
input:
  backbone_last_hidden: (B, H)
        |
        v
position 0 local transformer
  local_hidden_0 = local_transformer([backbone_last_hidden])[-1]
        |
        v
binary continue/stop head
  binary_logits = local_text_lm_head(local_hidden_0)
  binary_choice = sample(binary_logits)
  should_continue = binary_choice == 0
        |
        v
RVQ codebook autoregression inside one frame

  codebook 0:
    code_0 = sample(audio_lm_heads[0](local_hidden_0))
    embed_0 = audio_embeddings[0](code_0)

  codebook 1:
    local_hidden_1 = local_transformer([backbone_hidden, embed_0])[-1]
    code_1 = sample(audio_lm_heads[1](local_hidden_1))
    embed_1 = audio_embeddings[1](code_1)

  ...

  codebook n_vq-1:
    code_{n_vq-1} = sample(audio_lm_heads[n_vq-1](local_hidden_{n_vq-1}))
        |
        v
output:
  should_continue: (B,)
  audio_codes: (B, n_vq)
```

Important: `should_continue` is sampled by the local transformer binary head.
`compute_logits()` only maps that decision back into vLLM's main token stream.

## Qwen3-TTS Timeline

### Prefill

```text
runner
  input_ids
  additional_information:
    text / prompt / reference audio metadata
        |
        v
model.preprocess / preprocess_batch
  builds prompt embeddings
  builds hidden_states.trailing_text
  initializes meta.talker_text_offset
        |
        v
backbone forward
  Qwen backbone consumes prompt embeds
  outputs hidden_0
        |
        v
postprocess
  hidden_states.last = hidden_0[-1]
```

### Decode Step `i`

```text
vLLM sampler from previous step
  produces layer0 audio code token
        |
        v
runner
  input_ids_i = layer0 audio code
  req_info:
    hidden_states.last
    hidden_states.trailing_text
    meta.talker_text_offset
        |
        v
model.preprocess_decode_batch
  layer0_embed_i = embedding(input_ids_i)
  past_hidden = hidden_states.last
  text_step =
    trailing_text[text_offset] or tts_pad_embed
  update meta.talker_text_offset
        |
        v
runner.talker_mtp_forward
  passes:
    input_ids_i
    layer0_embed_i
    past_hidden
    text_step
        |
        v
model.talker_mtp
  residual_codes_i =
    code_predictor(
      layer0_code=input_ids_i,
      layer0_embed=layer0_embed_i,
      last_talker_hidden=past_hidden,
    )

  audio_codes_i =
    [layer0_code_i, residual_codes_i]

  input_embed_i =
    layer0_embed_i
    + residual_audio_embedding(residual_codes_i)
    + text_step
        |
        v
runner
  replace current input embed with input_embed_i
  write audio_codes_i to codes.audio
        |
        v
backbone forward
  consumes input_embed_i
  outputs hidden_i
        |
        v
postprocess
  hidden_states.last = hidden_i
        |
        v
make_omni_output
  emits codes.audio to downstream codec
```

## Stop Decision Comparison

```text
MOSS-TTS-Local:
  local_text_lm_head decides should_continue.
  compute_logits only forces the vLLM main token to match that decision.

Qwen3-TTS:
  the main token stream directly generates layer0 audio code / stop token.
  talker_mtp does not own a separate continue/stop state machine.
```

## State Ownership

```text
Common runner-side state:
  req_id -> state slot
  talker_state_indices
  last_hidden

MOSS-specific or generation-policy state:
  generated frame count / step
  max_new_frames frame budget
  finished

Ephemeral per-step outputs:
  emit
  current audio codes
```

`max_new_frames` is closer to a TTS sampling/generation budget than to a MOSS
model-intrinsic parameter. It is analogous to `max_tokens`, but counted in audio
frames instead of text tokens.

`emit` and `current_codes` do not fundamentally need to be long-lived state.
They can be treated as current-step outputs. The reason they appear in some
state paths is mostly implementation convenience while moving the hot path from
Python dictionaries to graph-safe tensor buffers.
