# MOSS-TTS-Local v1.5 流式问题排查：从超长音频到并发 zero-duration

> 记录时间：2026-06-29  
> 场景：`OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5` 在 vLLM-Omni 中接入 `/v1/audio/speech` 流式 serving。

这篇文章记录一次 MOSS-TTS-Local v1.5 流式链路的排查过程。问题表面上很杂：流式音频过长、音色和非流式不一致、TTFP 很高、并发 benchmark 偶发 `Audio duration is zero`。最后定位下来，其实是几类边界问题叠在一起：

- Local 流式 codec 不应该把 reference code 再喂给 codec decoder。
- Stage1 codec 收到的是 batch 内拼接后的 flat token，必须有框架级 per-request token count 才能正确切分。
- streaming slot 满时请求会 fallback 到 offline decode，但空 finish payload 可能无法触发 Stage1 flush。
- `codes.audio=[]` 这种空控制包不一定能稳定调度到 Stage1，最终要改成可调度 sentinel。

## 初始现象

最开始的复现方式很简单：

```bash
curl -N http://127.0.0.1:8123/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -o /root/vllm-omni-workspace/moss_local_stream.wav \
  -d '{
    "model": "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
    "input": "hello, I am canlin guo, how are you?",
    "ref_audio": "file:///root/vllm-omni-workspace/vllm-omni/local_v15_test.wav",
    "stream": true
  }'
```

非流式已经正常，但流式有几个明显异常：

- 人声主体能听出来，但音色和非流式有差别。
- 人声不够连续。
- 人声结束后会多出十几秒空白。
- 同一条文本偶现，有时正常，有时超长。

后来跑 benchmark 又看到：

```text
Audio duration is zero
first_chunk_ms=NA
Streaming continuity OK rate 很低
TTFP 仍在 1.8s 左右
```

这说明问题不只是“模型随机没停住”，而是 pipeline 的流式语义和 stage 间控制包处理都可能有问题。

## 先确认一个事实：服务端一直是 async chunk

一个容易误判的点是：服务端是否因为客户端 `stream=true` 才切换流式路径？

实际不是。当前服务端 deploy config 一直开着 async chunk。`stream=true` 只影响 OpenAI API 层怎么把音频回给 client；Stage0 到 Stage1 的 async chunk 链路无论客户端是否 stream 都在跑。

所以如果流式请求出问题，而非流式请求正常，不能简单理解为“服务端切换了另一个模型路径”。更准确地说：

- Stage 间的生成和 codec chunk 路径相同。
- API 层返回方式不同。
- codec 是否增量 decode、是否保持 streaming state、是否正确 flush，会直接影响客户端看到的流式音频。

## 问题一：为什么一开始音频会过长

早期日志里能看到类似内容：

```text
[talker2codec-emit] chunk_shape=(5, 12) code_flat=60 ref_flat=660 finished=False
[codec-input] streaming=True seg_tokens=60 codes_shape=(12, 5) prompt_shape=(12, 55)
[codec-stream-output] mode=step wav_shape=(2, 19200)
```

这里的关键是 `ref_flat=660` 和 `prompt_shape=(12, 55)`。Local-v1.5 的 reference audio 已经在 talker 侧作为 voice cloning 条件参与了生成；codec 阶段的职责是把 talker 生成出来的 audio codes decode 成 waveform。

把 reference code 再喂给 streaming codec，会带来两个问题：

1. 首包要先 prime 一大段 reference code，TTFP 被拉高。
2. streaming codec state 会被 reference audio 影响，和非流式只 decode generated code 的语义不一致。

修复方向是让 Local raw streaming 对齐非流式：codec 只 decode generated audio rows。reference audio 只用于 talker conditioning，不再传给 codec streaming state。

对应修改在 `talker2codec_raw_async_chunk`：

```python
# Raw/local streaming should mirror the non-streaming path: the codec
# decodes only generated audio rows. Reference audio conditions the
# talker, but feeding its codes into the codec streaming state adds a
# long first-packet prime step and changes the decoder state relative
# to non-streaming output.
ref_flat = None
```

这一步解决的是流式和非流式音色/时长语义不一致的问题。

## 问题二：batch 内 flat codec token 怎么切分

随后并发压测时出现更隐蔽的问题。Stage1 codec 的 `input_ids` 不是天然按 request 分好的，它是 batch 内拼接后的 flat tokens。

之前 codec 侧有过类似“按优先级猜”的思路：

```text
meta.code_flat_numel -> seq_token_counts -> num_scheduled_tokens
```

这个方向后来被否掉了。原因很简单：这些字段不应该由模型侧猜。Stage runner 已经知道每个 request 本轮 scheduled 了多少 token，应该通过统一字段明确传给模型。

最终采用的是 `seq_token_counts` 作为 code2wav 的共享契约：

- 正式 request 路径中，`GPUGenerationModelRunner` 把每个 request 的 token 数传给模型。
- stage-engine 路径中，`OmniGPUModelRunner` 从 `_omni_num_scheduled_tokens_np` 构造 `seq_token_counts`。
- dummy/profile run 中，也传一个 synthetic segment，避免 warmup 时 strict codec forward 崩掉。c
- codec forward 不再 fallback 猜测，缺字段直接报错。

codec 侧逻辑变成：

```python
ids_flat = input_ids.reshape(-1).to(dtype=torch.long)
token_counts = self._normalize_seq_token_counts(kwargs.get("seq_token_counts"))
if token_counts is None:
    raise RuntimeError(
        "MossTTS codec requires seq_token_counts; otherwise concatenated "
        "codec tokens cannot be split per request."
    )
if sum(token_counts) != int(ids_flat.shape[0]):
    raise RuntimeError(...)

offsets = [0]
for n in token_counts:
    offsets.append(offsets[-1] + int(n))
```

这个修复的核心不是 MOSS 私有逻辑，而是补齐框架侧 contract：code2wav 模型需要知道 flat `input_ids` 的 per-request 边界。

## 中间踩坑：dummy run 和重复 kwarg

把 `seq_token_counts` 改成 strict 后，马上暴露两个框架边界：

第一个是 profile / dummy run：

```text
RuntimeError: MossTTS codec requires seq_token_counts; otherwise concatenated codec tokens cannot be split per request.
```

原因是 dummy run 也会调用 model forward，但没有真实 request metadata。解决方法是在 dummy run 里传：

```python
model_kwargs["seq_token_counts"] = [int(num_tokens_padded)]
```

第二个是正式路径里重复传参：

```text
TypeError: GPUModelRunner._model_forward() got multiple values for keyword argument 'seq_token_counts'
```

原因是 generation runner 和 omni runner 都可能注入同名 kwarg。解决方法是在 `_model_forward` 合并 omni extra kwargs 前去重：

```python
model_kwargs_extra = self._build_model_kwargs_extra()
for key in tuple(model_kwargs_extra):
    if key in model_kwargs:
        model_kwargs_extra.pop(key)
```

这两个错误都不是模型问题，而是 strict contract 改完后把原来隐含的框架路径暴露了出来。

## 问题三：TTFP 为什么还是下不去

把 chunk 改小、修掉 ref code 后，TTFP 仍然没有明显下降。benchmark 里能看到：

```text
Mean AUDIO_TTFP: ~1800ms
```

这里要区分两个层面的“流式”：

1. Stage0 能否增量把 codes 发给 Stage1。
2. Stage1 codec 是否真正用低延迟 streaming session 增量吐音频。

当前 MOSS codec streaming path 仍有几个现实限制：

- 首包过小会导致声音不清楚，因为 codec 需要足够上下文才能稳定发声。
- chunk 太大会让 TTFP 接近 offline。
- 并发时 streaming slots 可能耗尽，请求 fallback 到 offline decode。
- fallback offline 的请求只有在 finish 时才会吐第一包。

所以看到 `mode=offline` 时，TTFP 接近 E2E 是正常结果。它不是 OpenAI API 没 stream，而是 Stage1 没法在当前 slot 状态下增量 decode。

这也是后续要继续优化的方向：slot 数、Stage1 capacity、codec streaming session、vocoder CUDA graph 和 chunk 策略需要一起调。

## 问题四：并发下偶发 zero-duration

单并发稳定后，4 并发 12 条请求仍偶发：

```text
WARNING [patch.py:1203] Audio duration is zero
```

第一次看日志，典型 request 是：

```text
MOSS codec streaming slots exhausted; buffering speech-80d... for offline decode.
talker2codec-empty-finish req=speech-80d... pending=0
SpeechE2E request_id=speech-80d... status=ok first_chunk_ms=NA
```

这个链路说明：

1. 请求第一帧进入 Stage1 时，streaming slot 满了。
2. codec 把 codes 暂存在 `_stream_pending_codes`，准备 final 时 offline decode。
3. Stage0 后续结束时没有新 code，只发了 empty finish。
4. codec empty finish 分支只 cleanup，没有把 pending codes decode 出来。
5. API 层没有收到任何 audio chunk，bench 看到 audio duration 为 0。

于是第一版修复是在 codec 里加 `_finish_empty_streaming_requests()`：如果 empty finish 到达时发现该 request 有 pending codes，就用 offline decode flush 出最后一包，然后释放状态。

核心逻辑：

```python
if pending:
    full_codes = self._pop_stream_pending(req_key)
    decode_codes, prompt_frames = self._with_stream_prompt(req_key, full_codes)
    wavs = session.decode_offline([decode_codes], max_step_frames=max_step_frames)
    if wavs:
        outputs[i] = self._trim_prompt_audio(wavs[0], prompt_frames)
```

新日志会显示：

```text
[codec-stream-output] req=... mode=offline-empty-finish ...
[codec-empty-finish] req=... had_pending=True
```

## 最后一层坑：空 payload 不一定会调度 Stage1

第一版修复后，又跑到一次 zero-duration。新的日志显示：

```text
talker2codec-empty-finish req=speech-82ad... pending=0
SpeechE2E request_id=speech-82ad... status=ok first_chunk_ms=NA
```

但没有看到：

```text
codec-empty-finish
offline-empty-finish
```

这说明 empty finish payload 根本没进入 Stage1 model forward。

当时 `talker2codec-empty-finish` 返回的是：

```python
CodesStruct(audio=[])
MetaStruct(code_flat_numel=0, finished=True, stream_finished=True)
```

从语义上它是对的，但从调度角度它是 0-token payload。Stage engine 不保证这种 payload 一定触发 Stage1 forward。也就是说，codec 里写再多 empty input 分支，如果 Stage1 没被调度，就没有机会执行。

最终修复是把 empty finish 改成非空 sentinel：

```python
CodesStruct(audio=torch.tensor([0], dtype=torch.long))
MetaStruct(
    code_flat_numel=0,
    finished=True,
    stream_finished=True,
)
```

`audio=[0]` 只用于让 Stage1 被调度；真正语义由 `code_flat_numel=0` 表达：这是控制包，不是 audio code。

codec 收到后在 `% n_vq` 检查前识别 sentinel：

```python
if (
    streaming_enabled
    and finished
    and code_flat_numel is not None
    and int(code_flat_numel) == 0
):
    for _, wav in self._finish_empty_streaming_requests([info]).items():
        audios[i] = wav.reshape(-1) if wav.ndim == 1 or int(wav.shape[0]) == 1 else wav
    continue
```

这一步才真正闭环了 “slot exhausted -> pending offline -> empty finish -> flush pending audio”。

## 修复后的关键日志

修复后，遇到 slot 满的请求应该有两种正常结局。

如果后续 final chunk 仍有 code：

```text
MOSS codec streaming slots exhausted; buffering req for offline decode.
[codec-stream-output] req=... mode=offline codes_shape=(12, N) wav_shape=(2, T) finished=True
[api-stream-audio] req=... final_shape=(2, T)
```

如果 final chunk 是 empty finish：

```text
[talker2codec-empty-finish] req=... pending=0
[codec-stream-output] req=... mode=offline-empty-finish codes_shape=(12, N) wav_shape=(2, T)
[codec-empty-finish] req=... releasing_slot=False had_pending=True
[api-stream-audio] req=... final_shape=(2, T)
```

不应该再出现同一 request 的：

```text
first_chunk_ms=NA
Audio duration is zero
```

## 经验总结

这次排查最有价值的地方，不是某一个 if 分支，而是几条边界原则。

第一，流式和非流式必须共享同一套音频语义。Local-v1.5 的 reference audio 是 talker conditioning，不应该在 streaming codec 里再 decode 一遍。

第二，框架已经知道的信息，不要让模型猜。`seq_token_counts` 应该作为 code2wav 的通用 contract 由 runner 传入，codec 严格校验，避免 batch 内 request 切错。

第三，控制包也要满足调度约束。`codes.audio=[]` 在语义上是 empty finish，但在 scheduler 看来可能是 0-token，不一定会跑 Stage1 forward。需要用可调度 sentinel 承载控制语义。

第四，slot exhausted 不是错误，但必须有完整 fallback。只要请求进入 `_stream_pending_codes`，后面无论 final chunk 有 code 还是 empty finish，都必须能 flush 出音频或者明确释放状态。

第五，TTFP 不只由 API streaming 决定。客户端 `stream=true` 只能让 API 边收边写；真正首包延迟取决于 Stage0 是否早发 chunk、Stage1 codec 是否能增量 decode、slot 是否充足、chunk size 是否合理。

## 后续优化方向

当前修复主要解决正确性。性能上还有明显空间：

- Stage1 streaming slots 和 `max_num_seqs` 需要按并发目标调优。
- `initial_codec_chunk_frames` 可以小于稳态 `codec_chunk_frames`，降低首包。
- codec streaming session 需要更稳定的 slot lifecycle 和更好的 batch coalescing。
- fallback offline decode 要监控比例；如果高并发下大量 fallback，TTFP 会自然退化到 E2E。
- benchmark 里应该同时看 `first_chunk_ms=NA`、`mode=offline` 比例、`audio_duration`、`audio_underrun`，不要只看平均 TTFP。

这次修复后，MOSS-TTS-Local v1.5 的流式链路至少具备了正确的状态闭环：正常 streaming 能增量出音频，slot 满时能 fallback，空 finish 也能触发 pending flush。后续再做低 TTFP 和高并发优化，才有稳定的基础。
