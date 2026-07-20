# MOSS-TTS-Local v1.5 调试复盘

本文记录一次 MOSS-TTS-Local v1.5 接入 vLLM-Omni 过程中，定位“输出噪声”和“请求卡住且没有日志”的完整过程。

结论先行：

- 输出噪声的主因是 vendored MOSS Audio Tokenizer v2 实现不完整，缺少官方 v2 的 stereo channel interleave / restore 逻辑。
- 后续非流式 curl 卡住、且看不到新增日志，并不是同一个推理链路问题，而是请求打到了错误端口或旧 server 进程。
- 端口问题解决后，更新后的 vendored tokenizer 路径能输出正确音频。

## 背景

MOSS-TTS-Local v1.5 是一个两阶段 TTS pipeline：

1. Stage 0: talker
   - 输入文本和 ref audio conditioning。
   - 输出 audio codec frame，也就是离散 audio code。

2. Stage 1: codec / vocoder
   - 输入 audio code。
   - 使用 MOSS Audio Tokenizer v2 decode 成 waveform。

框架侧通过 async chunk transfer 将 Stage 0 产生的增量 code frame 传给 Stage 1。理论上，接入一个“非新架构”的模型时，应该尽量只在模型适配层处理模型特有逻辑，不应该在 chunk transfer framework 里挂很多模型私有 state。

## 初始现象

最初的问题有两个：

1. 服务能启动，codec 权重也显示完整加载，但生成音频是噪声，后来有一点人声轮廓但仍然听不清。
2. 某些 curl 请求会一直卡住，而且 server 侧看不到我们后续加的 debug 日志。

这两个现象一开始容易混在一起看，但最后证明它们是两个独立问题。

## 第一阶段：先排除权重加载问题

早期日志里曾出现过：

```text
MOSS Audio Tokenizer weights: loaded=2084/2084 skipped=10
first skipped: ['encoder.11.input_proj.weight', ...]
```

这说明当时的 tokenizer module 结构和 checkpoint key 并不完全匹配。后续修正 weight remap 后，日志变成：

```text
MOSS Audio Tokenizer weights: loaded=2094/2094 skipped=0
MOSS Audio Tokenizer loaded: sampling_rate=48000, n_vq=32, n_channels=2
```

这一步排除了“明显少加载权重”的问题，但音频仍然不对，所以继续往数据语义查。

## 第二阶段：确认 codec 输入输出形状

我们在 talker2codec 和 codec decoder 之间加了几类 debug 日志：

- `talker2codec-input`
- `talker2codec-emit`
- `codec-input`
- `codec-output`
- `api-final-audio`

当时观察到的关键形状是：

```text
talker2codec-input input_shape=(1, 12), prompt_shape=(55, 12)
talker2codec-emit chunk_shape=(5, 12), code_flat=60, ref_flat=660
codec-input codes_shape=(12, 5), prompt_shape=(12, 55)
codec-output wav_shape=(1, 38400)
api-final-audio raw_shape=(261120,), final_shape=(2, 130560)
```

这里有一个明显矛盾：

- codec config 里 `n_channels=2`
- 但 codec 实际输出 `wav_shape=(1, T)`
- API 层又把 flat 1D audio 猜测性 reshape 成 `(2, T/2)`

如果 tokenizer 内部本来输出的是“interleaved mono stream”，API 层直接切半成 stereo 会破坏左右声道排列，结果就是音频听起来像噪声。

所以当时先把 API 层的 `_maybe_restore_moss_local_stereo()` 改成 no-op，避免继续错误 reshape。但这只是止损，根因还在 codec/tokenizer。

## 第三阶段：对比官方 MOSS Audio Tokenizer v2

用户下载了官方 v2 代码：

```text
/mnt/data1/huggingface/hub/models--OpenMOSS-Team--MOSS-Audio-Tokenizer-v2/snapshots/f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169/modeling_moss_audio_tokenizer.py
```

对比后发现官方 v2 有这些关键逻辑，而我们 vendored 版本缺失：

```python
self.number_channels = config.number_channels
self.enable_channel_interleave = getattr(config, "enable_channel_interleave", True)
```

官方构建 encoder 时会把 frame rate 乘以 channel interleave factor：

```python
channel_interleave_factor = (
    self.number_channels if self.enable_channel_interleave and self.number_channels > 1 else 1
)
current_frame_rate = float(self.sampling_rate * channel_interleave_factor)
```

官方 encode 前会把 `(B, C, T)` flatten 成 codec 内部的 `(B, 1, T*C)`：

```python
input_values = input_values.transpose(1, 2).contiguous().view(input_values.shape[0], 1, -1)
input_lengths = input_lengths * self.number_channels
```

官方 decode 后会把 codec 内部输出 restore 回 `(B, C, T)`：

```python
output_values = (
    output_values.squeeze(1)
    .contiguous()
    .view(output_values.shape[0], -1, self.number_channels)
    .transpose(1, 2)
    .contiguous()
    .float()
)
output_lengths = torch.div(output_lengths, self.number_channels, rounding_mode="floor")
```

这正好解释了我们看到的现象：v2 checkpoint 是双声道模型，但 vendored model 按单声道路径 decode，API 再事后猜 stereo，导致波形语义错位。

## 修复方向

没有整文件替换官方 `modeling_moss_audio_tokenizer.py`，因为我们 vendored 版本里已有 vLLM-Omni 适配逻辑，例如 streaming execution mask。实际采用的是小范围同步官方 v2 的数据语义：

1. 在 vendored config 中补齐 v2 字段：
   - `number_channels`
   - `enable_channel_interleave`
   - `attention_implementation`
   - `compute_dtype`
   - `codec_weight_dtype`
   - 兼容旧字段别名，例如 `channels_numbers`

2. 在 vendored tokenizer model 中补齐：
   - channel interleave aware 的 frame rate 计算
   - `_prepare_waveform_batch`
   - `_prepare_codes_batch`
   - `_flatten_channels_for_codec`
   - `_restore_channels_from_codec`

3. 修改 `_encode_frame()`：
   - 正确处理 `(T,)`, `(C, T)`, `(B, C, T)`
   - 对 stereo 输入执行官方 flatten 逻辑

4. 修改 `_decode_frame()`：
   - decoder stack 输出后执行官方 restore 逻辑
   - streaming decode session 直接拿到 `(B, 2, T)`，不再依赖 API 层猜测

5. API 层保持不再猜测 stereo：
   - codec 如果输出 `(2, T)`，直接保留。
   - codec 如果输出 `(1, T)`，不要在 API 层硬拆成 `(2, T/2)`。

修完后的期望日志是：

```text
MOSS Audio Tokenizer loaded: sampling_rate=48000, n_vq=32, n_channels=2
codec-output wav_shape=(2, ...)
api-final-audio raw_shape=(2, ...), final_shape=(2, ...)
```

## async chunk 相关判断

这次还顺带确认了 talker 到 codec 的 chunk 语义。

MOSS talker 内部原本维护 accumulated code，是为了模型自身的历史上下文和重复检测。框架 transfer 层不应该再挂一个 MOSS 私有 state 去维护累计快照。

最终选择：

- talker 内部仍可维护 accumulated，用于模型语义。
- talker 对外输出 current/new frame。
- `talker2codec_raw_async_chunk` 按通用 async chunk 逻辑自己 buffer 增量 frame，到达 chunk size 后传给 codec。
- 不再把完整 accumulated frame 每次传给 chunk transfer。

这样对齐了 Qwen3-TTS 等模型的通用模式，也避免 transfer manager 里出现 `_moss_tts_raw_state` 这种模型私有字段。

## 请求卡住问题

后来出现一个新现象：

```bash
curl -v http://127.0.0.1:8091/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
    "input": "hello, I am canlin guo, how are you?",
    "ref_audio": "file:///root/vllm-omni-workspace/vllm-omni/local_v15_test.wav"
  }' \
  --output /root/vllm-omni-workspace/moss_local_stream.wav
```

curl 显示已经连接，并上传了 body，但长时间没有响应。同时 server 日志里没有：

- `[MossTTSDebug][create-speech-entry]`
- `TTS speech request ...`
- talker2codec debug
- codec debug

这说明请求很可能没有进入我们以为的那条 serving path。

为了定位，我们又加了两层更早的日志：

1. ASGI 最外层 HTTP wrapper：

```text
[MossTTSDebug][asgi-http-enter]
[MossTTSDebug][asgi-http-response-start]
```

2. `/v1/audio/speech` route 函数第一行：

```text
[MossTTSDebug][api-create-speech-route]
```

判断逻辑是：

- 如果没有 `asgi-http-enter`，请求没有进入当前这份 app/server。
- 如果有 `asgi-http-enter` 但没有 `api-create-speech-route`，卡在 FastAPI/Pydantic/dependency/decorator 前置层。
- 如果有 route 日志但没有 serving 日志，卡在 handler 初始化或 `_check_model` 附近。
- 如果都有，再看 stage/talker/codec。

最后用户换端口后请求可以正常完成，并且输出正确。这说明此前卡住并不是 MOSS 推理链路问题，而是端口上存在旧进程、错误 server、或日志文件和实际请求进程不对应。

## 最终根因拆分

这次其实有两个根因：

### 1. 音频质量问题

根因：vendored MOSS Audio Tokenizer v2 缺少官方 v2 的 stereo channel interleave / restore 逻辑。

影响：

- codec config 显示双声道。
- decoder 实际按单声道输出。
- API 层错误 reshape，破坏声道排列。
- 生成音频表现为噪声或听不清的人声。

修复：

- 同步官方 v2 channel flatten/restore 语义。
- API 层不再猜测 stereo。

### 2. 请求卡住且无日志

根因：请求打到的不是预期的新 server 进程或新端口。

影响：

- curl 已连接但 server 日志中没有 route / serving / stage debug。
- 容易误判为模型推理卡住。

修复：

- 换端口后请求进入正确 server。
- 输出正常，说明模型链路已经修复。

## 经验

1. 先看“日志有没有进入入口”，再看模型。

   如果连 route 第一行日志都没有，不要直接怀疑模型 forward、scheduler 或 codec。

2. 对 codec 问题，shape 比主观听感更可靠。

   `n_channels=2` 但 `wav_shape=(1, T)` 是强信号。音频噪声通常不是“随机坏”，而是某个数据排列、采样率、声道或 codebook 顺序错了。

3. vendored remote code 时，不能只看权重数量。

   `loaded=2094/2094` 只能说明参数名和 shape 匹配，不代表数据前后处理语义匹配。MOSS Audio Tokenizer v2 的 channel interleave 属于无参数逻辑，权重加载日志无法发现它缺失。

4. 不要在 API 层猜测模型语义。

   如果 codec 是双声道模型，应由 codec 返回双声道 waveform。API 层看到 flat tensor 后自行 reshape，风险很高。

5. 框架 state 不应该承载模型私有语义。

   MOSS talker 内部 accumulated 是模型语义；transfer manager 里再挂 `_moss_tts_raw_state` 是框架污染。更好的边界是：模型输出增量 frame，框架只做通用 chunk buffer。

6. 端口和进程要作为排查的一等对象。

   当“加了日志但完全看不到”时，优先确认：

   - server 是否真的重启。
   - 请求是否打到同一个端口。
   - 端口上是否有旧进程。
   - 当前查看的 log 是否对应实际进程。

## 后续清理建议

这次为定位问题加了不少 debug 日志。问题确认后建议保留少量高价值日志，删除临时排查日志：

- 可保留：
  - codec weight fully loaded 日志
  - tokenizer config 中 `sampling_rate / n_vq / n_channels`
  - codec output shape 的 debug，至少在开发阶段保留

- 可删除或降级：
  - ASGI route 入口日志
  - API route 第一行日志
  - 过细的 processor/ref audio 构建日志

这样既保留未来定位 codec 语义问题的能力，也避免正常服务日志过噪。
