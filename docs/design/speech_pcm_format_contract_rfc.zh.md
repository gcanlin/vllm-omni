# RFC: Speech PCM Format Contract

## 背景

vLLM-Omni 的 `/v1/audio/speech` 同时服务多种 TTS 模型。大多数模型输出 24 kHz mono 音频，但 MOSS-TTS Local v1.5 使用 MOSS-Audio-Tokenizer-v2，原生输出 48 kHz stereo waveform。

当前 serving 和 benchmark 之间缺少一个明确的 raw PCM format contract。对 `response_format=pcm` 的请求，客户端只能拿到一段裸 PCM bytes；如果没有 sample rate、channel count、sample width 等元信息，benchmark 只能用默认假设解释这些 bytes。这个默认假设对 24 kHz mono 模型通常能工作，但对 MOSS Local 的 48 kHz stereo 输出会把音频读错，导致保存音频和 WER 评测都失真。

本文建议补齐 speech PCM format contract，让服务端、benchmark、WER 评测都基于显式音频格式工作。目标是修复 MOSS Local 的 stereo/48 kHz 评测问题，同时避免把 MOSS 特判扩散到 Seed-TTS WER 代码里。

## 当前问题

### MOSS Local 的真实输出

MOSS-TTS Local v1.5 的 codec stage 解码后输出通常是：

```text
waveform shape: [2, T]
sample_rate:    48000
dtype:          float32
layout:         channel-first, stereo
```

serving 最终会把 float waveform 编码成 s16le PCM。对于 stereo PCM，字节语义是 interleaved frame：

```text
L0 R0 L1 R1 L2 R2 ...
```

如果客户端知道这是 `sample_rate=48000, channels=2, sample_width=2`，这段 PCM 是可正确播放、保存和转码的。

### benchmark 当前看到的数据

`benchmarks/tts/bench_tts.py` 通过 `vllm-omni bench serve` 请求 `/v1/audio/speech`。在 streaming path 中，请求使用 PCM 响应以便统计首包和 chunk timing。

benchmark capture 层当前拿到的是：

```python
pcm_bytes: bytes
```

但如果 response header 或 capture metadata 没有告诉它真实格式，后续逻辑会按默认格式解释：

```text
sample_rate = 24000
channels    = 1
sample_width = 2
```

这对 MOSS Local 是错误的。48 kHz stereo PCM 被当成 24 kHz mono PCM 后会出现两个问题：

- 左右声道交错样本被当成一条 mono waveform。
- 48 kHz 被按 24 kHz 解释，时长、音高、速度都会错。

因此，保存出来的 eval WAV 可能听起来像噪声或严重变形，Seed-TTS WER 也会失去意义。

### 当前临时修复的问题

一个局部修复方式是在 WER 入口增加类似：

```python
_pcm_s16le_to_seed_tts_wer_bytes(
    pcm_bytes,
    sample_rate=48000,
    channels=2,
)
```

它可以把 MOSS Local 的 raw PCM downmix/resample 成 WER 需要的 24 kHz mono PCM。但这仍然是补丁式做法：

- 函数名和职责容易变成 Seed-TTS/MOSS 专属逻辑。
- 如果未来有 44.1 kHz stereo、48 kHz mono、不同 sample width 的模型，还会继续加分支。
- 根因没有消失：raw PCM bytes 本身没有携带格式 contract。

## SGLang-Omni 的做法

SGLang-Omni 的关键不是在 WER 里猜格式，而是在音频链路上一直保留格式语义。

MOSS Local codec 输出 waveform payload 时保留 channel 维度：

```python
audio_waveform_payload(
    audio,
    sample_rate=48000,
    keep_channels=True,
)
```

非流式 benchmark 默认请求 WAV。服务端写 WAV header，header 中包含真实 sample rate 和 channel count：

```text
WAV header:
  sample_rate = 48000
  channels    = 2
```

流式 PCM 路径也会通过 HTTP header 明确格式：

```text
Content-Type: audio/pcm
X-Sample-Rate: 48000
X-Channels: 1 or 2
X-Bit-Depth: 16
```

benchmark 解析 header 后再把 PCM 重新封装成 WAV。WER/ASR 读取的是 WAV 文件，由 audio loader 正确处理 downmix 和 resample。

这个设计的要点是：raw PCM 可以存在，但 raw PCM 的格式必须显式传递。

## 目标

1. `response_format=pcm` 的 response 必须携带真实音频格式。
2. benchmark capture 不再硬编码 24 kHz mono。
3. 保存到 benchmark result 的 WAV 必须能被普通播放器和 ASR 正确读取。
4. Seed-TTS WER 只消费标准化音频，不包含 MOSS 专属格式猜测。
5. 改动保持小粒度，不重构整个 TTS serving pipeline。

## 非目标

1. 不在本 RFC 中重写 `_generate_audio_chunks()` 的完整 AudioChunk contract。那属于 `audio_chunk_contract_rfc.zh.md` 的范围。
2. 不改变模型输出质量、采样策略或 stop 策略。
3. 不要求所有模型都输出 mono。MOSS Local 的 stereo 输出应该可以被保留。
4. 不引入复杂的音频容器协商。当前只需要覆盖 WAV 和 raw PCM。

## 设计

### 1. Serving 侧定义 PCM response header contract

当 `/v1/audio/speech` 返回 `response_format=pcm` 时，response 必须包含：

```text
Content-Type: audio/pcm
X-Sample-Rate: <positive int>
X-Channels: <positive int>
X-Bit-Depth: 16
```

对当前模型，典型值为：

| 模型 | X-Sample-Rate | X-Channels | X-Bit-Depth |
| --- | ---: | ---: | ---: |
| Qwen3-TTS | 24000 | 1 | 16 |
| Higgs Audio TTS | 24000 | 1 | 16 |
| MOSS-TTS Local v1.5 | 48000 | 2 | 16 |

如果 serving 层选择对某个模型 downmix 成 mono，则必须在编码 PCM 前实际 downmix，并返回 `X-Channels: 1`。header 必须描述 body 的真实格式，而不是模型原始格式。

### 2. Serving 侧统一推断 channel count

在把 waveform 编码成 PCM 之前，需要根据标准化后的 waveform shape 推断 channel count：

```text
[T]          -> channels = 1
[1, T]       -> channels = 1
[2, T]       -> channels = 2
[T, 1]       -> channels = 1
[T, 2]       -> channels = 2
```

建议 serving 内部先把 waveform 规范化到以下两种之一：

```text
[T]       # mono
[C, T]    # channel-first multi-channel
```

然后再编码：

```text
float32 waveform -> interleaved s16le PCM
```

### 3. 非流式 WAV 使用真实格式写 header

当 `response_format=wav` 时，服务端应该用真实 sample rate 和 channel count 写 WAV header。

MOSS Local 非流式应返回：

```text
sample_rate = 48000
channels    = 2
sample_width = 2
```

这样 benchmark 或 ASR 读取 WAV 时不需要任何模型知识。

### 4. Benchmark capture 解析 PCM header

benchmark 的 audio-speech request function 在收到 `audio/pcm` response 时，必须解析：

```python
sample_rate = int(headers.get("x-sample-rate", 24000))
channels = int(headers.get("x-channels", 1))
bit_depth = int(headers.get("x-bit-depth", 16))
sample_width = bit_depth // 8
```

并用这些元信息计算：

```python
duration_s = len(pcm_bytes) / (sample_rate * channels * sample_width)
```

而不是默认：

```python
duration_s = len(pcm_bytes) / (24000 * 1 * 2)
```

### 5. Benchmark 保存 WAV 时使用真实格式

benchmark 如果需要保存 streaming PCM 为 WAV，应按 header 中的格式写 WAV：

```python
with wave.open(path, "wb") as wf:
    wf.setframerate(sample_rate)
    wf.setnchannels(channels)
    wf.setsampwidth(sample_width)
    wf.writeframes(pcm_bytes)
```

这会让保存结果直接可播放，也让后续 ASR loader 自动识别 48 kHz/stereo。

### 6. WER 前做通用音频归一化

Seed-TTS WER 可以继续使用 24 kHz mono 作为内部输入格式，但归一化函数应该是通用 helper，而不是 MOSS 专属函数。

建议接口：

```python
def normalize_pcm_s16le_for_wer(
    pcm_bytes: bytes,
    *,
    sample_rate: int,
    channels: int,
    target_sample_rate: int = 24000,
) -> bytes:
    ...
```

职责：

1. 按 `channels` reshape interleaved PCM。
2. 如果 `channels > 1`，downmix 为 mono。
3. 如果 `sample_rate != target_sample_rate`，resample。
4. 输出 `target_sample_rate` mono s16le PCM。

如果 benchmark 已经保存了 WAV，也可以让 WER 入口直接读取 WAV，再由 audio loader 做 downmix/resample。长期看，WER 入口接受 WAV path 或 WAV bytes 会比接受裸 PCM 更稳。

## 推荐最小改动

为了把 PR 做小，建议先只改以下几处：

### `serving_speech.py`

1. 确保 `response_format=pcm` 的 streaming response 设置：

```text
X-Sample-Rate
X-Channels
X-Bit-Depth
```

2. MOSS Local 输出 `[2, T]` 时，不要在编码前错误 flatten 成 mono。
3. header 中的 `X-Channels` 必须和实际 PCM body 一致。

### `vllm_omni/benchmarks/patch/patch.py`

1. `_audio_pcm_format()` 优先读取 response header。
2. capture PCM 时记录真实 `sample_rate/channels/sample_width`。
3. 保存 WAV 和计算 duration 时使用真实格式。
4. 将临时的 `_pcm_s16le_to_seed_tts_wer_bytes` 改名为通用 helper，例如 `normalize_pcm_s16le_for_wer`。

### `vllm_omni/benchmarks/data_modules/seed_tts_eval.py`

1. WER 入口不要假设所有 `tts_output_pcm_bytes` 都是 24 kHz mono。
2. 如果仍传 raw PCM，需要同时传入格式 metadata。
3. 如果传 WAV path/WAV bytes，则让 audio loader 读取 header 后归一化。

## 验证

### 单元测试

增加覆盖：

1. 24 kHz mono PCM 保存成 WAV 后 header 正确。
2. 48 kHz stereo PCM 保存成 WAV 后 header 正确。
3. 48 kHz stereo PCM 归一化到 24 kHz mono 后，样本数约为原 stereo frame 数的一半。
4. `duration_s` 对 stereo PCM 使用 `sample_rate * channels * sample_width` 计算。

### 手工验证

用 MOSS Local streaming 请求：

```bash
curl -N http://127.0.0.1:8123/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -o /root/vllm-omni-workspace/moss_local_stream.pcm \
  -D /tmp/moss_headers.txt \
  -d '{
    "model": "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
    "input": "Hello, I am canlin guo. How are you?",
    "ref_audio": "file:///root/vllm-omni-workspace/vllm-omni/local_v15_test.wav",
    "stream": true,
    "response_format": "pcm"
  }'
```

检查 header：

```text
Content-Type: audio/pcm
X-Sample-Rate: 48000
X-Channels: 2
X-Bit-Depth: 16
```

然后用 header 中的格式封装 WAV，确认可正常播放。

### Benchmark 验证

运行 Seed-TTS 小样本：

```bash
SEED_TTS_EVAL_DEVICE=cuda:6 \
SEED_TTS_WER_SAVE_AUDIO_DIR=/root/vllm-omni-workspace/vllm-omni/results/moss_local_debug/wavs \
PYTHONPATH=/root/vllm-omni-workspace/vllm-omni \
python benchmarks/tts/bench_tts.py \
  --host 127.0.0.1 \
  --port 8123 \
  --model OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 \
  --task voice_clone \
  --locale en \
  --dataset-path /root/vllm-omni-workspace/seedtts_testset \
  --num-prompts 20 \
  --concurrency 1 \
  --wer-eval \
  --output-len 256 \
  --output-dir /root/vllm-omni-workspace/vllm-omni/results/moss_local_debug
```

预期：

1. 保存的 WAV 能直接听到正常人声。
2. WAV header 显示 48 kHz stereo 或归一化后的 24 kHz mono，取决于保存点设计，但不能再是错误解释的假 WAV。
3. WER 不再因为音频格式读错而大幅异常。

## 风险

1. 某些客户端可能已经默认把 `audio/pcm` 当 24 kHz mono。显式 header 不会破坏这些客户端，但如果它们忽略 header，仍然会读错 MOSS stereo PCM。
2. 如果服务端选择保留 stereo，streaming payload 带宽会比 mono 高一倍。
3. 如果服务端选择统一 downmix mono，会牺牲 MOSS Local 的 stereo 输出能力。
4. benchmark 结果在修复后可能和历史结果不可直接比较，因为历史 WER 可能是在错误音频格式上计算的。

## 结论

MOSS Local 暴露的问题不是 Seed-TTS WER 的特殊问题，而是 raw PCM 缺少格式 contract。正确方向是让 serving response 和 benchmark capture 都显式处理 `sample_rate/channels/sample_width`。

短期可以在 benchmark 中做通用 PCM normalization，修复 MOSS Local 的评测音频。长期应该让 speech serving 的 AudioChunk contract 和 PCM response contract 一起收敛：模型输出负责声明 waveform 的真实格式，serving 负责按格式编码，benchmark/ASR 负责按格式读取。
