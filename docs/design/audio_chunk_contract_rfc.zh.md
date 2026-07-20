# RFC: Speech Streaming AudioChunk Contract

## 背景

`serving_speech.py` 现在承担了太多模型适配逻辑。它本来应该只把模型已经生成好的音频 chunk 编码成 PCM/WAV 并写给 HTTP client，但实际代码需要判断：

- `audio` 是 list 还是 tensor。
- list 是累计列表还是本次新增列表。
- tensor 是本次 delta chunk 还是累计 waveform snapshot。
- tensor shape 里的维度到底是 batch、channel 还是 sample。
- `sr` 是 int、tensor 还是 list。
- MOSS Local v1.5 的 stereo 输出是否应该保留 `[2, T]`。

这些判断不是 OpenAI speech API 层天然应该知道的模型细节。根因是 Stage1 code2wav/codec 到 speech serving 之间缺少一个明确的音频 chunk contract。

本文建议增加一个很小的 AudioChunk contract，让 speech streaming 边界只消费标准化后的 waveform delta。目标是删除 `_generate_audio_chunks()` 中的猜测逻辑，让代码更短、更直接、更容易审查。

## 当前数据流

### 1. Stage0 talker 输出 audio codes

TTS streaming 请求通常先由 Stage0 talker 逐步生成 codec code。

以 MOSS Local v1.5 为例，Stage0 每个 local/global step 生成新的 codec rows，输出大致是：

```python
OmniOutput(
    text_hidden_states=hidden,
    multimodal_outputs={
        "codes": {
            "audio": new_codes,  # torch.Tensor, shape [T_step, n_vq], normally [1, 12]
        },
    },
)
```

Qwen3-TTS 的 Stage0 也会通过 `codes.audio` 输出 codec frame，但通常是 `[T, Q]`，其中 `Q` 是 quantizer 数量。

这个阶段的 `audio` 不是 waveform，而是离散 codec code。

### 2. async chunk processor 把 Stage0 codes 变成 Stage1 input

Stage0 输出不能直接给 HTTP。它先经过 stage input processor，变成 Stage1 codec/code2wav 的输入。

MOSS Local v1.5 当前路径是 `talker2codec_raw_async_chunk()`：

```python
new_frames = multimodal_output["codes"]["audio"]  # [T_step, n_vq]
pending_frames.append(...)

chunk_codes = torch.stack(pending_frames[:emit_frames], dim=0)  # [T_chunk, n_vq]
codec_flat = chunk_codes.transpose(0, 1).reshape(-1).tolist()

return OmniPayloadStruct(
    codes=CodesStruct(audio=codec_flat, ref=ref_flat),
    meta=MetaStruct(
        req_id=[req_id],
        left_context_size=0,
        codec_streaming=True,
        codec_chunk_frames=T_chunk,
        codec_left_context_frames=0,
        code_flat_numel=len(codec_flat),
        ref_code_len=...,
        stream_finished=finished,
        finished=finished,
    ),
    request_id=req_id,
)
```

这里 Stage1 收到的本质是：

```text
input_ids = flat codebook-major codec ids
meta      = request id, chunk size, finished flag, streaming flag
```

也就是说，Stage1 的输入仍然不是音频，而是 codec token 序列。

Qwen3-TTS 当前路径是 `talker2code2wav_async_chunk()`：

```python
window_frames = last_left_context_plus_new_frames
code_predictor_codes = torch.tensor(
    [window_frames[f][q] for q in range(num_quantizers) for f in range(num_frames)]
)

return OmniPayloadStruct(
    codes=CodesStruct(audio=code_predictor_codes),
    meta=MetaStruct(
        left_context_size=left_context_size,
        finished=finished,
        ref_context_size=...,
        ref_context_request_id=...,
        ref_context_included=...,
    ),
)
```

Qwen 的 Stage1 输入也同样是 codebook-major flat codec ids，只是它带了 left-context/ref-context，用于让 decoder 在 chunk 边界处更平滑。

### 3. Stage1 codec/code2wav 输出 waveform

Stage1 才是真正把 codec ids 变成 waveform 的位置。

MOSS codec 当前返回：

```python
OmniOutput(
    text_hidden_states=None,
    multimodal_outputs={
        "model_outputs": audios,  # list[torch.Tensor]
        "sr": srs,                # list[torch.Tensor]
    },
)
```

其中 `audios[i]` 可能是：

```text
torch.Tensor shape [T]       # mono
torch.Tensor shape [C, T]    # stereo/channel-first, MOSS Local v1.5 可能是 [2, T]
torch.Tensor shape [0]       # empty mono
torch.Tensor shape [C, 0]    # empty channel-first
```

MOSS codec 内部 decoder 原始输出通常是：

```text
out.audio shape [B, C, T]
```

当前代码会取单个 request：

```python
wav = out.audio[0]  # [C, T]
audios[i] = wav.reshape(-1) if wav.ndim == 1 or wav.shape[0] == 1 else wav
```

所以 mono 会被压成 `[T]`，stereo 会保留 `[2, T]`。

Qwen code2wav 当前返回：

```python
OmniOutput(
    text_hidden_states=None,
    multimodal_outputs={
        "model_outputs": audios,  # list[torch.Tensor], each [T]
        "sr": srs,                # list[torch.Tensor]
    },
)
```

Qwen decoder 原始输出大致是 `[B, 1, T]` 或 `[B, T]`，当前代码最终统一成 mono `[T]`。

### 4. output processor 把 `model_outputs` remap 成 `audio`

`OmniRequestState.add_multimodal_tensor()` 会把模型输出里的：

```python
{"model_outputs": audios, "sr": srs}
```

remap 成语义 modality key：

```python
{"audio": audios, "sr": srs}
```

然后通过 `MultimodalPayload.from_dict()` 拆分：

```text
torch.Tensor value -> payload.tensors
non-tensor/list    -> payload.metadata
```

这里有一个关键问题：Stage1 输出的 `audios` 是 `list[Tensor]`，不是单个 tensor，所以它会进入 metadata，而不是 tensors。后续输出可能表现为：

```python
mm["audio"] == list[torch.Tensor]  # list path
```

但如果某个 producer 或 processor 输出单个 tensor，则 API 层看到的是：

```python
mm["audio"] == torch.Tensor        # tensor path
```

两种路径的语义现在没有被 contract 明确描述。

### 5. serving_speech 当前看到的数据

`_generate_audio_chunks()` 当前通过 `_extract_audio_output()` 拿到：

```python
audio_output, audio_key = self._extract_audio_output(res)
audio_val = audio_output[audio_key]
sr_raw = audio_output.get("sr")
```

它实际可能看到以下几类数据：

#### MOSS Local v1.5 streaming

常见形态：

```python
audio_val: list[torch.Tensor]
audio_val[-1].shape == [2, T_chunk]  # stereo channel-first
sr_raw: list[torch.Tensor]
sr_raw[-1] == tensor(48000, dtype=torch.int32)
```

每个 list 元素通常对应一次 Stage1 codec 输出。语义上它应该是本次新增 audio delta，但这个语义没有字段表达，只能靠代码约定。

#### Qwen3-TTS streaming

常见形态：

```python
audio_val: list[torch.Tensor]
audio_val[-1].shape == [T_chunk]     # mono
sr_raw: list[torch.Tensor]
sr_raw[-1] == tensor(24000, dtype=torch.int32)
```

Qwen 的 chunk 可能包含 left-context decode 后再 trim 出来的新音频；serving 层不应该关心这个细节，只应该知道它收到的是 playable delta waveform。

#### Tensor path

某些输出路径下，API 层可能看到：

```python
audio_val: torch.Tensor
```

这个 tensor 可能是：

```text
[T]       # mono delta
[C, T]    # channel-first delta
[1, T]    # batch/channel singleton，需要规范化
[1, C, T] # batch + channel + samples
[T_total] or [C, T_total] # cumulative snapshot
```

当前没有字段说明它是 delta 还是 cumulative snapshot。于是 `_generate_audio_chunks()` 只能用 prefix compare 猜：

```python
if current[..., :prev_len] == previous:
    emit current[..., prev_len:]
else:
    emit current
```

这就是复杂性的主要来源。

## 当前问题

### 问题一：`audio` 的类型携带了隐式语义

当前代码把 list 当作一种 streaming 语义，把 tensor 当作另一种 streaming 语义：

```python
if isinstance(audio_val, list):
    # treat as cumulative list, emit audio_val[prev_count:]
else:
    # treat as tensor, then guess delta/cumulative
```

这很脆弱。`list[Tensor]` 只是 Python 容器类型，不应该承载“累计列表”或“delta 列表”的协议语义。

### 问题二：缺少 delta/cumulative 标记

HTTP streaming 最终应该发送 delta audio bytes。  
但 engine 层可能传：

- 本次新增 audio delta。
- 截至当前的完整 cumulative waveform。
- 累计 list，其中 list 的每个元素是历史 delta。

这些都能表示音频，但对 streaming consumer 来说语义完全不同。没有显式字段时，consumer 必须维护状态并猜测。

### 问题三：shape 没有标准化

现在 API 层会做多次 squeeze：

```python
if chunk_np.ndim == 3 and chunk_np.shape[0] == 1:
    chunk_np = chunk_np[0]
if chunk_np.ndim > 2:
    chunk_np = chunk_np.squeeze()
if chunk_np.ndim == 2 and 1 in chunk_np.shape:
    chunk_np = chunk_np.squeeze()
```

这对 mono 通常没问题，但对 stereo 很危险。  
MOSS Local v1.5 的合法输出是 `[2, T]`，不能被误判为 batch 维或被 flatten 成 `[2*T]`。

shape 规则应该在 producer 边界固定下来，而不是由 HTTP 层猜。

### 问题四：sample rate 不是标准字段

`sr` 现在可能是：

```text
int
torch.Tensor scalar
list[int]
list[torch.Tensor]
```

API 层因此需要：

```python
sr_val = sr_raw[-1] if isinstance(sr_raw, list) else sr_raw
sample_rate_val = sr_val.item() if hasattr(sr_val, "item") else int(sr_val)
```

这不是大问题，但它加剧了 streaming 函数的噪音。sample rate 应该在标准化函数里一次性解析。

### 问题五：模型输出和 API 输出之间没有“边界对象”

Stage1 现在直接把内部产物塞进 `multimodal_outputs`：

```python
{"model_outputs": audios, "sr": srs}
```

`serving_speech.py` 又直接解析这个内部结构。  
这让 OpenAI API 层知道了太多 engine/model 细节，包括 `model_outputs` remap、list metadata、tensor accumulation strategy 等。

## 设计目标

目标是定义一个简单、明确、可测试的 AudioChunk contract：

- Stage1 对 speech streaming 输出 playable waveform。
- serving speech 只接受 delta AudioChunk。
- audio sample axis 固定为最后一维。
- channel 语义显式，支持 mono 和 channel-first stereo。
- sample rate 必须显式提供。
- `_generate_audio_chunks()` 不再猜 delta/cumulative，不再做 prefix compare。

非目标：

- 不重构 Stage0 codec code 格式。
- 不统一所有 multimodal modality。
- 不改变 MOSS/Qwen 的 codec 算法。
- 不改变 stop/head/scheduler 逻辑。
- 不引入复杂的 media abstraction。

## AudioChunk Contract

新增一个最小内部 schema。可以先用 dataclass，也可以先用 dict；推荐 dataclass，因为它让 API 层代码更清楚。

```python
@dataclass(frozen=True)
class AudioChunk:
    samples: torch.Tensor | np.ndarray
    sample_rate: int
    is_delta: bool
    layout: Literal["mono", "channels_first"]
```

字段含义：

```text
samples:
    playable waveform, float32, range approximately [-1, 1]

sample_rate:
    output sample rate, required

is_delta:
    True means samples are newly generated audio for this streaming update.
    False is only allowed for final non-streaming output.

layout:
    "mono" means samples shape [T]
    "channels_first" means samples shape [C, T]
```

Streaming 边界只允许：

```text
is_delta == True
samples.ndim == 1 or samples.ndim == 2
sample axis == last dimension
```

禁止 streaming producer 输出 cumulative waveform snapshot。  
如果某个模型内部必须维护 cumulative waveform，应在模型或 output normalizer 内部裁成 delta 后再交给 `serving_speech.py`。

## Payload 表达

为了不大改 `MultimodalPayload`，Stage1 可以继续返回 dict，但需要显式附带 metadata：

```python
OmniOutput(
    text_hidden_states=None,
    multimodal_outputs={
        "model_outputs": audios,
        "sr": srs,
        "audio_meta": audio_metas,
    },
)
```

其中每个 request 对应：

```python
audio_meta = {
    "kind": "audio_chunk",
    "is_delta": True,
    "layout": "channels_first",  # or "mono"
}
```

MOSS Local v1.5 streaming 输出应该是：

```python
{
    "model_outputs": [wav],  # wav shape [2, T] for stereo, or [T] for mono
    "sr": [torch.tensor(48000, dtype=torch.int32)],
    "audio_meta": [{
        "kind": "audio_chunk",
        "is_delta": True,
        "layout": "channels_first",
    }],
}
```

Qwen3-TTS streaming 输出应该是：

```python
{
    "model_outputs": [wav],  # wav shape [T]
    "sr": [torch.tensor(24000, dtype=torch.int32)],
    "audio_meta": [{
        "kind": "audio_chunk",
        "is_delta": True,
        "layout": "mono",
    }],
}
```

非 streaming final output 可以使用：

```python
{
    "model_outputs": [full_wav],
    "sr": [sr],
    "audio_meta": [{
        "kind": "audio_chunk",
        "is_delta": False,
        "layout": "mono",
    }],
}
```

## Normalizer

新增一个集中式 helper，例如：

```python
def extract_audio_chunks(res, *, streaming: bool) -> list[AudioChunk]:
    ...
```

它做三件事：

1. 从 `res.multimodal_output` / `res.request_output.outputs[*].multimodal_output` 找到 audio payload。
2. 解析 `audio`、`sr`、`audio_meta`。
3. 返回标准 `AudioChunk` 列表。

这个 helper 是唯一允许理解 legacy payload 形态的地方。  
`_generate_audio_chunks()` 不再直接接触 `model_outputs`、list/tensor 差异、`MultimodalPayload` 内部细节。

推荐规则：

```python
if audio_meta.kind == "audio_chunk":
    validate shape/layout/is_delta/sample_rate
    return AudioChunk(...)

else:
    raise ValueError("Speech streaming output missing audio_meta")
```

为了保持实现干净，这次改造不建议继续长期支持无 `audio_meta` 的 silent fallback。测试里可以保留 legacy fixtures，确保错误信息清晰，但生产 streaming path 应该要求 Stage1 明确输出 contract。

## serving_speech 简化后形态

改造后 `_generate_audio_chunks()` 主体应该接近：

```python
async for res in generator:
    for chunk in extract_audio_chunks(res, streaming=True):
        if not chunk.is_delta:
            raise ValueError("Streaming speech requires delta AudioChunk")

        samples = normalize_audio_samples(chunk.samples, chunk.layout)

        if response_format == "wav" and first_chunk:
            yield create_wav_header(
                sample_rate=chunk.sample_rate,
                num_channels=infer_num_channels(samples),
            )
            first_chunk = False

        yield encode_pcm16(samples, sample_rate=chunk.sample_rate)
```

可以删除：

- `prev_count`
- `prev_tensor_snapshot`
- `tensor_stream_mode`
- prefix `np.allclose`
- `cumulative_snapshot` 日志
- `_maybe_restore_moss_local_stereo()`
- API 层对 MOSS Local 的特殊 shape 猜测

保留：

- WAV header first chunk 逻辑。
- PCM 编码逻辑。
- request cancelled / engine dead / e2e timing 日志。
- ref audio artifact cleanup。

## Producer 侧要求

### MOSS codec

MOSS codec 在返回 `OmniOutput` 前就知道当前输出 shape：

```python
wav = out.audio[0]  # [C, T]
audios[i] = wav.reshape(-1) if mono else wav
```

这里应该同步生成 `audio_meta`：

```python
if audios[i].ndim == 1:
    layout = "mono"
elif audios[i].ndim == 2:
    layout = "channels_first"
else:
    raise ValueError(...)

audio_metas[i] = {
    "kind": "audio_chunk",
    "is_delta": bool(streaming_enabled),
    "layout": layout,
}
```

对于 streaming path，`_decode_streaming_batch()` 输出的就是本次 codec step/sequence/offline flush 的 playable waveform delta，因此 `is_delta=True`。

对于 non-streaming path，一次性输出完整 waveform，因此 `is_delta=False`。

### Qwen code2wav

Qwen code2wav 当前最终统一输出 mono `[T]`：

```python
audios[idx] = wav.reshape(-1)
```

它应该输出：

```python
audio_metas[idx] = {
    "kind": "audio_chunk",
    "is_delta": True,
    "layout": "mono",
}
```

Qwen 内部的 left-context/ref-context decode 和 trim 仍然保留在 code2wav。serving 层只看到已经 trim 后的 delta waveform。

## 一次性改造清单

这件事粒度不大，建议一次完成，不拆分实施阶段：

1. 新增 `AudioChunk` dataclass 和 `extract_audio_chunks()` helper。
2. MOSS codec 输出 `audio_meta`。
3. Qwen code2wav 输出 `audio_meta`。
4. `_generate_audio_chunks()` 改成只消费 `AudioChunk`。
5. 删除 `_generate_audio_chunks()` 中的 list/tensor/cumulative guessing。
6. 删除 MOSS stereo restore hook。
7. 增加单元测试覆盖 MOSS stereo、Qwen mono、缺失 `audio_meta` 报错。

## 验收标准

### MOSS Local v1.5 stereo 不被破坏

输入：

```python
samples.shape == [2, 3840]
layout == "channels_first"
sample_rate == 48000
is_delta == True
```

期望：

- WAV header channel count 是 2。
- PCM bytes 按 stereo 编码。
- 不 squeeze 成 `[7680]`。
- 不走任何 MOSS-specific special case。

### Qwen mono 正常 streaming

输入：

```python
samples.shape == [T]
layout == "mono"
sample_rate == 24000
is_delta == True
```

期望：

- WAV header channel count 是 1。
- 每个 chunk 直接编码。
- API 层不关心 left-context/ref-context。

### cumulative snapshot 被拒绝

如果 streaming path 收到：

```python
is_delta == False
```

期望直接报错：

```text
Streaming speech requires delta AudioChunk
```

不要再尝试 prefix compare。cumulative-to-delta 应该由 producer 或 normalizer 在明确知道语义的地方完成。

### 缺少 audio_meta 被拒绝

如果 Stage1 输出：

```python
{"model_outputs": audios, "sr": srs}
```

但没有：

```python
{"audio_meta": ...}
```

期望 streaming path 报错：

```text
Speech streaming output missing audio_meta
```

这样能尽早暴露新模型没有遵守 contract，而不是让 API 层继续猜。

## 为什么不让所有模型内部都只发 delta

HTTP streaming 边界必须只发 delta。  
但模型内部不需要被这个要求污染。

例如：

- Qwen code2wav 需要 left-context/ref-context 做平滑 decode。
- MOSS codec streaming session 需要维护 codec state。
- 某些模型可能内部只方便生成 cumulative codes。

这些都可以保留。唯一要求是：到 Stage1 输出给 speech serving 时，必须已经变成标准 delta AudioChunk。

这条边界足够窄，也足够有约束：

```text
model-specific code / context / state
    -> Stage1 codec/code2wav
    -> AudioChunk(delta waveform)
    -> serving_speech encode/write bytes
```

## 预期收益

- `_generate_audio_chunks()` 只处理 HTTP streaming，不再处理模型协议。
- MOSS stereo 和 Qwen mono 都通过同一个 shape/layout contract 表达。
- 新模型接入时必须声明 audio chunk 语义，减少隐式行为。
- bug 定位更直接：如果音频重复/丢失，先看 producer 是否真的输出 delta；API 层不再背锅。
- 后续支持更多声道或采样率时，只扩展 AudioChunk validation，不需要在 serving 主循环里堆模型分支。
