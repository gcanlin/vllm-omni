# RFC: Runner-Owned TTS Vocoder CUDA Graph 最终态设计

## 一句话结论

TTS vocoder/code2wav CUDA graph 应该和 vLLM 原生模型 CUDA graph 一样，由 model runner 统一管理：

```text
runner owns:
  cudagraph_mode / enforce_eager
  cudagraph_capture_sizes
  cudagraph_num_of_warmups
  capture lifecycle
  graph pool
  memory accounting
  graph stats/logging

model owns:
  给定 shape 如何构造 static buffers
  给定 static buffers 如何执行 decode
  replay 前如何 copy runtime inputs
  streaming state 如何 reset
  graph miss 时如何 eager fallback
```

模型不读取 graph policy 配置。模型只暴露“我有哪些 vocoder graph routine、每个 routine 怎么 capture/replay”。

## 设计目标

1. **严格 follow vLLM CUDA graph 风格**

   graph capture 入口在 runner 的 `capture_model()` 生命周期里，而不是模型构造、`load_weights()`、首次请求、stream session lazy init。

2. **统一 graph policy**

   所有 TTS vocoder/code2wav graph 都只使用：

   ```python
   vllm_config.compilation_config.cudagraph_mode
   vllm_config.compilation_config.cudagraph_capture_sizes
   vllm_config.compilation_config.cudagraph_num_of_warmups
   ```

   `enforce_eager` 通过 vLLM config 解析后的 `cudagraph_mode == CUDAGraphMode.NONE` 生效。

3. **普通模型零特殊逻辑**

   新增架构变化不大的 code2wav/vocoder 模型，只要符合标准 tensor decode contract，就可以用默认 routine 接入 CUDA graph。

4. **特殊模型只开放最小自定义点**

   少数模型，例如 MOSS streaming codec，有 stateful KV/cache、exec mask、stream slot、reset 等特殊语义。它们只实现 routine hook，不接管 runner flow。

5. **connector extra 不再承载 graph policy**

   `connector.extra` 只描述 stage 间数据 contract，例如 chunk size、left context、streaming 开关。graph capture sizes、warmup 次数、eager/graph 开关都属于 runner/compilation config。

## 最终配置形态

stage1 vocoder/code2wav 只用 vLLM 风格配置：

```yaml
stages:
  - stage_id: 1
    enforce_eager: false
    compilation_config:
      cudagraph_mode: FULL_DECODE_ONLY
      cudagraph_capture_sizes: [1, 2, 4, 8, 16]
      cudagraph_num_of_warmups: 1
```

对 MOSS local 这种 streaming codec，如果希望 exact hit 所有 chunk tail：

```yaml
stages:
  - stage_id: 1
    enforce_eager: false
    compilation_config:
      cudagraph_capture_sizes: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
      cudagraph_num_of_warmups: 1
```

connector 只保留 inter-stage contract：

```yaml
connectors:
  shm:
    extra:
      codec_streaming: true
      initial_codec_chunk_frames: 1
      codec_chunk_frames: 15
      codec_left_context_frames: 0
```

不再出现：

```yaml
decode_cudagraph_capture_sizes
streaming_decode_cudagraph_capture_sizes
streaming_decode_cudagraph_min_free_gb
codec_max_step_frames
```

## 新增模块

建议新增两个模块。

```text
vllm_omni/model_executor/models/interfaces/vocoder_cudagraph.py
vllm_omni/worker/vocoder_cudagraph_manager.py
```

第一个模块定义模型侧需要实现的 protocol 和数据结构。

第二个模块实现 runner-owned graph manager，风格上对齐 vLLM 的 `CudagraphDispatcher` / `ModelCudaGraphManager`。

## 核心数据结构

### `VocoderCUDAGraphDescriptor`

描述一个可 capture 的 vocoder graph case。

```python
@dataclass(frozen=True)
class VocoderCUDAGraphDescriptor:
    routine_name: str
    size: int
    batch_size: int | None = None
    tag: str | None = None
```

语义：

- `routine_name`: graph routine 名字，例如 `"qwen3_code2wav"`、`"moss_streaming_decode"`。
- `size`: capture size。runner 不解释单位，模型 routine 自己解释。对 MOSS 是 streaming step frames `T`；对 Qwen3-TTS 可以是 decode frames/window length。
- `batch_size`: 可选。多数 TTS stage1 当前可以固定为 runner batch 或 stream slots。需要 batch-specialized graph 时再填。
- `tag`: 可选，用于区分同一 size 下的模型内部形态，例如 `"with_ref"` / `"no_ref"`。

为什么不一开始设计复杂 semantic enum：

```python
semantic: Literal["codec_frames", "mel_frames", "tokens", ...]
```

因为 runner 不应该根据 semantic 改变调度策略。最终接口保持 descriptor opaque，减少抽象面。

### `VocoderCUDAGraphBuffers`

模型 routine 自己定义 static buffers。runner 不理解 buffer 内容。

```python
class VocoderCUDAGraphBuffers(Protocol):
    pass
```

实际实现可以是 dataclass，例如 MOSS：

```python
@dataclass
class MossStreamingDecodeBuffers:
    codes: torch.Tensor        # [n_vq, stream_slots, T]
    lengths: torch.Tensor      # [stream_slots]
    exec_mask: torch.Tensor    # [stream_slots]
    audio: torch.Tensor | None = None
    audio_lengths: torch.Tensor | None = None
```

Qwen3：

```python
@dataclass
class Qwen3Code2WavBuffers:
    codes: torch.Tensor
    code_lengths: torch.Tensor
    audio: torch.Tensor | None = None
    audio_lengths: torch.Tensor | None = None
```

### `VocoderCUDAGraphReplayResult`

标准 replay 返回。

```python
@dataclass
class VocoderCUDAGraphReplayResult:
    output: Any
    hit: bool
    descriptor: VocoderCUDAGraphDescriptor | None = None
```

runner/manager 内部使用它记录 hit/miss。模型 forward 最终只关心 `output`。

### `VocoderCUDAGraphCaptureStats`

runner 统一记录 capture 结果。

```python
@dataclass
class VocoderCUDAGraphCaptureStats:
    requested: list[VocoderCUDAGraphDescriptor]
    captured: list[VocoderCUDAGraphDescriptor]
    failed: list[VocoderCUDAGraphDescriptor]
    elapsed_s: float
    memory_bytes: int
```

## 模型侧接口

### `SupportsVocoderCUDAGraph`

stage1 模型如果支持 vocoder graph，实现这个 protocol。

```python
class SupportsVocoderCUDAGraph(Protocol):
    def get_vocoder_cudagraph_routines(
        self,
    ) -> list["VocoderCUDAGraphRoutine"]:
        ...
```

模型不接收 `vllm_config` graph policy，不读取 `enforce_eager`，不读取 capture sizes。

### `VocoderCUDAGraphRoutine`

每个 routine 表示一个可 graph 的 vocoder/code2wav 子路径。

```python
class VocoderCUDAGraphRoutine(Protocol):
    name: str

    def default_capture_sizes(self) -> list[int]:
        ...

    def make_descriptors(
        self,
        capture_sizes: list[int],
    ) -> list[VocoderCUDAGraphDescriptor]:
        ...

    def allocate_static_buffers(
        self,
        desc: VocoderCUDAGraphDescriptor,
        device: torch.device,
    ) -> VocoderCUDAGraphBuffers:
        ...

    def prepare_for_capture(
        self,
        desc: VocoderCUDAGraphDescriptor,
        buffers: VocoderCUDAGraphBuffers,
    ) -> None:
        ...

    def forward_for_capture(
        self,
        desc: VocoderCUDAGraphDescriptor,
        buffers: VocoderCUDAGraphBuffers,
    ) -> Any:
        ...

    def finalize_capture(
        self,
        desc: VocoderCUDAGraphDescriptor,
        buffers: VocoderCUDAGraphBuffers,
        output: Any,
    ) -> None:
        ...

    def runtime_descriptor(
        self,
        runtime_inputs: Any,
    ) -> VocoderCUDAGraphDescriptor | None:
        ...

    def copy_inputs_to_static_buffers(
        self,
        runtime_inputs: Any,
        buffers: VocoderCUDAGraphBuffers,
    ) -> None:
        ...

    def output_from_static_buffers(
        self,
        desc: VocoderCUDAGraphDescriptor,
        buffers: VocoderCUDAGraphBuffers,
    ) -> Any:
        ...

    def eager_forward(
        self,
        runtime_inputs: Any,
    ) -> Any:
        ...
```

接口语义：

- `default_capture_sizes()`：没有显式 `compilation_config.cudagraph_capture_sizes` 时，模型给一个合理默认。MOSS 可以返回 `1..codec_chunk_frames`。Qwen3 可以返回常用 decode window。
- `make_descriptors()`：把 capture sizes 转成 routine-specific descriptors。
- `allocate_static_buffers()`：分配 captured graph 使用的固定地址 input/output buffers。
- `prepare_for_capture()`：capture 前 reset streaming state、设置 dummy mask 等。
- `forward_for_capture()`：runner 在 `torch.cuda.graph(...)` 里调用的实际 forward。
- `finalize_capture()`：保存 output tensor 引用，capture 后 reset state。
- `runtime_descriptor()`：根据 runtime input 选择 graph key。不能 hit 时返回 `None`。
- `copy_inputs_to_static_buffers()`：replay 前把真实输入 copy 到 static buffers。
- `output_from_static_buffers()`：replay 后从 static output buffer 取结果。
- `eager_forward()`：graph miss fallback。

这个 routine 是最小自定义面。模型只实现 shape/state 细节，runner 管 flow。

## Runner 侧新增类

### `VocoderCUDAGraphManager`

文件：

```text
vllm_omni/worker/vocoder_cudagraph_manager.py
```

职责：

```text
管理 routines
根据 compilation_config resolve descriptors
按大 shape 优先 capture
持有 CUDAGraph 和 static buffers
统一 replay dispatch
统一 memory/stats/logging
统一 clear
```

核心字段：

```python
class VocoderCUDAGraphManager:
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        device: torch.device,
        routines: list[VocoderCUDAGraphRoutine],
    ) -> None:
        self.vllm_config = vllm_config
        self.device = device
        self.routines = {routine.name: routine for routine in routines}
        self.graphs: dict[VocoderCUDAGraphDescriptor, torch.cuda.CUDAGraph] = {}
        self.buffers: dict[VocoderCUDAGraphDescriptor, VocoderCUDAGraphBuffers] = {}
        self.pool = current_platform.get_global_graph_pool()
```

### `needs_capture()`

```python
def needs_capture(self) -> bool:
    return (
        self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        and bool(self.routines)
    )
```

只看 vLLM graph mode，不直接读 `enforce_eager`。

### `resolve_descriptors()`

```python
def resolve_descriptors(self) -> list[VocoderCUDAGraphDescriptor]:
    config_sizes = self.vllm_config.compilation_config.cudagraph_capture_sizes
    descriptors = []
    for routine in self.routines.values():
        sizes = config_sizes or routine.default_capture_sizes()
        descriptors.extend(routine.make_descriptors(sizes))
    return sorted(descriptors, key=lambda d: d.size, reverse=True)
```

大 shape 优先，和 vLLM CUDA graph capture 策略一致。

### `capture()`

```python
@torch.inference_mode()
def capture(self) -> VocoderCUDAGraphCaptureStats:
    if not self.needs_capture():
        return empty_stats()

    descriptors = self.resolve_descriptors()
    start_time = time.perf_counter()

    torch.accelerator.synchronize()
    torch.accelerator.empty_cache()
    start_free = torch.cuda.mem_get_info()[0]

    set_cudagraph_capturing_enabled(True)
    with graph_capture(device=self.device):
        for desc in descriptors:
            routine = self.routines[desc.routine_name]
            buffers = routine.allocate_static_buffers(desc, self.device)

            for _ in range(self.vllm_config.compilation_config.cudagraph_num_of_warmups):
                routine.prepare_for_capture(desc, buffers)
                routine.forward_for_capture(desc, buffers)
                torch.accelerator.synchronize()

            routine.prepare_for_capture(desc, buffers)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool):
                output = routine.forward_for_capture(desc, buffers)

            routine.finalize_capture(desc, buffers, output)
            self.graphs[desc] = graph
            self.buffers[desc] = buffers

    set_cudagraph_capturing_enabled(False)
    torch.accelerator.synchronize()
    end_free = torch.cuda.mem_get_info()[0]

    return VocoderCUDAGraphCaptureStats(
        requested=descriptors,
        captured=list(self.graphs),
        failed=[d for d in descriptors if d not in self.graphs],
        elapsed_s=time.perf_counter() - start_time,
        memory_bytes=start_free - end_free,
    )
```

实际实现要用 `try/finally` 保证 `set_cudagraph_capturing_enabled(False)` 一定执行。

### `replay_or_none()`

```python
def replay_or_none(
    self,
    routine_name: str,
    runtime_inputs: Any,
) -> Any | None:
    routine = self.routines[routine_name]
    desc = routine.runtime_descriptor(runtime_inputs)
    if desc is None or desc not in self.graphs:
        return None

    buffers = self.buffers[desc]
    routine.copy_inputs_to_static_buffers(runtime_inputs, buffers)
    self.graphs[desc].replay()
    return routine.output_from_static_buffers(desc, buffers)
```

模型 forward 调用这个方法。miss 返回 `None`，模型调用 `routine.eager_forward(...)`。

### `clear()`

```python
def clear(self) -> None:
    self.graphs.clear()
    self.buffers.clear()
```

runner shutdown / teardown 时调用。

## Runner 改动

### `GPUGenerationModelRunner.__init__`

新增字段：

```python
self.vocoder_cudagraph_manager: VocoderCUDAGraphManager | None = None
```

### `GPUGenerationModelRunner.load_model()`

模型加载完成后发现 capability：

```python
def _maybe_init_vocoder_cudagraph_manager(self) -> None:
    model = self.model
    get_routines = getattr(model, "get_vocoder_cudagraph_routines", None)
    if not callable(get_routines):
        return

    routines = get_routines()
    if not routines:
        return

    self.vocoder_cudagraph_manager = VocoderCUDAGraphManager(
        vllm_config=self.vllm_config,
        device=self.device,
        routines=routines,
    )

    bind = getattr(model, "bind_vocoder_cudagraph_manager", None)
    if callable(bind):
        bind(self.vocoder_cudagraph_manager)
```

`bind_vocoder_cudagraph_manager()` 是可选的。普通模型可以让 routine 自己持有 manager；复杂模型可以在 model forward 里直接访问 manager。

### `GPUGenerationModelRunner.capture_model()`

最终态应类似 vLLM 原生 capture：

```python
@torch.inference_mode()
def capture_model(self) -> int:
    total_graph_memory = super().capture_model()

    if self.vocoder_cudagraph_manager is None:
        return total_graph_memory

    if not self.vocoder_cudagraph_manager.needs_capture():
        logger.warning(
            "Skipping vocoder CUDA graph capture. To enable it, ensure cudagraph_mode is not NONE."
        )
        return total_graph_memory

    stats = self.vocoder_cudagraph_manager.capture()
    logger.info(
        "Vocoder CUDA graph capture finished in %.0f secs, took %.2f GiB, captured=%s failed=%s",
        stats.elapsed_s,
        stats.memory_bytes / (1 << 30),
        stats.captured,
        stats.failed,
    )
    return total_graph_memory + stats.memory_bytes
```

如果 generation runner 本身没有主模型 graph，也仍然应该实现同样的 `capture_model()` 入口，而不是在 model 内 lazy capture。

### `GPUGenerationModelRunner.shutdown / clear`

如果 runner 有 teardown hook：

```python
def clear_cudagraphs(self) -> None:
    super().clear_cudagraphs()
    if self.vocoder_cudagraph_manager is not None:
        self.vocoder_cudagraph_manager.clear()
```

## 模型 forward 的标准模式

模型 forward 不判断 policy，只走：

```python
def forward(...):
    runtime_inputs = build_runtime_inputs(...)

    if self._vocoder_cudagraph_manager is not None:
        output = self._vocoder_cudagraph_manager.replay_or_none(
            "routine_name",
            runtime_inputs,
        )
        if output is not None:
            return output

    return self._routine.eager_forward(runtime_inputs)
```

如果 `enforce_eager=true`，runner 不 capture，manager 没有 graph，`replay_or_none()` miss，自动 eager。

## 默认 Routine：`StandardCode2WavCUDAGraphRoutine`

为了让大多数模型不用写太多代码，提供一个默认 routine。

文件：

```text
vllm_omni/model_executor/models/common/vocoder_cudagraph_routines.py
```

适用条件：

```text
输入可以整理成固定 shape tensor
decode 是纯 tensor 函数
没有跨 step streaming state
graph output shape 由 descriptor 决定
miss 时可以直接调用同一个 eager decode
```

接口：

```python
class StandardCode2WavCUDAGraphRoutine:
    def __init__(
        self,
        *,
        name: str,
        decode_fn: Callable[[Any], Any],
        allocate_buffers_fn: Callable[[VocoderCUDAGraphDescriptor, torch.device], Any],
        copy_inputs_fn: Callable[[Any, Any], None],
        output_fn: Callable[[VocoderCUDAGraphDescriptor, Any], Any],
        runtime_descriptor_fn: Callable[[Any], VocoderCUDAGraphDescriptor | None],
        default_capture_sizes_fn: Callable[[], list[int]],
    ) -> None:
        ...
```

模型只传几个函数，不自己写 manager/wrapper。

## Qwen3-TTS 接入示例

### 当前语义

Qwen3-TTS stage1 是 code2wav。它接收 stage0 发来的 codec codes，按 decode window 解出 waveform。它是相对 stateless 的 window decoder，不像 MOSS streaming codec 那样有跨 step KV state。

常见 capture sizes 表示 decode frame/window 长度，例如：

```text
25, 73, 97, 169, 325
```

最终态里这些 sizes 不放 connector extra，而放 stage1 `compilation_config.cudagraph_capture_sizes`。

### 模型新增方法

文件：

```text
vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_code2wav.py
```

新增：

```python
class Qwen3TTSCode2Wav(nn.Module):
    def get_vocoder_cudagraph_routines(self) -> list[VocoderCUDAGraphRoutine]:
        return [
            Qwen3Code2WavCUDAGraphRoutine(
                decoder=self.decoder,
                num_codebooks=self.num_codebooks,
                default_sizes=self._default_decode_graph_sizes(),
            )
        ]

    def bind_vocoder_cudagraph_manager(
        self,
        manager: VocoderCUDAGraphManager,
    ) -> None:
        self._vocoder_cudagraph_manager = manager
```

### `Qwen3Code2WavCUDAGraphRoutine`

```python
class Qwen3Code2WavCUDAGraphRoutine:
    name = "qwen3_code2wav"

    def default_capture_sizes(self) -> list[int]:
        return [25, 73, 97, 169, 325]

    def make_descriptors(self, capture_sizes: list[int]) -> list[VocoderCUDAGraphDescriptor]:
        return [
            VocoderCUDAGraphDescriptor(self.name, size=int(size))
            for size in capture_sizes
            if int(size) > 0
        ]

    def allocate_static_buffers(self, desc, device):
        return Qwen3Code2WavBuffers(
            codes=torch.zeros(..., desc.size, device=device, dtype=torch.long),
            code_lengths=torch.full(..., desc.size, device=device, dtype=torch.long),
        )

    def forward_for_capture(self, desc, buffers):
        return self.decoder.decode(buffers.codes, buffers.code_lengths)

    def finalize_capture(self, desc, buffers, output):
        buffers.audio = output.audio
        buffers.audio_lengths = output.audio_lengths

    def runtime_descriptor(self, runtime_inputs):
        frames = runtime_inputs.num_frames
        return VocoderCUDAGraphDescriptor(self.name, size=frames)

    def copy_inputs_to_static_buffers(self, runtime_inputs, buffers):
        buffers.codes.copy_(runtime_inputs.codes)
        buffers.code_lengths.copy_(runtime_inputs.code_lengths)

    def output_from_static_buffers(self, desc, buffers):
        return AudioOutput(buffers.audio, buffers.audio_lengths)

    def eager_forward(self, runtime_inputs):
        return self.decoder.decode(runtime_inputs.codes, runtime_inputs.code_lengths)
```

Qwen3 不需要自定义 manager，不需要读 `vllm_config`，不需要知道 `enforce_eager`。

## MOSS-TTS Local v1.5 接入示例

### 当前语义

MOSS local stage1 是 codec decoder，streaming decode 的核心输入是：

```text
codes_step: [n_vq, stream_slots, T]
codes_lengths: [stream_slots]
exec_mask: [stream_slots]
```

其中：

- `n_vq = 12`
- `stream_slots` 是 stage1 codec streaming slots
- `T` 是本次 streaming decode step 的 frame 数

MOSS 的特殊点：

```text
有 streaming context
有内部 KV/ring cache/state
每个 slot 有独立 offset
exec_mask 控制当前 step 哪些 slot active
capture 前后必须 reset streaming state
bucket padding 不安全，因为内部 state offset 按 tensor T 前进
```

因此 MOSS 需要自定义 routine，但不需要自定义 runner。

### 模型新增方法

文件：

```text
vllm_omni/model_executor/models/moss_tts/modeling_moss_tts_codec.py
```

新增：

```python
class MossTTSCodecDecoder(nn.Module):
    def get_vocoder_cudagraph_routines(self) -> list[VocoderCUDAGraphRoutine]:
        return [
            MossStreamingDecodeCUDAGraphRoutine(
                codec=self._codec,
                n_vq=self._n_vq,
                stream_slots=self._resolve_stream_slots(),
                default_chunk_frames=self._stream_chunk_frames,
                reset_slots=self._reset_streaming_slots,
            )
        ]

    def bind_vocoder_cudagraph_manager(self, manager: VocoderCUDAGraphManager) -> None:
        self._vocoder_cudagraph_manager = manager
```

### `MossStreamingDecodeCUDAGraphRoutine`

```python
class MossStreamingDecodeCUDAGraphRoutine:
    name = "moss_streaming_decode"

    def default_capture_sizes(self) -> list[int]:
        return list(range(1, self.default_chunk_frames + 1))

    def make_descriptors(self, capture_sizes: list[int]) -> list[VocoderCUDAGraphDescriptor]:
        return [
            VocoderCUDAGraphDescriptor(
                routine_name=self.name,
                size=int(size),
                batch_size=self.stream_slots,
            )
            for size in capture_sizes
            if int(size) > 0
        ]

    def allocate_static_buffers(self, desc, device):
        return MossStreamingDecodeBuffers(
            codes=torch.zeros(
                self.n_vq,
                self.stream_slots,
                desc.size,
                dtype=torch.long,
                device=device,
            ),
            lengths=torch.full(
                (self.stream_slots,),
                desc.size,
                dtype=torch.long,
                device=device,
            ),
            exec_mask=torch.ones(
                self.stream_slots,
                dtype=torch.bool,
                device=device,
            ),
        )

    def prepare_for_capture(self, desc, buffers):
        self.reset_all_slots()
        self.codec._set_streaming_exec_mask(buffers.exec_mask)

    def forward_for_capture(self, desc, buffers):
        return self.codec._decode_frame(buffers.codes, buffers.lengths)

    def finalize_capture(self, desc, buffers, output):
        buffers.audio = output.audio
        buffers.audio_lengths = output.audio_lengths
        self.reset_all_slots()

    def runtime_descriptor(self, runtime_inputs):
        T = runtime_inputs.codes_step.shape[-1]
        return VocoderCUDAGraphDescriptor(
            routine_name=self.name,
            size=int(T),
            batch_size=self.stream_slots,
        )

    def copy_inputs_to_static_buffers(self, runtime_inputs, buffers):
        buffers.exec_mask.copy_(runtime_inputs.exec_mask)
        self.codec._set_streaming_exec_mask(buffers.exec_mask)
        buffers.codes.copy_(runtime_inputs.codes_step)
        buffers.lengths.copy_(runtime_inputs.codes_lengths)

    def output_from_static_buffers(self, desc, buffers):
        return MossDecodeOutput(
            audio=buffers.audio,
            audio_lengths=buffers.audio_lengths,
        )

    def eager_forward(self, runtime_inputs):
        self.codec._set_streaming_exec_mask(runtime_inputs.exec_mask)
        return self.codec._decode_frame(
            runtime_inputs.codes_step,
            runtime_inputs.codes_lengths,
        )
```

MOSS forward/stream session 里不再直接持有 graph wrapper。它只做：

```python
graph_output = self._vocoder_cudagraph_manager.replay_or_none(
    "moss_streaming_decode",
    runtime_inputs,
)
if graph_output is None:
    graph_output = self._moss_streaming_routine.eager_forward(runtime_inputs)
```

## Standard Model 零改接入路径

为了让新增架构变化不大的模型“什么都不用改”或只改极少代码，定义一个 base class / mixin：

```python
class StandardCode2WavModelMixin:
    vocoder_graph_routine_name = "standard_code2wav"

    def get_vocoder_cudagraph_routines(self) -> list[VocoderCUDAGraphRoutine]:
        return [
            StandardCode2WavCUDAGraphRoutine(
                name=self.vocoder_graph_routine_name,
                decode_fn=self.decode_for_vocoder_graph,
                allocate_buffers_fn=self.allocate_vocoder_graph_buffers,
                runtime_descriptor_fn=self.vocoder_graph_runtime_descriptor,
                default_capture_sizes_fn=self.default_vocoder_graph_capture_sizes,
            )
        ]
```

模型只要实现标准方法：

```python
def decode_for_vocoder_graph(self, buffers): ...
def allocate_vocoder_graph_buffers(self, desc, device): ...
def vocoder_graph_runtime_descriptor(self, runtime_inputs): ...
def default_vocoder_graph_capture_sizes(self): ...
```

如果模型的 forward contract 更标准，可以进一步把这些方法也放进父类，模型只声明几个 class attributes：

```python
class NewCode2WavModel(StandardCode2WavModelMixin, nn.Module):
    num_codebooks = 16
    default_graph_capture_sizes = [25, 50, 100]
    graph_size_unit = "codec_frames"
```

## Replay Dispatch 策略

manager 默认只做 exact hit：

```text
runtime descriptor == captured descriptor -> replay
otherwise -> eager
```

不做统一 bucket padding。

原因：

- Qwen3 这类 stateless decoder 可以 bucket pad。
- MOSS streaming codec 不能随便 bucket pad，因为 padded T 会推进 streaming state。
- runner 不应该理解每个模型 padding 是否安全。

如果某个模型支持 padding，它可以在 `runtime_descriptor()` 里自己返回 bucket descriptor，并在 `copy_inputs_to_static_buffers()` 里填 padding，同时在 `output_from_static_buffers()` 里 slice valid output。

## Capture 顺序

manager 必须按大 shape 优先：

```python
descriptors = sorted(descriptors, key=lambda d: d.size, reverse=True)
```

这和 vLLM 原生 CUDA graph capture 保持一致，有利于复用 graph pool memory。

## Graph Pool

manager 统一使用 vLLM platform graph pool：

```python
self.pool = current_platform.get_global_graph_pool()
```

模型 routine 不创建自己的 graph pool。

## Capture Context

runner/manager capture 时使用 vLLM 现有上下文：

```python
set_cudagraph_capturing_enabled(True)
with graph_capture(device=self.device):
    ...
set_cudagraph_capturing_enabled(False)
```

模型 routine 不直接控制这些全局状态。

## Memory Accounting

runner 统一记录：

```python
torch.accelerator.synchronize()
torch.accelerator.empty_cache()
start_free = torch.cuda.mem_get_info()[0]

capture()

torch.accelerator.synchronize()
end_free = torch.cuda.mem_get_info()[0]
memory_bytes = start_free - end_free
```

模型不提供 `min_free_gb` 这种私有阈值。

## Logging

runner 统一输出：

```text
Vocoder CUDA graph capture:
  model=<model name>
  routines=[...]
  requested=[...]
  captured=[...]
  failed=[...]
  elapsed=...
  memory=... GiB
```

routine 只在单个 descriptor capture 失败时返回 failure 信息，不决定整体日志格式。

## 最终调用链

### 初始化

```text
GPUGenerationModelRunner.load_model()
  -> load model weights
  -> if model supports get_vocoder_cudagraph_routines()
       routines = model.get_vocoder_cudagraph_routines()
       manager = VocoderCUDAGraphManager(vllm_config, device, routines)
       model.bind_vocoder_cudagraph_manager(manager)
```

### Capture

```text
GPUGenerationModelRunner.capture_model()
  -> if compilation_config.cudagraph_mode == NONE:
       skip
  -> capture normal runner graphs if any
  -> manager.capture()
       resolve descriptors
       large shape first
       allocate static buffers
       warmup
       torch.cuda.graph(...)
       store graph + buffers
       collect stats
  -> return total graph memory bytes
```

### Runtime

```text
GPUGenerationModelRunner.execute_model()
  -> model.forward(...)
       build runtime vocoder inputs
       manager.replay_or_none(routine_name, runtime_inputs)
       if hit:
           return graph output
       else:
           return eager output
```

### Teardown

```text
GPUGenerationModelRunner.clear()
  -> manager.clear()
  -> normal runner cleanup
```

## 对当前 MOSS/Qwen3 语义的判断

### Qwen3-TTS

Qwen3-TTS 更接近 standard routine：

```text
输入窗口 -> code2wav decode -> waveform
无跨请求 streaming state
bucket padding 理论上更容易支持
```

它应该尽量复用 `StandardCode2WavCUDAGraphRoutine`，只提供 buffer schema 和 decode function。

### MOSS-TTS Local v1.5

MOSS-TTS 必须用 custom routine：

```text
有 streaming context
有 per-slot state
有 exec_mask
有 ring KV/cache
capture 前后必须 reset
不能默认 bucket pad
```

但它仍然不需要自定义 runner。所有特殊点都能限制在 `MossStreamingDecodeCUDAGraphRoutine`。

## 最终边界

```text
GPUGenerationModelRunner
  owns global policy and lifecycle

VocoderCUDAGraphManager
  owns graph objects, static buffers, descriptor dispatch, capture stats

VocoderCUDAGraphRoutine
  owns model-specific shape/state mechanics

TTS model
  owns decode math and runtime eager fallback
```

这个边界和 vLLM 原生 CUDA graph 的精神一致：

```text
runner decides when/what to capture
manager stores graphs and dispatches
model provides graphable computation
```

## 结论

最终态不应该是每个 TTS 模型各自实现一个 graph wrapper 并读取自己的配置。

最终态应该是：

```text
统一 runner flow
统一 vLLM compilation_config
统一 capture_model 生命周期
统一 memory/stats/logging
模型只实现 routine
普通模型走 standard routine
特殊模型实现 custom routine
```

这样新增 TTS vocoder/code2wav 模型时，默认不需要理解 CUDA graph 生命周期；只有当模型有 MOSS 这种 stateful streaming 特性时，才实现少量 routine hook。
