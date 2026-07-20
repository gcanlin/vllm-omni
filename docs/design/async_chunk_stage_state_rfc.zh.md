# RFC: Async Chunk Stage State

## 状态

草案。

## 背景

`async_chunk` 路径允许上游 stage 在请求尚未结束时，把中间产物按 chunk 发送给下游 stage。当前 `OmniChunkTransferAdapter` 会把自身作为 `transfer_manager` 传给模型侧 processor，例如：

```python
payload_data = self.custom_process_next_stage_input_func(
    transfer_manager=self,
    multimodal_output=multimodal_output,
    request=request,
    is_finished=is_segment_finished,
)
```

这让模型 processor 可以复用 adapter 上的 connector、chunk id、配置和 request 生命周期信息。但实际代码里，模型也开始直接在 `transfer_manager` 上挂自己的状态字段：

```python
transfer_manager.code_prompt_token_ids[request_id].append(frame)
transfer_manager.request_payload[request_id] = ref_code
transfer_manager._cached_ic[request_id] = initial_chunk_size
transfer_manager._moss_tts_raw_state[request_id] = {...}
```

这些字段分散在框架和模型 processor 之间，语义不统一，生命周期清理也需要 adapter 硬编码知道每个字段。

## 问题

当前写法可以工作，但它把模型私有状态暴露成了 `OmniChunkTransferAdapter` 的动态属性，带来几个问题：

1. 隐式 API：模型 processor 可以随意给 `transfer_manager` 加字段，框架没有正式约束。
2. 清理不完整：request finish、segment finish、abort、preemption 等路径需要清理状态，但 adapter 不知道所有模型私有字段。
3. 命名冲突：多个模型或后续框架字段可能复用同名属性。
4. 类型不清晰：`request_payload`、`code_prompt_token_ids` 等字段被不同模型复用成不同语义。
5. 测试困难：状态分散在动态属性里，难以做泄漏检测和生命周期断言。

MOSS-TTS-Local 暴露了这个问题，因为它需要 per-request cursor：

```text
talker 输出 accumulated snapshot
async_chunk 用 total_emitted 切出新增 frames
Stage1 codec streaming session 解码新增 frames
```

这需要保存 `total_emitted`、`prompt_emitted` 等状态。把这些状态挂成 `_moss_tts_raw_state` 可以跑通，但不是干净的框架接口。

## 目标

引入一个正式的 async chunk state 抽象，用来承载模型 processor 在 chunk 组装阶段需要的 per-request 状态。

目标包括：

1. 为模型 processor 提供 namespaced per-request state。
2. 由 `OmniChunkTransferAdapter` 统一管理 state 生命周期。
3. 避免模型直接给 `transfer_manager` 动态挂私有字段。
4. 支持 request finish、segment finish、abort 的统一清理。
5. 保留模型 processor 对 chunk 语义的控制权。

## 非目标

本 RFC 不试图统一所有模型的 chunk 策略。

这些逻辑仍然属于模型 processor：

1. 如何判断 frame 是否有效。
2. 什么时候发 chunk。
3. 是否带 left context。
4. ref code 是拼进 `codes.audio` 还是作为 `codes.ref` side-band。
5. 如何 flatten codec codes。
6. meta 字段如何组织。
7. terminal empty payload 如何表达。

本 RFC 也不覆盖 KV cache transfer、scheduler queue state、Stage1 decoder runtime state。

## 设计概览

新增一个 `StageStateStore`，由 `OmniChunkTransferAdapter` 持有：

```python
class OmniChunkTransferAdapter(...):
    def __init__(self, vllm_config):
        ...
        self.stage_state = StageStateStore()
```

模型 processor 不再写：

```python
transfer_manager._moss_tts_raw_state[req_id]
```

而是写：

```python
state = transfer_manager.stage_state.get(
    namespace="moss_tts_raw",
    request_id=req_id,
    factory=MossTTSRawState,
)
```

`namespace` 用来隔离模型或处理器的状态。`factory` 用来创建模型自己的 typed state。

## API 草案

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

最小实现可以是：

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

为了区分 request finish 和 realtime segment finish，state 可以选择实现 hook：

```python
class AsyncChunkRequestState(Protocol):
    def on_segment_end(self) -> None: ...
    def on_request_end(self) -> None: ...
    def on_abort(self) -> None: ...
```

`StageStateStore.cleanup_segment()` 调用 `on_segment_end()`。`cleanup_request()` 调用 `on_request_end()` 后删除该 request 的全部 state。

Hook 是可选的。普通 dataclass 不实现 hook 也可以使用。

## 模型侧 State 示例

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

Processor 使用方式：

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

Processor 使用方式：

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

这可以替代：

```python
transfer_manager.code_prompt_token_ids[request_id]
transfer_manager.request_payload[request_id]
transfer_manager._cached_ic[request_id]
```

## Transfer Manager 通用操作

除字段容器外，`OmniChunkTransferAdapter` 可以逐步提供这些通用 helper。

### Config 读取

当前多个 processor 重复读取：

```python
connector = getattr(transfer_manager, "connector", None)
raw_cfg = getattr(connector, "config", {}) or {}
cfg = raw_cfg.get("extra", raw_cfg)
```

建议提供：

```python
cfg = transfer_manager.chunk_config()
chunk_size = transfer_manager.get_extra_int("codec_chunk_frames", default=25)
```

### Chunk 计数器

当前 processor 会直接读取：

```python
transfer_manager.put_req_chunk[request_id]
```

建议提供：

```python
chunk_id = transfer_manager.put_chunk_id(request_id)
sent = transfer_manager.sent_chunk_count(request_id)
```

### FrameBuffer Helper

对于 Qwen3、Higgs、Mimo、Voxtral 这类逐帧 append 的模型，可以提供可选 helper：

```python
class FrameBuffer:
    def append(self, frame): ...
    def length(self) -> int: ...
    def window(self, context_frames: int, chunk_frames: int): ...
    def clear(self) -> None: ...
```

对于 MOSS Local 这类 accumulated snapshot 模型，可以提供 cursor helper：

```python
class SnapshotCursor:
    total_emitted: int

    def pending(self, snapshot) -> int: ...
    def take(self, snapshot, n: int): ...
```

这些 helper 应该是可选工具，不应成为 processor 必须继承的基类。

## 生命周期

### Request Finish

当 terminal payload 成功发送后，sender 侧调用：

```python
self.stage_state.cleanup_request(external_req_id)
```

这应该发生在 connector `put()` 成功之后，保持现有行为：避免 cleanup 和 save loop 的竞态。

### Segment Finish

对于 realtime/resumable request，segment finish 不等于 request finish。

当 `is_segment_finished=True` 时，adapter 可以调用：

```python
self.stage_state.cleanup_segment(external_req_id)
```

是否清理全部状态由 state hook 决定。例如某些 realtime 状态可能跨 segment 保留，某些 codec chunk 状态需要 segment 结束即清空。

### Abort / Scheduler Cleanup

当 scheduler 调用 adapter cleanup 时，应该统一清理：

```python
self.stage_state.cleanup_request(external_req_id)
```

如果可以区分 abort，也可以调用：

```python
self.stage_state.abort_request(external_req_id)
```

内部先调用 `on_abort()`，再删除 state。

## 迁移计划

### Phase 1: 引入 Store

1. 新增 `StageStateStore`。
2. 在 `OmniChunkTransferAdapter.__init__` 中创建 `self.stage_state`。
3. 在 `cleanup_sender()`、`cleanup()`、segment cleanup 路径中调用 state store cleanup。
4. 不改现有模型 processor 行为。

### Phase 2: 迁移 MOSS-TTS-Local

把：

```python
transfer_manager._moss_tts_raw_state
```

迁移为：

```python
transfer_manager.stage_state.get("moss_tts_raw", req_id, MossTTSRawState)
```

这一步能直接消除当前最明显的动态私有字段。

### Phase 3: 迁移通用 TTS Processor

逐步迁移：

```text
code_prompt_token_ids
request_payload
_cached_ic
higgs_v3_emitted_frames
_pending_streaming_prefills
```

其中 `put_req_chunk/get_req_chunk/finished_requests` 仍然留在 adapter，因为它们是 transfer 协议状态，不是模型 chunk 组装状态。

### Phase 4: 提供 Helper

在迁移稳定后，再添加：

```text
chunk_config()
get_extra_int()
FrameBuffer
SnapshotCursor
```

避免一开始引入过重抽象。

## 兼容性

该设计可以保持向后兼容：

1. `transfer_manager` 参数仍然传给现有 processor。
2. 旧字段短期保留。
3. 新模型优先使用 `stage_state`。
4. 旧 processor 可按模型逐步迁移。

## 备选方案

### 方案 A: 每个模型继承 TransferManager

不推荐。

这会把模型逻辑和 connector 生命周期强耦合。`OmniChunkTransferAdapter` 已经负责线程、connector、queue、chunk id、cleanup。如果每个模型继承它，会产生复杂的多继承或注册问题，也会让模型 processor 难以复用。

### 方案 B: 只用一个 dict

例如：

```python
transfer_manager.stage_state["moss_tts_raw"][req_id]
```

这比动态字段好，但仍缺少生命周期 hook、类型创建和清理入口。可以作为最小实现的内部结构，但不建议作为公开 API。

### 方案 C: 模型 processor 自己管理全局状态

不推荐。

全局状态会跨 adapter、跨 stage、跨 worker 泄漏，生命周期更难控制，也不适合多实例部署。

## 风险

1. 如果一次性迁移太多 processor，容易引入 streaming 行为回归。
2. 如果 hook 语义设计过重，模型侧会被迫实现不需要的接口。
3. 如果 state cleanup 时机不对，可能导致 save loop 中还未发送的 terminal chunk 读不到状态。

对应缓解：

1. 先只引入 store，不迁移行为。
2. Hook 全部 optional。
3. Sender 侧 request cleanup 继续放在 connector `put()` 成功之后。
4. 每迁移一个 processor，都加 async_chunk 单测覆盖首包、稳态包、尾包、abort/finish cleanup。

## 测试计划

1. `StageStateStore` 单测：
   - `get()` 懒创建。
   - namespace 隔离。
   - `cleanup_request()` 清理所有 namespace。
   - optional hook 被调用。

2. Adapter 生命周期单测：
   - terminal chunk 成功发送后清理 sender state。
   - non-terminal chunk 不清理 state。
   - segment finish 调用 segment cleanup。
   - abort cleanup 清理 state。

3. MOSS-TTS-Local processor 单测：
   - accumulated snapshot 只发送 delta。
   - `codes.ref` 只在首包发送。
   - finished flush 剩余 frames。
   - cleanup 后同 request id 不复用旧 cursor。

4. Qwen3-TTS 回归测试：
   - prefill zero codes 不进入 frame buffer。
   - decode frame append 正常。
   - left context window 不变。
   - ref context 首包行为不变。

## 结论

`stage_state` 的核心价值不是把模型 chunk 逻辑搬进框架，而是把当前隐式的 `transfer_manager` 动态字段正式化。

推荐采用：

```text
TransferManager owns lifecycle and namespaced state.
Model processor owns state schema and chunk semantics.
```

这样可以修复当前状态散落和清理困难的问题，同时保留每个模型对 streaming chunk 行为的控制权。
