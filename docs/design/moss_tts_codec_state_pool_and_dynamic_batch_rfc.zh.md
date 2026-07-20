# MOSS-TTS Codec State Pool 与动态批处理设计（实现版）

状态：Implemented，2026-07-20 根据实际代码更新；高并发 bugfix 待端到端压力复测。

主要实现提交：

```text
b8a7dbbf Refactor MOSS-TTS Local Streaming, c=16 RTF 0.6->0.4, TTFP 815ms-> ~350ms
```

## 1. 摘要

MOSS-TTS Stage 1 streaming codec 需要同时解决两个不同问题：

1. 最多保存多少条 live stream 的跨 chunk decoder state；
2. 当前 codec step 实际有多少条 stream 需要执行 decoder forward。

旧实现将二者都绑定到固定 `codec_stream_slots`。即使只有一个请求 ready，也按完整 slot 数执行 dense
decoder batch。新实现将持久 state capacity 与临时 execution batch 解耦：

- state capacity 直接复用 Stage 1 `scheduler_config.max_num_seqs`；
- 每条 live stream 持有一个稳定 physical state slot；
- 每一步只收集当前有 codes 的请求，形成紧凑 `B_actual`；
- `state_slot_ids` 将紧凑 execution rows 映射到持久 state slots；
- eager 按精确 `B_actual` 执行；
- CUDA Graph 按 `(B_bucket, T)` 捕获，只 padding 到最近的 B bucket；
- graph padding rows 使用不可租用 scratch slots，不接触其他请求的 live state。

例如 capacity 为 16，当前只有 slot 2、7、11 ready：

```text
leaseable state slots: [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15]
                              ^       ^          ^

compact execution rows: [row0, row1, row2, pad]
state_slot_ids:         [   2,    7,   11, scratch]
valid_rows:             [true, true, true, false]
selected graph:         (B=4, T=current_T)
```

capacity 16 只决定可以保留 16 份持久状态，本轮 decoder 只执行 B=4 graph，而不是 B=16。

## 2. 与最初 RFC 的主要差异

最初 RFC 描述了更彻底但尚未落地的 typed scheduler 架构。实际实现选择复用现有 vLLM-Omni
async-chunk、generation scheduler 和 model runner contract。以下表格是当前代码的权威结论：

| 最初 RFC | 实际实现 | 结论 |
| --- | --- | --- |
| 独立 `CodecStreamAdmissionController` | `OmniChunkTransferAdapter._active_streams` 提供 capacity window | 未新增 controller |
| scheduler 分配带 generation 的 `CodecStateLease` | `MossTTSCodecDecoder` 内维护 `request_id -> slot`，session 维护 free/leased slots | 未实现 generation/ACK 协议 |
| 强类型 `CodecChunk`、`CodecDecodeWorkItem` | 保留 flat `input_ids + seq_token_counts + runtime_additional_information` | runner contract 未重构 |
| 独立 Planner/Executor/StatePool 类 | planner/lease 在 `_MossCodecStreamSession`，graph 执行在 `CUDAGraphStreamingDecoderWrapper` | 采用较小落地范围 |
| model 不拥有 request/slot 映射 | model 的 `_stream_req_slots` 是 worker 侧实际 owner map | 与原设计不同 |
| empty terminal 使用显式 T=0 typed event | 继续使用非空 sentinel，并以 `code_flat_numel=0` 表示 control-only terminal | 必须特别处理零长度 slice |
| terminal residual T 一律 eager | 全组均为 terminal 时允许 padding 到更大 T graph，随后立即 reset state | 实现比原 RFC 更积极 |
| 超长 chunk 只允许 Stage 0 切分 | model 的 `_decode_stream_slot_sequence()` 可按 `codec_chunk_frames` 分段执行 | 已实现防御性切分 |
| batched pinned D2H materializer | session 将实际 audio batch `.to("cpu")`，再在 CPU 按 length 切行 | 未单独抽象 materializer |
| 删除 `_MossCodecStreamSession` | `_MossCodecStreamSession` 成为 state lease 与紧凑执行的核心适配层 | 原删除计划取消 |

独立 typed admission/lease ACK 仍可作为未来增强，但不再被本文描述为当前已实现接口。

## 3. 设计目标与不变量

### 3.1 性能目标

- `max_num_seqs` 增大时，C1 不得退化为 capacity-width dense forward。
- 每一步的 decoder 计算量由 ready work 数量决定，而不是 live state 数量决定。
- 常见 `(B,T)` 使用 CUDA Graph；graph miss 使用 eager exact batch。
- padding row 不得污染 leaseable state。
- graph replay 后只 materialize 实际 rows 的 PCM。

### 3.2 正确性目标

- 一个 request 同时最多持有一个 codec state slot。
- 一个 physical slot 同时最多属于一个 request。
- request 从首次 decode 到 terminal 始终复用同一 slot。
- slot reset 后才能返回 free list。
- idle live stream 虽然不参与当前 forward，仍计入 admission capacity。
- 超过 capacity 的请求在 Stage 1 排队，不得把 placeholder prompt 送入 codec。
- capacity invariant 失配时必须显式报错，不能丢 codes、复用 live slot 或返回静默空 PCM。
- normal terminal、empty terminal 和 abort/finished hook 都必须释放 slot。

运行时应满足：

```text
B_actual <= B_bucket <= max_num_seqs
leased codec slots <= max_num_seqs
admitted active streams <= max_num_seqs
```

正常稳定状态还要求 adapter active owner 与 model slot owner 对同一 request 生命周期保持一致。两者处于不同
进程/层级，不要求每个瞬间集合完全相同，但新的 stream 在 model acquire 前必须已经通过 active-window admission。

## 4. 总体数据流

```text
Stage 0 talker output
        |
        v
talker2codec_raw_async_chunk
  - buffer raw code rows
  - initial/steady chunking
  - attach req_id/code_flat_numel/terminal metadata
        |
        v
SharedMemoryConnector / OmniChunkTransferAdapter
  - async receive
  - bounded active-stream window
  - WAITING_FOR_CHUNK queue management
        |
        v
OmniGenerationScheduler
  - only active requests may enter fast path
  - scheduled new/cached request batches
        |
        v
GPUGenerationModelRunner
  - build flat input_ids
  - pass seq_token_counts in input_batch order
  - pass runtime additional information
        |
        v
MossTTSCodecDecoder
  - reconstruct per-request [NQ,T]
  - request_id -> state slot
  - group work by exact T
        |
        v
_MossCodecStreamSession
  - compact [NQ,B_actual,T]
  - eager or graph dispatch
        |
        v
CUDAGraphStreamingDecoderWrapper
  - select (B_bucket,T)
  - pad B with scratch slots
        |
        v
MossAudioTokenizerModel.decode_streaming_batch
  - state_slot_ids gather/scatter
  - persistent KV/offset update
        |
        v
actual-row PCM -> OmniOutput
```

## 5. 配置语义

当前配置不再定义以下 codec 专属字段：

```text
codec_stream_slots
codec_state_capacity
codec_max_decode_batch_size
codec_decode_batch_buckets
codec_decode_frame_sizes
```

复用字段如下：

```yaml
connectors:
  shm:
    extra:
      codec_streaming: true
      initial_codec_chunk_frames: 1
      codec_chunk_frames: 15

stages:
  - stage_id: 1
    max_num_seqs: 32
    enforce_eager: false
    compilation_config:
      cudagraph_capture_sizes: [1, 2, 4, 8, 16, 32]
      cudagraph_num_of_warmups: 1
```

字段语义：

- `max_num_seqs`：Stage 1 active-stream window 和 leaseable codec state capacity。
- `cudagraph_capture_sizes`：codec 内层 graph 的 B buckets，过滤掉大于 `max_num_seqs` 的值。
- `enforce_eager`：为 true 时 codec 不建立 streaming CUDA Graph wrapper。
- `initial_codec_chunk_frames`、`codec_chunk_frames`：定义常见 exact-T capture set。

Local v1.5 当前常见配置为：

```text
B buckets = [1, 2, 4, 8, 16, 32]
T shapes  = [1, 15]
```

注意：同一个 `cudagraph_capture_sizes` 在 Stage-1 外层 vLLM graph 和 codec 内层 wrapper 中可能处于不同
调用层级。codec wrapper 明确将它解释为 batch size，不是 sequence length 或 frame size。

## 6. 类职责与实际接口

### 6.1 `OmniChunkTransferAdapter`

文件：

```text
vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py
```

当 `codec_streaming=True` 时：

```python
self._active_window = model_max_num_seqs
self._active_streams: dict[str, Request] = {}
```

职责：

- 在 waiting/running queue 中按现有顺序提升 active streams；
- active window 满时不再提升新请求；
- active 请求没有 chunk 时转入内部 waiting-for-chunk deque；
- chunk ready 后恢复到 scheduler queue；
- terminal request 实际被 scheduler 选中后从 active set 移除；
- abort/cleanup 时幂等移除 receiver 和 active state；
- `is_active_stream(request_id)` 向 generation scheduler 暴露 admission 状态。

这里的 active window 是当前实现的 capacity admission。它不是 codec model 内的 slot allocator；model slot
allocator 仍作为第二层正确性保护。

### 6.2 `OmniGenerationScheduler`

文件：

```text
vllm_omni/core/sched/omni_generation_scheduler.py
```

generation fast path 在调度 waiting request 前必须检查：

```python
if chunk_transfer_adapter is not None:
    if not chunk_transfer_adapter.is_active_stream(request.request_id):
        skip_for_this_step(request)
        continue
```

非 active 请求放入本 step 的 skipped queue，step 结束时恢复原有等待顺序。它们不能进入 model runner，
因为 async-chunk 预提交 request 仍携带 placeholder prompt，而不是 codec codes。

### 6.3 `GPUGenerationModelRunner`

文件：

```text
vllm_omni/worker/gpu_generation_model_runner.py
vllm_omni/worker/gpu_model_runner.py
```

runner 保留通用 token contract。对 code2wav 模型额外传递：

```python
seq_token_counts: list[int]
runtime_additional_information: list[dict]
```

`seq_token_counts` 的顺序必须与 `input_batch.req_ids` 和 additional-information list 完全一致。优先从
`meta.code_flat_numel` 获取真实长度，以排除 vLLM padding/sentinel token；缺少 metadata 时才使用 scheduler
token counts。

runner 还在 `finished_req_ids` 路径调用 model 的 `on_requests_finished()`，用于 client abort、engine abort 或
没有正常 terminal payload 的清理场景。

### 6.4 `MossTTSCodecDecoder`

文件：

```text
vllm_omni/model_executor/models/moss_tts/modeling_moss_tts_codec.py
```

关键成员：

```python
self._stream_state_capacity = scheduler_config.max_num_seqs
self._stream_req_slots: dict[str, int] = {}
self._stream_session: _MossCodecStreamSession | None = None
```

实际 forward 接口仍为：

```python
forward(
    input_ids,
    runtime_additional_information,
    seq_token_counts=...,
) -> OmniOutput
```

职责：

1. 根据 `seq_token_counts` 切分 flat `input_ids`；
2. 从 metadata 读取 terminal、streaming 和真实 code length；
3. 将合法 segment reshape 为 `[NQ,T]`；
4. 解析稳定 request key；
5. 将 streaming work 组织为 `(output_index, request_id, codes, finished)`；
6. 按 T 分组并交给 session；
7. offline request 继续使用原 batch decode；
8. 将 mono/stereo PCM 恢复到 `OmniOutput.multimodal_outputs`。

request 第一次出现时由 session acquire slot，后续 chunk 查 `_stream_req_slots` 复用相同 slot。正常 terminal
decode 完成后 reset/release；control-only terminal 走专用 empty-finish 路径。

### 6.5 `_MossCodecStreamSession`

它是当前实现中 state lease 和紧凑 execution planning 的组合层。

初始化：

```python
_MossCodecStreamSession(
    codec,
    state_capacity=max_num_seqs,
    n_vq=n_vq,
    graph_batch_sizes=capture_sizes,
    graph_frame_sizes={initial_T, steady_T},
)
```

内部状态：

```python
_free_stream_slots: list[int]
_leased_slots: set[int]
_cudagraph_wrapper: CUDAGraphStreamingDecoderWrapper | None
```

主要接口：

```python
acquire() -> int | None
release(slot: int) -> None
reset_slots(slots: list[int]) -> None
step(
    slot_codes: dict[int, Tensor[NQ,T]],
    terminal_slots: set[int] | None = None,
) -> dict[int, Tensor]
```

`step()` 的输入必须满足：

- 所有 slot 已 lease；
- 同一调用中的 T 完全一致；
- terminal slots 是 work slots 的子集。

它按 dict insertion order 构造：

```text
codes_step:     [NQ, B_actual, T]
state_slot_ids: [B_actual]
```

graph miss 时调用 eager exact batch：

```python
codec.decode_streaming_batch(
    codes_step,
    codes_lengths=full(B_actual, T),
    state_slot_ids=actual_slots,
    valid_rows=ones(B_actual),
)
```

graph hit 时 wrapper 负责 B/T padding。session 只把实际 rows 的 audio 搬到 CPU，然后按 CPU
`audio_lengths` 切片并恢复 slot 映射。

### 6.6 `MossAudioTokenizerModel`

文件：

```text
vllm_omni/model_executor/models/moss_tts/audio_tokenizer_v2.py
```

实际 state-pool 接口：

```python
initialize_decoder_state_pool(
    state_capacity: int,
    scratch_capacity: int = 0,
) -> None

decode_streaming_batch(
    codes: Tensor[NQ,B,T],
    codes_lengths: Tensor[B],
    state_slot_ids: Tensor[B],
    valid_rows: Tensor[B],
) -> MossAudioTokenizerDecoderOutput

reset_decoder_state_slots(state_slot_ids: Tensor) -> None
close_decoder_state_pool() -> None
```

初始化一次性为 decoder-only streaming modules 分配：

```text
physical capacity = leaseable state capacity + scratch capacity
```

`StreamingExecutionContext` 显式携带：

```python
state_slot_ids: int64[B_execution]
valid_rows: bool[B_execution]
```

attention KV、MHA offset、Transformer offset 等 persistent tensors 以 physical slot 为状态维度。每一步
通过 `index_select` gather execution rows，通过 `index_copy_`/scatter 写回相应 slots。`valid_rows=False`
的 graph padding row 不推进有效状态。

### 6.7 `CUDAGraphStreamingDecoderWrapper`

文件：

```text
vllm_omni/model_executor/models/moss_tts/cuda_graph_streaming_decoder_wrapper.py
```

graph key：

```python
graphs: dict[tuple[int, int], CapturedGraph]  # (B_bucket, exact_T)
```

每个 graph entry 持有地址稳定的：

- `static_codes[NQ,B_bucket,T]`；
- `static_lengths[B_bucket]`；
- `static_state_slot_ids[B_bucket]`；
- `static_valid_rows[B_bucket]`；
- static audio 和 audio lengths。

runtime B 选择：

```python
B_bucket = first(capture_size >= B_actual)
```

实际 rows 使用 leaseable slots，padding rows 使用：

```text
state_capacity + arange(B_bucket)
```

即 scratch slot 范围。capture/warmup 后 reset 全部 leaseable + scratch slots，避免 warmup state 泄漏到服务请求。

## 7. CUDA Graph shape 与 padding 规则

### 7.1 B padding

B padding 始终允许，只要存在 `B_bucket >= B_actual` 的已捕获 graph：

| B actual | buckets | 选择 | padding |
| ---: | --- | ---: | ---: |
| 1 | `[1,2,4,8]` | 1 | 0 |
| 3 | `[1,2,4,8]` | 4 | 1 |
| 5 | `[1,2,4,8]` | 8 | 3 |
| 9 | `[1,2,4,8]` | eager 9 | 0 |

graph bucket 不是 execution hard cap。没有合适 graph 时使用 eager，不得丢 request 或强制拆分以伪装 graph hit。

### 7.2 T padding

常规非 terminal step 只允许 exact T graph。较小 T 放入较大 T graph 会推进 KV/offset，因此不能用于后续仍会
继续 decode 的 stream。

实际实现对“整个同-T work group 全部 terminal”的情况开放 T padding：

```python
allow_frame_padding = len(terminal_slots) == len(slots)
```

安全依据：

1. decoder attention 是 causal 的，补在真实 frames 后面的零 codes 不影响真实前缀输出；
2. static lengths 的实际 rows 仍记录真实 T，输出按 audio length 裁剪；
3. 所有相关 slots 在本 step 后立即 reset/release，不会再读取 padding 推进后的 state。

如果同一 group 中存在任意 non-terminal row，则未知 T graph miss 必须 eager exact T。

### 7.3 Scratch capacity

CUDA 环境下：

```text
scratch_capacity = max(codec graph B buckets)
```

eager-only 环境不需要 scratch slots。scratch slots 永不放入 session free list，也不会映射给 request。

## 8. 请求生命周期

### 8.1 首个实际 chunk

1. adapter active window 已 admission 该 request；
2. scheduler/runner 将实际 codec codes 送入 model；
3. model 查不到 request slot；
4. session `acquire()` 返回 free slot；
5. model 建立 `request_id -> slot`；
6. decoder 按该 slot 执行。

如果 active-window invariant 失效导致没有 free slot，model 抛出：

```text
MOSS codec state capacity exhausted ... refusing to drop audio codes.
```

这是系统 invariant error，不是正常 backpressure 路径。

### 8.2 普通 chunk

- request key 命中原 slot；
- 当前有 codes 才进入 execution batch；
- 等待下一 chunk 时持有 slot但不执行 decoder；
- adapter active window 仍将其计入 capacity。

### 8.3 带 codes 的 terminal

1. terminal codes 正常 decode；
2. PCM 已从 decoder result materialize；
3. `_finish_stream_request()` reset slot；
4. 删除 request owner map；
5. slot 返回 free list。

### 8.4 Control-only terminal

Stage 0 为确保 Stage 1 得到一次调度机会，当前仍发送非空 sentinel token，但 metadata 指定：

```text
code_flat_numel = 0
stream_finished = true
```

runner 将该 request 的 `seq_token_counts` 设为 0。model 必须在普通 `seg.numel()==0` early continue 之前处理
这个 terminal metadata，并释放已有 slot。

### 8.5 Abort 与非 payload finish

runner 在 scheduler `finished_req_ids` 路径调用：

```python
model.on_requests_finished(finished_req_ids)
```

hook 幂等查找 owner slot并 reset/release，用于 client disconnect、engine abort 或 terminal payload 未正常到达的
清理路径。正常 terminal 仍由 model forward 主路径释放。

## 9. 超过 capacity 时的排队

以 `max_num_seqs=32`、客户端 concurrency 64 为例，目标状态是：

```text
32 active streams  -> 允许 chunk polling，可持有 codec slots
32 pending streams -> 留在 Stage-1 waiting/hold queues
```

排队由 Chunk Adapter 和 Generation Scheduler 共同完成，不发生在 codec model 内：

1. adapter 从 waiting/running queues 按顺序选择最多 32 个 `_active_streams`；
2. active request 等待 chunk 时进入 adapter 私有 deque；
3. 非 active request 不允许进入 generation fast path；
4. terminal request 被实际调度后从 active set 移除；
5. 下一次 scheduler tick 按队列顺序 promote 新 request。

codec `session.acquire()` 不提供等待机制。它是最后一道 correctness guard，正常运行中不应耗尽。

## 10. 已修复的高并发问题

### 10.1 Empty terminal slot 泄漏

原 forward 顺序先执行：

```python
if seg.numel() == 0:
    continue
```

之后才检查 `code_flat_numel=0 && finished`，导致 control-only terminal 永远无法释放 slot。只有以空 terminal
结束的请求泄漏，因此系统能处理一部分请求，直到泄漏累计到 capacity 后才在下一批/下一轮报错。

修复：先解析 terminal metadata并调用 `_finish_empty_streaming_requests()`，再跳过普通空 segment。

### 10.2 Generation fast path 绕过 active window

高并发 `server.txt` 在 capacity exception 前出现：

```text
MossTTS codec input length 121 not divisible by n_vq 12; skipping.
...
MossTTS codec input length 177 not divisible by n_vq 12; skipping.
```

Local v1.5 的合法 flat codes 长度必须是 `NQ*T`，当前 `NQ=12`。121–177 实际是 Stage-1 async-chunk
预提交的 placeholder prompt 长度。

根因：active window 只从普通 queue 中取走 active requests；超过窗口的非 active requests 仍留在 scheduler
`waiting` queue。Generation fast path 没有检查 active owner，直接调度了这些 placeholder。

修复：

- adapter 增加 `is_active_stream(request_id)`；
- generation scheduler 调度 waiting request 前检查 active 状态；
- 非 active request 本 step skip，结束时恢复队列；
- 只有被 adapter admission 的 request 可以进入 model。

关键 invariant：

```text
streaming request enters Stage-1 model scheduler
    => request_id in adapter._active_streams
```

### 10.3 为什么两个 bug 会一起表现为 capacity exhausted

第一轮中，大部分带实际 terminal frames 的请求正常 release，空 terminal 请求逐步泄漏；因此不是第一批请求就
立即报错。第二轮高并发开始时：

1. 非 active placeholder 被 fast path 错误调度，出现非 `NQ` 倍数 warning；
2. active window 随 terminal 继续 admission 新请求；
3. model state pool 仍被上一轮泄漏 slot 占用；
4. 新合法 chunk acquire 失败，触发 capacity invariant error。

所以 EngineDead/ASGI `response already started` 都是 Stage-1 fatal error 的后续连锁反应，不是根因。

## 11. 性能结果

用户实测：

- C16 mean RTF 从约 `0.6` 降到约 `0.4`；
- `max_num_seqs=32` 时，C1 mean RTF 约 `0.15`；
- capacity 开大后 C1 仍按小 B graph/eager 执行，符合设计预期。

收益来源是消除 inactive slot 的 dense decoder rows。`max_num_seqs` 仍决定 state memory 和 admission capacity，
但不再决定每一步 forward width。

完整 benchmark、Nsight trace 和 commit 对照继续记录在仓库根目录 `main.md`。

## 12. 错误处理原则

以下事件属于正常等待：

- active-stream window 满；
- active request 的下一 chunk 尚未到达。

以下事件属于 request/system error：

- flat code length 不是 `NQ` 的倍数；
- `seq_token_counts` 与 input tensor 不匹配；
- execution batch 引用未 lease slot；
- terminal slot 未释放；
- `session.acquire()` 在已 admission request 上失败；
- graph padding row 指向 leaseable live slot。

禁止用以下方式处理 capacity error：

- 丢弃 audio codes；
- 覆盖最旧 slot；
- 返回 HTTP 200 + 空 PCM；
- 临时扩展 state tensor；
- 捕获异常后继续复用可能损坏的 worker state。

## 13. 当前验证状态

已确认：

- dynamic B 的 C1/C16 性能符合预期；
- CUDA Graph 已按配置捕获 `(B,T)` 组合；
- Ruff、`py_compile` 和 `git diff --check` 通过。

按当前开发约定未新增或运行单元测试。高并发 bugfix 仍需执行：

1. `max_num_seqs=32`、concurrency 64；
2. 第一轮至少 200 条请求；
3. 同一 server 进程立即运行第二轮；
4. 确认无非 `NQ` 倍数 warning；
5. 确认无 capacity exhaustion、zero PCM 和 EngineDead；
6. 检查超过 32 的请求体现为 TTFP/排队增加，而不是被送入 codec；
7. 复测 C1/C16，确认 admission fix 不影响正常性能。

## 14. 尚未落地的增强项

以下内容来自最初 RFC，但不属于当前实现：

- 带 generation 的 `CodecStateLease`，用于拒绝 stale release/abort；
- scheduler/controller 与 worker 之间的显式 reset ACK；
- 强类型 `CodecChunk` 和显式 T=0 terminal，替代 sentinel + flat token contract；
- 独立 `CodecBatchPlanner`、`CodecBatchExecutor` 和 `CodecOutputMaterializer`；
- waiting bytes/chunks 上限和 overload request error；
- active owner、leased slots、B actual/B bucket、release latency metrics；
- 多 Stage-1 replica 的显式 sticky-route invariant 检查；
- terminal、client abort、engine abort 的系统化生命周期测试。

这些增强可以改善可观测性和跨层协议严谨性，但当前动态 batch 的性能收益不依赖它们。若未来实现，应保持本文
已经验证的核心边界：capacity、admission、persistent state ownership 和 execution shape 必须继续解耦。

## 15. 代码索引

| 文件 | 当前职责 |
| --- | --- |
| `vllm_omni/deploy/moss_tts_local.yaml` | Stage-1 capacity、B buckets 和 T chunk contract |
| `vllm_omni/model_executor/stage_input_processors/moss_tts.py` | raw code buffering、chunking、terminal metadata |
| `vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py` | active window、chunk waiting queue、admission |
| `vllm_omni/core/sched/omni_generation_scheduler.py` | generation fast path；禁止非 active placeholder 调度 |
| `vllm_omni/worker/gpu_generation_model_runner.py` | code2wav `seq_token_counts` |
| `vllm_omni/worker/gpu_model_runner.py` | runtime additional information、finished hook |
| `vllm_omni/model_executor/models/moss_tts/modeling_moss_tts_codec.py` | request/slot owner、T grouping、PCM output |
| `vllm_omni/model_executor/models/moss_tts/audio_tokenizer_v2.py` | physical state pool、slot gather/scatter/reset |
| `vllm_omni/model_executor/models/moss_tts/cuda_graph_streaming_decoder_wrapper.py` | `(B,T)` graph capture、scratch padding、replay |

## 16. 最终结论

当前落地架构不是最初 RFC 中完整的 typed lease controller，而是对现有 vLLM-Omni async-chunk pipeline 的
最小侵入式改造：

```text
Chunk Adapter 管 admission
Model/session 管 request -> physical slot
Audio tokenizer 管 persistent state
Graph wrapper 管 compact execution bucket
```

这套职责划分已经实现核心目标：**最大容量决定可保存多少条 stream state，当前 ready work 决定本轮执行多少
decoder rows，`state_slot_ids` 将紧凑 execution batch 连接到持久状态。**

高并发修复进一步补齐了两个必要边界：control-only terminal 必须释放 slot，非 active request 必须停留在
Stage-1 waiting queue。只要这两个 invariant 在压力测试中成立，客户端 concurrency 可以高于 codec state capacity，
代价应当只是排队延迟，而不是 correctness failure。
