# MOSS-TTS Streaming Input Staging Triton 优化设计与实现

本文记录 MOSS-TTS Local v1.5 Stage 1 codec streaming input staging 的完整演进过程，包括原始实现、
初版非 Triton 方案、第一版 Triton 原型、参考 vLLM Model Runner V2 后的最终结构、逐行代码语义、
正确性边界和性能结果。

相关代码：

- `vllm_omni/model_executor/models/moss_tts/streaming_input.py`
- `vllm_omni/model_executor/models/moss_tts/cuda_graph_streaming_decoder_wrapper.py`
- `vllm_omni/model_executor/models/moss_tts/modeling_moss_tts_codec.py`
- `tests/model_executor/models/moss_tts/test_streaming_input.py`

## 1. 背景和目标

每个 codec streaming step 收到若干 active request 的 RVQ codes。session 使用固定数量的 slot 保存
每个请求的 streaming KV cache 和 offset，并通过 `exec_mask[B]` 表示本 step 哪些 slot 应推进状态。

CUDA Graph 对输入地址有稳定性要求，因此 graph capture 时创建固定地址的：

```text
static_codes: [NQ, B, T]
static_lengths: [B]
shared_exec_mask: [B]
```

运行时需要先把动态输入写到这些地址，再 replay graph。优化前的 graph 外路径是：

```text
Python 逐 slot 写 codes_lengths     多次 8-byte H2D
Python 逐 slot 写 exec_mask        多次 1-byte H2D
_set_streaming_exec_mask           mask D2D/传播
static_codes.copy_(codes_step)     codes D2D memcpy
CUDA Graph replay
```

数据量很小，但每个 Python scalar tensor assignment 都会经过 dispatcher、pageable H2D 和相关同步。
目标不是节省几百字节带宽，而是减少 graph replay 前的 CPU 调度、同步和 kernel/memcpy 提交次数。

最终目标：

1. active slots 不再通过 Python 逐元素写 CUDA bool tensor。
2. uniform `T` 不再通过 Python 逐元素写 CUDA long tensor。
3. graph 路径只用一次 launch 完成 codes staging 和 shared mask 更新。
4. 支持 C32，并且不引入随并发数指数增长的数据结构。
5. shared mask 地址保持不变，不破坏已 capture 的 CUDA Graph。

## 2. 原始瓶颈

原始 session 热路径近似为：

```python
codes_step = torch.zeros(n_vq, batch_size, step_t, device=device)
codes_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
exec_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

for slot, codes in slot_codes.items():
    codes_step[:, slot, :] = codes.to(device=device, dtype=torch.long)
    codes_lengths[slot] = int(codes.shape[1])
    exec_mask[slot] = True

self.codec._set_streaming_exec_mask(exec_mask)
entry.static_codes.copy_(codes_step)
entry.graph.replay()
```

其中有三类操作：

| 操作 | 数据 | 必要性 |
| --- | --- | --- |
| `codes_step[:, slot, :] = codes` | 实际 RVQ codes | 必须搬运，本次没有消除 |
| `codes_lengths[slot] = step_t` | 重复的 Python scalar | uniform `T` 下可以整体生成 |
| `exec_mask[slot] = True` | active-slot metadata | 可以编码为 bitset 后一次写入 |

Profile `20260713-110200_stage1_rank0_1783940520` 中观察到：

| 指标 | 数值 |
| --- | ---: |
| 1-byte pageable H2D | `43` |
| 8-byte pageable H2D | `43` |
| Pageable H2D 合计 | `86` |
| Scalar-associated sync | `86 calls / 155.135 ms` |

1 byte 对应 bool mask，8 byte 对应 `torch.long` length。这里的瓶颈是大量细粒度提交和同步，不是
PCIe 吞吐。

## 3. 初版非 Triton 实现

### 3.1 思路

第一版去 H2D 方案把 active-slot 组合编码成整数，并为小 batch 预生成 GPU mask table：

```text
active slots -> integer mask -> mask_table[integer mask]
```

例如 B3 一共有 8 种组合：

```text
000 001 010 011 100 101 110 111
```

运行时只需要选择预生成的一行，不再执行逐 slot bool assignment。对于表规模不可接受的 batch，
再使用另一条 bitset/materialize 路径。

### 3.2 优点

- 实现直接，容易验证。
- 小 batch 下 mask 内容提前生成。
- 可以消除原始的逐元素 1-byte H2D。

### 3.3 C32 扩展性问题

完整 mask table 的行数是 `2^B`，每行有 `B` 个 bool：

```text
B8:   256 rows
B16:  65,536 rows
B32:  4,294,967,296 rows
```

即使每个 bool 只按 1 byte 计算，C32 完整表也约为：

```text
2^32 * 32 bytes = 128 GiB
```

因此 table 方案只能人为设置阈值，例如小 batch 走 table，大 batch 走 fallback。它的问题不只是显存：

1. B16、B17、B32 走不同实现，性能和正确性边界分裂。
2. 新增 concurrency 需要重新判断阈值和 fallback。
3. mask table 解决了 mask materialization，却不能自然融合 `static_codes.copy_()`。
4. table lookup 仍需要把 active mask 选择信息送到 GPU。
5. 指数复杂度与 scheduler 的线性 slot 模型不匹配。

结论：mask table 可以作为小 batch 实验，但不能作为通用 continuous batching 结构。

## 4. 第一版 Triton 原型及其问题

为解决 C32 扩展性，第一版 Triton 原型改为固定 8 个 32-bit word，最多表示 256 slots，并用一个
kernel 写 mask 和复制 codes。方向是正确的，但接入层暴露了太多实现细节。

### 4.1 第一版公开接口

原型对业务层暴露了类似接口：

```python
MaskWords = tuple[int, int, int, int, int, int, int, int]

encode_slot_mask(...) -> MaskWords | None
full_slot_mask(...) -> MaskWords | None
stage_codes_and_mask(..., mask_words: MaskWords)
materialize_mask(..., mask_words: MaskWords)
```

这导致 session 和 graph wrapper 都理解 Triton 的 word layout：

```python
mask_words = self._slot_mask_words(slot_codes)
graph_output = wrapper.decode(codes_step, mask_words)
```

### 4.2 Triton availability 逻辑重复

原型直接执行：

```python
try:
    import triton
    import triton.language as tl
except Exception as exc:
    ...
```

并额外暴露 `is_available()`、`load_error()`。这重复了 vLLM 已有的 Triton 兼容层，CPU worker、
无 Triton 环境和不同 accelerator 的行为容易与 vLLM 主体不一致。

最终实现应统一使用：

```python
from vllm.triton_utils import tl, triton
```

### 4.3 业务层泄漏 kernel ABI

8 个 word 是 kernel launch ABI，不是 streaming session 的领域模型。业务层真正关心的只有：

```text
哪些 slot active
```

当 `MaskWords` 出现在 session、wrapper、测试和类型签名中时：

- 将 32-bit word layout 固化成公共协议。
- 以后改变 word 数量或表示方式需要修改所有调用点。
- CPU fallback 也被迫理解相同 word layout。
- 代码阅读者需要跨多层追踪 tuple 的每个位置。

### 4.4 状态所有权不清晰

原型在 session 内另外分配 `_slot_mask_output`，而 codec 本身已经在 streaming context 中创建了
address-stable `_streaming_exec_mask`。这样形成两份 GPU mask：

```text
session temporary mask -> codec shared mask -> all streaming states
```

即使只剩一次 copy，也没有必要。更合理的结构是 session、wrapper 和所有 streaming state 共同持有
同一个 shared mask 地址，Triton 直接写最终 tensor。

### 4.5 过多验证和 wrapper 层

原型把 Triton load error、availability、tensor validation、word validation、launch 和业务 wrapper
混在一个模块中。单项逻辑都合理，但整体不像 vLLM runner 的 hot-path kernel 接入方式：业务代码
看到了 kernel ABI，而 kernel launcher 又承担了运行环境管理。

### 4.6 Triton 普通全局变量编译错误

第一轮实际运行暴露了：

```text
NameError: Cannot access global variable _MASK_WORD_BITS from within
@jit'ed function
```

Triton 3.x 不允许 kernel 直接捕获普通 Python global：

```python
word_index = slot_offsets // _MASK_WORD_BITS  # 错误
```

最终改为显式 constexpr 参数：

```python
def kernel(..., mask_word_bits: tl.constexpr):
    word_index = slot_offsets // mask_word_bits

kernel[..., mask_word_bits=_MASK_WORD_BITS]
```

这里必须区分：

- `mask_word_bits=32` 是编译期结构常量，应该是 `tl.constexpr`。
- active mask 的 8 个 word 是每 step 变化的运行时参数，不能是 `tl.constexpr`。
- Triton 默认仍可能根据整数是否等于 1、是否满足特定对齐条件生成 specialization，因此 8 个 word
  还必须显式加入 `do_not_specialize`，才能避免 active pattern 导致额外编译。

## 5. 参考 vLLM Model Runner V2

主要参考：

- `vllm/v1/worker/gpu/input_batch.py`
- `vllm/v1/worker/gpu/buffer_utils.py`
- `vllm/v1/worker/gpu/sample/logit_bias.py`
- `vllm/model_executor/kernels/mhc/triton.py`

V2 runner 中 input preparation 类 kernel 通常采用：

```text
runner/state 持有 persistent GPU buffer
        |
薄 Python launcher 计算 grid/shape
        |
私有 @triton.jit kernel 原地更新 buffer
```

共同特点：

1. 使用 `vllm.triton_utils`，不在业务模块重复实现 Triton import fallback。
2. kernel 与薄 launcher 放在 input/sampling preparation 模块中。
3. runner/state 持有 buffer，kernel 不负责 session 生命周期。
4. launcher 接收 tensor 和实际运行时 metadata，不向上暴露内部展开细节。
5. 只有需要进入 `torch.compile` 模型图的模型算子才注册 custom op。

本 kernel 在 `_decode_frame` 和 CUDA Graph replay 之前执行，不属于 compiled model forward，因此直接
调用 Triton launcher，不注册 `direct_register_custom_op`。

## 6. 最终结构

最终运行关系：

```text
_MossCodecStreamSession
  |
  |-- active slots -> Python integer bitset
  |
  |-- owns/reference shared_exec_mask ---------------------------+
  |                                                              |
  +--> CUDAGraphStreamingDecoderWrapper                          |
         |                                                       |
         | prepare_streaming_inputs(...)                         |
         v                                                       |
      Triton kernel                                              |
         |-- codes_step -> graph static_codes                    |
         +-- integer bitset -> shared_exec_mask -----------------+
                                                                 |
      CUDA Graph replay                                          |
         +-- 92 MHA and other streaming states read same mask <--+
```

对外只有三个概念：

```python
active_slot_mask = encode_slot_mask(slots, batch_size)
prepare_streaming_exec_mask(exec_mask, active_slot_mask)
prepare_streaming_inputs(source_codes, static_codes, exec_mask, active_slot_mask)
```

8-word 展开完全留在 launcher 内部。

## 7. `streaming_input.py` 逐段说明

### 7.1 常量

```python
_MASK_WORD_BITS = 32
_NUM_MASK_WORDS = 8
MAX_STREAM_SLOTS = 256
_CODE_BLOCK_SIZE = 256
```

- 每个 scalar word 表示 32 个 slots。
- 8 个 word 支持 256 slots。
- 每个 Triton program 搬运 256 个 codes 元素。

### 7.2 Active slots 编码

```python
active_slot_mask = 0
for slot in slots:
    active_slot_mask |= 1 << slot
```

例如 `[0, 3, 7]` 编码为 `0b10001001`。这是普通 Python integer 运算，不创建 CPU/CUDA tensor。

### 7.3 拆成 signed int32

```python
word = (active_slot_mask >> (word_idx * 32)) & 0xffffffff
```

Python integer 没有固定宽度，因此先切出每个 32-bit word。若最高位为 1，再转换为相同 bit pattern 的
signed int32：

```text
0x80000000 -> -2147483648
0xffffffff -> -1
```

kernel 中使用算术右移后 `& 1`，仍可正确读取每一位。

### 7.4 Kernel 参数

```python
source_codes,
static_codes,
exec_mask,
num_codes,
num_slots,
mask_word_0 ... mask_word_7,
copy_codes: tl.constexpr,
code_block_size: tl.constexpr,
mask_block_size: tl.constexpr,
mask_word_bits: tl.constexpr,
```

动态参数：

- 三个 GPU tensor 指针。
- 实际元素数和 slot 数。
- 每 step 变化的 8 个 mask word。

编译期参数：

- 是否复制 codes。
- codes 和 mask block size。
- 每个 mask word 的 bit 数。

Kernel decorator 将 8 个 active word 全部加入 `do_not_specialize`。因此 active pattern 改变不会单独
编译新 kernel；这些 word 作为 kernel launch 参数进入 CUDA command，不形成独立的 pageable
`cudaMemcpyAsync`。

### 7.5 Codes staging

```python
code_offsets = program_id * code_block_size + tl.arange(0, code_block_size)
code_mask = code_offsets < num_codes
codes = tl.load(source_codes + code_offsets, mask=code_mask)
tl.store(static_codes + code_offsets, codes, mask=code_mask)
```

`source_codes` 和 `static_codes` 都要求 contiguous，所以 `[NQ, B, T]` 可以按一维连续地址搬运。
最后一个 program 通过 `code_mask` 防止越界。

当 `copy_codes=False` 时，Triton 在编译期删除整个分支；mask-only 路径不会读取或复制 codes。

### 7.6 Mask materialization

```python
if program_id == 0:
```

只有第一个 program 写 mask。否则 codes grid 变大时，每个 program 都会重复写同一 tensor。

```python
word_index = slot_offsets // mask_word_bits
bit_index = slot_offsets % mask_word_bits
```

slot 63 映射为 word 1 的 bit 31。随后使用 `tl.where` 从 8 个动态 word 中选择对应 word：

```python
active = ((word >> bit_index) & 1) != 0
tl.store(exec_mask + slot_offsets, active, mask=slot_offsets < num_slots)
```

结果直接写入所有 streaming state 已绑定的 shared mask。

### 7.7 Grid 和 mask block

```python
num_codes = source_codes.numel() if copy_codes else 0
grid = (max(1, triton.cdiv(num_codes, 256)),)
mask_block_size = triton.next_power_of_2(max(1, exec_mask.numel()))
```

mask-only 路径也至少启动一个 program。mask block 示例：

```text
B8   -> 8 lanes
B32  -> 32 lanes
B48  -> 64 lanes，其中 16 lanes 被 mask
B256 -> 256 lanes
```

### 7.8 两个公开 launcher

Graph 路径：

```python
prepare_streaming_inputs(source_codes, static_codes, exec_mask, active_slot_mask)
```

执行 codes copy 和 mask update。

Reset/eager 路径：

```python
prepare_streaming_exec_mask(exec_mask, active_slot_mask)
```

它把 `exec_mask` 同时作为未使用的 source/static 参数传入统一 kernel，并设置 `copy_codes=False`。
由于分支是 constexpr，编译结果不会执行自拷贝。

## 8. CUDA Graph Wrapper 接入

Wrapper 构造函数显式接收：

```python
exec_mask: torch.Tensor
```

它不再创建临时 mask，也不再调用 `_set_streaming_exec_mask()`。session、wrapper 和 codec states 持有
同一地址。

Capture warmup 使用：

```python
prepare_streaming_inputs(codes, codes, exec_mask, full_slot_mask)
```

source/destination alias 只发生在 warmup。每个元素独立 load/store，因此安全；作用是提前编译 fused
specialization 并设置 full mask。

运行时：

```python
prepare_streaming_inputs(
    codes_step,
    entry.static_codes,
    exec_mask,
    active_slot_mask,
)
entry.graph.replay()
```

Triton kernel 与 graph replay 提交到同一 current stream，CUDA stream ordering 保证 graph 在看到新的
codes 和 mask 后才执行，不需要 CPU synchronize。

Graph 只按精确 `T` 命中。不能把更短 step padding 到更大的 capture size，因为 codec 会按 tensor 的
`T` 推进 KV offset，padding 会改变状态语义。

## 9. Session 接入

Session 必须先进入 codec streaming context：

```python
self._exit_stack.enter_context(codec.decoder_streaming(batch_size))
shared_exec_mask = codec._streaming_exec_mask
```

shared mask 在 streaming context 中创建，并绑定给所有 live `StreamingState`。之后 session 将相同 tensor
传给 graph wrapper。

### 9.1 Reset

改前：

```python
reset_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
reset_mask[slots] = True
```

改后：

```python
active_slot_mask = encode_slot_mask(slots, batch_size)
prepare_streaming_exec_mask(shared_exec_mask, active_slot_mask)
reset_mask = shared_exec_mask
```

实际代码由 `_prepare_exec_mask()` 返回同一个 shared tensor。CPU 路径保留普通循环 fallback。

### 9.2 Streaming step

所有 active slot 必须具有相同的 `T`：

```python
step_lengths = {codes.shape[1] for codes in slot_codes.values()}
```

因此 eager fallback 不需要逐 slot 写 lengths：

```python
codes_lengths = torch.full((batch_size,), step_t, device=device)
```

inactive slot 的 length 也为 `step_t`，但：

1. inactive output 不会返回给请求。
2. shared `exec_mask` 阻止其 KV cache 和 offset 前进。

Graph 路径的 `static_lengths` 在 capture 时已经是对应精确 `T` 的完整向量，因此运行时不更新 lengths。

本次仍保留：

```python
codes_step[:, slot, :] = codes.to(device=device, dtype=torch.long)
```

这是实际 codes 数据搬运，不属于 scalar metadata H2D。后续若要继续优化，需要改变 codes batch packing，
不能把它与本次 mask 优化混为一项。

## 10. 正确性和 CUDA 语义

### 10.1 Address stability

CUDA Graph capture 的 layer 读取 `state.exec_mask` 的固定地址。Triton 只修改该地址中的值，不替换
tensor，因此 graph replay 合法。

### 10.2 Stream ordering

运行时没有 CPU synchronize：

```text
current stream: Triton staging kernel -> graph replay
```

同一 stream 上后提交的 graph node 能看到 staging kernel 的写入。

### 10.3 Scalar kernel argument 与 H2D

8 个 mask word 是 CUDA kernel launch 参数。它们需要由 host driver 填入 launch parameter buffer，
但不会形成独立 `cudaMemcpyAsync` H2D，也不会产生原始 tensor scalar assignment 对应的同步序列。

### 10.4 动态 mask 不重编译

Triton 不会把所有普通 scalar 的完整数值直接作为 constexpr，但默认会对整数的 `==1` 和某些对齐
属性进行特化。最终 kernel 因此显式声明：

```python
@triton.jit(
    do_not_specialize=[
        "mask_word_0",
        "mask_word_1",
        "mask_word_2",
        "mask_word_3",
        "mask_word_4",
        "mask_word_5",
        "mask_word_6",
        "mask_word_7",
    ]
)
```

这样 `[0, 1]` 和 `[3, 7, 15]` 等不同 active pattern 使用同一个已编译 specialization。

可能产生 specialization 的维度包括：

- `copy_codes=True/False`。
- 不同 `mask_block_size`，通常对应不同 batch size 区间。
- 不同 tensor dtype/layout signature。

### 10.5 B32 和 B256

该结构的 metadata 复杂度是：

```text
Python encode: O(number of active slots)
GPU mask write: O(B)
mask launch parameters: 固定 8 x int32
```

它不依赖 `2^B` table。C32、C64、C256 使用同一代码路径，当前显式上限为 256。

## 11. 测试

`test_streaming_input.py` 覆盖：

1. B8/B32/B64/B256 bitset 编解码。
2. slot 越界。
3. 超过 256 slots。
4. fused codes staging 的逐元素一致性。
5. fused mask 的一致性。
6. mask-only specialization。
7. 跨 word 边界的 slots，例如 31、32、63。

需要在有 CUDA/Triton 的 worker 环境运行：

```bash
pytest -q tests/model_executor/models/moss_tts/test_streaming_input.py
```

## 12. Profiling 和端到端结果

去 scalar H2D 的中间验证 profile：

```text
results/moss_local_stage1/batch_8/
20260713-125811_stage1_rank0_1783947491
```

| 指标 | 改前 `110200` | 改后 `125811` |
| --- | ---: | ---: |
| Pageable H2D | `43 x 1B + 43 x 8B` | `0` |
| Scalar-associated sync | `86 / 155.135 ms` | `0` |
| 全部 sync | `116 / 155.222 ms` | `32 / 0.107 ms` |
| `execute_context_7(84)_generation_0(0)` | `78.703 ms` | `44.241 ms` |

`125811` 早于最终通用 Triton 融合实现，因此只能证明消除 scalar assignment 的方向，不能用于声称
最终 fused kernel 的具体 CUDA 时间。

最终 Triton 版本 C16：

| 指标 | Shared mask C16 | Triton staging C16 | 变化 |
| --- | ---: | ---: | ---: |
| Mean RTF | `0.717` | `0.642` | `-10.5%` |
| Mean TTFP | `937 ms` | `806 ms` | `-14.0%` |
| Audio throughput | `22.317` | `24.824` | `+11.2%` |
| Mean WER | `0.0261` | `0.0248` | 均处于低位 |
| Successful samples | `100` | `100` | 无失败 |

最终实现已经得到端到端收益，但仍需采集新的 Nsight Systems trace，确认 graph replay 前只剩一个
Triton staging kernel，并测量它相对原生 `cudaMemcpyAsync` codes copy 的成本。

## 13. 已知边界和后续方向

1. 最大支持 256 streaming slots，超过后明确报错。
2. 对较大 `T`，Triton load/store kernel 的纯带宽可能不如原生 D2D memcpy，需要 profile 决定是否
   设置 size threshold；当前 streaming step 较小，launch 融合收益更重要。
3. `codes_step` 的构造仍按 active slot 执行 D2D slice copy，是下一层可优化对象。
4. eager 路径的 `torch.full(codes_lengths)` 仍是一个 GPU fill kernel，但没有 pageable scalar H2D。
5. 需要补 C8/C32、长时间 slot acquire/release/reset、waveform/SIM/UTMOS 和 streaming underrun 回归。
6. 若未来 active metadata 不止一个 bool mask，可以参考 vLLM V2 `StagedWriteTensor`/UVA buffer，但必须
   先确认不会重新引入高频 H2D 和同步。

## 14. 结论

本次优化的关键不是单独写了一个 Triton kernel，而是重新划分了状态和 ABI：

- session 只表达 active slots，不理解 8-word kernel ABI。
- launcher 隐藏 bitset 拆分和 Triton specialization。
- graph wrapper 持有 persistent buffers，并在 replay 前原地更新。
- codec 所有 streaming state 共享同一个 address-stable mask。
- 一次 Triton launch 融合 codes staging 和 mask materialization。

相比初版 mask table，它对 C32/C256 是线性复杂度；相比第一版 Triton 原型，它不再把 kernel ABI、
availability 和临时 mask 泄漏到业务层。这一结构与 vLLM Model Runner V2 的 input preparation 方式一致，
也为后续继续融合 codes batch packing 保留了清晰边界。
