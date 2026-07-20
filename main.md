# MOSS-TTS Local v1.5 性能优化记录

本文持续记录 `vllm-omni` 中 MOSS-TTS Local v1.5 的性能优化过程。每个优化点独立记录问题、
profiling 证据、代码修改和优化后的结果，避免多个变量混在一次实验里导致错误归因。

当前主要关注 Stage 1 codec streaming decode 的高并发性能。除非特别说明，表中的并发数均为 8，
RTF 越低越好，TTFP 越低越好。

## 优化汇总

| # | 日期 | 优化点 | 主要改动 | Profiling 结果 | 端到端结果 | 状态 |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | 2026-07-10 | Codec FP32 compute 改为 BF16 | decoder 全路径启用 BF16 autocast；CUDA streaming KV cache 固定为 BF16 | Attention kernel 从 FP32 变为 BF16，单层平均 `172.296 -> 64.480 us`，下降 62.6%；归一化 Self CUDA `93.17 -> 49.33 ms/codec step`，下降 47.1%（总时间受样本长度影响，仅作辅助参考） | RTF `0.6~0.7 -> 0.433`；TTFP `~700 -> 563 ms`；吞吐 `18.818` | 已生效 |
| 2 | 2026-07-13 | 消除 streaming `exec_mask` 重复 D2D | session 共享 address-stable mask；每步只复制一次；缓存 streaming module；仅初始化 decoder streaming state | `_set_streaming_exec_mask` `10.381 -> 0.017 ms/step`；D2D `780.6 -> 104.8 次/step`，下降 86.6%；D2D CUDA 时间下降 86.2% | RTF `0.433 -> 0.371`，下降 14.3%；TTFP `563 -> 431 ms`，下降 23.4%；吞吐 `18.818 -> 21.780`，提升 15.7% | 已生效 |
| 3 | 2026-07-15 | 消除 scalar H2D 并融合 streaming input staging | active slots 编码为 Python bitset；Triton 单 kernel 融合 `codes` staging 和 shared mask 更新；支持 256 slots | 中间版 profile 中 pageable H2D `86 -> 0`，scalar sync `86/155.135 ms -> 0`；最终 Triton trace 待补 | C16 RTF `0.717 -> 0.642`，下降 10.5%；TTFP `937 -> 806 ms`，下降 14.0%；吞吐 `22.317 -> 24.824`，提升 11.2% | 端到端已验证，待补最终 profile |
| 4 | 2026-07-20 | Codec stream state pool 与动态执行 batch | state capacity 与每步执行 batch 解耦；按 `(B_bucket, T)` 捕获 CUDA Graph；通过 `state_slot_ids` 间接寻址持久状态；超过 capacity 的流在 Stage 1 排队 | 动态 batch 的最终 trace 待补；CUDA Graph 已覆盖配置的 `(B,T)` 组合 | C16 RTF 约 `0.6 -> 0.4`；`max_num_seqs=32` 时 C1 RTF 约 `0.15` | 性能已验证；高并发 admission/terminal bugfix 待压力复测 |

### 端到端性能轨迹

| 版本 | Concurrency | RTF mean | TTFP | Throughput | 说明 |
| --- | ---: | ---: | ---: | ---: | --- |
| 初始 FP32 codec | 8 | `0.6~0.7` | `~700 ms` | 未记录 | 用户实测范围 |
| BF16 codec | 8 | `0.433` | `563 ms` | `18.818` | `voice_clone` benchmark |
| BF16 + shared `exec_mask` | 8 | `0.371` | `431 ms` | `21.780` | WER `0.0348`，100/100 成功 |
| BF16 + shared `exec_mask` | 16 | `0.717` | `937 ms` | `22.317` | WER `0.0261`，100/100 成功 |
| BF16 + shared mask + Triton input staging | 16 | `0.642` | `806 ms` | `24.824` | WER `0.0248`，100/100 成功 |
| Dynamic codec batch | 1 | `~0.15` | 未记录 | 未记录 | `max_num_seqs=32` 时不再按 32-row 固定 batch 执行 |
| Dynamic codec batch | 16 | `~0.4` | 未记录 | 未记录 | 用户实测，相对改造前约 `0.6`；完整 benchmark 待归档 |

## 测量和归因规则

1. 端到端 benchmark 是最终性能结论；profiler 用于解释收益来自哪里。
2. 不同 trace 包含的 codec step 数和每步输入帧数可能不同。总 CUDA 时间不能直接横向比较，优先比较
   `每 codec step`、`每 Transformer layer` 或相同算子的平均耗时。
3. 一次只改一个主要变量。若 profile 同时包含其他实验性代码，必须在结论中注明。
4. 每次保留 profile 目录、代码 commit/diff、C1/C8 benchmark 和音频质量结果。
5. WER、SIM、UTMOS 未测时明确标记为未测，不用性能结果代替正确性和质量验证。

---

## 1. Codec FP32 compute 改为 BF16

### 1.1 改前现状

Codec checkpoint 和整个模型以 FP32 加载。改造前 `_decode_frame()` 没有 autocast，quantizer、六个
decoder stage 和其中 92 个 Transformer layer 都沿 FP32 路径执行。streaming MHA 初始化 KV cache
时也直接继承 projection weight 的 FP32 dtype。

相关热路径：

- `vllm_omni/model_executor/models/moss_tts/audio_tokenizer_v2.py`
  - `MossAudioTokenizerModel._decode_frame()`
  - `MossAudioTokenizerMultiheadAttention._init_streaming_state()`
- `vllm_omni/model_executor/models/moss_tts/modeling_moss_tts_codec.py`
  - `MossTTSCodecDecoder.__init__()`

这里需要区分三个概念：

- **权重存储**：目前 codec 仍通过 `codec.to(device=device, dtype=torch.float32)` 以 FP32 存储。
- **计算 dtype**：decoder forward 在 CUDA 上通过 autocast 使用 BF16 GEMM/attention。
- **KV cache dtype**：streaming KV cache 在 CUDA 上直接分配为 BF16。

因此本次优化准确地说是 **FP32 storage + BF16 compute + BF16 KV cache**，不是把整个 checkpoint
和所有权重永久转换为 BF16。

### 1.2 改前 Profiling

Profile：

```text
results/moss_local_stage1/batch_8/
20260709-180509_stage1_rank0_1783620309
```

关键证据：

| 指标 | 改前结果 |
| --- | ---: |
| Attention kernel | `fmha_cutlassF_f32_aligned_64x64_rf_sm80` |
| Attention 调用数 | `1656 = 92 layers x 18 codec steps` |
| Attention CUDA 总时间 | `285.322 ms` |
| Attention 平均时间 | `172.296 us/layer` |
| Self CUDA 总时间 | `1.677 s` |
| 归一化 Self CUDA | `93.17 ms/codec step` |

Profile 中主要 GEMM 同样是 `f32f32` kernel。对于短序列、多层数的 codec decoder，FP32 不仅增加
算力开销，还让每层 attention、projection 和 FFN 读写更多字节。92 层的重复使这部分成本被放大。

### 1.3 分析结论

Codec inference 不需要训练梯度，H20 可以直接执行 BF16 Tensor Core GEMM。最小改动方案是：

1. 不改变 checkpoint 加载和权重存储，降低模型加载及兼容性风险。
2. 在完整 decode 区间外层统一启用 BF16 autocast，保证 eager 和 CUDA Graph capture 使用同一条路径。
3. streaming KV cache 直接分配为 BF16，避免 KV 长期以 FP32 占用显存和带宽。

只在零散 layer 上添加 autocast 会使 eager、warmup、graph capture 的 dtype 行为难以保持一致，因此
autocast 必须覆盖 quantizer decode 和全部 decoder stage。

### 1.4 代码修改

对应提交：

```text
b8de601b fp32 to bf 16, RTF 0.6 -> 0.4
```

第一处修改是将 CUDA streaming KV cache 固定为 BF16：

```python
in_proj = cast(nn.Linear, self.in_projs[0])
device = cast(torch.device, in_proj.weight.device)
weight_dtype = cast(torch.dtype, in_proj.weight.dtype)
dtype = torch.bfloat16 if device.type == "cuda" else weight_dtype
```

第二处修改是将完整 decoder forward 放入统一的 BF16 autocast：

```python
with torch.autocast(
    device_type=device.type,
    dtype=torch.bfloat16,
    enabled=device.type == "cuda",
):
    zq = quantizer.decode_codes(codes)
    for decoder_module in self.decoder:
        d, d_lengths = decoder_module(d, d_lengths)
```

CPU 路径仍继承原始 weight dtype，不强制使用 BF16。

### 1.5 改后 Profiling

用于对照的改后 profile：

```text
results/moss_local_stage1/batch_8/
20260713-101202_stage1_rank0_1783937522
```

| 指标 | FP32 | BF16 | 变化 |
| --- | ---: | ---: | ---: |
| Attention kernel | `fmha_cutlassF_f32...` | `fmha_cutlassF_bf16...` | dtype 已切换 |
| Codec steps | 18 | 16 | trace 工作量不同 |
| Attention calls/step | 92 | 92 | 计算结构未改变 |
| Attention 平均耗时 | `172.296 us` | `64.480 us` | **-62.6%** |
| Self CUDA/step | `93.17 ms` | `49.33 ms` | **-47.1%**，仅辅助参考 |

Attention 调用数保持每步 92 次，说明 BF16 没有跳过 layer；kernel 名称从 FP32 明确变为 BF16，
单层平均耗时下降是最直接、最可比的 profile 证据。

两个 profile 的 codec 输入帧分布不同，而且改后工作区还包含 LFQ LUT 实验，因此不能把 Self CUDA
总时间的全部变化都归因于 BF16。LFQ 实测没有观察到端到端收益，其 kernel 时间也远小于 Transformer；
BF16 的主要结论仍以 attention dtype、单层平均时间和独立端到端 benchmark 为准。

### 1.6 端到端结果

用户实测 C8：

```text
Task             Concurrency   RTF mean  TTFP (ms)   Throughput
voice_clone                8      0.433        563       18.818
```

相对初始结果：

- RTF 从 `0.6~0.7` 降到 `0.433`，下降约 27.8% 到 38.1%。
- 若以区间中点 `0.65` 估算，RTF 下降约 33.4%。
- TTFP 从约 `700 ms` 降到 `563 ms`，下降约 19.6%。

### 1.7 结论和遗留问题

BF16 是当前最明确的端到端收益点。改动小，覆盖了 92 层重复执行的 attention/GEMM，并把 streaming
KV cache 字节数减半。

仍需补充：

- C1 与 C8 的固定语料、固定 seed 重复 benchmark。
- 长音频、多 chunk 的 waveform/SIM/UTMOS 回归，检查 BF16 累积误差。
- 是否进一步把 decoder weight storage 转成 BF16，需要作为独立实验，不能混入本项结果。

---

## 2. 消除 streaming `exec_mask` 重复 D2D

### 2.1 改前现状

Streaming session 使用一个 active-slot `exec_mask` 表示本 codec step 中哪些 slot 有效。改造前每个
`StreamingState` 都持有独立的 `exec_mask` tensor。每个 decode step 在 CUDA Graph replay 之前执行：

```python
self.codec._set_streaming_exec_mask(exec_mask)
entry.static_codes.copy_(codes_step)
entry.graph.replay()
```

旧 `_set_streaming_exec_mask()` 通过 `self.apply(_set)` 遍历整个 codec module tree，并对每一个 live
streaming state 调用一次：

```python
module._streaming_state.set_exec_mask(
    exec_mask.to(module._streaming_state.device)
)
```

每个 state 随后执行 `self.exec_mask[:] = exec_mask`。mask 本身只有 `[stream_slots]`，但它被复制了
数百次，形成大量极小 D2D 和 Python module traversal。这些操作发生在 graph replay 外部，CUDA Graph
无法合并它们。

此外，decode-only session 原来调用整模型 `codec.streaming()`，会给根本不会执行的 encoder 也分配
streaming state 和 KV cache，进一步增加遍历对象和显存占用。

### 2.2 改前 Profiling

Profile：

```text
results/moss_local_stage1/batch_8/
20260713-101202_stage1_rank0_1783937522
```

该 trace 包含 16 个 codec step。关键结果：

| 指标 | 总计 | 每 codec step |
| --- | ---: | ---: |
| `_set_streaming_exec_mask` | `166.098 ms / 16 calls` | `10.381 ms` |
| `Module.apply` Python events | `67,416` | `4,213.5` |
| `aten::copy_` calls | `12,795` | `799.7` |
| `aten::copy_` Self CPU | `26.604 ms` | `1.663 ms` |
| D2D calls | `12,490` | `780.6` |
| D2D CUDA time | `10.033 ms` | `0.627 ms` |
| `aten::where` calls | `6,176` | `386.0` |

`Module.apply` 的事件持续时间包含递归嵌套，不能相加作为 wall time；但 `_set_streaming_exec_mask`
外层事件的每步 10.381 ms 是直接可见的 CPU 空泡。timeline 中 CUDA Graph 前密集的小 D2D 正是
逐 state 传播 mask 造成的。

### 2.3 分析结论

所有 streaming state 在同一个 session、同一个 codec step 中读取的 active-slot mask 完全相同，
不需要各自保存副本。CUDA Graph 要求输入 tensor 地址稳定，因此适合使用一个 session 生命周期内
地址不变的共享 tensor：

```text
state_0.exec_mask --+
state_1.exec_mask --+--> session_shared_exec_mask
state_2.exec_mask --+
...
```

每步只原地更新这个共享 tensor 一次。所有 graph-captured layer 仍从原来的 `state.exec_mask` 属性读取，
但这些属性引用同一块 device memory，因此不需要修改模型数学或 CUDA Graph 内部计算。

### 2.4 代码修改

修改文件：

- `vllm_omni/model_executor/models/moss_tts/audio_tokenizer_v2.py`
- `vllm_omni/model_executor/models/moss_tts/modeling_moss_tts_codec.py`

#### 共享 address-stable mask

`StreamingState` 增加 `bind_exec_mask()`，允许 state 引用 session 共享 tensor，并记录该 tensor 是否由
state 自己拥有。共享 mask 不在每个 state 的 `reset()` 中重复修改。

`MossAudioTokenizerModel._start_streaming()` 在 session 开始时只创建一次：

```python
shared_exec_mask = torch.ones(
    batch_size,
    dtype=torch.bool,
    device=next(self.parameters()).device,
)
```

初始化每个 streaming state 后，将其绑定到这个 tensor：

```python
state = module._init_streaming_state(batch_size)
state.bind_exec_mask(shared_exec_mask)
module._streaming_state = state
```

每步更新由遍历整棵 module tree 改为一次 device copy：

```python
shared_exec_mask.copy_(
    exec_mask.to(device=shared_exec_mask.device, dtype=torch.bool),
    non_blocking=True,
)
```

#### 缓存 streaming module 列表

session 初始化时缓存实际创建了 state 的 `StreamingModule`。后续 stop/reset 直接遍历这个扁平列表，
不再反复调用 `nn.Module.apply()`。

#### Decode-only streaming

新增 `decoder_streaming()` context，只初始化 `self.decoder` 下的 streaming state。Stage 1 codec session
优先使用它，不再为 encoder 分配未使用的 streaming state/KV cache。

兼容性上保留 fallback：如果 codec 没有 `decoder_streaming()` 或 `_reset_streaming_slots()`，session
仍使用旧接口。

### 2.5 改后 Profiling

Profile：

```text
results/moss_local_stage1/batch_8/
20260713-110200_stage1_rank0_1783940520
```

该 trace 包含 15 个 codec step。所有计数按 step 归一化后比较：

| 指标 | 改前 | 改后 | 变化 |
| --- | ---: | ---: | ---: |
| `_set_streaming_exec_mask` | `10.381 ms/step` | `0.017 ms/step` | **-99.84%** |
| `_set_streaming_exec_mask` 总时间 | `166.098 ms / 16` | `0.253 ms / 15` | 空泡基本消失 |
| `Module.apply` events | `67,416` | `0` | 热路径完全移除 |
| D2D calls | `780.6/step` | `104.8/step` | **-86.6%** |
| D2D CUDA time | `0.627 ms/step` | `0.086 ms/step` | **-86.2%** |
| `aten::copy_` calls | `799.7/step` | `122.9/step` | **-84.6%** |
| `aten::copy_` Self CPU | `1.663 ms/step` | `0.283 ms/step` | **-83.0%** |
| `aten::where` calls | `386.0/step` | `101.3/step` | **-73.8%** |
| Self CPU | `101.9 ms/step` | `72.5 ms/step` | **-28.8%**，受输入工作量影响 |

结构正确性的 profile 证据：

- Attention：`1380 = 92 layers x 15 steps`。
- Ring KV scatter：`2760 = 92 x 2 x 15`。
- LayerNorm：`2760 = 92 x 2 x 15`。
- Attention 平均时间：改前 `64.480 us`，改后 `65.005 us`，基本不变。
- BF16 copy：改前和改后均为 `1117/step`。

这些计数说明优化只移除了 graph 外的 mask 传播，没有跳过 decoder layer，也没有改变主要 graph
计算路径。

改前 trace 的 codec 输入帧总数为 174，改后为 119，工作量不同。因此不能使用 `789.209 ms ->
706.604 ms` 的 Self CUDA 总时间作为端到端收益。D2D、`copy_` 和 `_set_streaming_exec_mask` 按 step
归一化后与输入帧数基本无关，才是本项的有效比较指标。

### 2.6 端到端结果

#### C8 结果

Seed-TTS eval：

| 指标 | 结果 |
| --- | ---: |
| Evaluated | 100 |
| Mean WER | `0.0348` |
| Median WER | `0.0000` |
| Request/PCM/ASR failures | `0 / 0 / 0` |

Serving benchmark：

| 指标 | BF16 基线 | Shared `exec_mask` | 变化 |
| --- | ---: | ---: | ---: |
| Concurrency | 8 | 8 | 相同 |
| Mean RTF | `0.433` | `0.371` | **-14.3%** |
| Mean TTFP | `563 ms` | `431.10 ms` | **-23.4%** |
| Audio throughput | `18.818` | `21.780` | **+15.7%** |

本次 C8 benchmark 的其他结果：

| 指标 | Mean | Median | P99 |
| --- | ---: | ---: | ---: |
| E2E latency | `1565.24 ms` | `1574.11 ms` | `2398.98 ms` |
| Audio RTF | `0.37` | `0.36` | `0.52` |
| Audio TTFP | `431.10 ms` | `429.11 ms` | `683.39 ms` |
| Audio duration | `4.33 s` | `4.24 s` | `7.06 s` |
| Streaming underrun | `0.18 s` | `0.15 s` | `0.39 s` |

- 100 个请求全部成功，benchmark duration 为 `19.88 s`，request throughput 为 `5.03 req/s`。
- 共生成 `433.04 s` 音频和 `20,785,920` audio frames。
- Streaming continuity OK rate 为 `24.00%`。

#### C16 结果

Seed-TTS eval：

| 指标 | 结果 |
| --- | ---: |
| Evaluated | 100 |
| Mean WER | `0.0261` |
| Median WER | `0.0000` |
| Request/PCM/ASR failures | `0 / 0 / 0` |

Serving benchmark：

| 指标 | Mean | Median | P99 |
| --- | ---: | ---: | ---: |
| E2E latency | `2938.94 ms` | `2979.89 ms` | `4285.57 ms` |
| Audio RTF | `0.72` | `0.71` | `1.14` |
| Audio TTFP | `937.35 ms` | `998.36 ms` | `1314.86 ms` |
| Audio duration | `4.23 s` | `4.12 s` | `6.72 s` |
| Streaming underrun | `0.41 s` | `0.39 s` | `0.71 s` |

- Summary RTF 为 `0.717`，TTFP 为 `937 ms`，audio throughput 为 `22.317`。
- 100 个请求全部成功，benchmark duration 为 `18.97 s`，request throughput 为 `5.27 req/s`。
- 共生成 `423.28 s` 音频和 `20,317,440` audio frames。
- Streaming continuity OK rate 为 `4.00%`。

#### 收益和并发扩展性

C8 端到端结果与 profile 方向一致：原来每步约 10.4 ms 的 `_set_streaming_exec_mask` 区间降到约
17 us，最终 RTF 下降 14.3%，TTFP 下降 23.4%，audio throughput 提升 15.7%。收益不只来自 raw D2D
CUDA 时间每步减少约 0.541 ms，还来自 Python module traversal 和数百次 copy/launch 提交的消除。

从 C8 增加到 C16 后，audio throughput 仅从 `21.780` 增长到 `22.317`，提升约 2.5%；但 mean RTF
从 `0.371` 增加到 `0.717`，TTFP 从 `431.10 ms` 增加到 `937.35 ms`。这说明当前单卡 pipeline 在约
`22x realtime` 附近已经接近吞吐饱和，继续增加并发主要转化为排队和延迟，而不是有效吞吐。

C8/C16 的 WER 都较低且没有失败请求，说明服务功能正常。但两组 WER 不应被解释为 C16 质量更好，
也不能替代相同音频的 waveform/SIM/UTMOS 回归。Streaming continuity OK rate 从 C8 的 24% 降到 C16
的 4%，同时 mean underrun 从 0.18 s 增至 0.41 s，是下一阶段需要单独分析的高并发 QoS 问题。

### 2.7 结论和遗留问题

共享 mask 方案与 CUDA Graph 的 address-stable 要求匹配，profile 已明确验证优化生效。它解决的是
session 状态组织和 graph 外调度问题，不是 decoder kernel 算力问题。

改后仍然存在精确的 `190 memcpy32_post/codec step`：

```text
92 个 MHA x 2 次状态更新 + 6 个 Transformer x 1 次 offset 更新 = 190
```

它们来自每层独立更新 attention offset、RingKVCache `end_offset` 和 Transformer offset。下一步可考虑
把 position、write index 和 offset 提升为每个 decoder stage 共享的 metadata，在 stage 入口计算一次、
出口统一推进一次。这比共享 `exec_mask` 更侵入，需要单独设计和采集 profile，不能记入本项收益。

仍需补充：

- shared `exec_mask` 后的固定语料 C1 benchmark。
- 多请求 slot acquire/release/reset 的长时间稳定性测试。
- waveform 一致性或 SIM/UTMOS 回归。
- Streaming continuity/underrun 的指标定义、改前基线和 timeline 对照。

---

## 3. 消除 scalar H2D 并融合 streaming input staging

### 3.1 改前现状

共享 `exec_mask` 消除了逐 layer 传播，但 session 在每个 codec step 仍通过 Python tensor 索引构造
`codes_lengths` 和 active-slot mask：

```python
codes_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
exec_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
for slot, codes in slot_codes.items():
    codes_lengths[slot] = int(codes.shape[1])
    exec_mask[slot] = True
```

这些右值是 Python scalar。对 CUDA tensor 的逐元素赋值会产生很多极小的 pageable H2D，并在 CPU
提交路径触发 `cudaStreamSynchronize`。CUDA Graph replay 前还需要单独执行：

```python
self.codec._set_streaming_exec_mask(exec_mask)
entry.static_codes.copy_(codes_step)
entry.graph.replay()
```

因此 shared mask 虽然已经把数百次逐 state D2D 降为一次，graph 外仍存在 scalar H2D、一次 mask
更新和一次 `codes` staging，短 step 下这些固定调度开销不可忽略。

### 3.2 改前 Profiling

改前 profile：

```text
results/moss_local_stage1/batch_8/
20260713-110200_stage1_rank0_1783940520
```

该 trace 中每个被分析的 step 都能看到 graph launch 前的 pageable H2D 和 CPU
`cudaStreamSynchronize`。汇总结果：

| 指标 | 改前 |
| --- | ---: |
| 1-byte pageable H2D | `43` |
| 8-byte pageable H2D | `43` |
| pageable H2D 合计 | `86` |
| scalar-associated sync | `86 calls / 155.135 ms` |

1 byte 对应 bool active mask，8 byte 对应 `torch.long` code length。数据量本身很小，主要成本来自
逐 scalar dispatch、pageable H2D 和同步，而不是 PCIe 带宽。

### 3.3 分析结论

当前 streaming batch 要求同一个 codec step 内所有 active slot 的 `T` 相同，因此没有必要逐 slot
写入 `codes_lengths`：eager fallback 可以直接构造整向量 `torch.full(..., step_t)`，inactive slot 的
输出不会被消费，shared `exec_mask` 会阻止其 streaming state 前进。

active slots 也不需要先物化为 CPU/CUDA tensor。最大 256 个 slot 可以编码为一个 Python 整数 bitset，
再拆成 8 个动态 `int32` kernel launch 参数。它们通过 CUDA kernel argument 传递，不产生独立的
`cudaMemcpyAsync` H2D。参数不是 `tl.constexpr`，并显式设置 `do_not_specialize`，所以不同
active-slot 组合不会触发 Triton 重编译。

CUDA Graph 路径原本还需要把动态 `codes_step` 搬到 graph-owned static buffer。mask 更新和 codes
staging 都是简单的逐元素 store，可以融合为一次 Triton launch。对 C32/C64/C256，mask 工作只由
`program_id == 0` 执行，其余 program 只搬运 codes，不会随 copy grid 重复计算整个 mask。

### 3.4 代码修改

完整的方案演进、初版实现问题、vLLM Model Runner V2 接入方式、逐行代码语义和正确性分析见：

- [MOSS-TTS Streaming Input Staging Triton 优化设计与实现](docs/design/moss_tts_triton_streaming_input_staging.md)

新增：

```text
vllm_omni/model_executor/models/moss_tts/streaming_input.py
```

接入方式遵循 vLLM Model Runner V2 的 input preparation 模式：

- 通过 `from vllm.triton_utils import tl, triton` 使用 vLLM 的 Triton 兼容层。
- 私有 `@triton.jit` kernel 和薄 Python launcher 放在同一个 input preparation 模块中。
- 该 kernel 在 model forward/`torch.compile` 图之外执行，因此不注册 custom op。
- session 和 CUDA Graph wrapper 持有 address-stable GPU buffer，launcher 只负责原地更新。

对外接口只暴露一个整数 mask：

```python
active_slot_mask = encode_slot_mask(slot_codes, batch_size)
prepare_streaming_inputs(
    codes_step,
    entry.static_codes,
    shared_exec_mask,
    active_slot_mask,
)
entry.graph.replay()
```

`prepare_streaming_inputs()` 在一次 kernel launch 中完成：

1. `codes_step -> entry.static_codes`。
2. bitset -> address-stable shared `exec_mask`。

reset/eager 路径复用同一个 kernel 的 `copy_codes=False` specialization，只更新 shared mask。session
直接持有 codec streaming context 创建的 shared mask，不再调用 `_set_streaming_exec_mask()` 将 mask
复制给自己。

Triton 3.x 不允许 kernel 直接读取普通 Python 全局常量，因此 `mask_word_bits=32` 作为显式
`tl.constexpr` launch 参数传入；active mask 的 8 个 word 是动态 scalar 参数，并通过
`do_not_specialize` 避免按 active pattern 产生额外 specialization。

### 3.5 改后 Profiling

scalar H2D 消除后的中间验证 profile：

```text
results/moss_local_stage1/batch_8/
20260713-125811_stage1_rank0_1783947491
```

| 指标 | 改前 `110200` | 改后 `125811` | 变化 |
| --- | ---: | ---: | ---: |
| Pageable H2D | `43 x 1B + 43 x 8B` | `0` | **完全消除** |
| Scalar-associated sync | `86 / 155.135 ms` | `0` | **完全消除** |
| 全部 sync | `116 / 155.222 ms` | `32 / 0.107 ms` | 调用数 -72.4%，时间 -99.9% |
| `execute_context_7(84)_generation_0(0)` | `78.703 ms` | `44.241 ms` | **-43.8%** |

`125811` 验证了去除 scalar tensor assignment 的收益方向，但它早于最终的通用 Triton 融合实现，
不能用来证明最终 `prepare_streaming_inputs` kernel 的 launch 数和 CUDA 时间。最终实现仍需补一份相同
口径的 Nsight trace，重点确认 graph replay 前只剩一个 Triton staging kernel，并检查它相对原
`cudaMemcpyAsync` codes copy 的成本。

### 3.6 端到端结果

用户实测最终 Triton 版本 C16：

```text
Task             Concurrency   RTF mean  TTFP (ms)   Throughput
voice_clone               16      0.642        806       24.824
```

Seed-TTS eval：

| 指标 | 结果 |
| --- | ---: |
| Evaluated | `100` |
| Mean WER | `0.0248` |
| Median WER | `0.0000` |
| Request failed | `0` |
| No PCM captured | `0` |
| ASR / WER failed | `0` |

与相同 concurrency 的 shared-mask C16 基线对比：

| 指标 | Shared mask C16 | Triton staging C16 | 变化 |
| --- | ---: | ---: | ---: |
| Mean RTF | `0.717` | `0.642` | **-10.5%** |
| Mean TTFP | `937 ms` | `806 ms` | **-14.0%** |
| Audio throughput | `22.317` | `24.824` | **+11.2%** |
| Mean WER | `0.0261` | `0.0248` | 均处于低位，不解释为质量提升 |
| Successful eval samples | `100` | `100` | 无失败 |

吞吐从约 `22.3x realtime` 提升到 `24.8x realtime`，说明上一阶段观察到的约 `22x` 平台并不是
codec 纯计算的硬上限，graph 外 input staging 和同步仍在限制高并发吞吐。WER 没有显示明显回归，
但仍不能替代固定输入的 waveform/SIM/UTMOS 对照。

### 3.7 结论和遗留问题

最终 C16 benchmark 已确认端到端收益：RTF 和 TTFP 同时下降，吞吐提升 11.2%，100 个样本全部成功。
该方案不依赖按 batch size 预生成 mask table，C32 会使用同一个 kernel 和 bitset 编码路径；当前显式
上限为 256 slots。

仍需补充：

- 最终 Triton 版本的 Nsight Systems trace，并与 `110200`、`125811` 按 codec step 对齐。
- C8 和 C32 的相同语料 benchmark，确认收益随并发的变化。
- 检查 `prepare_streaming_inputs` 的 CUDA 时间；较大 `T` 下原生 D2D memcpy 可能有更高带宽。
- 固定输入 waveform、SIM、UTMOS 以及 streaming continuity/underrun 回归。

---

## 4. Codec stream state pool 与动态执行 batch

### 4.1 原问题：state capacity 被错误地当成执行 batch size

旧 streaming codec 使用固定数量的 slots 保存 decoder KV、offset 等跨 chunk 状态。配置
`codec_stream_slots=16/32` 后，每一步 codec forward 都按全部 slots 组织张量和计算，即使实际只有一个
请求有新 chunk，也会执行接近 `B=16/32` 的 batch。

这里混淆了两个独立概念：

- **State capacity**：最多允许多少条 live stream 同时保留跨 chunk decoder state。
- **Execution batch size**：当前 step 真正有 codec codes 需要 decode 的 stream 数量。

capacity 决定常驻状态显存和最大 live stream 数，不应该决定每一步的计算量。固定 slots batch 带来的
直接问题是：

1. `max_num_seqs` 开大后，C1 也执行完整 slot batch，低并发 RTF 显著恶化。
2. 无效 rows 仍经过 decoder，增加 GEMM、attention、D2D/D2H 和 graph replay 工作量。
3. slot 数同时承担容量、调度和 CUDA Graph shape 三种职责，正确性和性能难以分别约束。

动态 batch 的目标是保持 `max_num_seqs` 个持久状态槽，但每一步只计算有新 chunk 的请求，并继续让常用
shape 使用 CUDA Graph。

### 4.2 核心模型：持久 state slot 与紧凑执行 row 解耦

每条 live stream 首次进入 codec 时租用一个稳定的物理 state slot：

```text
request_id ──> persistent state slot
                  ├── attention KV
                  ├── ring/cache offset
                  └── decoder streaming metadata
```

某个 codec step 只收集当前有 work 的请求，组成紧凑执行 batch：

```text
compact row 0 ──state_slot_ids──> persistent slot 17
compact row 1 ──state_slot_ids──> persistent slot 3
compact row 2 ──state_slot_ids──> persistent slot 29
```

decoder kernel 通过 `state_slot_ids[B]` 访问对应的持久状态，因此执行 row 不需要与物理 slot 编号相同，
也不需要包含所有 live streams。一个正在等待下一段 Stage-0 codes 的请求继续持有 state slot，但不进入
当前执行 batch，不产生 codec 计算。

### 4.3 类与接口职责

#### `MossTTSCodecDecoder`

负责请求级协议和生命周期：

- 解析 runner 传入的 flat `input_ids`、`seq_token_counts` 和 additional information。
- 将每个请求的 flat codes 恢复成 `[NQ, T]`。
- 维护 `request_id -> state_slot` 映射。
- 按 `T` 对同一步 work 分组，因为一次 `session.step()` 要求组内 frame size 一致。
- terminal request decode 完最后一个 chunk 后释放 slot；空 terminal control packet 也必须释放。
- 非 streaming 请求继续走原 offline batch decode 路径。

#### `_MossCodecStreamSession`

负责 codec state pool 和一次紧凑 decode：

```python
step(
    slot_codes: dict[int, Tensor[NQ, T]],
    terminal_slots: set[int] | None,
) -> dict[int, Tensor[C, samples]]
```

约定：

- `slot_codes` 的 key 必须是已 lease 的物理 slot。
- 同一次调用的所有 codes 必须具有相同 `T`。
- `terminal_slots` 必须是 `slot_codes` 的子集。
- 输入按紧凑 row stack 成 `[NQ, B_actual, T]`，另传 `state_slot_ids[B_actual]`。
- 输出只保留实际 rows，并以物理 slot 为 key 返回。
- `release(slot)` 在归还 free list 前 reset 该 slot，防止下一个请求读到旧 KV/offset。

#### `CUDAGraphStreamingDecoderWrapper`

负责紧凑 batch 的 CUDA Graph capture、选择和 replay：

- graph key 是 `(B_bucket, exact_T)`。
- runtime 从 capture sizes 中选择首个 `B_bucket >= B_actual` 的 graph。
- 前 `B_actual` rows 指向真实 state slots；padding rows 指向专门的 scratch slots。
- `valid_rows` 保证 padding rows 不作为有效输出。
- replay 后只返回 `[:B_actual]`，因此 D2H 也只复制实际输出 rows。
- 找不到可用 graph 时回退 eager，eager 始终按精确 `B_actual` 计算。

scratch slots 不可 lease 给请求。它们将 graph padding 对 decoder state 的写入与真实 live state 隔离，
避免 padded row 污染另一个请求。

#### `OmniChunkTransferAdapter` 与 `OmniGenerationScheduler`

负责 codec capacity admission，而不是让 model forward 自己排队：

- `codec_streaming=True` 时，adapter 将 active window 设置为 `max_num_seqs`。
- `_active_streams` 表示已经获得 codec stream admission 的请求，包括暂时等待下一 chunk 的 idle live stream。
- 超过窗口的请求留在 waiting/hold queue，不进入 Stage-1 model scheduler。
- terminal chunk 被实际调度后才从 active window 移除，随后 FIFO 提升下一个等待请求。
- codec 内的 `session.acquire()` 不等待；如果 admission invariant 被破坏，它直接抛异常，拒绝丢弃 audio
  codes 或静默复用仍在使用的 state。

### 4.4 输入和输出协议

Stage 0 到 Stage 1 的每个请求携带：

| 字段 | 语义 |
| --- | --- |
| `codes.audio` | codebook-major flat codes，长度应为 `NQ * T` |
| `meta.req_id` | 跨 chunk 稳定的 stream owner ID |
| `meta.codec_streaming` | 是否使用持久 codec stream state |
| `meta.code_flat_numel` | 真实 codec token 数；用于 runner 构造 request slice |
| `meta.stream_finished` / `meta.finished` | 当前 packet 是否结束该 stream |
| `seq_token_counts` | runner 按当前 batch request 顺序提供的真实 token 数 |

`input_ids` 是 batch 内所有请求拼接后的 flat tensor。`seq_token_counts` 必须与
`runtime_additional_information` 和 runner 的 `input_batch.req_ids` 使用同一顺序。codec 只按这个长度切分，
不能从 padding 后的总 tensor shape 猜测 request 边界。

### 4.5 配置复用和 CUDA Graph shape

本次设计不再引入独立的 `codec_state_capacity`、`codec_max_decode_batch_size` 或
`codec_decode_batch_buckets`：

- state capacity 直接复用 Stage-1 `scheduler_config.max_num_seqs`。
- codec 的 B buckets 复用 `compilation_config.cudagraph_capture_sizes`，并裁剪到
  `B <= max_num_seqs`。
- eager 模式没有 capture sizes，直接按精确 `B_actual` forward。
- T shapes 来自实际 streaming chunk 协议，目前 Local v1.5 常见为初始 `T=1` 和常规 `T=15`。

因此 `cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32]` 在 codec wrapper 中表示 batch bucket，而不是
sequence length。Stage-1 外层 vLLM graph 与 codec 内层 streaming wrapper 的 shape 语义不同，不能因为
字段名称相同就把两层 graph 混为一谈。

尾部 `T=1..14` 只在 terminal step 出现时，可以选择更大的 T graph（通常 `T=15`）进行 padding：

- decoder attention 是 causal 的，真实前缀不依赖后补的 pad frames；
- `audio_lengths` 将输出裁剪回真实 T 对应的 samples；
- 该 step 的所有真实 slots 都是 terminal，replay 后立即 reset，因此 pad frame 推进的 state 不会再被读取。

非 terminal 的未知 T 不允许这样 padding，否则会永久推进 KV/offset；该情况必须 eager 或使用 exact-T
graph。

### 4.6 性能结果

用户实测动态 batch 达到预期，并优于固定-slot 实现：

- C16 mean RTF 从约 `0.6` 降到约 `0.4`。
- `max_num_seqs=32` 时，C1 mean RTF 约 `0.15`，说明 capacity 开大不再强制 C1 计算 32 rows。
- `max_num_seqs=32`、客户端并发高于 32 时，设计目标是最多 32 条 active codec streams，其余请求排队，
  而不是扩大每步 forward 到全部已提交请求。

这项收益主要来自减少无效 decoder rows，而不是单个 kernel 变快。其扩展性应分别观察：

- C1/C8：是否按小 B bucket replay，固定开销是否合理。
- C16/C32：GPU 吞吐是否随 B 提升。
- concurrency > capacity：等待时间会增加，但不得出现 audio 丢失、placeholder decode 或 slot corruption。

### 4.7 高并发正确性 bugfix

动态 batch 初版在 C1/C16 性能上符合预期，但 `max_num_seqs=32`、客户端并发 64、连续多轮请求时暴露了
两个独立的生命周期/admission bug。最终的 `capacity exhausted` 是保护性报错，真正根因发生在它之前。

#### Bug 1：空 terminal packet 被零长度分支提前跳过

当最后一批 pending frames 为空时，Stage 0 会发送 control-only terminal packet：

```text
codes.audio = [0]             # 让 Stage 1 得到一次调度机会
meta.code_flat_numel = 0      # 表明它不是有效 codec code
meta.stream_finished = true
```

runner 据此将该请求的 `seq_token_counts` 设为 0。原 codec forward 的顺序是：

```python
seg = ids_flat[start:end]
if seg.numel() == 0:
    continue
# 后面才检查 code_flat_numel == 0 and finished
```

所以 control-only terminal 永远无法执行 `_finish_empty_streaming_requests()`，对应 slot 会泄漏。尾部仍有
实际 frames 的请求能正常释放，因此现象不是“每个请求都泄漏”，而是运行一段时间后逐渐累计，最终下一轮
请求才撞上满池。

修复后先解析 metadata 和处理空 terminal，再跳过普通空 segment：

```python
if streaming_enabled and finished and int(code_flat_numel) == 0:
    finish_stream_request(...)
    continue
if seg.numel() == 0:
    continue
```

#### Bug 2：GenerationScheduler fast path 绕过 active window

`server.txt` 在 capacity exception 前给出了决定性证据：

```text
MossTTS codec input length 121 not divisible by n_vq 12; skipping.
...
MossTTS codec input length 177 not divisible by n_vq 12; skipping.
```

合法 Local v1.5 codec chunk 长度一定是 `NQ * T`，这里 `NQ=12`，所以这些 121–177 的输入不可能是
codec codes。它们实际是 async-chunk Stage-1 预提交时创建的 placeholder prompt 长度。

原 active-window 逻辑只处理已经 active 的请求：前 32 个请求会被移动到 chunk polling 队列；超过窗口的
请求仍留在 scheduler 普通 `waiting` 队列。`OmniGenerationScheduler` 随后的 generation fast path 没有再次
检查 active 状态，因而把非 active 请求的 placeholder 当作普通新 prompt 调度进 codec。

修复包括：

1. `OmniChunkTransferAdapter.is_active_stream(request_id)` 显式暴露 admission 状态。
2. GenerationScheduler 从 `waiting` 取请求时，非 active stream 放入本 step 的 skipped queue。
3. step 结束时按原队列顺序恢复，等 active slot 释放后再由 adapter FIFO promote。

修复后的关键 invariant 是：

```text
进入 Stage-1 model scheduler 的 streaming request
    => 已属于 adapter._active_streams
    => 最多 max_num_seqs 条 live stream 可持有 codec state
```

`session.acquire()` 的 capacity exception 继续保留。如果今后再次触发，它代表 scheduler admission、terminal
cleanup 或 abort cleanup 又与 model state pool 失配，不能改成覆盖旧 slot、丢弃 codes 或临时扩容来掩盖。

#### 验证状态

代码级检查已经通过：

- `git diff --check`
- Ruff
- `py_compile`

按当前开发约定没有新增或运行单元测试。两个 bugfix 仍需用以下场景做端到端压力复测：

1. `max_num_seqs=32`、concurrency 64。
2. 第一轮至少 200 条请求，完成后在同一 server 进程立即开始第二轮。
3. 检查没有 `input length ... not divisible by n_vq`、`state capacity exhausted` 和 zero PCM。
4. 检查第二轮 C1/C16 性能没有因 stale slots 或 waiting queue 顺序发生退化。

### 4.8 结论和后续观测

动态 codec batch 的核心不是简单缩小一个 tensor，而是把 capacity、admission、state ownership、执行 shape
和 CUDA Graph padding 分成独立职责。当前实现已经验证了低并发和 C16 性能收益；高并发问题也从
“codec slot 不够”定位为两个明确的协议/调度 bug，并在对应责任层修复。

后续应补充：

- 动态版本的 C1/C8/C16/C32 固定语料 benchmark 和 Nsight trace。
- concurrency 64、128 下的排队延迟、active window 数和 slot lease 数监控。
- terminal、client abort、engine abort 三条释放路径的计数指标。
- 在 capacity exception 中同时打印 active request IDs、leased slots 和当前 batch IDs，缩短下一次 invariant
  失配的定位时间。

---

## 后续优化记录模板

新增优化时，先在“优化汇总”和“端到端性能轨迹”追加一行，再复制以下结构：

```markdown
## N. 优化名称

### N.1 改前现状
说明代码路径、张量 shape、调度方式和瓶颈。

### N.2 改前 Profiling
记录 profile 目录、调用数、CPU/CUDA 时间，并按 step/layer 归一化。

### N.3 分析结论
说明为什么判断该瓶颈可优化，以及预期收益边界。

### N.4 代码修改
记录 commit、文件、关键接口和语义变化。

### N.5 改后 Profiling
使用同口径指标前后对照，注明工作量是否一致。

### N.6 端到端结果
记录 concurrency、RTF、TTFP、throughput、WER、SIM、UTMOS。

### N.7 结论和遗留问题
区分已验证结论、待补测试和下一步候选项。
```
