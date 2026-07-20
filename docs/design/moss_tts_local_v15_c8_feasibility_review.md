# MOSS-TTS Local v1.5 C8 性能优化可行性调研

## 1. 范围与结论

本文基于以下两条 PyTorch profiler trace 和当前代码做静态可行性分析：

- `results/moss_local_stage0/batch_8/20260709-180508_stage0_rank0_1783620308`
- `results/moss_local_stage1/batch_8/20260709-180509_stage1_rank0_1783620309`

当前观测基线是：

| 并发 | RTF | TTFP |
|---|---:|---:|
| 1 | 约 0.21 | 约 92 ms |
| 8 | 约 0.63 | 约 693 ms |

本次只做静态阅读，没有启动模型、运行 benchmark、修改实现或验证音质。

### 总体判断

| 优化项 | 可行性 | 预期价值 | 正确性风险 | 建议优先级 |
|---|---|---|---|---|
| Stage0 / Stage1 分卡 | 很高 | 很高，且是最干净的瓶颈隔离实验 | 低 | P0 |
| Codec BF16 compute | 高，但不能直接 cast 权重 | 很高 | 中 | P0 |
| Stage1 跨请求 coalescing | 中高 | 很高，直接解决 B1+B7 重复 codec pass | 中 | P0 |
| Stage1 尾包 waterfall 分段 | 高 | 很高，当前尾包约 740 ms | 中低 | P0 |
| 共享 `exec_mask` | 很高 | 中高，每个 codec step 可去掉数百次小 copy | 低 | P0 |
| Decoder-only streaming state | 很高 | 中，状态/遍历量约减半 | 低 | P0 |
| 合并 finished-slot reset | 很高 | 中高，当前 8 次 finish/reset 约 300 ms | 低 | P0 |
| Stage0 local static KV cache | 很高 | 高，local token-row 计算约降 6.5 倍 | 中低 | P0/P1 |
| Local top-k 后只在 k 内做 top-p | 高 | 中，trace 中 radix sort 明显 | 低到中 | P1 |
| 绕过 151936 维 forced-token logits | 中 | 中 | 中 | P1 |
| Stage1 只 D2H active audio rows | 高 | 中，尤其当前 B1 codec pass | 低 | P1 |
| Stable per-request last-hidden pool | 中 | 低到中 | 中高 | P2 |
| 动态 active-batch codec / paged KV | 理论可行，工程量大 | 长期很高 | 高 | P2/P3 |

当前最可信的主线是：

1. 先用分卡实验确认同卡争用的上限。
2. 正确接通 codec BF16 compute，而不是简单把权重改成 BF16。
3. 修 Stage1 的 B1+B7 cohort 和尾包 exact-T 分组。
4. 清理 graph 外 `exec_mask`、reset 和全 batch D2H。
5. Stage0 恢复 local static KV cache，再处理 sampling。

---

## 2. Trace 中可以确认的事实

### 2.1 两阶段同卡争用

`vllm_omni/deploy/moss_tts_local.yaml` 中 Stage0 和 Stage1 都配置为 `devices: "0"`。两个 stage 是独立进程和 CUDA context，而不是一个统一调度的 CUDA stream。

两条 trace 对齐后，GPU 活跃区间重叠约 **702 ms**。Stage1 相同的 `B1, T=15` codec 调用表现为：

- 与 Stage0 重叠时：约 179 / 197 / 199 ms。
- Stage0 结束后：约 104 / 104 / 104 ms。

这说明同一个 codec shape 在同卡竞争期接近慢 **1.8-1.9 倍**。同卡争用是实测事实，不只是配置推测。

### 2.2 Codec 是 FP32 计算

Stage1 profiler 的主要 kernel 包括：

- `sm80_xmma_gemm_f32...`：约 576 ms。
- `fmha_cutlassF_f32...`：约 285 ms。

加载代码在 `modeling_moss_tts_codec.py:770-772` 明确执行：

```python
codec.to(device=device, dtype=torch.float32)
```

HF codec config 虽然包含 `compute_dtype: "bf16"` 和 `dtype: "float32"`，但当前 vendored model 没有消费 `compute_dtype`。因此 profiler 中的 FP32 kernel 与代码一致。

### 2.3 Stage1 没有形成 B8 codec cohort

主要 Stage1 调用是：

| 形状 | 次数 | CUDA 总时间 | 平均时间 |
|---|---:|---:|---:|
| B1, T1 | 1 | 85 ms | 85 ms |
| B7, T1 | 1 | 103 ms | 103 ms |
| B1, T15 | 6 | 879 ms | 147 ms |
| B7, T15 | 2 | 441 ms | 220 ms |
| B7 final tails | 1 outer forward | 740 ms | - |

首包和稳态都形成了 `B1 + B7`，没有形成 B8。一个请求领先一个 Stage0 generation step 后，由于一次 codec pass 本身约 100-200 ms，其他请求会在它运行期间到达，之后 cohort 继续保持错相。

### 2.4 Stage1 session 的物理 batch 固定为 8

`_MossCodecStreamSession` 在 `modeling_moss_tts_codec.py:50-68` 把 batch 固定为 `codec_stream_slots`。每次 `step()` 都构造：

```text
codes_step: [n_vq, 8, T]
codes_lengths: [8]
exec_mask: [8]
```

所以 B1 不是一个真正的 batch-1 codec forward，只是固定 B8 session 中有一个 active slot。`exec_mask` 保证 inactive slot 不推进流式状态，但没有把所有 dense transformer/MLP 计算压缩成 B1。

### 2.5 Trace 只代表一个 8-request wave

Stage0 中只出现一轮从 B8 逐步缩到 B1/B2 的请求尾部：

- B8：34 次，平均约 13.26 ms。
- B2：18 次，平均约 14.42 ms。
- B1：16 次，平均约 12.83 ms。

它能证明小 batch graph 与 B8 几乎同价，也能证明一个封闭 8 请求波次的尾部利用率很差。但它不能单独证明 100 请求、持续补入 workload 的稳态也有同样比例的 B1/B2。后续 profile 应把“固定 8 请求 wave”和“100 请求、C8 持续补入”分开分析。

---

## 3. Stage0 / Stage1 分卡

### 可行性：很高

Stage runtime 已经支持每 stage 独立设置可见设备：

- `vllm_omni/entrypoints/stage_utils.py:14-88`
- `vllm_omni/engine/stage_engine_startup.py:763-887`

Stage 间使用 SharedMemoryConnector，Stage0 的 audio codes 已经转成 CPU payload 后写入 POSIX SHM，不依赖同卡 CUDA tensor 地址。因此 Stage1 放到另一张 GPU 不需要修改模型或 connector 数据契约。

### 配置注意事项

当前启动脚本使用：

```bash
CUDA_VISIBLE_DEVICES=7 vllm serve ...
```

如果要使用两张卡，建议显式暴露两张物理卡，再用 YAML 中的逻辑编号：

```bash
CUDA_VISIBLE_DEVICES=7,6 vllm serve ...
```

```yaml
stage0 devices: "0"  # physical 7
stage1 devices: "1"  # physical 6
```

只暴露一张卡然后把 Stage1 写成 `devices: "1"` 会触发物理 ID passthrough 逻辑，配置含义不够直观，不建议用于基准对照。

### 预期

分卡不一定是最终部署要求，但它是当前最重要的隔离实验。根据 trace，相同 B1/T15 codec pass 可从竞争期约 180-200 ms 回到约 104 ms，同时 Stage0 也不再被 codec graph 干扰。

### 单卡替代方案的边界

两个 stage 是独立进程/context，普通 PyTorch CUDA stream priority 无法形成可靠的跨进程 Stage0 优先级。单卡下减小 `codec_chunk_frames` 会缩短单次占用，但 T1 已经约 85 ms、T15 约 104 ms，说明 codec 有很大的固定/上下文成本；简单把 T15 改成 T3/T5 很可能显著恶化总 RTF。

单卡更应该先做 cohort coalescing 和 BF16，而不是先缩小 steady chunk。

---

## 4. Codec BF16 Compute

### 可行性：高，但不能只改一行 `.to(bfloat16)`

HF config 的意图明显是：

```text
codec weight dtype = FP32
codec compute dtype = BF16
```

当前 vendored config 已保留 `compute_dtype` 字段，但模型实现完全没有使用它。

### 为什么直接把 codec 权重改成 BF16 不安全

`audio_tokenizer_v2.py` 的 quantizer decode 路径显式构造 FP32 tensor：

```python
emb = torch.zeros(..., dtype=torch.float32)
emb += quantizer.decode_code(...).float()
emb = self.output_proj(emb)
```

Local v1.5 的 quantizer `rvq_dim=512, output_dim=768`，`output_proj` 不是 Identity。如果把 `output_proj` 权重直接变成 BF16，而输入仍被强制转为 FP32，eager 路径可能出现 dtype mismatch，或者引入隐式转换和额外 kernel。

### 推荐实现

保留 FP32 权重，增加真正的 compute-dtype plumbing：

1. 从 codec config 解析 `compute_dtype`，默认遵循 checkpoint 的 `bf16`。
2. codec 权重继续以 FP32 加载，避免改变 checkpoint 存储和 quantizer 数值路径。
3. `_decode_frame` 的 decoder compute 在 CUDA autocast BF16 上下文中执行。
4. `RingKVCache` 不再从 `in_proj.weight.dtype` 推断 dtype，而使用 session 的 compute dtype；否则 FP32 权重会继续创建 FP32 KV cache。
5. graph warmup 和 capture 必须在相同 autocast 上下文中完成。
6. eager fallback 也必须使用相同 compute context，防止 graph hit/miss 产生两套数值路径。
7. API 输出仍在 `_restore_channels_from_codec()` 或 session 输出处转 FP32。

### CUDA Graph 兼容性

可兼容，但必须销毁并重新捕获所有 streaming graphs。graph capture 时确定的 GEMM/SDPA kernel 和 KV cache storage dtype 都会变化，不能复用现有 FP32 graph。

### 额外收益

除了 GEMM/attention，BF16 KV cache 还会降低 codec streaming state 的显存和带宽。这个 codec 有 92 个 transformer layer，且每层持有 causal ring KV，状态 dtype 的收益不会很小。

### 验证标准

- FP32 与 BF16 对同一 code sequence 的 waveform 长度完全一致。
- waveform 无 NaN/Inf，chunk boundary 无新增突变。
- WER/CER、speaker similarity、UTMOS 不出现不可接受回退。
- graph hit 和 eager fallback 输出在允许误差内一致。
- profiler 的主 GEMM/attention kernel 不再是 F32 path。

不建议第一版同时引入 FP16。BF16 的动态范围更适合 codec 和长 streaming state。

---

## 5. Stage1 Coalescing 与尾包重组

### 5.1 跨 scheduler step 的 B1+B7 coalescing

### 可行性：中高

当前 chunk 到达后，`OmniChunkTransferAdapter` 立即把 request 放入 `_finished_load_reqs`，下一次 scheduler tick 就恢复为 ready/running：

- `chunk_transfer_adapter.py:195-290`
- `chunk_transfer_adapter.py:443-584`

因此 B1 先到时会立即启动一次固定 B8 codec graph；其运行期间 B7 到达，随后再启动第二次固定 B8 graph。

### 推荐位置

coalescing 应放在 Stage1 chunk-ready admission，而不是放在 codec model 内。模型只看得到本次已经被 scheduler 选中的 items，无法知道几毫秒后是否还有 sibling chunk 到达。

建议增加 MOSS codec 专用或通用可配置项，例如：

```yaml
codec_coalesce_wait_ms: 10
codec_coalesce_min_ready: 2
```

语义建议：

- 第一个有效音频包允许立即执行，或者只等待一个 scheduler tick。
- steady chunk 最多等待 `coalesce_wait_ms`。
- ready 数达到当前 active-stream 数或 slot 上限时立即释放。
- oldest ready chunk 到 deadline 必须释放，不能无限等慢请求。
- finish/control-only chunk 不应被等待窗口阻塞。

### 收益判断

当前 B1/T15 和 B7/T15 是两次约 100 ms 以上的固定 session replay。只要等待窗口小于被省掉的重复 replay，端到端平均延迟和 GPU 吞吐都会改善。trace 的 cohort 很可能只错开一个 Stage0 generation step，10-20 ms 是合理的第一组扫描范围。

### 风险

- 低负载单请求会多出等待时间，因此必须有 `min_ready`/低负载 fast path。
- 不同租户/无关请求不能被强行绑定生命周期，只能共享一次 compute admission。
- abort 和 timeout 必须从等待集合清理。

### 5.2 同一 forward 内的 final-tail waterfall

### 可行性：高

当前 `_decode_streaming_batch()` 按 exact T 分组：

```python
grouped.setdefault(int(codes_nq_t.shape[1]), []).append(...)
for group in grouped.values():
    session.step(group)
```

如果 7 个尾包长度都不同，固定 B8 session 的计算量近似与 `sum(unique_tail_lengths)` 成正比。这正是 trace 中 `B7 final tails` outer forward 达到约 740 ms 的原因；18 个 `session.step` 比 12 个 outer codec forward 多出的 6 次也与此吻合。

可以改为共同前缀 waterfall：

1. 对所有 ready slot 维护 remaining codes。
2. 取 active slot 中最小 remaining T 作为本段 `delta`。
3. 所有 active slot 一起 decode `delta` 帧。
4. 完成的 slot 退出，剩余 slot 继续下一段。
5. 每个 slot 按段拼接自己的 waveform。

若尾长为 `[2, 4, 7]`：

- 当前 exact-T：固定 B8 分别计算 T2、T4、T7，总计算长度 13。
- waterfall：计算 T2（3 active）、T2（2 active）、T3（1 active），总计算长度 7。

调用次数没有一定减少，但固定 B8 dense compute 的总 T 从 `sum(unique T)` 降到 `max(T)`。

### 为什么状态语义允许这样做

当前 decoder 是 causal streaming：

- active slot 的 KV/offset 按 T 推进。
- inactive slot 通过 `exec_mask` 不推进。
- decoder 中的 patch upsample 是无状态 reshape。
- 每个 slot 的 code 顺序保持不变。

因此把一个连续 tail 切成多个连续 streaming segment，理论上等价于一次性处理完整 tail。所有 delta 都在 1..15 capture size 内，仍可命中已有 graph。

### 必须验证

- 分段与一次 decode 的 waveform 长度一致。
- 分段拼接处不重复、不缺 sample。
- slot 在中途 finish 后不再推进任何 state。
- BF16 下允许小数值误差，但不能出现可听边界跳变。

### 5.3 不建议短期做 variable-B graph

固定 slot session 中，每个 request 的 KV/offset 绑定到稳定 slot。捕获 B1/B2 graph 并不能直接处理任意 slot；需要先 gather 该 slot 的 92 层状态，decode 后再 scatter 回去，成本和复杂度都很高。

真正的动态 active batch 需要 paged/block-table KV 或同等的间接寻址设计。它适合作为长期优化，但不应阻塞 coalescing、waterfall 和 BF16。

---

## 6. `exec_mask`、Streaming State 与 Reset

### 6.1 共享 `exec_mask`

### 可行性：很高

当前每个 `StreamingState` 都分配自己的 `[8]` bool tensor。每个 codec step：

1. `codec.apply()` 遍历整个 encoder + decoder module tree。
2. 找到 392 个 live streaming state。
3. 对每个 state 做一次 `state.exec_mask[:] = exec_mask`。

trace 中对应：

- `_set_streaming_exec_mask`：18 次，总计约 186 ms。
- `set_exec_mask`：7056 次，即 18 x 392。
- `_set` callback：50562 次，即 18 x 2809 个 module。

所有 state 只读取 `state.exec_mask`，没有要求它们拥有独立值。因此可以让同一个 session 的所有 decoder state 引用一个稳定的 `session_exec_mask`。

每步只需：

```python
session_exec_mask.copy_(new_exec_mask)
static_codes.copy_(codes_step)
graph.replay()
```

共享 tensor 地址在 graph 生命周期内稳定，符合 CUDA Graph 要求。

### Reset 注意事项

当前 `StreamingState.reset()` 会顺便把 reset slot 的 exec mask 改成 True。共享 mask 后必须解耦：

- `exec_mask` 只描述“本次 decode 哪些 slot active”。
- reset 只清 KV、offset 和其他 per-slot state。
- 下一次 step 在 graph replay 前统一写 exec mask。

否则 release 某个 slot 时可能意外修改其他 state 共同读取的 mask。

### 6.2 只为 decoder 建 streaming state

### 可行性：很高

当前 session 使用 `codec.streaming(batch_size)`，model-level `_start_streaming()` 对 encoder 和 decoder 全部执行 `apply()`。但 Stage1 `_decode_frame()` 根本不会调用 encoder。

encoder 和 decoder 结构基本对称。392 个 state 中，decoder 约 196 个。可以仿照 codec 自身 chunked decode 的写法，只对顶层 decoder streaming modules 建 context，避免初始化和遍历 encoder state。

这也让 `exec_mask` 和 reset 的作用域更清晰。

### 6.3 合并 finished-slot reset

### 可行性：很高

当前一个 group 解码后逐 request 调用 `_finish_stream_request()`，内部 `session.release(slot)` 会立即遍历并 reset 一次。8 个请求一起完成时会做 8 次完整 reset traversal。

trace 中：

- `_finish_stream_request` 8 次，总计约 300 ms。
- `_reset` callback 22472 次，即 8 x 2809。

应把生命周期操作拆成：

1. 收集本轮所有 finished slots。
2. 一次 `reset_slots(finished_slots)`。
3. 批量归还 free-slot bookkeeping。
4. 清理 request-id maps。

这个改动不改变每个 state 的 reset 数学，只把相同 reset mask 合并，风险低。

进一步把 decoder state list 缓存在 session 中，可同时去掉每次 `Module.apply()` 的 Python traversal。若 reset 小 kernel 仍明显，再考虑 multi-tensor/Triton reset；第一版不需要直接上自定义 kernel。

---

## 7. Stage0 Local Static KV Cache

### 可行性：很高

当前 `MossTTSLocalDepthTransformer.generate_frame()` 每帧执行 prefix 长度：

```text
1, 2, 3, ..., 12
```

也就是 12 次 local block，处理 78 个 token rows。静态增量 KV 后仍然是 12 次顺序调用，但每次只处理 1 个新 token，总共 12 个 token rows。

对主要 linear/LN/MLP 计算，理论工作量下降约：

```text
78 / 12 = 6.5x
```

attention 的 QK 工作量也从完整 prefix 的平方累积下降为单 query 对已有 KV，但这个模型主要成本仍是 hidden=2560、FFN=9728 的 MLP。

官方 Local v1.5 已经提供可对照实现：

- `MOSS-TTS/moss_tts_local_v1.5/modeling_moss_tts.py:242-313`
- `MOSS-TTS/moss_tts_local_v1.5/gpt2_decoder.py:314-360`

### 推荐在当前轻量实现内移植

不建议把整套 HF GPT2 decoder 搬进 vLLM。当前 local transformer 已经做到 checkpoint name 1:1 和数值语义清晰，只需给 `_MossTTSLocalAttention` 增加 incremental 接口：

```text
input hidden: [B, 1, H]
position: 0..11
static K/V: [B, 12, n_head, head_dim]
```

每个 frame 新建或复用一组短 static cache，frame 结束后逻辑 reset。B8、BF16 下 K+V 容量不到约 1 MiB，不是显存问题。

### CUDA Graph 兼容性

当前完整 `talker_mtp` 已按 batch bucket capture。local KV 的 12 个位置和 slice 都是静态的，可以完整记录进 graph。需要避免在 replay 语义中依赖动态 Python `cache.length`；更稳妥的是让 12 步 loop index 直接决定写入位置和可见 KV 长度。

对于单 query、已有 KV 的 SDPA，不能误用 `is_causal=True` 的 square-sequence 假设。应显式让 query 看见 `[0..current_position]` 的全部 cache，或使用与官方 `_sdpa_attention()` 相同的 q/k 长度处理。

### 数值风险

- 必须保持 GPT-J/interleaved RoPE，而不是 NeoX rotate-half。
- position 0 同时服务 binary head 和 codebook 0。
- codebook k 的输入仍是 codebook k-1 的 embedding。
- RNG 调用次数和顺序必须不变，否则即使 logits 等价，生成音频也无法逐 token 对照。

### 验证标准

1. 先比较每个 codebook step 的 hidden/logits，禁用 sampling 或固定输入 token。
2. 再在相同 RNG 状态下比较 12 个 sampled codes。
3. 覆盖 B1/B2/B8、stop frame 和 padded graph rows。
4. profiler 中 local linear 的 token-row 计算明显下降，graph hit 率不变。

---

## 8. Sampling 与 Forced Text Token

### 8.1 Top-k 后只在 k 个候选内做 top-p

### 可行性：高

当前 audio sampling 参数固定为 `top_k=25, top_p=0.8`。实现先用 top-k threshold 把完整 1024 vocab 其余位置设为 `-inf`，随后仍对完整 1024 做 sort、softmax、cumsum、scatter 和 multinomial。

trace 中出现 756 次 radix sort、CUDA 总计约 66 ms，以及 1920 次 softmax。这个数量与 local codebook 循环高度相关。

可以直接保留 `topk_values/topk_indices`，只在 25 个候选内做 top-p、softmax 和 multinomial，最后 gather 回原 token id。

唯一数值差异是 kth boundary 出现完全相同 logit 时：当前 threshold 语义可能保留超过 k 个并列 token，而 `torch.topk` 只返回 k 个。真实模型中精确 ties 预计很少，但验证中应明确接受该差异，或保留一个 eager tie fallback。

### 8.2 151936 维 forced text logits

### 可行性：中

`compute_logits()` 每步分配 `[B, 151936]`，只有 `audio_assistant_slot_token_id` 和 `im_end_token_id` 两列可能为有限值，随后仍进入通用 vLLM sampler。

最低风险短期方案是保证 Stage0 通用 sampler 使用 greedy 配置：

```yaml
temperature: 0.0
top_p: 1.0
top_k: -1
```

因为每行始终只有一个有限 logit，这不会改变 token 结果，但能去掉通用 top-k/top-p。它仍然保留完整 logits allocation 和 vocab argmax。

完整优化需要 model-owned/direct-token sampler：直接把 `should_continue` 映射为两个 token id，跳过 full-vocab logits 和通用 sampler。仓库已有 `prefer_model_sampler` 模式可参考，但该改动涉及 runner 的 logprobs、async sampling 和输出契约，风险高于 local KV 与 codec 优化，建议放在 P1。

### 8.3 Request seed scalar fallback

如果请求显式传 `request.seed`，serving 会写入 `extra_args["tts_local_seed"]`，`_talker_mtp_forward()` 在 batch>1 时会递归拆成 B1，以保证每请求 RNG 独立。

当前 profile 明确出现 B8 local decode，因此这不是本次 trace 的主因。但它是生产配置风险：一旦客户端显式传 seed，高并发性能会骤降。长期需要基于 `(request seed, frame index, codebook index)` 的 counter-based GPU RNG，短期应记录 scalar-fallback metric 并避免 benchmark 请求传 seed。

---

## 9. 其他可行优化

### 9.1 只把 active codec audio rows 拷回 CPU

当前 `session.step()` 对 graph 输出执行：

```python
audio_tensor.detach().to("cpu", torch.float32)
```

即使只有一个 active slot，也会把固定 B8 的完整 waveform 拷回 CPU。T15、48 kHz stereo 下，B8 FP32 输出约是数 MiB，而 B1 只需要其中 1/8。

建议先在 GPU 上 `index_select` active rows，再做一次 contiguous D2H，并保持一次批量 copy，避免按 slot 发起多个小 D2H。它不能减少 codec compute，但可降低 B1/B2 cohort 的 PCIe、CPU memory 和同步等待。

### 9.2 Last-hidden stable pool

当前上一轮 hidden 的路径是：

```text
per-request GPU clone
-> Python dict
-> 下一轮逐 request .to()
-> torch.cat()
-> runner packed buffer
```

已有实验直接复用了 runner packed buffer，但 mixed prefill/decode 时 source/destination row 会重叠，并且 persistent request slot 与 packed decode row 语义混在一起，因此已回退。

正确方案是两个独立 buffer：

- stable request-slot hidden store。
- current packed talker-MTP input buffer。

每轮通过 index gather 从前者复制到后者。这个方案可行，但需要覆盖 request reorder、preemption、resume、abort 和 mixed batch，且当前 profile 中 `preprocess_decode_batch` CPU 总计约 14 ms，优先级明显低于 codec 和 local KV。

### 9.3 Reference audio 不在这两条 stage trace 的完整范围内

MOSS reference audio 在 API 进程的 CPU tokenizer 中编码，通过 `asyncio.to_thread()` 并发执行，并按 ref string hash / named voice 缓存。对于唯一 reference 的 C8 workload，多个重 codec CPU encode 可能发生线程和内存带宽竞争；这部分发生在请求进入 Stage0 profiler 之前，不能从两条 stage trace 中排除。

当前 cache 没有 single-flight。同一未命中 ref 同时到达时可能重复编码。后续端到端 profiling 应单独记录：

- ref resolve time。
- ref codec encode queue/compute time。
- cache hit/miss。
- 同 key 并发 miss 数量。

但在没有该时间分解前，不应把 693 ms TTFP 全部归因于 Stage0/Stage1 GPU trace。

### 9.4 `connector_get_sleep_s` 当前不生效

SharedMemory chunk adapter 的 receive loop 在无进展时硬编码等待 1 ms；当前 YAML 的 `connector_get_sleep_s: 0.005` 没有被这条路径读取。因此只调 YAML 中这个值预计不会改变当前结果。

---

## 10. 推荐实施顺序

### Phase A：不改算法的隔离和低风险清理

1. Stage0/Stage1 分卡对照。
2. decoder-only streaming states。
3. shared session `exec_mask`。
4. batch finished-slot reset。
5. active-row D2H。

这组除了分卡外都不改变 codec 数学，适合先落地并建立稳定 profile。

### Phase B：高收益计算路径

1. 接通 FP32 weights + BF16 compute/KV cache。
2. Stage0 local static KV cache。
3. top-k candidate-space sampling。

这组需要数值和音质回归，但收益比继续微调 Python dict 更大。

### Phase C：调度与 cohort

1. 同一 forward 的 tail waterfall。
2. steady chunk 的 bounded coalescing window。
3. 根据 graph 时间扫描 5/10/15/20 ms，而不是先扫小 `codec_chunk_frames`。

调度改动必须同时看平均 TTFP、P99 TTFP、RTF、audio underrun 和 ready-wait 时间。

### Phase D：长期架构

1. direct forced-token sampler。
2. stable per-request hidden pool。
3. codec paged KV / dynamic active batch。
4. 跨 GPU 的 D2D/NCCL tensor connector，前提是 CPU SHM 已成为可见瓶颈。

---

## 11. 建议的验收矩阵

所有实验应区分：

- 8 prompts / C8 的单波次 profile。
- 100+ prompts / C8 的持续补入稳态。
- 相同 ref warm-cache。
- 唯一 ref cold-cache。
- 单卡共置与双卡分离。

至少记录：

| 层级 | 指标 |
|---|---|
| API | ref resolve、ref encode、request admission |
| Stage0 | prefill、backbone decode、local frame decode、sampling、batch size |
| Connector | chunk ready time、ready-to-scheduled wait、payload bytes |
| Stage1 | active slots、step T、graph hit、codec GPU time、D2H time |
| Client | TTFP、RTF、E2E、underrun、chunk interval |
| Quality | WER/CER、speaker similarity、UTMOS、waveform continuity |

特别建议增加以下结构化计数器：

```text
stage0_decode_batch_size
stage0_talker_mtp_graph_bucket
stage1_ready_streams
stage1_active_slots_per_step
stage1_codec_step_t
stage1_coalesce_wait_ms
stage1_tail_segment_count
stage1_exec_mask_updates
stage1_reset_passes
stage1_graph_hit_by_t
```

只有同时看到 ready cohort、active slots 和实际 graph T，才能判断 C8 退化是算力、调度还是流式状态碎片化。
