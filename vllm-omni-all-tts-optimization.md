# vLLM-Omni 的 TTS 工程实践

> 作者：vLLM-Omni maintainers

vLLM-Omni 从最初只支持 Omni model，到如今覆盖了 Qwen3-TTS、VoxCPM2、Fish Speech S2 Pro、Higgs Audio V3 等若干 TTS 模型。这篇文章会介绍我们在适配和优化这些模型过程中遇到的具体问题、解决方案，以及背后的工程判断。

---

## TTS 推理和传统的 LLM 推理有什么不同

TTS 和 LLM 文本推理都用自回归模型，但工程上的瓶颈点不一样。

**TTS 是 pipeline，由多个模型级联构成。** 典型的 TTS 系统至少有两阶段——Talker 做自回归 codec token 预测，Code2Wav 把 codec 转成音频波形。两个阶段的计算特征差得很远：Talker 是 latency-bound 的单 token decode，Code2Wav 是 throughput-bound 的并行 decode。如果调度策略不区分这两阶段，Talker 的延迟会阻塞 Code2Wav 的输入，Code2Wav 的并行能力也利用不起来——延迟和吞吐都受影响。

**流式输出对延迟有严格要求。** 语音合成场景里，用户期望在几百毫秒内听到第一段音频。pipeline 的 connector 层必须做 chunked streaming，chunk 大小直接影响 TTFP（Time To First Audio Packet）——太小则 Code2Wav 没有足够的上下文窗口保证跨 chunk 连续性，太大则首包延迟不可接受。

**吞吐同样关键。** 在线服务场景下，单卡能跑多少并发、每秒能产出多少秒音频，直接决定部署成本。TTS 的吞吐优化比 LLM 更复杂——Talker 和 Code2Wav 两个阶段的吞吐瓶颈完全不同，两个阶段之间还有 connector 的传输开销。提高吞吐需要在两个阶段之间找到平衡点，同时消除每个阶段的瓶颈。

![vLLM-Omni TTS serving pipeline](https://files.seeusercontent.com/2026/06/21/3wWz/2.png)

下面的内容先给出一张优化手段总览，然后用 Qwen3-TTS 作为主线走一遍完整的优化流程，最后用 VoxCPM2、Higgs Audio V3 和 Fish Speech S2 Pro 来说明当模型架构不同时，优化策略需要怎么调整。

---

## 优化手段总览

不同 TTS 模型的瓶颈并不一样。vLLM-Omni 没有用同一套优化硬套所有模型，而是根据每个模型的 pipeline 形态、decode 状态、batch shape 和数值敏感点选择不同路径。下表是四个模型在主要优化维度上的对比：

| 维度 | Qwen3-TTS | VoxCPM2 | Higgs Audio V3 | Fish Speech S2 Pro |
|------|-----------|---------|----------------|---------------------|
| Stage 设计 | Talker → Code2Wav 两阶段 | single-stage，AR + CFM/LocDiT + AudioVAE 在同一 runtime | Talker → Code2Wav 两阶段 | slow_ar + Fast AR → DAC decoder |
| Batching | Stage 0 decode 预处理 batch 化 | CFM/LocDiT decode-tail batching、VAE batch | GPU-resident batched decode state | DAC batching、async codec chunk |
| torch.compile | code predictor 部分路径保持 PyTorch/fp32 对齐 | whole-model forward compile，fullgraph=False | 不是主要收益点 | Fast AR compile（dynamic=True） |
| CUDA Graph | 用在稳定 shape 的路径 | reduce-overhead 与 KV cache 冲突，不用 | local MLP CUDA Graph 是吞吐主力 | Triton workspace 支持 capture |
| Kernel / fusion | mask 预计算、bucket lookup | LocDiT fused QKV / fused gate-up MLP | FlashInfer attention | Fish-specific q_len=1 Triton KV attention |
| State / 数据路径 | trailing_text offset/compact | `.item()` sync 移除、sliding-window VAE | 多码本状态迁移到 GPU tensor | Fast AR buffer/KV cache 复用、tensor payload |
| 核心原因 | 高并发下 Talker Python hot path 是瓶颈 | 单请求 CFM batch 太小，GPU 利用率低 | 多码本状态机复杂，batch 动态变化 | q_len=1 attention 和小 tensor 分配是主要开销 |

这张表的核心信息是：不是所有优化手段对所有 TTS 架构都有效。有意思的工程判断在于为正确的模型形态选择正确的杠杆。下面从 Qwen3-TTS 开始，走一遍完整的优化过程。

---

## Qwen3-TTS：一条完整的优化路径

Qwen3-TTS 是 Qwen 团队发布的语音生成模型系列，基于离散多码本语言模型架构，采用自研的 12Hz Tokenizer 做声学压缩与高保真重建[^11]。三个变体——Base（语音克隆）、CustomVoice（预定义说话人+情绪控制）、VoiceDesign（自然语言描述音色）——共享同一个两阶段架构：Talker 做自回归 codec token 预测，Code2Wav 做并行解码。Qwen3-TTS 的 Code2Wav 是非 DiT 的轻量架构，不需要扩散模型的多步推理。

Qwen3-TTS 的 pipeline 形态在四个模型里最"标准"——Talker → connector → Code2Wav——所以它很适合作为主线，展示 TTS 推理优化的完整流程。

### 1. 流式问题：connector chunk 和 Code2Wav decode window 的耦合

优化从流式输出开始。Qwen3-TTS 的早期实现里，connector 的 streaming chunking 和 Code2Wav 的 decode chunking 都依赖同一组 chunk 参数，主要是 `codec_chunk_frames`。

connector 负责从 Talker 向 Code2Wav 传输 codec tokens。流式模式下 connector 每次只发一个小 chunk，Code2Wav 的 decode chunk 也会变小，跨 chunk 的音频连续性变差。反过来，为了 Code2Wav 的质量把 chunk 调大，first-packet latency 就上去了。

解耦的做法是引入三个独立参数：

- `codec_chunk_frames`：connector 的流式 chunk 大小（控制 Talker→Code2Wav 的传输节奏）
- `decode_chunk_frames` / `decode_left_context_frames`：Code2Wav 内部的 decode 窗口大小和左侧上下文（保持默认值，不受 connector 影响）
- `initial_codec_chunk_frames`：首个 codec chunk 用更小的窗口发出，让 Code2Wav 尽早开始接收数据；后续 chunk 恢复正常大小

connector 用更小的 chunk size 让 Code2Wav 更早拿到第一批数据，Code2Wav 内部保持 300 帧 decode 窗口 + 25 帧左侧上下文，保证跨 chunk 音频质量。两者各自独立优化[^6]。

![Qwen3-TTS connector chunk decoupling](https://files.seeusercontent.com/2026/06/21/5Tty/3.png)

### 2. 吞吐问题：Stage 0 的 decode 预处理

流式延迟解决后，下一个瓶颈是吞吐。提高吞吐的第一个障碍是 Talker 的 decode 循环。Qwen3-TTS 的 Talker 每次 decode 需要做请求级别的预处理——准备 speaker embedding（声纹特征向量）、更新 trailing_text（已生成文本的上下文窗口，Talker 用它来决定接下来的语音特征）、构建输入 embedding。在 c=1 时这些操作开销不大，但 c=64 时，每个 decode step 都要串行遍历 64 个请求，Python 侧的 for 循环和 tensor slice 成为瓶颈。

为了定位具体开销，我们在 c=64 下对 Talker 的单步 decode 做了分段 profile。一个 decode step 里，模型外的 Python/runner 侧处理（`preprocess_decode_batch`、`make_omni_output`、`process_additional_info`、`build_mm_cpu`、bookkeeping sync）合计 p50 约 2.8–3.5ms，p90 约 3.9–5.7ms。单个 step 的几毫秒看起来不大，但 Qwen3-TTS 一次 utterance 需要 ~200 个 decode step，c=64 时这 200 步全都串行累积，模型外的 Python 开销在端到端延迟中占比显著。

另一个角度是 GPU 利用率。用 nvidia-smi 在 c=64 下采样，优化前 Stage 0（Talker）和 Stage 1（Code2Wav）的平均 GPU 利用率分别只有约 14% 和 6%。GPU 大部分时间不在计算，而是在等 Python 侧的调度、小 tensor 分配和 kernel launch。这组数据解释了为什么瓶颈不在算力，而在服务路径开销。

首先是 speaker embedding 的准备。Qwen3-TTS 在 voice-clone 模式下，每个请求需要使用 reference audio 跑 speaker encoder 提取 embedding，然后在 decode 过程中做 mel/STFT 处理。原来的做法是每个请求单独在 CPU 上算 mel spectrogram，再 copy 到 GPU。这在高并发下变成了大量的 CPU→GPU 小数据传输和 kernel launch。把 mel basis 和 window buffer 缓存到 GPU 上，整个 mel/STFT 计算直接在 GPU 上 batch 完成，省掉了每次的 CPU 计算和 H2D 传输。

然后是 `trailing_text` 的管理。Talker 在 decode 过程中需要维护一个 trailing_text 滑动窗口（已生成 token 的 embedding 缓存），每次 decode 取当前 token 对应的 embedding 追加到窗口末尾，再把最老的 token 移出去。原来的做法是每次 decode 都做一次 tensor slice 和 concat，频繁分配新 tensor。用 offset 跟踪当前位置，只在 offset 超过阈值或到达末尾时才做一次 compact（`_TRAILING_TEXT_COMPACT_MIN_FRAMES = 64`），中间的 decode step 直接通过 offset 索引，不分配新 tensor。

这两项改动加上 `preprocess_decode_batch` 把整个 decode 预处理变成一次 batched 操作，在 H20×2 上把 audio throughput 从 26.55× 提升到 42.88×（+61.5%），P99 E2EL 从 17.7s 降到 9.0s[^7]。61.5% 是多项优化叠加的结果——speaker embedding 批量化、trailing_text offset 管理、preprocess_decode_batch，以及后续 async D2H、runner hot-path 和 connector 等路径的叠加。

### 3. hot-path 清理

预处理批量化之后，profile 里剩下的热点是一些细碎的 Python 开销。单个看都不大，但在 c=64 的高频 decode 循环里累积起来就明显了。

`req_id_to_index` 原来用 `req_ids.index()` 做 O(N²) 的 list scan，每次 decode step 都要查一遍，改成 dict 查找后变成 O(1)。非流式请求不需要走 orchestrator 的 per-output streaming walk，直接在入口处跳过。codec-disallowed mask 预计算成一个 buffer，`compute_logits` 里每次直接 `masked_fill` 而不是重新算一遍[^1]。

Qwen3-TTS 的 CUDA Graph 用在 Code2Wav decoder 侧，不是 Talker 侧。Code2Wav 的 decoder 输入 shape 是 `(batch, num_quantizers, codec_frames)`，其中 `codec_frames` 在 chunked decode 模式下取值是有限的——要么是 streaming 的 `codec_chunk_frames + left_context`，要么是非流式的 `decode_chunk_size + left_context`（300+25=325），以及最后一个不完整的 tail chunk。这些取值在 warmup 阶段就能枚举出来，用 `CUDAGraphDecoderWrapper` 按 `(batch_size, frames)` 预先捕获所有 shape 的 graph，inference 时用 `bisect_left` 做 O(log n) 的 bucket 查找匹配最近的 padded size，查不到则 fallback 到 eager。多轮测试中 hit rate 稳定在 80% 以上。

### 4. 数值精度：code predictor 的 fp32 对齐

Talker 内部的 code predictor 子模块有一个精度问题。code predictor 序列长度很短（2-8 token），每一步都重新 prefill，vLLM 的 fused kernel 在 bfloat16 下和参考实现在数值上有微小差异，在这个短序列高频 prefill 的场景里会逐步累积，几十步后音质出问题。处理方式是拆开每一层，RMSNorm 方差、RoPE 的 cos/sin 都升为 float32 计算，Attention 和 QKV 投影也回到 PyTorch 原生实现，逐 bit 对齐。

### 5. 验证

以上优化叠加后，在 c=64、H20×2 的 voice-clone 场景下，audio throughput 提升 61.5%，P99 端到端延迟下降近一半。完整数据见文末性能数据章节。

在 warm 状态下跑了一个并发矩阵，更直观地看延迟随并发的变化：

| c | Mean TTFP | Mean E2E | P50 TTFP | P50 E2E |
|--:|----------:|---------:|---------:|--------:|
| 1 | 70.61ms | 564ms | 70.61ms | 564ms |
| 8 | 268.75ms | 1.55s | 287.15ms | 1.70s |
| 16 | 451.32ms | 2.62s | 516.15ms | 2.75s |
| 32 | 637.43ms | 5.05s | 634.22ms | 5.10s |
| 64 | 1127.93ms | 8.73s | 1051.05ms | 8.78s |

从 c=1 到 c=64，E2E 从 0.56s 增到 8.73s，不是 64 倍线性增长——warm 状态下高并发能摊薄固定成本。但 Talker/调度侧仍会在 c=64 时放大成主要排队来源，这也是为什么后续 hot-path 清理和 CUDA Graph 有实际收益。

---

## VoxCPM2：当模型是 single-stage hybrid 时

VoxCPM2 是 OpenBMB 发布的 tokenizer-free TTS 模型，基于扩散自回归混合范式，在 AudioVAE V2 的 latent 空间中运行[^12]。它的 Talker 是一个四阶段级联：

```
MiniCPM4 (28层, PagedAttention) → FSQ → MiniCPM4 ResidualLM (8层) → LocDiT (CFM solver) → AudioVAE
```

其中 LocDiT 做 CFM（Conditional Flow Matching）扩散去噪，AudioVAE 重建 48kHz 波形。在 vLLM-Omni 里，VoxCPM2 没有拆成多个 runtime stage，而是作为一个 single-stage AR TTS pipeline 运行：MiniCPM4、FSQ、ResidualLM、LocDiT 和 AudioVAE 都在同一个模型实例里完成，最后直接输出 audio。这样做可以避免 latent 在 stage 之间传输，也让 decode tail 的 CFM/LocDiT 和 VAE 更容易做跨请求 batching。

![VoxCPM2 single-stage hybrid pipeline](https://files.seeusercontent.com/2026/06/21/H1mf/5.png)

和 Qwen3-TTS 的两阶段 Talker → Code2Wav 不同，VoxCPM2 的优化重心不在 stage 之间的 connector 和 streaming 边界，而在两个问题上：怎么让 28 层的 MiniCPM4 跑得更快，怎么让 CFM/LocDiT 在高并发下不空转。

### torch.compile 的探索

VoxCPM2 的 28 层 MiniCPM4 是 Talker 中计算量最大的部分，首先考虑的是用 `torch.compile` 加速。但实际收益最大的方向并不在最初预期的路径上。

最初的做法是对每层的 `mlp` 和 `o_proj` 单独 compile（28 层 × 2 = 56 个 compiled region，`fullgraph=True`）[^3]。问题出在 Dynamo 不能跨 compiled boundaries 做优化，每个 boundary 都有 Python→compiled→Python 的切换开销，56 个 region 意味着每个 decode step 都要反复进出 compiled region。

随后调整为把整个 `Model.forward` 用 `torch.compile` 包起来（`fullgraph=False`）[^4]。Dynamo 看到完整的 28 层 loop，PagedAttention 处会 graph break，但它只需要 memoize 少数几个 sub-graph，per-step dispatch 从几十次降到几次。RTF 从 ~0.21 降到 ~0.13，这是 VoxCPM2 整个优化过程中最大的单项收益。

为了量化 whole-model compile 的效果，我们用 torch profiler 对比了三种配置：eager、per-layer compile、whole/unified graph。per-layer compile 已经减少了一部分 kernel 数和 kernel time，但 launch 次数没有下降；whole graph 才是关键一步——cudaLaunchKernel 次数减少约 71%，kernel events 减少约 30%，kernel time 减少约 27%。单请求 E2E 方面，per-layer compile 只减少了约 2.6%，whole graph 减少了约 6.5%。

后来还试过升级为 `mode="reduce-overhead"` 来启用自动 CUDA Graph capture，但发现 reduce-overhead 的 CUDA Graph 和 PagedAttention 的 stateful KV cache 有冲突——graph capture 时 `slot_mapping` 被固定，replay 时 attention 写入了错误的 KV cache 位置，stop logits 会产生错误，生成过早截断。

RoPE 和 RMSNorm 也解释了为什么不能用 `fullgraph=True`。RoPE 为了数值精度需要在 float32 下计算，代码里会做 `tensor.to(torch.float32)` 这样的显式 dtype 转换；RMSNorm 算 variance 时也是类似的情况。`fullgraph=True` 遇到这些 dtype 转换和 PagedAttention graph break 会直接失败，而 `fullgraph=False` 可以保留 whole-forward 的视角，同时允许这些边界退回 eager。

### CFM/LocDiT 的 decode-tail batching

单请求延迟解决后，高并发下的瓶颈在 CFM/LocDiT 阶段。每个请求的 CFM 扩散去噪需要跑 LocDiT attention/GEMM，但单个请求的 batch 极小（CFG 下 B=2），远不足以填满 GPU。高并发时，多个请求各跑各的 LocDiT，GPU 利用率很低。

我们的思路是把 CFM/LocDiT 的 decode tail 做 batch：将多个请求的 `lm_h`、residual 输出、prefix feature condition 收集起来，一次性跑 `dit_proj`、CFM/LocDiT、`feat_encoder`、`stop_head`，然后 scatter 回各请求状态。配合 VAE decode 每 3 个 latent chunk 触发一次（而不是每步）、跨请求 VAE batch、音频 D2H 拷贝合并，以及 LocDiT fused-QKV/fused gate-up MLP 等 fusion 优化，在 H20×1 c=64 下 request throughput 从 4.19 req/s 提升到 10.83 req/s（+158.8%），audio throughput 从 12.16 audio-s/s 提升到 33.07 audio-s/s（+172.0%）[^5]。

CFM 扩散的 Euler 积分循环中，对 0-dim GPU tensor 调用 `.item()` 会强制 GPU→CPU 同步。每个 diffusion step 调用 4 次，10 timesteps × 4 syncs × ~60 decode steps，一次请求下来就是 ~2,400 次同步。改为 `.copy_()` 直接在 GPU 上广播，省去全部 CPU 参与。

VAE 解码也有一个结构性问题。初版用的是 accumulate-and-re-decode 模式，每 5 步将所有已生成的 latent patches 拼接后送入 VAE 重新解码，计算总量是 O(N²)。改成 sliding-window decode（12-frame pad context + 4-frame new），每次只解码新的一小段，总量降为 O(N)。长文本 RTF 不再随长度增长，所有长度 RTF ≈ 0.132-0.138[^4]。

---

## Higgs Audio V3：当多码本状态机遇上动态 batch

Higgs Audio V3 是 Boson AI 发布的 TTS 模型，支持 100+ 语言和 zero-shot 语音克隆。架构上几个关键点：Qwen3 backbone（36 层，2560 hidden，GQA 结构）、fused multi-codebook embedding（`[N×V, D]` 大矩阵 + offset 查找）、MusicGen-style delay pattern `[0, 1, 2, ..., 7]` 配合 BOC/EOC 特殊 token。Talker → Code2Wav 的总体框架和 Qwen3-TTS 类似，但 Talker 内部的多码本预测机制和延迟模式有显著差异。

和 Qwen3-TTS 相比，Higgs v3 的优化重心不同：Qwen3-TTS 的瓶颈在 Python hot path 和 streaming chunk 边界，Higgs v3 的瓶颈在复杂的多码本 decode 状态管理和 CUDA Graph 适配。

### 把 decode 状态迁移到 GPU

Higgs v3 吞吐提升的核心是把原先 per-request Python dict 状态机迁移到 GPU-resident batched tensor 状态机[^10]。覆盖的状态包括 `_decode_last_codes`、`_decode_has_codes`、delay count、EOC countdown、generation done 等。主要收益来自减少 Python per-request 循环、减少 D2H 同步、让采样/状态更新走 batched GPU hot path。这套改动再加上 local MLP CUDA Graph 和 FlashInfer attention，在 H20 单卡 c=16 下达到 35.26 audio_s/s。

难点在于 vLLM scheduler 中 batch 可能被 reorder、shrink、finish/remove request，row-level 状态不能默认等价于 request-level 状态。音频 AR 的状态机比文本复杂——有延迟 codebook、EOC ramp-down、terminal frame 等特殊语义，任何一个状态落后一拍，表现为音频质量问题而非 crash。GPU state、CPU override、scheduler token 三者必须保持状态源唯一性，否则 stop 语义就会混乱。

### CUDA Graph 与动态 batch 的适配

在给 Higgs v3 的 Talker decode 阶段做 CUDA Graph 捕获时，遇到了一个棘手的问题。Talker 在 decode 过程中有一个 audio feedback 机制——上一步生成的 audio token 的 embedding 会替换掉下一步 continuation token 的 embedding。这个替换是通过 boolean mask 来筛选"哪些请求当前处于 decode 状态"的，筛选后的 tensor shape 取决于实际有多少请求在 decode。

CUDA graph capture 期间，stream 里所有操作的 input/output shape 必须是确定的、不依赖数据的。而这个 boolean mask 筛选的结果恰恰是依赖数据的——batch 里有多少 decode 请求在 capture 时是未知的，导致代码直接崩溃。

解决思路是在 CUDA graph 路径下，让 decode batch 始终是均匀的单 token decode（每个 span 都等于 1）。这样 `decode_mask` 天然是全 True，筛选操作退化为无操作，直接返回原始 tensor。graph 里看到的 shape 始终是完整的 batch 维度，不再依赖运行时状态。

### local MLP CUDA Graph 与 PIECEWISE 的取舍

local MLP CUDA Graph 是保留下来的一个重要优化。它覆盖 `post_attention_layernorm + mlp` 这部分 GPU 时间的主要开销。vLLM 的 PIECEWISE CUDA Graph 看起来是更完整的方案——把整个 decode step 都包进 graph，理论上能消除更多 kernel launch gap。但 Higgs v3 的模型结构比较特殊：multi-codebook 的 delay pattern 让每次 decode 的 token 布局不同，embedding lookup 和 attention 前面的 index 操作本身是 data-dependent 的，PIECEWISE graph 在这些地方要么 graph break 退回到 eager，要么需要额外的 metadata 同步。实测下来，PIECEWISE graph 需要禁用掉 local MLP graph 才能工作，但 local MLP graph 在 `post_attention_layernorm + mlp` 这块的收益远大于 PIECEWISE graph 在 attention 前后的 gap 消除。端到端对比，eager + local MLP graph 比 PIECEWISE graph 更快。

### staging overlap 的尝试

放弃的方案中，one-step audio staging overlap 值得单独讨论。这个想法是把 audio staging 的 D2H 拷贝和下一步计算 overlap 起来，减少 GPU 空等。dry-run 结果正常，但压测下发现 vLLM scheduler 在 decode 过程中可能 reorder、shrink 或 finish 请求，cursor 指着的 row 和 request 之间的对应关系会被打破。这是一个 cursor-lag 设计在 batch 动态变化下的结构性不安全，不是修修边界条件能解决的。后续如果要再做 overlap，需要采用 request-id keyed 的方式，加上 finish/remove 的 drain hook。

---

## Fish Speech S2 Pro：当通用 attention 变成瓶颈

Fish Speech S2 Pro 是 Fish Audio 发布的 TTS 模型，基于 Dual-AR（双自回归）架构，在 1000 万小时+ 的音频数据上训练，覆盖 80+ 语言[^13]。在 vLLM-Omni 的实现里，Fish Speech S2 Pro 走的是 slow_ar + Fast AR + DAC decoder：slow_ar 沿时间轴预测语义码本，Fast AR 在每个 decode step 生成剩余残差码本，DAC decoder 接收 10 个 codebook 后重建波形。

和 Qwen3-TTS 相比，Fish Speech 的优化重心在另一个方向上：Qwen3-TTS 的瓶颈在 Python 侧预处理，Fish Speech 的瓶颈在 GPU 侧——q_len=1 的 attention 在高并发下成了主要开销，通用 paged attention 的 shape check 和分支逻辑对纯 decode 场景来说全是浪费。

### model-specific attention kernel

Fish Speech 的 slow_ar 在高并发 decode 时，profile 下来大部分时间花在两个地方：q_len=1 的 SlowAR attention，以及 DAC 和 runtime 之间的数据交接。通用 paged/varlen attention 要兼容 prefill、chunked prefill、decode 等各种形状，代码路径上有一层层的 shape check 和分支。对 Fish 的纯 decode 场景——q_len=1、head_dim=128、block size 16、GQA 布局——这些通用逻辑全是开销。

我们给 Fish SlowAR 写了一个只做 decode 的 Triton kernel，不处理 prefill 和其他模型。这个 kernel 的要求很明确：q_len=1、fp16/bf16、head_dim=128、block size 16、Fish 的 GQA。不满足条件就回退到原来的 attention，不增加任何分支。

kernel 分两条路径。短序列（≤1024 token）用标准的 online softmax 一次完成，grid 是 `(batch_size, num_kv_heads)`，每个 program 处理一个 batch 行的一个 KV head 对应的所有 Q head。block size 硬编码为 16，和 vLLM 的 KV cache block 大小对齐，block table 查表直接走 `tl.load` 不用额外 gather。长序列走 split-partial-combine 两阶段——把序列切成若干段，每段独立算 partial m/l/acc，再按 online softmax 的归约公式合并。这让带 reference audio 的长序列请求也能走 fast path。

dispatch 的判断条件有一个容易忽略的细节。kernel 需要知道每个请求的 seq_len 才能决定走短路径还是长路径，但 seq_len 在 GPU 上，读到 CPU 需要同步。做法是在 runner 层面提前算一个 CPU 侧的 `seq_lens_cpu_upper_bound`（已计算的 token 数 + 本轮 scheduled token 数），这个上界一定 ≥ 真实 seq_len，短路径不会越界，长路径的 split 也不会少算。CUDA graph capture 时，这个 upper bound 直接填 `max_model_len`，保证 graph 里所有路径都覆盖。

整个 fast path 只对 Fish SlowAR 的 attention layer 生效，安装方式是在模型加载时遍历 `model.layers`，把每个 attention layer 的 `impl.forward` 替换成带 Fish fast path dispatch 的 wrapper。prefill 请求、非 Fish 模型、不满足 shape 约束的 decode 请求都走原来的 attention 实现。

### Fast AR 的 buffer 复用与 compile

Fish Speech 的 Fast AR 是一个 4 层轻量 transformer，负责在 slow_ar 每一步之后预测残差码本。它维护 per-call KV cache：每个残差码本 step 只 decode 新 token，并把 K/V 写入预分配的 `_k_cache` / `_v_cache`，避免短序列高频 decode 下的重复计算。

Fast AR 的每步 decode 需要完成以下操作：把 slow_ar 的 hidden state 投影到 Fast AR 的维度，嵌入当前语义 token，逐层计算 attention 和 MLP，最后从 logits 中采样。即使序列很短（最多 10 个 token），在 c=64 的高并发场景下，重复 allocation 和重复 prefill 仍会累积成可见开销。

`_embed_buf`、`_pos_ids`、`_k_cache` 和 `_v_cache` 在第一次调用时分配，后续调用直接复用。`_embed_buf` 的 shape 是 `(batch_size, num_codebooks+1, hidden_dim)`，覆盖一次 Fast AR decode 的所有时间步；`_k_cache` / `_v_cache` 按 layer、batch、KV head、sequence position 和 head_dim 预分配，供 `forward_one` 逐步写入和读取。

`torch.compile` 也用在 Fast AR 上。和 VoxCPM2 的 MiniCPM4 不同，Fast AR 只有 4 层，compile 的 overhead 小。`fullgraph=False` 是因为 attention 走的是 `F.scaled_dot_product_attention`（SDPA），不是 paged attention，SDPA 内部会 graph break，但 Dynamo 只需要 memoize 几个 sub-graph。`dynamic=True` 让编译结果在 batch size 变化时也能复用。

### DAC 和 runtime 侧

DAC 和 runtime 侧的优化包括几个方面。codec payload 的传输格式从 Python `list[int]` 改成 tensor payload——2D code tensor 直接序列化传输，不展开成整数列表，省去高并发下的 int 分配和 GC。fp16 DAC 支持让显存和计算量都降了一半，配合 frame-work bounded DAC batching，控制每次 DAC forward 处理的帧数上限，避免单个请求的 DAC 过久阻塞其他请求。async chunk 处理让 connector 的传输和 DAC 计算可以 overlap——SlowAR 和 Fast AR 每 decode 一步产出一帧 10-codebook 的 codec token，connector 积累到 `codec_chunk_frames` 帧后批量传输给 DAC decoder，DAC decoder 在处理当前 chunk 时，connector 已经开始累积下一个 chunk。

---

## 性能数据

以下数据来自 vLLM-Omni cookbook 的 benchmark。指标说明：

- **RTF**（Real-Time Factor）：generation time / audio duration，<1 表示比实时快
- **TTFP**（Time To First Audio Packet）：从请求发出到收到第一个音频包的延迟
- **Tput**：`audio_throughput`，audio duration / wall time，即每秒产出的音频秒数
- **E2EL**：端到端延迟

### Qwen3-TTS（c=64, p=512, H20×2, voice-clone）

| 指标 | 优化前 | 优化后 | 变化 |
|------|------:|------:|-----:|
| Audio throughput | 26.55× | 42.88× | +61.5% |
| Median E2EL | 9654ms | 5699ms | −41.0% |
| P99 E2EL | 17686ms | 8956ms | −49.4% |
| P99 TTFP | 7558ms | 5563ms | −26.4% |

### VoxCPM2（c=64, H20×1, CFM 批量化前后）

| 指标 | 优化前 | 优化后 | 变化 |
|------|------:|------:|-----:|
| Request throughput | 4.19 req/s | 10.83 req/s | +158.8% |
| Audio throughput | 12.16 audio-s/s | 33.07 audio-s/s | +172.0% |

### Fish Speech S2 Pro（H20, 单卡, c=64, Triton kvcache + tensor payload）

| 指标 | 数值 |
|------|-----:|
| Audio throughput | 23.72 audio-s/s |
| Request throughput | 5.95 req/s |
| Mean TTFP | 899.67 ms |
| Mean E2EL | 10.47 s |

### Higgs Audio V3（H20, 单卡, c=16, FULL_DECODE）

| 指标 | 数值 |
|------|-----:|
| Request throughput | 5.18 req/s |
| Audio throughput | 35.26 audio_s/s |
| Wall time | 96.5s |
| Speedup vs baseline | 2.70× |

---

## 参考链接

[^1]: Qwen3-TTS hot-path 微优化 — [PR #3689](https://github.com/vllm-project/vllm-omni/pull/3689)
[^3]: VoxCPM2 per-layer compile + PagedAttention — [PR #2690](https://github.com/vllm-project/vllm-omni/pull/2690)
[^4]: VoxCPM2 whole-model compile + streaming VAE + CFM sync fix — [PR #2758](https://github.com/vllm-project/vllm-omni/pull/2758)
[^5]: VoxCPM2 CFM/LocDiT batching + decode-tail 优化 — [PR #3882](https://github.com/vllm-project/vllm-omni/pull/3882)
[^6]: Qwen3-TTS 流式 connector 解耦 — [PR #3485](https://github.com/vllm-project/vllm-omni/pull/3485)
[^7]: Qwen3-TTS 高并发 Stage 0 批量化 — [PR #3662](https://github.com/vllm-project/vllm-omni/pull/3662)
[^9]: Fish Speech S2 Pro KV cache fastpath + DAC 优化 — [PR #3773](https://github.com/vllm-project/vllm-omni/pull/3773)
[^10]: Higgs Audio V3 GPU-resident state machine + CUDA Graph — [PR #4204](https://github.com/vllm-project/vllm-omni/pull/4204)
[^11]: Qwen3-TTS — [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
[^12]: VoxCPM2 — [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM)
[^13]: Fish Speech S2 Pro — [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech)

---

*如果你对 TTS 推理优化感兴趣，欢迎来 [vLLM Slack](https://slack.vllm.ai) 的 `#vllm-omni` 频道，或者在 [vLLM-Omni GitHub](https://github.com/vllm-project/vllm-omni) 上开 issue。*