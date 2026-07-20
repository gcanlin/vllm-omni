# MOSS-TTS-Local-v1.5 性能差距与优化路线

## 背景和目标

本文分析 vLLM-Omni 当前 MOSS-TTS-Local-Transformer-v1.5 与 Qwen3-Omni / Qwen3-TTS 以及 SGLang-Omni MOSS Local 实现之间的性能差距，并给出后续优化路线。

当前观察到的现象：

- MOSS-TTS-Local 在 vLLM-Omni 中单并发 RTF 约 0.5-0.6。
- Qwen3-Omni / Qwen3-TTS 在成熟优化路径下可以做到更低 RTF，用户侧预期 Qwen3-Omni 可到约 0.2，TTFP 可到约 200ms。
- 当前 MOSS Local 的流式代码已回退到稳定非流式路径，Stage 1 实际仍是收完整 code 后再 decode，因此 TTFP 目前不具备与 Qwen3 系列对齐的条件。

本文引用和对照的本地材料：

- `sglang-omni-moss-tts-local-v15.md`
- `vllm-omni-all-tts-optimization.md`
- `docs/design/qwen3_omni_tts_performance_optimization.md`
- 当前 vLLM-Omni MOSS Local 代码：
  - `vllm_omni/model_executor/models/moss_tts/modeling_moss_tts_talker.py`
  - `vllm_omni/model_executor/models/moss_tts/modeling_moss_tts_local_depth.py`
  - `vllm_omni/model_executor/stage_input_processors/moss_tts.py`
  - `vllm_omni/model_executor/models/moss_tts/modeling_moss_tts_codec.py`
  - `vllm_omni/deploy/moss_tts_local.yaml`
- SGLang-Omni MOSS Local 参考代码：
  - `sglang_omni/models/moss_tts_local/model_runner.py`
  - `sglang_omni/models/moss_tts_local/state_pool.py`
  - `sglang_omni/models/moss_tts_local/local_transformer.py`
  - `sglang_omni/models/moss_tts_local/streaming_vocoder.py`
  - `sglang_omni/models/moss_tts_local/vocoder_cuda_graph.py`

## 结论摘要

MOSS Local 当前和 Qwen3-Omni / Qwen3-TTS 的 gap 不是一个单点问题。核心差距可以分成四层：

1. 模型结构本身更重：MOSS 使用 Qwen3-4B backbone、12 个 RVQ codebook 的 local transformer、MOSS-Audio-Tokenizer-v2 48kHz stereo codec。它的 codec 和每帧 local decode 都比很多 Qwen3 TTS 路径更重，因此不能简单假设完全达到 Qwen3-Omni 的 RTF。
2. vLLM 当前 MOSS Local 的 AR 帧内 decode 仍是 per-request eager 小 kernel 路径：local transformer 在 `make_omni_output` 中逐请求运行，且每帧做 CPU history、Python sampling、tensor concat，没有 GPU-resident state pool，也没有 frame decode CUDA Graph。
3. Stage0 到 Stage1 的数据路径还不是流式高性能路径：当前 `talker2codec_async_chunk` 把 `chunk_frames` 固定成极大值，相当于只在 finish 时发给 codec；而且 payload 是累计快照和 Python list 风格，容易带来复制和序列化开销。
4. vocoder 侧还没有用上 MOSS-Audio-Tokenizer-v2 的原生 streaming session、slot 管理、coalesced batch 和有状态 CUDA Graph；同时 `moss_tts_local.yaml` 中 Stage1 `max_num_seqs: 1` 且 `enforce_eager: true`，无法形成 Qwen3/SGLang 那种稳定 shape 的 batch/graph decode。

因此后续路线应按依赖顺序推进：

1. 先建立分段 profile 和正确的 benchmark 基线。
2. 先修掉当前非流式路径中的 O(T^2) 状态累积、CPU sync、payload 膨胀和参考音频重复编码。
3. 再实现 MOSS Local 专用 GPU state pool + batched local transformer。
4. 在状态地址稳定后做 frame decode CUDA Graph 和 GPU seeded sampler。
5. 最后恢复并完善 MOSS-Audio-Tokenizer-v2 streaming vocoder，配合 Stage1 batch 和 vocoder CUDA Graph。

## Pipeline 对比

### Qwen3-Omni / Qwen3-TTS 已有的高性能形态

Qwen3 系列的优化路径已经比较成熟：

- Stage 间 async chunk 和客户端 streaming 是连通的，首包不必等完整音频生成。
- Talker 侧把 decode 预处理做了批量化，例如 trailing_text offset/compact、常量 embedding buffer、mask 预计算、O(1) request index。
- Code2Wav 侧把 connector chunk 和内部 decode window 解耦，支持 `codec_chunk_frames`、`initial_codec_chunk_frames`、`decode_chunk_frames`、`decode_left_context_frames`。
- Code2Wav 对稳定 shape 做 CUDA Graph，按 `(batch, frames)` bucket 捕获，hit 时减少大量 launch overhead。
- 部分模型还把多码本状态迁移到 GPU tensor，避免 per-request Python dict 成为高并发瓶颈。

这些优化对 Qwen3-TTS 的收益很明确：在 `vllm-omni-all-tts-optimization.md` 中，Qwen3-TTS 的高并发吞吐和 RTF 主要来自 Stage0 decode 预处理 batch、Code2Wav graph、async chunk、hot-path 清理。

### SGLang-Omni MOSS Local 的目标形态

SGLang-Omni 针对 MOSS Local 的设计更贴近这个模型本身：

- 预处理阶段：参考音频编码做 batch、LRU cache、single-flight 去重。
- AR 阶段：
  - Qwen3 backbone 走常规 graph / cache 路径。
  - local transformer 的完整帧内微循环独立 capture 成 frame decode graph。
  - `MossTTSLocalDecodeStatePool` 预分配固定 GPU row，持有 feedback embedding、sampling 参数、seed、step、repetition 历史。
  - 下一帧输入 embedding 在 graph 内计算并写入 pool，下一步 backbone 通过 `_decode_input_embedding` staging table 读取。
  - sampler 使用 GPU seeded branchless 版本，避免 CPU 控制流并保持 batch-invariant。
- vocoder 阶段：
  - 使用持久化 `codec.streaming()` session。
  - 每个请求分配 streaming slot，多个 slot coalesced decode。
  - 首包小阈值、稳态大阈值。
  - 有状态 vocoder CUDA Graph 修复 KV/cache buffer 地址稳定性后按帧数 capture。

SGLang 文档里强调：MOSS Local 单帧除了 1 次 backbone，还包含 local transformer 的二元决策、12 个 codebook head、12 次采样、12 次 embedding 反馈、11 次 local transformer step。这个帧内微循环计算量不大，但 kernel launch 数量多，不 graph 化时 launch overhead 会主导。

### vLLM-Omni 当前 MOSS Local 的形态

当前 vLLM-Omni MOSS Local 主要还处在功能正确的初始 serving 形态：

- Stage0 `MossTTSLocalTalkerForGeneration.forward` 只调用 Qwen3 backbone，这是合理的。MOSS Local 的 audio code 不是由 backbone logits 直接输出，而是在 `make_omni_output` 中用 backbone hidden state 驱动 local transformer 生成下一帧 codes。
- `preprocess` 中 prefill 会把 text embedding 和 reference audio code embedding 相加；decode 时把 forced text slot embedding 与上一帧 audio code embedding 相加。
- `make_omni_output` 中逐请求处理 hidden state，调用 `local_transformer.generate_frame` 生成一帧 12-codebook audio codes。
- `local_transformer.generate_frame` 当前每个 codebook 重新 `_forward_prefix` 一次，使用 PyTorch eager attention/MLP/head/sampling。
- `talker2codec_async_chunk` 当前将 `chunk_frames` 固定为 `1 << 30`，意味着 streaming chunk 实际被禁用，只在请求结束时把所有 codes 发到 Stage1。
- Stage1 `MossTTSCodecDecoder.forward` 当前对每个请求调用 `batch_decode`；如果 `enforce_eager` 为 false 才会启用 codec CUDA Graph wrapper，但当前部署配置中 Stage1 是 `enforce_eager: true`。
- `moss_tts_local.yaml` 里 Stage1 `max_num_seqs: 1`，这使 vocoder 没有 batch 能力，也容易让未来 streaming slot 与 stage capacity 不匹配。

## 差距拆解

### 1. 模型结构的天然成本

MOSS Local 的目标输出是 48kHz stereo，使用 MOSS-Audio-Tokenizer-v2。这个 codec 是 MOSS 质量优势的一部分，但也是成本来源。相比一些 24kHz mono 或更轻的 DAC/Code2Wav 路径：

- 输出采样率和声道数更高，单位音频秒的 decoder 工作更多。
- 12 个 RVQ codebook 每帧都要顺序采样，AR 帧内依赖更强。
- 参考音频编码需要完整 codec encoder，voice_clone 场景的冷请求成本明显。

这意味着即便 infra 完全优化，MOSS Local 也未必能无条件达到 Qwen3-Omni 的所有数值。优化目标应该拆成两个：

- TTFP：通过真正流式 vocoder 和 async chunk，可以接近 Qwen3 的用户体验目标。
- RTF：通过 frame graph、vocoder graph、batch/coalescing、缓存减少大量 serving overhead，但最终下限受 codec 和 local decode 结构影响。

### 2. AR backbone 不是当前最大问题，local frame decode 才是

`MossTTSLocalTalkerForGeneration.forward` 只跑 `self.model(...)`，也就是 Qwen3 backbone。这不是漏掉 local transformer，而是 MOSS Local 架构的正确拆分：

- backbone 负责把文本、参考音频、上一帧 feedback embedding 转成当前帧 hidden state。
- local transformer 用这个 hidden state 生成一帧 audio codes。
- `compute_logits` 只合成继续生成 slot token 或结束 token，不负责 audio code logits。

当前瓶颈在于 local transformer 被放在 `make_omni_output` 的 Python loop 里逐请求 eager 执行：

- 每个 request 独立调用 `generate_frame`，batch 维度基本没有利用。
- `history_per_codebook` 通过 `acc_for_hist[-rep_window:].long().cpu().tolist()` 构造，有 GPU 到 CPU 同步。
- 每帧 `torch.cat([acc, new_codes])` 更新 accumulated codes，长音频会形成 O(T^2) copy。
- `generate_frame` 中每个 codebook 都调用 `_forward_prefix`，最多 12 次小 shape transformer forward；这些 kernel 很小，launch overhead 明显。
- sampling 逻辑复用通用 `_sample_token`，没有 MOSS Local 专用的 seeded branchless GPU sampler，也不适合直接 graph。

这解释了为什么当前单并发 RTF 仍然偏高：即使 batch=1，local decode 的 kernel launch 和 Python overhead 也无法靠 vLLM backbone 优化覆盖。

### 3. State 管理还不是 graph-friendly

当前状态主要存在 `info_dict` 里的 Python dict 和临时 tensor：

- `audio_state`: `is_stopping`、`step`、`max_new_frames`
- `audio_codes.current`: 上一帧 codes
- `audio_codes.accumulated`: 全量历史 codes
- `hidden_states.last`: 上一步 hidden

这种形式适合先打通功能，但不适合高性能：

- batch reorder / finish / abort 时，Python dict 和 scheduler row 的一致性难以保证。
- CUDA Graph 需要固定地址输入输出，Python dict 和动态 tensor allocation 不满足。
- repetition penalty 历史每步从 accumulated tensor 转 CPU list，既慢又阻断 graph。
- 下一帧 feedback embedding 每次在 preprocess 中由 text/audio embedding 临时计算，无法像 SGLang 那样在 frame graph 内直接写入 state pool。

Qwen3-TTS、Higgs、SGLang MOSS 的共同经验是：多码本 TTS 的 decode 状态必须逐步迁移到 GPU-resident tensor pool，Python 只保留 request-id 到 row 的映射和少量控制面信息。

### 4. Stage0 到 Stage1 数据路径没有真正流式

当前 `talker2codec_async_chunk` 中虽然接口名字是 async chunk，但有一段关键逻辑：

```python
chunk_frames: int = 1 << 30
```

这会让请求在生成完成前都不向 Stage1 发有效 chunk。结果是：

- 客户端 `stream=true` 也只能在完整 Stage0 + Stage1 后拿到第一包。
- benchmark 中 AUDIO_TTFP 接近 E2E 是预期现象。
- async chunk 没法和 vocoder compute overlap。
- RTF 也会受影响，因为 codec 只能做整段 decode，无法和 AR 并行。

此外，当前 Stage0 传出的 codes 是 accumulated snapshot，Stage input processor 再通过 `prev_t` 去重尾部。这能保证正确性，但会产生两个问题：

- Stage0 每步向 connector 暴露越来越大的 tensor，虽然下游去重，但上游 copy/对象生命周期仍然膨胀。
- 传给 Stage1 时最终被 transpose/flatten 成 list，Python int 序列化和 GC 成本在高并发下会放大。

后续应改成 delta payload：每帧或每 chunk 只发送新增 `[T_delta, n_vq]` tensor，并尽量保持 tensor payload 到 Stage1。

### 5. Vocoder 侧还没有利用 MOSS v2 的核心能力

MOSS-Audio-Tokenizer-v2 的一个重要特性是原生 streaming decoder。SGLang 利用了 `codec.streaming(batch_size)` 形成持久 session，并用 slot 管理多个请求。

vLLM 当前 Stage1 的实现仍是：

- 收到 flat ids。
- reshape 成 `(n_vq, T)`。
- 调用 `batch_decode(codes_list=[codes_nq_t])`。
- 输出 waveform。

这条路径的问题：

- 没有持久 streaming state，无法低 TTFP 增量输出。
- 没有 stream/offline slot 分离，流式和非流式请求会互相影响。
- 没有 coalesced step，多个请求不能共享一次 codec forward。
- 没有有状态 vocoder CUDA Graph。
- Stage1 `max_num_seqs: 1` 使 batch decode 和 slot 池都没有空间。
- Stage1 `enforce_eager: true` 关闭了已有的 codec graph wrapper。

即使先不做 streaming，非流式 codec 也可以做 batch decode、chunked decode graph、Stage1 capacity 调整；但要达到 TTFP 目标，必须做原生 streaming session。

### 6. 参考音频编码和缓存还没成为一等公民

voice_clone 的参考音频编码是明显冷启动成本。SGLang 文档中提到单条参考编码可达约 250ms，并且通过缓存扩容在 SeedTTS 工作负载上拿到显著吞吐提升。

vLLM 当前已经有 `reference_encoder.py` 负责解析 URL/file/base64 并编码 reference codes，但后续需要补上：

- 内容寻址 cache，而不是只依赖路径。
- single-flight，避免并发请求同一参考音频时重复编码。
- 批量 reference encoding。
- 统一缓存 dtype/device 契约，返回 clone，避免调用方修改共享缓存。
- cache metrics：hit/miss/single-flight/eviction。

这对单个 warm request 的 RTF 不一定有直接收益，但对 voice_clone benchmark 和生产固定音色池非常关键。

### 7. 配置和内存预算未按 MOSS Local 特性调优

当前 `moss_tts_local.yaml` 中：

- Stage0 `max_num_seqs: 8`。
- Stage1 `max_num_seqs: 1`。
- Stage1 `gpu_memory_utilization: 0.15`。
- Stage1 `enforce_eager: true`。
- connector 中配置了 `codec_streaming: true` 和 `codec_chunk_frames: 15`，但 processor 实际把 chunk 固定成 full flush。

这会造成配置语义和实际行为不一致。后续需要把配置拆成明确模式：

- `offline_fast`: 非流式、整段或大 chunk、最大吞吐。
- `streaming_low_latency`: 小 initial chunk、稳态 chunk、vocoder session slot。
- `streaming_balanced`: 根据负载自适应增大 chunk。

同时要为同卡共置预留 codec activation / streaming state 内存。SGLang 的经验是显式 `codec_mem_reserve` 可以避免 AR KV cache 抢占 codec 运行余量。

## 优化路线

### Phase 0：建立可解释的 profile 基线

目标：先知道 0.5-0.6 RTF 里每部分占多少，避免盲目 graph。

需要补充的统计：

- Stage0 Qwen3 backbone decode time。
- `make_omni_output` 总耗时。
- local transformer 每帧耗时、每 codebook 平均耗时。
- sampling 耗时。
- `cpu().tolist()` repetition history 耗时。
- `torch.cat` accumulated codes 耗时和累计 bytes。
- Stage0 -> Stage1 payload bytes、payload 类型、每请求 chunk 数。
- Stage1 codec decode time、输入 frames、输出 samples、sample rate/channels。
- ref audio resolve/encode/cache 时间。
- CUDA Graph hit/fallback 计数。
- GPU utilization 和 kernel launch 统计。

建议 benchmark 分组：

- warmup 后再测，剔除首次加载 codec weight / graph warmup。
- non-streaming 与 streaming 分开测。
- voice_clone 与 fixed speaker/base TTS 分开测。
- `repetition_penalty=1.0` 和 `1.1` 分开测，因为前者更适合 graph。
- c=1/2/4/8/16 分开测，避免只看单并发。
- 记录真实 audio duration，避免 48k stereo PCM 解析错误导致 RTF 虚高或虚低。

验收：

- 单请求日志能分解 `E2E = ref encode + Stage0 backbone + local frame decode + connector + Stage1 codec + postprocess`。
- benchmark 输出的 audio duration 与 wav/pcm 真实时长一致。
- 所有优化 PR 都能用同一套指标比较。

### Phase 1：低风险修掉当前非流式路径的明显 overhead

目标：不改变模型数值路径，先减少 Python/tensor 分配和 payload 膨胀。

#### 1.1 移除 accumulated codes 的 O(T^2) concat

当前每帧：

```python
updated_acc = torch.cat([acc.to(new_codes.device), new_codes.unsqueeze(0)], dim=0)
```

改法：

- 请求状态中维护 list of frame tensors，finish 时一次 stack；或维护预分配增长 buffer。
- `make_omni_output` 返回新增 delta，而不是全量 accumulated snapshot。
- 若为了兼容旧 processor 仍需要 snapshot，可只在非 streaming final 阶段 materialize。

收益：

- 长文本不再随帧数产生二次 copy。
- Stage0 -> Stage1 connector 不再处理越来越大的快照。

风险：

- stop frame 不应 append。
- max_new_frames 边界不能多发最后一帧。

#### 1.2 repetition penalty 的 fast path

当前默认 repetition penalty 是 1.1，因此每帧都构造 CPU history。短期可以：

- 当 `repetition_penalty == 1.0` 时完全跳过 history。
- 把 history 维护为 GPU `[n_vq, vocab]` presence 或 rolling buffer，不每帧 `.cpu().tolist()`。
- graph 初期可以规定 `repetition_penalty != 1.0` 走 eager fallback，与 SGLang 一致。

收益：

- 减少 D2H sync。
- 为 frame graph 准备条件。

#### 1.3 Tensor payload 替代 list payload

当前 Stage1 input 最终走 flat list。建议：

- Stage0 输出 `codes.audio` 保持 `[T, n_vq]` 或 `[n_vq, T]` tensor。
- connector 传 tensor payload，Stage1 直接消费 tensor。
- 只有 fallback 兼容路径才 `.tolist()`。

收益：

- 高并发下减少 Python int allocation 和 GC。
- 为 coalesced codec batch 准备统一 shape。

#### 1.4 参考音频 cache

先实现单机 LRU + single-flight：

- key 使用内容 hash 或强一致 stat+sentinel。
- value 用 CPU int32 存储。
- 返回 clone 到下游所需 dtype/device。
- 增加 cache metrics。

收益：

- voice_clone 固定说话人池吞吐提升。
- benchmark 更稳定，减少 ref encode 噪声。

#### 1.5 配置语义修正

当前配置声明 `codec_streaming: true`，但实际 processor full flush。建议先明确：

- 如果 streaming 未完成，将配置改为 `codec_streaming: false` 或在文档中标注当前为 non-streaming。
- Stage1 `max_num_seqs` 暂时保持 1 也可以，但不要让用户误以为 streaming 生效。

验收目标：

- non-streaming 正确性不回退。
- c=1 RTF 有可测改善。
- 长文本 RTF 不随生成帧数异常增长。

### Phase 2：GPU state pool + batched local transformer

目标：把 MOSS Local 的帧内 decode 从 per-request Python eager 迁移到 batched GPU hot path。

#### 2.1 DecodeStatePool

参考 SGLang `MossTTSLocalDecodeStatePool`，在 vLLM 模型内维护固定 GPU buffer：

- `feedback_embeds: [P, hidden]`
- `current_codes: [P, n_vq]`
- `generated_codes` 或 rolling history buffer
- `generation_steps: [P]`
- `sampling_steps: [P]`
- `seeds: [P]`
- temperature/top_p/top_k/repetition penalty
- `audio_token_presence: [P, n_vq, vocab]` 或 rolling history
- `active_mask` / `stop_mask`

CPU 侧只保留：

- request_id -> row
- free rows
- row lifecycle hooks
- finish/abort/preempt cleanup

关键点：

- row 必须用 request_id 管理，不能默认 scheduler batch index 稳定。
- vLLM scheduler 可能 reorder/shrink/finish，state pool 需要在每步根据 active request 显式 gather row。
- cleanup 必须覆盖 normal finish、error、client abort。

#### 2.2 Decode input embedding staging

SGLang 的做法是维护 `_decode_input_embedding`：

- 每步 decode 前，把 pool 中 active rows 的 `feedback_embeds` copy 到 embedding table 前 `B` 行。
- 把 `input_ids` 改成 `[0, 1, ..., B-1]`。
- backbone 看到的是普通 embedding lookup，模型特定的 13 通道融合被隔离。

vLLM 可以复用这个思路：

- prefill 仍使用当前 text+ref audio embedding 融合。
- decode 步使用 staging table 输入上一帧反馈 embedding。
- frame decode graph 内负责写下一帧 feedback embedding。

收益：

- decode preprocess 从每请求构造 embedding 变成一次 batched copy/gather。
- backbone graph 更容易捕获，因为 input embedding 形状和地址更稳定。

#### 2.3 Batched local transformer eager

先不要一步到 graph，先实现 `generate_frame_batch`：

- 输入：`last_h: [B, H]`、state pool rows、sampling params。
- 输出：`should_continue: [B]`、`codes: [B, n_vq]`、`feedback_embeds: [B, H]`。
- 12 个 codebook 的顺序依赖保留，但每个 codebook 在 batch 维上一起跑。
- local transformer 改成增量 KV 或至少 batched `_forward_prefix`。
- feedback embedding 在 GPU 上一次性 sum。

收益：

- 高并发首先受益。
- c=1 也会为后续 graph 减少 Python loop。

风险：

- batch-invariant sampling：同一个 request 在不同 batch 组合下应尽量生成一致结果。
- stop request 的 code 不应污染下一帧 feedback。
- repetition penalty 如果还没 GPU 化，先 fallback。

验收目标：

- 与当前 eager 单请求在固定 seed 下尽量一致；如 sampler 数值不同，需要单独记录并做音质/WER 验证。
- c=4/8 local decode time 明显下降。
- c=1 没有显著回退。

### Phase 3：Frame decode CUDA Graph 和 GPU sampler

目标：消除 local transformer 帧内大量小 kernel launch。

#### 3.1 Frame graph bucket

按 batch size bucket 捕获：

- `[1, 2, 4, 8, 16]` 起步。
- 实际 batch padding 到 bucket，输出裁剪。
- 超过最大 bucket 或带不支持参数时 eager fallback。

Graph 覆盖范围：

- local transformer step 0。
- binary continue/stop head。
- binary sampler。
- 12 个 codebook head。
- 12 次 codebook sampler。
- code embedding lookup。
- feedback embedding sum。
- 写 state pool。

不建议一开始把 backbone 和 frame decode 放进同一个 graph。SGLang 也是 backbone graph 与 frame graph 分开捕获，原因是两者 shape、状态和 fallback 条件不同。

#### 3.2 GPU seeded branchless sampler

移植或参考 SGLang `sample_seeded_branchless`：

- RNG 由 `(request_seed, frame_step, channel_index)` 派生。
- temperature/top_k/top_p 在 GPU 上完成。
- 避免 host 分支。
- 支持 compile。
- batch composition 不改变单请求采样序列。

Repetition penalty 策略：

- 第一版 graph 仅支持 `repetition_penalty == 1.0`。
- `!= 1.0` fallback eager。
- 后续把 `audio_token_presence` 迁移到 state pool 后再 graph 化。

#### 3.3 Local transformer KV / prefix 优化

当前 `generate_frame` 每个 codebook 都重新 `_forward_prefix(embeds[:, :k])`。数学上因为只读最后位置，可以改成增量 local KV：

- position 0 输入 backbone hidden。
- 每个 codebook 采样后，将 embedding 作为下一个 position。
- local attention KV cache 在帧内复用，帧间 reset。
- 对 batch bucket 预分配固定 KV buffer。

收益：

- 减少重复计算。
- 更适合 graph capture。

风险：

- RoPE 是 GPT-J/interleaved 风格，不能误用 vLLM neox-style RoPE。
- 当前实现强调数值对齐官方 gpt2 decoder，改 KV 需要严格比对 logits/codes。

验收目标：

- frame graph hit rate 可观。
- c=1 RTF 明显下降，目标先从 0.5-0.6 降到 0.35-0.45 区间。
- graph fallback 有日志和计数。

### Phase 4：恢复 MOSS v2 streaming vocoder

目标：让 TTFP 不再等完整 Stage0，并让 codec 与 AR overlap。

#### 4.1 持久 streaming session

参考 SGLang `_CodecStreamSession`：

- 初始化时创建 `codec.streaming(batch_size=total_slots)`。
- slots 分为 stream slots 和 offline slots。
- 每个 request acquire/release slot。
- slot state 包括 pending frames、threshold、emitted_any、finished。
- request finish/error/abort 必须释放 slot。

需要避免之前遇到的 streaming slots exhausted：

- slot 数必须和 Stage1 `max_num_seqs` / vocoder session capacity 对齐。
- acquire 失败不能直接 crash；应 fallback offline 或排队。
- release 必须在所有退出路径执行。

#### 4.2 双阈值和 coalesced step

建议初始参数：

- `initial_codec_chunk_frames`: 3-5 帧，MOSS 12.5Hz 下约 240-400ms。
- `codec_chunk_frames`: 15-25 帧，约 1.2-2s。
- `max_step_frames`: 100。

策略：

- 首包用小阈值，降低 TTFP。
- 稳态用大阈值，提升 throughput 和连续性。
- 任一 slot 到期时，允许其他已积累足够帧的 slot join。
- 单次 step 中所有参与 slot 使用相同 T，exec mask 控制有效 slot。

#### 4.3 Stage input processor 真正 chunk 化

恢复 `chunk_frames` 为配置值，但不能回到之前不稳定实现。需要：

- Stage0 输出 delta codes。
- processor 累积 pending frames。
- 到 `initial_codec_chunk_frames` 或 `codec_chunk_frames` 后发送 tensor payload。
- meta 中携带 `left_context_size`、`finished`、`streaming`、`chunk_index`。
- finish 时 flush 剩余帧。

MOSS v2 原生 streaming 如果能保证与 offline bit-level 或近似一致，就不需要 Qwen3-TTS 那种 overlap decode window；但仍要做流式/非流式一致性验证。

#### 4.4 Vocoder CUDA Graph

MOSS v2 streaming decoder 是有状态的，graph 的关键是状态 buffer 地址稳定：

- attention KV cache 不能每步重新赋值新 tensor。
- RoPE position/state 更新要原地写。
- capture 按 T bucket。
- batch 宽度固定为 total slots。
- exec mask 控制参与 slot。
- OOM 或 capture 失败 fallback eager。

短 chunk 下 vocoder graph 收益最大，因为 codec 每步 kernel 多、T 小，launch overhead 占比高。

验收目标：

- streaming 与 offline 音质/WER 无明显回归。
- c=1 TTFP 降到 300-600ms 以内，再继续调 initial frames 和 pipeline overlap。
- Stage1 slot exhaustion 不再导致请求失败。
- streaming RTF 与 non-streaming 的 tradeoff 可解释。

### Phase 5：调度、内存和跨阶段优化

目标：把单点优化变成稳定 serving 系统。

#### 5.1 Stage1 batching 和 capacity

当前 Stage1 `max_num_seqs: 1` 不适合 streaming vocoder。需要：

- 根据 GPU memory 设定 `max_num_seqs`，至少覆盖目标 stream slots。
- 非流式也允许 batch decode 多请求。
- 对不同 T 做 bucket/padding。
- codec decode 太重时考虑 Stage1 独占 GPU 或与 Stage0 分卡。

#### 5.2 显存预算

同卡共置时需要显式预算：

- Qwen3 backbone weights + KV cache。
- local transformer/state pool。
- codec weights。
- codec streaming KV/cache/activation。
- graph static buffers。

建议引入 MOSS Local 专用配置项：

- `codec_mem_reserve_gb` 或 fraction。
- `state_pool_max_rows`。
- `stream_slots` / `offline_slots`。
- `frame_graph_capture_batch_sizes`。
- `vocoder_graph_capture_frames`。

#### 5.3 Backpressure 和优先级

流式 vocoder 会更频繁抢 GPU。高负载下需要避免 Stage1 过度打断 Stage0：

- vocoder coalesced step 合并更多 slot。
- 负载高时自动增大 steady chunk。
- 首包优先，但首包之后转吞吐优先。
- 如果 Stage1 排队过长，Stage0 可适当积累更大 chunk。

#### 5.4 Graph 和 fallback 可观测性

所有 graph 都需要指标：

- frame graph hit/fallback，fallback reason。
- vocoder graph hit/fallback，fallback reason。
- sampler compile enabled/hit。
- state pool row acquire/release/leak。
- streaming slot acquire/release/fallback offline。
- cache hit/miss/single-flight。

没有这些指标，后续性能回归很难定位。

## 优先级建议

### P0：必须先做

- 分段 profile 和 benchmark 修正。
- 修掉 accumulated `torch.cat` O(T^2)。
- repetition penalty fast path / 避免无必要 CPU history。
- delta tensor payload。
- 明确当前配置是否 streaming，避免误测 TTFP。
- 参考音频 cache + single-flight。

原因：这些改动风险较低，且能让后续 profile 更干净。

### P1：RTF 主力优化

- GPU DecodeStatePool。
- decode input embedding staging。
- batched local transformer eager。
- GPU sampler。
- frame decode CUDA Graph。

原因：当前 c=1 RTF 的主要 gap 很可能在 local frame decode 的 eager 小 kernel 和 Python 路径上。SGLang 对 MOSS Local 的优化也把 frame graph 作为核心。

### P2：TTFP 主力优化

- MOSS v2 streaming session。
- streaming slot lifecycle。
- initial/steady chunk。
- Stage0 delta chunk。
- vocoder coalesced step。
- vocoder CUDA Graph。

原因：没有这部分，TTFP 永远接近 E2E。仅把客户端请求设为 `stream=true` 不会有实际首包收益。

### P3：生产吞吐和稳定性

- Stage1 capacity 和内存预算。
- backpressure。
- adaptive chunk。
- 多 GPU placement。
- 完整可观测性。

原因：这些决定高并发下是否稳定，而不是单请求 demo 是否能跑通。

## 预期收益和现实边界

粗略预期如下，具体需要 Phase 0 profile 验证：

- 移除 O(T^2) concat、CPU history、payload 膨胀：对长文本和高并发更明显，c=1 也会有小幅收益。
- 参考音频 cache：对固定 speaker / SeedTTS 这类重复参考工作负载收益明显，对完全冷随机参考收益有限。
- batched local transformer：高并发收益明显，c=1 主要是为 graph 铺路。
- frame decode CUDA Graph：对 c=1 也应有明显收益，因为 MOSS Local 帧内有大量小 kernel。
- vocoder streaming：主要降低 TTFP，并通过 AR/vocoder overlap 改善 E2E；但高并发下可能牺牲部分吞吐。
- vocoder CUDA Graph：对短 streaming chunk 收益最大，对整段大 T decode 收益较小。

需要保持现实预期：

- Qwen3-Omni/Qwen3-TTS 的 RTF 数字不能直接作为 MOSS Local 的硬目标，因为 MOSS v2 48k stereo codec 和 12-codebook local decode 更重。
- 但是当前 vLLM MOSS Local 还有大量 serving overhead 未优化，RTF 从 0.5-0.6 继续下降是合理目标。
- TTFP 要接近 200ms，需要 initial chunk 很小、Stage0 首帧很快、Stage1 streaming session 已 warm、codec graph 可用、无排队。MOSS 12.5Hz 下 1 帧是 80ms，初始 3-5 帧天然是 240-400ms 音频上下文；若 codec 需要至少 5 帧稳定首包，200ms 可能偏激进，需要用音质换延迟或验证 codec 是否支持更小首块。

## 风险清单

- Stop 语义：binary head 选择 stop 时，当前帧 codes 不应追加到 accumulated，也不应送 codec。
- Batch-invariant sampling：同 request 不应因为 batch 组合变化产生不可解释差异，至少 seed 语义要稳定。
- Scheduler reorder/shrink：state pool row 必须 request-id keyed，不能依赖 batch index。
- Abort cleanup：client 断开、异常、Stage1 失败都必须释放 state row 和 streaming slot。
- Graph capture：shape、地址、dtype、control flow 必须稳定；不满足条件要 eager fallback。
- Repetition penalty：如果保留默认 1.1，frame graph 初期可能大量 fallback；要么先调默认到 1.0 做性能模式，要么实现 GPU history。
- Codec streaming consistency：streaming/offline 应做 WER、SIM、波形长度、chunk continuity 验证。
- Sample rate/channels：MOSS Local 是 48kHz stereo，benchmark 的 PCM 解析必须匹配，否则 RTF/audio duration 会错。
- Weight loading warmup：首次请求加载 codec weight 或 graph warmup 会污染 TTFP，需要服务启动时 warmup。

## 建议里程碑

### M1：可解释 baseline

- 增加 MOSS Local 分段 timing。
- benchmark 能区分 warm/cold、stream/non-stream。
- 输出真实 audio duration。
- 产出 c=1/4/8 的 baseline 表。

### M2：non-streaming fast path

- 移除 accumulated O(T^2)。
- delta tensor payload。
- ref audio cache。
- repetition penalty fast path。
- 配置语义修正。

目标：非流式正确性保持，c=1 RTF 有明确下降，长文本不劣化。

### M3：batched local decode

- DecodeStatePool。
- embedding staging。
- batched local transformer eager。
- GPU feedback embedding。

目标：高并发 local decode time 下降，state lifecycle 稳定。

### M4：frame graph

- GPU seeded sampler。
- frame graph bucket。
- graph/fallback metrics。
- repetition penalty fallback。

目标：c=1 RTF 明显下降，开始接近 0.3-0.4 区间；具体数值以硬件和数据集为准。

### M5：streaming vocoder

- 持久 codec streaming session。
- slot 管理。
- initial/steady chunk。
- coalesced step。
- Stage1 capacity 调整。
- vocoder graph。

目标：c=1 TTFP 降到亚秒级，再向 300-600ms 调优；在音质不回退前提下探索 200ms 附近可行性。

## 代码入口建议

优先修改和审查这些位置：

- Stage0 MOSS Local：
  - `vllm_omni/model_executor/models/moss_tts/modeling_moss_tts_talker.py`
  - `MossTTSLocalTalkerForGeneration.preprocess`
  - `MossTTSLocalTalkerForGeneration.make_omni_output`
- Local transformer：
  - `vllm_omni/model_executor/models/moss_tts/modeling_moss_tts_local_depth.py`
  - 新增 batched/incremental local decode 和 sampler。
- Stage connector：
  - `vllm_omni/model_executor/stage_input_processors/moss_tts.py`
  - 改 accumulated snapshot 为 delta tensor chunk。
- Stage1 codec：
  - `vllm_omni/model_executor/models/moss_tts/modeling_moss_tts_codec.py`
  - 增加 streaming session、slot、coalesced decode、graph。
- 部署配置：
  - `vllm_omni/deploy/moss_tts_local.yaml`
  - 拆分 offline/streaming 配置，调整 Stage1 capacity。
- 参考音频：
  - `vllm_omni/model_executor/models/moss_tts/reference_encoder.py`
  - 增加 cache/single-flight/batch encode。
- 可复用参考：
  - Qwen3-TTS `qwen3_tts_code2wav.py` 的 decode graph config/bucket 思路。
  - Higgs 的 GPU state 管理和 request-id keyed lifecycle。
  - Fish 的 tensor payload 和 bounded codec batching。
  - SGLang MOSS Local 的 state pool、frame sampler、streaming vocoder。

## 最终判断

当前 MOSS Local 的主要 gap 不只是 batch、graph、streaming 三个开关没打开，而是 MOSS Local 这类模型需要一套更专用的 serving 数据面：

- 多通道帧向量不能一直用 Python dict 和临时 embedding 构造。
- local transformer 帧内微循环必须 batch/graph 化。
- feedback embedding 必须变成 GPU 常驻状态。
- codec 必须用 MOSS v2 原生 streaming session，而不是把 streaming 模拟成小段 `batch_decode`。
- Stage connector 必须传 delta tensor，而不是累计快照/list。

建议先用 Phase 0/M2 清理当前路径，然后集中投入 M3/M4。RTF 的最大收益大概率来自 M3/M4；TTFP 的最大收益一定来自 M5。只有这些都完成后，MOSS Local 才能和 Qwen3-Omni/Qwen3-TTS 的成熟优化栈做公平比较。
