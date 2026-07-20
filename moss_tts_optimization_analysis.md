# vLLM-Omni MOSS-TTS 优化分析

日期：2026-06-22

## 1. 背景与信息源

本文结合 sglang-omni 的 MOSS-TTS 性能路线图、相关 issue/PR，以及本地 `vllm-omni`/`sglang-omni` 代码，分析 vLLM-Omni MOSS-TTS 当前还可以做的优化。

主要参考：

- sglang-omni issue [#637: MOSS TTS Performance Optimization Roadmap](https://github.com/sgl-project/sglang-omni/issues/637)
- sglang-omni PR [#609: Add MOSS-TTS support](https://github.com/sgl-project/sglang-omni/pull/609)
- sglang-omni issue [#730: decouple preprocessing and reference audio encoding](https://github.com/sgl-project/sglang-omni/issues/730)
- sglang-omni issue [#731: LRU cache for reference audio encoder](https://github.com/sgl-project/sglang-omni/issues/731)
- sglang-omni issue [#734: validate and optimize async decode path](https://github.com/sgl-project/sglang-omni/issues/734)
- sglang-omni issue [#738: batched vocoder decode](https://github.com/sgl-project/sglang-omni/issues/738)
- 本地代码：
  - `vllm-omni/vllm_omni/model_executor/models/moss_tts/`
  - `vllm-omni/vllm_omni/model_executor/stage_input_processors/moss_tts.py`
  - `vllm-omni/vllm_omni/entrypoints/openai/serving_speech.py`
  - `vllm-omni/vllm_omni/deploy/moss_tts.yaml`
  - `sglang-omni/sglang_omni/models/moss_tts/`

## 2. sglang-omni 路线图要点

sglang-omni 的 #637 把 MOSS-TTS 优化拆成几条主线：

1. Pipeline：把 `preprocessing -> AR -> vocoder` 拆成更清晰的 4-stage：`preprocessing -> audio_encoder -> AR -> vocoder`，为参考音频 GPU encode、batch encode、cache 创造边界。
2. Encoder：参考音频 LRU cache、batched audio encode、encoder `torch.compile`。
3. AR step：async decode、state-pool、CUDA Graph、`torch.compile`、per-frame launch orchestration vectorization。
4. Vocoder：batched vocoder decode、vocoder CUDA Graph、streaming vocoder。
5. CI/benchmark：用 Seed-TTS 等基准持续看 RTF、吞吐、WER/CER、speaker similarity。

#609 暴露过几个关键瓶颈和修复经验：

- 初版 eager baseline 在 concurrency 16 下 RTF 很高。
- per-row/per-codebook Python sampling loop 曾经是 decode step 热点，后续通过 batched tensor ops 明显降低 RTF。
- preprocessing 串行会让 AR engine 饿到 batch=1。
- audio logits processor 使用 text vocab 会导致 padding column 泄露，被采样成 out-of-range audio code，影响 WER/CER 尾部。

这些经验对 vLLM-Omni 很有参考价值，但不能直接照搬，因为 vLLM-Omni 当前管线结构不一样。

## 3. vLLM-Omni 当前实现概况

### 3.1 Pipeline

本地 vLLM-Omni 的 full MOSS-TTS family 是 2-stage：

```text
stage 0: moss_tts       LLM_AR         Qwen3 backbone + audio heads
stage 1: moss_tts_codec LLM_GENERATION MOSS Audio Tokenizer decode
```

对应文件：

- `vllm-omni/vllm_omni/model_executor/models/moss_tts/pipeline.py`
- `vllm-omni/vllm_omni/deploy/moss_tts.yaml`
- `vllm-omni/vllm_omni/deploy/moss_tts_realtime.yaml`

与 sglang-omni 不同，vLLM-Omni 的参考音频编码目前发生在 OpenAI serving 构参阶段，而不是独立 pipeline stage。

### 3.2 参考音频编码与缓存

`vllm-omni/vllm_omni/model_executor/models/moss_tts/reference_encoder.py` 已经实现：

- named voice cache：按 `voice_name + created_at + model_type` 命中。
- anonymous reference cache：按 `ref_str` 的 SHA1 命名。
- cold encode 使用 `asyncio.to_thread()`，避免阻塞 event loop。
- cache value 是 CPU 上的 reference RVQ codes。

这已经覆盖 sglang-omni #731 的一部分目标。不过需要注意：anonymous cache 当前是基于 `ref_str`，如果同一音频通过不同 URL、不同本地路径、不同 base64 表达传入，可能不会命中真正的内容级缓存。

### 3.3 AR talker

Delay talker 位于：

- `vllm-omni/vllm_omni/model_executor/models/moss_tts/modeling_moss_tts_talker.py`

已有优化：

- audio heads 被 stack 成 `_stacked_audio_head_w`，采样时用单次 batched matmul 替代 `n_vq` 个 `nn.Linear`。
- audio embeddings 被 stack 成 `_stacked_audio_emb_w`，prefill/decode 时减少 per-codebook loop。
- 每请求维护 `audio_state`，并在 `compute_logits()` 中做 delay-pattern 强制 token、audio/text mask、`max_new_frames` cap。

仍然存在的热点：

- `make_omni_output()` 仍按 request 循环处理，每个 request 取最后 hidden state 后调用 `_sample_audio_codes()`。
- 累积 codes 使用逐步 `torch.cat`，长音频会产生重复分配。
- Realtime local transformer 在每个 request 内逐 frame 生成，batch 维度没有充分利用。

### 3.4 Stage 0 -> Stage 1 传输与 streaming

`vllm-omni/vllm_omni/model_executor/stage_input_processors/moss_tts.py` 中：

- non-streaming `talker2codec()` 会把最终 `(T, NQ)` codes 转成 `(NQ, T)` flat ids。
- streaming `talker2codec_async_chunk()` 维护 per-request accumulated codes。
- 但当前代码把 `chunk_frames` 强制设为 `1 << 30`，注释说明原因是 codec causal decoder 尚未接好 left-context，25-frame 首 chunk 会触发内部 reshape 问题。因此实际效果是：配置里 `async_chunk: true` 和 `codec_chunk_frames` 存在，但 full delay/realtime 路径会攒到结束才送 codec。

这意味着 MOSS-TTS full variants 当前的 TTFA/首包延迟仍有明显优化空间。

### 3.5 Codec decode

`vllm-omni/vllm_omni/model_executor/models/moss_tts/modeling_moss_tts_codec.py` 已经支持：

- 独立加载 `OpenMOSS-Team/MOSS-Audio-Tokenizer`。
- `enforce_eager=False` 时尝试启用 `MossTTSCUDAGraphCodecWrapper`。
- `decode_cudagraph_capture_sizes` 可通过 deploy YAML 配置。

限制：

- `forward()` 里仍按 request 循环调用 `batch_decode(codes_list=[codes_nq_t])` 或 CUDA Graph wrapper。
- stage 1 默认 `max_num_seqs: 1`，基本没有真正 batched vocoder decode。
- CUDA Graph wrapper 是单 segment decode wrapper，不等价于跨请求 batched vocoder。

## 4. 优化建议

### P0：补一套可重复 benchmark 与 profiler

优先级最高。没有可重复指标，后续优化很容易只改善局部、拖慢整体。

建议新增：

- MOSS-TTS v1.5 / MOSS-TTS-Realtime 的 serving benchmark。
- 指标至少包括：RTF mean/median/p95、TTFA、throughput req/s、stage 0 decode step time、stage 1 codec time、reference encode/cache hit rate、audio duration、失败率。
- 对齐 sglang-omni 的 Seed-TTS 评估思路，至少保留 WER/CER 和 speaker similarity 的离线任务入口。
- 对 `moss_tts.yaml` 下 `max_num_seqs`、`async_scheduling`、`max_num_batched_tokens`、stage 1 `enforce_eager` 做矩阵测试。

落点：

- 新增或扩展 `vllm-omni/vllm_omni/benchmarks/`
- 扩展 `tests/e2e/offline_inference/test_moss_tts_v1_5.py`
- 增加 profiler 标记，区分 serving 构参、reference encode、stage 0 AR、connector、stage 1 codec。

预期收益：为 P1/P2 优化提供客观排序。风险低。

### P1：恢复真正 streaming codec chunk

当前 `talker2codec_async_chunk()` 名义上支持 chunk，但实际强制等结束才 emit。这会直接影响首包延迟。

建议：

1. 修复 MOSS Audio Tokenizer decoder 对小 chunk 的 reshape/left-context 问题。
2. 让 `codec_chunk_frames` 和 `codec_left_context_frames` 真正生效。
3. 对 delay variant 做 de-delay 后按连续非 pad segment 输出。
4. 对 realtime variant 跳过 de-delay，但仍按 chunk 输出。
5. 增加 A/B 测试：完整解码 vs chunk 解码拼接后的波形长度、边界连续性、主观/客观质量。

落点：

- `vllm-omni/vllm_omni/model_executor/stage_input_processors/moss_tts.py`
- `vllm-omni/vllm_omni/model_executor/models/moss_tts/modeling_moss_tts_codec.py`
- `vllm-omni/vllm_omni/model_executor/models/moss_tts/audio_tokenizer.py`
- `vllm-omni/vllm_omni/deploy/moss_tts.yaml`
- `vllm-omni/vllm_omni/deploy/moss_tts_realtime.yaml`

预期收益：显著降低 TTFA/streaming 体感延迟。风险中等，主要是边界音质和短 chunk decoder 正确性。

### P1：codec stage 真正 batched decode

sglang-omni #738 指出 vocoder batch path 如果只是 serial loop，GPU 利用率会差。vLLM-Omni 当前 codec `forward()` 也是按 request 循环 decode，并且 deploy 里 stage 1 `max_num_seqs: 1`。

建议：

1. 将同一 step 内多个 request 的 `codes_nq_t` 收集起来。
2. 按 frame length bucket 或 padding 成 `[B, NQ, T_max]`。
3. 调用 `MossAudioTokenizerModel.batch_decode()` 一次完成 batch decode。
4. 根据 `audio_lengths` trim，分发回每个 request。
5. 单请求或长度差异很大时 fallback 到现有路径。
6. 将 stage 1 `max_num_seqs` 从 1 提高到可配置值，例如 4/8，并用 benchmark 找最佳点。

落点：

- `vllm-omni/vllm_omni/model_executor/models/moss_tts/modeling_moss_tts_codec.py`
- `vllm-omni/vllm_omni/model_executor/models/moss_tts/audio_tokenizer.py`
- `vllm-omni/vllm_omni/deploy/moss_tts.yaml`

预期收益：并发场景下提高 codec 吞吐，降低 stage 1 排队。风险中等，主要是 padding artifact 和 batch shape 对 CUDA Graph capture 的影响。

### P1：把 delay talker 的 per-request sampling 进一步 batch 化

目前 audio head 内部已经 vectorized，但 batch 维度仍是 request loop。sglang-omni #609 的经验说明 per-row/per-codebook sampling 曾是大瓶颈；vLLM-Omni 已经解决 per-codebook 的一半，但 per-request 仍可优化。

建议：

1. 在 `make_omni_output()` 中一次性收集所有 active requests 的 last hidden states，形成 `[B_active, H]`。
2. 将 `_sample_audio_codes()` 扩展成 batch 版本，输入 `[B, H]` 和 state tensors，输出 `[B, NQ]`。
3. 将 `audio_lengths`、`delayed_lengths`、`sampling_mask` 从 Python dict/list 转成 tensor。
4. 对 `torch.multinomial` 使用 batched probs。
5. 累积 codes 时避免每步 `torch.cat`，改成 list buffer 或预分配 tensor pool。

落点：

- `vllm-omni/vllm_omni/model_executor/models/moss_tts/modeling_moss_tts_talker.py`

预期收益：并发下减少 Python overhead 和 kernel launch 数。风险中等，需要确保 seeded sampling 和现有输出分发语义不变。

### P1：参考音频 cache 改成内容哈希，并补 singleflight

现有 anonymous cache key 是 `sha1(ref_str)`，不是音频内容 hash。对 URL、路径、data URI 的等价内容不稳定。

建议：

1. 在 `_resolve_ref_audio()` 后对 normalized waveform bytes + sample_rate + `n_vq` + variant 做 BLAKE2/SHA256。
2. named voice 仍按 `voice_created_at` invalidation。
3. anonymous reference 改为内容 hash。
4. 对相同 key 的并发 cold miss 做 singleflight：第一个请求执行 encode，其余 await 同一个 future，避免同一参考音频并发重复 encode。
5. 记录 cache hit/miss/evict 指标。

落点：

- `vllm-omni/vllm_omni/model_executor/models/moss_tts/reference_encoder.py`
- `vllm-omni/vllm_omni/entrypoints/openai/serving_speech.py`

预期收益：高复用 speaker 的线上场景收益明显；对 cold 并发更稳定。风险低到中。

### P2：参考音频 encoder GPU/batch 化，作为可选模式

vLLM-Omni 当前注释说明 MOSS audio tokenizer 放 CPU 是为了避免 8B talker 旁边多占约 6.7 GiB。这个选择合理，但会牺牲 cold reference latency。

建议做成可选配置，而不是默认替换：

- `reference_encoder_device: cpu|cuda|auto`
- `reference_encoder_max_batch_size`
- `reference_encoder_cache_max_entries`

可行路径：

1. 小模型或双 GPU 部署时，将 reference encoder 放到 stage 1 GPU 或独立 GPU。
2. 将多个 cold reference encode bucket 后 batch encode。
3. CPU OOM/GPU OOM 时自动 fallback CPU encode。

预期收益：冷启动、多 speaker、大量新 reference 场景更快。风险中高，主要是显存预算、跨设备数据拷贝、pipeline 调度复杂度。

### P2：Realtime local transformer batch 化

Realtime variant 的核心低延迟路径是 local transformer frame decode。当前每个 request 逐个调用 `local_transformer.generate_frame()`，并且 history/repetition penalty 在 Python list 中处理。

建议：

1. 为 `generate_frame()` 增加 batch input：`hidden: [B, H]`。
2. 将 `history_per_codebook` 转成 padded tensor。
3. repetition penalty、top-k/top-p、sampling 在 batch 维度执行。
4. 对已停止 request 做 mask，而不是跳过导致 batch shape 抖动。

落点：

- `vllm-omni/vllm_omni/model_executor/models/moss_tts/modeling_moss_tts_talker.py`
- `vllm-omni/vllm_omni/model_executor/models/moss_tts/modeling_moss_tts_local.py`

预期收益：Realtime 并发吞吐提升。风险中等，采样一致性与停止条件要单测覆盖。

### P2：减少跨 stage CPU 往返和 list 化

当前 Stage 0 -> Stage 1 传输会将 codes `.cpu()`、`.tolist()` 后通过 connector 传输，Stage 1 再转回 tensor。对长音频或高并发来说，这是额外 CPU copy 和 Python 对象开销。

建议：

1. 如果 SharedMemoryConnector 支持 tensor payload，优先传 tensor 而不是 list[int]。
2. 对 `codes.audio` 使用 compact dtype，例如 `torch.int16` 或 `torch.int32`，因为 codebook size 约 1024。
3. 在 Stage 1 直接从 shared memory tensor view 构造 `codes_nq_t`。
4. 保留 list path 作为兼容 fallback。

落点：

- `vllm-omni/vllm_omni/model_executor/stage_input_processors/moss_tts.py`
- stage connector 相关实现

预期收益：降低 connector overhead 和 CPU 压力。风险中等，涉及通用 connector contract。

### P2：CUDA Graph capture 策略扩展到 batched/streaming codec

现有 codec CUDA Graph wrapper 是单 decode shape capture。引入 batched decode 和 streaming chunk 后，需要重新设计 capture sizes。

建议：

- capture key 从 `T` 扩展为 `(B, T_bucket, NQ)` 或固定 `NQ` 后 `(B, T_bucket)`。
- 对常用 chunk size，例如 15/25/50/100 frames，预热 `B=1/2/4/8`。
- 对非常长尾长度 fallback eager。
- benchmark 比较 graph replay vs padding overhead。

落点：

- `vllm-omni/vllm_omni/model_executor/models/moss_tts/moss_codec_cudagraph.py`
- `vllm-omni/vllm_omni/model_executor/models/moss_tts/modeling_moss_tts_codec.py`

预期收益：在稳定 chunk/batch shape 下减少 launch overhead。风险中等。

### P3：`torch.compile` 可选实验

sglang-omni roadmap 提到 AR backbone、encoder compile。vLLM-Omni 依赖 vLLM 执行栈，直接 compile 整个模型未必划算，但可以局部实验：

- `MossTTSRealtimeLocalTransformer.generate_frame()` 内部可尝试 compile。
- codec `_decode` 可与 CUDA Graph 对比。
- delay talker 的 batch sampling helper 可尝试 compile，但 sampling/随机数/动态 shape 可能限制收益。

建议作为实验开关，不应先于 P0/P1。

## 5. 建议实施顺序

1. 建 benchmark/profiler，确认当前 RTF、TTFA、stage 占比和 cache 命中率。
2. 修复 streaming codec chunk，让配置里的 `codec_chunk_frames` 生效。
3. 实现 codec batched decode，并把 stage 1 `max_num_seqs` 调到可 benchmark 的值。
4. batch 化 delay talker per-request sampling，顺带消除逐步 `torch.cat`。
5. 改进 reference cache：内容哈希、singleflight、指标。
6. 视 benchmark 决定是否做 GPU/batched reference encoder、Realtime local transformer batch、batched CUDA Graph。

## 6. 风险与验证清单

必须验证：

- 生成音频长度与 max_new_frames/token_count 控制一致。
- delay-pattern de-delay 后没有 pad row 泄漏。
- audio code clamp/mask 没有采样到 out-of-range code。
- streaming chunk 拼接不引入明显边界噪声。
- batched codec 与 serial codec 输出一致或误差在可接受范围。
- cache hit 与 miss 产生 bit-identical reference codes。
- seeded sampling 在单请求下保持可复现；并发 seeded 请求至少不互相污染全局 RNG。

建议测试：

- 单请求短文本、长文本。
- concurrency 1/2/4/8/16。
- 相同 reference 重复请求。
- 相同 reference 并发 cold miss。
- 不同长度 reference 混合。
- MOSS-TTS v1.5、MOSS-TTS-Realtime、MOSS-TTSD 至少各覆盖一条。

## 7. 结论

vLLM-Omni 的 MOSS-TTS 已经不是 sglang-omni #637 中的最初状态：它已经有参考音频 cache、async thread cold encode、audio head/embedding stacking、codec CUDA Graph wrapper。但当前最值得优先投入的点也很明确：

1. 真正启用 streaming codec chunk，降低首包延迟。
2. stage 1 codec 做 batched decode，提升并发吞吐。
3. delay/realtime talker 的 per-request Python loop 继续 batch 化。
4. reference cache 从字符串 key 升级到内容 hash，并处理并发 cold miss。
5. 用稳定 benchmark 把这些优化排出真实收益。

如果目标是快速对齐 sglang-omni 的“RTF < 1 at concurrency 16”方向，建议先做 P0 + P1。P2/P3 更像是性能天花板优化，应该等 profiler 明确指出瓶颈后再推进。
