# MOSS-TTS Local 高并发性能优化点

本文记录 MOSS-TTS-Local-v1.5 在高并发 benchmark 中的可优化方向。当前观测数据：

```text
concurrency=1   RTF=0.213  TTFP=92ms
concurrency=8   RTF=0.698  TTFP=873ms
concurrency=16  RTF=2.311  TTFP=2978ms
concurrency=32  RTF=3.069  TTFP=3864ms
```

单并发表现正常，但并发升高后 RTF/TTFP 都急剧劣化，说明主要问题不是单条请求的 kernel 速度，而是 batching、stage 间排队、graph bucket padding、CPU postprocess 或 codec stage 吞吐在高并发下被放大。

## P0: 确认 talker MTP 是否被显式 seed 打散成单行执行

`gpu_model_runner._talker_mtp_forward` 中有一个特殊路径：

```python
if decode_batch_size > 1 and any(_explicit_talker_seed(req_id) is not None for req_id in decode_req_ids):
    ...
    for row, req_id in enumerate(decode_req_ids):
        self._talker_mtp_forward([req_id], inputs_embeds, row_offsets)
```

只要 batch 内任一 request 带 `extra_args["tts_local_seed"]`，多请求 MTP batch 就会被拆成逐行递归执行。这会直接破坏高并发 batching：

```text
C=16 正常期望：一次 B=16 local MTP graph/replay
C=16 seeded fallback：16 次 B=1 local MTP forward/replay + Python 循环
```

需要检查：

- benchmark 请求是否传了 `seed`。
- `serving_speech.py` 是否把请求级 `seed` 写入了 `extra_args["tts_local_seed"]`。
- stage0 默认 `seed` 是否会通过当前服务路径变成 `tts_local_seed`。
- profiling 中 `_talker_mtp_forward([req_id])` 是否在高并发下被重复调用。

优化方向：

- 高吞吐 benchmark 不传 request-level seed。
- 如果必须可复现，不要用“每个 request 一个 torch.Generator 导致 scalar fallback”的实现；需要实现 batched deterministic sampling，或接受高并发路径不保证 per-request seed 隔离。
- 对 MOSS local 可以考虑默认不把 stage0 sampling seed 映射到 `tts_local_seed`，只在用户显式要求 local seed 时启用。

## P0: 控制 talker MTP graph bucket padding

当前 local MTP graph 会按 batch descriptor/capture bucket 运行。如果真实 active request 数较小，但命中了更大的 bucket，会产生额外行计算。

需要重点看高并发下：

```text
decode_batch_size
num_tokens_padded
batch_desc.num_tokens
```

风险模式：

```text
真实 active B=9  -> 命中 B=16 graph
真实 active B=17 -> 命中 B=32 graph
```

对 local transformer 来说，这些 padded 行也会进入 local transformer 的 batched matmul/attention，成本会被真实并发放大。

优化方向：

- 显式配置更细的 `cudagraph_capture_sizes`，覆盖高并发常见 batch size。
- 优先捕获 `[1, 2, 4, 8, 16, 32]`，如果 workload 经常出现非 2 次幂 batch，再补 `[3, 5, 6, 7, 9, 10, 12, 14, 24]`。
- 在 `_talker_mtp_forward` 周围加 debug 统计，记录 `decode_batch_size -> num_tokens_padded` 的分布，先确认 padding 放大倍数。

## P0: Stage0 和 Stage1 当前同卡，可能在高并发下互相抢占

`moss_tts_local.yaml` 当前两个 stage 都在 `devices: "0"`。低并发时影响不明显；高并发时 stage0 backbone/local MTP 和 stage1 codec 同卡排队，TTFP 会被放大。

优化方向：

- 如果机器有多张 GPU，优先改成 stage0/stage1 分卡：

```yaml
stage0 devices: "0"
stage1 devices: "1"
```

- 如果必须单卡，需要分别 profiling stage0 和 stage1 的 device timeline，确认是 local MTP、backbone replay、还是 codec 占满 GPU。
- 对单卡部署，适当降低 stage1 `max_num_seqs` 或调大 codec chunk，减少 stage1 插队频率；但这会影响首包/流式粒度，需要权衡。

## P1: Stage1 codec chunk 策略会影响高并发排队

当前配置：

```yaml
initial_codec_chunk_frames: 1
codec_chunk_frames: 15
```

含义：

- 首包 1 frame，目标低 TTFP。
- 后续每 15 frame 才送 codec，大约 1.2s 音频一个 chunk。

高并发下的风险：

- `initial_codec_chunk_frames=1` 会让所有请求很早进入 stage1，造成 stage1 首包风暴。
- 后续 `codec_chunk_frames=15` 单个 chunk 较大，如果 stage1 codec 是重计算，会形成较长 GPU 占用片段，阻塞其他请求首包。

优化方向：

- 分别测试：

```yaml
initial_codec_chunk_frames: 1 / 2 / 3
codec_chunk_frames: 8 / 12 / 15 / 20
```

- 低 TTFP 优先：小 initial chunk。
- 高吞吐优先：适当增大 initial chunk，减少 stage1 小包调度风暴。
- 如果 stage1 是瓶颈，优先分卡，其次再调 chunk。

## P1: connector polling sleep 可能放大高并发延迟

当前配置：

```yaml
connector_get_sleep_s: 0.005
connector_get_max_wait_first_chunk: 1000
connector_get_max_wait: 300
```

`connector_get_sleep_s=5ms` 在单请求时不明显，但高并发下 stage1 每个请求等待 chunk 的轮询延迟会累积，并影响首包调度及时性。

优化方向：

- 测试 `connector_get_sleep_s: 0.001` 或更低。
- 观察 CPU 占用和 stage1 request ready 延迟。
- 如果降低 sleep 后 TTFP 明显下降，说明 stage 间调度 wakeup 是瓶颈之一。

## P1: `postprocess_talker_mtp` 是 CPU 串行路径

`talker_mtp` 入图后，`postprocess_talker_mtp` 仍然在 graph 外处理：

- `mtp_outputs` GPU -> CPU flags。
- 遍历每个 request 更新 `audio_state`。
- 更新 `audio_codes.current / accumulated / emit`。
- 合并到 `model_intermediate_buffer`。

这条路径不能直接入图，但高并发下 Python per-request loop 会变重。

优化方向：

- profiling `postprocess_talker_mtp` CPU self time。
- 减少 CPU copy 的字段，只 copy 必需的 `active/continue` flags。
- 避免每步拼接 `accumulated` 大 tensor；如果只需要 streaming emit，尽量只保留 current frame，历史累积延后或按需维护。
- 合并 `_merge_additional_information_update` 的小 dict 更新，减少 per-request Python 对象操作。

## P1: 检查 stage1 codec 是否仍在跳过或处理 padding/control token

之前已经修复了 `seq_token_counts` 的主要问题：

```text
meta.code_flat_numel -> seq_token_counts -> codec trim graph padding
```

但高并发仍需确认：

- 不再出现 `input length 16 not divisible by n_vq 12; skipping`。
- control-only finish packet 的 `code_flat_numel=0` 不会被当成真实 audio token。
- mixed batch 中 `[0, 12, 180, ...]` 这类 seq counts 能正确切分。

如果 codec 仍跳过首包，服务端 `first_chunk_ms` 可能看起来快，但首个有效音频会延后。

## P2: Stage0 `max_num_batched_tokens` 过大，可能导致调度形态不适合低延迟

当前 stage0：

```yaml
max_num_batched_tokens: 65536
max_num_seqs: 32
```

对 TTS streaming decode 来说，单步通常是 decode token，过大的 token budget 可能让 prefill 和 decode 混批过重，增加首包排队。

参考 qwen3 high concurrency 配置，stage0 使用了更小的 batched token budget：

```yaml
max_num_batched_tokens: 512
```

优化方向：

- 测试 stage0 `max_num_batched_tokens: 512 / 1024 / 2048 / 4096`。
- 观察 TTFT/TPOT/TTFP 和 throughput。
- 如果高并发 TTFP 明显改善，说明调度队列中过大的 prefill/decode 混批是问题之一。

## P2: Stage1 `max_num_seqs=32` 未必适合 codec

codec stage 和 text AR stage 的最佳 batch size 不一定一致。codec 可能在小 batch 更低延迟，在大 batch 更高吞吐。

优化方向：

- 测试 stage1 `max_num_seqs: 4 / 8 / 16 / 32`。
- 如果 stage1 单卡和 stage0 共享 GPU，高 stage1 batch 可能拉高 stage0 空泡和 TTFP。
- 如果 stage1 独立 GPU，可更激进地增大 batch。

## P2: capture sizes 与真实 workload 不匹配

stage1 yaml 里 capture sizes 目前是注释状态：

```yaml
# compilation_config:
#   cudagraph_capture_sizes: [1, 2, 3, ..., 15]
```

如果 stage1 codec 或 stage0 backbone/local MTP 没有覆盖真实 batch shape，会出现 eager fallback 或过大 bucket replay。

优化方向：

- 分 stage 打开 cudagraph stats/debug 日志。
- 记录每个 stage 的 graph hit/miss、bucket size、padding size。
- 对常见 shape 补 capture sizes，而不是盲目捕获全部。

## P2: 服务端 first chunk 指标需要区分“有 chunk”和“有效音频”

服务端 `first_chunk_ms` 目前更接近“第一次 yield audio bytes/chunk”的时间，不一定等于“首个有效 codec 音频完成”的时间。

如果 codec skip、空 tensor、finish sentinel、或空音频也触发 first chunk 记录，高并发下指标会误导判断。

优化方向：

- 增加 debug 指标：

```text
first_stage0_output_ms
first_stage1_input_ms
first_codec_non_empty_audio_ms
first_served_audio_bytes_ms
```

- benchmark TTFP 应以 first non-empty valid audio 为准。

## 建议的排查顺序

1. 先确认 `tts_local_seed` 是否触发高并发 scalar fallback。这是最可能导致 C=8/16 崩掉的代码路径。
2. 打印 `_talker_mtp_forward` 的 `decode_batch_size -> num_tokens_padded` 分布，确认 graph bucket padding 放大。
3. 分卡跑 stage0/stage1，判断是否单卡竞争导致 codec 和 local MTP 互相阻塞。
4. 调小 connector sleep，观察 TTFP 是否下降。
5. 扫 `initial_codec_chunk_frames / codec_chunk_frames`，确认 stage1 首包风暴和后续大 chunk 是否是瓶颈。
6. 调 stage0 `max_num_batched_tokens`，参考 qwen3 high concurrency 的较小 token budget。
7. profiling `postprocess_talker_mtp` CPU self time，确认 Python per-request loop 的占比。

## 推荐第一组实验

优先做一组最小变量实验：

```text
实验 A: 不传 request seed，确认没有 tts_local_seed，其他不变
实验 B: A + stage0 max_num_batched_tokens=1024
实验 C: B + connector_get_sleep_s=0.001
实验 D: C + stage0/stage1 分卡
实验 E: D + codec_chunk_frames sweep: 8/12/15/20
```

每组记录：

```text
RTF, TTFP, TPOT, TTFT
stage0 wall time
stage1 wall time
talker_mtp decode_batch_size
talker_mtp num_tokens_padded
codec first non-empty audio time
```

这样能把问题拆成 batching、padding、connector、codec、单卡竞争几类，而不是只看最终 RTF/TTFP。
