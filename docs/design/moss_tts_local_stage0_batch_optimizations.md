# MOSS-TTS Local Stage0 Batch Optimizations

本文记录 MOSS-TTS-Local-Transformer-v1.5 Stage0 最近两处 batch 化优化。目标是减少 decode 热路径里的 Python loop、小 GPU kernel、D2D scatter/copy，以及由细碎 CPU 调度带来的 GPU 空泡。

## 背景

在 8 并发 profile 中，Stage0 已经能入 CUDA graph，但 forward 前后仍有明显 CPU/GPU 空泡。截图里可以看到：

- CPU 侧 `_preprocess -> MossTTSLocalTalker.preprocess` 被拆成多段很小的 embedding/copy。
- GPU 侧 `execute_context_0(0)_generation_8(8)` 前有多个零散 kernel 和 memcpy，中间空白较多。
- `compute_logits` 里曾出现 `aten::nonzero`、`aten::index_put_` 和 `cudaStreamSynchronize`。

优化前 CPU 侧截图：

![Stage0 CPU profile before](../../cpu.png)

优化前 GPU 侧截图：

![Stage0 GPU profile before](../../device.png)

优化后截图占位：

![Stage0 CPU profile after](TODO_cpu_after.png)

![Stage0 GPU profile after](TODO_device_after.png)

## 优化一：decode preprocess batch 化

### 优化前

MOSS local 没有实现 `preprocess_decode_batch()`，所以 runner 在 decode step 会对每个 request 逐个调用 scalar `preprocess()`：

```text
for req in decode_requests:
    model.preprocess(input_ids[s:e], **req_infos)
    copy req_embeds -> inputs_embeds
    copy req_embeds -> talker_mtp_inputs_embeds
    copy last_hidden -> last_talker_hidden
    copy text_step -> text_step
```

8 并发时，这会产生 8 次小 embedding lookup、8 次 hidden reshape/to、8 次 `zeros_like`/control 构造，以及多组很小的 D2D copy。每个 kernel 很短，但 CPU launch 和调度空泡很多。

### 优化后

新增 `MossTTSLocalTalkerForGeneration.preprocess_decode_batch()`，复用 runner 里已有的 batch decode preprocess 快路径：

```text
batch input_ids -> 一次 embed_tokens
batch req_infos -> 拼出 mtp_hidden
batch req_infos -> 构造 mtp_control
runner 一次 index_copy_ 回写 inputs_embeds
runner 一次 batch copy 到 talker_mtp buffers
```

这样 decode batch 从 N 次 scalar preprocess 变成一次 batch preprocess。对 8 并发来说，forward 前段原本很散的 embedding/copy 小段应该明显合并。

关键行为：

- `input_ids_out`: `(B,)`
- `text_embeds`: `(B, H)`
- `mtp_hidden`: `(B, H)`，来自每个 request 的 `hidden_states.last`
- `mtp_control`: `(B, H)`，第 0 维表示 active/stop 控制
- `updates`: 当前为空列表，保持 runner 接口一致

## 优化二：talker_mtp 输出 batch-local transport

### 优化前

为了把 MOSS local 的 `should_continue` 从 `talker_mtp` 传给 `compute_logits`，最初采用了显式 per-request additional_information 写回：

```text
talker_mtp -> output_codes, should_continue
runner:
  additional_information[req_id]["audio_codes"]["current"] = output_codes[i:i+1]
  additional_information[req_id]["talker_mtp"]["should_continue"] = should_continue[i:i+1]
make_omni_output:
  再按 input_batch.req_ids 从 additional_information 里取回
```

这个设计语义清楚，但 Stage0 decode 每步都会多一批 per-request GPU tensor slice/clone/copy。8 并发下会看到多组很小的 D2D 更新，容易继续制造 CPU/GPU 空泡。

### 优化后

保留三返回值协议：

```python
return input_embeds_out, output_codes, should_continue
```

但 MOSS local 声明：

```python
self.talker_mtp_outputs_batch_local = True
self.talker_mtp_output_key = ("audio_codes", "current")
self.talker_mtp_aux_output_key = ("talker_mtp", "should_continue")
```

runner 对这类模型不再逐 request 写入 `additional_information`，而是把整批输出作为 batch-local payload 传给 `make_omni_output`：

```text
talker_mtp_batch_outputs = {
    req_ids,
    output_key,
    output,
    aux_key,
    aux,
}
```

`make_omni_output()` 再按 `request_id -> batch row` 映射取出当前 step 的 code 和 `should_continue`。这样保留了 req_id 对齐语义，又避免了 per-request tensor 存储热路径。

关键效果：

- `audio_codes.current` 不再作为跨轮 state 写回。
- `talker_mtp.should_continue` 不再作为跨轮 state 写回。
- 两者都是当前 decode step 的 batch-local 输出。
- `hidden_states.last` 仍然通过原有 `postprocess -> additional_information["hidden_states"]["last"]` 作为跨轮状态。

## 额外修正

### compute_logits 去掉 boolean indexing

之前 `compute_logits` 曾使用：

```python
stop_rows = rows_t[~should_continue]
logits[stop_rows, assistant_id] = -inf
logits[stop_rows, im_end_id] = 0
```

这会触发 `aten::nonzero` 和 `aten::index_put_`，profile 中能看到 `cudaStreamSynchronize`。

现在改为两列 `torch.where`：

```python
logits[:, assistant_id] = torch.where(should_continue, zeros, neg_inf)
logits[:, im_end_id] = torch.where(should_continue, neg_inf, zeros)
```

### make_omni_output 去掉 D2H 判断

之前有：

```python
bool(raw_should.reshape(-1)[0].detach().to("cpu").item())
```

这会强制 D2H 同步。现在不再用 CPU bool 判断 emit；停止轮会携带全 pad code row，后续 raw codec streaming processor 会过滤全 pad 行。

### make_omni_output 不再每步输出 ref codes

Local raw codec streaming path 不消费 `codes.ref`，所以 `make_omni_output()` 不再把 reference codes 放进 `multimodal_outputs`，避免每步重复 H2D。

## 预期 Profile 变化

优化后应重点观察：

- `_preprocess` 中 scalar `preprocess()` 调用数量减少，出现 batch `preprocess_decode_batch()`。
- forward 前段 embedding/copy 小 kernel 数量减少。
- `compute_logits` 中 `aten::nonzero`、`aten::index_put_` 消失或显著减少。
- `make_omni_output` 中不再有 `should_continue.to("cpu").item()` 造成的 `cudaStreamSynchronize`。
- `talker_mtp` 输出写回不再出现每个 request 一组 `audio_codes.current` / `talker_mtp.should_continue` 小 tensor copy。

