# MOSS-TTS Local Graph `seq_token_counts` 问题总结

本文记录 MOSS-TTS-Local-v1.5 接入 talker MTP CUDA graph wrapper 过程中遇到的首包延迟、codec 长度错配，以及最终对 `seq_token_counts` 生成逻辑的修复。

## 背景

当前 MOSS-TTS Local 的 stage0 talker 通过 MTP 接口生成 local codec code，并通过 async chunk 传给 stage1 codec。为了让 local transformer 进入 CUDA graph，需要把原本 eager 的 per-request Python 状态更新拆出来：

- `talker_mtp` 保持 tensor-only 路径，适合进入 graph。
- `postprocess_talker_mtp` 处理 CPU 状态、字典、request state、emit 等逻辑，不进入 graph。

这部分 graph-safe 改造本身可以降低 steady-state RTF，但它暴露了 stage1 codec 在 CUDA graph padding 下的长度语义问题。

## 观察到的问题

开启 local graph-safe 改造后，出现过以下现象：

- graph 本身对 RTF 有收益，关闭 graph 后 RTF 会劣化。
- 但 TTFP 曾从约 100ms 劣化到 300ms 或 450ms。
- 服务端日志里的 `first_chunk_ms` 看起来很快，例如 37ms 左右，但实际首个有效音频并不一定已经产生。
- stage1 codec 日志出现：

```text
MossTTS codec input length 16 not divisible by n_vq 12; skipping.
```

这个 warning 说明 codec 收到了 16 个 token，但 MOSS local codec 的一个 frame 应该是 `n_vq=12` 个 code。首包 1 frame 时真实长度应该是 12，不应该是 16。

## 正确的数据路径

MOSS raw local streaming 的正确路径如下：

1. Stage0 local talker 生成 raw codec rows，形状通常是 `[T, n_vq]`。

2. `talker2codec_raw_async_chunk` 将其转成 stage1 codec 需要的一维 codebook-major 序列：

```python
codec_flat = chunk_codes.transpose(0, 1).contiguous().reshape(-1).to(torch.long)
```

3. 同一个 payload 里会写入真实 codec token 数：

```python
meta=MetaStruct(
    ...
    code_flat_numel=int(codec_flat.numel()),
    ...
)
```

首包如果 `T=1, n_vq=12`，那么：

```text
code_flat_numel = 12
```

4. connector 将 `codes.audio` 作为下一 stage 的 token 序列，同时把 `meta.code_flat_numel` 放入 request 的 additional/runtime information。

5. Stage1 runner 生成传给 codec 的 `seq_token_counts`。

6. Stage1 codec 只应该用 `seq_token_counts` 作为每个 request 的逻辑 codec token 长度来切 `input_ids`。

关键点是：`seq_token_counts` 是 codec 的唯一切分入口；`meta.code_flat_numel` 是跨 stage payload 里的真实长度来源。两者不是两套切分逻辑，而是：

```text
meta.code_flat_numel -> runner 归一化成 seq_token_counts -> codec 用 seq_token_counts 切 input_ids
```

## 三种长度的区别

这次问题的核心是混淆了三种长度：

| 字段 | 含义 | 是否可能包含 graph padding |
| --- | --- | --- |
| `input_ids.numel()` | 实际传入模型的物理 tensor 长度 | 是 |
| `scheduler_output.num_scheduled_tokens` | scheduler 本轮调度的 transport token 数 | 否，但可能包含 finish sentinel |
| `meta.code_flat_numel` | codec 语义上的真实 audio code token 数 | 否 |

CUDA graph 会把 `input_ids` pad 到 bucket。例如真实首包是 12 个 codec token，但 graph bucket 可能是 16：

```text
真实 codec 长度: 12
input_ids.numel(): 16
```

因此不能用 `input_ids.numel()` 作为 codec 的逻辑长度。

另一个例子是 control-only finish packet：为了唤醒 stage1，payload 可能带一个 sentinel token `[0]`，scheduler 看到的 token 数是 1，但真实 codec audio 长度是 0：

```text
scheduler token count: 1
meta.code_flat_numel: 0
```

这也是为什么 `meta.code_flat_numel` 比 scheduler count 更接近 codec 语义。

## 根因

当时新增的 `_code2wav_seq_token_counts` 逻辑里有两个问题：

```python
total = int(input_ids.reshape(-1).shape[0])
if len(scheduled_token_counts) <= 1:
    return [total]
```

单请求时，它在读取 `meta.code_flat_numel` 之前就直接返回了 `input_ids.numel()`。在 graph padding 场景下，这个 `total` 是 padded bucket 长度，不是真实 codec 长度。

所以首包真实长度 12 被错误写成 16：

```text
seq_token_counts = [16]
```

codec 随后用 `[16]` 切分 input ids，发现 16 不能被 `n_vq=12` 整除，于是跳过这个 chunk。这样服务端可能已经记录了很快的 first chunk，但首个有效音频实际上被推迟了。

第二个问题是原逻辑要求：

```python
sum(sizes) == total
```

在 CUDA graph padding 下，正确情况经常是：

```text
sum(code_flat_numel) < input_ids.numel()
```

例如 `12 < 16`。因此这里应该允许小于等于，而不是必须相等。

## 修复方案

修复点在 `vllm_omni/worker/gpu_generation_model_runner.py`。

runner 仍然只给 codec 传一个标准字段：

```python
model_kwargs["seq_token_counts"]
```

但生成规则改成：

1. 如果 `runtime_additional_information[*].meta.code_flat_numel` 存在，并且数量与 request 数匹配，则优先使用它。
2. 允许 `sum(code_flat_numel) <= input_ids.numel()`，因为 `input_ids` 可以包含 graph padding。
3. 如果没有 `code_flat_numel`，再 fallback 到 scheduler 的 `num_scheduled_tokens`。
4. 不再优先使用 padded `input_ids.numel()`。

修复后的核心逻辑：

```python
if input_ids is None:
    return scheduled_token_counts
total = int(input_ids.reshape(-1).shape[0])
if isinstance(runtime_additional_information, list):
    sizes: list[int] = []
    for info in runtime_additional_information:
        meta = info.get("meta", {}) if isinstance(info, dict) else {}
        value = meta.get("code_flat_numel") if isinstance(meta, dict) else None
        if value is None:
            break
        sizes.append(int(value))
    if len(sizes) == len(scheduled_token_counts) and sum(sizes) <= total:
        return sizes
return scheduled_token_counts
```

## 修复后的行为

首包场景：

```text
code_flat_numel = 12
input_ids.numel() = 16
```

修复前：

```text
seq_token_counts = [16]
codec skip: 16 % 12 != 0
```

修复后：

```text
seq_token_counts = [12]
codec trim padding 后正常 decode 1 frame
```

control-only finish packet 场景：

```text
code_flat_numel = 0
scheduler token count = 1
```

runner 应该生成：

```text
seq_token_counts = [0]
```

这样 codec 不会把 sentinel 当成真实 audio code。

## 相关结论

- `postprocess_talker_mtp` 不应该入图，它处理的是 Python/CPU/request state，不是 graph-safe tensor 计算。
- 服务端日志里的 `first_chunk_ms` 不一定等价于首个有效音频时间；如果 codec 产出了空输出或 skip 了首包，这个指标可能显得过快。
- local graph 对 RTF 是有收益的；TTFP 劣化不一定来自 graph replay 本身，也可能来自 graph-safe 适配过程中引入的 payload 长度语义变化。
- 对 code2wav/codec stage，`seq_token_counts` 必须表达逻辑 codec token 数，而不是 padded tensor 长度。

## 后续注意点

当前主修复解决了首包 `12 -> 16` 的 graph padding 问题。另一个可以继续检查的边界是 codec 对 `code_flat_numel=0` 的 finish packet 处理：如果 segment 长度为 0，codec 需要确保仍能执行 streaming state flush，而不是在 `seg.numel() == 0` 时直接跳过所有 finish 逻辑。
