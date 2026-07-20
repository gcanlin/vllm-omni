# MOSS-TTS-Local-Transformer-v1.5 接入 vLLM-Omni 报告

## 1. 背景与架构

将 `OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5` 接入 vLLM-Omni，作为 MOSS-TTS 系列的第三个变体（此前已有 Delay、Realtime 两个变体）。

- **Talker（Stage 0，LLM_AR）**：Qwen3 backbone（36 层，hidden=2560）+ 1 层 GPT2 风格的「local depth transformer」，每帧内部重置 KV cache/位置，先输出二分类的 continue/stop 决策与第 0 通道 code，再依次采样剩余 11 个通道（`n_vq=12`）。
- **Codec（Stage 1，LLM_GENERATION）**：MOSS-Audio-Tokenizer-**v2**，48 kHz **立体声**、通道交织（channel-interleave）RVQ 编解码器。
- **两段式流水线**：Talker → Codec，通过 `SharedMemoryConnector`（`async_chunk: true`）连接。

整体接入遵循仓库内 `add-tts-model` skill 的标准流程（HF 参考实现 → Stage 拆分 → 在线服务 → async_chunk → CUDA Graph），并严格遵守约束：**未参考 `moss-test` 分支**（该分支已知存在精度问题），所有设计依据均来自官方 HuggingFace 参考代码、checkpoint 的 `config.json`/权重文件头，以及仓库已有的 Delay/Realtime 变体实现模式。

## 2. 遇到的问题与修复

### 2.1 「没有人声、像噪音」—— 根因 1：v1/v2 codec 权重命名不兼容

`audio_tokenizer.py` 中权重加载使用的 `_SUFFIX_REMAP` 表只覆盖了 v1（`MOSS-Audio-Tokenizer`）的命名约定（如 `self_attn.in_projs.0`、`linear1`/`linear2`），而 v1.5 用的是 v2（`MOSS-Audio-Tokenizer-v2`）codec，其子模块命名不同（`self_attn.in_proj`/`out_proj`，无下标；`ffn.0`/`ffn.2` 而非独立的 `linear1`/`linear2`）。

**修复**：在 `modeling_moss_tts_codec.py` 的 `_SUFFIX_REMAP` 中新增 4 条 v2 专用映射规则，同时保留全部 v1 规则不变。

### 2.2 「没有人声、像噪音」—— 根因 2：错误地用 Identity 替代 Linear

`audio_tokenizer.py` 的 `_ProjectedTransformer.__init__` 中，当 `input_dimension == d_model` 时错误地用 `nn.Identity()` 替代了应有的 `nn.Linear`。但上游官方实现里，无论维度是否相等，`input_proj`/`output_proj` 都是**真实的、训练过的** `nn.Linear`。这导致 encoder/decoder 中多个关键投影层的权重被静默丢弃（用随机初始化代替）。

**修复**：始终使用 `nn.Linear`，不再按维度做条件替换。

**验证方法**（吸取的经验：「不报错、波形幅度正常」≠「权重真的加载对了」）：
1. 生产环境日志确认权重加载计数：`loaded=2094/2094 skipped=0`（修复前并非如此）。
2. 编写独立的 codec 编解码往返测试：原始音频 → encode → decode，与原始波形做 **梅尔频谱余弦相似度** 对比（而非直接对原始波形做皮尔逊相关系数——后者即使编解码器完全正常，也会因相位/群延迟差异显示出很低的相关性，是误导性指标）。修复后相似度为 **0.934**，证明编解码器在结构上是正确的。

### 2.3 在线服务 OOM：窗口注意力的 (T,T) 稠密 mask

跑 benchmark 时，服务端在 codec 解码阶段崩溃，报错 `CUDA out of memory`，在构造窗口因果注意力的布尔 mask 时一次性申请约 4 GiB 显存。

**根因排查**：
- MOSS-TTS 系列的流式分块逻辑（`stage_input_processors/moss_tts.py` 的 `talker2codec_async_chunk`）目前对所有变体都被显式禁用（`chunk_frames: int = 1 << 30`，注释说明是因为 codec 的左上下文 padding 还没接好），即所有音频 codes 会**累积到生成结束后一次性**送入 codec 解码，而不是按 `codec_chunk_frames=15` 增量分块解码。
- 这本身是已知的、影响所有 MOSS-TTS 变体的既有限制（不是本次新引入的问题），但 v1.5 codec 的帧率/通道数更高，导致单次解码调用中间层的序列长度 T 可以达到上万，远超 v1 mono codec 变体常见的规模。
- `audio_tokenizer.py` 中 `_Attention.forward()` 对窗口因果注意力的实现方式是显式构造一个 `(T, T)` 的稠密布尔 mask，显存占用是 `O(T²)`——T 一旦变大（如 65536），mask 本身就需要约 4.3 GB，这是真正的瓶颈。

**修复**：将该 mask 的构造方式改为按 query 分块处理（block=4096），每块只与其可见的局部 key 范围构造小尺寸 mask，使峰值显存变为 `O(block × (block + context))`，与 T 无关，数值上与原版完全等价。

### 2.4 发现但尚未修复：talker 的 stop 条件可能从未触发

跑 benchmark 时发现一个新问题：5 条**不同长度**的 seed-tts 测试文本，生成的音频时长**全部精确等于 163.84 秒**（2048 帧）。这极不正常——短句子理论上应该比长段落生成更短的音频。

排查代码发现：
- Local-v1.5 talker 的停止逻辑依赖 `local_transformer.generate_frame()` 返回的二分类 `should_continue`；
- 代码中存在一个 `max_new_frames` 强制截断机制（`modeling_moss_tts_talker.py` 中类似机制在 Delay 变体上有明确注释：「如果没有这个机制，短 prompt 也会跑到 deploy 默认的 max_tokens，生成 100+ 秒音频」），但该值默认是 `-1`（关闭），因为 `/v1/audio/speech` 请求本身不会设置它；
- 因此目前唯一能让生成停下来的是 vLLM stage-0 的硬性 `max_tokens=4096` 限制（对应约 2048 帧），**强烈怀疑二分类停止头从未真正触发**，每次请求都是被硬上限截断，而不是模型自己决定结束。

**当前状态**：已定位疑点，**尚未修复**。这意味着下面的 benchmark 数值反映的是「固定长度解码」的性能，而非真实场景下随文本长度变化的延迟/RTF。

## 3. Benchmark 结果

使用 vLLM-Omni 自带的 TTS 通用 benchmark 框架（`benchmarks/tts/bench_tts.py`），数据集为 **Seed-TTS-eval**（`voice_clone` 任务，`en` locale，工作区中已下载好的 `/root/vllm-omni-workspace/seedtts_testset`）。

> 修复了 benchmark 客户端的一个 bug：`vllm_omni/benchmarks/patch/patch.py` 中 PCM 时长/RTF/连续性统计硬编码为 24 kHz 单通道（适用于 Qwen3-TTS、VoxCPM2 等其它模型），但 v1.5 实际输出是 **48 kHz 立体声**，不修复会导致时长被错算成实际的 1/4。已新增环境变量 `VLLM_OMNI_BENCH_AUDIO_SAMPLE_RATE` / `VLLM_OMNI_BENCH_AUDIO_CHANNELS`（默认值不变，向后兼容）。

### 测试配置

- 并发数（concurrency）：1
- 请求数（num-prompts）：5
- 数据集：Seed-TTS-eval（en）

### 结果汇总

| 指标 | 数值 |
|---|---|
| RTF（均值 / 中位数 / P99） | 0.230 / 0.230 / 0.230 |
| TTFP 首包延迟（均值 / 中位数 / P99） | 37,602 ms / 37,628 ms / 37,719 ms |
| E2E 延迟（均值 / 中位数 / P99） | 37,612 ms / 37,638 ms / 37,729 ms |
| 音频吞吐（生成音频时长/秒，real time） | 4.356 |
| 平均单条音频时长 | 163.84 s（5 条请求完全一致，见 §2.4） |
| 流式连续性达标率（无明显卡顿） | 100% |
| 平均欠载时间（audio underrun） | 0.00 s |
| 成功 / 失败请求数 | 5 / 0 |

**解读**：
- RTF 0.23 表示生成速度约为播放速度的 4.3 倍（生成 1 秒音频只需 0.23 秒），单看数字是不错的，但因为 §2.4 的停止条件问题，这是「生成固定 163.84 秒音频」场景下的 RTF，并不直接代表真实使用场景（短句子）下的体验。
- TTFP ≈ 37.6 秒看起来很差，但这并非延迟优化的问题，而是 §2.3 提到的「MOSS-TTS 全系列流式分块目前被禁用，必须等整段生成完才能开始 codec 解码并返回第一个音频包」的既有架构限制，所有 MOSS-TTS 变体（Delay/Realtime/Local）目前都受此限制，不是 v1.5 独有的新问题。

## 4. 后续建议

1. **修复 stop 条件 bug**（§2.4）：确认二分类停止头（`local_text_lm_head`）的判定逻辑、采样温度，以及 checkpoint 中该权重的行顺序是否正确，使生成长度能随文本内容自适应，而不是每次都跑满硬上限。修完后应重新跑 benchmark，得到有意义的 RTF/TTFP 数据。
2. 若需要降低 TTFP，需要解除 §2.3 提到的「累积到结束才解码」限制，给 v2 codec 的因果解码补上左上下文（left-context）支持，实现真正的增量流式分块解码——这是影响全部 MOSS-TTS 变体的通用工作，工作量较大，建议单独立项。
