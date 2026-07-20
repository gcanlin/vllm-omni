以部分掩盖，但不能完全靠调度掩盖。

  这段链路是强依赖：

  local MTP graph -> 写 inputs_embeds -> backbone graph replay

  backbone 的输入 embedding 依赖 local 生成的 audio embedding，所以 backbone 不能提前在 GPU 上跑。local graph 结束到 backbone
  graph launch 之间的 CPU replay/调度 gap，本质是两个独立 CUDA graph launch 之间的 host gap。只要还是两个 graph，就一定有一个
  launch 间隙。

  能做的有三类：

  1. 把 local + backbone 合成一个 graph
     这是唯一能真正消掉中间 host bubble 的方式。也就是捕获：

     talker_mtp -> inputs_embeds update -> backbone forward
     成一个完整 decode graph。
     但这会更侵入：backbone graph 当前由 vLLM wrapper 管，local MTP 是我们额外包的 wrapper；要合并就得改 runner 的 capture 边
     界，不能简单嵌套两个 CUDAGraphWrapper。
     界，不能简单嵌套两个 CUDAGraphWrapper。

  2. 把 CPU postprocess 延后，避免它挡在 backbone 前面
     当前 _talker_mtp_forward 里：

     talker_mtp(...)
     postprocess_talker_mtp(...)
     inputs_embeds[...] = req_embeds

     postprocess_talker_mtp 不需要在 backbone 前完成，backbone 只需要 req_embeds。所以可以改成：

     talker_mtp replay
     立刻写 inputs_embeds
     立刻 launch backbone replay
     backbone 之后/并行 copy stream 再处理 mtp_outputs -> audio_state

     这不能消掉 backbone replay CPU 自身，但可以避免 postprocess_talker_mtp 的同步和 dict 更新进入首包关键空泡。

  3. 减少两个 graph launch 的 Python path
     比如 MTP graph dispatch 不走通用 _determine_batch_execution_and_padding、不重复构造 kwargs/list，固定 B=1 热路径直接取
     graph entry replay。这个能省一点 CPU，但通常不会完全消掉 1ms。

  所以如果你看到的 1ms 是 “local graph 完成后，backbone graph replay 发起前 GPU 空闲”，最根本方案是 合并 graph。短期更现实的是
  先把 postprocess_talker_mtp 从 backbone 前移走，保证这段 gap 只剩必要的 graph replay 调度，而不是夹杂 CPU 状态维护。