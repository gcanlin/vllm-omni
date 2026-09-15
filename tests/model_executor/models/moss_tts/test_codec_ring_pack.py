# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.model_executor.models.moss_tts.codec_kernels import pack_ring_kv

pytestmark = [pytest.mark.core_model, pytest.mark.cuda]


@pytest.mark.parametrize(
    "batch,heads,frames,capacity", [(1, 20, 1, 125), (2, 20, 15, 125), (8, 20, 15, 125), (32, 12, 32, 400)]
)
@torch.inference_mode()
def test_pack_ring_kv_state_and_replay(batch, heads, frames, capacity):
    torch.manual_seed(17)
    k = torch.randn(batch, frames, heads, 64, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    v = torch.randn_like(k)
    cache = torch.randn(2, batch + 3, heads, capacity, 64, device="cuda", dtype=k.dtype)
    reference = cache.clone()
    slots = torch.randperm(batch + 3, device="cuda")[:batch]
    offsets = torch.randint(0, 4 * capacity, (batch + 3,), device="cuda")

    def expected():
        packed = reference.index_select(1, slots)
        indices = (offsets[slots, None] + torch.arange(frames, device="cuda")) % capacity
        indices = indices[:, None, :, None].expand(batch, heads, frames, 64)
        packed[0].scatter_(2, indices, k)
        packed[1].scatter_(2, indices, v)
        reference.index_copy_(1, slots, packed)
        return packed

    for _ in range(3):
        result = pack_ring_kv(k, v, cache, offsets, slots)
        torch.testing.assert_close(result, expected(), rtol=0, atol=0)
        torch.testing.assert_close(cache, reference, rtol=0, atol=0)
        offsets.add_(frames)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        pack_ring_kv(k, v, cache, offsets, slots)
    torch.cuda.current_stream().wait_stream(stream)
    expected()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = pack_ring_kv(k, v, cache, offsets, slots)
    for _ in range(3):
        slots.copy_(torch.randperm(batch + 3, device="cuda")[:batch])
        offsets.add_(frames)
        k.normal_()
        v.normal_()
        graph.replay()
        torch.testing.assert_close(captured, expected(), rtol=0, atol=0)
        torch.testing.assert_close(cache, reference, rtol=0, atol=0)


@torch.inference_mode()
def test_compiled_ring_complete_preserves_state_and_positions():
    from vllm_omni.model_executor.models.moss_tts.audio_tokenizer_v2 import RingKVCache, StreamingExecutionContext

    capacity, heads, dim = 125, 2, 64
    cache = RingKVCache(12, heads, dim, capacity, device=torch.device("cuda"), dtype=torch.bfloat16)
    reference = cache.cache.clone()
    ref_offsets = cache.end_offset.clone()

    def run(k, v, slots, valid):
        result = cache.complete(k, v, execution_context=StreamingExecutionContext(slots, valid))
        return result.keys, result.values, result.positions

    compiled = torch.compile(run, fullgraph=True, dynamic=True)
    for batch, frames in [(2, 1), (4, 15), (2, 15), (4, 1)]:
        k = torch.randn(batch, heads, frames, dim, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        slots = torch.randperm(12, device="cuda")[:batch]
        valid = torch.arange(batch, device="cuda") % 2 == 0
        # Exercise wraparound, inactive graph-padding rows and changing slots.
        cache.end_offset.fill_(capacity - 1)
        ref_offsets.copy_(cache.end_offset)
        old = ref_offsets[slots]
        expected = reference.index_select(1, slots)
        indices = (old[:, None] + torch.arange(frames, device="cuda")) % capacity
        indices = indices[:, None, :, None].expand(batch, heads, frames, dim)
        expected[0].scatter_(2, indices, k)
        expected[1].scatter_(2, indices, v)
        reference.index_copy_(1, slots, expected)
        last = old[:, None] + frames - 1
        columns = torch.arange(capacity, device="cuda")
        delta = columns - last % capacity
        positions = torch.where(delta <= 0, last + delta, last + delta - capacity)
        next_offset = torch.where(valid, old + frames, old)
        positions = torch.where(columns >= next_offset[:, None], -1, positions)
        ref_offsets.index_copy_(0, slots, next_offset)

        actual_k, actual_v, actual_pos = compiled(k, v, slots, valid)
        torch.testing.assert_close(actual_k, expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual_v, expected[1], rtol=0, atol=0)
        torch.testing.assert_close(actual_pos, positions, rtol=0, atol=0)
        torch.testing.assert_close(cache.cache, reference, rtol=0, atol=0)
        torch.testing.assert_close(cache.end_offset, ref_offsets, rtol=0, atol=0)
