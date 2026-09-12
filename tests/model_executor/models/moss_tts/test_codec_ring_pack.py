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
