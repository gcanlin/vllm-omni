# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare PDL on/off across dependent kernels and repeated graph replays."""

import pytest
import torch

from vllm_omni.model_executor.models.moss_tts import codec_pdl
from vllm_omni.model_executor.models.moss_tts.codec_fused_ops import codec_causal_mask, codec_rope_unpack_qkv
from vllm_omni.model_executor.models.moss_tts.codec_gemm import ffn_gemm
from vllm_omni.model_executor.models.moss_tts.codec_kernels import pack_ring_kv
from vllm_omni.model_executor.models.moss_tts.streaming_attention import masked_attention

pytestmark = [pytest.mark.core_model, pytest.mark.cuda]


def _capture(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = fn()
    return graph, outputs


@pytest.mark.parametrize("batch,heads,frames,capacity", [(1, 20, 1, 125), (8, 20, 15, 125), (32, 12, 32, 400)])
@torch.inference_mode()
def test_pdl_attention_chain_replay(monkeypatch, batch, heads, frames, capacity):
    monkeypatch.setattr(codec_pdl, "ENABLED", True)
    if not codec_pdl.enabled():
        pytest.skip("PDL requires CUDA SM90+")
    torch.manual_seed(79)
    projected = torch.randn(batch, frames, 3, heads, 64, device="cuda", dtype=torch.bfloat16)
    offsets = torch.full((batch + 3,), capacity - frames // 2, device="cuda", dtype=torch.long)
    slots = torch.randperm(batch + 3, device="cuda")[:batch]
    valid = torch.ones(batch, device="cuda", dtype=torch.bool)
    initial = torch.randn(2, batch + 3, heads, capacity, 64, device="cuda", dtype=projected.dtype)
    caches = [initial.clone(), initial.clone()]
    captured = []
    for enabled, cache in zip([False, True], caches):
        monkeypatch.setattr(codec_pdl, "ENABLED", enabled)

        def run():
            cache.copy_(initial)
            offset = offsets.index_select(0, slots)
            q, k, v = codec_rope_unpack_qkv(projected, offset, 10000.0)
            packed = pack_ring_kv(k, v, cache, offsets, slots)
            mask = codec_causal_mask(offset, offset, valid, frames, capacity, capacity)
            out = masked_attention(q, packed[0], packed[1], mask)
            return out, packed, mask

        captured.append(_capture(run))
    for _ in range(5):
        projected.normal_()
        offsets.add_(frames)
        slots.copy_(torch.randperm(batch + 3, device="cuda")[:batch])
        valid.copy_(torch.rand(batch, device="cuda") > 0.3)
        for graph, _ in captured:
            graph.replay()
        for a, b in zip(captured[0][1], captured[1][1]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(caches[0], caches[1], rtol=0, atol=0)


@pytest.mark.parametrize("split", [1, 4])
@pytest.mark.parametrize("mode", [0, 1, 2])
@torch.inference_mode()
def test_pdl_ffn_replay(monkeypatch, split, mode):
    monkeypatch.setattr(codec_pdl, "ENABLED", True)
    if not codec_pdl.enabled():
        pytest.skip("PDL requires CUDA SM90+")
    x = torch.randn(15, 768, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(3072, 768, device="cuda", dtype=x.dtype) / 28
    scale = torch.randn(3072, device="cuda", dtype=x.dtype) * 0.01
    residual = torch.randn(15, 3072, device="cuda", dtype=x.dtype)
    captured = []
    for enabled in [False, True]:
        monkeypatch.setattr(codec_pdl, "ENABLED", enabled)
        captured.append(_capture(lambda: ffn_gemm(x, w, scale, residual, mode, 16, 64, 64, split)))
    for _ in range(5):
        x.normal_()
        residual.normal_()
        for graph, _ in captured:
            graph.replay()
        torch.testing.assert_close(captured[0][1], captured[1][1], rtol=0, atol=0)
