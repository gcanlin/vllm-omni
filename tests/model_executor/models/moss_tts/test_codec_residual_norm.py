# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest
import torch
import torch.nn.functional as F

from vllm_omni.model_executor.models.moss_tts.codec_residual_norm import residual_norm

pytestmark = [pytest.mark.core_model, pytest.mark.cuda]


@pytest.mark.parametrize("width", [768, 1280])
@pytest.mark.parametrize("rows", [1, 15, 120])
def test_residual_rounding_and_norm(width, rows):
    torch.manual_seed(17)
    x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
    update = torch.randn_like(x)
    scale = torch.randn(width, device="cuda", dtype=x.dtype) * 0.1
    weight = torch.randn_like(scale)
    bias = torch.randn_like(scale)
    ref = x + update * scale
    out, norm = residual_norm(update, x, scale, weight, bias, 1e-5)
    torch.testing.assert_close(out, ref, atol=0, rtol=0)
    expected = F.layer_norm(ref, (width,), weight, bias, 1e-5)
    torch.testing.assert_close(norm, expected, atol=0, rtol=0)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        residual_norm(update, x, scale, weight, bias, 1e-5)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_out, captured_norm = residual_norm(update, x, scale, weight, bias, 1e-5)
    x.add_(0.25)
    graph.replay()
    ref = x + update * scale
    torch.testing.assert_close(captured_out, ref, atol=0, rtol=0)
    torch.testing.assert_close(captured_norm, F.layer_norm(ref, (width,), weight, bias, 1e-5), atol=0, rtol=0)
