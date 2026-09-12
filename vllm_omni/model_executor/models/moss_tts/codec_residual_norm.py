# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fuse codec LayerScale/residual while retaining native LayerNorm numerics."""

import os

import torch
from vllm.triton_utils import tl, triton

ENABLED = os.getenv("MOSS_CODEC_RESIDUAL_NORM", "0") == "1"


@triton.jit
def _residual_scale(update, residual, scale, out, size: tl.constexpr, width: tl.constexpr, block: tl.constexpr):
    index = tl.program_id(0) * block + tl.arange(0, block)
    u = tl.load(update + index, index < size, 0).to(tl.float32)
    r = tl.load(residual + index, index < size, 0).to(tl.float32)
    s = tl.load(scale + index % width).to(tl.float32)
    x = (u * s).to(tl.bfloat16).to(tl.float32) + r
    tl.store(out + index, x, index < size)


@torch.library.custom_op("moss_codec::residual_norm", mutates_args=())
def residual_norm(
    update: torch.Tensor,
    residual: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    supported = all(
        t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous() for t in (update, residual, scale, weight, bias)
    )
    if supported:
        out = torch.empty_like(residual)
        _residual_scale[(triton.cdiv(residual.numel(), 256),)](
            update,
            residual,
            scale,
            out,
            residual.numel(),
            residual.shape[-1],
            256,
            enable_fp_fusion=False,
        )
    else:
        out = residual + update * scale
    # The single-kernel LayerNorm candidate changed reduction rounding enough
    # to fail multichunk waveform validation. Keep the native reduction.
    return out, torch.nn.functional.layer_norm(out, (out.shape[-1],), weight, bias, eps)


@residual_norm.register_fake
def _(update, residual, scale, weight, bias, eps):
    return torch.empty_like(residual), torch.empty_like(residual)
