# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch


@dataclass
class FSDPInferenceConfig:
    """Configuration for FSDP inference."""

    enabled: bool = False
    hsdp_replicate_dim: int = 1
    hsdp_shard_dim: int = -1  # -1 = auto
    cpu_offload: bool = False
    pin_cpu_memory: bool = True
    param_dtype: torch.dtype = torch.bfloat16
    reduce_dtype: torch.dtype = torch.float32
    output_dtype: torch.dtype | None = None
    reshard_after_forward: bool = True
