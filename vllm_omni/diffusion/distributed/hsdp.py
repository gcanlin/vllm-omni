# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)
from vllm.logger import init_logger

from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)


@dataclass
class HSDPInferenceConfig:
    """Configuration for HSDP inference.

    This is a runtime config created from DiffusionParallelConfig's HSDP settings.
    """

    enabled: bool = False
    hsdp_replicate_size: int = 1
    hsdp_shard_size: int = -1  # -1 = auto (shard across entire world)
    cpu_offload: bool = False
    pin_cpu_memory: bool = True
    param_dtype: torch.dtype = torch.bfloat16
    reduce_dtype: torch.dtype = torch.float32
    output_dtype: torch.dtype | None = None
    reshard_after_forward: bool = True


def apply_hsdp_to_model(
    model: nn.Module,
    hsdp_config: HSDPInferenceConfig,
) -> nn.Module:
    """
    Apply HSDP sharding to a model that already has weights loaded.

    This function redistributes the model's parameters across GPUs using HSDP.
    The model should already have its weights loaded via the standard load_weights method.

    Args:
        model: Model instance with weights already loaded
        hsdp_config: HSDP configuration with HSDP mesh dimensions

    Returns:
        HSDP-wrapped model ready for inference
    """
    if not hsdp_config.enabled:
        raise ValueError("HSDP is not enabled in config")

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    hsdp_replicate_size = hsdp_config.hsdp_replicate_size
    hsdp_shard_size = hsdp_config.hsdp_shard_size

    if hsdp_shard_size == -1:
        hsdp_shard_size = world_size // hsdp_replicate_size

    assert hsdp_replicate_size * hsdp_shard_size == world_size, (
        f"HSDP dimensions ({hsdp_replicate_size} × {hsdp_shard_size}) must equal world_size ({world_size})"
    )

    logger.info(
        "HSDP Inference: replicate_size=%d, shard_size=%d, world_size=%d, rank=%d",
        hsdp_replicate_size,
        hsdp_shard_size,
        world_size,
        rank,
    )

    mp_policy = MixedPrecisionPolicy(
        param_dtype=hsdp_config.param_dtype,
        reduce_dtype=hsdp_config.reduce_dtype,
        output_dtype=hsdp_config.output_dtype,
        cast_forward_inputs=False,
    )

    device_type = current_omni_platform.device_type

    # Default (replicate=1, shard=world_size) = FULL_SHARD
    device_mesh = init_device_mesh(
        device_type,
        mesh_shape=(hsdp_replicate_size, hsdp_shard_size),
        mesh_dim_names=("replicate", "shard"),
    )

    # Apply HSDP sharding, this will automatically handle weight distribution
    shard_model(
        model,
        cpu_offload=hsdp_config.cpu_offload,
        reshard_after_forward=hsdp_config.reshard_after_forward,
        mp_policy=mp_policy,
        mesh=device_mesh,
        hsdp_shard_conditions=hsdp_shard_conditions,
        pin_cpu_memory=hsdp_config.pin_cpu_memory,
    )

    for param in model.parameters():
        param.requires_grad = False

    logger.info("HSDP applied to model: %s", type(model).__name__)
    return model


def shard_model(
    model: nn.Module,
    *,
    cpu_offload: bool,
    reshard_after_forward: bool = True,
    mp_policy: MixedPrecisionPolicy | None = None,
    mesh: DeviceMesh | None = None,
    hsdp_shard_conditions: list[Callable[[str, nn.Module], bool]],
    pin_cpu_memory: bool = True,
) -> None:
    """Apply HSDP sharding to model modules based on shard conditions."""
    hsdp_kwargs: dict[str, Any] = {
        "reshard_after_forward": reshard_after_forward,
        "mesh": mesh,
        "mp_policy": mp_policy,
    }

    if cpu_offload:
        hsdp_kwargs["offload_policy"] = CPUOffloadPolicy(pin_memory=pin_cpu_memory)

    num_sharded = 0
    for name, module in reversed(list(model.named_modules())):
        if any(cond(name, module) for cond in hsdp_shard_conditions):
            fully_shard(module, **hsdp_kwargs)
            num_sharded += 1

    if num_sharded == 0:
        raise ValueError("No modules were sharded")

    fully_shard(model, **hsdp_kwargs)
    logger.info("Sharded %d modules + root", num_sharded)
