# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
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

from vllm_omni.diffusion.distributed.fsdp.config import FSDPInferenceConfig
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)


def apply_fsdp_to_model(
    model: nn.Module,
    fsdp_config: FSDPInferenceConfig,
) -> nn.Module:
    """
    Apply FSDP sharding to a model that already has weights loaded.

    This function redistributes the model's parameters across GPUs using FSDP.
    The model should already have its weights loaded via the standard load_weights method.

    Args:
        model: Model instance with weights already loaded
        fsdp_config: FSDP configuration

    Returns:
        FSDP-wrapped model ready for inference
    """
    if not fsdp_config.enabled:
        raise ValueError("FSDP is not enabled in config")

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    hsdp_replicate_dim = fsdp_config.hsdp_replicate_dim
    hsdp_shard_dim = fsdp_config.hsdp_shard_dim
    if hsdp_shard_dim == -1:
        hsdp_shard_dim = world_size // hsdp_replicate_dim

    assert hsdp_replicate_dim * hsdp_shard_dim == world_size, (
        f"HSDP dimensions ({hsdp_replicate_dim} × {hsdp_shard_dim}) "
        f"must equal world_size ({world_size})"
    )

    logger.info(
        "FSDP Inference: replicate_dim=%d, shard_dim=%d, world_size=%d, rank=%d",
        hsdp_replicate_dim,
        hsdp_shard_dim,
        world_size,
        rank,
    )

    mp_policy = MixedPrecisionPolicy(
        param_dtype=fsdp_config.param_dtype,
        reduce_dtype=fsdp_config.reduce_dtype,
        output_dtype=fsdp_config.output_dtype,
        cast_forward_inputs=False,
    )

    device_type = current_omni_platform.device_type
    device_mesh = init_device_mesh(
        device_type,
        mesh_shape=(hsdp_replicate_dim, hsdp_shard_dim),
        mesh_dim_names=("replicate", "shard"),
    )

    fsdp_shard_conditions = getattr(model, "_fsdp_shard_conditions", None)
    if not fsdp_shard_conditions:
        raise ValueError(
            f"Model {type(model).__name__} has no _fsdp_shard_conditions defined"
        )

    # Apply FSDP sharding, this will automatically handle weight distribution
    shard_model(
        model,
        cpu_offload=fsdp_config.cpu_offload,
        reshard_after_forward=fsdp_config.reshard_after_forward,
        mp_policy=mp_policy,
        mesh=device_mesh,
        fsdp_shard_conditions=fsdp_shard_conditions,
        pin_cpu_memory=fsdp_config.pin_cpu_memory,
    )

    for param in model.parameters():
        param.requires_grad = False

    logger.info("FSDP applied to model: %s", type(model).__name__)
    return model


def shard_model(
    model: nn.Module,
    *,
    cpu_offload: bool,
    reshard_after_forward: bool = True,
    mp_policy: MixedPrecisionPolicy | None = None,
    mesh: DeviceMesh | None = None,
    fsdp_shard_conditions: list[Callable[[str, nn.Module], bool]],
    pin_cpu_memory: bool = True,
) -> None:
    """Apply FSDP sharding to model modules based on shard conditions."""
    fsdp_kwargs: dict[str, Any] = {
        "reshard_after_forward": reshard_after_forward,
        "mesh": mesh,
        "mp_policy": mp_policy,
    }

    if cpu_offload:
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy(pin_memory=pin_cpu_memory)

    num_sharded = 0
    for name, module in reversed(list(model.named_modules())):
        if any(cond(name, module) for cond in fsdp_shard_conditions):
            fully_shard(module, **fsdp_kwargs)
            num_sharded += 1

    if num_sharded == 0:
        raise ValueError("No modules were sharded")

    fully_shard(model, **fsdp_kwargs)
    logger.info("Sharded %d modules + root", num_sharded)
