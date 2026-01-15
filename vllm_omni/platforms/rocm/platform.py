# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.logger import init_logger
from vllm.platforms.rocm import RocmPlatform

from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
from vllm_omni.platforms.interface import OmniPlatform, OmniPlatformEnum

logger = init_logger(__name__)


class RocmOmniPlatform(OmniPlatform, RocmPlatform):
    """ROCm/AMD GPU implementation of OmniPlatform.

    Inherits all ROCm-specific implementations from vLLM's RocmPlatform,
    and adds Omni-specific interfaces from OmniPlatform.
    """

    _omni_enum = OmniPlatformEnum.ROCM

    @classmethod
    def get_omni_ar_worker_cls(cls) -> str:
        return "vllm_omni.worker.gpu_ar_worker.GPUARWorker"

    @classmethod
    def get_omni_generation_worker_cls(cls) -> str:
        return "vllm_omni.worker.gpu_generation_worker.GPUGenerationWorker"

    @classmethod
    def get_diffusion_attn_backend_cls(
        cls,
        selected_backend: str | None,
        head_size: int,
    ) -> str:
        if selected_backend is not None:
            backend_upper = selected_backend.upper()
            backend = DiffusionAttentionBackendEnum[backend_upper]
            logger.info("Using diffusion attention backend '%s' for ROCm", backend_upper)
            return backend.get_path()

        logger.info("Using SDPA backend for diffusion")
        return DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()

    @classmethod
    def get_default_stage_config_path(cls) -> str:
        return "vllm_omni/platforms/rocm/stage_configs"

    @classmethod
    def get_torch_device(cls, local_rank: int | None = None) -> torch.device:
        if local_rank is None:
            return torch.device("cuda")
        return torch.device("cuda", local_rank)

    @classmethod
    def get_device_count(cls) -> int:
        return torch.cuda.device_count()

    @classmethod
    def get_device_version(cls) -> str | None:
        if torch.version.hip is not None:
            hip_version = torch.version.hip
            return hip_version.split("-")[0]
        return None

    @classmethod
    def synchronize(cls) -> None:
        torch.cuda.synchronize()

    @classmethod
    def get_free_memory(cls, device: torch.device | None = None) -> int:
        free, _ = torch.cuda.mem_get_info(device)
        return free
