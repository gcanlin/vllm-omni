"""
ROCm/AMD GPU implementation of OmniPlatform.

Uses multiple inheritance to combine:
- OmniPlatform: Omni-specific interfaces
- RocmPlatform: vLLM's ROCm platform implementation
"""
import torch
from vllm.logger import init_logger
from vllm.platforms.rocm import RocmPlatform

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

    # Diffusion attention backend configuration for ROCm
    # Similar to CUDA but may have different Flash Attention support
    _DIFFUSION_BACKEND_CONFIG = {
        "FLASH_ATTN": "vllm_omni.diffusion.attention.backends.flash_attn.FlashAttentionBackend",
        "TORCH_SDPA": "vllm_omni.diffusion.attention.backends.sdpa.SDPABackend",
    }

    @classmethod
    def get_diffusion_attn_backend_cls(
        cls,
        selected_backend: str | None,
        head_size: int,
    ) -> str:
        if selected_backend is not None:
            backend_upper = selected_backend.upper()
            if backend_upper in cls._DIFFUSION_BACKEND_CONFIG:
                logger.info(
                    "Using diffusion attention backend '%s' for ROCm",
                    backend_upper,
                )
                return cls._DIFFUSION_BACKEND_CONFIG[backend_upper]
            raise ValueError(
                f"Invalid diffusion attention backend '{selected_backend}' for ROCm. "
                f"Valid backends: {list(cls._DIFFUSION_BACKEND_CONFIG.keys())}"
            )

        logger.info("Using SDPA backend for diffusion (ROCm)")
        return cls._DIFFUSION_BACKEND_CONFIG["TORCH_SDPA"]

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
