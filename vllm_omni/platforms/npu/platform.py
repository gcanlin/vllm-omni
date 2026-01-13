"""
NPU/Ascend implementation of OmniPlatform.

Uses multiple inheritance to combine:
- OmniPlatform: Omni-specific interfaces
- NPUPlatform: vllm-ascend's NPU platform implementation
"""
import torch
from vllm.logger import init_logger
from vllm_ascend.platform import NPUPlatform

from vllm_omni.platforms.interface import OmniPlatform, OmniPlatformEnum

logger = init_logger(__name__)


class NPUOmniPlatform(OmniPlatform, NPUPlatform):
    """NPU/Ascend implementation of OmniPlatform.

    Inherits all NPU-specific implementations from vllm-ascend's NPUPlatform,
    and adds Omni-specific interfaces from OmniPlatform.
    """

    _omni_enum = OmniPlatformEnum.NPU

    @classmethod
    def get_omni_ar_worker_cls(cls) -> str:
        return "vllm_omni.platforms.npu.worker.npu_ar_worker.NPUARWorker"

    @classmethod
    def get_omni_generation_worker_cls(cls) -> str:
        return "vllm_omni.platforms.npu.worker.npu_generation_worker.NPUGenerationWorker"

    @classmethod
    def get_default_stage_config_path(cls) -> str:
        return "vllm_omni/platforms/npu/stage_configs"

    # Diffusion attention backend configuration for NPU
    # NPU uses Ascend-specific backend, with SDPA as fallback
    _DIFFUSION_BACKEND_CONFIG = {
        "ASCEND": "vllm_omni.platforms.npu.attention.ascend_attn.AscendAttentionBackend",
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
                    "Using diffusion attention backend '%s' for NPU",
                    backend_upper,
                )
                return cls._DIFFUSION_BACKEND_CONFIG[backend_upper]
            raise ValueError(
                f"Invalid diffusion attention backend '{selected_backend}' for NPU. "
                f"Valid backends: {list(cls._DIFFUSION_BACKEND_CONFIG.keys())}"
            )

        logger.info("Using Ascend attention backend for diffusion (NPU)")
        return cls._DIFFUSION_BACKEND_CONFIG["ASCEND"]

    @classmethod
    def get_torch_device(cls, local_rank: int | None = None) -> torch.device:
        if local_rank is None:
            return torch.device("npu")
        return torch.device("npu", local_rank)

    @classmethod
    def get_device_count(cls) -> int:
        return torch.npu.device_count()

    @classmethod
    def get_device_version(cls) -> str | None:
        return None

    @classmethod
    def synchronize(cls) -> None:
        torch.npu.synchronize()
