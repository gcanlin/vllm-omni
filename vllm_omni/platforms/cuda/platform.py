"""
CUDA/GPU implementation of OmniPlatform.

Uses multiple inheritance to combine:
- OmniPlatform: Omni-specific interfaces
- CudaPlatform: vLLM's CUDA platform implementation
"""
import torch
from vllm.logger import init_logger
from vllm.platforms.cuda import CudaPlatformBase

from vllm_omni.platforms.interface import OmniPlatform, OmniPlatformEnum

logger = init_logger(__name__)


class CudaOmniPlatform(OmniPlatform, CudaPlatformBase):
    """CUDA/GPU implementation of OmniPlatform (default).

    Inherits all CUDA-specific implementations from vLLM's CudaPlatform,
    and adds Omni-specific interfaces from OmniPlatform.
    """

    _omni_enum = OmniPlatformEnum.CUDA

    @classmethod
    def get_omni_ar_worker_cls(cls) -> str:
        return "vllm_omni.worker.gpu_ar_worker.GPUARWorker"

    @classmethod
    def get_omni_generation_worker_cls(cls) -> str:
        return "vllm_omni.worker.gpu_generation_worker.GPUGenerationWorker"

    # Diffusion attention backend configuration for CUDA
    # Priority order: user selection > Flash Attention > SDPA (default)
    _DIFFUSION_BACKEND_CONFIG = {
        "FLASH_ATTN": "vllm_omni.diffusion.attention.backends.flash_attn.FlashAttentionBackend",
        "TORCH_SDPA": "vllm_omni.diffusion.attention.backends.sdpa.SDPABackend",
        "SAGE_ATTN": "vllm_omni.diffusion.attention.backends.sage_attn.SageAttentionBackend",
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
                    "Using diffusion attention backend '%s' for CUDA",
                    backend_upper,
                )
                return cls._DIFFUSION_BACKEND_CONFIG[backend_upper]
            raise ValueError(
                f"Invalid diffusion attention backend '{selected_backend}' for CUDA. "
                f"Valid backends: {list(cls._DIFFUSION_BACKEND_CONFIG.keys())}"
            )

        try:
            from flash_attn import __version__

            if __version__ >= "2.6.0":
                logger.info("Using Flash Attention backend for diffusion (CUDA)")
                return cls._DIFFUSION_BACKEND_CONFIG["FLASH_ATTN"]
        except ImportError:
            pass

        logger.info("Using SDPA backend for diffusion (CUDA)")
        return cls._DIFFUSION_BACKEND_CONFIG["TORCH_SDPA"]

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
        return torch.version.cuda

    @classmethod
    def synchronize(cls) -> None:
        torch.cuda.synchronize()

    # ================== Diffusion Worker Methods ==================
    # Note: set_device() is inherited from CudaPlatformBase
    # Note: is_sleep_mode_available() is inherited from Platform base class

    @classmethod
    def supports_torch_compile(cls) -> bool:
        return True

    @classmethod
    def get_free_memory(cls, device: torch.device | None = None) -> int:
        free, _ = torch.cuda.mem_get_info(device)
        return free
