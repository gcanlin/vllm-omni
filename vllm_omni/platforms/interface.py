"""
vllm-omni Platform interface.

This module defines the OmniPlatform abstraction for vllm-omni.
OmniPlatform inherits from vLLM's Platform and adds Omni-specific interfaces.
Both CUDA and NPU are built-in platforms (not OOT).
"""
from abc import abstractmethod
from enum import Enum

import torch

from vllm.platforms import Platform


class OmniPlatformEnum(Enum):
    """Enum for supported Omni platforms."""
    CUDA = "cuda"
    ROCM = "rocm"
    NPU = "npu"
    XPU = "xpu"
    UNSPECIFIED = "unspecified"


class OmniPlatform(Platform):
    """
    Abstract base class for vllm-omni Platform.

    Inherits from vLLM's Platform and adds Omni-specific interfaces.
    This gives OmniPlatform all vLLM Platform capabilities plus
    Omni-specific methods.
    """

    _omni_enum: OmniPlatformEnum

    def is_npu(self) -> bool:
        return self._omni_enum == OmniPlatformEnum.NPU

    @classmethod
    @abstractmethod
    def get_omni_ar_worker_cls(cls) -> str:
        ...

    @classmethod
    @abstractmethod
    def get_omni_generation_worker_cls(cls) -> str:
        ...

    @classmethod
    @abstractmethod
    def get_default_stage_config_path(cls) -> str:
        raise NotImplementedError

    @classmethod
    def get_diffusion_attn_backend_cls(
        cls,
        selected_backend: str | None,
        head_size: int,
    ) -> str:
        """Get the diffusion attention backend class path for this platform.

        This method selects the appropriate attention backend for diffusion 
        models based on platform capabilities and user preferences.

        Args:
            selected_backend: User-selected backend name (e.g., "FLASH_ATTN",
                "TORCH_SDPA", "SAGE_ATTN"). If None, uses platform default.
            head_size: Attention head size.

        Returns:
            Fully qualified class path of the selected backend.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_torch_device(cls, local_rank: int | None = None) -> torch.device:
        """Get the torch.device for the current platform."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_device_count(cls) -> int:
        """Return the device count for the current platform."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_device_version(cls) -> str | None:
        """Return the device runtime version (e.g., CUDA/ROCm), or None."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def synchronize(cls) -> None:
        """Synchronize the current device."""
        raise NotImplementedError

    @classmethod
    def supports_torch_compile(cls) -> bool:
        return False

    @classmethod
    def get_free_memory(cls, device: torch.device | None = None) -> int:
        raise NotImplementedError
