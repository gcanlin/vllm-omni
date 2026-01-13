# Copyright 2024 xDiT team.
# Adapted from
# https://github.com/xdit-project/xDiT/blob/main/xfuser/envs.py
"""
Diffusion environment utilities.

This module provides environment checking utilities for diffusion models.
Most device-related functionality has been moved to the platform layer
(vllm_omni.platforms). For device detection and configuration, use
`current_omni_platform` from `vllm_omni.platforms`.

Example:
    from vllm_omni.platforms import current_omni_platform

    # Get device for a given rank
    device = current_omni_platform.get_torch_device(local_rank)

    # Get distributed backend
    backend = current_omni_platform.dist_backend

    # Check platform type
    if current_omni_platform.is_cuda():
        ...
"""
from vllm.logger import init_logger
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)


class PackagesEnvChecker:
    """Singleton class for checking package availability."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        packages_info = {}
        packages_info["has_flash_attn"] = self._check_flash_attn(packages_info)
        self.packages_info = packages_info

    def _check_flash_attn(self, packages_info) -> bool:
        """Check if flash attention is available and compatible."""
        platform = current_omni_platform

        # Flash attention requires CUDA-like platforms (CUDA or ROCm)
        if not platform.is_cuda_alike():
            return False

        # Check if devices are available
        if platform.get_device_count() == 0:
            return False

        try:
            gpu_name = platform.get_device_name()
            # Turing/Tesla/T4 GPUs don't support flash attention well
            if "Turing" in gpu_name or "Tesla" in gpu_name or "T4" in gpu_name:
                return False

            from flash_attn import __version__

            if __version__ < "2.6.0":
                raise ImportError("install flash_attn >= 2.6.0")
            return True
        except ImportError:
            if not packages_info.get("has_aiter", False):
                logger.warning(
                    'Flash Attention library "flash_attn" not found, '
                    "using pytorch attention implementation"
                )
            return False

    def get_packages_info(self) -> dict:
        """Get the packages info dictionary."""
        return self.packages_info


PACKAGES_CHECKER = PackagesEnvChecker()
