# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Platform-aware dependency routing for vLLM-Omni.

This module implements install-time detection of the target hardware platform
and automatically selects the appropriate platform-specific dependencies.

Detection Priority:
1. Explicit override via VLLM_OMNI_TARGET_DEVICE environment variable
2. Torch backend detection (CUDA, ROCm, NPU, XPU)
3. Fallback to common dependencies only (treated as CPU)

Supported platforms: cuda, rocm, npu, xpu, cpu
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from setuptools import setup

try:
    import torch
except Exception:  # pragma: no cover - torch may not be installed at build time
    torch = None

ROOT = Path(__file__).parent

# Supported target devices
TargetDevice = Literal["cuda", "rocm", "npu", "xpu", "cpu"]


def _read_requirements(filename: str) -> list[str]:
    """Read and resolve requirements from a file, handling -r includes."""
    requirements_path = ROOT / "requirements" / filename
    if not requirements_path.exists():
        return []
    lines = requirements_path.read_text().splitlines()
    resolved: list[str] = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r "):
            resolved += _read_requirements(line.split()[1])
        else:
            resolved.append(line)
    return resolved


def _detect_target_device() -> TargetDevice | None:
    """
    Detect the target device platform for dependency selection.

    Priority rules:
    1. VLLM_OMNI_TARGET_DEVICE environment variable (highest priority)
    2. Torch backend detection via torch.version.cuda/hip and device availability
    3. None (fallback - only common dependencies will be installed)

    Returns:
        The detected target device, or None if no platform can be determined.
    """
    # Priority 1: Explicit environment variable override
    env_target = os.getenv("VLLM_OMNI_TARGET_DEVICE", "").lower()
    if env_target:
        valid_devices = {"cuda", "rocm", "npu", "xpu", "cpu"}
        if env_target in valid_devices:
            return env_target  # type: ignore[return-value]
        # Invalid value - log warning and continue with auto-detection
        print(
            f"Warning: Invalid VLLM_OMNI_TARGET_DEVICE='{env_target}'. "
            f"Valid values: {valid_devices}. Falling back to auto-detection."
        )

    # Priority 2: Torch backend detection
    if torch is not None:
        # Check CUDA
        if getattr(torch.version, "cuda", None) is not None:
            return "cuda"

        # Check ROCm
        if getattr(torch.version, "hip", None) is not None:
            return "rocm"

        # Check NPU
        try:
            if hasattr(torch, "npu") and torch.npu.is_available():
                return "npu"
        except Exception:
            pass

        # Check XPU
        try:
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                return "xpu"
        except Exception:
            pass

    # Priority 3: Fallback - no specific platform detected
    return None


def _get_install_requires() -> list[str]:
    """
    Build the complete list of install requirements based on detected platform.

    Always includes common.txt, then adds platform-specific dependencies
    based on the detected target device.
    """
    install_requires = _read_requirements("common.txt")

    target_device = _detect_target_device()

    if target_device is not None:
        platform_requirements_file = f"{target_device}.txt"
        platform_requirements = _read_requirements(platform_requirements_file)
        install_requires += platform_requirements

    return install_requires


install_requires = _get_install_requires()

setup(install_requires=install_requires)
