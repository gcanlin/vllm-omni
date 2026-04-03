# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch.nn as nn
from vllm.logger import init_logger

from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)


def _get_compile_backend():
    """Get the appropriate compile backend for the current platform."""
    if current_omni_platform.is_npu():
        from mindiesd.compilation import MindieSDBackend

        return MindieSDBackend()
    return None  # Use default inductor backend


def regionally_compile(model: nn.Module, *compile_args: Any, **compile_kwargs: Any) -> nn.Module:
    """
    Apply regional compilation to a PyTorch model.

    Args:
        model: The PyTorch model instance to compile
        *compile_args: Positional arguments forwarded to torch.compile
        **compile_kwargs: Keyword arguments forwarded to torch.compile

    Returns:
        The same model instance (modified in-place)
    """
    # Check if platform supports compile
    if not current_omni_platform.supports_compile():
        logger.warning(
            "Regional compilation skipped because platform %s does not support compile.",
            current_omni_platform.device_type,
        )
        return model

    # Get the list of repeated blocks from the model
    repeated_blocks = getattr(model, "_repeated_blocks", None)

    if not repeated_blocks:
        logger.warning("Regional compilation skipped because the model does not define `_repeated_blocks`.")
        return model

    # Get platform-specific backend
    backend = _get_compile_backend()
    if backend is not None:
        compile_kwargs["backend"] = backend

    # Check if we have modules with the specified class names
    has_compiled_region = False
    for submod in model.modules():
        if submod.__class__.__name__ in repeated_blocks:
            # Compile this submodule
            submod.compile(*compile_args, **compile_kwargs)
            has_compiled_region = True

    if not has_compiled_region:
        logger.warning(f"Regional compilation skipped because {repeated_blocks} classes are not found in the model.")

    return model
