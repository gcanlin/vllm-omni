# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import TYPE_CHECKING

from .torch_profiler import TorchProfiler

if TYPE_CHECKING:
    from vllm.config import ProfilerConfig

    from .base import ProfilerBase

# Default profiler – can be changed later via config
CurrentProfiler: type[ProfilerBase] = TorchProfiler


def get_profiler_class(config: ProfilerConfig | None) -> type[ProfilerBase] | None:
    """Get the appropriate profiler class based on configuration.

    Args:
        config: The profiler configuration. If None or profiler is None,
                returns None (profiling disabled).

    Returns:
        The profiler class to use, or None if profiling is disabled.
    """
    if config is None or config.profiler is None:
        return None

    if config.profiler == "torch":
        return TorchProfiler
    elif config.profiler == "cuda":
        # CUDA profiler is not yet implemented for diffusion
        raise NotImplementedError("CUDA profiler is not yet implemented for diffusion models")
    else:
        raise ValueError(f"Unknown profiler type: {config.profiler}")


def configure_profiler(config: ProfilerConfig | None) -> None:
    """Configure the profiler with the given configuration.

    This should be called during worker initialization to set up the profiler
    with CLI-provided settings.

    Args:
        config: The profiler configuration.
    """
    if config is None or config.profiler is None:
        return

    if config.profiler == "torch":
        TorchProfiler.set_config(config)


__all__ = [
    "CurrentProfiler",
    "TorchProfiler",
    "configure_profiler",
    "get_profiler_class",
]
