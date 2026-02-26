# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import subprocess
from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch
from torch.profiler import ProfilerActivity, profile
from vllm.logger import init_logger

from .base import ProfilerBase

if TYPE_CHECKING:
    from vllm.config import ProfilerConfig

logger = init_logger(__name__)


class TorchProfiler(ProfilerBase):
    """
    Torch-based profiler configured for End-to-End continuous recording.
    Uses 'on_trace_ready' to handle Trace export.
    Compression is offloaded to a background subprocess to avoid blocking the worker loop.

    Now supports config-based initialization following vLLM's profiler pattern.
    """

    _profiler: profile | None = None
    _trace_template: str = ""
    _config: ProfilerConfig | None = None

    @classmethod
    def set_config(cls, config: ProfilerConfig) -> None:
        """Set the profiler configuration."""
        cls._config = config

    @classmethod
    def start(cls, trace_path_template: str) -> str:
        """
        Start the profiler with the given trace path template.
        """
        # 1. Cleanup any existing profiler
        if cls._profiler is not None:
            logger.warning("[Rank %s] Stopping existing Torch profiler", cls._get_rank())
            cls._profiler.stop()
            cls._profiler = None

        rank = cls._get_rank()

        # 2. Make path absolute
        trace_path_template = os.path.abspath(trace_path_template)
        cls._trace_template = trace_path_template

        # Expected paths
        json_file = f"{trace_path_template}_rank{rank}.json"

        os.makedirs(os.path.dirname(json_file), exist_ok=True)

        logger.info(f"[Rank {rank}] Starting End-to-End Torch profiler")

        # Get config values (use defaults if no config set)
        config = cls._config
        use_gzip = config.torch_profiler_use_gzip if config else True
        record_shapes = config.torch_profiler_record_shapes if config else False
        profile_memory = config.torch_profiler_with_memory if config else False
        with_stack = config.torch_profiler_with_stack if config else True
        with_flops = config.torch_profiler_with_flops if config else False

        # 3. Define the on_trace_ready handler
        def trace_handler(p):
            nonlocal json_file

            # A. Export JSON Trace
            try:
                p.export_chrome_trace(json_file)
                logger.info(f"[Rank {rank}] Trace exported to {json_file}")

                if use_gzip:
                    try:
                        subprocess.Popen(["gzip", "-f", json_file])
                        logger.info(f"[Rank {rank}] Triggered background compression for {json_file}")
                        # Update variable to point to the eventual file
                        json_file = f"{json_file}.gz"
                    except Exception as compress_err:
                        logger.warning(f"[Rank {rank}] Background gzip failed to start: {compress_err}")

            except Exception as e:
                logger.warning(f"[Rank {rank}] Failed to export trace: {e}")

        # 4. Initialize profiler with a large active window
        cls._profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=0,
                warmup=0,
                active=100000,
            ),
            on_trace_ready=trace_handler,
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack,
            with_flops=with_flops,
        )

        # 5. Start profiling
        cls._profiler.start()

        # Return the expected final path
        suffix = ".gz" if use_gzip else ""
        return f"{trace_path_template}_rank{rank}.json{suffix}"

    @classmethod
    def stop(cls) -> dict | None:
        if cls._profiler is None:
            return None

        rank = cls._get_rank()
        config = cls._config
        use_gzip = config.torch_profiler_use_gzip if config else True

        # Determine expected paths
        base_path = f"{cls._trace_template}_rank{rank}"
        suffix = ".gz" if use_gzip else ""
        trace_path = f"{base_path}.json{suffix}"

        try:
            # This triggers trace_handler synchronously
            # Since we removed table generation and backgrounded compression, this returns fast.
            cls._profiler.stop()
        except Exception as e:
            logger.warning(f"[Rank {rank}] Profiler stop failed: {e}")

        cls._profiler = None

        # We return the path assuming background compression will succeed (if enabled).
        return {"trace": trace_path, "table": None}

    @classmethod
    def step(cls):
        if cls._profiler is not None:
            cls._profiler.step()

    @classmethod
    def is_active(cls) -> bool:
        return cls._profiler is not None

    @classmethod
    def get_step_context(cls):
        return nullcontext()
