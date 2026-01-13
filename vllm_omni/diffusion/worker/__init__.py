# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker classes for diffusion models."""

from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker, WorkerProc

__all__ = ["DiffusionWorker", "WorkerProc"]
