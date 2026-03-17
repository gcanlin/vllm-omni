# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Base NPU worker class for vLLM-Omni with OmniProfiler support."""

import time

from vllm_ascend.worker.worker import NPUWorker


class OmniNPUWorkerBase(NPUWorker):
    """Base NPU worker for vLLM-Omni with OmniProfiler support.

    This class replaces vllm-ascend's profiler with OmniProfiler for
    unified profiling across all platforms.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Replace vllm-ascend's profiler with OmniProfiler
        profiler_config = self.vllm_config.profiler_config
        if profiler_config and profiler_config.profiler == "torch":
            from vllm_omni.profiler import create_omni_profiler

            worker_name = f"npu-rank-{self.rank}"
            self.profiler = create_omni_profiler(
                profiler_config=profiler_config,
                worker_name=worker_name,
                local_rank=self.local_rank,
            )

    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        """Override to set trace filename before starting the profiler.

        NPUWorker's profile() accepts profile_prefix, so we use it to generate
        a descriptive trace filename for OmniProfiler.
        """
        if self.profiler is None:
            raise RuntimeError("Profiling is not enabled. Please set --profiler-config to enable profiling.")
        if is_start:
            from vllm_omni.profiler import OmniTorchProfilerWrapper

            if isinstance(self.profiler, OmniTorchProfilerWrapper):
                prefix = f"{profile_prefix}_" if profile_prefix else ""
                filename = f"{prefix}npu_{int(time.time())}"
                self.profiler.set_trace_filename(filename)
            self.profiler.start()
        else:
            self.profiler.stop()
