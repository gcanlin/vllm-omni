# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_ascend.worker.worker_v1 import NPUWorker

from vllm_omni.platforms.npu.worker.npu_generation_model_runner import NPUGenerationModelRunner
from vllm_omni.worker.mixins import OmniWorkerMixin


class NPUGenerationWorker(OmniWorkerMixin, NPUWorker):
    """NPU generation worker for code2wav stage in Omni model."""

    def init_device(self):
        device = self._init_device()

        self.model_runner: NPUGenerationModelRunner = NPUGenerationModelRunner(self.vllm_config, device)
