# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in programmatic dependent launch for the codec's custom CUDA kernels.

Read before graph capture; changing the environment requires a server restart.
PDL does not change which GEMM or residual-fusion implementation is selected.
"""

import os

from vllm.platforms import current_platform

ENABLED = os.getenv("MOSS_CODEC_PDL", "0") == "1"


def enabled() -> bool:
    return ENABLED and current_platform.is_cuda() and current_platform.has_device_capability(90)
