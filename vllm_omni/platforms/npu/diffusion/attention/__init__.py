"""NPU/Ascend attention backends for diffusion models."""
from vllm_omni.platforms.npu.attention.ascend_attn import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
)

__all__ = ["AscendAttentionBackend", "AscendAttentionBackendImpl"]
