# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HunyuanImage3 rotary embedding layers."""

import torch
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbeddingBase


class HunyuanImage3RotaryEmbedding(RotaryEmbeddingBase):
    """Interleaved 2D RoPE used by HunyuanImage3.

    HunyuanImage3's original 2D RoPE interleaves height and width frequencies:

        [y*f0, x*f1, y*f2, x*f3, ...]

    This differs from vLLM's generic MRoPE, which assigns contiguous frequency
    blocks to each position dimension. The implementation keeps the vLLM rotary
    layer interface while preserving HunyuanImage3's pretrained layout.
    """

    def __init__(
        self,
        head_dim: int,
        rope_theta: float = 10000.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if head_dim % 4 != 0:
            raise ValueError(f"head_dim must be divisible by 4, got {head_dim}")

        super().__init__(
            head_size=head_dim,
            rotary_dim=head_dim,
            max_position_embeddings=0,
            base=rope_theta,
            is_neox_style=True,
            dtype=dtype,
            init_cache=False,
        )
        inv_freq = self._compute_inv_freq(rope_theta)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _compute_inv_freq(self, base: float) -> torch.Tensor:
        return 1.0 / (base ** (torch.arange(0, self.head_size, 2, dtype=torch.float32) / self.head_size))

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def _get_2d_positions(
        positions: torch.Tensor,
        num_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if positions.dim() == 2:
            y_pos = positions[1, :num_tokens].float()
            x_pos = positions[2, :num_tokens].float()
        else:
            y_pos = positions[:num_tokens].float()
            x_pos = positions[:num_tokens].float()
        return y_pos, x_pos

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        num_tokens = query.shape[0]
        y_pos, x_pos = self._get_2d_positions(positions, num_tokens)

        dtype = query.dtype
        query_shape = query.shape
        key_shape = key.shape if key is not None else None

        inv_freq = self.inv_freq.to(device=y_pos.device, dtype=torch.float32)
        inv_freq_y = inv_freq[0::2]
        inv_freq_x = inv_freq[1::2]

        y_freqs = y_pos.unsqueeze(-1) * inv_freq_y.unsqueeze(0)
        x_freqs = x_pos.unsqueeze(-1) * inv_freq_x.unsqueeze(0)
        freqs = torch.stack([y_freqs, x_freqs], dim=-1).flatten(-2)

        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(dtype).unsqueeze(1)
        sin = emb.sin().to(dtype).unsqueeze(1)

        query = query.reshape(num_tokens, -1, self.head_size)
        query = query * cos + self._rotate_half(query) * sin
        query = query.reshape(query_shape)

        if key is None:
            return query, None

        key = key.reshape(num_tokens, -1, self.head_size)
        key = key * cos + self._rotate_half(key) * sin
        return query, key.reshape(key_shape)

    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.forward_native(positions, query, key)

    def forward_hip(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.forward_native(positions, query, key)
