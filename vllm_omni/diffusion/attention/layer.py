# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) Microsoft Corporation and Jiarui Fang
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team & Jiarui Fang
# Adapted from
# https://github.com/feifeibear/long-context-attention/blob/main/yunchang/attention/layer.py


import torch
import torch.nn as nn
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend
from vllm_omni.diffusion.attention.parallel import build_parallel_attention_strategy
from vllm_omni.diffusion.attention.parallel.base import NoParallelAttention
from vllm_omni.diffusion.attention.parallel.ring import RingParallelAttention
from vllm_omni.diffusion.attention.selector import get_attn_backend_for_role
from vllm_omni.diffusion.config import get_current_diffusion_config_or_none
from vllm_omni.diffusion.distributed.parallel_state import get_sp_group
from vllm_omni.diffusion.forward_context import get_forward_context, is_forward_context_available

logger = init_logger(__name__)


def _parse_selector_indices(selector: str | list[int] | tuple[int, ...] | set[int] | None) -> set[int] | None:
    if selector is None:
        return None
    if isinstance(selector, set):
        values = selector
    elif isinstance(selector, (list, tuple)):
        values = set(selector)
    elif isinstance(selector, str):
        text = selector.strip()
        if not text:
            return None
        values: set[int] = set()
        for chunk in text.split(","):
            token = chunk.strip()
            if not token:
                continue
            if "-" in token:
                start_str, end_str = token.split("-", 1)
                try:
                    start = int(start_str.strip())
                    end = int(end_str.strip())
                except ValueError as exc:
                    raise ValueError(f"Invalid range token '{token}' in selector '{selector}'.") from exc
                if start < 0 or end < 0 or start > end:
                    raise ValueError(f"Invalid range token '{token}' in selector '{selector}'.")
                values.update(range(start, end + 1))
            else:
                try:
                    index = int(token)
                except ValueError as exc:
                    raise ValueError(f"Invalid index token '{token}' in selector '{selector}'.") from exc
                if index < 0:
                    raise ValueError(f"Negative index '{index}' is not allowed in selector '{selector}'.")
                values.add(index)
    else:
        raise TypeError(f"Unsupported selector type: {type(selector)!r}")

    for idx in values:
        if not isinstance(idx, int):
            raise TypeError(f"Selector index must be int, got {type(idx)!r}")
        if idx < 0:
            raise ValueError("Selector indices must be non-negative.")
    return values


class Attention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        # Per-role backend selection (RFC: per-role attention backend)
        role: str = "self",
        role_category: str | None = None,
        # Model-defined Q/K/V tensor layout hint for backend execution.
        qkv_layout: str | None = None,
        # ulysses attention
        scatter_idx: int = 2,
        gather_idx: int = 1,
        use_sync: bool = False,
        skip_sequence_parallel: bool = False,
        # Opt-out for KV-cache quantization at this specific attention layer.
        # Set by the model author when quant is known to degrade quality or
        # perf for this layer (e.g. Wan2.2 cross-attn has short sequences and
        # block-FP8 quant offers no win). Default False = follow global config.
        disable_kv_quant: bool = False,
    ):
        super().__init__()

        self.role = role
        self.role_category = role_category
        self.qkv_layout = qkv_layout

        # Resolve backend via role-aware config.
        # The global diffusion config is set during model init via
        # set_current_diffusion_config(); no env-var re-parsing needed here.
        backend_kwargs: dict | None = None
        self.backend_pref = None

        config = get_current_diffusion_config_or_none()
        attention_config = config.attention_config if config is not None else None

        attn_backend_cls, spec = get_attn_backend_for_role(
            role=role,
            head_size=head_size,
            attention_config=attention_config,
            role_category=role_category,
        )
        if spec is not None:
            backend_kwargs = spec.extra or None
            self.backend_pref = spec.backend
            logger.debug("Attention(role=%s) → backend=%s", role, spec.backend)
        else:
            logger.debug("Attention(role=%s) → platform default", role)

        self.attn_backend = attn_backend_cls
        self.attn_impl_cls = self.attn_backend.get_impl_cls()
        self.attention = self.attn_impl_cls(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=softmax_scale,
            causal=causal,
            num_kv_heads=num_kv_heads,
            qkv_layout=qkv_layout,
            backend_kwargs=backend_kwargs,
        )
        # Instantiate fallback backend for float32 support
        self.sdpa_fallback = SDPABackend.get_impl_cls()(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=softmax_scale,
            causal=causal,
            num_kv_heads=num_kv_heads,
            qkv_layout=qkv_layout,
        )

        self.softmax_scale = softmax_scale
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        self.use_sync = use_sync
        self.causal = causal
        self.skip_sequence_parallel = skip_sequence_parallel

        self.use_ring = False
        self.ring_pg = None
        self.ring_runner = None

        if config is not None:
            if config.parallel_config.ring_degree > 1:
                self.use_ring = True
                try:
                    sp_group = get_sp_group()
                    self.ring_pg = sp_group.ring_group
                    self.ring_runner = RingParallelAttention(
                        sp_group,
                        attn_backend_pref=self.backend_pref,
                    )
                except Exception:
                    self.use_ring = False
                    self.ring_runner = None

        self.parallel_strategy = build_parallel_attention_strategy(
            scatter_idx=scatter_idx,
            gather_idx=gather_idx,
            use_sync=use_sync,
        )
        # Fallback strategy when SP is not active (outside sharded regions)
        self._no_parallel_strategy = NoParallelAttention()

        # KV cache quantization: resolved lazily in forward() because
        # forward_context is not available during model loading.
        self._kv_cache_dtype: str | None = None
        self._kv_cache_dtype_resolved: bool = False
        self._kv_cache_skip_steps: set[int] | None = None
        self._kv_cache_skip_layers: set[int] | None = None
        self._kv_cache_skip_selectors_resolved: bool = False
        # Per-layer opt-out from KV-cache quantization (set by model author).
        self._disable_kv_quant: bool = disable_kv_quant

    def _get_active_parallel_strategy(self):
        """Get the parallel strategy based on current SP active state.

        Returns NoParallelAttention if we're outside an SP sharded region
        (e.g., in noise_refiner/context_refiner before unified_prepare in Z-Image).
        This avoids unnecessary SP communication for layers not covered by _sp_plan.
        """
        if self.skip_sequence_parallel:
            return self._no_parallel_strategy
        if is_forward_context_available():
            ctx = get_forward_context()
            if not ctx.sp_active:
                return self._no_parallel_strategy
        return self.parallel_strategy

    def _resolve_kv_cache_dtype(self) -> str | None:
        """Lazily resolve kv_cache_dtype from forward context."""
        if self._kv_cache_dtype_resolved:
            return self._kv_cache_dtype
        try:
            config = get_forward_context().omni_diffusion_config
            dtype = config.kv_cache_dtype
        except Exception:
            dtype = None
        if dtype:
            if not self.attn_backend.supports_kv_cache_dtype(dtype):
                logger.warning(
                    "Attention backend %s does not support kv_cache_dtype='%s'. KV quantization will be disabled.",
                    self.attn_backend.get_name(),
                    dtype,
                )
                dtype = None
            elif self.use_ring:
                raise ValueError(
                    "KV quantization is not compatible with ring attention "
                    "(ring_degree > 1). Ring kernels do not propagate quantization descale "
                    "factors. Use Ulysses SP instead."
                )
        self._kv_cache_dtype = dtype
        self._kv_cache_dtype_resolved = True
        return dtype

    def _resolve_kv_cache_skip_selectors_from_config(self) -> tuple[set[int] | None, set[int] | None]:
        if self._kv_cache_skip_selectors_resolved:
            return self._kv_cache_skip_steps, self._kv_cache_skip_layers
        try:
            config = get_forward_context().omni_diffusion_config
        except Exception:
            return self._kv_cache_skip_steps, self._kv_cache_skip_layers
        self._kv_cache_skip_steps = _parse_selector_indices(config.kv_cache_skip_steps)
        self._kv_cache_skip_layers = _parse_selector_indices(config.kv_cache_skip_layers)
        self._kv_cache_skip_selectors_resolved = True
        return self._kv_cache_skip_steps, self._kv_cache_skip_layers

    def _should_apply_kv_cache_quant(self, attn_metadata: AttentionMetadata | None) -> bool:
        skip_steps = self._kv_cache_skip_steps
        skip_layers = self._kv_cache_skip_layers
        if skip_steps is not None:
            step_idx = attn_metadata.extra.get("denoise_step_idx") if attn_metadata is not None else None
            if step_idx is not None and step_idx in skip_steps:
                return False
        if skip_layers is not None:
            layer_idx = attn_metadata.extra.get("layer_idx") if attn_metadata is not None else None
            if layer_idx is not None and layer_idx in skip_layers:
                return False
        return True

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        # Get the appropriate parallel strategy based on SP active state
        strategy = self._get_active_parallel_strategy()

        # 1. Prepare inputs (Communication / Resharding)
        # For Ulysses: AllToAll Q/K/V; Slicing joint_q/k/v
        # For Ring: Concat joint_q
        query, key, value, attn_metadata, ctx = strategy.pre_attention(query, key, value, attn_metadata)

        # Resolve kv_cache runtime knobs only once after forward context is available.
        if not self._kv_cache_dtype_resolved:
            kv_cache_dtype = self._resolve_kv_cache_dtype()
        else:
            kv_cache_dtype = self._kv_cache_dtype
        if not self._kv_cache_skip_selectors_resolved:
            self._resolve_kv_cache_skip_selectors_from_config()
        if kv_cache_dtype is not None and not self._disable_kv_quant:
            if attn_metadata is None:
                attn_metadata = AttentionMetadata(extra={"kv_cache_dtype": kv_cache_dtype})
            elif self._should_apply_kv_cache_quant(attn_metadata):
                attn_metadata.extra["kv_cache_dtype"] = kv_cache_dtype

        # 2. Kernel Execution (Computation)
        if self.use_ring and strategy is not self._no_parallel_strategy:
            out = self._run_ring_attention(query, key, value, attn_metadata)
        else:
            out = self._run_local_attention(query, key, value, attn_metadata)

        # 3. Post-processing (Reverse Communication)
        # For Ulysses: AllToAll Output, and AllGather Joint Output
        out = strategy.post_attention(out, ctx)

        return out

    def _run_local_attention(self, query, key, value, attn_metadata):
        if query.dtype == torch.float32:
            logger.warning_once(
                f"Only SDPA supports float32. Overriding user config {type(self.attention)} "
                f"attention_backend='{self.backend_pref}' to 'sdpa' for dtype={query.dtype}."
            )
            return self.sdpa_fallback.forward(query, key, value, attn_metadata)

        # Fallback to standard attention
        return self.attention.forward(query, key, value, attn_metadata)

    def _run_ring_attention(self, query, key, value, attn_metadata):
        # Delegate to RingParallelAttention strategy if available
        if self.ring_runner is not None:
            return self.ring_runner.run_attention(
                query, key, value, attn_metadata, softmax_scale=self.softmax_scale, causal=self.causal
            )

        raise RuntimeError("Ring attention is enabled but strategy is not RingParallelAttention")
