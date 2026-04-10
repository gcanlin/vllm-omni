# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Diffusion attention backend selector.

This module provides the interface for selecting diffusion attention backends.
The actual backend selection logic is delegated to the platform layer
(vllm_omni.platforms), similar to how vLLM handles attention backend selection.

Usage:
    from vllm_omni.diffusion.attention.selector import get_attn_backend

    # Get the appropriate backend for current platform (legacy, no role)
    backend_cls = get_attn_backend(head_size=64)

    # Role-aware selection with AttentionConfig
    from vllm_omni.diffusion.data import AttentionConfig
    backend_cls, spec = get_attn_backend_for_role(
        role="self",
        head_size=64,
        attention_config=config,
    )
"""

from __future__ import annotations

import importlib
import os
from functools import cache, lru_cache
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
)

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec

logger = init_logger(__name__)


def _load_backend_cls(cls_path: str) -> type[AttentionBackend]:
    """Load a backend class from its fully qualified path.

    Args:
        cls_path: Fully qualified class path (e.g.,
            "vllm_omni.diffusion.attention.backends.sdpa.SDPABackend")

    Returns:
        The loaded backend class
    """
    module_path, class_name = cls_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
        backend_class = getattr(module, class_name)
        return backend_class
    except ImportError as e:
        raise ImportError(f"Failed to import module {module_path}: {e}")
    except AttributeError as e:
        raise AttributeError(f"Class {class_name} not found in module: {e}")


def _get_platform_default_backend(
    selected_backend: str | None,
    head_size: int,
) -> type[AttentionBackend]:
    """Get the platform default backend class."""
    from vllm_omni.platforms import current_omni_platform

    backend_cls_path = current_omni_platform.get_diffusion_attn_backend_cls(
        selected_backend=selected_backend,
        head_size=head_size,
    )
    return _load_backend_cls(backend_cls_path)


@cache
def get_attn_backend(head_size: int) -> type[AttentionBackend]:
    """
    Get attention backend for diffusion models (legacy API, no role awareness).

    Kept for backwards compatibility. New code should use get_attn_backend_for_role().
    """
    selected_backend = os.environ.get("DIFFUSION_ATTENTION_BACKEND")
    return _get_platform_default_backend(selected_backend, head_size)


def _freeze_extra(extra: dict[str, Any]) -> tuple:
    """Convert extra dict to a hashable tuple for caching."""
    return tuple(sorted(extra.items()))


@lru_cache(maxsize=128)
def _cached_get_backend_cls(
    backend_name: str,
    head_size: int,
) -> type[AttentionBackend]:
    """Cache backend class resolution by (backend_name, head_size)."""
    return _get_platform_default_backend(
        selected_backend=backend_name,
        head_size=head_size,
    )


def get_attn_backend_for_role(
    role: str,
    head_size: int,
    attention_config: AttentionConfig | None = None,
    role_category: str | None = None,
) -> tuple[type[AttentionBackend], AttentionSpec | None]:
    """
    Get attention backend for a specific attention role.

    Lookup precedence:
      1. attention_config.per_role[role]           — exact match
      2. attention_config.per_role[role_category]   — category fallback
      3. attention_config.default                   — global default
      4. DIFFUSION_ATTENTION_BACKEND env var        — env var fallback
      5. Platform default                           — hardware-specific

    Args:
        role: Attention role string (e.g. "self", "cross", "joint",
              "ltx2.audio_to_video")
        head_size: Head size for attention computation
        attention_config: The AttentionConfig from OmniDiffusionConfig.
            If None, falls back to legacy behavior.
        role_category: Optional category for fallback (e.g. "cross" for
            "ltx2.audio_to_video")

    Returns:
        Tuple of (backend_class, AttentionSpec or None).
        AttentionSpec is None when using platform default without explicit config.
    """
    spec = None
    if attention_config is not None:
        spec = attention_config.resolve(role=role, role_category=role_category)

    if spec is not None:
        backend_cls = _cached_get_backend_cls(spec.backend, head_size)
        return backend_cls, spec

    # Fall back to env var / platform default
    selected_backend = os.environ.get("DIFFUSION_ATTENTION_BACKEND")
    backend_cls = _get_platform_default_backend(selected_backend, head_size)
    return backend_cls, None
