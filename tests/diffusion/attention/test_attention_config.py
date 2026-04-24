# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for per-role attention backend configuration (RFC: per-role-attention-backend).

Tests cover:
- AttentionSpec and AttentionConfig normalization
- Role-aware backend resolution with category fallback
- OmniDiffusionConfig attention shorthand handling
- AttentionMetadata.extra field
"""

import pytest

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec, build_attention_config


class TestAttentionSpec:
    def test_construct_no_extra(self):
        spec = AttentionSpec(backend="FLASH_ATTN")
        assert spec.extra == {}

    def test_mapping_extra_normalized(self):
        spec = AttentionSpec(backend="SAGE_ATTN", extra={"quant": "int8"})
        assert spec.backend == "SAGE_ATTN"
        assert spec.extra == {"quant": "int8"}

    def test_invalid_backend_type(self):
        with pytest.raises(TypeError):
            AttentionSpec(backend=123)  # type: ignore[arg-type]


class TestAttentionConfig:
    def test_empty_config(self):
        config = AttentionConfig()
        assert config.default is None
        assert config.per_role == {}

    def test_constructor_normalizes_mappings(self):
        config = AttentionConfig(
            default={"backend": "FLASH_ATTN"},
            per_role={
                "self": {"backend": "SPARSE_BLOCK", "extra": {"block_size": 128}},
                "cross": "SAGE_ATTN",
            },
        )
        assert config.default.backend == "FLASH_ATTN"
        assert config.per_role["self"].backend == "SPARSE_BLOCK"
        assert config.per_role["self"].extra == {"block_size": 128}
        assert config.per_role["cross"].backend == "SAGE_ATTN"

    def test_constructor_flattens_nested_per_role_tree(self):
        config = AttentionConfig(
            per_role={
                "ltx2": {
                    "audio_self": {"backend": "FLASH_ATTN"},
                    "audio_to_video": {"backend": "SAGE_ATTN"},
                }
            }
        )
        assert config.per_role["ltx2.audio_self"].backend == "FLASH_ATTN"
        assert config.per_role["ltx2.audio_to_video"].backend == "SAGE_ATTN"

    def test_constructor_normalizes_auto_to_unset(self):
        config = AttentionConfig(
            default={"backend": "auto"},
            per_role={
                "self": "auto",
                "cross": {"backend": "SAGE_ATTN"},
            },
        )
        assert config.default is None
        assert "self" not in config.per_role
        assert config.per_role["cross"].backend == "SAGE_ATTN"

    def test_resolve_exact_match(self):
        config = AttentionConfig(
            default=AttentionSpec(backend="FLASH_ATTN"),
            per_role={
                "self": AttentionSpec(backend="SPARSE_BLOCK"),
                "cross": AttentionSpec(backend="SAGE_ATTN"),
            },
        )
        spec = config.resolve(role="self")
        assert spec.backend == "SPARSE_BLOCK"

        spec = config.resolve(role="cross")
        assert spec.backend == "SAGE_ATTN"

    def test_resolve_with_source_reports_match_origin(self):
        config = AttentionConfig(
            default=AttentionSpec(backend="FLASH_ATTN"),
            per_role={
                "cross": AttentionSpec(backend="SAGE_ATTN"),
                "ltx2.audio_to_video": AttentionSpec(backend="SPARSE_BLOCK"),
            },
        )

        spec, source = config.resolve_with_source(role="ltx2.audio_to_video", role_category="cross")
        assert spec is not None
        assert spec.backend == "SPARSE_BLOCK"
        assert source == "attention_config.per_role['ltx2.audio_to_video']"

        spec, source = config.resolve_with_source(role="ltx2.video_to_audio", role_category="cross")
        assert spec is not None
        assert spec.backend == "SAGE_ATTN"
        assert source == "attention_config.per_role['cross'] (role_category fallback)"

        spec, source = config.resolve_with_source(role="self")
        assert spec is not None
        assert spec.backend == "FLASH_ATTN"
        assert source == "attention_config.default"

    def test_resolve_category_fallback(self):
        config = AttentionConfig(
            default=AttentionSpec(backend="FLASH_ATTN"),
            per_role={
                "cross": AttentionSpec(backend="SAGE_ATTN"),
            },
        )
        # "ltx2.audio_to_video" falls back to category "cross"
        spec = config.resolve(role="ltx2.audio_to_video", role_category="cross")
        assert spec.backend == "SAGE_ATTN"

    def test_resolve_exact_overrides_category(self):
        config = AttentionConfig(
            per_role={
                "cross": AttentionSpec(backend="SAGE_ATTN"),
                "ltx2.audio_to_video": AttentionSpec(backend="FLASH_ATTN"),
            },
        )
        # Exact match wins over category
        spec = config.resolve(role="ltx2.audio_to_video", role_category="cross")
        assert spec.backend == "FLASH_ATTN"

    def test_resolve_default_fallback(self):
        config = AttentionConfig(
            default=AttentionSpec(backend="FLASH_ATTN"),
        )
        spec = config.resolve(role="self")
        assert spec.backend == "FLASH_ATTN"

        spec = config.resolve(role="joint")
        assert spec.backend == "FLASH_ATTN"

    def test_resolve_returns_none_when_empty(self):
        config = AttentionConfig()
        spec = config.resolve(role="self")
        assert spec is None

    def test_resolve_no_category_no_default(self):
        config = AttentionConfig(
            per_role={"self": AttentionSpec(backend="SPARSE_BLOCK")},
        )
        # Unknown role with no category and no default
        spec = config.resolve(role="joint")
        assert spec is None

    def test_full_ltx2_scenario(self):
        """Test the LTX2 6-role stress test from the RFC."""
        config = AttentionConfig(
            default=AttentionSpec(backend="FLASH_ATTN"),
            per_role={
                "self": AttentionSpec(backend="SPARSE_BLOCK", extra={"block_size": 128}),
                "cross": AttentionSpec(backend="SAGE_ATTN"),
                "ltx2.audio_self": AttentionSpec(backend="FLASH_ATTN"),
                "ltx2.audio_to_video": AttentionSpec(backend="FLASH_ATTN", extra={"causal_window": 64}),
            },
        )

        # video self → exact match "self"
        assert config.resolve("self").backend == "SPARSE_BLOCK"

        # audio self → exact match "ltx2.audio_self"
        assert config.resolve("ltx2.audio_self", "self").backend == "FLASH_ATTN"

        # video-text cross → exact match "cross"
        assert config.resolve("cross").backend == "SAGE_ATTN"

        # audio-text cross → category fallback to "cross"
        assert config.resolve("ltx2.audio_text_cross", "cross").backend == "SAGE_ATTN"

        # audio-to-video → exact match
        spec = config.resolve("ltx2.audio_to_video", "cross")
        assert spec.backend == "FLASH_ATTN"
        assert spec.extra == {"causal_window": 64}

        # video-to-audio → category fallback to "cross"
        assert config.resolve("ltx2.video_to_audio", "cross").backend == "SAGE_ATTN"


class TestAttentionMetadataExtra:
    def test_default_extra_is_empty(self):
        meta = AttentionMetadata()
        assert meta.extra == {}

    def test_extra_passthrough(self):
        import torch

        block_mask = torch.ones(4, 4)
        meta = AttentionMetadata(extra={"block_mask": block_mask, "kv_indices": [0, 1, 2]})
        assert torch.equal(meta.extra["block_mask"], block_mask)
        assert meta.extra["kv_indices"] == [0, 1, 2]

    def test_extra_does_not_affect_existing_fields(self):
        import torch

        mask = torch.ones(2, 8)
        meta = AttentionMetadata(attn_mask=mask, extra={"foo": "bar"})
        assert meta.attn_mask is mask
        assert meta.extra == {"foo": "bar"}


class TestBuildAttentionConfig:
    def test_env_sets_default_when_no_higher_priority_input(self, monkeypatch):
        monkeypatch.setenv("DIFFUSION_ATTENTION_BACKEND", "TORCH_SDPA")

        config = build_attention_config()

        assert config.default is not None
        assert config.default.backend == "TORCH_SDPA"

    def test_attention_backend_overrides_env(self, monkeypatch):
        monkeypatch.setenv("DIFFUSION_ATTENTION_BACKEND", "TORCH_SDPA")

        config = build_attention_config(attention_backend="SAGE_ATTN")

        assert config.default is not None
        assert config.default.backend == "SAGE_ATTN"

    def test_attention_backend_auto_disables_env_fallback(self, monkeypatch):
        monkeypatch.setenv("DIFFUSION_ATTENTION_BACKEND", "TORCH_SDPA")

        config = build_attention_config(attention_backend="auto")

        assert config.default is None

    def test_explicit_default_ignores_env(self, monkeypatch):
        monkeypatch.setenv("DIFFUSION_ATTENTION_BACKEND", "self=FLASH_ATTN,cross=TORCH_SDPA")

        config = build_attention_config(
            AttentionConfig(default=AttentionSpec(backend="FLASH_ATTN")),
        )

        assert config.default is not None
        assert config.default.backend == "FLASH_ATTN"

    def test_env_auto_does_not_set_default(self, monkeypatch):
        monkeypatch.setenv("DIFFUSION_ATTENTION_BACKEND", "auto")

        config = build_attention_config()

        assert config.default is None

    def test_attention_backend_conflicts_with_explicit_default(self):
        with pytest.raises(ValueError):
            build_attention_config(
                AttentionConfig(default=AttentionSpec(backend="FLASH_ATTN")),
                attention_backend="SAGE_ATTN",
            )


class TestOmniDiffusionConfigAttentionParsing:
    """Test OmniDiffusionConfig attention shorthand and structured config."""

    def test_attention_backend_sets_default(self):
        from vllm_omni.diffusion.data import OmniDiffusionConfig

        config = OmniDiffusionConfig(attention_backend="SAGE_ATTN")
        assert isinstance(config.attention_config, AttentionConfig)
        assert config.attention_config.default is not None
        assert config.attention_config.default.backend == "SAGE_ATTN"

    def test_attention_backend_auto_means_platform_default(self):
        from vllm_omni.diffusion.data import OmniDiffusionConfig

        config = OmniDiffusionConfig(attention_backend="auto")
        assert isinstance(config.attention_config, AttentionConfig)
        assert config.attention_config.default is None

    def test_attention_backend_and_default_are_mutually_exclusive(self):
        from vllm_omni.diffusion.data import OmniDiffusionConfig

        with pytest.raises(ValueError):
            OmniDiffusionConfig(
                attention_backend="SAGE_ATTN",
                attention_config=AttentionConfig(default=AttentionSpec(backend="FLASH_ATTN")),
            )

    def test_dict_attention_config(self):
        from vllm_omni.diffusion.data import OmniDiffusionConfig

        config = OmniDiffusionConfig(
            attention_config={
                "default": {"backend": "FLASH_ATTN"},
                "per_role": {"self": "SPARSE_BLOCK"},
            }
        )
        assert config.attention_config.default.backend == "FLASH_ATTN"
        assert config.attention_config.per_role["self"].backend == "SPARSE_BLOCK"

    def test_old_attention_name_raises(self):
        from vllm_omni.diffusion.data import OmniDiffusionConfig

        with pytest.raises(TypeError):
            OmniDiffusionConfig.from_kwargs(attention={})  # type: ignore[call-arg]

    def test_no_attention_config_defaults_to_empty(self):
        from vllm_omni.diffusion.data import OmniDiffusionConfig

        config = OmniDiffusionConfig()
        assert isinstance(config.attention_config, AttentionConfig)
        assert config.attention_config.default is None
        assert config.attention_config.per_role == {}
