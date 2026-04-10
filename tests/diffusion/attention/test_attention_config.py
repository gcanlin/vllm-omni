# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for per-role attention backend configuration (RFC: per-role-attention-backend).

Tests cover:
- AttentionSpec and AttentionConfig construction and serialization
- Role-aware backend resolution with category fallback
- Legacy attention_backend migration
- AttentionMetadata.extra field
"""

import pytest

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec


class TestAttentionSpec:
    def test_from_string(self):
        spec = AttentionSpec.from_dict("FLASH_ATTN")
        assert spec.backend == "FLASH_ATTN"
        assert spec.extra == {}

    def test_from_dict(self):
        spec = AttentionSpec.from_dict({"backend": "SAGE_ATTN", "extra": {"quant": "int8"}})
        assert spec.backend == "SAGE_ATTN"
        assert spec.extra == {"quant": "int8"}

    def test_from_dict_no_extra(self):
        spec = AttentionSpec.from_dict({"backend": "FLASH_ATTN"})
        assert spec.extra == {}

    def test_from_dict_invalid_type(self):
        with pytest.raises(TypeError):
            AttentionSpec.from_dict(123)


class TestAttentionConfig:
    def test_empty_config(self):
        config = AttentionConfig()
        assert config.default is None
        assert config.per_role == {}

    def test_from_legacy(self):
        config = AttentionConfig.from_legacy("FLASH_ATTN")
        assert config.default is not None
        assert config.default.backend == "FLASH_ATTN"
        assert config.per_role == {}

    def test_from_legacy_none(self):
        config = AttentionConfig.from_legacy(None)
        assert config.default is None

    def test_from_dict(self):
        data = {
            "default": {"backend": "FLASH_ATTN"},
            "per_role": {
                "self": {"backend": "SPARSE_BLOCK", "extra": {"block_size": 128}},
                "cross": "SAGE_ATTN",
            },
        }
        config = AttentionConfig.from_dict(data)
        assert config.default.backend == "FLASH_ATTN"
        assert config.per_role["self"].backend == "SPARSE_BLOCK"
        assert config.per_role["self"].extra == {"block_size": 128}
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


class TestOmniDiffusionConfigMigration:
    """Test that the legacy attention_backend field migrates correctly."""

    def test_legacy_attention_backend_migrates(self):
        from vllm_omni.diffusion.data import OmniDiffusionConfig

        config = OmniDiffusionConfig(attention_backend="SAGE_ATTN")
        assert isinstance(config.attention, AttentionConfig)
        assert config.attention.default is not None
        assert config.attention.default.backend == "SAGE_ATTN"

    def test_new_attention_config_takes_precedence(self):
        from vllm_omni.diffusion.data import OmniDiffusionConfig

        attn_cfg = AttentionConfig(default=AttentionSpec(backend="FLASH_ATTN"))
        config = OmniDiffusionConfig(
            attention_backend="SAGE_ATTN",  # legacy
            attention=attn_cfg,  # new — has a default, so legacy is ignored
        )
        assert config.attention.default.backend == "FLASH_ATTN"

    def test_dict_attention_config(self):
        from vllm_omni.diffusion.data import OmniDiffusionConfig

        config = OmniDiffusionConfig(
            attention={
                "default": {"backend": "FLASH_ATTN"},
                "per_role": {"self": "SPARSE_BLOCK"},
            }
        )
        assert config.attention.default.backend == "FLASH_ATTN"
        assert config.attention.per_role["self"].backend == "SPARSE_BLOCK"

    def test_no_attention_config_defaults_to_empty(self):
        from vllm_omni.diffusion.data import OmniDiffusionConfig

        config = OmniDiffusionConfig()
        assert isinstance(config.attention, AttentionConfig)
        assert config.attention.default is None
        assert config.attention.per_role == {}
