# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 checkpoint resolution for the AR text-encoder stage."""

from pathlib import Path

from vllm_omni.model_executor.model_loader.weight_utils import (
    download_weights_from_hf_specific,
)

MINIMAX_H3_TEXT_ENCODER_DOWNLOAD_PATTERNS = ["FL2VA/text_encoder/**"]


def resolve_minimax_h3_model_root(
    model: str,
    revision: str | None,
) -> str:
    path = Path(model)
    if path.is_dir():
        return str(path)
    return download_weights_from_hf_specific(
        model_name_or_path=model,
        cache_dir=None,
        allow_patterns=MINIMAX_H3_TEXT_ENCODER_DOWNLOAD_PATTERNS,
        revision=revision,
        require_all=True,
    )


__all__ = ["resolve_minimax_h3_model_root"]
