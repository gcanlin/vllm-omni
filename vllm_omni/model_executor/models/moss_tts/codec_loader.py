# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared MOSS codec loading, independent of stage and scheduler configuration."""

from functools import partial

import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.config import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import DefaultModelLoader
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.utils.torch_utils import set_default_torch_dtype

from .audio_tokenizer import MossAudioTokenizerConfig, MossAudioTokenizerModel
from .audio_tokenizer_v2 import MossAudioTokenizerModel as MossAudioTokenizerV2Model
from .configuration_moss_audio_tokenizer_v2 import MossAudioTokenizerConfig as MossAudioTokenizerV2Config

logger = init_logger(__name__)


def load_codec(
    codec_path: str,
    *,
    device: torch.device,
    load_config: LoadConfig,
    num_quantizers: int,
    attention_backend: str = "sdpa",
    skip_empty_tiles: bool = False,
    fused_slots: bool = False,
) -> tuple[PretrainedConfig, nn.Module]:
    """Load and prepare a codec without creating stream state or CUDA graphs."""
    logger.info("Loading MOSS Audio Tokenizer from %s directly on %s", codec_path, device)

    # This codec comes from a secondary checkpoint, so it cannot be built
    # by the outer stage's normal initialize_model() call. Re-enter the
    # same target-device/default-dtype contexts used by vLLM's
    # BaseModelLoader.load_model() instead of constructing an 8 GiB FP32
    # model on CPU and copying the whole module to the GPU afterwards.
    with set_default_torch_dtype(torch.float32):
        with device:
            codec_cfg, codec = _build_codec(codec_path)

    model_loader = DefaultModelLoader(load_config)
    source = DefaultModelLoader.Source(
        model_or_path=codec_path,
        revision=None,
        subfolder=None,
    )
    codec_weights = model_loader._get_weights_iterator(source)
    params_dict = dict(codec.named_parameters())

    # Upstream MossAudioTokenizer uses different submodule names than the
    # vendored re-implementation in ``audio_tokenizer.py``. Without this
    # remap only ~half the codec parameters load (codebooks + WN convs)
    # and the rest stay at their random init, which produces noise that
    # sounds correct in duration but is structurally garbage.
    _SUFFIX_REMAP: list[tuple[str, str]] = [
        # v1 (MOSS-Audio-Tokenizer) naming.
        (".self_attn.in_projs.0.", ".attn.in_proj."),
        (".self_attn.out_projs.0.", ".attn.out_proj."),
        (".linear1.", ".ff1."),
        (".linear2.", ".ff2."),
        # v2 checkpoint names use singular in_proj/out_proj and ffn.{0,2};
        # the vendored module keeps the original MOSS layer names.
        (".self_attn.in_proj.", ".self_attn.in_projs.0."),
        (".self_attn.out_proj.", ".self_attn.out_projs.0."),
        (".ffn.0.", ".linear1."),
        (".ffn.2.", ".linear2."),
        (".layer_scale_1.", ".ls1."),
        (".layer_scale_2.", ".ls2."),
        (".input_proj.", ".in_proj."),
        (".output_proj.", ".out_proj."),
    ]

    def _remap(name: str) -> str:
        for src, dst in _SUFFIX_REMAP:
            if src in name:
                return name.replace(src, dst)
        return name

    loaded_names: set[str] = set()
    skipped: list[str] = []
    shape_mismatches: list[tuple[str, str, tuple[int, ...], tuple[int, ...]]] = []
    for name, tensor in codec_weights:
        # Try direct name first (e.g. ``quantizer.input_proj.*`` exists
        # under the same name in both layouts), then the remap (transformer
        # submodules need ``.linear1.``→``.ff1.`` etc.).
        tgt = name if name in params_dict else _remap(name)
        if tgt in params_dict:
            expected_shape = tuple(params_dict[tgt].shape)
            actual_shape = tuple(tensor.shape)
            if expected_shape != actual_shape:
                shape_mismatches.append((name, tgt, actual_shape, expected_shape))
                continue
            default_weight_loader(params_dict[tgt], tensor)
            loaded_names.add(tgt)
        else:
            skipped.append(name)

    missing = sorted(set(params_dict) - loaded_names)
    if missing or skipped or shape_mismatches:
        raise RuntimeError(
            "MOSS Audio Tokenizer weights were not fully loaded: "
            f"loaded={len(loaded_names)}/{len(params_dict)} "
            f"missing={len(missing)} skipped={len(skipped)} "
            f"shape_mismatches={len(shape_mismatches)}; "
            f"first_missing={missing[:5]} "
            f"first_skipped={skipped[:5]} "
            f"first_shape_mismatches={shape_mismatches[:3]}"
        )
    logger.info(
        "MOSS Audio Tokenizer weights: loaded=%d/%d skipped=%d (first skipped: %s)",
        len(loaded_names),
        len(params_dict),
        len(skipped),
        skipped[:3] if skipped else "none",
    )

    codec.eval()
    # The v1 quantizer emits FP32 tensors, so its decoder must remain FP32.
    if device.type != "cpu" and isinstance(codec, MossAudioTokenizerV2Model):
        codec.decoder.to(dtype=torch.bfloat16)
    if attention_backend != "sdpa":
        if attention_backend not in {"triton", "triton_slot"} or device.type != "cuda":
            raise ValueError(f"Unsupported codec attention backend/device: {attention_backend}/{device.type}")
        from vllm_omni.model_executor.models.moss_tts.audio_tokenizer_v2 import MossAudioTokenizerMultiheadAttention
        from vllm_omni.model_executor.models.moss_tts.streaming_attention import masked_attention

        if attention_backend == "triton_slot":
            from vllm_omni.model_executor.models.moss_tts.slot_attention import (
                slot_ring_attention,
                slot_ring_attention_rows,
            )

            if skip_empty_tiles and fused_slots:
                raise ValueError("codec_skip_empty_attention_tiles requires unfused slot attention")
            if skip_empty_tiles:
                slot_ring_attention = partial(slot_ring_attention, skip_empty_tiles=True)
                logger.info("MOSS codec slot attention: skipping empty tiles for T <= 32")

        for module in codec.decoder.modules():
            if isinstance(module, MossAudioTokenizerMultiheadAttention):
                module._streaming_attention = masked_attention
                if attention_backend == "triton_slot":
                    module._slot_attention = slot_ring_attention
                    if fused_slots:
                        module._slot_attention_rows = slot_ring_attention_rows
        logger.info("Enabled codec attention backend=%s", attention_backend)
    build_decode_lut = getattr(codec.quantizer, "build_decode_lut", None)
    if callable(build_decode_lut):
        lut_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        build_decode_lut(num_quantizers, dtype=lut_dtype)
        lut = codec.quantizer._decode_lut
        logger.info(
            "MOSS Audio Tokenizer LFQ decoded LUT: shape=%s dtype=%s size=%.1f MiB",
            tuple(lut.shape),
            lut.dtype,
            lut.numel() * lut.element_size() / (1024**2),
        )
    return codec_cfg, codec


def _build_codec(codec_path: str) -> tuple[PretrainedConfig, nn.Module]:
    config_dict, _ = MossAudioTokenizerV2Config.get_config_dict(codec_path)
    is_v2 = config_dict.get("number_channels", 1) >= 2

    if is_v2:
        try:
            codec_cfg = MossAudioTokenizerV2Config.from_pretrained(codec_path)
            codec = MossAudioTokenizerV2Model(codec_cfg)
            logger.info("Using vendored MOSS Audio Tokenizer v2 classes from %s", codec_path)
            return codec_cfg, codec
        except Exception:
            logger.exception(
                "Failed to instantiate vendored MOSS Audio Tokenizer v2; falling back to legacy vendored codec."
            )

    codec_cfg = MossAudioTokenizerConfig.from_pretrained(codec_path)
    codec = MossAudioTokenizerModel(codec_cfg)
    logger.info("Using vendored MOSS Audio Tokenizer v1 classes from %s", codec_path)
    return codec_cfg, codec
