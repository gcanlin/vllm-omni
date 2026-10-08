# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MOSS Local first-frame decoding in the Talker process.

The empty-history path specializes attention and owns no stream state. The
streaming fallback reuses the codec stage's session with a private state pool.
Both retain every upsampling block; the regular codec stage independently
primes its persistent state with the same codes.
"""

from __future__ import annotations

import torch
from torch import nn
from vllm.config import VllmConfig

from .modeling_moss_tts_codec import _MossCodecStreamSession, load_codec

_FIRST_FRAME_BATCH_SIZES = (1, 2, 4, 8)


def first_audio_enabled(config) -> bool:
    """Default to first audio on supported runners; allow explicit opt-out."""
    connector = getattr(config.model_config, "stage_connector_config", {}) or {}
    extra = connector.get("extra", connector) if isinstance(connector, dict) else getattr(connector, "extra", {})
    extra = extra or {}
    enabled = extra.get("moss_talker_first_audio", True)
    if not isinstance(enabled, bool):
        raise ValueError("moss_talker_first_audio must be a boolean")
    if not enabled:
        return False
    # The system profile also serves V1 platform fallbacks. They retain
    # regular codec delivery rather than constructing a CUDA-only decoder.
    if not bool(getattr(config.model_config, "use_v2_model_runner", False)):
        return False
    model, parallel = config.model_config, config.parallel_config
    supported = (
        torch.device(config.device_config.device).type == "cuda"
        and bool(getattr(model, "async_chunk", False))
        and parallel.tensor_parallel_size == parallel.pipeline_parallel_size == 1
        and parallel.distributed_executor_backend in (None, "uni")
        and bool(getattr(model.hf_config, "mrv2_gpu_slot_state", False))
        and not bool(getattr(model.hf_config, "mrv2_eager_mtp", False))
        and getattr(config, "speculative_config", None) is None
    )
    # Unsupported deployments retain regular codec delivery by default.
    # An explicit opt-in still reports a misconfigured first-audio path.
    if not supported and "moss_talker_first_audio" in extra:
        raise ValueError(
            "MOSS first audio requires CUDA MRV2 GPU slots, regular MTP, "
            "async chunks, in-process TP/PP=1, no speculation"
        )
    return supported


class MossFirstFrameDecoder(nn.Module):
    def __init__(self, codec_path: str, num_quantizers: int, *, empty_history: bool = False):
        super().__init__()
        self._codec_path = codec_path
        self._num_quantizers = num_quantizers
        self._empty_history = empty_history

    def load(self, config: VllmConfig) -> set[str]:
        # Read the framework's loading/compilation context without changing it.
        # The first decoder owns its graph sizes, not a scheduler or stage config.
        codec_config, self._codec = load_codec(
            self._codec_path,
            device=config.device_config.device,
            load_config=config.load_config,
            num_quantizers=self._num_quantizers,
            attention_backend="sdpa" if self._empty_history else "triton_slot",
        )
        # Reference encoding belongs to the API processor, never this decoder.
        self._codec.encoder = None
        self._sr_tensor = torch.tensor(int(codec_config.sampling_rate), dtype=torch.int32)
        if self._empty_history:
            from .first_frame_special import StatelessFirstGraphs, specialize

            specialize(self._codec)
            self._special_graphs = StatelessFirstGraphs(self._codec, self._num_quantizers, _FIRST_FRAME_BATCH_SIZES)
        else:
            self._session = _MossCodecStreamSession(
                self._codec,
                state_capacity=max(_FIRST_FRAME_BATCH_SIZES),
                n_vq=self._num_quantizers,
                vllm_config=config,
                graph_batch_sizes=list(_FIRST_FRAME_BATCH_SIZES) if not config.model_config.enforce_eager else [],
                graph_frame_sizes=[1],
                gpu_output=True,
                chunk_frames=1,
                private_graph_pool=True,
            )
        return set(dict(self.named_parameters()))

    @property
    def sample_rate(self) -> torch.Tensor:
        return self._sr_tensor

    @torch.inference_mode()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """[B, NQ] codes -> owned float32 [B, channels, samples] PCM."""
        if self._empty_history:
            return self._special_graphs(codes)
        session = self._session
        parts = []
        batch_size = max(_FIRST_FRAME_BATCH_SIZES)
        for start in range(0, codes.shape[0], batch_size):
            chunk = codes[start : start + batch_size]
            slots = [session.acquire() for _ in range(len(chunk))]
            if any(slot is None for slot in slots):
                raise RuntimeError("First-frame decoder exhausted its private state pool")
            try:
                output = session.step(
                    {slot: chunk[row, :, None] for row, slot in enumerate(slots)},
                    terminal_slots=set(slots),
                )
                parts.append(torch.stack([output[slot] for slot in slots]))
            finally:
                for slot in slots:
                    session.release(slot)
        return torch.cat(parts, dim=0)
