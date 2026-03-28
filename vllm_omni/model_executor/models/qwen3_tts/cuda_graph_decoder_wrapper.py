# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility wrapper for decoder static-graph execution.

Historically this file implemented a CUDA-only graph wrapper. It now delegates
to the reusable static-graph runtime so model-local code no longer owns raw
graph API calls directly. The public class name is kept for compatibility with
existing call sites and tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from vllm.logger import init_logger

from vllm_omni.compilation import BucketedStaticGraphRunner, GraphWorkloadAdapter, ListBucketPolicy
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)


@dataclass
class _DecoderStaticGraphState:
    static_input: torch.Tensor


class _DecoderStaticGraphWorkload(GraphWorkloadAdapter[_DecoderStaticGraphState, torch.Tensor]):
    def __init__(self, decoder: torch.nn.Module, num_quantizers: int):
        self.decoder = decoder
        self.num_quantizers = num_quantizers

    def can_run_graph(self, codes: torch.Tensor) -> bool:
        return codes.shape[0] == 1

    def bucket_value(self, codes: torch.Tensor) -> int:
        return int(codes.shape[-1])

    def build_static_state(self, bucket_size: int, device: torch.device, dtype: torch.dtype) -> _DecoderStaticGraphState:
        return _DecoderStaticGraphState(
            static_input=torch.zeros(1, self.num_quantizers, bucket_size, dtype=dtype, device=device)
        )

    def warmup(self, state: _DecoderStaticGraphState) -> None:
        _ = self.decoder(state.static_input)

    def run(self, state: _DecoderStaticGraphState) -> torch.Tensor:
        return self.decoder(state.static_input)

    def eager(self, codes: torch.Tensor) -> torch.Tensor:
        return self.decoder(codes)

    def copy_inputs(self, state: _DecoderStaticGraphState, codes: torch.Tensor) -> None:
        actual_size = int(codes.shape[-1])
        state.static_input.zero_()
        state.static_input[:, :, :actual_size] = codes

    def process_output(self, output: torch.Tensor, state: _DecoderStaticGraphState, codes: torch.Tensor) -> torch.Tensor:
        actual_out_len = int(codes.shape[-1]) * int(self.decoder.total_upsample)
        return output[..., :actual_out_len].clone()


class CUDAGraphDecoderWrapper:
    """Backwards-compatible wrapper around the reusable static-graph runner."""

    def __init__(
        self,
        decoder: torch.nn.Module,
        capture_sizes: list[int] | None = None,
        num_quantizers: int = 8,
        enabled: bool = True,
    ):
        self.decoder = decoder
        self._explicit_sizes = capture_sizes is not None
        self.capture_sizes = sorted(capture_sizes) if capture_sizes else []
        self.num_quantizers = num_quantizers
        self.enabled = enabled

        self._warmed_up = False
        self._device: torch.device | None = None
        self._runner: BucketedStaticGraphRunner[_DecoderStaticGraphState, torch.Tensor] | None = None

    @staticmethod
    def compute_capture_sizes(
        codec_chunk_frames: int = 0,
        codec_left_context_frames: int = 0,
        decode_chunk_size: int = 300,
        decode_left_context: int = 25,
    ) -> list[int]:
        """Compute capture sizes from chunking config for high graph hit rate."""
        sizes: set[int] = set()

        if codec_chunk_frames > 0:
            sizes.add(codec_chunk_frames)
            if codec_left_context_frames > 0:
                sizes.add(codec_chunk_frames + codec_left_context_frames)

        non_stream_max = decode_chunk_size + decode_left_context
        sizes.add(non_stream_max)

        for p2 in [2, 4, 8, 16, 32, 64, 128, 256]:
            if p2 <= non_stream_max:
                sizes.add(p2)

        return sorted(sizes)

    def _get_padded_size(self, actual_size: int) -> int | None:
        for size in self.capture_sizes:
            if actual_size <= size:
                return size
        return None

    def warmup(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.long,
        codec_chunk_frames: int = 0,
        codec_left_context_frames: int = 0,
    ) -> None:
        if not self.enabled or self._warmed_up:
            return

        self._device = device
        self.decoder.eval()

        if not self._explicit_sizes:
            self.capture_sizes = self.compute_capture_sizes(
                codec_chunk_frames=codec_chunk_frames,
                codec_left_context_frames=codec_left_context_frames,
            )

        backend = current_omni_platform.create_static_graph_backend()
        if backend is None or not backend.is_available(device):
            logger.info("Static graph backend not available for decoder on device=%s", device)
            return

        logger.info("Starting decoder static-graph warmup for %d sizes: %s", len(self.capture_sizes), self.capture_sizes)
        self._runner = BucketedStaticGraphRunner(
            backend=backend,
            bucket_policy=ListBucketPolicy(self.capture_sizes),
            workload=_DecoderStaticGraphWorkload(self.decoder, self.num_quantizers),
            enabled=self.enabled,
            use_global_graph_pool=False,
        )
        self._runner.warmup(device=device, dtype=dtype, continue_on_capture_failure=True)
        if not self._runner.captures:
            logger.warning(
                "Decoder static-graph warmup captured 0/%d buckets on device=%s; falling back to eager decode",
                len(self.capture_sizes),
                device,
            )
            self._runner = None
            self._warmed_up = False
            return

        self._warmed_up = True
        logger.info(
            "Decoder static-graph warmup complete: %d/%d captured",
            len(self.graphs),
            len(self.capture_sizes),
        )

    @property
    def is_enabled(self) -> bool:
        return self._runner is not None and bool(self._runner.captures)

    @property
    def graphs(self) -> dict[int, object]:
        if self._runner is None:
            return {}
        return {bucket: capture.graph for bucket, capture in self._runner.captures.items()}

    @property
    def static_inputs(self) -> dict[int, torch.Tensor]:
        if self._runner is None:
            return {}
        return {bucket: capture.state.static_input for bucket, capture in self._runner.captures.items()}

    @property
    def static_outputs(self) -> dict[int, torch.Tensor]:
        if self._runner is None:
            return {}
        return {bucket: capture.output for bucket, capture in self._runner.captures.items()}

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        if self._runner is None:
            return self.decoder(codes)
        self._runner.enabled = self.enabled
        return self._runner.run(codes)

    def chunked_decode_with_static_graph(
        self,
        codes: torch.Tensor,
        chunk_size: int = 300,
        left_context_size: int = 25,
    ) -> torch.Tensor:
        wavs = []
        start_index = 0
        total_len = codes.shape[-1]
        total_upsample = self.decoder.total_upsample

        while start_index < total_len:
            end_index = min(start_index + chunk_size, total_len)
            context_size = left_context_size if start_index - left_context_size > 0 else start_index

            codes_chunk = codes[..., start_index - context_size : end_index]
            wav_chunk = self.decode(codes_chunk)

            wavs.append(wav_chunk[..., context_size * total_upsample :])
            start_index = end_index

        return torch.cat(wavs, dim=-1)

    def chunked_decode_with_cudagraph(
        self,
        codes: torch.Tensor,
        chunk_size: int = 300,
        left_context_size: int = 25,
    ) -> torch.Tensor:
        return self.chunked_decode_with_static_graph(
            codes,
            chunk_size=chunk_size,
            left_context_size=left_context_size,
        )
