# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 text-encoder stage input and output adapters."""

from __future__ import annotations

import copy
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from vllm_omni.model_executor.models.minimax_h3.preprocessing import (
    MINIMAX_H3_OUTPUT_SHORT_EDGE,
    load_minimax_h3_images,
    resolve_minimax_h3_aspect_ratio,
    resolve_minimax_h3_output_canvas,
    resolve_minimax_h3_reference_image_shape,
)
from vllm_omni.model_executor.models.minimax_h3.reference_video import (
    MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS,
    prepare_reference_videos,
    sample_reference_video_frames,
)
from vllm_omni.errors import OmniClientError
from vllm_omni.model_executor.models.minimax_h3.conditioning import (
    MINIMAX_H3_CONDITION_LABELS_KEY,
    MINIMAX_H3_PRESENTATION_TASK_KEY,
)

MINIMAX_H3_DIT_STAGE_ID = 1


def _audio_items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)) and len(value) == 2 and isinstance(value[1], (int, np.integer)):
        return [value]
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _resolve_task(
    extra_args: Mapping[str, Any],
    multi_modal_data: Mapping[str, Any],
) -> str:
    requested = extra_args.get("task")
    if requested is not None:
        return str(requested).lower()
    if multi_modal_data.get("video") is not None or multi_modal_data.get("audio") is not None:
        return "ref2va"
    if multi_modal_data.get("image") is not None:
        return "fl2va"
    return "t2va"


def _prepare_qwen_images(
    task: str,
    values: Any,
    sampling: Any,
) -> list[Any]:
    if values is None:
        return []
    images = load_minimax_h3_images(values)
    if task == "ref2va":
        return [
            image.resize(
                resolve_minimax_h3_reference_image_shape(image),
                Image.Resampling.LANCZOS,
            )
            for image in images
        ]
    if task != "fl2va":
        return images

    extra_args = sampling.extra_args or {}
    target = extra_args.get("target")
    if target is not None and not isinstance(target, Mapping):
        raise OmniClientError("MiniMax H3 extra_args['target'] must be an object")
    target = target if isinstance(target, Mapping) else {}
    aspect_ratio = resolve_minimax_h3_aspect_ratio(
        task,
        target.get("aspect_ratio", extra_args.get("aspect_ratio")),
        images[0],
    )
    if not 0.25 <= aspect_ratio <= 4.0:
        raise OmniClientError(f"MiniMax H3 canvas aspect ratio must be in [1:4, 4:1], got {aspect_ratio}")
    height = sampling.height
    width = sampling.width
    if height is None or width is None:
        short_edge = target.get(
            "short_edge",
            extra_args.get("short_edge", MINIMAX_H3_OUTPUT_SHORT_EDGE),
        )
        if isinstance(short_edge, bool) or not isinstance(short_edge, (int, np.integer)):
            raise OmniClientError(
                f"MiniMax H3 target.short_edge must be {MINIMAX_H3_OUTPUT_SHORT_EDGE}, got {short_edge!r}"
            )
        height, width = resolve_minimax_h3_output_canvas(aspect_ratio, int(short_edge))
    height = int(height) // 32 * 32
    width = int(width) // 32 * 32
    if min(height, width) <= 0:
        raise OmniClientError(f"invalid MiniMax H3 canvas {width}x{height}")
    if width > 4 * height or height > 4 * width:
        raise OmniClientError("MiniMax H3 canvas aspect ratio must be in [1:4, 4:1]")
    return [image.resize((width, height), Image.Resampling.LANCZOS) for image in images]


def prepare_text_encoder_prompt(
    prompt: Any,
    sampling_params_list: Sequence[Any],
) -> Any:
    """Build H3's labeled Qwen3-VL presentation for Stage 0.

    The upstream Qwen3-VL multimodal processor expands each image/video
    placeholder into the exact number of vision tokens and adds timestamped
    video blocks.  Audio is represented only by its H3 text label and is not
    sent to Qwen3-VL.
    """
    if isinstance(prompt, str):
        return prompt
    if not isinstance(prompt, dict):
        raise TypeError(f"MiniMax H3 expects a string or dict prompt, got {type(prompt)!r}")

    text = str(prompt.get("prompt") or "")
    if not text:
        raise OmniClientError("MiniMax H3 requires a non-empty prompt")
    multi_modal_data = prompt.get("multi_modal_data") or {}
    if not isinstance(multi_modal_data, Mapping):
        raise TypeError("multi_modal_data must be a mapping")

    videos = multi_modal_data.get("video")
    audios = _audio_items(multi_modal_data.get("audio"))
    diffusion_sampling = sampling_params_list[MINIMAX_H3_DIT_STAGE_ID]
    extra_args = diffusion_sampling.extra_args or {}
    task = _resolve_task(extra_args, multi_modal_data)
    images = _prepare_qwen_images(task, multi_modal_data.get("image"), diffusion_sampling)
    qwen_video_inputs: list[tuple[np.ndarray, dict[str, Any]]] = []
    condition_labels: list[tuple[str, int]] = []

    if task == "t2va":
        if images or videos is not None or audios:
            raise OmniClientError("t2va does not accept image, video, or audio conditions")
    elif task == "fl2va":
        if not images or videos is not None or audios:
            raise OmniClientError("fl2va requires image conditions only")
        condition_labels.extend(("image", index) for index in range(1, len(images) + 1))
    elif task == "ref2va":
        if not images and videos is None:
            raise OmniClientError("ref2va requires an image or video condition")
        condition_labels.extend(("image", index) for index in range(1, len(images) + 1))
        prepared_videos: list[dict[str, Any]] = []
        if videos is not None:
            with tempfile.TemporaryDirectory(prefix="minimax_h3_text_encoder_") as workdir:
                prepared_videos = prepare_reference_videos(
                    videos,
                    target_frame_count=0,
                    workdir=workdir,
                    start_time_seconds=extra_args.get("start_time_seconds"),
                )
                for index, item in enumerate(prepared_videos):
                    sampled = sample_reference_video_frames(
                        item["prepared_path"],
                        workdir=str(Path(workdir) / f"qwen_frames_{index}"),
                    )
                    frames = np.stack(sampled["frames"])
                    frame_count = int(frames.shape[0])
                    qwen_video_inputs.append(
                        (
                            frames,
                            {
                                "total_num_frames": frame_count,
                                "fps": MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS,
                                "duration": frame_count / MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS,
                                "video_backend": "minimax_h3",
                                "frames_indices": list(range(frame_count)),
                                "do_sample_frames": False,
                            },
                        )
                    )
        audio_index = 0
        for video_index, item in enumerate(prepared_videos, start=1):
            if item["input_has_audio"]:
                audio_index += 1
                condition_labels.append(("audio", audio_index))
            condition_labels.append(("video", video_index))
        for _ in audios:
            audio_index += 1
            condition_labels.append(("audio", audio_index))
    else:
        raise OmniClientError(f"unsupported MiniMax H3 task {task!r}")

    transformed = copy.copy(prompt)
    if isinstance(prompt.get("additional_information"), Mapping):
        transformed["additional_information"] = dict(prompt["additional_information"])
    transformed["prompt"] = text
    qwen_mm_data = dict(multi_modal_data)
    qwen_mm_data.pop("audio", None)
    if images:
        qwen_mm_data["image"] = images
    if qwen_video_inputs:
        qwen_mm_data["video"] = qwen_video_inputs
    transformed["multi_modal_data"] = qwen_mm_data or None

    mm_processor_kwargs = dict(prompt.get("mm_processor_kwargs") or {})
    mm_processor_kwargs[MINIMAX_H3_PRESENTATION_TASK_KEY] = task
    mm_processor_kwargs[MINIMAX_H3_CONDITION_LABELS_KEY] = condition_labels
    transformed["mm_processor_kwargs"] = mm_processor_kwargs
    return transformed


def _original_prompt(prompt: Any) -> dict[str, Any]:
    if isinstance(prompt, list):
        prompt = prompt[0] if prompt else {}
    if isinstance(prompt, dict):
        return copy.copy(prompt)
    if isinstance(prompt, str):
        return {"prompt": prompt}
    raise TypeError(f"invalid MiniMax H3 prompt type {type(prompt)!r}")


def text_encoder2diffusion(
    source_outputs: list[Any],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
) -> dict[str, Any] | None:
    """Attach Stage 0 conditioning to the original request."""
    del requires_multimodal_data, streaming_context
    if not source_outputs:
        return None

    payload = source_outputs[0].outputs[0].multimodal_output
    conditioning = {
        "hidden_states": payload["hidden"],
        "token_tags": payload["meta"]["token_role_ids"].squeeze(-1),
    }

    diffusion_prompt = _original_prompt(prompt)
    additional_information = dict(diffusion_prompt.get("additional_information") or {})
    additional_information["text_encoder_output"] = conditioning
    diffusion_prompt["additional_information"] = additional_information
    return diffusion_prompt
