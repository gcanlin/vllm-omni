# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from io import BytesIO

import numpy as np
import pytest
import soundfile

from vllm_omni.entrypoints.openai.audio_utils_mixin import AudioMixin, _float32_to_pcm16_bytes
from vllm_omni.entrypoints.openai.protocol.audio import CreateAudio


def reference(audio):
    with BytesIO() as buffer:
        soundfile.write(buffer, audio, 48000, format="RAW", subtype="PCM_16")
        return buffer.getvalue()


def test_pcm_matches_libsndfile_at_every_quantization_boundary_and_special_values():
    centers = np.arange(-32768, 32768, dtype=np.float32) / 32768
    audio = np.concatenate(
        [
            centers,
            np.nextafter(centers, -np.inf),
            np.nextafter(centers, np.inf),
            np.array([np.nan, np.inf, -np.inf, -2, 2, -0.0, 0.0], dtype=np.float32),
            np.random.default_rng(42).uniform(-1.1, 1.1, 1_000_000).astype(np.float32),
        ]
    )
    original = audio.copy()
    assert _float32_to_pcm16_bytes(audio) == reference(audio)
    np.testing.assert_array_equal(audio, original)


@pytest.mark.parametrize("samples", [0, 1, 3840, 57600])
@pytest.mark.parametrize("stereo", [False, True])
def test_streaming_pcm_bytes_interleaving_and_metadata(samples, stereo):
    shape = (2, samples) if stereo else (samples,)
    audio = np.random.default_rng(42).uniform(-1, 1, shape).astype(np.float32)
    response = AudioMixin().create_audio(
        CreateAudio(
            audio_tensor=audio,
            sample_rate=48000,
            response_format="pcm",
            base64_encode=False,
        )
    )
    expected = audio.T if stereo else audio
    assert response.audio_data == reference(expected)
    assert response.audio_metadata.frame_count == samples
    assert response.audio_metadata.channels == (2 if stereo else 1)


pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
