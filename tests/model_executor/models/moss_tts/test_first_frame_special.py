# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest
import torch

from vllm_omni.model_executor.models.moss_tts.audio_tokenizer_v2 import (
    MossAudioTokenizerMultiheadAttention,
    MossAudioTokenizerRotaryEmbedding,
)
from vllm_omni.model_executor.models.moss_tts.first_frame_special import EmptyHistoryAttention

pytestmark = [pytest.mark.core_model, pytest.mark.cuda]


@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("length", [1, 2, 8, 32])
def test_empty_history_attention_matches_original_causal_attention(batch, length):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(8133)
    original = MossAudioTokenizerMultiheadAttention(
        embed_dim=384,
        num_heads=6,
        causal=True,
        context=64,
        rope=MossAudioTokenizerRotaryEmbedding(),
        device="cuda",
        dtype=torch.bfloat16,
    ).eval()
    special = EmptyHistoryAttention(original)
    query = torch.randn(batch, length, 384, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        expected = original(query, query, query)
        actual = special(query, query, query)
    # Different BF16 GEMM/attention accumulation orders need not be bitwise equal.
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.005)
