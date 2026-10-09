# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from contextlib import nullcontext

import pytest
import torch

from vllm_omni.model_executor.models.moss_tts import audio_tokenizer_v2 as codec_module
from vllm_omni.model_executor.models.moss_tts.audio_tokenizer_v2 import (
    MHAState,
    MossAudioTokenizerModel,
    TransformerState,
)
from vllm_omni.model_executor.models.moss_tts.configuration_moss_audio_tokenizer_v2 import MossAudioTokenizerConfig

pytestmark = [pytest.mark.core_model]


def make_codec(device="cpu", backend="sdpa"):
    block = dict(
        module_type="Transformer",
        d_model=128,
        num_heads=2,
        num_layers=3,
        dim_feedforward=128,
        causal=True,
        norm="layer_norm",
        positional_embedding="rope",
        gating="none",
        context_duration=5.0,
    )
    config = MossAudioTokenizerConfig(
        sampling_rate=8,
        downsample_rate=8,
        number_channels=1,
        encoder_kwargs=[dict(module_type="PatchedPretransform", patch_size=8)],
        decoder_kwargs=[
            dict(block, input_dimension=8, output_dimension=16),
            dict(module_type="PatchedPretransform", patch_size=2),
            dict(block, input_dimension=8, output_dimension=4),
            dict(module_type="PatchedPretransform", patch_size=4),
        ],
        quantizer_kwargs=dict(input_dim=8, rvq_dim=8, output_dim=8, num_quantizers=2, codebook_size=16, codebook_dim=4),
    )
    model = MossAudioTokenizerModel(config).eval().to(device)
    if device != "cpu":
        model.decoder.to(dtype=torch.bfloat16)
    if backend != "sdpa":
        from vllm_omni.model_executor.models.moss_tts.slot_attention import (
            slot_ring_attention,
            slot_ring_attention_rows,
        )

        for module in model.decoder.modules():
            if isinstance(module, codec_module.MossAudioTokenizerMultiheadAttention):
                module._slot_attention = slot_ring_attention
                if backend == "triton_slot_fused":
                    module._slot_attention_rows = slot_ring_attention_rows
    return model


def pair(device="cpu", backend="sdpa", headroom=0):
    torch.manual_seed(17)
    baseline = make_codec(device, backend)
    candidate = make_codec(device, backend)
    candidate.load_state_dict(baseline.state_dict())
    baseline.initialize_decoder_state_pool(4, 4, chunk_frames=headroom)
    baseline.close_decoder_state_pool()
    # Fixed-width streaming retains independent offsets for the reference.
    baseline._start_streaming(8, decoder_only=True)
    baseline._decoder_state_capacity = 8
    candidate.initialize_decoder_state_pool(4, 4, chunk_frames=headroom)
    return baseline, candidate


def check_states(baseline, candidate):
    for left, right in zip(baseline._streaming_modules, candidate._streaming_modules):
        a, b = left._streaming_state, right._streaming_state
        if isinstance(a, MHAState):
            torch.testing.assert_close(a.offset[:4], b.offset[:4], rtol=0, atol=0)
            torch.testing.assert_close(a.kv_cache.end_offset[:4], b.kv_cache.end_offset[:4], rtol=0, atol=0)
            torch.testing.assert_close(a.kv_cache.cache[:, :4], b.kv_cache.cache[:, :4], rtol=0, atol=0)
            assert b.kv_cache.cache[:, 4].count_nonzero() == 0
        elif isinstance(a, TransformerState):
            torch.testing.assert_close(a.offsets[:4], b.offsets[:4], rtol=0, atol=0)
    assert candidate._decoder_slot_offsets[:, 4].count_nonzero() == 0


@pytest.mark.cpu
def test_default_pool_capacity_aliases_and_legacy_lifecycle():
    baseline, candidate = pair()
    assert baseline._decoder_state_capacity == 8
    assert candidate._decoder_state_capacity == 5
    assert baseline._decoder_slot_offsets.shape == (14, 8)
    assert candidate._decoder_slot_offsets.shape == (2, 5)
    for group in (candidate.decoder[0].transformer, candidate.decoder[2].transformer):
        for layer in group.layers:
            state = layer.self_attn._streaming_state
            assert state.offset is group._streaming_state.offsets
            assert state.kv_cache.end_offset is state.offset
    with pytest.raises(RuntimeError, match="already streaming"):
        candidate.initialize_decoder_state_pool(2, 128)
    assert candidate._decoder_null_slot == 4
    candidate.close_decoder_state_pool()
    with candidate.decoder_streaming(1):
        assert candidate._decoder_null_slot is None
        assert not candidate.decoder[0].transformer._shared_kv_metadata
        group = candidate.decoder[0].transformer
        assert group.layers[0].self_attn._streaming_state.offset is not group._streaming_state.offsets
    candidate.initialize_decoder_state_pool(2, 128)
    assert candidate._decoder_state_capacity == 3


@pytest.mark.cpu
@pytest.mark.parametrize("headroom", [0, 7])
def test_cpu_wrap_reorder_reset_padding_and_oversized_chunks(mocker, headroom):
    baseline, candidate = pair(headroom=headroom)
    prepare = mocker.spy(codec_module, "prepare_streaming_attention_metadata")
    with torch.inference_mode():
        for step, frames in enumerate([1, 3, 3, 7, 1, 3, 7, 3]):
            codes = torch.randint(0, 16, (2, 4, frames))
            valid = torch.tensor([True, True, False, False]) if step != 5 else torch.zeros(4, dtype=torch.bool)
            slots = torch.tensor([1, 0, 6, 7]) if step % 2 else torch.tensor([0, 1, 6, 7])
            if step == 5:
                slots = torch.arange(4, 8)
            lengths = torch.full((4,), frames) * valid
            ref = baseline.decode_streaming_tensors(codes, lengths, slots, valid)
            # Invalid IDs must be normalized before any gather, including -1.
            candidate_slots = torch.where(valid, slots, torch.tensor(-1))
            prepare.reset_mock()
            out = candidate.decode_streaming_tensors(codes, lengths, candidate_slots, valid)
            assert prepare.call_count == 2
            torch.testing.assert_close(out[0], ref[0], rtol=0, atol=0)
            torch.testing.assert_close(out[1], ref[1], rtol=0, atol=0)
            check_states(baseline, candidate)
            if step == 3:
                for model in (baseline, candidate):
                    model.reset_decoder_state_slots(torch.tensor([0]))


@pytest.mark.cpu
def test_slot_metadata_does_not_allocate_dense_attention_mask():
    offsets = torch.tensor([0, 401, 0])
    valid = torch.tensor([True, True, False])
    metadata = codec_module.prepare_streaming_attention_metadata(
        offsets, valid, 480, 400, 400, True, slot_attention=True
    )
    assert metadata.scatter_indexes is None and metadata.positions is None and metadata.attn_bias is None
    assert metadata.valid_lengths.tolist() == [480, 480, 0]
    assert metadata.next_offsets.tolist() == [480, 881, 0]


@pytest.mark.cpu
def test_v2_initialization_failure_does_not_load_legacy_codec(mocker):
    from vllm_omni.model_executor.models.moss_tts import modeling_moss_tts_codec as module

    mocker.patch.object(
        module.MossAudioTokenizerV2Config, "get_config_dict", return_value=({"number_channels": 2}, None)
    )
    mocker.patch.object(
        module.MossAudioTokenizerV2Config, "from_pretrained", side_effect=RuntimeError("invalid v2 config")
    )
    legacy = mocker.patch.object(module.MossAudioTokenizerConfig, "from_pretrained")
    with pytest.raises(RuntimeError, match="invalid v2 config"):
        module._build_codec("unused")
    legacy.assert_not_called()


@pytest.mark.cpu
def test_compile_capture_failure_propagates(mocker):
    from vllm_omni.model_executor.models.moss_tts import cuda_graph_streaming_decoder_wrapper as module

    wrapper = module.CUDAGraphStreamingDecoderWrapper.__new__(module.CUDAGraphStreamingDecoderWrapper)
    wrapper.batch_sizes, wrapper.frame_sizes, wrapper.num_quantizers = [1], [1], 2
    wrapper._warmed_up = False
    wrapper._compiled_decode = torch.nn.Identity()
    mocker.patch.object(module.torch.cuda, "device", return_value=nullcontext())
    capture = mocker.patch.object(wrapper, "_capture_with_decode", side_effect=RuntimeError("compile/capture failed"))
    with pytest.raises(RuntimeError, match="compile/capture failed"):
        wrapper.warmup(torch.device("cuda"))
    capture.assert_called_once_with(1, 1, torch.device("cuda"), wrapper._compiled_decode)
    assert not wrapper._warmed_up


@pytest.mark.cuda
@pytest.mark.parametrize("backend", ["sdpa", "triton_slot", "triton_slot_fused"])
@pytest.mark.parametrize("headroom", [0, 7])
def test_cuda_graph_null_slot_and_state_parity(backend, headroom):
    baseline, candidate = pair("cuda", backend, headroom)
    codes = torch.randint(0, 16, (2, 4, 7), device="cuda")
    lengths = torch.full((4,), 7, device="cuda", dtype=torch.long)
    slots = torch.tensor([0, 1, 6, 7], device="cuda")
    valid = torch.tensor([True, True, False, False], device="cuda")
    with torch.inference_mode():
        for _ in range(2):
            candidate.decode_streaming_tensors(codes, lengths, slots, valid)
        candidate.reset_decoder_state_slots(torch.arange(4, device="cuda"))
        for module in candidate._streaming_modules:
            state = module._streaming_state
            if isinstance(state, MHAState):
                state.kv_cache.cache.zero_()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = candidate.decode_streaming_tensors(codes, lengths, slots, valid)
        for step in range(8):
            codes.random_(0, 16)
            slots[:2] = torch.tensor([step % 2, 1 - step % 2], device="cuda")
            if step == 5:
                slots[:2] = torch.tensor([4, 5], device="cuda")
            valid.fill_(step != 5)
            valid[2:] = False
            ref = baseline.decode_streaming_tensors(codes, lengths, slots, valid)
            graph.replay()
            torch.testing.assert_close(out[0], ref[0], rtol=0, atol=0)
            check_states(baseline, candidate)
            if step == 3:
                for model in (baseline, candidate):
                    model.reset_decoder_state_slots(torch.tensor([0], device="cuda"))


@pytest.mark.cuda
@pytest.mark.parametrize("frames", [1, 15, 240, 480])
def test_masked_commit_touched_ring_entries_and_duplicate_null_slots(frames):
    from vllm_omni.model_executor.models.moss_tts.codec_kv_state import commit_cache, commit_offsets

    pool = torch.randn(2, 5, 2, 400, 64, device="cuda", dtype=torch.bfloat16)
    before = pool.clone()
    slots = torch.tensor([2, 0, 4, 4], device="cuda")
    valid = torch.tensor([True, True, False, False], device="cuda")
    offsets = torch.tensor([397, 1234, 0, 0], device="cuda")
    rows = pool.index_select(1, slots)
    length = min(frames, 400)
    time = torch.arange(frames - length, frames, device="cuda")
    indexes = ((offsets[:, None] + time) % 400)[:, None, :, None]
    for kv in (0, 1):
        values = torch.randn(4, 2, length, 64, device="cuda", dtype=torch.bfloat16)
        rows[kv].scatter_(2, indexes.expand_as(values), values)
    expected = before.clone()
    expected.index_copy_(1, slots[:2], rows[:, :2])
    commit_cache(rows, pool, slots, valid, offsets, frames)
    torch.testing.assert_close(pool, expected, rtol=0, atol=0)
    state = torch.zeros(5, device="cuda", dtype=torch.long)
    commit_offsets(offsets + frames, state, slots, valid)
    assert state.tolist() == [1234 + frames, 0, 397 + frames, 0, 0]


@pytest.mark.cuda
@pytest.mark.parametrize("backend", ["sdpa", "triton_slot", "triton_slot_fused"])
def test_strict_dynamic_compilation_and_wrapper_padding(backend):
    from vllm_omni.model_executor.models.moss_tts.cuda_graph_streaming_decoder_wrapper import (
        CUDAGraphStreamingDecoderWrapper,
    )

    # Each backend is an independent compiler regression. Avoid combining its
    # shape specializations with the previous parameter's Dynamo guard cache.
    torch._dynamo.reset()
    baseline, candidate = pair("cuda", backend)
    compiled = torch.compile(candidate.decode_streaming_tensors, fullgraph=True, dynamic=True)
    with torch.inference_mode():
        for batch, frames in [(4, 3), (2, 1), (3, 4)]:
            codes = torch.randint(0, 16, (2, batch, frames), device="cuda")
            lengths = torch.full((batch,), frames, device="cuda", dtype=torch.long)
            slots = torch.arange(batch, device="cuda")
            valid = torch.ones(batch, device="cuda", dtype=torch.bool)
            for tensor, dims in [(codes, [1, 2]), (lengths, [0]), (slots, [0]), (valid, [0])]:
                for dim in dims:
                    torch._dynamo.mark_dynamic(tensor, dim)
            ref = baseline.decode_streaming_tensors(codes, lengths, slots, valid)
            out = compiled(codes, lengths, slots, valid)
            torch.testing.assert_close(out[0], ref[0], rtol=0.02, atol=0.002)
            torch.testing.assert_close(
                candidate._decoder_slot_offsets[0, :4], baseline.decoder[0].transformer._streaming_state.offsets[:4]
            )
        wrapper = CUDAGraphStreamingDecoderWrapper.__new__(CUDAGraphStreamingDecoderWrapper)
        wrapper.codec, wrapper.state_capacity = candidate, 4
        wrapper.batch_sizes, wrapper.frame_sizes, wrapper.num_quantizers = [4], [3], 2
        wrapper._shared_kv, wrapper._pool, wrapper.graphs = True, None, {}
        wrapper._capture_with_decode(4, 3, torch.device("cuda"), compiled)
        assert wrapper.scratch_capacity == 1
        for _ in range(3):
            codes = torch.randint(0, 16, (2, 2, 3), device="cuda")
            result = wrapper.decode(codes, torch.tensor([0, 1], device="cuda"))
            assert result is not None and result[2] == 2
            assert wrapper.graphs[(4, 3)].static_state_slot_ids.tolist() == [0, 1, 4, 4]
            assert candidate._decoder_slot_offsets[:, 4].count_nonzero() == 0
            for module in candidate._streaming_modules:
                if isinstance(module._streaming_state, MHAState):
                    assert module._streaming_state.kv_cache.cache[:, 4].count_nonzero() == 0
