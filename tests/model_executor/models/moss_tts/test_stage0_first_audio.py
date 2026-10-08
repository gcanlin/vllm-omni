# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.model_executor.models.moss_tts.test_local_model_state import _batch, _state
from vllm_omni.model_executor.models.moss_tts.first_audio_state import MossEarlyFirstAudioState
from vllm_omni.model_executor.models.moss_tts.first_frame_decoder import first_audio_enabled
from vllm_omni.model_executor.models.moss_tts.local_model_state import MossLocalModelState, _CodeRowsSnapshot

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("enabled", [False, True])
def test_system_profile_first_audio_keeps_v1_fallback(enabled, mocker):
    config = mocker.Mock()
    config.model_config.stage_connector_config = {"extra": {"moss_talker_first_audio": enabled}}
    config.model_config.use_v2_model_runner = False
    assert not first_audio_enabled(config)


def early_state():
    owner = SimpleNamespace(model=SimpleNamespace(audio_pad_token_id=16, audio_assistant_slot_token_id=7))
    state = MossEarlyFirstAudioState(owner, None)
    owner._first_audio_sender = object()
    return state


@pytest.mark.parametrize("batch_prefill", [False, True])
@pytest.mark.parametrize("computed,count,eligible", [(0, 2, False), (2, 2, True), (3, 1, True)])
def test_only_completed_prefill_arms_first_audio_including_prefix_hits(batch_prefill, computed, count, eligible):
    state = _state(MossLocalModelState, torch.device("cpu"))
    state._batch_prefill = batch_prefill
    state._early_first_audio = MossEarlyFirstAudioState(state, None)
    state._first_audio_sender = object()
    state.intermediate_buffer.buffers[0] = {
        "req_id": "a",
        "codes": {"ref": torch.tensor([[1, 2], [3, 4], [5, 6], [2, 3]])},
        "sampling_params": SimpleNamespace(max_tokens=10),
    }
    batch = _batch(torch.device("cpu"), [0], [count])
    req = SimpleNamespace(prompt_len=np.full(5, 4), num_computed_tokens=np.full(5, computed))
    with torch.inference_mode():
        state.run_preprocess(batch, {"input_ids": batch.input_ids}, req)
    assert ("a" in state._early_first_audio.waiting) == eligible


def test_owned_first_codes_batch_reorder_and_one_time_promise(mocker):
    state = early_state()
    state.record_prefill("b", SimpleNamespace(max_tokens=10))
    publish = mocker.patch.object(state, "_publish", return_value=["b"])
    codes = torch.tensor([[1, 2], [3, 4], [5, 6]])
    state.after_mtp(["a", "b", "c"], codes, torch.tensor([7, 7, 7]))
    codes.fill_(99)
    assert publish.call_args.args[1].tolist() == [[3, 4]]
    flags = state.take_flags(["c", "b", "a"], codes.device)
    assert flags.tolist() == [False, True, False]
    snapshot = _CodeRowsSnapshot(codes.clone(), [0, 1, 2], 3, flags)
    copies = []

    def copy(tensor):
        copies.append(tensor)
        return tensor.clone()

    host = snapshot.copy_to_cpu(copy)
    flags.zero_()
    assert len(copies) == 2
    assert [bool(x) for x in host["meta"]["first_audio"]] == [False, True, False]
    assert state.take_flags(["b"], codes.device) is None
    state.after_mtp(["a", "b", "c"], codes, torch.tensor([7, 7, 7]))
    assert publish.call_count == 1
    state.remove("b")
    assert not state.seen and not state.waiting and not state.delivered and not state.updates


@pytest.mark.parametrize("codes,token,valid", [([16, 16], 7, False), ([1, 2], 9, False), ([1, 2], 7, True)])
def test_stop_and_non_audio_tokens_do_not_promise_pcm(codes, token, valid, mocker):
    state = early_state()
    state.record_prefill("a", SimpleNamespace(max_tokens=10))
    publish = mocker.patch.object(state, "_publish", return_value=["a"])
    state.after_mtp(["a"], torch.tensor([codes]), torch.tensor([token]))
    assert publish.call_args.args[2].tolist() == [valid]
    assert state.take_flags(["a"], torch.device("cpu")).tolist() == [valid]


def test_rejected_route_and_cancel_keep_regular_path(mocker):
    state = early_state()
    state.record_prefill("a", SimpleNamespace(max_tokens=10))
    publish = mocker.patch.object(state, "_publish", return_value=[])
    state.after_mtp(["a"], torch.tensor([[1, 2]]), torch.tensor([7]))
    assert state.take_flags(["a"], torch.device("cpu")) is None
    state.record_prefill("b", SimpleNamespace(max_tokens=10))
    state.remove("b")
    state.after_mtp(["b"], torch.tensor([[1, 2]]), torch.tensor([7]))
    assert publish.call_count == 1


def test_cap_one_and_unbound_route_do_not_arm_first_audio():
    state = early_state()
    state.record_prefill("a", SimpleNamespace(max_tokens=1))
    state.owner._first_audio_sender = None
    state.record_prefill("b", SimpleNamespace(max_tokens=10))
    assert not state.waiting
