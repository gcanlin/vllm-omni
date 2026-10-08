# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared checkpoint validation and first-frame loading without stage config edits."""

from dataclasses import dataclass, field

import pytest
import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.config import LoadConfig

from vllm_omni.model_executor.models.moss_tts import codec_loader
from vllm_omni.model_executor.models.moss_tts.first_frame_decoder import MossFirstFrameDecoder

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TinyCodec(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(2, 2, bias=False)
        self.decoder = nn.Module()
        self.decoder.self_attn = nn.Module()
        self.decoder.self_attn.in_projs = nn.ModuleList([nn.Linear(2, 6, bias=False)])
        self.quantizer = nn.Module()


@pytest.fixture
def checkpoint(mocker):
    codec = TinyCodec()
    config = PretrainedConfig(sampling_rate=48000, num_quantizers=12, number_channels=2)
    mocker.patch.object(codec_loader, "_build_codec", return_value=(config, codec))
    weights = [("encoder.weight", torch.full((2, 2), 3.0)), ("decoder.self_attn.in_proj.weight", torch.ones(6, 2))]
    iterator = mocker.patch.object(codec_loader.DefaultModelLoader, "_get_weights_iterator", return_value=iter(weights))
    return codec, config, weights, iterator


def test_shared_loader_remaps_and_loads_every_parameter(checkpoint):
    codec, config, _, _ = checkpoint
    loaded_config, loaded = codec_loader.load_codec(
        "local-checkpoint", device=torch.device("cpu"), load_config=LoadConfig(), num_quantizers=12
    )
    assert loaded is codec and loaded_config is config
    assert not loaded.training
    torch.testing.assert_close(loaded.encoder.weight, torch.full((2, 2), 3.0))
    torch.testing.assert_close(loaded.decoder.self_attn.in_projs[0].weight, torch.ones(6, 2))


@pytest.mark.parametrize("problem", ["missing", "unexpected", "shape"])
def test_shared_loader_rejects_incomplete_checkpoint(checkpoint, problem):
    _, _, weights, iterator = checkpoint
    if problem == "missing":
        weights.pop()
    elif problem == "unexpected":
        weights.append(("unknown.weight", torch.ones(1)))
    else:
        weights[-1] = (weights[-1][0], torch.ones(1))
    iterator.return_value = iter(weights)
    with pytest.raises(RuntimeError, match="weights were not fully loaded"):
        codec_loader.load_codec(
            "local-checkpoint", device=torch.device("cpu"), load_config=LoadConfig(), num_quantizers=12
        )


@dataclass(frozen=True)
class DeviceContext:
    device: torch.device = torch.device("cpu")


@dataclass(frozen=True)
class ModelContext:
    enforce_eager: bool = False


@dataclass(frozen=True)
class LoadingContext:
    device_config: DeviceContext = field(default_factory=DeviceContext)
    load_config: LoadConfig = field(default_factory=LoadConfig)
    model_config: ModelContext = field(default_factory=ModelContext)

    @property
    def scheduler_config(self):
        pytest.fail("First-frame loading must not access the stage scheduler")

    @property
    def compilation_config(self):
        pytest.fail("First-frame loading must not rewrite the stage compilation config")


@pytest.mark.parametrize("empty_history", [False, True])
def test_first_frame_loads_without_constructing_stage1_or_mutating_config(checkpoint, mocker, empty_history):
    from vllm_omni.model_executor.models.moss_tts import first_frame_special, modeling_moss_tts_codec

    codec, _, _, _ = checkpoint
    config = LoadingContext()
    stage = mocker.patch.object(modeling_moss_tts_codec.MossTTSCodecDecoder, "__init__", side_effect=AssertionError)
    loader = mocker.patch(
        "vllm_omni.model_executor.models.moss_tts.first_frame_decoder.load_codec", return_value=(checkpoint[1], codec)
    )
    specialize = mocker.patch.object(first_frame_special, "specialize")
    graphs = mocker.patch.object(first_frame_special, "StatelessFirstGraphs")
    session = mocker.patch.object(modeling_moss_tts_codec, "_MossCodecStreamSession")
    decoder = MossFirstFrameDecoder("local-checkpoint", 12, empty_history=empty_history)
    loaded = decoder.load(config)
    assert codec.encoder is None
    assert decoder.sample_rate.item() == 48000
    assert loaded == {"_codec.decoder.self_attn.in_projs.0.weight"}
    assert loader.call_args.kwargs["load_config"] is config.load_config
    stage.assert_not_called()
    if empty_history:
        specialize.assert_called_once_with(codec)
        graphs.assert_called_once_with(codec, 12, (1, 2, 4, 8))
        session.assert_not_called()
    else:
        specialize.assert_not_called()
        graphs.assert_not_called()
        assert session.call_args.kwargs["vllm_config"] is config
        assert session.call_args.kwargs["state_capacity"] == 8
        assert session.call_args.kwargs["graph_batch_sizes"] == [1, 2, 4, 8]
