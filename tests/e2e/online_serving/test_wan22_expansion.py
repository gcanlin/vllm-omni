"""
Comprehensive tests of diffusion features that are available in online serving mode
and are supported by the following models:
- Wan-AI/Wan2.2-I2V-A14B-Diffusers (NPU only, HSDP always-on)

Coverage (HSDP is mandatory baseline, ring-attn excluded):
- Cache-DiT + HSDP
- HSDP only
- CFG-Parallel + HSDP
- Ulysses-SP + HSDP
- Tensor-Parallel + VAE-Patch-Parallel + HSDP

assert_diffusion_response validates successful generation
"""

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.media import generate_synthetic_image
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler

pytestmark = [pytest.mark.diffusion, pytest.mark.full_model]

PROMPT = "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
NEGATIVE_PROMPT = "low quality, blurry, distorted face, extra limbs, bad anatomy, watermark, logo, text, ugly, deformed, mutated, jpeg artifacts"
TWO_CARD_MARKS = hardware_marks(res={"npu": "A2"}, num_cards=2)
FOUR_CARD_MARKS = hardware_marks(res={"npu": "A2"}, num_cards=4)

WAN22_MODELS = [
    ("Wan-AI/Wan2.2-I2V-A14B-Diffusers", "i2v"),
]

HSDP_ARGS = ["--use-hsdp", "--hsdp-shard-size", "2"]

# Each entry: (feat_id, extra_args, marks). HSDP is appended automatically.
PARALLEL_CONFIGS = [
    ("hsdp", [], TWO_CARD_MARKS),
    ("cfg_parallel", ["--cfg-parallel-size", "2"], FOUR_CARD_MARKS),
    ("ulysses_sp", ["--usp", "2"], FOUR_CARD_MARKS),
    ("tp_vae_patch", ["--tensor-parallel-size", "2", "--vae-patch-parallel-size", "2"], FOUR_CARD_MARKS),
]


def _get_wan22_feature_cases():
    """
    Generate parameterized test cases for I2V-A14B with HSDP always enabled.
    Ring-attention is intentionally excluded.
    """
    cases = []

    # Cache-DiT + HSDP (2 cards because HSDP shard_size=2)
    for model_path, model_key in WAN22_MODELS:
        cases.append(
            pytest.param(
                OmniServerParams(
                    model=model_path,
                    server_args=["--cache-backend", "cache_dit", "--enable-layerwise-offload", *HSDP_ARGS],
                ),
                id=f"{model_key}_cache_dit_hsdp",
                marks=TWO_CARD_MARKS,
            )
        )

    # Other parallelism features stacked with HSDP
    for model_path, model_key in WAN22_MODELS:
        for feat_id, extra_args, marks in PARALLEL_CONFIGS:
            cases.append(
                pytest.param(
                    OmniServerParams(model=model_path, server_args=[*extra_args, *HSDP_ARGS]),
                    id=f"{model_key}_{feat_id}",
                    marks=marks,
                )
            )

    return cases


@pytest.mark.parametrize(
    "omni_server",
    _get_wan22_feature_cases(),
    indirect=True,
)
def test_wan22_diffusion_features(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
):
    model_path = omni_server.model
    is_i2v_or_ti2v = any(kw in model_path for kw in ["I2V", "TI2V"])
    is_moe_model = "I2V-A14B" in model_path  # Only I2V-A14B uses MoE per spec

    form_data = {
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "height": 512,
        "width": 512,
        "num_frames": 8,
        "fps": 8,
        "num_inference_steps": 2,
        "guidance_scale": 4.0,
        "seed": 42,
        # flow_shift omitted: Service uses resolution-based defaults (12.0 for 512px)
        # vae_use_slicing/tiling omitted: Service-side optimization, not request param
    }

    if is_moe_model:
        form_data.update(
            {
                "guidance_scale_2": 1.0,
                "boundary_ratio": 0.5,
            }
        )

    request_config = {
        "model": model_path,
        "form_data": form_data,
    }

    if is_i2v_or_ti2v:
        request_config["image_reference"] = f"data:image/jpeg;base64,{generate_synthetic_image(512, 512)['base64']}"

    openai_client.send_video_diffusion_request(request_config)
