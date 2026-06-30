# SPDX-License-Identifier: Apache-2.0
"""MOSS-TTS serving adapters (Nano + full family).

Both variants share the same build/validate flow (``_build_moss_tts_params``
handles each); they are registered under distinct model-type names.
"""

from typing import TYPE_CHECKING

from vllm.inputs import tokens_input
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest, conditioning_cache_salt

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest


class _MossTTSAdapterBase(ARTTSAdapter):
    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        err = self.ctx.server._apply_uploaded_speaker(request)
        if err:
            return err
        return self.ctx.server._validate_moss_tts_request(request)

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        server = self.ctx.server
        logger.info(
            "[MossTTSDebug][adapter-build-start] model_type=%s variant=%s stream=%s "
            "ref_audio=%s voice=%s",
            getattr(server, "_tts_model_type", None),
            getattr(server, "_moss_variant", None),
            bool(getattr(request, "stream", False)),
            bool(getattr(request, "ref_audio", None)),
            getattr(request, "voice", None),
        )
        tts_params = await server._build_moss_tts_params(request)
        logger.info(
            "[MossTTSDebug][adapter-build-after-params] keys=%s has_prompt_ids=%s codes_keys=%s",
            sorted(tts_params.keys()),
            isinstance(tts_params.get("prompt_token_ids"), list),
            sorted((tts_params.get("codes") or {}).keys()) if isinstance(tts_params.get("codes"), dict) else None,
        )
        if request.voice:
            voice_lower = request.voice.lower()
            if voice_lower in server.uploaded_speakers and not has_inline_ref_audio:
                tts_params["voice_name"] = [voice_lower]
                tts_params["voice_created_at"] = [server._voice_created_at(voice_lower)]
        # MOSS reads the resolved seed at build time (it samples internally).
        if sampling_params_list and getattr(sampling_params_list[0], "seed", None) is not None:
            tts_params["seed"] = [sampling_params_list[0].seed]
        if isinstance(tts_params.get("prompt_token_ids"), list):
            prompt_token_ids = tts_params.pop("prompt_token_ids")
            prompt = tokens_input(prompt_token_ids=prompt_token_ids)
        else:
            prompt = tokens_input(prompt_token_ids=[1])
        prompt["additional_information"] = tts_params
        prompt["cache_salt"] = conditioning_cache_salt(request, tts_params)
        logger.info(
            "[MossTTSDebug][adapter-build-done] prompt_len=%d additional_keys=%s",
            len(prompt.get("prompt_token_ids", [])),
            sorted(tts_params.keys()),
        )
        return PreparedRequest(prompt=prompt, tts_params=tts_params, model_type=self.name)


@register_tts_adapter
class MossTTSNanoAdapter(_MossTTSAdapterBase):
    stage_keys = frozenset({"moss_tts_nano"})
    name = "moss_tts_nano"


@register_tts_adapter
class MossTTSAdapter(_MossTTSAdapterBase):
    stage_keys = frozenset({"moss_tts", "moss_tts_codec", "moss_tts_local", "moss_tts_local_codec"})
    name = "moss_tts"
