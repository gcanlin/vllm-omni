from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.config.vllm import set_current_vllm_config
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.qwen2 import Qwen2MLP as Qwen3MLP
from vllm.model_executor.models.utils import is_pp_missing_parameter
from vllm.transformers_utils.config import set_default_rope_theta

from .configuration_qwen3_tts import Qwen3TTSTalkerCodePredictorConfig, Qwen3TTSTalkerConfig

# Type alias for per-layer KV cache: (k_cache, v_cache) each of shape
# [max_batch_size, num_kv_heads, max_seq_len, head_dim].
KVCache = tuple[torch.Tensor, torch.Tensor]


class CodePredictorAttention(nn.Module):
    """Standalone attention using SDPA + dense KV buffers.

    Reuses QKVParallelLinear, RowParallelLinear, RMSNorm (QK-norm), and RoPE
    from vLLM but replaces the paged-attention backend with
    ``F.scaled_dot_product_attention``.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict,
        max_position: int = 32768,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        self.num_heads = num_heads
        self.total_num_kv_heads = num_kv_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or hidden_size // num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position,
            rope_parameters=rope_parameters,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        seq_len: int,
    ) -> torch.Tensor:
        """
        Args:
            positions: [B, qlen] position ids.
            hidden_states: [B, qlen, hidden_size].
            kv_cache: (k_cache, v_cache) each [B, num_kv_heads, max_seq_len, head_dim].
            seq_len: total sequence length *after* this forward (past + current query).

        Returns:
            output: [B, qlen, hidden_size].
        """
        bsz, qlen, _ = hidden_states.shape
        k_cache, v_cache = kv_cache

        qkv, _ = self.qkv_proj(hidden_states.reshape(bsz * qlen, -1))
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # QK-norm (per head).
        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(q.shape)
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(k.shape)

        # RoPE.
        q, k = self.rotary_emb(positions.reshape(-1), q, k)

        # Reshape to [B, heads, qlen, head_dim].
        q = q.view(bsz, qlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, qlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, qlen, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Write new K/V into the dense cache at the correct positions.
        start_pos = seq_len - qlen
        k_cache[:bsz, :, start_pos:seq_len, :] = k
        v_cache[:bsz, :, start_pos:seq_len, :] = v

        # Attend over the full sequence so far.
        k_full = k_cache[:bsz, :, :seq_len, :]
        v_full = v_cache[:bsz, :, :seq_len, :]

        attn_out = F.scaled_dot_product_attention(
            q,
            k_full,
            v_full,
            scale=self.scaling,
            is_causal=(qlen == seq_len),  # True for prefill, False for decode
            enable_gqa=(self.num_heads != self.num_kv_heads),
        )
        # [B, num_heads, qlen, head_dim] -> [B*qlen, num_heads * head_dim]
        attn_out = attn_out.transpose(1, 2).reshape(bsz * qlen, -1)
        output, _ = self.o_proj(attn_out)
        return output.view(bsz, qlen, -1)


class CodePredictorDecoderLayer(nn.Module):
    """Standalone decoder layer for the code predictor.

    Same architecture as ``Qwen3DecoderLayer`` (attention + MLP with
    pre-norm residuals) but uses ``CodePredictorAttention`` instead of
    vLLM's ``Attention`` backend.  Weight names are identical so existing
    checkpoints load without changes.
    """

    def __init__(
        self,
        config: Qwen3TTSTalkerCodePredictorConfig,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        set_default_rope_theta(config, default_theta=1000000)

        self.self_attn = CodePredictorAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=getattr(config, "head_dim", None),
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            quant_config=quant_config,
            rope_parameters=config.rope_parameters,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        kv_cache: KVCache,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            seq_len=seq_len,
        )

        # MLP
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3TTSTalkerCodePredictorModelVLLM(nn.Module):
    def __init__(
        self,
        config: Qwen3TTSTalkerCodePredictorConfig,
        *,
        talker_hidden_size: int | None = None,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config

        self.layers = nn.ModuleList(
            [
                CodePredictorDecoderLayer(config, quant_config=quant_config, prefix=f"{prefix}.layers.{i}")
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Official code_predictor uses one embedding table per residual group.
        # Some Qwen3-TTS checkpoints store codec embeddings in the talker hidden
        # space, even when `code_predictor_config.hidden_size` is smaller.
        # We keep the embedding dim aligned with the checkpoint and project down
        # via `small_to_mtp_projection` in the wrapper module.
        emb_dim = int(talker_hidden_size) if talker_hidden_size is not None else int(config.hidden_size)
        self.codec_embedding = nn.ModuleList(
            [nn.Embedding(config.vocab_size, emb_dim) for _ in range(config.num_code_groups - 1)]
        )

    def get_input_embeddings(self) -> nn.ModuleList:
        return self.codec_embedding

    def forward(
        self,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor,
        kv_caches: list[KVCache],
        seq_len: int,
    ) -> torch.Tensor:
        """
        Args:
            positions: [B, qlen] position ids.
            inputs_embeds: [B, qlen, hidden_size].
            kv_caches: list of (k_cache, v_cache) per layer.
            seq_len: total sequence length after this forward.
        """
        hidden_states = inputs_embeds
        residual = None
        for layer, kv_cache in zip(self.layers, kv_caches):
            hidden_states, residual = layer(positions, hidden_states, residual, kv_cache, seq_len)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Match vLLM Qwen2/Qwen3 packing conventions: q_proj/k_proj/v_proj -> qkv_proj,
        # gate_proj/up_proj -> gate_up_proj.
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if self.quant_config is not None and (scale_name := self.quant_config.get_cache_scale(name)):
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                if mapped.endswith(".bias") and mapped not in params_dict:
                    continue
                if is_pp_missing_parameter(mapped, self):
                    continue
                if mapped.endswith("scale"):
                    mapped = maybe_remap_kv_scale_name(mapped, params_dict)
                    if mapped is None:
                        continue
                param = params_dict.get(mapped)
                if param is None:
                    continue
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(mapped)
                break
            else:
                mapped = maybe_remap_kv_scale_name(name, params_dict)
                if mapped is None:
                    continue
                if name.endswith(".bias") and mapped not in params_dict:
                    continue
                if is_pp_missing_parameter(mapped, self):
                    continue
                param = params_dict.get(mapped)
                if param is None:
                    continue
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(mapped)
        return loaded_params


class Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM(nn.Module):
    """vLLM-native code_predictor used by the AR talker (residual codebooks)."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Qwen3TTSTalkerCodePredictorConfig,
        talker_config: Qwen3TTSTalkerConfig,
        prefix: str = "code_predictor",
    ) -> None:
        super().__init__()
        self._vllm_config = vllm_config
        self.config = config
        self.talker_config = talker_config

        # Keep module/weight names aligned with official checkpoint (talker.code_predictor.model.*).
        self.model = Qwen3TTSTalkerCodePredictorModelVLLM(
            config,
            talker_hidden_size=int(talker_config.hidden_size),
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.model",
        )

        # One head per residual group.
        self.lm_head = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.vocab_size, bias=False) for _ in range(config.num_code_groups - 1)]
        )

        if config.hidden_size != talker_config.hidden_size:
            self.small_to_mtp_projection = nn.Linear(talker_config.hidden_size, config.hidden_size, bias=True)
        else:
            self.small_to_mtp_projection = nn.Identity()

        # Dense KV cache state (allocated lazily).
        self._kv_caches: list[KVCache] | None = None
        self._max_seq_len = int(getattr(config, "num_code_groups", 16) or 16)
        self._num_layers = int(config.num_hidden_layers)
        self._num_kv_heads = int(config.num_key_value_heads)
        self._head_dim = int(getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads)

    def get_input_embeddings(self) -> nn.ModuleList:
        return self.model.get_input_embeddings()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        with set_current_vllm_config(self._vllm_config):
            loaded: set[str] = set()
            model_weights: list[tuple[str, torch.Tensor]] = []
            other_weights: list[tuple[str, torch.Tensor]] = []
            for name, w in weights:
                if name.startswith("model."):
                    model_weights.append((name[len("model.") :], w))
                else:
                    other_weights.append((name, w))

            loaded_model = self.model.load_weights(model_weights)
            loaded |= {f"model.{n}" for n in loaded_model}

            params = dict(self.named_parameters(remove_duplicate=False))
            for name, w in other_weights:
                if name not in params:
                    continue
                default_weight_loader(params[name], w)
                loaded.add(name)
            return loaded

    def _allocate_kv_caches(self, batch_size: int, device: torch.device) -> list[KVCache]:
        """Allocate dense KV cache tensors for all layers."""
        caches: list[KVCache] = []
        for _ in range(self._num_layers):
            k = torch.zeros(
                batch_size,
                self._num_kv_heads,
                self._max_seq_len,
                self._head_dim,
                dtype=torch.bfloat16,
                device=device,
            )
            v = torch.zeros(
                batch_size,
                self._num_kv_heads,
                self._max_seq_len,
                self._head_dim,
                dtype=torch.bfloat16,
                device=device,
            )
            caches.append((k, v))
        return caches

    @torch.inference_mode()
    def reset_cache(self) -> None:
        if self._kv_caches is not None:
            for k, v in self._kv_caches:
                k.zero_()
                v.zero_()

    @torch.inference_mode()
    def prefill_logits(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """Prefill with 2 tokens: [past_hidden, layer0_embed]. Returns logits for residual group 0."""
        bsz = int(inputs_embeds.shape[0])
        qlen = 2
        device = inputs_embeds.device

        # Allocate / re-allocate KV caches if needed.
        if self._kv_caches is None or self._kv_caches[0][0].shape[0] < bsz:
            self._kv_caches = self._allocate_kv_caches(bsz, device)

        hs = inputs_embeds.to(dtype=torch.bfloat16)  # [B, 2, H]
        hs = self.small_to_mtp_projection(hs.reshape(bsz * qlen, -1)).view(bsz, qlen, -1)

        positions = torch.arange(qlen, dtype=torch.long, device=device).unsqueeze(0).expand(bsz, -1)

        out = self.model(positions=positions, inputs_embeds=hs, kv_caches=self._kv_caches, seq_len=qlen)

        # Gather last token per request.
        last_h = out[:, -1, :]  # [B, hidden]
        logits = self.lm_head[0](last_h)
        return logits

    @torch.inference_mode()
    def decode_logits(self, input_ids: torch.Tensor, *, generation_step: int, past_seq_len: int) -> torch.Tensor:
        """Decode one new token for residual group `generation_step` (1..Q-1)."""
        assert self._kv_caches is not None
        bsz = int(input_ids.shape[0])
        if generation_step <= 0:
            raise ValueError("generation_step must be >= 1 for decode_logits")

        embed_idx = generation_step - 1
        hs = self.model.get_input_embeddings()[embed_idx](input_ids.to(dtype=torch.long).reshape(bsz, 1))
        hs = self.small_to_mtp_projection(hs.reshape(bsz, -1)).view(bsz, 1, -1)

        seq_len = past_seq_len + 1
        positions = torch.full((bsz, 1), past_seq_len, dtype=torch.long, device=input_ids.device)

        out = self.model(positions=positions, inputs_embeds=hs, kv_caches=self._kv_caches, seq_len=seq_len)

        logits = self.lm_head[generation_step](out[:, 0, :])
        return logits

    @torch.inference_mode()
    def forward(
        self,
        layer0_code: torch.Tensor,
        layer0_embed: torch.Tensor,
        last_talker_hidden: torch.Tensor,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """Full autoregressive prediction of residual codebooks 1..Q-1.

        Args:
            layer0_code: [B, 1] first-layer codec token ids.
            layer0_embed: [B, 1, H] embedding of layer0_code.
            last_talker_hidden: [B, 1, H] hidden state from the talker.
            do_sample: whether to sample or take argmax.
            temperature: sampling temperature.
            top_k: top-k filtering.
            top_p: top-p (nucleus) filtering.

        Returns:
            audio_codes: [B, Q] all codebook tokens (layer0 + residuals).
        """
        bsz = int(layer0_code.shape[0])
        num_groups = int(self.config.num_code_groups)
        max_steps = num_groups - 1

        # Reset KV cache for a fresh sequence.
        self.reset_cache()

        # Prefill: feed [last_talker_hidden, layer0_embed] → logits for group 1.
        prefill_input = torch.cat([last_talker_hidden, layer0_embed], dim=1)  # [B, 2, H]
        logits = self.prefill_logits(prefill_input)  # [B, vocab]

        all_codes = [layer0_code.reshape(bsz, 1)]
        past_seq_len = 2

        for step in range(1, num_groups):
            # Sample or argmax from logits.
            if do_sample and temperature > 0:
                scaled = logits / temperature
                if top_k > 0:
                    topk_vals, _ = scaled.topk(top_k, dim=-1)
                    scaled = scaled.masked_fill(scaled < topk_vals[:, -1:], float("-inf"))
                probs = torch.softmax(scaled, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1)  # [B, 1]
            else:
                next_ids = logits.argmax(dim=-1, keepdim=True)  # [B, 1]
            all_codes.append(next_ids)

            # If not the last step, decode one more token.
            if step < max_steps:
                logits = self.decode_logits(
                    next_ids.reshape(bsz),
                    generation_step=step,
                    past_seq_len=past_seq_len,
                )
                past_seq_len += 1

        return torch.cat(all_codes, dim=1)  # [B, Q]
