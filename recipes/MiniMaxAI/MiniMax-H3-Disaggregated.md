# MiniMax H3 disaggregated text encoder

This opt-in topology runs the Qwen3-VL text encoder as a vLLM stage and sends
its hidden states and token-role metadata to an encoder-free diffusion stage.
The standard MiniMax H3 recipes remain single-stage and continue to load the
text encoder inside the diffusion pipeline.

## Start the server

Choose the topology explicitly and load its deployment defaults:

```bash
vllm-omni serve MiniMaxAI/MiniMax-H3 \
  --omni \
  --deploy-config vllm_omni/deploy/minimax_h3_disaggregated.yaml
```

The default deployment assigns stage 0 to GPUs 0-1 with tensor parallel size
2 and stage 1 to GPUs 2-5 with diffusion tensor parallel size 4. Adjust the
`devices`, `tensor_parallel_size`, and stage 1 `parallel_config` values in a
deployment override for the available hardware. Diffusion quantization,
layerwise offload, distributed layerwise offload, VAE parallelism, and USP
settings use the same stage 1 options documented in [MiniMax-H3.md](MiniMax-H3.md).

Stage 1 sets `model_loaded.text_encoder: false`; it must not load or download
text-encoder weights. The initial topology uses the standard stage subprocess
transport and expects all stages to run in one deployment. Cross-node payload
transport and inline multi-stage diffusion are outside this configuration.

The `/v1/videos` request schema and `extra_params.task` values (`t2va`,
`fl2va`, and `ref2va`) are unchanged from the single-stage recipe.
