# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
FSDP shard conditions for Wan2.2 transformer models.

These conditions define which modules should be wrapped with fully_shard().
FSDP will shard the parameters of these modules across GPUs to reduce memory.

For Wan2.2, we shard:
- Transformer blocks (blocks.0, blocks.1, ...): The main computational units
  containing self-attention, cross-attention, and FFN layers. Each block has
  significant parameters that benefit from sharding.
"""

from torch import nn


def is_transformer_block(name: str, module: nn.Module) -> bool:
    """Match WanTransformerBlock layers.

    Matches modules named like 'blocks.0', 'blocks.1', etc.
    These are the main transformer blocks containing attention and FFN layers.
    """
    return "blocks" in name and name.split(".")[-1].isdigit()


# List of shard condition functions for Wan2.2 models
WAN_FSDP_SHARD_CONDITIONS = [
    is_transformer_block,
]
