# Copyright 2026 OpenMOSS and the vLLM-Omni team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""GPU input preparation for MOSS-TTS streaming decode."""

from collections.abc import Iterable

import torch
from vllm.triton_utils import tl, triton

_MASK_WORD_BITS = 32
_NUM_MASK_WORDS = 8
MAX_STREAM_SLOTS = _MASK_WORD_BITS * _NUM_MASK_WORDS
_CODE_BLOCK_SIZE = 256


def encode_slot_mask(slots: Iterable[int], num_slots: int) -> int:
    """Encode active stream slots without creating a host or device tensor."""
    if not 0 <= num_slots <= MAX_STREAM_SLOTS:
        raise ValueError(f"num_slots must be in [0, {MAX_STREAM_SLOTS}], got {num_slots}")

    active_slot_mask = 0
    for slot in slots:
        slot = int(slot)
        if not 0 <= slot < num_slots:
            raise IndexError(f"Codec stream slot {slot} is outside [0, {num_slots}).")
        active_slot_mask |= 1 << slot
    return active_slot_mask


def _to_signed_int32(value: int) -> int:
    return value if value < (1 << 31) else value - (1 << 32)


def _mask_words(active_slot_mask: int) -> tuple[int, ...]:
    if active_slot_mask < 0 or active_slot_mask.bit_length() > MAX_STREAM_SLOTS:
        raise ValueError(
            f"active_slot_mask must fit in {MAX_STREAM_SLOTS} bits, got bit_length={active_slot_mask.bit_length()}"
        )
    word_mask = (1 << _MASK_WORD_BITS) - 1
    return tuple(
        _to_signed_int32((active_slot_mask >> (word_idx * _MASK_WORD_BITS)) & word_mask)
        for word_idx in range(_NUM_MASK_WORDS)
    )


@triton.jit(
    do_not_specialize=[
        "mask_word_0",
        "mask_word_1",
        "mask_word_2",
        "mask_word_3",
        "mask_word_4",
        "mask_word_5",
        "mask_word_6",
        "mask_word_7",
    ]
)
def _prepare_streaming_inputs_kernel(
    source_codes,
    static_codes,
    exec_mask,
    num_codes,
    num_slots,
    mask_word_0,
    mask_word_1,
    mask_word_2,
    mask_word_3,
    mask_word_4,
    mask_word_5,
    mask_word_6,
    mask_word_7,
    copy_codes: tl.constexpr,
    code_block_size: tl.constexpr,
    mask_block_size: tl.constexpr,
    mask_word_bits: tl.constexpr,
):
    program_id = tl.program_id(0)

    if copy_codes:
        code_offsets = program_id * code_block_size + tl.arange(0, code_block_size)
        code_mask = code_offsets < num_codes
        codes = tl.load(source_codes + code_offsets, mask=code_mask)
        tl.store(static_codes + code_offsets, codes, mask=code_mask)

    if program_id == 0:
        slot_offsets = tl.arange(0, mask_block_size)
        word_index = slot_offsets // mask_word_bits
        bit_index = slot_offsets % mask_word_bits
        word = mask_word_0
        word = tl.where(word_index == 1, mask_word_1, word)
        word = tl.where(word_index == 2, mask_word_2, word)
        word = tl.where(word_index == 3, mask_word_3, word)
        word = tl.where(word_index == 4, mask_word_4, word)
        word = tl.where(word_index == 5, mask_word_5, word)
        word = tl.where(word_index == 6, mask_word_6, word)
        word = tl.where(word_index == 7, mask_word_7, word)
        active = ((word >> bit_index) & 1) != 0
        tl.store(
            exec_mask + slot_offsets,
            active,
            mask=slot_offsets < num_slots,
        )


def _prepare_streaming_inputs(
    source_codes: torch.Tensor,
    static_codes: torch.Tensor,
    exec_mask: torch.Tensor,
    active_slot_mask: int,
    *,
    copy_codes: bool,
) -> None:
    assert exec_mask.is_cuda and exec_mask.dtype == torch.bool
    assert exec_mask.ndim == 1 and exec_mask.numel() <= MAX_STREAM_SLOTS
    num_codes = source_codes.numel() if copy_codes else 0
    grid = (max(1, triton.cdiv(num_codes, _CODE_BLOCK_SIZE)),)
    mask_block_size = triton.next_power_of_2(max(1, exec_mask.numel()))
    _prepare_streaming_inputs_kernel[grid](
        source_codes,
        static_codes,
        exec_mask,
        num_codes,
        exec_mask.numel(),
        *_mask_words(active_slot_mask),
        copy_codes=copy_codes,
        code_block_size=_CODE_BLOCK_SIZE,
        mask_block_size=mask_block_size,
        mask_word_bits=_MASK_WORD_BITS,
    )


def prepare_streaming_inputs(
    source_codes: torch.Tensor,
    static_codes: torch.Tensor,
    exec_mask: torch.Tensor,
    active_slot_mask: int,
) -> None:
    """Copy dynamic codes and update the graph's address-stable exec mask."""
    assert source_codes.is_cuda and source_codes.is_contiguous()
    assert static_codes.is_cuda and static_codes.is_contiguous()
    assert source_codes.shape == static_codes.shape
    assert source_codes.dtype == static_codes.dtype
    assert source_codes.device == static_codes.device == exec_mask.device
    _prepare_streaming_inputs(
        source_codes,
        static_codes,
        exec_mask,
        active_slot_mask,
        copy_codes=True,
    )


def prepare_streaming_exec_mask(
    exec_mask: torch.Tensor,
    active_slot_mask: int,
) -> None:
    """Update an address-stable exec mask without a host-to-device copy."""
    _prepare_streaming_inputs(
        exec_mask,
        exec_mask,
        exec_mask,
        active_slot_mask,
        copy_codes=False,
    )


__all__ = [
    "MAX_STREAM_SLOTS",
    "encode_slot_mask",
    "prepare_streaming_exec_mask",
    "prepare_streaming_inputs",
]
