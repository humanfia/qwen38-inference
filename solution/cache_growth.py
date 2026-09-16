"""Grow all paged KV tensors with one allocation and one copy/zero kernel.

The input may contain stale tokens after HybridCache.reset(). Only each slot's
live prefix is copied; all other tokens, including the reserved page, are zero.
The returned views remain ordinary, disjoint, contiguous tensors, so the
reference cache's clone(), reset(), and attention views continue to work.
"""
from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _grow_kv(OLD_PTRS, LENGTHS, NEW, OLD_CAP: tl.constexpr, CAP: tl.constexpr,
             BATCH: tl.constexpr, PAGE: tl.constexpr, BLOCK: tl.constexpr):
    # The packed allocation can exceed 2^31 elements even though each tensor's
    # offsets fit int32. Widen before multiplying the tensor index by its size.
    tensor = tl.program_id(1).to(tl.int64)
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    size: tl.constexpr = (BATCH * CAP + PAGE) * 1024
    token = offsets // 1024
    slot = (token - PAGE) // CAP
    position = (token - PAGE) % CAP
    in_slot = (token >= PAGE) & (slot < BATCH)
    length = tl.load(LENGTHS + slot, mask=in_slot, other=0)
    old = tl.load(OLD_PTRS + tensor).to(tl.pointer_type(NEW.dtype.element_ty))
    source = (PAGE + slot * OLD_CAP + position) * 1024 + offsets % 1024
    value = tl.load(old + source, mask=(offsets < size) & in_slot & (position < length), other=0)
    tl.store(NEW + tensor * size + offsets, value, mask=offsets < size)


def allocate_packed(cache, capacity, page_size, device, dtype, *, zero):
    """Allocate one backing storage and expose the reference per-layer layout."""
    shape = (2 * len(cache.kv), len(cache.lengths) * capacity + page_size, 4, 256)
    storage = (torch.zeros if zero else torch.empty)(shape, device=device, dtype=dtype)
    kv = {layer: (storage[2 * j], storage[2 * j + 1]) for j, layer in enumerate(cache.kv)}
    return storage, kv


def grow_fused(cache, capacity, page_size, device, dtype, ring):
    """Include pointer metadata staging and all prefix copy/zero work in this call."""
    has_history = any(cache.lengths)
    storage, kv = allocate_packed(cache, capacity, page_size, device, dtype, zero=not has_history)
    if has_history:
        pointers = np.asarray([t.data_ptr() for pair in cache.kv.values() for t in pair], dtype=np.int64)
        lengths = np.asarray(cache.lengths, dtype=np.int64)
        old_ptrs, lens = ring.upload([pointers, lengths])
        size = storage[0].numel()
        _grow_kv[(triton.cdiv(size, 4096), storage.shape[0])](
            old_ptrs, lens, storage, cache.capacity, capacity, len(cache.lengths), page_size,
            BLOCK=4096, num_warps=4,
        )
    return kv
