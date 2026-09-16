"""Token-parallel prefill convolution with optional fused Q/K L2 normalization.

Copied verbatim from Lane 3's runtime-validated turn-2 package, module SHA-256
854c6da38b820a74e1610e26bf3c2cf68aee8b53466b8af273962b329ddc96c5.
Use explicit block_t=32, block_n=128, num_warps=4, feat_first=False. The copied
wrapper's historical defaults are retained for source identity, not selected by
the integration. The engine self-checks the exact lean launcher before use.
"""
from __future__ import annotations

from typing import List, Optional, Union

import torch
import triton
import triton.language as tl

from .qwen38_optimized import PAD_SLOT_ID


@triton.jit
def _conv_tile_history(x_base, stride_x_token, conv_states_base, stride_state_tok, tok, tok_ok,
                       mask_w, load_init_state, SHIFT: tl.constexpr, STATE_LEN: tl.constexpr):
    src = tok - SHIFT
    in_x = (src >= 0) & tok_ok
    vals = tl.load(x_base[None, :] + src.to(tl.int64)[:, None] * stride_x_token,
                   mask=in_x[:, None] & mask_w[None, :], other=0.0)
    if load_init_state:
        from_state = (src < 0) & tok_ok
        state_vals = tl.load(conv_states_base[None, :] + (STATE_LEN + src)[:, None] * stride_state_tok,
                             mask=from_state[:, None] & mask_w[None, :], other=0.0)
        vals = tl.where(from_state[:, None], state_vals, vals)
    return vals


@triton.jit()
def _causal_conv1d_split_tile_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    conv_states_ptr,
    conv_state_indices_ptr,
    has_initial_states_ptr,
    query_start_loc_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    eps,
    dim: tl.constexpr,
    num_cache_lines: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_state_seq: tl.constexpr,
    stride_state_dim: tl.constexpr,
    stride_state_tok: tl.constexpr,
    pad_slot_id: tl.constexpr,
    Q_DIM: tl.constexpr,
    K_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FEAT_FIRST: tl.constexpr,
    FUSE_L2NORM: tl.constexpr,
):
    tl.static_assert(KERNEL_WIDTH == 4, "this model uses a width-4 convolution")
    tl.static_assert(BLOCK_T >= KERNEL_WIDTH - 1, "only the first tile may read the conv state")
    state_len: tl.constexpr = KERNEL_WIDTH - 1

    # FEAT_FIRST makes the feature block the fastest-varying program index (programs
    # resident together read the same token rows). Measured slower than the
    # sequence-first order on B200 (585 vs 475 us at 64x512 for 32x128/4 warps); kept
    # as an ablation switch.
    if FEAT_FIRST:
        feat_block = tl.program_id(0) * BLOCK_N
        chunk = tl.program_id(1)
        idx_seq = tl.program_id(2)
    else:
        idx_seq = tl.program_id(0)
        chunk = tl.program_id(1)
        feat_block = tl.program_id(2) * BLOCK_N
    idx_feats = feat_block + tl.arange(0, BLOCK_N)
    mask_w = idx_feats < dim

    sequence_start_index = tl.load(query_start_loc_ptr + idx_seq)
    sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1)
    seqlen = sequence_end_index - sequence_start_index
    token_offset = chunk * BLOCK_T
    if token_offset >= seqlen:
        return
    conv_state_batch_coord = tl.load(conv_state_indices_ptr + idx_seq).to(tl.int64)
    if conv_state_batch_coord == pad_slot_id:
        return

    # [BLOCK_N] pointers to token 0 of this sequence, its conv state row and its weights.
    x_base = x_ptr + sequence_start_index.to(tl.int64) * stride_x_token + idx_feats * stride_x_dim
    conv_states_base = (
        conv_states_ptr
        + (conv_state_batch_coord * stride_state_seq)
        + (idx_feats * stride_state_dim)
    )
    w_base = w_ptr + (idx_feats * stride_w_dim)

    # Output region of this feature block: [all_q | all_k | all_v], token-major.
    if feat_block < Q_DIM:
        o_base = q_ptr + idx_feats
        o_stride = Q_DIM
    elif feat_block < Q_DIM + K_DIM:
        o_base = k_ptr + (idx_feats - Q_DIM)
        o_stride = K_DIM
    else:
        o_base = v_ptr + (idx_feats - Q_DIM - K_DIM)
        o_stride = V_DIM

    tok = token_offset + tl.arange(0, BLOCK_T)
    tok_ok = tok < seqlen
    tile_mask = tok_ok[:, None] & mask_w[None, :]

    if chunk == 0:
        load_init_state = tl.load(has_initial_states_ptr + idx_seq).to(tl.int1)
        x3 = _conv_tile_history(x_base, stride_x_token, conv_states_base, stride_state_tok, tok, tok_ok, mask_w, load_init_state, 3, state_len)
        x2 = _conv_tile_history(x_base, stride_x_token, conv_states_base, stride_state_tok, tok, tok_ok, mask_w, load_init_state, 2, state_len)
        x1 = _conv_tile_history(x_base, stride_x_token, conv_states_base, stride_state_tok, tok, tok_ok, mask_w, load_init_state, 1, state_len)

        # Conv-state update, performed only by the first tile of each sequence (upstream code).
        if state_len <= seqlen:
            idx_tokens_last = (seqlen - state_len) + tl.arange(0, NP2_STATELEN)
            x_ptrs = (
                x_ptr
                + ((sequence_start_index + idx_tokens_last).to(tl.int64) * stride_x_token)[:, None]
                + (idx_feats * stride_x_dim)[None, :]
            )
            mask_x = (
                (idx_tokens_last >= 0)[:, None]
                & (idx_tokens_last < seqlen)[:, None]
                & (idx_feats < dim)[None, :]
            )
            new_conv_state = tl.load(x_ptrs, mask_x, 0.0)
            idx_tokens_conv = tl.arange(0, NP2_STATELEN)
            conv_states_ptrs_target = (
                conv_states_base[None, :] + (idx_tokens_conv * stride_state_tok)[:, None]
            )
            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
            tl.debug_barrier()
            tl.store(conv_states_ptrs_target, new_conv_state, mask)
        else:
            idx_tokens_conv = tl.arange(0, NP2_STATELEN)
            VAL = state_len - seqlen
            x_ptrs = x_base[None, :] + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
            mask_x = (
                (idx_tokens_conv - VAL >= 0)[:, None]
                & (idx_tokens_conv - VAL < seqlen)[:, None]
                & (idx_feats < dim)[None, :]
            )
            loaded_x = tl.load(x_ptrs, mask_x, 0.0)
            if load_init_state:
                conv_states_ptrs_source = (
                    conv_states_ptr
                    + (conv_state_batch_coord * stride_state_seq)
                    + (idx_feats * stride_state_dim)[None, :]
                    + ((idx_tokens_conv + seqlen) * stride_state_tok)[:, None]
                )
                mask = (
                    (conv_state_batch_coord < num_cache_lines)
                    & ((idx_tokens_conv + seqlen) < state_len)[:, None]
                    & (idx_feats < dim)[None, :]
                )
                conv_state = tl.load(conv_states_ptrs_source, mask, other=0.0)
                tl.debug_barrier()
                new_conv_state = tl.where(mask, conv_state, loaded_x)
            else:
                new_conv_state = loaded_x
            conv_states_ptrs_target = (
                conv_states_base + (idx_tokens_conv * stride_state_tok)[:, None]
            )
            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
            tl.debug_barrier()
            tl.store(conv_states_ptrs_target, new_conv_state, mask)
    else:
        x3 = tl.load(x_base[None, :] + (tok - 3).to(tl.int64)[:, None] * stride_x_token, mask=tile_mask, other=0.0)
        x2 = tl.load(x_base[None, :] + (tok - 2).to(tl.int64)[:, None] * stride_x_token, mask=tile_mask, other=0.0)
        x1 = tl.load(x_base[None, :] + (tok - 1).to(tl.int64)[:, None] * stride_x_token, mask=tile_mask, other=0.0)
    x0 = tl.load(x_base[None, :] + tok.to(tl.int64)[:, None] * stride_x_token, mask=tile_mask, other=0.0)

    if HAS_BIAS:
        bias = tl.load(bias_ptr + idx_feats, mask=mask_w, other=0.0).to(tl.float32)
        acc = tl.broadcast_to(bias[None, :], (BLOCK_T, BLOCK_N))
    else:
        acc = tl.zeros((BLOCK_T, BLOCK_N), dtype=tl.float32)
    w_col0 = tl.load(w_base + (0 * stride_w_width), mask_w, other=0.0)
    w_col1 = tl.load(w_base + (1 * stride_w_width), mask_w, other=0.0)
    w_col2 = tl.load(w_base + (2 * stride_w_width), mask_w, other=0.0)
    w_col3 = tl.load(w_base + (3 * stride_w_width), mask_w, other=0.0)
    # Same operand types and order as upstream: bf16 products, fp32 accumulation.
    acc += x3 * w_col0[None, :]
    acc += x2 * w_col1[None, :]
    acc += x1 * w_col2[None, :]
    acc += x0 * w_col3[None, :]
    if SILU_ACTIVATION:
        acc = acc / (1 + tl.exp(-acc))
    o_rows = (sequence_start_index + tok).to(tl.int64) * o_stride
    o_rows = tl.multiple_of(o_rows, BLOCK_N)
    o_ptrs = o_base[None, :] + o_rows[:, None]
    if FUSE_L2NORM:
        # Q/K heads: the L2 normalization `l2norm_fwd_kernel` would apply to the stored
        # bf16 conv output. BLOCK_N == head_dim, and the [BLOCK_T, 128] tile has the same
        # per-row lane layout (16 lanes x 8 contiguous elements) as the reference's
        # [16, 128] tile, so the per-row FMA chain and butterfly tree are the same.
        y_bf16 = acc.to(o_base.dtype.element_ty)
        if feat_block < Q_DIM + K_DIM:
            b_x = y_bf16.to(tl.float32)
            b_var = tl.sum(b_x * b_x, axis=1)
            b_y = b_x / tl.sqrt(b_var + eps)[:, None]
            tl.store(o_ptrs, b_y.to(o_base.dtype.element_ty), mask=tile_mask)
        else:
            tl.store(o_ptrs, y_bf16, mask=tile_mask)
    else:
        tl.store(o_ptrs, acc, mask=tile_mask)


def causal_conv1d_split_tile_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Union[torch.Tensor, None],
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens_cpu: List[int],
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    activation: Optional[str],
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_dim: int,
    pad_slot_id: int = PAD_SLOT_ID,
    block_t: int = 64,
    block_n: int = 128,
    num_warps: int = 4,
    feat_first: bool = True,
    fuse_l2norm: bool = False,
    eps: float = 1e-6,
):
    """Token-parallel conv1d over the packed [dim, T] channel-last view, returning contiguous q/k/v.

    Returns tensors shaped [1, T, heads, head_dim] like `causal_conv1d_split_fn`; block_t,
    block_n, num_warps and feat_first only change work distribution and program order
    (every element is computed identically). With fuse_l2norm the Q and K outputs are the
    L2-normalized heads (what `gdn_prefill_qkv_prepare_fwd` returns), so that call is skipped.
    """
    dim, total_tokens = x.shape
    _, width = weight.shape
    if width != 4:
        raise ValueError("Fused conv/split supports the width-4 convolution only")
    if x.stride(0) != 1:
        raise ValueError("x must be channel-last: feature stride 1")
    q_dim, k_dim, v_dim = num_q_heads * head_dim, num_k_heads * head_dim, num_v_heads * head_dim
    if q_dim + k_dim + v_dim != dim:
        raise ValueError("q/k/v head layout does not cover the packed feature dimension")
    if q_dim % block_n or k_dim % block_n or v_dim % block_n:
        raise ValueError("q/k/v feature ranges must be multiples of the feature block")
    if block_t < width - 1 or block_t & (block_t - 1) or block_n & (block_n - 1):
        raise ValueError("block_t/block_n must be powers of two with block_t >= width - 1")
    if fuse_l2norm and block_n != head_dim:
        raise ValueError("fused Q/K L2 normalization needs one head per feature block")
    if conv_states.shape[1] != dim or conv_states.shape[2] < width - 1:
        raise ValueError("conv state shape does not match the packed features")
    q = torch.empty((1, total_tokens, num_q_heads, head_dim), dtype=x.dtype, device=x.device)
    k = torch.empty((1, total_tokens, num_k_heads, head_dim), dtype=x.dtype, device=x.device)
    v = torch.empty((1, total_tokens, num_v_heads, head_dim), dtype=x.dtype, device=x.device)
    num_seqs, num_chunks, num_feat_blocks = len(seq_lens_cpu), triton.cdiv(max(seq_lens_cpu), block_t), triton.cdiv(dim, block_n)
    grid = (num_feat_blocks, num_chunks, num_seqs) if feat_first else (num_seqs, num_chunks, num_feat_blocks)
    _causal_conv1d_split_tile_kernel[grid](
        x,
        weight,
        bias,
        conv_states,
        cache_indices,
        has_initial_state,
        query_start_loc,
        q,
        k,
        v,
        eps,
        dim,
        conv_states.shape[0],
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        conv_states.stride(0),
        conv_states.stride(1),
        conv_states.stride(2),
        pad_slot_id,
        Q_DIM=q_dim,
        K_DIM=k_dim,
        V_DIM=v_dim,
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ("silu", "swish"),
        NP2_STATELEN=triton.next_power_of_2(width - 1),
        BLOCK_T=block_t,
        BLOCK_N=block_n,
        FEAT_FIRST=feat_first,
        FUSE_L2NORM=fuse_l2norm,
        num_warps=num_warps,
    )
    return q, k, v
