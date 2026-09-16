"""Bitwise-neutral fused kernels from Lane 3's validated turn-1b package, as plain functions.

Source: shared/artifacts/lane-3/turn1b/source.tar.gz, solution/qwen38_optimized.py
(sha256 adbddd672ac0cd2f804eeac6cfb086aa8af64e8e9ce6e6cf67b235ff6e8e60a4) and
solution/lane3_trials.py (sha256 bac24c6d1ae552ac86e6fb5f6f5dd4f4eed651bdc911bdd7abd1c25bf5e88f53).
The Triton kernels below are copied verbatim from that package; the reference kernel
library copy ``solution.qwen38_optimized`` stays identical to ``qwen38_inference.py``.

Each kernel reproduces the reference arithmetic element for element (same operand
dtypes, accumulation order and math functions) and only changes data movement or
work distribution, which Lane 3 validated bitwise on the complete 40-case suite:

* ``_fused_qk_rmsnorm_rope_gate_kvstore_kernel``: QK GemmaRMSNorm + RoPE + gate split
  whose K programs write the rotated key and copy the value straight into the paged
  cache rows (replaces the RoPE kernel plus the separate ``store`` launch).
* ``_causal_conv1d_split_fwd_kernel``: the prefill causal conv1d writing token-major
  Q/K/V directly (replaces conv1d plus the strided split kernel).
* ``fused_gdn_gating_blocked_kernel``: token-blocked gating grid with the forget gate
  emitted in the linear domain (libdevice exp equals torch.exp bitwise), removing the
  per-layer ``torch.exp`` launch.
* ``_joint_l2norm``: Q and K L2 normalisation in one launch with the reference's tile.
* ``rms_norm_gated_rows``: the reference gated RMSNorm with an explicit rows-per-program
  choice (prefill uses 8 where the reference heuristic picks 4; decode is unchanged).

``check_*`` helpers compare a launcher against the reference kernels on random data and
are used both by ``solution.engine`` (one-time self-check before a fused path is
enabled) and by ``solution.check_kernels``.
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from . import qwen38_optimized as K

_ENABLE_PDL = K._ENABLE_PDL
PAD_SLOT_ID = K.PAD_SLOT_ID


# Lane 3: same math as `_fused_qk_rmsnorm_rope_gate_kernel` (text RoPE only), but the
# K programs write the rotated key and copy the value straight into the paged cache
# rows given by `locations`, so the separate store_kvcache launch and the [T, 1024]
# k_out intermediate disappear. Q and gate outputs are unchanged. Each program handles
# Q_PER_PROG query heads or K_PER_PROG key heads of one token in sequence; every head
# is processed by exactly the upstream 1D code path (same tensor shapes, same layout,
# same reduction tree), so grouping heads only cuts the CTA count (bitwise identical).
@triton.jit
def _rope_one_head(
    in_base,
    w_ptr,
    out_base,
    cos_sin_cache_ptr,
    pos,
    stride_cos_t,
    HEAD_DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    HALF_ROTARY: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    ROT_HALF_BLOCK: tl.constexpr,
    EPS: tl.constexpr,
    FP16: tl.constexpr,
    HAS_PASS: tl.constexpr,
):
    out_dtype = tl.float16 if FP16 else tl.bfloat16
    head_offs = tl.arange(0, HEAD_BLOCK)
    head_mask = head_offs < HEAD_DIM
    x = tl.load(in_base + head_offs, mask=head_mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + head_offs, mask=head_mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / HEAD_DIM
    inv_rms = tl.rsqrt(var + EPS)
    x_norm = (x * inv_rms * (w + 1.0)).to(out_dtype).to(tl.float32)

    if HAS_PASS:
        pass_mask = head_mask & (head_offs >= ROTARY_DIM)
        tl.store(out_base + head_offs, x_norm, mask=pass_mask)

    rot_offs = tl.arange(0, ROT_HALF_BLOCK)
    rot_mask = rot_offs < HALF_ROTARY
    xr1 = tl.load(in_base + rot_offs, mask=rot_mask, other=0.0).to(tl.float32)
    xr2 = tl.load(in_base + HALF_ROTARY + rot_offs, mask=rot_mask, other=0.0).to(
        tl.float32
    )
    wr1 = tl.load(w_ptr + rot_offs, mask=rot_mask, other=0.0).to(tl.float32)
    wr2 = tl.load(w_ptr + HALF_ROTARY + rot_offs, mask=rot_mask, other=0.0).to(
        tl.float32
    )
    xr1 = (xr1 * inv_rms * (wr1 + 1.0)).to(out_dtype).to(tl.float32)
    xr2 = (xr2 * inv_rms * (wr2 + 1.0)).to(out_dtype).to(tl.float32)

    cache_off = pos.to(tl.int64) * stride_cos_t
    cos = tl.load(
        cos_sin_cache_ptr + cache_off + rot_offs, mask=rot_mask, other=0.0
    ).to(tl.float32)
    sin = tl.load(
        cos_sin_cache_ptr + cache_off + HALF_ROTARY + rot_offs, mask=rot_mask, other=0.0
    ).to(tl.float32)
    tl.store(out_base + rot_offs, (xr1 * cos - xr2 * sin), mask=rot_mask)
    tl.store(out_base + HALF_ROTARY + rot_offs, (xr2 * cos + xr1 * sin), mask=rot_mask)


@triton.jit
def _fused_qk_rmsnorm_rope_gate_kvstore_kernel(
    q_gate_ptr,
    k_ptr,
    v_ptr,
    q_out_ptr,
    gate_out_ptr,
    k_cache_ptr,
    v_cache_ptr,
    locations_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    stride_qg_t,
    stride_k_t,
    stride_v_t,
    stride_qo_t,
    stride_gate_t,
    stride_kc_row,
    stride_vc_row,
    stride_cos_t,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    Q_PER_PROG: tl.constexpr,
    K_PER_PROG: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    HALF_ROTARY: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    ROT_HALF_BLOCK: tl.constexpr,
    EPS: tl.constexpr,
    FP16: tl.constexpr,
    HAS_PASS: tl.constexpr,
    HAS_GATE: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    tl.static_assert(NUM_Q_HEADS % Q_PER_PROG == 0)
    tl.static_assert(NUM_KV_HEADS % K_PER_PROG == 0)
    NUM_Q_GROUPS: tl.constexpr = NUM_Q_HEADS // Q_PER_PROG
    token = tl.program_id(0)
    group = tl.program_id(1)
    pos = tl.load(positions_ptr + token)
    head_offs = tl.arange(0, HEAD_BLOCK)
    head_mask = head_offs < HEAD_DIM

    if group < NUM_Q_GROUPS:
        for j in tl.static_range(Q_PER_PROG):
            local_head = group * Q_PER_PROG + j
            if HAS_GATE:
                in_base = q_gate_ptr + token * stride_qg_t + local_head * 2 * HEAD_DIM
            else:
                in_base = q_gate_ptr + token * stride_qg_t + local_head * HEAD_DIM
            out_base = q_out_ptr + token * stride_qo_t + local_head * HEAD_DIM
            _rope_one_head(in_base, q_weight_ptr, out_base, cos_sin_cache_ptr, pos, stride_cos_t,
                           HEAD_DIM, ROTARY_DIM, HALF_ROTARY, HEAD_BLOCK, ROT_HALF_BLOCK, EPS, FP16, HAS_PASS)
            if HAS_GATE:
                gate_in = in_base + HEAD_DIM
                gate_out = gate_out_ptr + token * stride_gate_t + local_head * HEAD_DIM
                g = tl.load(gate_in + head_offs, mask=head_mask, other=0.0)
                tl.store(gate_out + head_offs, g, mask=head_mask)
    else:
        location = tl.load(locations_ptr + token).to(tl.int64)
        for j in tl.static_range(K_PER_PROG):
            local_head = (group - NUM_Q_GROUPS) * K_PER_PROG + j
            in_base = k_ptr + token * stride_k_t + local_head * HEAD_DIM
            out_base = k_cache_ptr + location * stride_kc_row + local_head * HEAD_DIM
            # Value pass-through into the paged cache (pure data movement).
            v_in = v_ptr + token * stride_v_t + local_head * HEAD_DIM
            v_out = v_cache_ptr + location * stride_vc_row + local_head * HEAD_DIM
            v_values = tl.load(v_in + head_offs, mask=head_mask, other=0.0)
            tl.store(v_out + head_offs, v_values, mask=head_mask)
            _rope_one_head(in_base, k_weight_ptr, out_base, cos_sin_cache_ptr, pos, stride_cos_t,
                           HEAD_DIM, ROTARY_DIM, HALF_ROTARY, HEAD_BLOCK, ROT_HALF_BLOCK, EPS, FP16, HAS_PASS)

    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def fused_qk_gemma_rmsnorm_rope_gate_kvstore(
    q_gate: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    locations: torch.Tensor,
    k_cache_rows: torch.Tensor,
    v_cache_rows: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    q_per_prog: Optional[int] = None,
    k_per_prog: Optional[int] = None,
    group_threshold: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """QK GemmaRMSNorm + RoPE + gate split; K and V are written into the cache rows.

    Heads per program default to 1 below `group_threshold` tokens (latency-bound decode)
    and to 8 query / 4 key heads above it (CTA-launch-bound prefill).
    """
    T = q_gate.shape[0]
    kv_size = num_kv_heads * head_dim
    if k.shape != (T, kv_size) or v.shape != (T, kv_size):
        raise ValueError("k and v must be [T, num_kv_heads * head_dim]")
    if k_cache_rows.ndim != 2 or v_cache_rows.ndim != 2 or k_cache_rows.shape[1] != kv_size or v_cache_rows.shape[1] != kv_size:
        raise ValueError("cache rows must be [rows, num_kv_heads * head_dim]")
    if k_cache_rows.stride(1) != 1 or v_cache_rows.stride(1) != 1 or k.stride(1) != 1 or v.stride(1) != 1:
        raise ValueError("feature dimensions must be contiguous")
    if locations.shape != (T,) or positions.shape != (T,):
        raise ValueError("locations and positions must be [T]")
    if q_per_prog is None:
        q_per_prog = 8 if T >= group_threshold else 1
    if k_per_prog is None:
        k_per_prog = 4 if T >= group_threshold else 1
    if num_q_heads % q_per_prog or num_kv_heads % k_per_prog:
        raise ValueError("heads per program must divide the head counts")
    q_size = num_q_heads * head_dim
    q_out = torch.empty(T, q_size, dtype=q_gate.dtype, device=q_gate.device)
    gate_out = torch.empty(T, num_q_heads, head_dim, dtype=q_gate.dtype, device=q_gate.device)
    half_rotary = rotary_dim // 2
    grid = (T, num_q_heads // q_per_prog + num_kv_heads // k_per_prog)
    _fused_qk_rmsnorm_rope_gate_kvstore_kernel[grid](
        q_gate,
        k,
        v,
        q_out,
        gate_out,
        k_cache_rows,
        v_cache_rows,
        locations,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        q_gate.stride(0),
        k.stride(0),
        v.stride(0),
        q_out.stride(0),
        gate_out.stride(0),
        k_cache_rows.stride(0),
        v_cache_rows.stride(0),
        cos_sin_cache.stride(0),
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        Q_PER_PROG=q_per_prog,
        K_PER_PROG=k_per_prog,
        HEAD_DIM=head_dim,
        ROTARY_DIM=rotary_dim,
        HALF_ROTARY=half_rotary,
        HEAD_BLOCK=triton.next_power_of_2(head_dim),
        ROT_HALF_BLOCK=triton.next_power_of_2(half_rotary),
        EPS=eps,
        FP16=q_gate.dtype == torch.float16,
        HAS_PASS=rotary_dim < head_dim,
        HAS_GATE=True,
        ENABLE_PDL=_ENABLE_PDL,
    )
    return q_out, gate_out



# Lane 3: causal conv1d that writes token-major Q/K/V directly.
#
# The upstream pair `causal_conv1d_fn` + `fused_qkv_split_gdn_prefill` materializes a
# channel-major [dim, T] conv output (torch.empty_like falls back to a contiguous
# [dim, T] buffer for the non-dense transposed view) and then reads it back with a
# stride of T elements between neighbouring lanes. This kernel keeps the exact
# arithmetic of `_causal_conv1d_fwd_kernel` (same operand dtypes, accumulation
# order, activation, and conv-state update) but stores every token's 256-feature
# block contiguously into the separate q/k/v tensors the chunk kernel consumes.
@triton.jit()
def _causal_conv1d_split_fwd_kernel(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tl.static_assert(KERNEL_WIDTH == 4, "this model uses a width-4 convolution")
    state_len: tl.constexpr = KERNEL_WIDTH - 1

    idx_seq = tl.program_id(0)
    chunk_offset = tl.program_id(1)
    feat_block = tl.program_id(2) * BLOCK_N
    idx_feats = feat_block + tl.arange(0, BLOCK_N)

    sequence_start_index = tl.load(query_start_loc_ptr + idx_seq)
    sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1)
    seqlen = sequence_end_index - sequence_start_index
    token_offset = BLOCK_M * chunk_offset
    segment_len = min(BLOCK_M, seqlen - token_offset)
    if segment_len <= 0:
        return

    x_base = (
        x_ptr
        + sequence_start_index.to(tl.int64) * stride_x_token
        + idx_feats * stride_x_dim
    )
    conv_state_batch_coord = tl.load(conv_state_indices_ptr + idx_seq).to(tl.int64)
    if conv_state_batch_coord == pad_slot_id:
        return
    conv_states_base = (
        conv_states_ptr
        + (conv_state_batch_coord * stride_state_seq)
        + (idx_feats * stride_state_dim)
    )
    w_base = w_ptr + (idx_feats * stride_w_dim)
    mask_w = idx_feats < dim

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

    if chunk_offset == 0:
        load_init_state = tl.load(has_initial_states_ptr + idx_seq).to(tl.int1)
        if load_init_state:
            x_elem_ty = x_ptr.dtype.element_ty
            prior_tokens = conv_states_base + (state_len - 1) * stride_state_tok
            col2 = tl.load(prior_tokens, mask_w, 0.0).to(x_elem_ty)
            col1 = tl.load(prior_tokens - 1 * stride_state_tok, mask_w, 0.0).to(x_elem_ty)
            col0 = tl.load(prior_tokens - 2 * stride_state_tok, mask_w, 0.0).to(x_elem_ty)
        else:
            col0 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            col1 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            col2 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)

        # Conv-state update, performed only by the first chunk of each sequence.
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
            tl.store(conv_states_ptrs_target, new_conv_state, mask)
    else:
        prior_tokens = x_base + (token_offset - 1).to(tl.int64) * stride_x_token
        col2 = tl.load(prior_tokens, mask_w, 0.0, cache_modifier=".ca")
        col1 = tl.load(prior_tokens - 1 * stride_x_token, mask_w, 0.0, cache_modifier=".ca")
        col0 = tl.load(prior_tokens - 2 * stride_x_token, mask_w, 0.0, cache_modifier=".ca")

    if HAS_BIAS:
        acc_preload = tl.load(bias_ptr + idx_feats, mask=mask_w, other=0.0).to(tl.float32)
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    x_base_1d = x_base + token_offset.to(tl.int64) * stride_x_token
    w_col0 = tl.load(w_base + (0 * stride_w_width), mask_w, other=0.0)
    w_col1 = tl.load(w_base + (1 * stride_w_width), mask_w, other=0.0)
    w_col2 = tl.load(w_base + (2 * stride_w_width), mask_w, other=0.0)
    w_col3 = tl.load(w_base + (3 * stride_w_width), mask_w, other=0.0)
    o_row = (sequence_start_index + token_offset).to(tl.int64)
    for idx_token in range(segment_len):
        matrix_x = tl.load(x_base_1d + idx_token * stride_x_token, mask=mask_w)
        acc = acc_preload
        # Same operand types and order as upstream: bf16 products, fp32 accumulation.
        acc += col0 * w_col0
        acc += col1 * w_col1
        acc += col2 * w_col2
        acc += matrix_x * w_col3
        col0 = col1
        col1 = col2
        col2 = matrix_x
        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        tl.store(o_base + (o_row + idx_token) * o_stride, acc, mask=mask_w)


def causal_conv1d_split_fn(
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
    block_n: int = 256,
    num_warps: int = 4,
):
    """Conv1d over the packed [dim, T] channel-last view, returning contiguous q/k/v.

    Returns tensors shaped [1, T, heads, head_dim] like `fused_qkv_split_gdn_prefill`.
    block_n/num_warps only change work distribution; every element is computed identically.
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
        raise ValueError("q/k/v feature ranges must be multiples of the 256-lane block")
    if conv_states.shape[1] != dim or conv_states.shape[2] < width - 1:
        raise ValueError("conv state shape does not match the packed features")
    q = torch.empty((1, total_tokens, num_q_heads, head_dim), dtype=x.dtype, device=x.device)
    k = torch.empty((1, total_tokens, num_k_heads, head_dim), dtype=x.dtype, device=x.device)
    v = torch.empty((1, total_tokens, num_v_heads, head_dim), dtype=x.dtype, device=x.device)
    block_m = 8
    grid = (len(seq_lens_cpu), triton.cdiv(max(seq_lens_cpu), block_m), triton.cdiv(dim, block_n))
    _causal_conv1d_split_fwd_kernel[grid](
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
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=2,
    )
    return q, k, v



# Lane 3: the upstream gating grid launches one single-warp program per (token, 8 heads),
# i.e. 6*T programs. This 2D version covers BLOCK_T tokens x all heads per program with
# the same per-element arithmetic (bitwise identical values).
@triton.jit
def fused_gdn_gating_blocked_kernel(
    g,
    beta_output,
    A_log,
    a,
    b,
    dt_bias,
    T,
    stride_a,
    stride_b,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLK_HEADS: tl.constexpr,
    OUTPUT_EXP: tl.constexpr,
):
    i_t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    head_off = tl.arange(0, BLK_HEADS)
    mask = (i_t < T)[:, None] & (head_off < NUM_HEADS)[None, :]
    off = i_t[:, None] * NUM_HEADS + head_off[None, :]
    blk_A_log = tl.load(A_log + head_off, mask=head_off < NUM_HEADS)
    blk_a = tl.load(a + i_t[:, None] * stride_a + head_off[None, :], mask=mask)
    blk_b = tl.load(b + i_t[:, None] * stride_b + head_off[None, :], mask=mask)
    blk_bias = tl.load(dt_bias + head_off, mask=head_off < NUM_HEADS)
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)[None, :]
    softplus_x = tl.where(
        beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
    )
    blk_g = -tl.exp(blk_A_log.to(tl.float32))[None, :] * softplus_x
    if OUTPUT_EXP:
        blk_g = libdevice.exp(blk_g)
    tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)
    blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))
    tl.store(beta_output + off, blk_beta_output.to(b.dtype.element_ty), mask=mask)


def fused_gdn_gating_blocked(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
    output_exp: bool = False,
    block_t: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Same outputs as fused_gdn_gating ([1, T, H] fp32 each) with a token-blocked grid."""
    tokens, num_heads = a.shape
    g = torch.empty(1, tokens, num_heads, dtype=torch.float32, device=a.device)
    beta_output = torch.empty(1, tokens, num_heads, dtype=torch.float32, device=b.device)
    fused_gdn_gating_blocked_kernel[(triton.cdiv(tokens, block_t),)](
        g,
        beta_output,
        A_log,
        a,
        b,
        dt_bias,
        tokens,
        a.stride(0),
        b.stride(0),
        NUM_HEADS=num_heads,
        beta=beta,
        threshold=threshold,
        BLOCK_T=block_t,
        BLK_HEADS=triton.next_power_of_2(num_heads),
        OUTPUT_EXP=output_exp,
        num_warps=4,
    )
    return g, beta_output




# Lane 3 (lane3_trials.py): Q and K L2 normalisation in one launch. Same [16, D] tile,
# arithmetic and launch configuration as the reference l2norm_fwd_kernel; the second grid
# dimension merges the Q and K work.
@triton.jit(do_not_specialize=["T"])
def _joint_l2norm(Q, K, QO, KO, eps, T, D: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr):
    is_k = tl.program_id(1) == 1
    x = tl.where(is_k, K, Q)
    y = tl.where(is_k, KO, QO)
    i_t = tl.program_id(0)
    px = tl.make_block_ptr(x, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    v = tl.load(px, boundary_check=(0, 1)).to(tl.float32)
    var = tl.sum(v * v, axis=1)
    out = v / tl.sqrt(var + eps)[:, None]
    py = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    tl.store(py, out.to(py.dtype.element_ty), boundary_check=(0, 1))


def joint_l2norm(q: torch.Tensor, k: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """l2norm_fwd(q, eps), l2norm_fwd(k, eps) for contiguous [T, H, D] inputs (D <= 512) in one launch."""
    if not (q.is_contiguous() and k.is_contiguous()) or q.shape != k.shape or q.dtype != k.dtype or q.shape[-1] > 512:
        raise ValueError("joint_l2norm needs equal contiguous q/k with head dim <= 512")
    qo, ko = torch.empty_like(q), torch.empty_like(k)
    d = q.shape[-1]
    rows = q.numel() // d
    _joint_l2norm[(triton.cdiv(rows, 16), 2)](q, k, qo, ko, eps, rows, d, 16, triton.next_power_of_2(d), num_warps=8, num_stages=3)
    return qo, ko


def rms_norm_gated_rows(x, weight, z, eps, rows_per_block=None):
    """Reference rms_norm_gated(norm_before_gate=True, is_rms_norm=True, activation='swish') with an explicit rows-per-program."""
    x_shape_og = x.shape
    x = x.reshape(-1, x.shape[-1])
    if x.stride(-1) != 1:
        x = x.contiguous()
    if z.shape == x_shape_og:
        z = z.reshape(-1, z.shape[-1])
        if z.stride(-1) != 1:
            z = z.contiguous()
        z_is_3d = False
    else:
        assert z.ndim == 3 and z.shape[0] * z.shape[1] == x.shape[0] and z.shape[2] == x.shape[1] and z.stride(-1) == 1
        z_is_3d = True
    weight = weight.contiguous()
    M, N = x.shape
    out = torch.empty_like(x)
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    block_n = min(65536 // x.element_size(), triton.next_power_of_2(N))
    if N > block_n:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    num_warps = min(max(block_n // 256, 1), 8)
    if rows_per_block is None:
        rows_per_block = K.calc_rows_per_block(M, x.device)
    grid = (triton.cdiv(M, rows_per_block), 1)
    pdl_kwargs = {"USE_GDC": True, "launch_pdl": True} if K.is_arch_support_pdl() else {}
    with torch.cuda.device(x.device.index):
        K._layer_norm_fwd_1pass_kernel[grid](
            x, out, weight, None, z, None, rstd, x.stride(0), out.stride(0),
            0 if z_is_3d else z.stride(0), z.stride(0) if z_is_3d else 0, z.stride(1) if z_is_3d else 0,
            M, N, eps, BLOCK_N=block_n, ROWS_PER_BLOCK=rows_per_block, HAS_BIAS=False, HAS_Z=True, Z_IS_3D=z_is_3d,
            Z_HEADS=z.shape[1] if z_is_3d else 1, NORM_BEFORE_GATE=True, IS_RMS_NORM=True, ACTIVATION="swish",
            num_warps=num_warps, **pdl_kwargs,
        )
    return out.reshape(x_shape_og)


# ------------------------------------------------------------------------------ self-checks
# Each helper builds random model-shaped inputs, runs the reference kernels from the
# pristine library copy and the given launcher on independent state copies, and returns
# (bitwise_equal, detail). Launchers use the engine's calling convention (see solution.lean).

def _maxdiff(a, b):
    return (a.float() - b.float()).abs().max().item()


def check_conv_split(launch, device, dtype, lens, slots, has_init, seed, *, normalize=False):
    """launch(x_T, weight, conv_states, query_start(int32), lens, cache_indices(int32), has_initial(bool)) -> q, k, v as [1, T, H, D]."""
    gen = torch.Generator(device=device).manual_seed(seed)
    T = sum(lens)
    S = max(slots) + 1
    qkvz = torch.randn((T, 16384), generator=gen, device=device).to(dtype)
    mixed = qkvz[:, :10240]
    weight = (torch.randn((10240, 4), generator=gen, device=device) * 0.3).to(dtype)
    state0 = torch.randn((S, 10240, 3), generator=gen, device=device).to(dtype)
    cu = [0]
    for n in lens:
        cu.append(cu[-1] + n)
    query_start = torch.tensor(cu, dtype=torch.int32, device=device)
    idx = torch.tensor(slots, dtype=torch.int32, device=device)
    hi = torch.tensor(has_init, dtype=torch.bool, device=device)
    st_ref = state0.clone()
    out = K.causal_conv1d_fn(mixed.T, weight, None, st_ref, query_start, lens, cache_indices=idx, has_initial_state=hi, activation="silu").T
    q1, k1, v1 = K.fused_qkv_split_gdn_prefill(out, 16, 16, 48, 128, 128, 128)
    if normalize:
        q1, k1 = K.l2norm_fwd(q1, 1e-6), K.l2norm_fwd(k1, 1e-6)
    st_new = state0.clone()
    q2, k2, v2 = launch(mixed.T, weight, st_new, query_start, lens, idx, hi)
    torch.cuda.synchronize()
    ok = torch.equal(q1, q2) and torch.equal(k1, k2) and torch.equal(v1, v2) and torch.equal(st_ref, st_new)
    return ok, {"q": _maxdiff(q1, q2), "k": _maxdiff(k1, k2), "v": _maxdiff(v1, v2), "state": _maxdiff(st_ref, st_new), "lens": list(lens), "slots": list(slots)}


def check_rope_kvstore(launch, store, device, dtype, T, q_per_prog, k_per_prog, seed, max_positions=8192):
    """launch(qg, k, v, q_w, k_w, rope, positions(int64), locations(int64), kc_rows, vc_rows, eps, q_per_prog, k_per_prog) -> q, gate."""
    gen = torch.Generator(device=device).manual_seed(seed)
    projected = torch.randn((T, 14336), generator=gen, device=device).to(dtype)
    qg, k, v = projected.split([12288, 1024, 1024], dim=-1)
    q_w = (torch.randn((256,), generator=gen, device=device) * 0.1).to(dtype)
    k_w = (torch.randn((256,), generator=gen, device=device) * 0.1).to(dtype)
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 64, 2, dtype=torch.float32, device=device) / 64))
    freqs = torch.einsum("i,j->ij", torch.arange(max_positions, dtype=torch.float32, device=device), inv_freq)
    rope = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
    positions = torch.randint(0, max_positions, (T,), generator=gen, device=device, dtype=torch.int64)
    # Synthetic paged cache: the reserved page plus enough rows for every token at a distinct random row.
    rows = 64 + max(4 * 1024, ((T + 63) // 64) * 64)
    locations = torch.randperm(rows - 64, generator=gen, device=device)[:T] + 64
    # Validation only: nonzero history exposes writes to untouched/reserved rows.
    # Poison the destinations so omitted stores cannot pass on a zero output,
    # even when the reference store omits the same coordinate.
    kc1 = torch.empty((rows, 4, 256), device=device, dtype=dtype).uniform_(0.5, 1.5, generator=gen)
    vc1 = torch.empty_like(kc1).uniform_(-1.5, -0.5, generator=gen)
    kc1.index_fill_(0, locations, float("nan"))
    vc1.index_fill_(0, locations, float("nan"))
    kc2, vc2 = kc1.clone(), vc1.clone()
    q1, k1, g1 = K.fused_qk_gemma_rmsnorm_rope_gate(qg, k, q_w, k_w, rope, positions, 1e-6, 24, 4, 256, 64, has_gate=True)
    store(k1.view(-1, 1024), v, kc1.view(-1, 1024), vc1.view(-1, 1024), locations, 4, kc1.shape[0], 0)
    q2, g2 = launch(qg, k, v, q_w, k_w, rope, positions, locations, kc2.view(-1, 1024), vc2.view(-1, 1024), 1e-6, q_per_prog, k_per_prog)
    torch.cuda.synchronize()
    ok = torch.equal(q1, q2) and torch.equal(g1, g2) and torch.equal(kc1, kc2) and torch.equal(vc1, vc2)
    return ok, {"q": _maxdiff(q1, q2), "gate": _maxdiff(g1, g2), "kc": _maxdiff(kc1, kc2), "vc": _maxdiff(vc1, vc2), "T": T, "q_per_prog": q_per_prog, "k_per_prog": k_per_prog}


def check_gating_exp(launch, device, dtype, T, seed):
    """launch(A_log, a, b, dt_bias) -> (exp(g), beta), both [1, T, 48] fp32; a/b are strided views of a [T, 96] tensor."""
    gen = torch.Generator(device=device).manual_seed(seed)
    A_log = torch.randn((48,), generator=gen, device=device) * 0.5
    ba = (torch.randn((T, 96), generator=gen, device=device) * 3).to(dtype)
    b, a = ba[:, :48], ba[:, 48:]
    dt_bias = torch.randn((48,), generator=gen, device=device).to(dtype)
    g1, beta1 = K.fused_gdn_gating(A_log, a, b, dt_bias)
    g1e = torch.exp(g1[0].float())
    g2, beta2 = launch(A_log, a, b, dt_bias)
    torch.cuda.synchronize()
    ok = torch.equal(g1e, g2[0]) and torch.equal(beta1, beta2) and g2.shape == g1.shape and beta2.shape == beta1.shape
    return ok, {"g": _maxdiff(g1e, g2[0]), "beta": _maxdiff(beta1, beta2), "T": T}


def check_joint_l2norm(launch, device, dtype, T, seed):
    """launch(q, k, eps) -> (qn, kn) for contiguous [T, 16, 128] q/k."""
    gen = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn((T, 16, 128), generator=gen, device=device).to(dtype)
    k = torch.randn((T, 16, 128), generator=gen, device=device).to(dtype)
    q1, k1 = K.l2norm_fwd(q, 1e-6), K.l2norm_fwd(k, 1e-6)
    q2, k2 = launch(q, k, 1e-6)
    torch.cuda.synchronize()
    ok = torch.equal(q1, q2) and torch.equal(k1, k2)
    return ok, {"q": _maxdiff(q1, q2), "k": _maxdiff(k1, k2), "T": T}


def check_norm_rows(launch, device, dtype, T, rows, seed):
    """launch(x[T*48, 128], weight, z[T, 48, 128] strided view, eps, rows) -> y; reference uses its own heuristic."""
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((T * 48, 128), generator=gen, device=device).to(dtype)
    zfull = torch.randn((T, 16384), generator=gen, device=device).to(dtype)
    z = zfull[:, 10240:].view(T, 48, 128)
    w = (1.0 + 0.1 * torch.randn((128,), generator=gen, device=device)).to(dtype)
    y1 = K.rms_norm_gated(x=x, weight=w, bias=None, z=z, eps=1e-6, norm_before_gate=True, is_rms_norm=True, activation="swish")
    y2 = launch(x, w, z, 1e-6, rows)
    torch.cuda.synchronize()
    ok = torch.equal(y1, y2)
    return ok, {"y": _maxdiff(y1, y2), "T": T, "rows": rows, "reference_rows": K.calc_rows_per_block(T * 48, x.device)}


def check_dense(linear, weight, M, seed):
    """Compare linear(x, weight) with torch.nn.functional.linear on random bf16 activations."""
    gen = torch.Generator(device=weight.device).manual_seed(seed)
    x = torch.randn((M, weight.shape[1]), generator=gen, device=weight.device).to(weight.dtype)
    y1 = torch.nn.functional.linear(x, weight)
    y2 = linear(x, weight)
    torch.cuda.synchronize()
    ok = torch.equal(y1, y2)
    return ok, {"y": _maxdiff(y1, y2), "M": M, "N": weight.shape[0], "K": weight.shape[1]}
