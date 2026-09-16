"""GDN prefill convolution with a sub-tile loop, program-resident weights and a packed worklist (Lane 3 turn 4).

Source: Lane 3's runtime-validated turn-4 package (artifacts/lane-3/turn4; source.tar.gz sha256
b256814448996ebbbae75e042248e7f380c9e4b200c7d5497c84598d71cb9a19). Everything between this docstring and
the "Lane 1 additions" marker below is a byte-identical copy of their ``solution/conv_triton4.py``
(sha256 95a2302f9f144a9b731fd43c613f64e1e861604a3295c83f4af68afccb402dab) minus its module docstring,
which read:

    The retained `_causal_conv1d_split_tile_kernel` (turn 2) computes one [32, 128] tile per
    program. Its offline SASS shows three costs that are not arithmetic: the four [128]
    weight vectors are broadcast to the tile layout through shared memory (25 barriers and
    about 40 shared loads per program), every element of the four shifted input tiles passes
    through a masked select even for interior tiles, and 121 registers cap residency at four
    programs per SM. This kernel keeps every per-element operation, operand type and order
    (bf16 products, fp32 accumulation, `acc / (1 + exp(-acc))`, bf16 rounding, and the fused
    Q/K L2 normalization with the same [2 x 16 lanes, 8 elements] row layout) and changes only
    the work distribution:

    * a program owns BLOCK_T consecutive tokens of one sequence for one 128-feature block and
      walks them in SUB-row sub-tiles (a `tl.range` loop bounds the live registers);
    * the weights are loaded once per program directly in the sub-tile layout (row-invariant
      [SUB, BLOCK_N] loads from the contiguous transposed weight [4, dim]), so no layout
      conversion through shared memory is needed;
    * complete interior sub-tiles use unmasked loads/stores; only the first sub-tile of a
      sequence (conv-state history) and a ragged tail are masked;
    * the grid enumerates only (sequence, chunk) pairs that contain tokens, from a small
      int32 worklist built from the call's own sequence lengths once per forward.

    The conv-state update is the upstream code, executed by the chunk-0 program of each
    sequence after it consumed the previous state. Outputs are the token-major q/k/v tensors
    of `causal_conv1d_split_tile_fn`.

Integration (solution.engine option ``conv_split="loop"``): the engine prepares the contiguous
transposed conv weights once per engine, chooses the tokens per program from the call's lengths
(``choose_block_t``, Lane 3's validated heuristic) and stages the packed worklist through the pinned
metadata ring together with the other prefill metadata (``worklist_items``), then launches the kernel
through ``solution.lean.conv_loop`` (cached direct launch) for every GDN layer of a prefill. The
kernel is enabled only after ``Engine._check_fused`` reproduced the reference conv + split + L2 norm
kernels bitwise, including the complete conv-state tensors (flag ``conv_loop``).
"""

from typing import List, Optional, Union

import torch
import triton
import triton.language as tl

PAD_SLOT_ID = -1


@triton.jit
def _rows_history(x_base, stride_x_token, conv_states_base, stride_state_tok, tok, tok_ok,
                  load_init_state, SHIFT: tl.constexpr, STATE_LEN: tl.constexpr):
    """Tap SHIFT for the first sub-tile of a sequence: rows before token 0 come from the state."""
    src = tok - SHIFT
    in_x = (src >= 0) & tok_ok
    vals = tl.load(x_base[None, :] + src.to(tl.int64)[:, None] * stride_x_token, mask=in_x[:, None], other=0.0)
    if load_init_state:
        from_state = (src < 0) & tok_ok
        state_vals = tl.load(conv_states_base[None, :] + (STATE_LEN + src)[:, None] * stride_state_tok,
                             mask=from_state[:, None], other=0.0)
        vals = tl.where(from_state[:, None], state_vals, vals)
    return vals


@triton.jit
def _conv_subtile(x3, x2, x1, x0, w0, w1, w2, w3, bias, eps,
                  HAS_BIAS: tl.constexpr, SILU: tl.constexpr, L2NORM: tl.constexpr,
                  SUB: tl.constexpr, BLOCK_N: tl.constexpr):
    """Per-element arithmetic of the upstream conv kernel followed by the fused L2 norm."""
    if HAS_BIAS:
        acc = bias
    else:
        acc = tl.zeros((SUB, BLOCK_N), dtype=tl.float32)
    # Same operand types and order as upstream: bf16 products, fp32 accumulation.
    acc += x3 * w0
    acc += x2 * w1
    acc += x1 * w2
    acc += x0 * w3
    if SILU:
        acc = acc / (1 + tl.exp(-acc))
    y_bf16 = acc.to(tl.bfloat16)
    if L2NORM:
        b_x = y_bf16.to(tl.float32)
        b_var = tl.sum(b_x * b_x, axis=1)
        b_y = b_x / tl.sqrt(b_var + eps)[:, None]
        return b_y.to(tl.bfloat16)
    else:
        return y_bf16


@triton.jit
def _load_weights(w_tile, bias_tile, stride_w_width: tl.constexpr, HAS_BIAS: tl.constexpr):
    """Taps (and bias) as row-invariant [SUB, BLOCK_N] tiles, i.e. already in the sub-tile layout."""
    w0 = tl.load(w_tile + 0 * stride_w_width)
    w1 = tl.load(w_tile + 1 * stride_w_width)
    w2 = tl.load(w_tile + 2 * stride_w_width)
    w3 = tl.load(w_tile + 3 * stride_w_width)
    if HAS_BIAS:
        bias = tl.load(bias_tile).to(tl.float32)
    else:
        bias = 0.0
    return w0, w1, w2, w3, bias


@triton.jit
def _conv_program(x_ptr, x_base, conv_states_base, o_base, w_tile, bias_tile, eps,
                  sequence_start_index, seqlen, token_offset, chunk, load_init_state,
                  stride_x_token: tl.constexpr, stride_state_tok: tl.constexpr, stride_w_width: tl.constexpr,
                  o_stride: tl.constexpr,
                  HAS_BIAS: tl.constexpr, SILU: tl.constexpr, L2NORM: tl.constexpr, STATE_LEN: tl.constexpr,
                  BLOCK_T: tl.constexpr, SUB: tl.constexpr, BLOCK_N: tl.constexpr,
                  UNROLL: tl.constexpr, STAGES: tl.constexpr):
    rows = tl.arange(0, SUB)
    n_valid = tl.minimum(seqlen - token_offset, BLOCK_T)
    n_full = n_valid // SUB
    # Loop-invariant [SUB, BLOCK_N] offsets of the four taps and of the output rows.
    off_x0 = (rows * stride_x_token)[:, None]
    off_x1 = ((rows - 1) * stride_x_token)[:, None]
    off_x2 = ((rows - 2) * stride_x_token)[:, None]
    off_x3 = ((rows - 3) * stride_x_token)[:, None]
    off_o = (rows * o_stride)[:, None]

    if chunk == 0:
        # First sub-tile: history taps come from the conv state (or zeros), masked to seqlen.
        # The strided state tile has its own register layout; Triton resolves the mixed
        # layouts with shared-memory conversions. Loading the weights again inside this
        # branch keeps those conversions inside the chunk-0 programs (interior programs
        # and the loop below use the direct sub-tile layout without any conversion).
        wh0, wh1, wh2, wh3, biash = _load_weights(w_tile, bias_tile, stride_w_width, HAS_BIAS)
        tok = rows
        tok_ok = tok < n_valid
        x3 = _rows_history(x_base, stride_x_token, conv_states_base, stride_state_tok, tok, tok_ok, load_init_state, 3, STATE_LEN)
        x2 = _rows_history(x_base, stride_x_token, conv_states_base, stride_state_tok, tok, tok_ok, load_init_state, 2, STATE_LEN)
        x1 = _rows_history(x_base, stride_x_token, conv_states_base, stride_state_tok, tok, tok_ok, load_init_state, 1, STATE_LEN)
        x0 = tl.load(x_base[None, :] + tok.to(tl.int64)[:, None] * stride_x_token, mask=tok_ok[:, None], other=0.0)
        y = _conv_subtile(x3, x2, x1, x0, wh0, wh1, wh2, wh3, biash, eps, HAS_BIAS, SILU, L2NORM, SUB, BLOCK_N)
        o_rows = (sequence_start_index + tok).to(tl.int64) * o_stride
        o_rows = tl.multiple_of(o_rows, BLOCK_N)
        tl.store(o_base[None, :] + o_rows[:, None], y, mask=tok_ok[:, None])
        s_begin = 1
    else:
        s_begin = 0

    w0, w1, w2, w3, bias = _load_weights(w_tile, bias_tile, stride_w_width, HAS_BIAS)
    # Complete interior sub-tiles: no masks, taps lie inside the sequence.
    for s in tl.range(s_begin, n_full, loop_unroll_factor=UNROLL, num_stages=STAGES):
        # rel0: first row of the sub-tile relative to the sequence start (x_base already
        # points at the sequence); stores use the absolute token row.
        rel0 = token_offset + s * SUB
        xp = x_base[None, :] + rel0.to(tl.int64) * stride_x_token
        x3 = tl.load(xp + off_x3)
        x2 = tl.load(xp + off_x2)
        x1 = tl.load(xp + off_x1)
        x0 = tl.load(xp + off_x0)
        y = _conv_subtile(x3, x2, x1, x0, w0, w1, w2, w3, bias, eps, HAS_BIAS, SILU, L2NORM, SUB, BLOCK_N)
        op = o_base[None, :] + tl.multiple_of((sequence_start_index + rel0).to(tl.int64) * o_stride, BLOCK_N)
        tl.store(op + off_o, y)

    # Ragged tail (fewer than SUB remaining rows), never the history sub-tile.
    tail0 = n_full * SUB
    if (tail0 < n_valid) & ((chunk != 0) | (n_full >= 1)):
        tok_ok = rows < (n_valid - tail0)
        tail_mask = tok_ok[:, None]
        rel0 = token_offset + tail0
        xp = x_base[None, :] + rel0.to(tl.int64) * stride_x_token
        x3 = tl.load(xp + off_x3, mask=tail_mask, other=0.0)
        x2 = tl.load(xp + off_x2, mask=tail_mask, other=0.0)
        x1 = tl.load(xp + off_x1, mask=tail_mask, other=0.0)
        x0 = tl.load(xp + off_x0, mask=tail_mask, other=0.0)
        y = _conv_subtile(x3, x2, x1, x0, w0, w1, w2, w3, bias, eps, HAS_BIAS, SILU, L2NORM, SUB, BLOCK_N)
        op = o_base[None, :] + tl.multiple_of((sequence_start_index + rel0).to(tl.int64) * o_stride, BLOCK_N)
        tl.store(op + off_o, y, mask=tail_mask)


@triton.jit()
def _causal_conv1d_split_loop_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    conv_states_ptr,
    conv_state_indices_ptr,
    has_initial_states_ptr,
    query_start_loc_ptr,
    work_ptr,
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
    SUB: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FEAT_FIRST: tl.constexpr,
    FUSE_L2NORM: tl.constexpr,
    USE_WORKLIST: tl.constexpr,
    UNROLL: tl.constexpr,
    STAGES: tl.constexpr,
):
    tl.static_assert(KERNEL_WIDTH == 4, "this model uses a width-4 convolution")
    tl.static_assert(SUB >= KERNEL_WIDTH - 1, "the history sub-tile must cover the state taps")
    tl.static_assert(BLOCK_T % SUB == 0, "sub-tiles must tile the program's tokens")
    tl.static_assert(Q_DIM == K_DIM, "q and k output strides are shared by the L2-normalized branch")
    state_len: tl.constexpr = KERNEL_WIDTH - 1

    if USE_WORKLIST:
        # Packed (sequence << 16 | chunk) items; only chunks that contain tokens are listed.
        if FEAT_FIRST:
            feat_block = tl.program_id(0) * BLOCK_N
            item = tl.program_id(1)
        else:
            item = tl.program_id(0)
            feat_block = tl.program_id(1) * BLOCK_N
        packed = tl.load(work_ptr + item)
        idx_seq = packed >> 16
        chunk = packed & 0xFFFF
    else:
        if FEAT_FIRST:
            feat_block = tl.program_id(0) * BLOCK_N
            chunk = tl.program_id(1)
            idx_seq = tl.program_id(2)
        else:
            idx_seq = tl.program_id(0)
            chunk = tl.program_id(1)
            feat_block = tl.program_id(2) * BLOCK_N
    idx_feats = feat_block + tl.arange(0, BLOCK_N)

    sequence_start_index = tl.load(query_start_loc_ptr + idx_seq)
    sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1)
    seqlen = sequence_end_index - sequence_start_index
    token_offset = chunk * BLOCK_T
    if token_offset >= seqlen:
        return
    conv_state_batch_coord = tl.load(conv_state_indices_ptr + idx_seq).to(tl.int64)
    if conv_state_batch_coord == pad_slot_id:
        return
    load_init_state = tl.load(has_initial_states_ptr + idx_seq).to(tl.int1)

    # [BLOCK_N] pointers to token 0 of this sequence, its conv state row and its weights.
    x_base = x_ptr + sequence_start_index.to(tl.int64) * stride_x_token + idx_feats * stride_x_dim
    conv_states_base = (
        conv_states_ptr
        + (conv_state_batch_coord * stride_state_seq)
        + (idx_feats * stride_state_dim)
    )
    # Weight (and bias) pointer tiles: row-invariant [SUB, BLOCK_N], loaded in _conv_program.
    zero_rows = tl.zeros((SUB, 1), dtype=tl.int32)
    w_tile = w_ptr + (idx_feats * stride_w_dim)[None, :] + zero_rows
    if HAS_BIAS:
        bias_tile = bias_ptr + idx_feats[None, :] + zero_rows
    else:
        bias_tile = w_tile

    if feat_block < Q_DIM + K_DIM:
        if feat_block < Q_DIM:
            o_base = q_ptr + idx_feats
        else:
            o_base = k_ptr + (idx_feats - Q_DIM)
        _conv_program(x_ptr, x_base, conv_states_base, o_base, w_tile, bias_tile, eps,
                      sequence_start_index, seqlen, token_offset, chunk, load_init_state,
                      stride_x_token, stride_state_tok, stride_w_width, Q_DIM,
                      HAS_BIAS, SILU_ACTIVATION, FUSE_L2NORM, state_len, BLOCK_T, SUB, BLOCK_N, UNROLL, STAGES)
    else:
        o_base = v_ptr + (idx_feats - Q_DIM - K_DIM)
        _conv_program(x_ptr, x_base, conv_states_base, o_base, w_tile, bias_tile, eps,
                      sequence_start_index, seqlen, token_offset, chunk, load_init_state,
                      stride_x_token, stride_state_tok, stride_w_width, V_DIM,
                      HAS_BIAS, SILU_ACTIVATION, False, state_len, BLOCK_T, SUB, BLOCK_N, UNROLL, STAGES)

    if chunk == 0:
        # Conv-state update (upstream code), after this program consumed the previous state.
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


def build_worklist(seq_lens_cpu: List[int], block_t: int, device) -> torch.Tensor:
    """Packed (sequence << 16 | chunk) items in chunk-major order, from this call's lengths.

    Chunk-major order keeps the retained sequence-first program order (all sequences' chunk
    c before chunk c+1) while skipping chunks beyond a sequence's length. Built once per
    forward from the current inputs and shared by the 48 GDN layers.
    """
    if len(seq_lens_cpu) >= 1 << 16:
        raise ValueError("worklist packs the sequence index into 16 bits")
    max_chunks = -(-max(seq_lens_cpu) // block_t)
    if max_chunks >= 1 << 16:
        raise ValueError("worklist packs the chunk index into 16 bits")
    items = [(s << 16) | c for c in range(max_chunks) for s, n in enumerate(seq_lens_cpu) if c * block_t < n]
    return torch.tensor(items, dtype=torch.int32, device=device)


def causal_conv1d_split_loop_fn(
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
    feat_first: bool = False,
    fuse_l2norm: bool = True,
    eps: float = 1e-6,
    sub_t: int = 8,
    worklist: Optional[torch.Tensor] = None,
    use_worklist: bool = True,
    unroll: int = 1,
    stages: int = 1,
    weight_t: Optional[torch.Tensor] = None,
):
    """Drop-in for `causal_conv1d_split_tile_fn` (same inputs, outputs and cache contract).

    `worklist` may be prebuilt once per forward with `build_worklist(seq_lens_cpu, block_t)`;
    otherwise it is built here. With use_worklist=False the retained 3D grid is launched
    (programs beyond a sequence's length return immediately), which isolates the worklist.
    `weight_t` is the contiguous transposed weight [width, dim] (features contiguous), so the
    kernel loads the four taps as 16-byte vectors in the sub-tile layout; it is derived from
    `weight` here when not supplied (callers cache it per model, see conv_trials4).
    """
    dim, total_tokens = x.shape
    _, width = weight.shape
    if width != 4:
        raise ValueError("Fused conv/split supports the width-4 convolution only")
    if weight_t is None:
        weight_t = weight.t().contiguous()
    if weight_t.shape != (width, dim) or weight_t.stride(1) != 1 or weight_t.dtype != weight.dtype:
        raise ValueError("weight_t must be the contiguous [width, dim] transpose of weight")
    if x.stride(0) != 1:
        raise ValueError("x must be channel-last: feature stride 1")
    q_dim, k_dim, v_dim = num_q_heads * head_dim, num_k_heads * head_dim, num_v_heads * head_dim
    if q_dim + k_dim + v_dim != dim or q_dim != k_dim:
        raise ValueError("q/k/v head layout does not cover the packed feature dimension")
    if q_dim % block_n or k_dim % block_n or v_dim % block_n:
        raise ValueError("q/k/v feature ranges must be multiples of the feature block")
    if block_t % sub_t or sub_t < width - 1 or block_t & (block_t - 1) or block_n & (block_n - 1) or sub_t & (sub_t - 1):
        raise ValueError("block_t/sub_t/block_n must be powers of two with sub_t | block_t")
    if fuse_l2norm and block_n != head_dim:
        raise ValueError("fused Q/K L2 normalization needs one head per feature block")
    if conv_states.shape[1] != dim or conv_states.shape[2] < width - 1:
        raise ValueError("conv state shape does not match the packed features")
    q = torch.empty((1, total_tokens, num_q_heads, head_dim), dtype=x.dtype, device=x.device)
    k = torch.empty((1, total_tokens, num_k_heads, head_dim), dtype=x.dtype, device=x.device)
    v = torch.empty((1, total_tokens, num_v_heads, head_dim), dtype=x.dtype, device=x.device)
    num_seqs, num_chunks, num_feat_blocks = len(seq_lens_cpu), triton.cdiv(max(seq_lens_cpu), block_t), triton.cdiv(dim, block_n)
    if use_worklist:
        if worklist is None:
            worklist = build_worklist(seq_lens_cpu, block_t, x.device)
        grid = (num_feat_blocks, worklist.numel()) if feat_first else (worklist.numel(), num_feat_blocks)
    else:
        worklist = query_start_loc  # unused placeholder pointer
        grid = (num_feat_blocks, num_chunks, num_seqs) if feat_first else (num_seqs, num_chunks, num_feat_blocks)
    _causal_conv1d_split_loop_kernel[grid](
        x,
        weight_t,
        bias,
        conv_states,
        cache_indices,
        has_initial_state,
        query_start_loc,
        worklist,
        q,
        k,
        v,
        eps,
        dim,
        conv_states.shape[0],
        x.stride(0),
        x.stride(1),
        weight_t.stride(1),
        weight_t.stride(0),
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
        SUB=sub_t,
        BLOCK_N=block_n,
        FEAT_FIRST=feat_first,
        FUSE_L2NORM=fuse_l2norm,
        USE_WORKLIST=use_worklist,
        UNROLL=unroll,
        STAGES=stages,
        num_warps=num_warps,
    )
    return q, k, v


# ----------------------------------------------------------------------------- Lane 1 additions
# ``choose_block_t`` is copied from Lane 3's ``solution/conv_trials4.py`` (same package, sha256
# 3cdd1c82624cc6488ad050d6453f887362af38df970f9e39bc9413525ddaaaa2); ``worklist_items`` is the numpy form of
# ``build_worklist`` above (same items in the same chunk-major order) so the engine can stage the worklist
# through its pinned ring instead of a synchronous pageable copy per forward.
import numpy as np

FEATURE_BLOCKS = 80        # 10240 / 128
CONV_SUB_T = 8
CONV_NUM_WARPS = 4


def choose_block_t(seq_lens_cpu):
    """Tokens per program from this call's lengths (Lane 3 turn-4 component timings, B200).

    128 wins from about 2000 programs per layer (1.17-1.30x hot at 8k-32k tokens), 64 in
    between (1.22x at 1792 tokens over three sequences), 32 for small calls (a single
    256-token sequence: bt32 1.13x, bt64 0.86x, bt128 0.63x versus the retained kernel).
    """
    def programs(bt):
        return FEATURE_BLOCKS * sum(-(-n // bt) for n in seq_lens_cpu)
    if programs(128) >= 2048:
        return 128
    if programs(64) >= 1024:
        return 64
    return 32


def worklist_items(seq_lens_cpu, block_t):
    """int32 numpy array equal to ``build_worklist(seq_lens_cpu, block_t, device).cpu().numpy()``."""
    lens = np.asarray(seq_lens_cpu, dtype=np.int64)
    if lens.size >= 1 << 16:
        raise ValueError("worklist packs the sequence index into 16 bits")
    chunks = -(-lens // block_t)
    max_chunks = int(chunks.max()) if lens.size else 0
    if max_chunks >= 1 << 16:
        raise ValueError("worklist packs the chunk index into 16 bits")
    c = np.arange(max_chunks, dtype=np.int64)[:, None]
    s = np.arange(lens.size, dtype=np.int64)[None, :]
    return ((s << 16) | c)[c < chunks[None, :]].astype(np.int32)
