"""Lean launch paths for the hottest Triton kernels of the forward pass.

The upstream wrappers (copied in solution.qwen38_optimized) validate arguments,
derive strides/constexprs from shapes and go through Triton's JIT dispatch on
every call; for a decode step that is 176 launches. Here each call site keeps a
cache keyed by everything Triton specialises on (integer argument values,
pointer 16-byte alignment, dtypes, constexpr values) and, on a hit, launches
the already compiled kernel directly through CompiledKernel.run. The first call
of a key goes through the normal JITFunction.run (compiling if needed).

The launched kernel, its arguments and its launch options are identical to the
wrapper's, so the arithmetic is unchanged. Keys are finer than Triton's own
specialisation (exact integer values instead of ==1 / %16 classes), which is
safe: at worst a kernel is cached twice.
"""
from __future__ import annotations

import torch
import triton

from . import qwen38_optimized as K
from . import lane3_turn3 as T3


def _align(*tensors):
    return tuple(t.data_ptr() & 15 for t in tensors)


class _Site:
    """One kernel call site: (key -> compiled kernel handles)."""

    __slots__ = ("fn", "cache", "misses")

    def __init__(self, fn):
        self.fn = fn
        self.cache = {}
        self.misses = 0

    def launch(self, key, grid, values, options, stream=None):
        entry = self.cache.get(key)
        if entry is None:
            self.misses += 1
            kernel = self.fn.run(*values, grid=grid, warmup=False, **options)
            # Accessing .run initialises the module handles; cache the bound callables.
            self.cache[key] = (kernel.run, kernel.function, kernel.packed_metadata)
            return
        run, function, packed_metadata = entry
        if stream is None:
            # Query per launch unless the caller passes the stream handle it already queried
            # for the whole forward call (the stream cannot change inside one call).
            stream = torch.cuda.current_stream().cuda_stream
        run(grid[0], grid[1] if len(grid) > 1 else 1, grid[2] if len(grid) > 2 else 1, stream, function, packed_metadata,
            None, None, None, *values)


_ROPE = _Site(K._fused_qk_rmsnorm_rope_gate_kernel)
_SIGMOID = _Site(K._fused_sigmoid_mul_kernel)
_CONV_UPDATE = _Site(K._fused_qkvzba_causal_conv1d_update_contiguous_kernel)
_RECURRENT = _Site(K.fused_recurrent_gated_delta_rule_packed_decode_kernel)
_LAYER_NORM = _Site(K._layer_norm_fwd_1pass_kernel)

SITES = {"rope_gate": _ROPE, "sigmoid_mul": _SIGMOID, "conv_update": _CONV_UPDATE, "recurrent_decode": _RECURRENT, "rms_norm_gated": _LAYER_NORM}


def rope_gate(q_gate, k, q_weight, k_weight, cos_sin_cache, positions, eps, stream=None):
    """fused_qk_gemma_rmsnorm_rope_gate(..., 24, 4, 256, 64, has_gate=True) for this model."""
    T = q_gate.shape[0]
    q_out = torch.empty(T, 6144, dtype=q_gate.dtype, device=q_gate.device)
    k_out = torch.empty(T, 1024, dtype=k.dtype, device=k.device)
    gate_out = torch.empty(T, 24, 256, dtype=q_gate.dtype, device=q_gate.device)
    sq, sk, sp, sc = q_gate.stride(0), k.stride(0), positions.stride(0), cos_sin_cache.stride(0)
    key = (T, sq, sk, sp, sc, q_gate.dtype, positions.dtype, eps, _align(q_gate, k, positions, q_weight, k_weight, cos_sin_cache))
    values = (q_gate, k, q_out, k_out, gate_out, q_weight, k_weight, cos_sin_cache, positions, None,
              sq, sk, q_out.stride(0), k_out.stride(0), gate_out.stride(0), sc, sp,
              24, 4, 256, 64, 32, 256, 32, eps, q_gate.dtype == torch.float16, True, True, False, K._ENABLE_PDL)
    _ROPE.launch(key, (T, 28), values, {}, stream)
    return q_out, k_out, gate_out


def sigmoid_mul(attn_output, gate, stream=None):
    """fused_sigmoid_mul(attn_output, gate, inplace=True) for 2D [T, 6144] tensors."""
    T, hidden_dim = attn_output.shape
    block_h = 1024 if T < 1024 else 2048
    key = (T, hidden_dim, block_h, attn_output.dtype, gate.dtype, _align(attn_output, gate))
    values = (attn_output, attn_output, gate, hidden_dim, hidden_dim, hidden_dim, hidden_dim, block_h)
    _SIGMOID.launch(key, (T, triton.cdiv(hidden_dim, block_h)), values, {"num_warps": 4}, stream)
    return attn_output


def conv_update(mixed_qkvz, mixed_ba, conv_state, conv_weight, conv_state_indices, stream=None):
    """fused_qkvzba_causal_conv1d_update_contiguous(..., qkv_dim=10240, v_dim=6144, num_v_heads=48, head_v_dim=128, activation='silu')."""
    batch = mixed_qkvz.shape[0]
    mixed_qkv = torch.empty((batch, 10240), dtype=mixed_qkvz.dtype, device=mixed_qkvz.device)
    z = torch.empty((batch, 48, 128), dtype=mixed_qkvz.dtype, device=mixed_qkvz.device)
    b = torch.empty((batch, 48), dtype=mixed_ba.dtype, device=mixed_ba.device)
    a = torch.empty_like(b)
    strides = (mixed_qkvz.stride(0), mixed_qkvz.stride(1), mixed_ba.stride(0), mixed_ba.stride(1), conv_state.stride(0), conv_state.stride(1),
               conv_state.stride(2), conv_weight.stride(0), conv_weight.stride(1), conv_state_indices.stride(0))
    slots, state_len, width = conv_state.shape[0], conv_state.shape[2], conv_weight.shape[1]
    key = (batch, slots, state_len, width, strides, mixed_qkvz.dtype, mixed_ba.dtype, conv_state.dtype, conv_weight.dtype, conv_state_indices.dtype,
           _align(mixed_qkvz, mixed_ba, conv_state, conv_weight, conv_state_indices))
    values = (mixed_qkv, z, b, a, mixed_qkvz, mixed_ba, conv_state, conv_weight, None, conv_state_indices, *strides,
              10240, 6144, 48, slots, state_len, width, False, True, -1, 256)
    _CONV_UPDATE.launch(key, (batch, 40), values, {"num_warps": 8, "num_stages": 2}, stream)
    return mixed_qkv, z, b, a


def recurrent_decode(mixed_qkv, a, b, A_log, dt_bias, state, out, ssm_state_indices, stream=None, bv=32):
    """fused_recurrent_gated_delta_rule_packed_decode(..., scale=128**-0.5, use_qk_l2norm_in_kernel=True) for H=16, HV=48, K=V=128.

    ``bv`` is the number of FP32 state rows per program: 32 is the wrapper's fixed tile (grid (4, B * 48)); 16, 8 and 4
    launch the identical kernel function with grid (128 // bv, B * 48), still one warp per program (Lane 3's turn-12
    retile, EngineOptions.decode_bv; enabled by the engine only after its bitwise self-check).
    """
    if bv not in (32, 16, 8, 4):
        raise RuntimeError("lean recurrent_decode supports 32, 16, 8 or 4 state rows per program")
    B = mixed_qkv.shape[0]
    strides = (mixed_qkv.stride(0), a.stride(0), b.stride(0), state.stride(0), state.stride(0), ssm_state_indices.stride(0))
    key = (B, bv, strides, mixed_qkv.dtype, a.dtype, b.dtype, A_log.dtype, dt_bias.dtype, state.dtype, out.dtype, ssm_state_indices.dtype,
           _align(mixed_qkv, a, b, A_log, dt_bias, out, state, ssm_state_indices))
    values = (mixed_qkv, a, b, A_log, dt_bias, out, state, state, ssm_state_indices, 128 ** -0.5, *strides,
              16, 48, 128, 128, 128, bv, 20.0, True)
    _RECURRENT.launch(key, (128 // bv, B * 48), values, {"num_warps": 1, "num_stages": 3}, stream)
    return out, state


def rms_norm_gated(x, weight, z, eps, stream=None, rows_per_block=None):
    """rms_norm_gated(x=x, weight=weight, bias=None, z=z, eps=eps, norm_before_gate=True, is_rms_norm=True, activation='swish').

    ``rows_per_block`` overrides the reference heuristic (prefill uses 8 where it picks 4); it is part of the key.
    """
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
        z_is_3d = True
    M, N = x.shape
    out = torch.empty_like(x)
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    block_n = min(65536 // x.element_size(), triton.next_power_of_2(N))
    num_warps = min(max(block_n // 256, 1), 8)
    rows_per_block = K.calc_rows_per_block(M, x.device) if rows_per_block is None else rows_per_block
    pdl = K.is_arch_support_pdl()
    stride_z_row = 0 if z_is_3d else z.stride(0)
    stride_z_token = z.stride(0) if z_is_3d else 0
    stride_z_head = z.stride(1) if z_is_3d else 0
    z_heads = z.shape[1] if z_is_3d else 1
    key = (M, N, x.stride(0), out.stride(0), stride_z_row, stride_z_token, stride_z_head, z_heads, eps, x.dtype, z.dtype, weight.dtype, rows_per_block, pdl,
           _align(x, out, weight, z))
    values = (x, out, weight, None, z, None, rstd, x.stride(0), out.stride(0), stride_z_row, stride_z_token, stride_z_head, M, N, eps,
              block_n, rows_per_block, False, True, z_is_3d, z_heads, True, True, "swish", pdl)
    options = {"num_warps": num_warps}
    if pdl:
        options["launch_pdl"] = True
    _LAYER_NORM.launch(key, (triton.cdiv(M, rows_per_block), 1), values, options, stream)
    return out.reshape(x_shape_og)


# ----------------------------------------------------------------------------- prefill-only kernels
_QKV_SPLIT = _Site(K.fused_qkv_split_gdn_prefill_kernel)
_GATING = _Site(K.fused_gdn_gating_kernel)
_L2NORM = _Site(K.l2norm_fwd_kernel)
_CONV1D = _Site(K._causal_conv1d_fwd_kernel)
SITES.update({"qkv_split": _QKV_SPLIT, "gdn_gating": _GATING, "l2norm": _L2NORM, "causal_conv1d": _CONV1D})


def qkv_split(mixed_qkv, stream=None):
    """fused_qkv_split_gdn_prefill(mixed_qkv, 16, 16, 48, 128, 128, 128)."""
    T = mixed_qkv.shape[0]
    q = torch.empty((1, T, 16, 128), dtype=mixed_qkv.dtype, device=mixed_qkv.device)
    k = torch.empty((1, T, 16, 128), dtype=mixed_qkv.dtype, device=mixed_qkv.device)
    v = torch.empty((1, T, 48, 128), dtype=mixed_qkv.dtype, device=mixed_qkv.device)
    st, sd = mixed_qkv.stride(0), mixed_qkv.stride(1)
    key = (T, st, sd, mixed_qkv.dtype, _align(mixed_qkv))
    values = (q, k, v, mixed_qkv, st, sd, 16, 16, 48, 128, 128, 128, 16384)
    _QKV_SPLIT.launch(key, (T,), values, {"num_warps": 8, "num_stages": 3}, stream)
    return q, k, v


def gdn_gating(A_log, a, b, dt_bias, stream=None):
    """fused_gdn_gating(A_log, a, b, dt_bias) with beta=1.0, threshold=20.0."""
    batch, num_heads = a.shape
    sa, sb = a.stride(0), b.stride(0)
    g = torch.empty(1, batch, num_heads, dtype=torch.float32, device=a.device)
    beta_output = torch.empty(1, batch, num_heads, dtype=torch.float32, device=b.device)
    key = (batch, num_heads, sa, sb, a.dtype, b.dtype, A_log.dtype, dt_bias.dtype, _align(A_log, a, b, dt_bias))
    values = (g, beta_output, A_log, a, b, dt_bias, 1, sa, sb, num_heads, 1.0, 20.0, 8)
    _GATING.launch(key, (batch, 1, triton.cdiv(num_heads, 8)), values, {"num_warps": 1}, stream)
    return g, beta_output


def l2norm(x, eps=1e-6, stream=None):
    """l2norm_fwd(x, eps) for contiguous [..., D<=512] inputs (T is not specialised by Triton)."""
    x_shape_og = x.shape
    x = x.view(-1, x.shape[-1])
    y = torch.empty_like(x)
    T, D = x.shape
    BD = min(65536 // x.element_size(), triton.next_power_of_2(D))
    if D > 512:
        raise RuntimeError("lean l2norm supports D <= 512 only")
    key = (D, BD, eps, x.dtype, x.stride(0), y.stride(0), _align(x, y))
    values = (x, y, eps, T, D, 16, BD)
    _L2NORM.launch(key, (triton.cdiv(T, 16),), values, {"num_warps": 8, "num_stages": 3}, stream)
    return y.view(x_shape_og)


def causal_conv1d(x, weight, conv_states, query_start_loc, seq_lens, cache_indices, has_initial_state, stream=None):
    """causal_conv1d_fn(x, weight, None, conv_states, query_start_loc, seq_lens, cache_indices=..., has_initial_state=..., activation='silu')."""
    out = torch.empty_like(x)
    dim, cu_seqlen = x.shape
    width = weight.shape[1]
    np2_statelen = triton.next_power_of_2(width - 1)
    num_cache_lines = conv_states.shape[0]
    strides = (0, x.stride(0), x.stride(1), weight.stride(0), weight.stride(1), conv_states.stride(0), conv_states.stride(1), conv_states.stride(2),
               0, out.stride(0), out.stride(1))
    batch = len(seq_lens)
    max_seq_len = max(seq_lens)
    key = (dim, cu_seqlen, width, num_cache_lines, strides, x.dtype, weight.dtype, conv_states.dtype, cache_indices.dtype, has_initial_state.dtype,
           query_start_loc.dtype, _align(x, weight, conv_states, cache_indices, has_initial_state, query_start_loc))
    values = (x, weight, None, conv_states, cache_indices, has_initial_state, query_start_loc, out, dim, cu_seqlen, num_cache_lines, *strides,
              K.PAD_SLOT_ID, False, width, True, True, True, True, True, np2_statelen, 8, 256)
    grid = (batch, (max_seq_len + 7) // 8, triton.cdiv(dim, 256))
    _CONV1D.launch(key, grid, values, {"num_stages": 2}, stream)
    return out


# ----------------------------------------------------------------------------- lane-3 fused kernels
# Bitwise-neutral fusions from Lane 3's validated package (solution.lane3_kernels). The engine
# enables each one only after a self-check against the reference kernels (Engine._bind_direct).
from . import lane3_kernels as K3
from . import lane3_conv_tile as KT

_ROPE_KVSTORE = _Site(K3._fused_qk_rmsnorm_rope_gate_kvstore_kernel)
_CONV_SPLIT = _Site(K3._causal_conv1d_split_fwd_kernel)
_GATING_BLOCKED = _Site(K3.fused_gdn_gating_blocked_kernel)
_JOINT_L2NORM = _Site(K3._joint_l2norm)
_CONV_TILE = _Site(KT._causal_conv1d_split_tile_kernel)
SITES.update({"rope_kvstore": _ROPE_KVSTORE, "conv_split": _CONV_SPLIT, "gating_blocked": _GATING_BLOCKED, "joint_l2norm": _JOINT_L2NORM,
              "conv_tile": _CONV_TILE})


def rope_kvstore(q_gate, k, v, q_weight, k_weight, cos_sin_cache, positions, locations, k_rows, v_rows, eps, q_per_prog, k_per_prog, stream=None):
    """fused_qk_gemma_rmsnorm_rope_gate_kvstore(..., 24, 4, 256, 64, q_per_prog=..., k_per_prog=...) for this model.

    Writes the rotated keys and the values into ``k_rows``/``v_rows`` ([rows, 1024] cache views) at
    ``locations``; returns (q [T, 6144], gate [T, 24, 256]).
    """
    T = q_gate.shape[0]
    if 24 % q_per_prog or 4 % k_per_prog:
        raise RuntimeError("heads per program must divide the head counts")
    q_out = torch.empty(T, 6144, dtype=q_gate.dtype, device=q_gate.device)
    gate_out = torch.empty(T, 24, 256, dtype=q_gate.dtype, device=q_gate.device)
    strides = (q_gate.stride(0), k.stride(0), v.stride(0), q_out.stride(0), gate_out.stride(0), k_rows.stride(0), v_rows.stride(0), cos_sin_cache.stride(0))
    key = (T, strides, q_per_prog, k_per_prog, eps, q_gate.dtype, k.dtype, v.dtype, k_rows.dtype, v_rows.dtype, positions.dtype, locations.dtype, cos_sin_cache.dtype,
           _align(q_gate, k, v, q_out, gate_out, k_rows, v_rows, locations, q_weight, k_weight, cos_sin_cache, positions))
    values = (q_gate, k, v, q_out, gate_out, k_rows, v_rows, locations, q_weight, k_weight, cos_sin_cache, positions, *strides,
              24, 4, q_per_prog, k_per_prog, 256, 64, 32, 256, 32, eps, q_gate.dtype == torch.float16, True, True, K._ENABLE_PDL)
    _ROPE_KVSTORE.launch(key, (T, 24 // q_per_prog + 4 // k_per_prog), values, {}, stream)
    return q_out, gate_out


def conv_split(x, weight, conv_states, query_start_loc, seq_lens, cache_indices, has_initial_state, stream=None):
    """causal_conv1d_split_fn(x, weight, None, conv_states, query_start_loc, seq_lens, cache_indices, has_initial_state, 'silu', 16, 16, 48, 128, block_n=512, num_warps=4).

    ``x`` is the channel-last [10240, T] view of the projection; returns contiguous q/k/v as [1, T, H, 128].
    """
    dim, total = x.shape
    if x.stride(0) != 1 or dim != 10240 or tuple(weight.shape) != (10240, 4) or conv_states.shape[1] != dim or conv_states.shape[2] < 3:
        raise RuntimeError("lean conv_split expects the channel-last [10240, T] projection view and the width-4 convolution")
    q = torch.empty((1, total, 16, 128), dtype=x.dtype, device=x.device)
    k = torch.empty((1, total, 16, 128), dtype=x.dtype, device=x.device)
    v = torch.empty((1, total, 48, 128), dtype=x.dtype, device=x.device)
    slots = conv_states.shape[0]
    strides = (x.stride(0), x.stride(1), weight.stride(0), weight.stride(1), conv_states.stride(0), conv_states.stride(1), conv_states.stride(2))
    key = (slots, strides, x.dtype, weight.dtype, conv_states.dtype, cache_indices.dtype, has_initial_state.dtype, query_start_loc.dtype,
           _align(x, weight, conv_states, cache_indices, has_initial_state, query_start_loc, q, k, v))
    values = (x, weight, None, conv_states, cache_indices, has_initial_state, query_start_loc, q, k, v, dim, slots, *strides, K.PAD_SLOT_ID,
              2048, 2048, 6144, False, 4, True, 4, 8, 512)
    grid = (len(seq_lens), (max(seq_lens) + 7) // 8, 20)
    _CONV_SPLIT.launch(key, grid, values, {"num_warps": 4, "num_stages": 2}, stream)
    return q, k, v


def gating_blocked_exp(A_log, a, b, dt_bias, stream=None):
    """fused_gdn_gating_blocked(A_log, a, b, dt_bias, output_exp=True): exp(g) and beta, both [1, T, 48] fp32."""
    tokens, num_heads = a.shape
    sa, sb = a.stride(0), b.stride(0)
    g = torch.empty(1, tokens, num_heads, dtype=torch.float32, device=a.device)
    beta = torch.empty(1, tokens, num_heads, dtype=torch.float32, device=b.device)
    blk_heads = triton.next_power_of_2(num_heads)
    key = (tokens, num_heads, sa, sb, a.dtype, b.dtype, A_log.dtype, dt_bias.dtype, _align(A_log, a, b, dt_bias, g, beta))
    values = (g, beta, A_log, a, b, dt_bias, tokens, sa, sb, num_heads, 1.0, 20.0, 32, blk_heads, True)
    _GATING_BLOCKED.launch(key, (triton.cdiv(tokens, 32),), values, {"num_warps": 4}, stream)
    return g, beta


def conv_tile(x, weight, conv_states, query_start_loc, seq_lens, cache_indices, has_initial_state, stream=None, *, fuse_l2norm=True):
    """Lane 3's validated 32x128/4-warp sequence-first convolution tile.

    Q/K output is L2-normalized when ``fuse_l2norm`` is true. All input-dependent
    work and the convolution-state write occur in the launched kernel.
    """
    dim, total = x.shape
    if x.stride(0) != 1 or dim != 10240 or tuple(weight.shape) != (10240, 4) or conv_states.shape[1] != dim or conv_states.shape[2] < 3:
        raise RuntimeError("lean conv_tile expects the channel-last [10240, T] projection view and the width-4 convolution")
    q = torch.empty((1, total, 16, 128), dtype=x.dtype, device=x.device)
    k = torch.empty((1, total, 16, 128), dtype=x.dtype, device=x.device)
    v = torch.empty((1, total, 48, 128), dtype=x.dtype, device=x.device)
    slots = conv_states.shape[0]
    strides = (x.stride(0), x.stride(1), weight.stride(0), weight.stride(1), conv_states.stride(0), conv_states.stride(1), conv_states.stride(2))
    block_t, block_n, warps, feat_first, eps = 32, 128, 4, False, 1e-6
    key = (slots, strides, block_t, block_n, warps, feat_first, fuse_l2norm, eps,
           x.dtype, weight.dtype, conv_states.dtype, cache_indices.dtype, has_initial_state.dtype, query_start_loc.dtype,
           _align(x, weight, conv_states, cache_indices, has_initial_state, query_start_loc, q, k, v))
    values = (x, weight, None, conv_states, cache_indices, has_initial_state, query_start_loc, q, k, v, eps, dim, slots, *strides, K.PAD_SLOT_ID,
              2048, 2048, 6144, False, 4, True, 4, block_t, block_n, feat_first, fuse_l2norm)
    _CONV_TILE.launch(key, (len(seq_lens), triton.cdiv(max(seq_lens), block_t), dim // block_n), values, {"num_warps": warps}, stream)
    return q, k, v


def joint_l2norm(q, k, eps=1e-6, stream=None):
    """joint_l2norm(q, k, eps): l2norm_fwd of contiguous [T, 16, 128] q and k in one launch (T is not specialised)."""
    if not (q.is_contiguous() and k.is_contiguous()) or q.shape != k.shape or q.dtype != k.dtype or q.shape[-1] > 512:
        raise RuntimeError("lean joint_l2norm needs equal contiguous q/k with head dim <= 512")
    qo, ko = torch.empty_like(q), torch.empty_like(k)
    d = q.shape[-1]
    rows = q.numel() // d
    bd = triton.next_power_of_2(d)
    key = (d, bd, eps, q.dtype, _align(q, k, qo, ko))
    values = (q, k, qo, ko, eps, rows, d, 16, bd)
    _JOINT_L2NORM.launch(key, (triton.cdiv(rows, 16), 2), values, {"num_warps": 8, "num_stages": 3}, stream)
    return qo, ko


# ----------------------------------------------------------------------------- Lane 3 turn-3 prefill kernels
_NORM_TOKEN = _Site(T3._norm_token_heads_kernel)
SITES["rms_norm_gated_token"] = _NORM_TOKEN


def rms_norm_gated_token(x, weight, z, eps, stream=None):
    """lane3_turn3.rms_norm_gated_token(x, weight, z, eps) through the cached direct launch.

    ``x`` is the contiguous [T*48, 128] GDN output, ``z`` the [T, 48, 128] view of the qkvz projection; the
    caller checks eligibility once per forward (``lane3_turn3.norm_token_eligible``).
    """
    if z.stride(1) != 128 or z.stride(2) != 1 or not x.is_contiguous():
        raise ValueError("token-major gated norm expects contiguous x and a [T, 48, 128] view with unit/128 inner strides")
    out = torch.empty_like(x)
    T = x.shape[0] // 48
    pdl = K.is_arch_support_pdl()
    key = (z.stride(0), eps, x.dtype, z.dtype, weight.dtype, pdl, _align(x, z, weight, out))
    values = (x, z, weight, out, eps, T3.NORM_ORDER0_T, z.stride(0), T3.NORM_HEADS, T3.NORM_ORDER, pdl)
    options = {"num_warps": T3.NORM_WARPS}
    if pdl:
        options["launch_pdl"] = True
    _NORM_TOKEN.launch(key, T3.norm_grid(T), values, options, stream)
    return out


def rope_kvstore_cuda(q_gate, k, v, q_weight, k_weight, cos_sin_cache, positions, locations, k_rows, v_rows, eps, stream=None):
    """CUDA RoPE/QK-norm/KV-store kernel (lane3_turn3); the extension reads the current stream itself, ``stream`` is accepted for symmetry."""
    return T3.rope_kvstore_cuda(q_gate, k, v, q_weight, k_weight, cos_sin_cache, positions, locations, k_rows, v_rows, eps)


# ----------------------------------------------------------------------------- Lane 3 turn-4 prefill kernels
from . import lane3_conv_loop as KL
from . import lane3_turn4 as T4

_CONV_LOOP = _Site(KL._causal_conv1d_split_loop_kernel)
SITES["conv_loop"] = _CONV_LOOP


def conv_loop(x, weight_t, conv_states, query_start_loc, seq_lens, cache_indices, has_initial_state, worklist, block_t, stream=None, *, use_worklist=True):
    """lane3_conv_loop.causal_conv1d_split_loop_fn(x, weight, None, conv_states, query_start_loc, seq_lens, cache_indices, has_initial_state,
    'silu', 16, 16, 48, 128, block_t=block_t, block_n=128, num_warps=4, feat_first=False, fuse_l2norm=True, sub_t=8, worklist=worklist,
    use_worklist=use_worklist, weight_t=weight_t) through the cached direct launch.

    ``x`` is the channel-last [10240, T] projection view, ``weight_t`` the contiguous [4, 10240] transpose of the conv weight,
    ``worklist`` the int32 packed (sequence << 16 | chunk) items of this call for ``block_t`` (unused when ``use_worklist`` is
    false: the 3-D grid then covers every chunk of every sequence). Returns L2-normalised q and k plus v as [1, T, H, 128];
    the conv state of every active slot is updated inside the kernel.
    """
    dim, total = x.shape
    if (x.stride(0) != 1 or dim != 10240 or tuple(weight_t.shape) != (4, 10240) or weight_t.stride(1) != 1
            or conv_states.shape[1] != dim or conv_states.shape[2] < 3):
        raise RuntimeError("lean conv_loop expects the channel-last [10240, T] projection view, the transposed width-4 weight and a [slots, 10240, >=3] state")
    if block_t not in (32, 64, 128):
        raise RuntimeError("lean conv_loop supports 32, 64 or 128 tokens per program")
    q = torch.empty((1, total, 16, 128), dtype=x.dtype, device=x.device)
    k = torch.empty((1, total, 16, 128), dtype=x.dtype, device=x.device)
    v = torch.empty((1, total, 48, 128), dtype=x.dtype, device=x.device)
    slots = conv_states.shape[0]
    strides = (x.stride(0), x.stride(1), weight_t.stride(1), weight_t.stride(0), conv_states.stride(0), conv_states.stride(1), conv_states.stride(2))
    eps = 1e-6
    if use_worklist:
        if worklist is None or worklist.dtype != torch.int32 or worklist.ndim != 1 or worklist.numel() == 0:
            raise RuntimeError("lean conv_loop needs the non-empty int32 worklist of this call")
        work = worklist
        grid = (worklist.numel(), dim // 128)
    else:
        work = query_start_loc  # unused placeholder pointer, as in the wrapper
        grid = (len(seq_lens), triton.cdiv(max(seq_lens), block_t), dim // 128)
    key = (slots, strides, block_t, use_worklist, x.dtype, weight_t.dtype, conv_states.dtype, cache_indices.dtype, has_initial_state.dtype,
           query_start_loc.dtype, work.dtype, _align(x, weight_t, conv_states, cache_indices, has_initial_state, query_start_loc, work, q, k, v))
    values = (x, weight_t, None, conv_states, cache_indices, has_initial_state, query_start_loc, work, q, k, v, eps, dim, slots, *strides, K.PAD_SLOT_ID,
              2048, 2048, 6144, False, 4, True, 4, block_t, KL.CONV_SUB_T, 128, False, True, use_worklist, 1, 1)
    _CONV_LOOP.launch(key, grid, values, {"num_warps": KL.CONV_NUM_WARPS}, stream)
    return q, k, v


def rope_kvstore_cuda4(q_gate, k, v, q_weight, k_weight, cos_sin_cache, positions, locations, k_rows, v_rows, eps, stream=None):
    """CUDA RoPE/QK-norm/KV-store kernel with four heads per warp (lane3_turn4); the extension reads the current stream itself."""
    return T4.rope_kvstore_cuda4(q_gate, k, v, q_weight, k_weight, cos_sin_cache, positions, locations, k_rows, v_rows, eps)
