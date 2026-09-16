"""Lane 3 turn-3 prefill kernels (CUDA RoPE/QK-norm/KV store, token-major gated norm), explicit dispatch.

Source: Lane 3's runtime-validated turn-3 package (artifacts/lane-3/turn3; source.tar.gz sha256
95a9cd9988e917493b101094b344e3a2c0c0f28efe1d2e3f369961b701396cc4). Two kernels are taken over:

* ``solution/lane3_rope_cuda3.cu`` is a byte-identical copy of their ``rope_cuda3.cu`` (sha256
  recorded in their source-manifest.json). One warp per 256-feature head, four warps per block:
  Gemma-RMSNorm of the 24 query / 4 key heads with a lane-permuted but bitwise-identical reduction
  tree, RoPE on the first 64 features, the query gate passed through as raw bits, and the rotated
  keys plus the values written straight into the paged K/V cache rows (width 1024). It replaces
  the Triton fused RoPE + KV-store kernel (solution.lane3_kernels) for prefills of at least 256
  tokens, Lane 3's validated eligibility; decode and smaller prefills keep the Triton path.
* ``_norm_token_heads_kernel`` is copied verbatim from their ``norm_triton3.py``: GDN gated RMSNorm
  (norm before the swish gate) with one token and BH heads per program, scalar token addressing and
  no rstd output. Lane 3 selected 16 heads, two warps and grid order 0. It replaces the reference
  ``_layer_norm_fwd_1pass_kernel`` (8 rows per program) for prefills of at least 256 tokens.

Neither kernel caches token-dependent results; callers own every tensor. The engine enables each
kernel only after ``Engine._check_fused`` reproduced the reference kernels bitwise on random
model-shaped data (flags ``rope_cuda`` and ``norm_token``); a failed check keeps the previous path.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
import triton
import triton.language as tl

from . import qwen38_optimized as K

CUDA_SOURCE = Path(__file__).with_name("lane3_rope_cuda3.cu")
EXTENSION_NAME = "lane3_rope_cuda3_v1"
_CPP = """
#include <torch/extension.h>
void rope_cuda3_launch(const at::Tensor&, const at::Tensor&, const at::Tensor&,
 const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
 const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
 const at::Tensor&, double, int, int);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("launch", &rope_cuda3_launch); }
"""
_EXTENSION = None
# Lane 3's selected launch configuration: four warps per block, token-major work order.
ROPE_WARPS = 4
ROPE_ORDER = 0


def extension():
    """Build (once per persistent kernel cache) and return the CUDA extension."""
    global _EXTENSION
    if _EXTENSION is None:
        from torch.utils.cpp_extension import load_inline
        root = Path(os.environ.get("QWEN38_KERNEL_CACHE", Path.home() / ".cache/qwen38_standalone")) / EXTENSION_NAME
        root.mkdir(parents=True, exist_ok=True)
        _EXTENSION = load_inline(EXTENSION_NAME, cpp_sources=_CPP, cuda_sources=CUDA_SOURCE.read_text(),
                                 extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_100,code=sm_100"],
                                 with_cuda=True, verbose=False, build_directory=str(root))
    return _EXTENSION


def rope_kvstore_cuda(q_gate, k, v, q_weight, k_weight, cos_sin_cache, positions, locations, k_rows, v_rows, eps):
    """QK Gemma-RMSNorm + RoPE + gate split for this model's [T, 12288] q/gate projection; K and V stored into the cache rows.

    Same contract as ``lane3_kernels.fused_qk_gemma_rmsnorm_rope_gate_kvstore(..., 24, 4, 256, 64)``: returns
    (q [T, 6144], gate [T, 24, 256]); ``k_rows``/``v_rows`` are [rows, 1024] views of the paged cache and receive the
    rotated keys and the values at ``locations``. Runs on the current CUDA stream.
    """
    T = q_gate.shape[0]
    if (q_gate.dtype != torch.bfloat16 or q_gate.shape[1] != 12288 or k.shape != (T, 1024) or v.shape != (T, 1024)
            or q_gate.stride(1) != 1 or k.stride(1) != 1 or v.stride(1) != 1 or k_rows.stride() != (1024, 1) or v_rows.stride() != (1024, 1)
            or cos_sin_cache.dtype != torch.float32 or cos_sin_cache.stride() != (64, 1) or positions.dtype != torch.int64 or locations.dtype != torch.int64
            or positions.shape != (T,) or locations.shape != (T,) or not q_weight.is_contiguous() or not k_weight.is_contiguous()
            or q_weight.dtype != torch.bfloat16 or k_weight.dtype != torch.bfloat16):
        raise ValueError("unsupported layout for the CUDA RoPE/KV-store kernel")
    q_out = torch.empty(T, 6144, dtype=q_gate.dtype, device=q_gate.device)
    gate_out = torch.empty(T, 24, 256, dtype=q_gate.dtype, device=q_gate.device)
    extension().launch(q_gate, k, v, q_weight, k_weight, cos_sin_cache, positions, locations, q_out, gate_out, k_rows, v_rows, float(eps), ROPE_WARPS, ROPE_ORDER)
    return q_out, gate_out


# --------------------------------------------------------------------------------------- gated norm
# Verbatim copy of Lane 3's norm_triton3._norm_token_heads_kernel (see module docstring).
@triton.jit
def _norm_token_heads_kernel(X, Z, W, Y, eps, T: tl.constexpr,
                             Z_STRIDE: tl.constexpr, BH: tl.constexpr,
                             ORDER: tl.constexpr, PDL: tl.constexpr):
    if PDL:
        tl.extra.cuda.gdc_wait()
    if ORDER == 0:
        token = tl.program_id(0) // tl.cdiv(48, BH)
        group = tl.program_id(0) % tl.cdiv(48, BH)
    else:
        token = tl.program_id(0) % T
        group = tl.program_id(0) // T
    heads = group * BH + tl.arange(0, BH)
    cols = tl.arange(0, 128)
    mask = heads[:, None] < 48
    offsets = heads[:, None] * 128 + cols[None, :]
    x = tl.load(X + token * 6144 + offsets, mask, other=0).to(tl.float32)
    z = tl.load(Z + token * Z_STRIDE + offsets, mask, other=0).to(tl.float32)
    w = tl.load(W + cols).to(tl.float32)
    var = tl.sum(x * x, axis=1) / 128
    rstd = tl.rsqrt(var + eps)
    out = (x * rstd[:, None]) * w[None, :]
    out *= z * tl.sigmoid(z)
    tl.store(Y + token * 6144 + offsets, out, mask)
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()


# Lane 3's selected configuration: 16 heads per program, two warps, grid order 0 (token-major).
NORM_HEADS = 16
NORM_WARPS = 2
NORM_ORDER = 0
# ``T`` is only read by grid order 1; with order 0 the token index comes from the program id, so a
# fixed placeholder avoids compiling one kernel per distinct token count (the compiled code is the
# same for every value, and the engine's self-check plus the full suite validate this launch).
NORM_ORDER0_T = 0


def norm_token_eligible(x, z, weight):
    """Lane 3's eligibility: contiguous bf16 x [T*48, 128], z a [T, 48, 128] view with unit/128 inner strides, T >= 256, 32-bit offsets."""
    return (x.dtype == torch.bfloat16 and x.ndim == 2 and x.shape[1] == 128 and x.is_contiguous() and x.shape[0] % 48 == 0
            and z.dtype == torch.bfloat16 and z.ndim == 3 and z.shape == (x.shape[0] // 48, 48, 128) and z.stride(1) == 128 and z.stride(2) == 1
            and z.shape[0] >= 256 and z.shape[0] * max(6144, z.stride(0)) < 2 ** 31
            and weight.dtype == torch.bfloat16 and weight.is_contiguous() and weight.shape == (128,))


def norm_grid(T):
    return (T * triton.cdiv(48, NORM_HEADS),)


def rms_norm_gated_token(x, weight, z, eps):
    """Plain Triton launch (JIT dispatch per call) of the token-major gated norm; ``solution.lean`` caches the compiled kernel."""
    if not norm_token_eligible(x, z, weight):
        raise ValueError("unsupported layout for the token-major gated norm")
    out = torch.empty_like(x)
    T = x.shape[0] // 48
    pdl = K.is_arch_support_pdl()
    _norm_token_heads_kernel[norm_grid(T)](x, z, weight, out, eps, T=NORM_ORDER0_T, Z_STRIDE=z.stride(0), BH=NORM_HEADS, ORDER=NORM_ORDER, PDL=pdl,
                                           num_warps=NORM_WARPS, **({"launch_pdl": True} if pdl else {}))
    return out
