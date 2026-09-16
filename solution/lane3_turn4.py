"""Lane 3 turn-4 prefill kernels as integrated: RoPE/QK-norm/KV store with four heads per warp.

Source: Lane 3's runtime-validated turn-4 package (artifacts/lane-3/turn4; source.tar.gz sha256
b256814448996ebbbae75e042248e7f380c9e4b200c7d5497c84598d71cb9a19). ``solution/lane3_rope_cuda4.cu`` is a
byte-identical copy of their ``solution/rope_cuda4.cu`` (sha256
753ad5e3ecfe27034388bd764d21281fe6e1b88e245062fe04183a59b0745227). It keeps every per-element operation,
intrinsic and rounding of the turn-3 kernel (``solution/lane3_rope_cuda3.cu``, one warp per 256-feature
head) and changes only the warp-to-work mapping: a warp owns G = 4 heads of one token (all 24 query heads
in six warps, the 4 key heads in one), issues every input and pass-through load before any arithmetic and
loads the normalization weight and the RoPE table row once per warp. Lane 3 measured 1.29x over the turn-3
kernel at 32768 tokens (413 vs 535 us hot) and 120/120 bitwise component comparisons.

The engine selects it with ``kv_store="cuda4"`` for prefills of at least 256 tokens (Lane 3's eligibility;
decode and smaller prefills keep the Triton fused store) after ``Engine._check_fused`` reproduced the
reference RoPE kernel plus the reference paged store bitwise (flag ``rope_cuda``). The kernel's
convolution companion lives in ``solution.lane3_conv_loop``.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

CUDA_SOURCE = Path(__file__).with_name("lane3_rope_cuda4.cu")
EXTENSION_NAME = "lane3_rope_cuda4_v1"
_CPP = """
#include <torch/extension.h>
void rope_cuda4_launch(const at::Tensor&, const at::Tensor&, const at::Tensor&,
 const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
 const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
 const at::Tensor&, double, int, int);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("launch", &rope_cuda4_launch); }
"""
_EXTENSION = None
# Lane 3's selected launch configuration: four warps per block, four heads per warp.
ROPE_WARPS = 4
ROPE_HEADS = 4


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


def rope_kvstore_cuda4(q_gate, k, v, q_weight, k_weight, cos_sin_cache, positions, locations, k_rows, v_rows, eps, heads=ROPE_HEADS, warps=ROPE_WARPS):
    """Same contract as ``lane3_turn3.rope_kvstore_cuda``: returns (q [T, 6144], gate [T, 24, 256]); K/V rows written at ``locations``.

    ``heads`` (1, 2 or 4) is the number of heads per warp and ``warps`` (4 or 8) the warps per block; the
    kernel's arithmetic does not depend on either. Runs on the current CUDA stream.
    """
    T = q_gate.shape[0]
    if (q_gate.dtype != torch.bfloat16 or q_gate.shape[1] != 12288 or k.shape != (T, 1024) or v.shape != (T, 1024)
            or q_gate.stride(1) != 1 or k.stride(1) != 1 or v.stride(1) != 1 or k_rows.stride() != (1024, 1) or v_rows.stride() != (1024, 1)
            or cos_sin_cache.dtype != torch.float32 or cos_sin_cache.stride() != (64, 1) or positions.dtype != torch.int64 or locations.dtype != torch.int64
            or positions.shape != (T,) or locations.shape != (T,) or not q_weight.is_contiguous() or not k_weight.is_contiguous()
            or q_weight.dtype != torch.bfloat16 or k_weight.dtype != torch.bfloat16 or heads not in (1, 2, 4) or warps not in (4, 8)):
        raise ValueError("unsupported layout for the CUDA RoPE/KV-store kernel (four heads per warp)")
    q_out = torch.empty(T, 6144, dtype=q_gate.dtype, device=q_gate.device)
    gate_out = torch.empty(T, 24, 256, dtype=q_gate.dtype, device=q_gate.device)
    extension().launch(q_gate, k, v, q_weight, k_weight, cos_sin_cache, positions, locations, q_out, gate_out, k_rows, v_rows, float(eps), warps, heads)
    return q_out, gate_out
