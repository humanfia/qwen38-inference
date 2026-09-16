// Exact RoPE/QK normalization (turn-3 permuted reference tree) with several heads per warp.
//
// The turn-3 kernel (rope_cuda3.cu) gives one warp one 256-feature head: 22 loads and 12
// stores per warp, each lane holding four feature pairs, and only its own head's loads in
// flight. Offline SASS (256 instructions per warp, 32 registers) and turn-3 NCU (52.6%
// issue utilization, long-scoreboard stalls) show it is bound by memory-level parallelism,
// not by instruction issue. This kernel keeps every per-element operation, intrinsic and
// rounding of the turn-3 kernel and changes only the mapping: a warp owns G heads of one
// token (G | 24 and G | 4), issues all of their input and pass-through loads before any
// arithmetic, loads the normalization weight and the RoPE table once per warp, and then
// runs the unchanged per-head sequence (products, permuted reduction, rsqrt, normalization,
// rotation, stores).
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

using bf16 = __nv_bfloat16;

__device__ __forceinline__ float2 load_pair(const bf16* p) {
  return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(p));
}

__device__ __forceinline__ void store_pair(bf16* p, float2 v) {
  *reinterpret_cast<__nv_bfloat162*>(p) = __floats2bfloat162_rn(v.x, v.y);
}

__device__ __forceinline__ float rounded_bf16(float x) {
  return __bfloat162float(__float2bfloat16_rn(x));
}

__device__ __forceinline__ float reference_rsqrt(float x) {
  float result;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
  return result;
}

template<int WARPS, int G>
__global__ void __launch_bounds__(WARPS * 32) rope_group_kernel(
    const bf16* __restrict__ qg, const bf16* __restrict__ k,
    const bf16* __restrict__ v, const bf16* __restrict__ qw,
    const bf16* __restrict__ kw, const float* __restrict__ rope,
    const int64_t* __restrict__ positions, const int64_t* __restrict__ locations,
    bf16* __restrict__ qo, bf16* __restrict__ go,
    bf16* __restrict__ kc, bf16* __restrict__ vc,
    int tokens, int sq, int sk, int sv, float eps) {
  constexpr int Q_GROUPS = 24 / G;
  constexpr int GROUPS = Q_GROUPS + 4 / G;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int item = blockIdx.x * WARPS + warp;
  if (item >= tokens * GROUPS) return;
  const int token = item / GROUPS;
  const int group = item % GROUPS;
  const bool query = group < Q_GROUPS;
  const int h0 = query ? group * G : (group - Q_GROUPS) * G;
  const int offset = (lane & 7) * 2 + (lane >> 3) * 64;
  const int64_t location = query ? 0 : locations[token];
  // Per-head source rows: q heads are 512 wide (q | gate); k heads 256 wide.
  const bf16* src0 = query ? qg + int64_t(token) * sq + h0 * 512
                           : k + int64_t(token) * sk + h0 * 256;
  const int src_head_stride = query ? 512 : 256;
  const bf16* weight = query ? qw : kw;
  bf16* dst0 = query ? qo + int64_t(token) * 6144 + h0 * 256
                     : kc + location * 1024 + h0 * 256;
  const bf16* pass0 = query ? src0 + 256 : v + int64_t(token) * sv + h0 * 256;
  const int pass_head_stride = query ? 512 : 256;
  bf16* pass_dst0 = query ? go + int64_t(token) * 6144 + h0 * 256
                          : vc + location * 1024 + h0 * 256;

  // Issue every load first: G heads x 4 pairs of input, G heads x 4 raw pass-through words,
  // the shared normalization weight and (lanes < 8) the shared RoPE table row.
  float2 x[G][4];
  uint32_t raw[G][4];
  float2 w[4];
  #pragma unroll
  for (int g = 0; g < G; ++g) {
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      x[g][j] = load_pair(src0 + g * src_head_stride + offset + j * 16);
      raw[g][j] = *reinterpret_cast<const uint32_t*>(pass0 + g * pass_head_stride + offset + j * 16);
    }
  }
  #pragma unroll
  for (int j = 0; j < 4; ++j) w[j] = load_pair(weight + offset + j * 16);
  float2 cs[2], sn[2];
  if (lane < 8) {
    const float* table = rope + positions[token] * 64 + offset;
    #pragma unroll
    for (int j = 0; j < 2; ++j) {
      cs[j] = *reinterpret_cast<const float2*>(table + j * 16);
      sn[j] = *reinterpret_cast<const float2*>(table + 32 + j * 16);
    }
  }

  #pragma unroll
  for (int g = 0; g < G; ++g) {
    float p[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) p[j] = __fmaf_rn(x[g][j].y, x[g][j].y, __fmul_rn(x[g][j].x, x[g][j].x));
    // Reference: adjacent pair, then feature offsets 32,16,8,4,2,128,64 (turn-3 permutation).
    float total = __fadd_rn(__fadd_rn(p[0], p[2]), __fadd_rn(p[1], p[3]));
    total = __fadd_rn(total, __shfl_xor_sync(0xffffffff, total, 4));
    total = __fadd_rn(total, __shfl_xor_sync(0xffffffff, total, 2));
    total = __fadd_rn(total, __shfl_xor_sync(0xffffffff, total, 1));
    total = __fadd_rn(total, __shfl_xor_sync(0xffffffff, total, 16));
    total = __fadd_rn(total, __shfl_xor_sync(0xffffffff, total, 8));
    const float inv = reference_rsqrt(__fadd_rn(__fmul_rn(total, 1.0f / 256), eps));
    float2 y[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      y[j].x = rounded_bf16(__fmul_rn(__fmul_rn(x[g][j].x, inv), __fadd_rn(w[j].x, 1.f)));
      y[j].y = rounded_bf16(__fmul_rn(__fmul_rn(x[g][j].y, inv), __fadd_rn(w[j].y, 1.f)));
    }
    if (lane < 8) {
      #pragma unroll
      for (int j = 0; j < 2; ++j) {
        const float2 a = y[j], b = y[j + 2];
        y[j] = {__fmaf_rn(a.x, cs[j].x, -__fmul_rn(b.x, sn[j].x)),
                __fmaf_rn(a.y, cs[j].y, -__fmul_rn(b.y, sn[j].y))};
        // Triton's second half fuses a*sin and rounds b*cos first.
        y[j + 2] = {__fmaf_rn(a.x, sn[j].x, __fmul_rn(b.x, cs[j].x)),
                    __fmaf_rn(a.y, sn[j].y, __fmul_rn(b.y, cs[j].y))};
      }
    }
    bf16* dst = dst0 + g * 256;
    bf16* pass_dst = pass_dst0 + g * 256;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      store_pair(dst + offset + j * 16, y[j]);
      // Pure pass-through preserves the raw BF16 gate/value bits.
      *reinterpret_cast<uint32_t*>(pass_dst + offset + j * 16) = raw[g][j];
    }
  }
}

void rope_cuda4_launch(const at::Tensor& qg, const at::Tensor& k, const at::Tensor& v,
                      const at::Tensor& qw, const at::Tensor& kw, const at::Tensor& rope,
                      const at::Tensor& positions, const at::Tensor& locations,
                      const at::Tensor& qo, const at::Tensor& go,
                      const at::Tensor& kc, const at::Tensor& vc, double eps, int warps, int heads) {
  TORCH_CHECK(qg.is_cuda() && qg.scalar_type() == at::kBFloat16, "CUDA BF16 required");
  TORCH_CHECK(positions.scalar_type() == at::kLong && locations.scalar_type() == at::kLong, "int64 indices required");
  TORCH_CHECK(warps == 4 || warps == 8, "unsupported warp count");
  TORCH_CHECK(heads == 1 || heads == 2 || heads == 4, "unsupported heads per warp");
  const int t = qg.size(0);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  #define RUN(W, G) rope_group_kernel<W, G><<<(t * (28 / G) + W - 1) / W, W * 32, 0, stream>>>( \
    reinterpret_cast<const bf16*>(qg.data_ptr()), reinterpret_cast<const bf16*>(k.data_ptr()), \
    reinterpret_cast<const bf16*>(v.data_ptr()), reinterpret_cast<const bf16*>(qw.data_ptr()), \
    reinterpret_cast<const bf16*>(kw.data_ptr()), rope.data_ptr<float>(), positions.data_ptr<int64_t>(), \
    locations.data_ptr<int64_t>(), reinterpret_cast<bf16*>(qo.data_ptr()), reinterpret_cast<bf16*>(go.data_ptr()), \
    reinterpret_cast<bf16*>(kc.data_ptr()), reinterpret_cast<bf16*>(vc.data_ptr()), t, qg.stride(0), k.stride(0), v.stride(0), float(eps))
  if (warps == 4) {
    if (heads == 1) { RUN(4, 1); } else if (heads == 2) { RUN(4, 2); } else { RUN(4, 4); }
  } else {
    if (heads == 1) { RUN(8, 1); } else if (heads == 2) { RUN(8, 2); } else { RUN(8, 4); }
  }
  #undef RUN
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
