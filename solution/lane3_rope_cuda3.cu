// Exact RoPE/QK normalization with a permuted reference reduction tree.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

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

template<int WARPS, int ORDER>
__global__ void rope_permuted_kernel(
    const bf16* __restrict__ qg, const bf16* __restrict__ k,
    const bf16* __restrict__ v, const bf16* __restrict__ qw,
    const bf16* __restrict__ kw, const float* __restrict__ rope,
    const int64_t* __restrict__ positions, const int64_t* __restrict__ locations,
    bf16* __restrict__ qo, bf16* __restrict__ go,
    bf16* __restrict__ kc, bf16* __restrict__ vc,
    int tokens, int sq, int sk, int sv, float eps) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int item = blockIdx.x * WARPS + warp;
  if (item >= tokens * 28) return;
  int token, head;
  if constexpr (ORDER == 0) {
    token = item / 28;
    head = item % 28;
  } else {
    token = item % tokens;
    head = item / tokens;
  }
  const bool query = head < 24;
  const int h = query ? head : head - 24;
  const int offset = (lane & 7) * 2 + (lane >> 3) * 64;
  const int64_t location = query ? 0 : locations[token];
  const bf16* src = query ? qg + int64_t(token) * sq + h * 512
                          : k + int64_t(token) * sk + h * 256;
  const bf16* weight = query ? qw : kw;
  bf16* dst = query ? qo + int64_t(token) * 6144 + h * 256
                    : kc + location * 1024 + h * 256;
  float2 x[4], w[4], y[4];
  float p[4];
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    x[j] = load_pair(src + offset + j * 16);
    w[j] = load_pair(weight + offset + j * 16);
    p[j] = __fmaf_rn(x[j].y, x[j].y, __fmul_rn(x[j].x, x[j].x));
  }
  // Reference: adjacent pair, then feature offsets 32,16,8,4,2,128,64.
  // The first two shuffle stages become lane-local register additions.
  float total = __fadd_rn(__fadd_rn(p[0], p[2]), __fadd_rn(p[1], p[3]));
  total = __fadd_rn(total, __shfl_xor_sync(0xffffffff, total, 4));
  total = __fadd_rn(total, __shfl_xor_sync(0xffffffff, total, 2));
  total = __fadd_rn(total, __shfl_xor_sync(0xffffffff, total, 1));
  total = __fadd_rn(total, __shfl_xor_sync(0xffffffff, total, 16));
  total = __fadd_rn(total, __shfl_xor_sync(0xffffffff, total, 8));
  const float inv = reference_rsqrt(__fadd_rn(__fmul_rn(total, 1.0f / 256), eps));
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    y[j].x = rounded_bf16(__fmul_rn(__fmul_rn(x[j].x, inv), __fadd_rn(w[j].x, 1.f)));
    y[j].y = rounded_bf16(__fmul_rn(__fmul_rn(x[j].y, inv), __fadd_rn(w[j].y, 1.f)));
  }
  if (lane < 8) {
    const float* table = rope + positions[token] * 64 + offset;
    #pragma unroll
    for (int j = 0; j < 2; ++j) {
      const float2 cs = *reinterpret_cast<const float2*>(table + j * 16);
      const float2 sn = *reinterpret_cast<const float2*>(table + 32 + j * 16);
      const float2 a = y[j], b = y[j + 2];
      y[j] = {__fmaf_rn(a.x, cs.x, -__fmul_rn(b.x, sn.x)),
              __fmaf_rn(a.y, cs.y, -__fmul_rn(b.y, sn.y))};
      // Triton's second half fuses a*sin and rounds b*cos first.
      y[j + 2] = {__fmaf_rn(a.x, sn.x, __fmul_rn(b.x, cs.x)),
                  __fmaf_rn(a.y, sn.y, __fmul_rn(b.y, cs.y))};
    }
  }
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    store_pair(dst + offset + j * 16, y[j]);
    // Pure pass-through preserves the raw BF16 gate/value bits.
    if (query) {
      const uint32_t raw = *reinterpret_cast<const uint32_t*>(src + 256 + offset + j * 16);
      *reinterpret_cast<uint32_t*>(go + int64_t(token) * 6144 + h * 256 + offset + j * 16) = raw;
    } else {
      const uint32_t raw = *reinterpret_cast<const uint32_t*>(v + int64_t(token) * sv + h * 256 + offset + j * 16);
      *reinterpret_cast<uint32_t*>(vc + location * 1024 + h * 256 + offset + j * 16) = raw;
    }
  }
}

void rope_cuda3_launch(const at::Tensor& qg, const at::Tensor& k, const at::Tensor& v,
                      const at::Tensor& qw, const at::Tensor& kw, const at::Tensor& rope,
                      const at::Tensor& positions, const at::Tensor& locations,
                      const at::Tensor& qo, const at::Tensor& go,
                      const at::Tensor& kc, const at::Tensor& vc, double eps, int warps, int order) {
  TORCH_CHECK(qg.is_cuda() && qg.scalar_type() == at::kBFloat16, "CUDA BF16 required");
  TORCH_CHECK(positions.scalar_type() == at::kLong && locations.scalar_type() == at::kLong, "int64 indices required");
  TORCH_CHECK(warps == 4 || warps == 8 || warps == 16, "unsupported warp count");
  TORCH_CHECK(order == 0 || order == 1, "unsupported grid order");
  const int t = qg.size(0);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  #define RUN(W, O) rope_permuted_kernel<W, O><<<(t * 28 + W - 1) / W, W * 32, 0, stream>>>( \
    reinterpret_cast<const bf16*>(qg.data_ptr()), reinterpret_cast<const bf16*>(k.data_ptr()), \
    reinterpret_cast<const bf16*>(v.data_ptr()), reinterpret_cast<const bf16*>(qw.data_ptr()), \
    reinterpret_cast<const bf16*>(kw.data_ptr()), rope.data_ptr<float>(), positions.data_ptr<int64_t>(), \
    locations.data_ptr<int64_t>(), reinterpret_cast<bf16*>(qo.data_ptr()), reinterpret_cast<bf16*>(go.data_ptr()), \
    reinterpret_cast<bf16*>(kc.data_ptr()), reinterpret_cast<bf16*>(vc.data_ptr()), t, qg.stride(0), k.stride(0), v.stride(0), float(eps))
  if (order == 0) {
    if (warps == 4) { RUN(4, 0); } else if (warps == 8) { RUN(8, 0); } else { RUN(16, 0); }
  } else {
    if (warps == 4) { RUN(4, 1); } else if (warps == 8) { RUN(8, 1); } else { RUN(16, 1); }
  }
  #undef RUN
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
