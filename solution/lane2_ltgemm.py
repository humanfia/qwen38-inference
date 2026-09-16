"""cuBLASLt algorithm enumeration and dispatch for bf16 GEMMs (Lane 2 dense-op research).

Computes out[M, N] = x[M, K] @ w[N, K]^T with fp32 accumulation, exactly the operation
performed by torch.nn.functional.linear(x, w) (cublasGemmEx, CUBLAS_COMPUTE_32F). It exposes
cuBLASLt's heuristic algorithm list so that per-shape algorithms can be timed and filtered for
bitwise agreement with the F.linear result.

The extension is built with torch.utils.cpp_extension.load_inline on first use.
"""
import os
from pathlib import Path

import torch

_EXT = None

_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cstring>
#include <map>
#include <tuple>
#include <vector>

#define LT_CHECK(x) do { cublasStatus_t _s = (x); TORCH_CHECK(_s == CUBLAS_STATUS_SUCCESS, "cublasLt error ", (int)_s, ": ", #x); } while (0)

static cublasLtHandle_t lt_handle() {
  static cublasLtHandle_t h = nullptr;
  if (h == nullptr) LT_CHECK(cublasLtCreate(&h));
  return h;
}

struct Plan {
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
};

// out(row-major M x N) = x(row-major M x K) * w(row-major N x K)^T
// Column-major view: C^T[N x M] = op(A) * op(B), A = w viewed as K x N (ld K, transposed), B = x viewed K x M (ld K).
static Plan& plan_for(int64_t M, int64_t N, int64_t K, bool out_f32) {
  static std::map<std::tuple<int64_t, int64_t, int64_t, bool>, Plan> plans;
  auto key = std::make_tuple(M, N, K, out_f32);
  auto it = plans.find(key);
  if (it != plans.end()) return it->second;
  Plan p;
  LT_CHECK(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  LT_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
  LT_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
  LT_CHECK(cublasLtMatrixLayoutCreate(&p.a, CUDA_R_16BF, K, N, K));
  LT_CHECK(cublasLtMatrixLayoutCreate(&p.b, CUDA_R_16BF, K, M, K));
  LT_CHECK(cublasLtMatrixLayoutCreate(&p.c, out_f32 ? CUDA_R_32F : CUDA_R_16BF, N, M, N));
  return plans.emplace(key, p).first->second;
}

static int algo_attr(const cublasLtMatmulAlgo_t& algo, cublasLtMatmulAlgoConfigAttributes_t attr) {
  int v = -1; size_t written = 0;
  cublasStatus_t s = cublasLtMatmulAlgoConfigGetAttribute(&algo, attr, &v, sizeof(v), &written);
  return s == CUBLAS_STATUS_SUCCESS ? v : -1;
}

// Returns (algos uint8 [n, sizeof(algo)], workspace int64 [n], waves float64 [n], attrs int64 [n, 8])
// attrs columns: id, tile, splitk, reduction, swizzle, custom, stages, inner/cluster
std::vector<torch::Tensor> lt_heuristics(int64_t M, int64_t N, int64_t K, int64_t max_algos, int64_t max_workspace) {
  Plan& p = plan_for(M, N, K, false);
  cublasLtMatmulPreference_t pref;
  LT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  size_t ws = (size_t)max_workspace;
  LT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
  std::vector<cublasLtMatmulHeuristicResult_t> res(max_algos);
  int returned = 0;
  cublasStatus_t s = cublasLtMatmulAlgoGetHeuristic(lt_handle(), p.op, p.a, p.b, p.c, p.c, pref, (int)max_algos, res.data(), &returned);
  cublasLtMatmulPreferenceDestroy(pref);
  TORCH_CHECK(s == CUBLAS_STATUS_SUCCESS || s == CUBLAS_STATUS_NOT_SUPPORTED, "heuristic query failed ", (int)s);
  auto algos = torch::empty({returned, (int64_t)sizeof(cublasLtMatmulAlgo_t)}, torch::kUInt8);
  auto wss = torch::empty({returned}, torch::kInt64);
  auto waves = torch::empty({returned}, torch::kFloat64);
  auto attrs = torch::empty({returned, 8}, torch::kInt64);
  for (int i = 0; i < returned; ++i) {
    std::memcpy(algos[i].data_ptr(), &res[i].algo, sizeof(cublasLtMatmulAlgo_t));
    wss[i] = (int64_t)res[i].workspaceSize;
    waves[i] = (double)res[i].wavesCount;
    attrs[i][0] = algo_attr(res[i].algo, CUBLASLT_ALGO_CONFIG_ID);
    attrs[i][1] = algo_attr(res[i].algo, CUBLASLT_ALGO_CONFIG_TILE_ID);
    attrs[i][2] = algo_attr(res[i].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM);
    attrs[i][3] = algo_attr(res[i].algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME);
    attrs[i][4] = algo_attr(res[i].algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING);
    attrs[i][5] = algo_attr(res[i].algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION);
    attrs[i][6] = algo_attr(res[i].algo, CUBLASLT_ALGO_CONFIG_STAGES_ID);
    attrs[i][7] = algo_attr(res[i].algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID);
  }
  return {algos, wss, waves, attrs};
}

// algo: CPU uint8 tensor of sizeof(cublasLtMatmulAlgo_t) bytes, or empty for the heuristic default.
void lt_matmul(torch::Tensor x, torch::Tensor w, torch::Tensor out, torch::Tensor algo, torch::Tensor workspace) {
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && out.dim() == 2, "2D tensors required");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && out.is_contiguous(), "contiguous tensors required");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && w.scalar_type() == torch::kBFloat16, "bf16 inputs required");
  int64_t M = x.size(0), K = x.size(1), N = w.size(0);
  TORCH_CHECK(w.size(1) == K && out.size(0) == M && out.size(1) == N, "shape mismatch");
  bool out_f32 = out.scalar_type() == torch::kFloat32;
  TORCH_CHECK(out_f32 || out.scalar_type() == torch::kBFloat16, "output must be bf16 or fp32");
  Plan& p = plan_for(M, N, K, out_f32);
  float alpha = 1.0f, beta = 0.0f;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const cublasLtMatmulAlgo_t* algo_ptr = nullptr;
  cublasLtMatmulAlgo_t algo_s;
  if (algo.numel() > 0) {
    TORCH_CHECK(algo.numel() == (int64_t)sizeof(cublasLtMatmulAlgo_t) && algo.device().is_cpu(), "bad algo blob");
    std::memcpy(&algo_s, algo.contiguous().data_ptr(), sizeof(algo_s));
    algo_ptr = &algo_s;
  }
  void* ws = workspace.numel() > 0 ? workspace.data_ptr() : nullptr;
  LT_CHECK(cublasLtMatmul(lt_handle(), p.op, &alpha, w.data_ptr(), p.a, x.data_ptr(), p.b, &beta,
                          out.data_ptr(), p.c, out.data_ptr(), p.c, algo_ptr, ws, (size_t)workspace.numel(), stream));
}

int64_t lt_version() { return (int64_t)cublasLtGetVersion(); }

// Heuristic results can contain private nvjet configuration fields that AlgoInit plus
// public ConfigSet attributes cannot reconstruct (observed CUBLAS_STATUS_NOT_SUPPORTED).
// Recover the complete result from the same shape/workspace heuristic query instead.
static cublasLtMatmulHeuristicResult_t lt_resolve_attrs(
    Plan& p, const std::vector<int64_t>& attrs) {
  TORCH_CHECK(attrs.size() == 8, "attrs = [id, tile, splitk, reduction, swizzle, custom, stages, cluster]");
  cublasLtMatmulPreference_t pref;
  LT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  size_t ws = 64 * 1024 * 1024;
  LT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
  const cublasLtMatmulAlgoConfigAttributes_t keys[] = {
      CUBLASLT_ALGO_CONFIG_ID, CUBLASLT_ALGO_CONFIG_TILE_ID, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
      CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
      CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, CUBLASLT_ALGO_CONFIG_STAGES_ID,
      CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID};
  for (int count : {8, 64, 256}) {
    std::vector<cublasLtMatmulHeuristicResult_t> results(count);
    int returned = 0;
    cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
        lt_handle(), p.op, p.a, p.b, p.c, p.c, pref, count, results.data(), &returned);
    if (status != CUBLAS_STATUS_SUCCESS) continue;
    for (int i = 0; i < returned; ++i) {
      bool match = results[i].state == CUBLAS_STATUS_SUCCESS;
      for (int j = 0; j < 8 && match; ++j)
        match = attrs[j] < 0 || attrs[j] == algo_attr(results[i].algo, keys[j]);
      if (match) {
        cublasLtMatmulPreferenceDestroy(pref);
        return results[i];
      }
    }
  }
  cublasLtMatmulPreferenceDestroy(pref);
  TORCH_CHECK(false, "selected cuBLASLt heuristic attributes are unavailable for this shape/runtime");
}

std::vector<torch::Tensor> lt_algo_from_attrs(int64_t M, int64_t N, int64_t K, std::vector<int64_t> attrs) {
  Plan& p = plan_for(M, N, K, false);
  auto result = lt_resolve_attrs(p, attrs);
  auto blob = torch::empty({(int64_t)sizeof(cublasLtMatmulAlgo_t)}, torch::kUInt8);
  std::memcpy(blob.data_ptr(), &result.algo, sizeof(result.algo));
  auto ws = torch::empty({1}, torch::kInt64);
  ws[0] = (int64_t)result.workspaceSize;
  return {blob, ws};
}

std::vector<int64_t> lt_algo_ids(int64_t max_ids) {
  std::vector<int> ids(max_ids);
  int returned = 0;
  LT_CHECK(cublasLtMatmulAlgoGetIds(lt_handle(), CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, (int)max_ids, ids.data(), &returned));
  return std::vector<int64_t>(ids.begin(), ids.begin() + returned);
}

// Capabilities of one algorithm id: {tiles, stages, (unused), [splitk_support, reduction_mask, swizzle_support, custom_max]}
std::vector<std::vector<int64_t>> lt_algo_caps(int64_t algo_id) {
  cublasLtMatmulAlgo_t algo;
  LT_CHECK(cublasLtMatmulAlgoInit(lt_handle(), CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, (int)algo_id, &algo));
  auto list = [&](cublasLtMatmulAlgoCapAttributes_t attr) {
    size_t written = 0;
    cublasLtMatmulAlgoCapGetAttribute(&algo, attr, nullptr, 0, &written);
    std::vector<int> buf(written / sizeof(int) + 1);
    if (written > 0) cublasLtMatmulAlgoCapGetAttribute(&algo, attr, buf.data(), written, &written);
    return std::vector<int64_t>(buf.begin(), buf.begin() + written / sizeof(int));
  };
  auto scalar = [&](cublasLtMatmulAlgoCapAttributes_t attr) {
    int v = -1; size_t written = 0;
    cublasLtMatmulAlgoCapGetAttribute(&algo, attr, &v, sizeof(v), &written);
    return (int64_t)v;
  };
  std::vector<int64_t> misc = {scalar(CUBLASLT_ALGO_CAP_SPLITK_SUPPORT), scalar(CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK),
                               scalar(CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT), scalar(CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX)};
  return {list(CUBLASLT_ALGO_CAP_TILE_IDS), list(CUBLASLT_ALGO_CAP_STAGES_IDS), std::vector<int64_t>{}, misc};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("lt_heuristics", &lt_heuristics, "cuBLASLt heuristic algorithm list");
  m.def("lt_matmul", &lt_matmul, "cuBLASLt matmul with explicit algorithm");
  m.def("lt_version", &lt_version, "cuBLASLt version");
  m.def("lt_algo_from_attrs", &lt_algo_from_attrs, "Resolve a complete heuristic result from config attributes");
  m.def("lt_algo_ids", &lt_algo_ids, "Algorithm ids for bf16/fp32-compute matmul");
  m.def("lt_algo_caps", &lt_algo_caps, "Capabilities of an algorithm id");
}
'''


def load(verbose=False):
    global _EXT
    if _EXT is not None:
        return _EXT
    from torch.utils.cpp_extension import load_inline
    root = Path(os.environ.get("QWEN38_KERNEL_CACHE", Path.home() / ".cache/qwen38_standalone")) / "lane2_ltgemm_v2"
    root.mkdir(parents=True, exist_ok=True)
    _EXT = load_inline(name="lane2_ltgemm_v2", cpp_sources=[_SRC], functions=None, extra_cflags=["-O2"],
                       extra_ldflags=["-lcublasLt", "-lcublas"], build_directory=str(root), verbose=verbose,
                       with_cuda=True)
    return _EXT


def heuristics(M, N, K, max_algos=32, max_workspace=32 * 1024 * 1024):
    ext = load()
    algos, ws, waves, attrs = ext.lt_heuristics(int(M), int(N), int(K), int(max_algos), int(max_workspace))
    names = ["id", "tile", "splitk", "reduction", "swizzle", "custom", "stages", "cluster"]
    return [{"algo": algos[i].clone(), "workspace": int(ws[i]), "waves": float(waves[i]),
             "attrs": {n: int(attrs[i][j]) for j, n in enumerate(names)}} for i in range(algos.shape[0])]


def matmul(x, w, algo=None, workspace=None, out=None, out_dtype=torch.bfloat16):
    ext = load()
    if out is None:
        out = torch.empty((x.shape[0], w.shape[0]), dtype=out_dtype, device=x.device)
    if algo is None:
        algo = torch.empty(0, dtype=torch.uint8)
    if workspace is None:
        workspace = torch.empty(0, dtype=torch.uint8, device=x.device)
    ext.lt_matmul(x, w, out, algo, workspace)
    return out


ATTR_NAMES = ["id", "tile", "splitk", "reduction", "swizzle", "custom", "stages", "cluster"]


def algo_from_attrs(M, N, K, attrs):
    """attrs: dict or list in ATTR_NAMES order; returns (blob, workspace_bytes)."""
    ext = load()
    if isinstance(attrs, dict):
        attrs = [int(attrs.get(n, -1)) for n in ATTR_NAMES]
    blob, ws = ext.lt_algo_from_attrs(int(M), int(N), int(K), [int(v) for v in attrs])
    return blob, int(ws[0])


def algo_ids(max_ids=64):
    return list(load().lt_algo_ids(int(max_ids)))


def algo_caps(algo_id):
    tiles, stages, clusters, misc = load().lt_algo_caps(int(algo_id))
    return {"tiles": list(tiles), "stages": list(stages), "clusters": list(clusters),
            "splitk_support": misc[0], "reduction_mask": misc[1], "swizzle_support": misc[2], "custom_max": misc[3]}
