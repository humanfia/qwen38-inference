"""cuBLASLt table dispatch in C++ to avoid Python planning/allocation overhead.

Uses the exact algorithms selected by lane2_dense_table.json. Unknown shapes and
unsupported input layouts use ATen linear. No weights or token-dependent results
are cached. Plans depend on shape; scratch allocations are private to each shape,
device and CUDA stream, and remain alive for captured graph replays.
"""

import json
import os
from pathlib import Path

if __package__:
    from . import lane2_ltgemm
else:
    import lane2_ltgemm


_EXTRA_SRC = r'''
#include <ATen/ops/linear.h>
#include <c10/cuda/CUDAGuard.h>

using DenseKey = std::tuple<int64_t, int64_t, int64_t>;
struct DenseEntry {
  std::vector<int64_t> attrs;
  cublasLtMatmulAlgo_t algo;
  bool ready = false;
  size_t workspace_bytes = 0;
  std::map<std::pair<int, uintptr_t>, torch::Tensor> workspaces;
};

static std::map<DenseKey, DenseEntry>& dense_entries() {
  // Lifetime is the loaded extension's process lifetime, including graph replays.
  static auto* entries = new std::map<DenseKey, DenseEntry>();
  return *entries;
}

void dense_configure(std::vector<std::vector<int64_t>> rows) {
  TORCH_CHECK(dense_entries().empty(), "dense table may only be configured once");
  for (auto& row : rows) {
    TORCH_CHECK(row.size() == 11, "row = M,N,K,id,tile,splitk,reduction,swizzle,custom,stages,cluster");
    auto key = std::make_tuple(row[0], row[1], row[2]);
    DenseEntry e;
    e.attrs.assign(row.begin() + 3, row.end());
    TORCH_CHECK(dense_entries().emplace(key, std::move(e)).second, "duplicate dense shape");
  }
}

static void dense_prepare(Plan& p, DenseEntry& e) {
  if (e.ready) return;
  auto result = lt_resolve_attrs(p, e.attrs);
  e.algo = result.algo;
  e.workspace_bytes = result.workspaceSize;
  e.ready = true;
}

torch::Tensor dense_linear(torch::Tensor x, torch::Tensor w) {
  if (!x.is_cuda() || x.dim() != 2 || w.dim() != 2 || !x.is_contiguous() ||
      !w.is_contiguous() || x.scalar_type() != torch::kBFloat16 ||
      w.scalar_type() != torch::kBFloat16 || x.device() != w.device() ||
      x.size(1) != w.size(1)) {
    return at::linear(x, w);
  }
  int64_t M = x.size(0), K = x.size(1), N = w.size(0);
  auto it = dense_entries().find(std::make_tuple(M, N, K));
  if (it == dense_entries().end()) return at::linear(x, w);
  c10::cuda::CUDAGuard guard(x.device());
  auto& entry = it->second;
  Plan& plan = plan_for(M, N, K, false);
  dense_prepare(plan, entry);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device());
  void* workspace_ptr = nullptr;
  if (entry.workspace_bytes > 0) {
    auto key = std::make_pair(x.get_device(), reinterpret_cast<uintptr_t>(stream));
    auto& workspace = entry.workspaces[key];
    if (!workspace.defined())
      workspace = torch::empty({(int64_t)entry.workspace_bytes}, x.options().dtype(torch::kUInt8));
    workspace_ptr = workspace.data_ptr();
  }
  auto output = torch::empty({M, N}, x.options());
  float alpha = 1.0f, beta = 0.0f;
  LT_CHECK(cublasLtMatmul(lt_handle(), plan.op, &alpha, w.data_ptr(), plan.a,
                         x.data_ptr(), plan.b, &beta, output.data_ptr(), plan.c,
                         output.data_ptr(), plan.c, &entry.algo,
                         workspace_ptr, entry.workspace_bytes, stream));
  return output;
}
'''

_EXT = None
_LINEAR = None


def load():
    global _EXT, _LINEAR
    if _EXT is None:
        from torch.utils.cpp_extension import load_inline
        root = Path(os.environ.get("QWEN38_KERNEL_CACHE", Path.home() / ".cache/qwen38_standalone")) / "lane2_fastlt_v1"
        root.mkdir(parents=True, exist_ok=True)
        binding = 'PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {'
        source = lane2_ltgemm._SRC.replace(binding, _EXTRA_SRC + '\n' + binding +
                                         '\n  m.def("dense_configure", &dense_configure);' +
                                         '\n  m.def("dense_linear", &dense_linear);')
        extension = load_inline(name="lane2_fastlt_v1", cpp_sources=[source], functions=None,
                                extra_cflags=["-O2"], extra_ldflags=["-lcublasLt", "-lcublas"],
                                build_directory=str(root), with_cuda=True)
        table = json.loads(Path(__file__).with_name('lane2_dense_table.json').read_text())
        rows = []
        for shape, entries in table.items():
            N, K = map(int, shape.split(','))
            for entry in entries:
                if entry['kind'] != 'lt':
                    raise ValueError('fastlt requires a cuBLASLt-only dispatch table')
                for M in range(entry['min_m'], entry['max_m'] + 1):
                    rows.append([M, N, K, *entry['attrs']])
        extension.dense_configure(rows)
        _EXT = extension
        _LINEAR = extension.dense_linear
    return _EXT


def linear(x, w):
    if _LINEAR is None:
        load()
    return _LINEAR(x, w)

