"""Extract first-kernel-start to last-kernel-end spans from CUDA activities."""
import gzip
import json
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def benchmark_span_start(ptr):
    tl.store(ptr, 1)


@triton.jit
def benchmark_span_end(ptr):
    tl.store(ptr, 2)


class KernelTrace:
    def __init__(self):
        self.marker = torch.empty((), dtype=torch.int32, device="cuda")
        benchmark_span_start[(1,)](self.marker)
        benchmark_span_end[(1,)](self.marker)
        torch.cuda.synchronize()

    def call(self, function, rows):
        torch.cuda.synchronize()
        benchmark_span_start[(1,)](self.marker)
        logits = function(rows)
        benchmark_span_end[(1,)](self.marker)
        torch.cuda.synchronize()
        return logits

    def profile(self):
        return torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False, with_stack=False, profile_memory=False,
            experimental_config=torch._C._profiler._ExperimentalConfig(
                disable_external_correlation=True, trace_only=True),
        )

    def extract(self, profiler, path, expected_calls):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".chrome.json")
        profiler.export_chrome_trace(str(temporary))
        document = json.loads(temporary.read_text())
        kernels = sorted((e for e in document["traceEvents"] if e.get("cat") == "kernel"),
                         key=lambda e: e["ts"])
        calls, active = [], None
        for event in kernels:
            name = event["name"]
            if "benchmark_span_start" in name:
                assert active is None, "Nested or missing marker"
                active = []
            elif "benchmark_span_end" in name:
                assert active, "No kernels between markers"
                first = min(e["start_us"] for e in active)
                last = max(e["end_us"] for e in active)
                calls.append({"first_kernel_start_us": first, "last_kernel_end_us": last,
                              "kernel_span_ms": (last - first) / 1000,
                              "kernel_sum_ms": sum(e["end_us"] - e["start_us"] for e in active) / 1000,
                              "kernel_count": len(active), "kernels": active})
                active = None
            elif active is not None:
                active.append({"name": name, "start_us": event["ts"],
                               "end_us": event["ts"] + event["dur"],
                               "stream": event.get("args", {}).get("stream")})
        assert active is None and len(calls) == expected_calls, (len(calls), expected_calls)
        with gzip.open(path, "wt") as handle:
            json.dump({"units": "microseconds", "source": "CUPTI CUDA kernel activities via PyTorch/Kineto",
                       "scope": "markers excluded; memcpy/memset excluded as endpoints; intervening gaps included",
                       "calls": calls}, handle, separators=(",", ":"))
        temporary.unlink()
        return [{key: value for key, value in call.items() if key != "kernels"} for call in calls]


def self_check(output):
    trace = KernelTrace()
    x = torch.ones(1024, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = (x + 1).relu()
    graph.replay()
    torch.cuda.synchronize()
    with trace.profile() as profile:
        trace.call(lambda unused: (x + 1).relu(), None)
        trace.call(lambda unused: graph.replay(), None)
    calls = trace.extract(profile, output, 2)
    assert all(call["kernel_count"] >= 2 and call["kernel_span_ms"] > 0 for call in calls)
    print("KERNEL_TRACE_SELF_CHECK", json.dumps(calls), flush=True)


if __name__ == "__main__":
    import sys
    self_check(sys.argv[1])
