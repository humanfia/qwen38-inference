"""Whole-forward latency comparison over the complete 40-case, 98-call workload.

Unlike benchmark_forward_step.py, this harness keeps every original test case,
every ordered call (45 prefill + 53 decode), the original input representations
(flat lists, int32/int64 tensors, strided views, CUDA tensors), request_indices
selection/reordering, continuations, and return_all_logits. Both implementations
receive identical token ids, start from independent caches created with the
documented initial_capacity=64 contract, and reset their histories between
repetitions. Each timed call includes input conversion, metadata preparation,
required cache growth, all 64 layers, the requested logits, and cache updates.
Model loading, one-time compilation and input generation are outside timing.

Timers per call: synchronized wall latency (perf_counter around a synchronized
call) and CUDA-event latency (events enclosing the call). Kernel spans from
gpu_kernel_span.KernelTrace are optional diagnostics (--trace-repeats).
With --check, validate all calls in a separate pass before warmups; no reference
logits or comparison kernels are retained in recorded passes. Failures exit
nonzero. --order balanced exchanges arm positions and adjacent predecessors in
complete blocks, with the complete execution schedule retained in metadata.

Example (from the repository root, on the GPU sandbox):
    env PYTHONPATH=. python3 benchmark.py --model-path /path/to/Qwen3.8-27B --output runs/bench-x \
        --candidate candidate=solution.entry:run --repeats 5 --warmups 1 \
        --trace-repeats 1 --check
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from test_forward_step import TEST_CASES, VOCAB_SIZE, make_input  # noqa: E402
from solution.benchmark_order import make_orders, order_counts  # noqa: E402
from solution.benchmark_bindings import snapshot_model_bindings  # noqa: E402
from solution.benchmark_output import benchmark_startup  # noqa: E402

REFERENCE_DIR = REPO_ROOT / "reference"


def sha256_file(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def source_hashes():
    files = {"qwen38_inference.py": REPO_ROOT / "qwen38_inference.py",
             "test_forward_step.py": REPO_ROOT / "test_forward_step.py",
             "gpu_kernel_span.py": REPO_ROOT / "gpu_kernel_span.py",
             "benchmark.py": REPO_ROOT / "benchmark.py"}
    for path in sorted(p for p in (REPO_ROOT / "solution").iterdir() if p.is_file() and p.suffix in (".py", ".json", ".cu", ".sh")):
        files[f"solution/{path.name}"] = path
    if REFERENCE_DIR.is_dir():
        for path in sorted(REFERENCE_DIR.glob("*.py")):
            files[f"reference/{path.name}"] = path
    return {name: sha256_file(path) for name, path in files.items()}


def environment():
    import torch
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "torch_num_threads": torch.get_num_threads(),
            "cuda_available": torch.cuda.is_available(), "versions": {}}
    for package in ("torch", "triton", "flashinfer-python", "sglang-kernel", "apache-tvm-ffi", "nvidia-cutlass-dsl", "safetensors"):
        try:
            info["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            info["versions"][package] = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu"] = {"name": props.name, "capability": f"{props.major}.{props.minor}",
                       "total_memory_bytes": props.total_memory, "multi_processor_count": props.multi_processor_count}
        try:
            query = "name,driver_version,clocks.max.sm,clocks.max.memory,power.limit,memory.total"
            out = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
            info["gpu"]["nvidia_smi"] = out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            info["gpu"]["nvidia_smi"] = None
    try:
        cpu = [line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")]
        info["cpu"] = {"model": cpu[0] if cpu else None, "count": len(cpu), "affinity": len(os.sched_getaffinity(0)), "loadavg": os.getloadavg()}
    except OSError:
        info["cpu"] = None
    return info


def plan_case(case):
    """Enumerate the ordered calls of one case with their prior/append lengths."""
    batch_size = case.get("batch_size", len(case["steps"][0]["lengths"]))
    state = [0] * batch_size
    calls = []
    for index, step in enumerate(case["steps"]):
        slots = list(step["slots"]) if "slots" in step else list(range(batch_size))
        lengths = list(step["lengths"])
        priors = [state[s] for s in slots]
        is_decode = all(n == 1 and p > 0 for n, p in zip(lengths, priors))
        kwargs = {}
        if "slots" in step:
            kwargs["request_indices"] = list(step["slots"])
        if step.get("all_logits", False):
            kwargs["return_all_logits"] = True
        count = sum(lengths) if step.get("all_logits", False) else len(lengths)
        calls.append({"index": index, "kind": "decode" if is_decode else "prefill", "slots": slots,
                      "lengths": lengths, "priors": priors, "total_tokens": sum(lengths),
                      "max_context_after": max(p + n for p, n in zip(priors, lengths)),
                      "kwargs": kwargs, "output_rows": count, "form": case.get("form", "lists"),
                      "input_device": case.get("input_device", "cpu")})
        for s, n in zip(slots, lengths):
            state[s] += n
    return batch_size, calls


def percentile(sorted_values, p):
    return sorted_values[int(p * (len(sorted_values) - 1))]


def summarize(samples):
    ordered = sorted(samples)
    return {"count": len(samples), "median_ms": statistics.median(samples), "mean_ms": statistics.fmean(samples),
            "min_ms": ordered[0], "max_ms": ordered[-1], "p10_ms": percentile(ordered, 0.1), "p90_ms": percentile(ordered, 0.9),
            "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0}


def input_digest(value):
    import torch
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()[:16]


class Implementation:
    """A forward callable plus cache factory, both taking the reference model."""

    def __init__(self, label, spec, model):
        self.label = label
        self.spec = spec
        self.model = model
        if spec == "baseline":
            # Share loaded weights, but bind every method to the original class.
            # Each implementation creates its own cache using its own cache type.
            from reference.qwen38_inference import Qwen38 as ReferenceModel
            self.model = object.__new__(ReferenceModel)
            self.model.__dict__ = vars(model).copy()
            self.forward = self.model.forward_step
            self.new_cache = self.model.new_cache
            self.module = None
        else:
            module_name, function_name = spec.split(":", 1)
            self.module = importlib.import_module(module_name)
            function = getattr(self.module, function_name)
            self.forward = lambda ids, cache, **kw: function(model, ids, cache, **kw)
            factory = getattr(self.module, "new_cache", None)
            if factory is None:
                self.new_cache = model.new_cache
            else:
                self.new_cache = lambda batch_size, initial_capacity=64: factory(model, batch_size, initial_capacity=initial_capacity)


def parse_candidates(values):
    implementations = [("baseline", "baseline")]
    for value in values or []:
        if "=" in value.split(":", 1)[0]:
            label, spec = value.split("=", 1)
        else:
            label, spec = value.replace(":", "_"), value
        implementations.append((label, spec))
    labels = [label for label, _ in implementations]
    if len(set(labels)) != len(labels):
        raise SystemExit("Duplicate implementation labels")
    return implementations


def compare_logits(logits, expected, atol, rtol):
    """Tolerance is the acceptance criterion; bit equality is diagnostic only."""
    import torch
    finite = bool(torch.isfinite(logits).all().item())
    diff = (logits - expected).abs()
    within = bool((diff <= atol + rtol * expected.abs()).all().item()) and finite
    return {"role": "candidate", "finite": finite, "max_abs": diff.max().item(),
            "within_tolerance": within,
            "bitwise_equal": bool(torch.equal(logits.view(torch.int32), expected.view(torch.int32)))}


def validate_case(impls, case, calls, atol, rtol):
    """Untimed baseline-first pass, with independent caches and owned oracles."""
    import torch
    expected = {}
    records = {}
    if impls[0].label != "baseline":
        raise ValueError("Validation requires the baseline first")
    for impl in impls:
        def checker(case, call, logits):
            if impl.label == "baseline":
                if not bool(torch.isfinite(logits).all().item()):
                    raise AssertionError(f"Non-finite baseline: {case['name']} call {call['index']}")
                expected[call["index"]] = logits.detach().clone()
                return {"role": "reference"}
            if call["index"] not in expected:
                raise AssertionError("Missing reference logits")
            result = compare_logits(logits, expected[call["index"]], atol, rtol)
            if not result["within_tolerance"]:
                raise AssertionError(f"{impl.label} failed {case['name']} call {call['index']}: {result}")
            return result
        sequence = run_case_sequence(impl, case, calls, lambda fn: (fn(), {}), checker=checker)
        records[impl.label] = [r["check"] for r in sequence]
        print(f"check {impl.label} {case['name']}: calls={len(sequence)} passed", flush=True)
    torch.cuda.synchronize()
    return records


def run_case_sequence(impl, case, calls, timed, checker=None, tracer=None):
    """Run one case's ordered calls on a fresh cache; return per-call records."""
    import torch
    batch_size = case.get("batch_size", len(case["steps"][0]["lengths"]))
    cache = impl.new_cache(batch_size, initial_capacity=64)
    records = []
    for call in calls:
        inputs = make_input(case, case["steps"][call["index"]], call["index"])
        kwargs = {key: list(value) if isinstance(value, list) else value for key, value in call["kwargs"].items()}
        counters = getattr(impl.module, "benchmark_counters", None)
        before = counters(impl.model, impl.spec) if counters is not None else {}
        if tracer is not None:
            torch.cuda.synchronize()
            logits = tracer.call(lambda rows: impl.forward(rows, cache, **kwargs), inputs)
            record = {}
        else:
            logits, record = timed(lambda: impl.forward(inputs, cache, **kwargs))
        if counters is not None:
            after = counters(impl.model, impl.spec)
            record["diagnostics"] = {name: value - before.get(name, 0) for name, value in after.items()}
        # Cheap sanity checks outside timing catch missing or truncated computation.
        assert tuple(logits.shape) == (call["output_rows"], VOCAB_SIZE), (impl.label, case["name"], call["index"], tuple(logits.shape))
        assert logits.dtype == torch.float32, logits.dtype
        if checker is not None:
            record["check"] = checker(case, call, logits)
        records.append(record)
        del logits
    del cache
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", type=Path, required=True, help="Directory for JSON results and traces")
    parser.add_argument("--candidate", action="append", default=[], help="label=module:function with signature (model, input_ids, cache, **kwargs); repeatable")
    parser.add_argument("--repeats", type=int, default=5, help="Recorded repetitions of the whole suite per implementation")
    parser.add_argument("--warmups", type=int, default=1, help="Unrecorded full-suite passes per implementation (compilation, autotuning)")
    parser.add_argument("--trace-repeats", type=int, default=0, help="Extra passes under the CUDA kernel tracer (diagnostic kernel spans)")
    parser.add_argument("--case", action="append", help="Restrict to named cases (iteration only)")
    parser.add_argument("--check", action="store_true", help="Check every candidate call in a separate untimed pass before warmups; fail on any mismatch")
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--max-context", type=int, default=8192, help="Model context capacity, as in the original test script")
    parser.add_argument("--no-alternate", action="store_true", help="Keep implementation order fixed instead of reversing it on odd repetitions")
    parser.add_argument("--order", choices=("alternate", "fixed", "balanced"), default="alternate")
    parser.add_argument("--order-seed", type=int, default=10010, help="Reproducible balanced-block row and arm permutations")
    parser.add_argument("--plan-output", type=Path, help="Write the workload and exact schedule before model loading (also with --dry-run)")
    parser.add_argument("--dry-run", action="store_true", help="Print the call plan and exit without loading the model")
    args = parser.parse_args()
    if args.warmups < 0 or args.trace_repeats < 0:
        parser.error("warmups and trace-repeats must be nonnegative")
    if not all(math.isfinite(v) and v >= 0 for v in (args.atol, args.rtol)):
        parser.error("atol and rtol must be finite and nonnegative")
    if args.no_alternate:
        if args.order == "balanced":
            parser.error("--no-alternate conflicts with --order balanced")
        args.order = "fixed"

    cases = TEST_CASES if args.case is None else [c for c in TEST_CASES if c["name"] in args.case]
    if args.case and set(args.case) - {c["name"] for c in cases}:
        parser.error("Unknown case name")
    plans = {case["name"]: plan_case(case) for case in cases}
    total_calls = sum(len(calls) for _, calls in plans.values())
    kinds = {"prefill": 0, "decode": 0}
    for _, calls in plans.values():
        for call in calls:
            kinds[call["kind"]] += 1
    implementations = parse_candidates(args.candidate or ["entry=solution.entry:run"])
    labels = [label for label, _ in implementations]
    try:
        orders, block_size = make_orders(labels, [c["name"] for c in cases], args.repeats, args.order, args.order_seed)
    except ValueError as error:
        parser.error(str(error))
    schedule = {"mode": args.order, "seed": args.order_seed, "block_size": block_size,
                "orders": orders, "counts": {name: order_counts(labels, rows) for name, rows in orders.items()},
                "balance_scope": "positions and adjacent predecessors within each case; excludes boundaries between cases/repetitions"}
    if args.plan_output:
        args.plan_output.parent.mkdir(parents=True, exist_ok=True)
        args.plan_output.write_text(json.dumps({"cases": plans, "counts": kinds, "schedule": schedule}, indent=1) + "\n")
    print(f"cases={len(cases)} calls={total_calls} prefill={kinds['prefill']} decode={kinds['decode']} implementations={[l for l, _ in implementations]}", flush=True)
    if args.dry_run:
        for name, (batch_size, calls) in plans.items():
            for call in calls:
                print(name, batch_size, json.dumps({k: v for k, v in call.items() if k not in ("slots",)}))
        return

    with benchmark_startup(args.output) as previous_attempt:
        import torch
        torch.set_num_threads(1)
        from qwen38_inference import Qwen38
        load_start = time.perf_counter()
        model = Qwen38(args.model_path, max_context=args.max_context)
        load_seconds = time.perf_counter() - load_start
        impls = [Implementation(label, spec, model) for label, spec in implementations]
    started = _dt.datetime.now(_dt.timezone.utc).isoformat()
    meta = {"command": " ".join(sys.argv), "started_utc": started, "status": "running", "completed_repeats": 0,
            "previous_attempt": previous_attempt,
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "model_path": args.model_path, "model_load_seconds": load_seconds, "environment": environment(), "source_hashes": source_hashes(),
            "implementations": [{"label": label, "spec": spec} for label, spec in implementations],
            "protocol": {"caches": "independent per implementation, new_cache(batch_size, initial_capacity=64) per case per repetition",
                         "timing": "torch.cuda.synchronize(); perf_counter and CUDA event around forward; synchronize; includes input conversion, metadata, cache growth, all layers, logits, cache update",
                         "order": args.order, "schedule": schedule,
                         "check": "separate untimed pass before warmups" if args.check else "disabled",
                         "suite_estimator": "sum of per-call medians; cases also retain raw per-repetition sequence totals",
                         "warmups": args.warmups, "repeats": args.repeats, "trace_repeats": args.trace_repeats}}

    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def timed(fn):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        start_event.record()
        result = fn()
        t_issued = time.perf_counter()
        end_event.record()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        # cpu_issue_ms: host time until forward returned (all work issued); wall_ms: until the GPU finished.
        return result, {"wall_ms": (t1 - t0) * 1e3, "event_ms": start_event.elapsed_time(end_event), "cpu_issue_ms": (t_issued - t0) * 1e3}

    # results[case][impl] -> list over repetitions of list over calls of records
    results = {case["name"]: {impl.label: [] for impl in impls} for case in cases}
    meta["validation"] = {}

    failed = {}
    meta["failed_implementations"] = failed

    def fail(label, case, phase, error):
        import traceback
        failed[label] = {"case": case, "repeat": phase, "error": repr(error)[:2000],
                         "traceback": traceback.format_exc()[-4000:]}
        meta["status"] = "failed"
        meta["engine_bindings"] = snapshot_model_bindings(model)
        (args.output / "failure.json").write_text(json.dumps(meta, indent=1) + "\n")
        (args.output / "status.json").write_text(json.dumps({"status": "failed", "failures": failed}, indent=1) + "\n")
        print(f"FAILED {label} on {case} ({phase}): {error!r}", flush=True)

    (args.output / "status.json").write_text('{"status": "running"}\n')
    with torch.inference_mode():
        if args.check:
            for case in cases:
                try:
                    meta["validation"][case["name"]] = validate_case(impls, case, plans[case["name"]][1], args.atol, args.rtol)
                except Exception as error:
                    fail("validation", case["name"], "validation", error)
                    raise
            (args.output / "validation.json").write_text(json.dumps(meta["validation"], indent=1) + "\n")
        for warmup in range(args.warmups):
            for case in cases:
                for impl in impls:
                    if impl.label in failed:
                        continue
                    t0 = time.perf_counter()
                    try:
                        run_case_sequence(impl, case, plans[case["name"]][1], timed)
                    except Exception as error:  # noqa: BLE001
                        fail(impl.label, case["name"], "warmup", error)
                        raise
                    print(f"warmup {warmup} {impl.label} {case['name']}: {time.perf_counter() - t0:.2f}s", flush=True)
        for repeat in range(args.repeats):
            by_label = {impl.label: impl for impl in impls}
            for case in cases:
                order = [by_label[label] for label in orders[case["name"]][repeat]]
                for position, impl in enumerate(order):
                    if impl.label in failed:
                        continue
                    try:
                        records = run_case_sequence(impl, case, plans[case["name"]][1], timed)
                    except Exception as error:  # noqa: BLE001 - preserve the failure and exit nonzero
                        fail(impl.label, case["name"], repeat, error)
                        raise
                    for record in records:
                        record["order_position"] = position
                    results[case["name"]][impl.label].append(records)
                    walls = [r["wall_ms"] for r in records]
                    print(f"repeat {repeat} {impl.label} {case['name']}: calls={len(records)} sum={sum(walls):.2f}ms first={walls[0]:.2f}ms", flush=True)
            meta["completed_repeats"] = repeat + 1
            write_report(args, meta, cases, plans, impls, results, {})
        traces = {case["name"]: {impl.label: [] for impl in impls} for case in cases}
        if args.trace_repeats > 0:
            from gpu_kernel_span import KernelTrace
            tracer = KernelTrace()
            for repeat in range(args.trace_repeats):
                for case in cases:
                    calls = plans[case["name"]][1]
                    for impl in impls:
                        torch.cuda.synchronize()
                        with tracer.profile() as profile:
                            run_case_sequence(impl, case, calls, timed, tracer=tracer)
                        trace_file = args.output / "traces" / impl.label / f"{case['name']}_{repeat}.json.gz"
                        summaries = tracer.extract(profile, trace_file, len(calls))
                        for summary in summaries:
                            summary["file"] = str(trace_file.relative_to(args.output))
                        traces[case["name"]][impl.label].append(summaries)
                        print(f"trace {repeat} {impl.label} {case['name']}: span={[round(s['kernel_span_ms'], 2) for s in summaries]} kernels={[s['kernel_count'] for s in summaries]}", flush=True)
        meta["status"] = "complete"
        write_report(args, meta, cases, plans, impls, results, traces)
        (args.output / "status.json").write_text(json.dumps({"status": "complete", "repeats": args.repeats,
                "cases": len(cases), "calls": total_calls, "check": args.check}) + "\n")
    print("DONE", flush=True)


def geometric_mean(values):
    values = [v for v in values if v > 0]
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else float("nan")


def write_report(args, meta, cases, plans, impls, results, traces):
    # One shared model, possibly many engines. Snapshot only at the existing
    # report boundary; per-call counter deltas and the timed path stay unchanged.
    meta["engine_bindings"] = snapshot_model_bindings(next((impl.model for impl in impls if impl.spec != "baseline"), impls[0].model))
    # Preserve absolute self-check flags as well as per-call counter deltas. A
    # constant true flag has zero delta and otherwise cannot prove activation.
    meta["implementation_counters"] = {}
    for impl in impls:
        counters = getattr(impl.module, "benchmark_counters", None)
        if counters is not None:
            meta["implementation_counters"][impl.label] = counters(impl.model, impl.spec)
    labels = [impl.label for impl in impls]
    report = {"meta": dict(meta, finished_utc=_dt.datetime.now(_dt.timezone.utc).isoformat()), "cases": [], "summary": {}}
    totals = {label: {"prefill": 0.0, "decode": 0.0, "prefill_event": 0.0, "decode_event": 0.0} for label in labels}
    case_totals = {label: {} for label in labels}
    per_call_speedups = {label: [] for label in labels}
    paired_wins = {label: [0, 0] for label in labels}
    checks = {label: {"calls": 0, "within_tolerance": 0, "bitwise_equal": 0, "max_abs": 0.0} for label in labels}
    regressions = {label: [] for label in labels}
    for case in cases:
        name = case["name"]
        batch_size, calls = plans[name]
        entry = {"name": name, "batch_size": batch_size, "distribution": case["distribution"], "form": case.get("form", "lists"),
                 "input_device": case.get("input_device", "cpu"), "calls": []}
        reps = {label: results[name][label] for label in labels if results[name][label]}
        if labels[0] not in reps:
            continue
        for index, call in enumerate(calls):
            call_entry = {key: call[key] for key in ("index", "kind", "slots", "lengths", "priors", "total_tokens", "max_context_after", "kwargs", "output_rows")}
            call_entry["per_implementation"] = {}
            for label in reps:
                walls = [rep[index]["wall_ms"] for rep in reps[label]]
                events = [rep[index]["event_ms"] for rep in reps[label]]
                issues = [rep[index].get("cpu_issue_ms") for rep in reps[label] if rep[index].get("cpu_issue_ms") is not None]
                item = {"wall_ms": walls, "event_ms": events, "wall": summarize(walls), "event": summarize(events)}
                item["order_positions"] = [rep[index]["order_position"] for rep in reps[label] if "order_position" in rep[index]]
                diagnostics = [rep[index]["diagnostics"] for rep in reps[label] if "diagnostics" in rep[index]]
                if diagnostics:
                    item["diagnostics"] = diagnostics
                if issues:
                    item["cpu_issue_ms"] = issues
                    item["cpu_issue"] = summarize(issues)
                validation = meta.get("validation", {}).get(name, {}).get(label, [])
                check_records = [validation[index]] if validation and validation[index].get("role") == "candidate" else []
                if check_records:
                    item["check"] = check_records[0]
                    checks[label]["calls"] += 1
                    checks[label]["within_tolerance"] += int(check_records[0]["within_tolerance"])
                    checks[label]["bitwise_equal"] += int(check_records[0]["bitwise_equal"])
                    checks[label]["max_abs"] = max(checks[label]["max_abs"], check_records[0]["max_abs"])
                trace_records = [call_summary for rep in traces.get(name, {}).get(label, []) for call_summary in [rep[index]]]
                if trace_records:
                    item["kernel_span_ms"] = [t["kernel_span_ms"] for t in trace_records]
                    item["kernel_sum_ms"] = [t["kernel_sum_ms"] for t in trace_records]
                    item["kernel_count"] = [t["kernel_count"] for t in trace_records]
                    item["trace_files"] = sorted({t["file"] for t in trace_records})
                call_entry["per_implementation"][label] = item
                median = item["wall"]["median_ms"]
                totals[label][call["kind"]] += median
                totals[label][call["kind"] + "_event"] += item["event"]["median_ms"]
                case_totals[label][name] = case_totals[label].get(name, 0.0) + median
            base = call_entry["per_implementation"][labels[0]]
            for label in labels[1:]:
                if label not in reps:
                    continue
                cand = call_entry["per_implementation"][label]
                speedup = base["wall"]["median_ms"] / cand["wall"]["median_ms"]
                cand["speedup_vs_baseline"] = speedup
                per_call_speedups[label].append(speedup)
                pairs = list(zip(base["wall_ms"], cand["wall_ms"]))
                wins = sum(1 for b, c in pairs if c < b)
                paired_wins[label][0] += wins
                paired_wins[label][1] += len(pairs)
                cand["paired_faster_fraction"] = wins / len(pairs) if pairs else None
                # A regression is a slower median outside the baseline's own spread.
                if cand["wall"]["median_ms"] > base["wall"]["max_ms"] and speedup < 0.98:
                    regressions[label].append({"case": name, "call": index, "kind": call["kind"], "speedup": speedup,
                                               "baseline_median_ms": base["wall"]["median_ms"], "candidate_median_ms": cand["wall"]["median_ms"]})
            entry["calls"].append(call_entry)
        entry["sequence_ms"] = {label: case_totals[label].get(name) for label in labels}
        entry["sequence_repetitions_ms"] = {label: [sum(r["wall_ms"] for r in rep) for rep in rows]
                                             for label, rows in reps.items()}
        entry["sequence_repetitions_event_ms"] = {label: [sum(r["event_ms"] for r in rep) for rep in rows]
                                                   for label, rows in reps.items()}
        for label in labels[1:]:
            if entry["sequence_ms"][labels[0]] and entry["sequence_ms"].get(label):
                entry["sequence_speedup"] = entry.get("sequence_speedup", {})
                entry["sequence_speedup"][label] = entry["sequence_ms"][labels[0]] / entry["sequence_ms"][label]
        report["cases"].append(entry)
    for label in labels:
        if label != labels[0] and not case_totals[label]:
            report["summary"][label] = {"failed": meta.get("failed_implementations", {}).get(label)}
            continue
        report["summary"][label] = {"total_suite_ms": totals[label]["prefill"] + totals[label]["decode"],
                                    "prefill_total_ms": totals[label]["prefill"], "decode_total_ms": totals[label]["decode"],
                                    "total_suite_event_ms": totals[label]["prefill_event"] + totals[label]["decode_event"],
                                    "case_sequence_ms": case_totals[label]}
        if label != labels[0]:
            base = report["summary"][labels[0]]
            cand = report["summary"][label]
            case_speedups = {name: base["case_sequence_ms"][name] / cand["case_sequence_ms"][name] for name in cand["case_sequence_ms"] if cand["case_sequence_ms"][name]}
            cand["speedup_vs_baseline"] = {
                "total_suite": base["total_suite_ms"] / cand["total_suite_ms"] if cand["total_suite_ms"] else None,
                "prefill_total": base["prefill_total_ms"] / cand["prefill_total_ms"] if cand["prefill_total_ms"] else None,
                "decode_total": base["decode_total_ms"] / cand["decode_total_ms"] if cand["decode_total_ms"] else None,
                "geomean_case_sequence": geometric_mean(case_speedups.values()),
                "geomean_per_call": geometric_mean(per_call_speedups[label]),
                "per_call_min": min(per_call_speedups[label]) if per_call_speedups[label] else None,
                "per_call_max": max(per_call_speedups[label]) if per_call_speedups[label] else None,
                "paired_faster_fraction": paired_wins[label][0] / paired_wins[label][1] if paired_wins[label][1] else None,
                "case_sequence_speedups": case_speedups,
            }
            cand["regressions"] = regressions[label]
            cand["checks"] = checks[label]
    (args.output / "bench.json").write_text(json.dumps(report, indent=1) + "\n")
    (args.output / "summary.md").write_text(render_markdown(report))


def render_markdown(report):
    labels = [impl["label"] for impl in report["meta"]["implementations"]]
    lines = [f"# Whole-suite latency: {', '.join(labels)}", "", f"Command: `{report['meta']['command']}`", ""]
    env = report["meta"]["environment"]
    gpu = env.get("gpu", {})
    lines.append(f"GPU: {gpu.get('name')} (cc {gpu.get('capability')}); torch {env['torch']}; triton {env['versions'].get('triton')}; flashinfer {env['versions'].get('flashinfer-python')}; repeats={report['meta']['protocol']['repeats']} warmups={report['meta']['protocol']['warmups']}")
    lines.append("")
    lines.append("| implementation | total suite ms | prefill ms | decode ms | total speedup | prefill speedup | decode speedup | geomean case speedup | paired faster |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for label in labels:
        s = report["summary"].get(label)
        if not s:
            continue
        sp = s.get("speedup_vs_baseline", {})
        fmt = lambda v: f"{v:.4f}" if isinstance(v, (int, float)) and v is not None else "-"
        lines.append(f"| {label} | {s['total_suite_ms']:.1f} | {s['prefill_total_ms']:.1f} | {s['decode_total_ms']:.1f} | {fmt(sp.get('total_suite'))} | {fmt(sp.get('prefill_total'))} | {fmt(sp.get('decode_total'))} | {fmt(sp.get('geomean_case_sequence'))} | {fmt(sp.get('paired_faster_fraction'))} |")
    lines.append("")
    lines.append("| case | call | kind | B | tokens | " + " | ".join(f"{label} median ms (min..max)" for label in labels) + " | speedup |")
    lines.append("| --- | ---: | --- | ---: | ---: | " + " | ".join("---:" for _ in labels) + " | ---: |")
    for case in report["cases"]:
        for call in case["calls"]:
            cells = []
            for label in labels:
                w = call["per_implementation"][label]["wall"]
                cells.append(f"{w['median_ms']:.2f} ({w['min_ms']:.2f}..{w['max_ms']:.2f})")
            speedups = [call["per_implementation"][label].get("speedup_vs_baseline") for label in labels[1:]]
            lines.append(f"| {case['name']} | {call['index']} | {call['kind']} | {case['batch_size']} | {call['total_tokens']} | " + " | ".join(cells) + " | " + ", ".join(f"{s:.3f}" for s in speedups if s) + " |")
    lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
