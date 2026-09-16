"""Data-driven comparisons of two forward_step implementations, on one GPU.

Edit TEST_CASES below, or pass your own cases to run_tests(). Each step consumes
tokens and updates a separate cache for each implementation. No generate() calls.
The CLI candidate has signature candidate(model, input_ids, kv_cache, **kwargs).
run_tests() also accepts bound methods and separate cache factories directly.
"""
import argparse
import importlib
import json
import math
from pathlib import Path

import torch


VOCAB_SIZE = 248320

# lengths describes the unpadded token count of each active request in one call.
# The first step creates the request slots; subsequent steps reuse their history.
# All random data uses private, seeded CPU generators, independently per caller.
TEST_CASES = [
    {"name": "zeros_256", "distribution": "zeros", "form": "flat", "steps": [{"lengths": [256]}, {"lengths": [1]}]},
    {"name": "maximum_512", "distribution": "maximum", "form": "tensor32", "steps": [{"lengths": [512]}, {"lengths": [1]}]},
    {"name": "flat_integer_tensor", "distribution": "uniform", "form": "flat_tensor64", "steps": [{"lengths": [1024]}, {"lengths": [1]}]},
    {"name": "constant_768", "distribution": "constant", "value": 19, "steps": [{"lengths": [768]}, {"lengths": [1]}]},
    {"name": "alternating_1536", "distribution": "alternating", "steps": [{"lengths": [1536]}, {"lengths": [1]}]},
    {"name": "ramp_3072", "distribution": "ramp", "steps": [{"lengths": [3072]}, {"lengths": [1]}]},
    {"name": "uniform_4096", "distribution": "uniform", "seed": 11, "steps": [{"lengths": [4096]}, {"lengths": [1]}]},
    {"name": "batch_2_tensor64", "distribution": "uniform", "form": "tensor64", "seed": 12, "steps": [{"lengths": [512, 512]}, {"lengths": [1, 1]}]},
    {"name": "batch_4_tensor32", "distribution": "low_uniform", "form": "tensor32", "steps": [{"lengths": [1024] * 4}, {"lengths": [1] * 4}]},
    {"name": "batch_8", "distribution": "alternating", "steps": [{"lengths": [512] * 8}, {"lengths": [1] * 8}]},
    {"name": "batch_16_length_2048", "distribution": "uniform", "seed": 25, "steps": [{"lengths": [2048] * 16}, {"lengths": [1] * 16}]},
    {"name": "batch_32_length_1024", "distribution": "uniform", "seed": 26, "steps": [{"lengths": [1024] * 32}, {"lengths": [1] * 32}]},
    {"name": "batch_64_length_512", "distribution": "uniform", "seed": 27, "steps": [{"lengths": [512] * 64}, {"lengths": [1] * 64}]},
    {"name": "batch_32_ragged", "distribution": "uniform", "seed": 28, "steps": [{"lengths": [256, 512, 1024, 2048] * 8}, {"lengths": [1] * 32}]},
    {"name": "batch_3_length_2053", "distribution": "uniform", "seed": 29, "steps": [{"lengths": [2053] * 3}, {"lengths": [1] * 3}]},
    {"name": "batch_7_length_1000", "distribution": "uniform", "seed": 30, "steps": [{"lengths": [1000] * 7}, {"lengths": [1] * 7}]},
    {"name": "batch_17_length_769", "distribution": "uniform", "seed": 31, "steps": [{"lengths": [769] * 17}, {"lengths": [1] * 17}]},
    {"name": "batch_33_length_513", "distribution": "uniform", "seed": 32, "steps": [{"lengths": [513] * 33}, {"lengths": [1] * 33}]},
    {"name": "batch_63_length_257", "distribution": "uniform", "seed": 33, "steps": [{"lengths": [257] * 63}, {"lengths": [1] * 63}]},
    {"name": "ragged_lengths", "distribution": "uniform", "seed": 13, "steps": [{"lengths": [128, 255, 512, 1023, 2048, 4096]}, {"lengths": [1] * 6}]},
    {"name": "page_1023_to_1025", "distribution": "uniform", "seed": 14, "steps": [{"lengths": [1023]}, {"lengths": [1]}, {"lengths": [1]}, {"lengths": [1]}]},
    {"name": "page_2048_to_2049", "distribution": "maximum", "steps": [{"lengths": [2048]}, {"lengths": [1]}]},
    {"name": "page_4097", "distribution": "zeros", "steps": [{"lengths": [4097]}, {"lengths": [1]}]},
    {"name": "length_255_256_257", "distribution": "ramp", "steps": [{"lengths": [255, 256, 257]}, {"lengths": [1, 1, 1]}]},
    {"name": "length_511_512_513", "distribution": "uniform", "seed": 15, "steps": [{"lengths": [511, 512, 513]}, {"lengths": [1, 1, 1]}]},
    {"name": "length_1023_1024_1025", "distribution": "uniform", "seed": 16, "steps": [{"lengths": [1023, 1024, 1025]}, {"lengths": [1, 1, 1]}]},
    {"name": "length_2047_2048_2049", "distribution": "uniform", "seed": 23, "steps": [{"lengths": [2047, 2048, 2049]}, {"lengths": [1, 1, 1]}]},
    {"name": "length_4095_4096_4097", "distribution": "uniform", "seed": 24, "steps": [{"lengths": [4095, 4096, 4097]}, {"lengths": [1, 1, 1]}]},
    {"name": "normal_token_ids", "distribution": "normal", "seed": 17, "steps": [{"lengths": [512, 1536]}, {"lengths": [1, 1]}]},
    {"name": "zipf_token_ids", "distribution": "zipf", "seed": 18, "steps": [{"lengths": [1024, 3072]}, {"lengths": [1, 1]}]},
    {"name": "rare_maximum_ids", "distribution": "sparse_extremes", "steps": [{"lengths": [2049, 1025]}, {"lengths": [1, 1]}]},
    {"name": "special_token_ids", "distribution": "cycle", "values": [0, 198, 248044, 248045, 248046, 248319], "steps": [{"lengths": [768, 2048]}, {"lengths": [1, 1]}]},
    {"name": "shared_prefix", "distribution": "shared_prefix", "prefix_length": 256, "seed": 19, "steps": [{"lengths": [1024, 2048, 4096]}, {"lengths": [1, 1, 1]}]},
    {"name": "strided_integer_tensor", "distribution": "uniform", "form": "strided64", "steps": [{"lengths": [1024, 1024]}, {"lengths": [1, 1]}]},
    {"name": "cuda_integer_tensor", "distribution": "uniform", "form": "tensor64", "input_device": "cuda", "steps": [{"lengths": [2048, 2048]}, {"lengths": [1, 1]}]},
    {"name": "chunked_prefill", "distribution": "uniform", "seed": 20, "steps": [{"lengths": [512, 1024, 2048]}, {"lengths": [1024, 512, 1024]}, {"lengths": [512, 2048, 1024]}, {"lengths": [1, 1, 1]}]},
    {"name": "select_and_reorder", "distribution": "uniform", "seed": 21, "steps": [{"lengths": [1023, 2048, 4095]}, {"lengths": [1, 1], "slots": [2, 0]}, {"lengths": [1], "slots": [1]}, {"lengths": [512, 1024, 256], "slots": [2, 1, 0]}]},
    {"name": "new_and_existing_slots", "batch_size": 3, "distribution": "uniform", "steps": [{"lengths": [512], "slots": [1]}, {"lengths": [1, 1024], "slots": [1, 0]}, {"lengths": [2048, 1, 1], "slots": [2, 0, 1]}]},
    {"name": "all_position_logits", "distribution": "uniform", "steps": [{"lengths": [256, 128], "all_logits": True}, {"lengths": [1, 1], "all_logits": True}]},
    {"name": "long_decode_history", "distribution": "uniform", "seed": 22, "steps": [{"lengths": [4093, 2045]}] + [{"lengths": [1, 1]} for _ in range(12)]},
]


def make_input(case, step, step_index):
    """Generate legal integer token IDs, not arbitrary floating-point inputs."""
    lengths = step["lengths"]
    assert lengths and all(type(n) is int and n > 0 for n in lengths)
    rng = torch.Generator(device="cpu").manual_seed(case.get("seed", 1234) + step_index)
    n = sum(lengths)
    pattern = case["distribution"]
    if pattern in ("zeros", "maximum", "constant"):
        value = {"zeros": 0, "maximum": VOCAB_SIZE - 1}.get(pattern, case.get("value", 0))
        ids = torch.full((n,), value, dtype=torch.int64)
    elif pattern == "alternating":
        ids = (torch.arange(n) % 2) * (VOCAB_SIZE - 1)
    elif pattern == "ramp":
        ids = (torch.arange(n) + VOCAB_SIZE - 8) % VOCAB_SIZE
    elif pattern in ("uniform", "low_uniform", "shared_prefix"):
        ids = torch.randint(256 if pattern == "low_uniform" else VOCAB_SIZE, (n,), generator=rng)
    elif pattern == "normal":
        ids = (torch.randn(n, generator=rng) * (VOCAB_SIZE / 8) + VOCAB_SIZE / 2).round().clamp(0, VOCAB_SIZE - 1).long()
    elif pattern == "zipf":
        weights = torch.arange(1, VOCAB_SIZE + 1, dtype=torch.float64).pow(-1.2)
        ids = torch.multinomial(weights, n, replacement=True, generator=rng)
    elif pattern == "sparse_extremes":
        ids = torch.zeros(n, dtype=torch.int64)
        ids[::31] = VOCAB_SIZE - 1
    elif pattern == "cycle":
        values = torch.tensor(case["values"], dtype=torch.int64)
        ids = values[torch.arange(n) % len(values)]
    else:
        raise ValueError(f"Unknown distribution: {pattern}")
    assert ids.min() >= 0 and ids.max() < VOCAB_SIZE
    rows = [row.tolist() for row in ids.split(lengths)]
    if pattern == "shared_prefix":
        for row in rows[1:]:
            count = min(case.get("prefix_length", 16), len(rows[0]), len(row))
            row[:count] = rows[0][:count]
    form = case.get("form", "lists")
    if form == "lists":
        return rows
    if form == "flat":
        assert len(rows) == 1
        return rows[0]
    assert form in ("tensor32", "tensor64", "strided64", "flat_tensor64")
    assert len(set(lengths)) == 1, "Rectangular tensors cannot represent ragged requests"
    dtype = torch.int32 if form == "tensor32" else torch.int64
    result = torch.tensor(rows, dtype=dtype, device=case.get("input_device", "cpu"))
    if form == "flat_tensor64":
        assert len(rows) == 1
        return result[0]
    if form == "strided64":
        storage = torch.empty((len(rows), lengths[0] * 2), dtype=dtype, device=result.device)
        storage[:, ::2] = result
        result = storage[:, ::2]
        assert result.stride(-1) == 2
    return result


def assert_logits(actual, expected, shape, *, atol=0.0, rtol=0.0, bitwise=True):
    """Compare full-vocabulary bytes or values; bitwise mode ignores tolerances."""
    assert isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor)
    assert tuple(actual.shape) == tuple(expected.shape) == shape, "Wrong output shape"
    assert actual.dtype == expected.dtype == torch.float32, "Expected float32 logits"
    a, b = actual.detach().contiguous().cpu(), expected.detach().contiguous().cpu()
    assert torch.isfinite(a).all() and torch.isfinite(b).all(), "Non-finite logits"
    byte_equal = torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    if bitwise:
        assert byte_equal, "raw logit bytes differ"
    else:
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol, check_dtype=True)
    return {"max_abs": (a - b).abs().max().item(), "bitwise_equal": byte_equal}


@torch.inference_mode()
def run_tests(reference_forward, candidate_forward, reference_new_cache, candidate_new_cache,
              *, cases=TEST_CASES, atol=0.0, rtol=0.0, bitwise=True):
    """Callables take (input_ids, cache, **kwargs); cache factories take (B, initial_capacity=...).

    Both implementations receive identical tokens, call boundaries, and slot order.
    Cache formats may differ. Continuations check that each implementation preserves
    its own history. Chunked and unchunked evaluation need not be bitwise identical.
    """
    if type(bitwise) is not bool:
        raise TypeError("bitwise must be a bool")
    if not bitwise and not all(math.isfinite(x) and x >= 0 for x in (atol, rtol)):
        raise ValueError("atol and rtol must be finite and nonnegative")
    assert cases, "Select at least one test case"
    results = []
    for case in cases:
        batch_size = case.get("batch_size", len(case["steps"][0]["lengths"]))
        ref_cache = reference_new_cache(batch_size, initial_capacity=64)
        got_cache = candidate_new_cache(batch_size, initial_capacity=64)
        assert ref_cache is not got_cache, "Implementations must use independent caches"
        checks = []
        for index, step in enumerate(case["steps"]):
            kwargs = {}
            if "slots" in step:
                kwargs["request_indices"] = step["slots"]
            if step.get("all_logits", False):
                kwargs["return_all_logits"] = True
            count = sum(step["lengths"]) if step.get("all_logits", False) else len(step["lengths"])
            try:
                # Regenerate independently so input mutation cannot affect the other call.
                ref_kwargs = {key: list(value) if isinstance(value, list) else value for key, value in kwargs.items()}
                expected = reference_forward(make_input(case, step, index), ref_cache, **ref_kwargs)
                # Snapshot the result in case a candidate reuses an output workspace.
                expected = expected.detach().clone()
                got_kwargs = {key: list(value) if isinstance(value, list) else value for key, value in kwargs.items()}
                actual = candidate_forward(make_input(case, step, index), got_cache, **got_kwargs)
                checks.append(assert_logits(actual, expected, (count, VOCAB_SIZE), atol=atol, rtol=rtol, bitwise=bitwise))
            except Exception as error:
                raise AssertionError(f"{case['name']} step {index}, lengths={step['lengths']}, kwargs={kwargs}: {error}") from error
        results.append({"name": case["name"], "calls": len(checks), "max_abs": max(c["max_abs"] for c in checks), "bitwise_equal": all(c["bitwise_equal"] for c in checks)})
        print("PASS", json.dumps(results[-1]), flush=True)
        del ref_cache, got_cache, expected, actual
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, help="module:function, signature (model, input_ids, cache, **kwargs)")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--atol", type=float, default=0.0, help="Absolute tolerance; ignored with --bitwise")
    parser.add_argument("--rtol", type=float, default=0.0, help="Relative tolerance; ignored with --bitwise")
    parser.add_argument("--bitwise", action=argparse.BooleanOptionalAction, default=True,
                        help="Compare raw bytes instead of using atol/rtol (default: enabled)")
    parser.add_argument("--case", action="append", help="Run named cases only; repeat to select several")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = TEST_CASES if args.case is None else [c for c in TEST_CASES if c["name"] in args.case]
    if args.case and set(args.case) - {c["name"] for c in cases}:
        parser.error("Unknown case name")
    module, name = args.candidate.split(":", 1)
    candidate = getattr(importlib.import_module(module), name)
    from qwen38_inference import Qwen38
    model = Qwen38(args.model_path, max_context=8192)
    from reference.qwen38_inference import Qwen38 as ReferenceModel
    reference = object.__new__(ReferenceModel)
    reference.__dict__ = vars(model).copy()
    results = run_tests(reference.forward_step, lambda ids, cache, **kw: candidate(model, ids, cache, **kw),
                        reference.new_cache, model.new_cache, cases=cases,
                        atol=args.atol, rtol=args.rtol, bitwise=args.bitwise)
    report = {"reference": "reference.qwen38_inference.Qwen38.forward_step", "candidate": args.candidate,
              "atol": args.atol, "rtol": args.rtol, "bitwise": args.bitwise,
              "cases": results, "total_calls": sum(r["calls"] for r in results)}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"All {len(results)} cases passed ({report['total_calls']} forward_step comparisons)", flush=True)


if __name__ == "__main__":
    main()
