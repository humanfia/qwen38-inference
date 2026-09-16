# Qwen3.8-27B inference

Single-B200 text inference with optimized prefill and decode, the original reference, and reproducible benchmarks. Model weights are external.

```text
qwen38_inference.py     Inference API and CLI
solution/              Engine, cache, Triton/CUDA kernels and GEMM table
reference/             Original unmodified inference baseline
test_forward_step.py   Original 40 cases / 98 comparisons, adapted CLI bindings
benchmark.py           Complete-suite latency benchmark
results/               Historical measurements and per-shape results
```

## Run

Python 3.12, NVIDIA B200, CUDA 13 toolkit, a C++20 compiler and a local Qwen3.8-27B checkpoint are required. First use compiles kernels into the local cache.

```bash
pip install -r requirements.txt
python qwen38_inference.py --model-path /path/to/Qwen3.8-27B \
  --prompt 'Explain attention briefly.' --max-new-tokens 32
```

```python
from qwen38_inference import Qwen38
model = Qwen38('/path/to/Qwen3.8-27B', max_context=8192)
cache = model.new_cache(batch_size=2)
logits = model.forward_step([[11, 12], [21]], cache)
tokens = model.generate([11, 12], max_new_tokens=16)
```

## Evaluate

Use an exclusively reserved GPU. Both commands compare against `reference/qwen38_inference.py` with shared read-only weights and independent caches.

```bash
python test_forward_step.py --model-path /path/to/Qwen3.8-27B \
  --candidate solution.entry:run --atol 1e-3 --rtol 1e-3 --no-bitwise
python benchmark.py --model-path /path/to/Qwen3.8-27B --output runs/bench \
  --repeats 20 --warmups 1 --order balanced --check
```

The benchmark includes all 40 cases / 98 calls, separating 45 prefill and 53 decode calls. Timing includes input handling, cache growth, all layers and logits; loading and compilation are excluded.

## Results

Historical best mainline round, 2026-09-14: B200, Xeon Platinum 8581C, 20 repeats. Speedup is baseline / optimized synchronized wall latency. Each call uses its median latency.

| Metric | Prefill | Decode |
| --- | ---: | ---: |
| Sum of call medians | 1.063x | 2.006x |
| Mean per-call speedup | 1.153x | 2.032x |
| B16, L2048 | 1.041x | 1.803x |
| B32, L1024 | 1.046x | 2.069x |
| B64, L512 | 1.048x | 2.278x |

For prefill, L is input length; for decode, L is prior context and each request appends one token. Full-suite aggregate: 1.116x. Correctness: 98/98 passed, maximum absolute error 0.

[Release verification](results/verification.json) · [Per-shape CSV](results/shapes.csv) · [Raw best-round samples](results/best.json) · [Separate 128k measurement](results/long128k.json).

Best-round samples retain the baseline and selected implementation from the original five-arm run. Results depend on host and shape. The separate B1/131072 test measured 1.023x prefill and 0.927x decode; it is excluded from the main average. The inherited single-call prefill indexing limit remains: keep total input per call below 131072 tokens; `generate` uses 512-token chunks per request by default.

Apache-2.0; original notices are retained in the source and [LICENSE](LICENSE).
