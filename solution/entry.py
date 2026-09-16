"""Candidate interface consumed by the supplied forward-step test script.

Forward uses fused cache growth and shape-keyed decode graphs for small caches.
The graph templates import current cache contents on every call. Generate keeps
the existing per-cache graph loop, which also supports larger batches without
state transfers. Both paths preserve the reference cache and input interfaces.
Prefill hands the recurrent-state pool to the chunk kernel directly when the
active slots are the whole cache in order, and the host issues launches with one
stream query per call and direct flashinfer launchers (self-checked bitwise).
Lane 3's bitwise-neutral fused kernels (their turn-4 loop-structured conv1d writing
L2-normalised q/k and v directly with a per-call worklist, blocked gating emitting
exp(g), RoPE writing k/v into the cache, gated norm with 8 rows per program in
prefill; for prefills of >= 256 tokens their turn-4 CUDA RoPE/QK-norm/KV-store
kernel with four heads per warp and their turn-3 token-major gated norm) and
Lane 2's cuBLASLt algorithm table are enabled after one-time bitwise self-checks
against the reference kernels; see solution.engine.EngineOptions and DEFAULT_OVERRIDES below.
"""

from .engine import EngineOptions, get_engine as _get_eager_engine
from .template_graph import get_template_engine, register_runner as _register_runner

# Default configuration on top of EngineOptions' defaults: Lane 2's measured cuBLASLt algorithm table
# for the dense projections (bound after a bitwise self-check; F.linear otherwise; turn 5: 1.00436x total,
# paired 95% interval [1.0010, 1.0079]) plus Lane 3's turn-3 prefill kernels for calls of >= 256 tokens:
# the CUDA RoPE/QK-norm/KV-store kernel (kv_store="cuda") and the token-major gated norm
# (norm_rows="token16"), both self-checked bitwise against the reference kernels before activation.
# Turn 7 measured the pair at 1.00584x total over the turn-6 default (paired 95% interval
# [1.0033, 1.0080], prefill 1.00605x, decode unchanged); solution.variants.run_turn6 keeps the old default.
# Turn 9 adds Lane 3's turn-4 kernels: the loop-structured prefill conv with fused Q/K L2 norm and per-call
# worklist (conv_split="loop", flag conv_loop) and the four-heads-per-warp RoPE/QK-norm/KV-store kernel
# (kv_store="cuda4"); measured in job 23 (see results/turn9/README.md); solution.variants.run_lane3_turn3
# keeps the turn-7 default as the paired control.
DEFAULT_OVERRIDES = {"dense": "lt", "kv_store": "cuda4", "norm_rows": "token16", "conv_split": "loop"}


def run(model, input_ids, kv_cache, **kwargs):
    """Candidate forward_step: fused growth, decode graph templates, fused kernels, dense table."""
    return get_template_engine(model, "fused", **DEFAULT_OVERRIDES).forward_step(input_ids, kv_cache, **kwargs)


def generate(model, input_ids, **kwargs):
    """Candidate generate with the default configuration (per-cache decode graphs inside)."""
    return _get_eager_engine(model, EngineOptions(**DEFAULT_OVERRIDES)).generate(input_ids, **kwargs)


def get_engine(model, options=None):
    """Default forward engine, or an explicitly configured ordinary engine."""
    if options is not None:
        return _get_eager_engine(model, options)
    return get_template_engine(model, "fused", **DEFAULT_OVERRIDES)


_register_runner("qwen38_inference:run", "fused", **DEFAULT_OVERRIDES)


def benchmark_counters(model, spec):
    from .template_graph import benchmark_counters as counters
    return counters(model, "qwen38_inference:run")

__all__ = ["run", "generate", "get_engine", "DEFAULT_OVERRIDES"]
