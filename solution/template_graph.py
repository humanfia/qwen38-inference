"""Experimental decode graphs that accept current state from any small cache.

Each template owns scratch cache tensors and captures the unmodified engine
body. A call uploads actual tensor addresses, copies current cache contents
into scratch, replays the graph once, copies the updated cache back, and returns
owned logits. No input-dependent inference result is retained between calls.
Capacity buckets change only the cache pitch; GEMM batch sizes stay unchanged.

The first call executes the real decode eagerly on scratch and then captures
the now-compiled body without executing it. Later calls reuse that compilation
even when the caller supplies a newly created, cloned, or reset cache.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import triton
import triton.language as tl

from .engine import Engine, EngineOptions, _GraphEntry, _PinnedRing


@triton.jit
def _transfer_cache(REAL_PTRS, STATIC_PTRS, SIZES, REAL_CAP: tl.constexpr, STATIC_CAP: tl.constexpr,
                    KV_COUNT: tl.constexpr, TO_STATIC: tl.constexpr, BLOCK: tl.constexpr):
    tensor = tl.program_id(1)
    size = tl.load(SIZES + tensor).to(tl.int32)
    real = tl.load(REAL_PTRS + tensor)
    scratch = tl.load(STATIC_PTRS + tensor)
    # Descriptors give counts in 32-bit words. KV/conv are copied as raw pairs of
    # bf16 values, recurrent state as raw fp32 values; the copy has no arithmetic.
    real_ptr = real.to(tl.pointer_type(tl.int32))
    static_ptr = scratch.to(tl.pointer_type(tl.int32))
    for first in range(tl.program_id(0) * BLOCK, size, tl.num_programs(0) * BLOCK):
        offsets = first + tl.arange(0, BLOCK)
        static_offsets = offsets
        if tensor < KV_COUNT:
            token = offsets // 512
            slot = (token - 64) // REAL_CAP
            position = (token - 64) % REAL_CAP
            static_token = tl.where(token < 64, token, 64 + slot * STATIC_CAP + position)
            static_offsets = static_token * 512 + offsets % 512
        if TO_STATIC:
            value = tl.load(real_ptr + offsets, mask=offsets < size, other=0)
            tl.store(static_ptr + static_offsets, value, mask=offsets < size)
        else:
            value = tl.load(static_ptr + static_offsets, mask=offsets < size, other=0)
            tl.store(real_ptr + offsets, value, mask=offsets < size)


def tensors(cache):
    return ([t for pair in cache.kv.values() for t in pair]
            + list(cache.conv.values()) + list(cache.recurrent.values()))


class Template:
    def __init__(self, model, slots, batch, capacity):
        self.cache = model.new_cache(slots, initial_capacity=capacity)
        self.entry = _GraphEntry(self.cache, batch, self.cache.capacity, model.device)
        ts = tensors(self.cache)
        self.static_ptrs = torch.tensor([t.data_ptr() for t in ts], dtype=torch.int64, device=model.device)
        self.bytes = sum(t.numel() * t.element_size() for t in ts)
        self.tick = 0
        self.capture_failed = False


class TemplateEngine(Engine):
    """Use shape-keyed graphs for caches with at most four slots; larger caches run eagerly or, with
    ``large_decode="cache_graph"``, through Engine's per-cache decode graphs (no scratch transfers)."""

    def __init__(self, model, *, growth="batched", max_slots=4, **overrides):
        super().__init__(model, EngineOptions(growth=growth, graphs="decode", **overrides))
        self.templates = {}
        self.transfer_ring = _PinnedRing(self.device, torch.int64, 1 << 12)
        self.max_slots = max_slots
        self.template_budget_bytes = 16 << 30
        self.template_stats = {"captures": 0, "replays": 0, "warm_decodes": 0, "fallbacks": 0, "evictions": 0,
                               "large_eager": 0, "large_captures": 0, "large_replays": 0}

    def _transfer(self, cache, template, to_static, descriptors=None):
        if descriptors is None:
            ts = tensors(cache)
            pointers = np.asarray([t.data_ptr() for t in ts], dtype=np.int64)
            sizes = np.asarray([t.numel() * t.element_size() // 4 for t in ts], dtype=np.int64)
            descriptors = self.transfer_ring.upload([pointers, sizes])
        pointers, sizes = descriptors
        _transfer_cache[(128, pointers.numel())](
            pointers, template.static_ptrs, sizes, cache.capacity, template.cache.capacity,
            KV_COUNT=2 * len(cache.kv), TO_STATIC=to_static, BLOCK=2048, num_warps=4,
        )
        return descriptors

    def _decode_large(self, cache, flat, slots, prefixes, lengths, batch):
        """Caches with more slots than the templates serve: Engine's per-cache graphs (turn 17 option).

        No scratch copy is involved; the graph is bound to this cache's tensors and captured on the
        ``large_graph_warm_calls``-th decode of its signature. Counted separately from template activity.
        """
        captures, replays = self.graph_stats["captures"], self.graph_stats["replays"]
        output = self._decode_with_cache_graph(cache, flat, slots, prefixes, lengths, batch, self.options.large_graph_warm_calls)
        stats = self.template_stats
        stats["large_captures"] += self.graph_stats["captures"] - captures
        stats["large_replays"] += self.graph_stats["replays"] - replays
        if output is None:
            stats["large_eager"] += 1
        return output

    def _decode_with_graph(self, cache, flat, slots, prefixes, lengths, batch):
        if len(cache.lengths) > self.max_slots:
            if self.options.large_decode == "cache_graph":
                return self._decode_large(cache, flat, slots, prefixes, lengths, batch)
            self.graph_stats["eager_decodes"] += 1
            return None
        capacity = min(1 << (cache.capacity - 1).bit_length(),
                       math.ceil(cache.max_context / self.page_size) * self.page_size)
        key = (len(cache.lengths), batch, capacity)
        template = self.templates.get(key)
        if template is None:
            # LRU bounds retained state. This is a conservative memory guard,
            # not a promise that every shape remains cached indefinitely.
            estimate = (2 * len(cache.kv) * (len(cache.lengths) * capacity + self.page_size) * 2048
                        + sum(t.numel() * t.element_size() for t in list(cache.conv.values()) + list(cache.recurrent.values())))
            # Eviction cannot make a single oversized scratch cache fit. Fall back before
            # allocating it or evicting reusable templates; forward_step runs the live cache once.
            if estimate > self.template_budget_bytes:
                self.template_stats["fallbacks"] += 1
                self.graph_stats["eager_decodes"] += 1
                return None
            while self.templates and (len(self.templates) >= 32 or sum(t.bytes for t in self.templates.values()) + estimate > self.template_budget_bytes):
                oldest = min(self.templates, key=lambda k: self.templates[k].tick)
                del self.templates[oldest]
                self.template_stats["evictions"] += 1
            template = self.templates[key] = Template(self.model, len(cache.lengths), batch, capacity)
        self._graph_tick += 1
        template.tick = self._graph_tick
        entry = template.entry
        if template.capture_failed:
            self.template_stats["fallbacks"] += 1
            return None
        self._fill_static(entry, flat, slots, prefixes, lengths)
        descriptors = self._transfer(cache, template, True)
        if entry.graph is None:
            # Perform this call exactly once, and warm every kernel before capture.
            output = self._body(template.cache, entry.meta, True, [1] * batch, batch, batch, False, graph_dense=True)
            self.template_stats["warm_decodes"] += 1
            self.graph_stats["eager_decodes"] += 1
            # Compile the output transfer outside capture too; no inference work is omitted.
            self._transfer(cache, template, False, descriptors)
            graph = torch.cuda.CUDAGraph()
            try:
                torch.cuda.synchronize()
                with torch.cuda.stream(self._capture_stream):
                    graph.capture_begin(capture_error_mode="thread_local")
                    try:
                        captured_output = self._body(template.cache, entry.meta, True, [1] * batch, batch, batch, False, graph_dense=True)
                    finally:
                        graph.capture_end()
                torch.cuda.synchronize()
                entry.graph = graph
                entry.output = captured_output
                self.template_stats["captures"] += 1
                self.graph_stats["captures"] += 1
            except Exception:
                # The eager decode was already copied back; return that result,
                # and use the ordinary eager path on future calls of this shape.
                template.capture_failed = True
                self.template_stats["fallbacks"] += 1
                self.graph_stats["capture_failures"] += 1
                torch.cuda.synchronize()
            return output
        entry.graph.replay()
        self._transfer(cache, template, False, descriptors)
        self.template_stats["replays"] += 1
        self.graph_stats["replays"] += 1
        return entry.output.clone()


def engine_key(growth="batched", **overrides):
    return ("template_graph", growth, tuple(sorted(overrides.items())))


def get_template_engine(model, growth="batched", **overrides):
    # Reuse the same model-scoped engine table as solution.engine without changing
    # model weights/configuration or reference methods. A distinct key keeps the
    # experiment separate from default dispatch. ``overrides`` are EngineOptions fields.
    from .engine import _ENGINES
    table = _ENGINES.get(model)
    if table is None:
        table = _ENGINES[model] = {}
    key = engine_key(growth, **overrides)
    if key not in table:
        table[key] = TemplateEngine(model, growth=growth, **overrides)
    return table[key]


# candidate spec (module:function) -> engine key, for the harness's read-only counters
COUNTER_KEYS = {}


def register_runner(spec, growth="batched", **overrides):
    COUNTER_KEYS[spec] = engine_key(growth, **overrides)


def run(model, input_ids, kv_cache, **kwargs):
    return get_template_engine(model).forward_step(input_ids, kv_cache, **kwargs)


def run_fused_growth(model, input_ids, kv_cache, **kwargs):
    return get_template_engine(model, "fused").forward_step(input_ids, kv_cache, **kwargs)


register_runner("solution.template_graph:run")
register_runner("solution.template_graph:run_fused_growth", "fused")


def benchmark_counters(model, spec):
    """Read-only cumulative counters; the harness records per-call deltas outside timing."""
    from .engine import _ENGINES
    key = COUNTER_KEYS.get(spec)
    engine = _ENGINES.get(model, {}).get(key) if key is not None else None
    if engine is None:
        return {}
    counters = dict(engine.template_stats)
    if engine.direct is not None:
        for name, value in engine.direct.items():
            if isinstance(value, bool):
                # add_norm / gdn_prefill direct launchers and the fused-kernel self-check outcomes
                counters[f"direct_{name}"] = int(value)
    prefetcher = getattr(engine, "_prefetcher", None)
    if prefetcher is not None:
        # Turn 25: capture-time L2 prefetch branch activity (launches happen at capture, never during replay).
        for name, value in prefetcher.counters.items():
            counters[f"prefetch_{name}"] = int(value)
    return counters
