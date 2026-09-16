"""Candidate execution engine bound to a reference-created Qwen38 model.

Why a separate engine: solution.entry used to call the copied class method
``qwen38_optimized.Qwen38.forward_step(model, ...)`` on a *reference* model
instance. Every ``self.<method>`` inside that copy therefore resolved to the
reference class, so edits to copied helper methods (``_reserve``, ``new_cache``,
...) silently had no effect. The engine dispatches explicitly: weights, kernels
and the RoPE table are read from the reference model, while all orchestration
(input conversion, metadata staging, cache growth, layer loop, logits, optional
CUDA-graph replay) lives here and in the kernel library copy
``solution.qwen38_optimized``.

The engine consumes the reference ``HybridCache`` objects created by
``model.new_cache`` and preserves their semantics: slots are stable, lengths
count consumed tokens, growth keeps the zero-initialised tail and reserved page,
and ``clone``/``reset`` on the reference dataclass keep working because no
history is stored outside the cache. Decode graphs hold only device pointers of
one specific cache object; they are keyed by that object's identity and tensor
addresses, so growth (new KV tensors), cloning (new object) or a freed cache can
never be replayed against the wrong memory.

Execution options are explicit so that controlled ablations can run inside one
process (see ``solution.variants``).
"""
from __future__ import annotations

import dataclasses
import math
import weakref

import numpy as np
import torch
import torch.nn.functional as F

from .qwen38_optimized import HybridCache

from . import lean as L
from . import cache_growth as C
from . import lane3_kernels as K3
from . import lane3_conv_tile as KT
from . import lane3_turn3 as T3
from . import lane3_conv_loop as KL
from . import lane3_turn4 as T4
from . import qwen38_optimized as K


# cuBLASLt version (cublasLtGetVersion) on which Lane 2 recorded solution/lane2_dense_table.json (their probe2.json and
# turn-30 workspace context: 130101). A different runtime version is where recorded heuristic attributes can stop resolving.
DENSE_TABLE_LT_VERSION = 130101


@dataclasses.dataclass(frozen=True)
class EngineOptions:
    """Execution-orchestration switches; none of them change the arithmetic."""

    # "reference": one torch.tensor(list) upload per metadata array and per-layer index
    # dtype conversions, as in the baseline. "pinned": numpy construction, a reusable
    # pinned ring buffer with one transfer per dtype, and hoisted conversions.
    staging: str = "pinned"
    # "reference": per-slot copies. "batched": strided copy when all slots are full.
    # "packed": batched copy with one zeroed allocation. "fused": one allocation and
    # one kernel copying live prefixes while zeroing everything else.
    growth: str = "fused"
    # "wrapper": upstream Triton wrappers. "lean": cached direct launches (solution.lean) for the
    # rope/gate, sigmoid-gate, conv-update, recurrent-decode and gated-RMSNorm kernels.
    launch: str = "lean"
    # "off" or "decode": replay captured CUDA graphs for repeated decode signatures.
    graphs: str = "off"
    # Capture a decode signature (cache identity, batch, capacity) on its N-th call.
    graph_warm_calls: int = 2
    # Least-recently-used bound on retained decode graphs.
    max_graphs: int = 16
    # Prefill recurrent-state handling. "gather": index_select the active slots' states and
    # index_copy the final states back (baseline behaviour). "identity": when the active slots are
    # exactly 0..N-1 of an N-slot cache, hand the cache tensor to the chunk kernel directly and
    # install the freshly written final-state tensor in the cache (no gather, no scatter, no alias).
    state_path: str = "identity"
    # Host-side issue path. "reference": per-launch stream queries and the flashinfer Python
    # wrappers with int64 cu_seqlens (turn-2 behaviour). "lean": one stream query per call, int32
    # cu_seqlens, and direct calls into flashinfer's inner launchers when a one-time self-check
    # shows they reproduce the wrapper results bitwise (otherwise the wrappers are kept).
    issue: str = "lean"
    # Lane 3's bitwise-neutral fused kernels (solution.lane3_kernels). Each is enabled only after a
    # one-time self-check against the reference kernels on random data (Engine.direct records it);
    # a failed check keeps the reference path. "fused": the prefill conv1d writes token-major q/k/v
    # directly; "separate": conv1d followed by the split kernel. "tile": token-parallel
    # conv/split; "tile_norm": the same 32x128/4-warp tile with Q/K L2 norm fused; "loop": Lane 3's
    # turn-4 sub-tile loop kernel with program-resident transposed weights, a per-call packed
    # (sequence, chunk) worklist staged with the other prefill metadata, tokens per program chosen
    # from the call's lengths, and the Q/K L2 norm fused (solution.lane3_conv_loop). A prefill that
    # carries no worklist (graph-captured prefill templates) uses the "fused" path instead.
    conv_split: str = "fused"
    # "joint": q and k L2-normalised in one launch; "separate": two l2norm launches.
    qk_norm: str = "joint"
    # "blocked_exp": token-blocked gating grid emitting exp(g) (no torch.exp launch); "reference".
    gating: str = "blocked_exp"
    # "fused": the QK-norm/RoPE kernel writes k and v into the paged cache; "separate": RoPE + store;
    # "cuda": Lane 3's turn-3 CUDA kernel (one warp per head, solution.lane3_turn3) for prefills of
    # >= 256 tokens, the Triton fused kernel otherwise; "cuda4": their turn-4 kernel with four heads per
    # warp (solution.lane3_turn4), same arithmetic and eligibility.
    kv_store: str = "fused"
    # Heads per RoPE program for calls of >= 256 tokens with the fused store: "q2k1" (Lane 3's final
    # default), "q8k4", or "none" (one head per program, the reference grid).
    rope_group: str = "q2k1"
    # Gated RMSNorm rows per program in prefill: "prefill8" uses 8 where the reference heuristic picks
    # 4 (same compiler reduction layout, validated bitwise on the suite by Lane 3); "heuristic";
    # "token16": Lane 3's turn-3 token-major kernel (16 heads per program) for prefills of >= 256
    # tokens, rows 8 otherwise.
    norm_rows: str = "prefill8"
    # Dense projections: "torch" (F.linear) or "lt" (Lane 2's measured cuBLASLt algorithm table via
    # solution.lane2_fastlt with ATen fallback outside the table), bound after a bitwise self-check.
    dense: str = "torch"
    # Lane 1 turn 53: with dense="lt", resolve every (M, N, K) row of Lane 2's table on the current cuBLASLt runtime once
    # after the bitwise self-check (heuristic lookup through lt_algo_from_attrs, no GEMM, no allocation) and route rows
    # whose recorded algorithm attributes no longer resolve to F.linear. Without it a row resolves lazily on its first
    # use and an unresolvable row raises inside a timed forward (lane2_ltgemm's resolver throws). The table was recorded
    # on cuBLASLt DENSE_TABLE_LT_VERSION; Engine.direct records the runtime version, the row count and unresolved rows.
    # False keeps the lazy resolution of job 24 exactly.
    dense_preresolve: bool = True
    # Use Lane 2's validated GEMVs only in decode graph bodies. Shape misses use
    # the ordinary dense callable; eager forwards and all prefills are unchanged.
    graph_gemv: bool = False
    # SiLU-and-multiply after the gate/up projection: "reference" (the sgl_kernel op), "ilp"
    # (solution.silu_ilp: identical per-element arithmetic, constant-divisor addressing, read-only
    # loads; prefills of >= silu_ilp.ILP_MIN_TOKENS tokens only) or "ilp_pdl" (the same kernel launched
    # with programmatic dependent launch like the reference op), bound after a bitwise self-check.
    silu: str = "reference"
    # Decode calls on caches with more slots than the template engine serves (TemplateEngine.max_slots,
    # four): "eager" keeps every such call eager (the behaviour up to turn 16); "cache_graph" replays the
    # per-cache decode graphs that generate already uses (keyed by cache identity, capacity, batch and the
    # cache tensor addresses, captured on the ``large_graph_warm_calls``-th decode of one signature, no
    # scratch copies). Plain Engine instances with graphs="decode" serve every batch size that way already
    # and ignore this field; it only routes TemplateEngine calls.
    large_decode: str = "eager"
    large_graph_warm_calls: int = 2
    # Rows of the FP32 recurrent state per program in the decode GDN recurrence kernel (Lane 3 turn 12, integrated turn
    # 19): 32 is the reference wrapper's fixed tile (grid (4, B*48)); 16, 8 or 4 launch the identical kernel function
    # with grid (128 // rows, B*48) through the lean site (launch="lean" required), enabled only after a one-time bitwise
    # self-check against the wrapper on the current device (Engine.direct["recurrent_bv"]); otherwise 32 is kept.
    decode_bv: int = 32
    # Lane 3 turn 13: compute FP32 decay and BF16-rounded beta once in the existing decode convolution.
    # Research option, enabled only after a paired bitwise self-check. The retained path stays "reference".
    decode_gates: str = "reference"
    # Lane 1 turn 25: prefetch of the next dense projection's weight rows into L2 on a forked branch of captured decode
    # graphs (solution.l2_prefetch25). "off" (retained); "rows": uniformly spread whole rows through cp.async.bulk.prefetch.L2
    # with an evict_last policy, demoted to evict_normal by the next branch node once the projection consumed them;
    # "rows_normal": the same rows with the default priority and no demote; "lines": per-128-byte-line prefetch instructions
    # with demote; "rows_sticky": evict_last without demote (diagnostic arm). Only captured decode bodies prefetch (templates,
    # per-cache graphs, generate); eager forwards and prefills are unchanged. Enabled only after the self-check in
    # Engine._check_fused (Engine.direct["l2_prefetch"]); the arithmetic is untouched (the kernels write nothing).
    l2_prefetch: str = "off"
    # Multiplies the per-site byte budgets of solution.l2_prefetch25.SITE_BUDGET_BYTES (each capped by CAP_BYTES and the weight).
    l2_prefetch_scale: float = 1.0


class _PinnedRing:
    """Two pinned host buffers with device twins; reuse is guarded by CUDA events.

    Host buffer k may be rewritten only after the copy issued from it finished
    (event k). Device buffer k is stream-ordered, so kernels of the previous
    call that read it complete before the next copy into it.
    """

    def __init__(self, device, dtype, initial=1 << 16):
        self.device = device
        self.dtype = dtype
        self.size = 0
        self.host = [None, None]
        self.dev = [None, None]
        self.events = [torch.cuda.Event(), torch.cuda.Event()]
        self.turn = 0
        self._allocate(initial)

    def _allocate(self, size):
        self.size = size
        self.host = [torch.empty(size, dtype=self.dtype, pin_memory=True) for _ in range(2)]
        self.dev = [torch.empty(size, dtype=self.dtype, device=self.device) for _ in range(2)]

    def upload(self, arrays):
        """Copy a list of numpy arrays in one transfer; return device views in order."""
        total = sum(int(a.size) for a in arrays)
        if total > self.size:
            torch.cuda.synchronize()
            self._allocate(max(total, 2 * self.size))
        k = self.turn
        self.turn ^= 1
        self.events[k].synchronize()
        host = self.host[k]
        view = host.numpy()
        offset = 0
        spans = []
        for array in arrays:
            n = int(array.size)
            view[offset:offset + n] = array.reshape(-1)
            spans.append((offset, n))
            offset += n
        dev = self.dev[k]
        if total:
            dev[:total].copy_(host[:total], non_blocking=True)
        self.events[k].record()
        return [dev[o:o + n] for o, n in spans]


class _Metadata:
    __slots__ = ("tokens", "positions", "locations", "last_indices", "slots64", "query_start", "query_start64",
                 "state_indices", "page_table", "sequence_lengths", "kv_starts", "has_initial", "identity_slots",
                 "conv_worklist", "conv_block_t")

    def __init__(self):
        for name in self.__slots__:
            setattr(self, name, None)


class _GraphEntry:
    """One captured decode graph with its static metadata buffers and output."""

    __slots__ = ("graph", "cache_ref", "batch", "capacity", "host64", "dev64", "host32", "dev32", "meta", "output", "event", "tick")

    def __init__(self, cache, batch, capacity, device):
        self.graph = None
        self.cache_ref = weakref.ref(cache)
        self.batch = batch
        self.capacity = capacity
        pages = capacity // 64
        self.host64 = torch.empty(3 * batch, dtype=torch.int64, pin_memory=True)
        self.dev64 = torch.empty(3 * batch, dtype=torch.int64, device=device)
        self.host32 = torch.empty(2 * batch + batch * pages, dtype=torch.int32, pin_memory=True)
        self.dev32 = torch.empty(2 * batch + batch * pages, dtype=torch.int32, device=device)
        meta = _Metadata()
        meta.tokens = self.dev64[0:batch]
        meta.positions = self.dev64[batch:2 * batch]
        meta.locations = self.dev64[2 * batch:3 * batch]
        meta.state_indices = self.dev32[0:batch]
        meta.sequence_lengths = self.dev32[batch:2 * batch]
        meta.page_table = self.dev32[2 * batch:].view(batch, pages)
        self.meta = meta
        self.output = None
        self.event = torch.cuda.Event()
        self.tick = 0


class Engine:
    """Optimised forward_step/generate over a reference Qwen38 instance's weights."""

    def __init__(self, model, options: EngineOptions | None = None):
        self.model = model
        self.options = options or EngineOptions()
        o = self.options
        if o.staging not in ("reference", "pinned") or o.growth not in ("reference", "batched", "packed", "fused") or o.graphs not in ("off", "decode") or o.launch not in ("wrapper", "lean"):
            raise ValueError(f"Unknown engine options: {o}")
        if o.state_path not in ("gather", "identity") or o.issue not in ("reference", "lean"):
            raise ValueError(f"Unknown engine options: {o}")
        if o.large_decode not in ("eager", "cache_graph") or type(o.large_graph_warm_calls) is not int or o.large_graph_warm_calls < 1 or o.graph_warm_calls < 1:
            raise ValueError(f"Unknown engine options: {o}")
        if (o.decode_bv != 32 or o.decode_gates != "reference" or o.l2_prefetch != "off"
                or o.graph_gemv or o.silu != "reference"):
            raise ValueError("This standalone build contains only the retained inference paths")
        if (o.conv_split not in ("fused", "separate", "tile", "tile_norm", "loop") or o.qk_norm not in ("joint", "separate") or o.gating not in ("blocked_exp", "reference")
                or o.kv_store not in ("fused", "separate", "cuda", "cuda4") or o.rope_group not in ("q2k1", "q8k4", "none") or o.norm_rows not in ("prefill8", "heuristic", "token16")
                or o.dense not in ("torch", "lt") or o.silu not in ("reference", "ilp", "ilp_pdl")):
            raise ValueError(f"Unknown engine options: {o}")
        if type(o.dense_preresolve) is not bool:
            raise ValueError(f"dense_preresolve must be a bool: {o}")
        self._linear = F.linear
        self._graph_linear = None
        # Lane 1 turn 25: capture-time L2 prefetch helper, created by _check_fused after its self-check passed.
        self._prefetcher = None
        # Direct flashinfer launchers are bound lazily (first eager forward) after a bitwise self-check.
        self.direct = None
        self._add_norm = getattr(model, "_add_norm", None)
        self._gdn_direct = None
        self._gdn_cu32 = o.issue == "lean"
        self.lean = o.launch == "lean"
        self.device = model.device
        self.page_size = model.page_size
        self.vocab_size = model.config["vocab_size"]
        pinned = o.staging == "pinned"
        # Pinned allocation is slow on some hosts (seconds), so size the rings once for the largest
        # supported call (max_context tokens per request, up to 64 requests) instead of growing lazily.
        tokens = max(int(model.max_context) * 4, 1 << 18)
        self._ring64 = _PinnedRing(self.device, torch.int64, tokens) if pinned else None
        self._ring32 = _PinnedRing(self.device, torch.int32, 1 << 16) if pinned else None
        self._ring8 = _PinnedRing(self.device, torch.uint8, 1 << 12) if pinned else None
        self._growth_ring = _PinnedRing(self.device, torch.int64, 1 << 12) if o.growth == "fused" else None
        self._graphs: dict[tuple, _GraphEntry] = {}
        self._capture_stream = torch.cuda.Stream(device=self.device) if o.graphs == "decode" else None
        self._graph_seen: dict[tuple, int] = {}
        self._graph_tick = 0
        self.graph_stats = {"captures": 0, "replays": 0, "eager_decodes": 0, "capture_failures": 0}
        # Lane 3 turn 4: the loop conv kernel reads the four taps from the contiguous transposed weight [4, 10240].
        # Prepared once per engine (a weight layout, like load-time preprocessing) and keyed by the live weight
        # tensor's storage pointer/shape/dtype so a replaced weight is re-derived; weights are never mutated here.
        self._conv_wt: dict[int, tuple] = {}
        if o.conv_split == "loop":
            for i, (kind, w) in enumerate(zip(model.layer_types, model.layers)):
                if kind != "full_attention":
                    self._conv_weight_t(i, w["conv"])

    def _conv_weight_t(self, layer, weight):
        signature = (weight.data_ptr(), tuple(weight.shape), weight.dtype)
        entry = self._conv_wt.get(layer)
        if entry is None or entry[0] != signature:
            entry = self._conv_wt[layer] = (signature, weight.t().contiguous())
        return entry[1]

    # ------------------------------------------------------------------ cache
    def new_cache(self, batch_size=1, initial_capacity=256, max_context=None):
        return self.model.new_cache(batch_size, initial_capacity=initial_capacity, max_context=max_context)

    def _reserve(self, cache, needed):
        model = self.model
        if needed > cache.max_context:
            raise ValueError(f"Context limit exceeded: {needed} > {cache.max_context}")
        if needed <= cache.capacity:
            return
        ps = self.page_size
        capacity = math.ceil(min(cache.max_context, max(needed, cache.capacity * 2)) / ps) * ps
        batch = len(cache.lengths)
        if self.options.growth == "fused":
            cache.kv = C.grow_fused(cache, capacity, ps, self.device, model.dtype, self._growth_ring)
            cache.capacity = capacity
            return
        packed = C.allocate_packed(cache, capacity, ps, self.device, model.dtype, zero=True)[1] if self.options.growth == "packed" else None
        full = all(length == cache.capacity for length in cache.lengths)
        new_kv = {}
        for i, pair in cache.kv.items():
            new_pair = []
            for j, old in enumerate(pair):
                new = packed[i][j] if packed is not None else torch.zeros((batch * capacity + ps, 4, 256), dtype=model.dtype, device=self.device)
                if self.options.growth == "reference" or not full:
                    for slot, length in enumerate(cache.lengths):
                        if length:
                            new[ps + slot * capacity:ps + slot * capacity + length].copy_(old[ps + slot * cache.capacity:ps + slot * cache.capacity + length])
                else:
                    # reset() leaves old KV tokens in place; copying tails would preserve stale
                    # values that the reference growth discards. Batch only fully occupied slots.
                    new[ps:].view(batch, capacity, 4, 256)[:, :cache.capacity].copy_(old[ps:].view(batch, cache.capacity, 4, 256))
                new_pair.append(new)
            new_kv[i] = tuple(new_pair)
        cache.kv = new_kv
        cache.capacity = capacity

    # --------------------------------------------------------------- metadata
    def _metadata(self, flat, slots, lens, prefixes, lengths, cu, kv_cu, capacity, is_decode, return_all_logits):
        ps = self.page_size
        meta = _Metadata()
        pages_per_slot = capacity // ps
        max_pages = math.ceil(max(lengths) / ps)
        if self.options.staging == "reference":
            def tensor(data, dtype=torch.int32):
                return torch.tensor(data, dtype=dtype, device=self.device)
            meta.query_start = tensor(cu)
            meta.state_indices = tensor(slots)
            meta.positions = tensor([p + j for p, n in zip(prefixes, lens) for j in range(n)], torch.int64)
            meta.locations = tensor([ps + s * capacity + p + j for s, p, n in zip(slots, prefixes, lens) for j in range(n)], torch.int64)
            meta.page_table = tensor([[1 + s * pages_per_slot + j if j < math.ceil(n / ps) else 0 for j in range(max_pages)] for s, n in zip(slots, lengths)])
            meta.sequence_lengths = tensor(lengths)
            meta.kv_starts = tensor(kv_cu)
            meta.has_initial = tensor([p > 0 for p in prefixes], torch.bool)
            meta.tokens = tensor(flat, torch.int64)
            if not (is_decode and not return_all_logits):
                meta.last_indices = None if return_all_logits else tensor([end - 1 for end in cu[1:]], torch.int64)
            if not is_decode and self.options.conv_split == "loop":
                meta.conv_block_t = KL.choose_block_t(lens)
                meta.conv_worklist = tensor(KL.worklist_items(lens, meta.conv_block_t).tolist())
            return meta
        slots_np = np.asarray(slots, dtype=np.int64)
        lens_np = np.asarray(lens, dtype=np.int64)
        prefixes_np = np.asarray(prefixes, dtype=np.int64)
        tokens_np = np.asarray(flat, dtype=np.int64)
        lengths_np = np.asarray(lengths, dtype=np.int64)
        cu_np = np.asarray(cu, dtype=np.int64)
        if is_decode:
            positions_np = prefixes_np
            locations_np = ps + slots_np * capacity + prefixes_np
        else:
            offsets = np.arange(int(cu[-1]), dtype=np.int64) - np.repeat(cu_np[:-1], lens_np)
            positions_np = np.repeat(prefixes_np, lens_np) + offsets
            locations_np = np.repeat(ps + slots_np * capacity + prefixes_np, lens_np) + offsets
        columns = np.arange(max_pages, dtype=np.int64)[None, :]
        pages_needed = ((lengths_np + ps - 1) // ps)[:, None]
        table = np.where(columns < pages_needed, 1 + slots_np[:, None] * pages_per_slot + columns, 0).astype(np.int32)
        arrays32 = [cu_np.astype(np.int32), slots_np.astype(np.int32), table, lengths_np.astype(np.int32), np.asarray(kv_cu, dtype=np.int32)]
        arrays64 = [tokens_np, positions_np, locations_np]
        if not is_decode:
            arrays64 += [slots_np, cu_np]
        if not (is_decode or return_all_logits):
            arrays64.append(cu_np[1:] - 1)
        conv_loop = (not is_decode) and self.options.conv_split == "loop"
        if conv_loop:
            # Lane 3 turn 4: the packed (sequence << 16 | chunk) worklist of this call, staged with the other
            # int32 metadata; padded to a 16-byte offset so its pointer specialisation is stable.
            meta.conv_block_t = KL.choose_block_t(lens)
            work = KL.worklist_items(lens_np, meta.conv_block_t)
            pad = (-sum(int(a.size) for a in arrays32)) % 4
            if pad:
                arrays32.append(np.zeros(pad, dtype=np.int32))
            arrays32.append(work)
        views64 = self._ring64.upload(arrays64)
        views32 = self._ring32.upload(arrays32)
        meta.tokens, meta.positions, meta.locations = views64[:3]
        if not is_decode:
            meta.slots64, meta.query_start64 = views64[3:5]
            if not return_all_logits:
                meta.last_indices = views64[5]
            (flags,) = self._ring8.upload([(prefixes_np > 0).astype(np.uint8)])
            meta.has_initial = flags.view(torch.bool)
        meta.query_start, meta.state_indices, page_table, meta.sequence_lengths, meta.kv_starts = views32[:5]
        meta.page_table = page_table.view(len(slots), max_pages)
        if conv_loop:
            meta.conv_worklist = views32[-1]
        return meta

    # ------------------------------------------------------- direct launchers
    def _bind_direct(self):
        """Bind flashinfer's inner launch functions when they reproduce the model's wrappers bitwise.

        The wrappers (``gemma_fused_add_rmsnorm`` via sgl_kernel, ``chunk_gated_delta_rule``) spend
        tens of microseconds per call on argument handling, custom-op dispatch and dtype casts.
        The inner functions launch the identical compiled kernels with identical arguments, so
        the arithmetic is unchanged; a one-time comparison on random data guards against version
        differences, and any failure keeps the wrapper.
        """
        if torch.cuda.is_current_stream_capturing():
            return  # keep the wrappers for this call; bind on the next eager call
        model = self.model
        direct = {"add_norm": False, "gdn_prefill": False, "errors": []}
        if self.options.issue == "lean":
            device = self.device
            try:
                import flashinfer.norm as fnorm
                from flashinfer.norm.kernels.fused_add_rmsnorm import fused_add_rmsnorm_cute
                from flashinfer.utils import device_support_pdl
                if getattr(fnorm, "_USE_CUDA_NORM", True):
                    raise RuntimeError("flashinfer uses its CUDA norm module; keeping the wrapper")
                pdl = bool(device_support_pdl(device))

                def add_norm(hidden, residual, weight, eps):
                    fused_add_rmsnorm_cute(hidden, residual, weight, eps, weight_bias=1.0, enable_pdl=pdl)

                gen = torch.Generator(device=device).manual_seed(11)
                x = torch.randn((37, 5120), generator=gen, device=device).to(model.dtype)
                r = torch.randn((37, 5120), generator=gen, device=device).to(model.dtype)
                x1, r1, x2, r2 = x.clone(), r.clone(), x.clone(), r.clone()
                model._add_norm(x1, r1, model.final_norm, model.eps)
                add_norm(x2, r2, model.final_norm, model.eps)
                torch.cuda.synchronize()
                if torch.equal(x1, x2) and torch.equal(r1, r2):
                    self._add_norm = add_norm
                    direct["add_norm"] = True
                else:
                    direct["errors"].append("add_norm self-check differs")
            except Exception as error:  # noqa: BLE001 - keep the wrapper
                direct["errors"].append(f"add_norm: {error!r}"[:300])
            try:
                from flashinfer.gdn_kernels import chunk_gated_delta_rule_sm100
                scale = 1.0 / math.sqrt(128)

                def gdn(q, k, v, gate, beta, initial, final, cu32):
                    out = torch.empty((q.shape[0], 48, 128), dtype=q.dtype, device=q.device)
                    chunk_gated_delta_rule_sm100(q, k, v, gate, beta, out, cu32, initial, final, scale)
                    return out

                gen = torch.Generator(device=device).manual_seed(12)
                T = 200
                q = torch.randn((T, 16, 128), generator=gen, device=device).to(model.dtype)
                k = torch.randn((T, 16, 128), generator=gen, device=device).to(model.dtype)
                v = torch.randn((T, 48, 128), generator=gen, device=device).to(model.dtype)
                gate = torch.rand((T, 48), generator=gen, device=device)
                beta = torch.rand((T, 48), generator=gen, device=device)
                cu32 = torch.tensor([0, 64, T], dtype=torch.int32, device=device)
                initial = torch.randn((2, 48, 128, 128), generator=gen, device=device)
                final_w, final_d = torch.empty_like(initial), torch.empty_like(initial)
                out_w, final_w = model._gdn_prefill(q=q, k=k, v=v, g=gate, beta=beta, scale=None, initial_state=initial, output_final_state=True,
                                                    cu_seqlens=cu32.to(torch.int64), use_qk_l2norm_in_kernel=False, output_state=final_w)
                out_d = gdn(q, k, v, gate, beta, initial, final_d, cu32)
                torch.cuda.synchronize()
                if torch.equal(out_w, out_d) and torch.equal(final_w, final_d):
                    self._gdn_direct = gdn
                    direct["gdn_prefill"] = True
                else:
                    direct["errors"].append("gdn_prefill self-check differs")
            except Exception as error:  # noqa: BLE001 - keep the wrapper
                direct["errors"].append(f"gdn_prefill: {error!r}"[:300])
        self._check_fused(direct)
        self.direct = direct

    def _check_fused(self, direct):
        """Enable each fused kernel only after it reproduces the reference kernels bitwise.

        The check runs the exact launcher the body uses (lean direct launch or wrapper) twice per
        shape, so the cached direct launch path is exercised as well as the first JIT dispatch.
        """
        o = self.options
        model = self.model
        device, dtype = self.device, model.dtype
        lean = self.lean
        stream = torch.cuda.current_stream().cuda_stream
        direct.update({"conv_split": False, "conv_l2norm": False, "kv_store": False, "gating": False, "qk_norm": False, "dense": False, "graph_gemv": False,
                       "rope_cuda": False, "norm_token": False, "silu_ilp": False, "conv_loop": False, "recurrent_bv": False})
        direct["recurrent_bv_rows"] = 32
        direct["decode_gates"] = False

        def attempt(name, cases, checker):
            try:
                ok = True
                for seed, case in enumerate(cases):
                    for pass_ in range(2):
                        passed, detail = checker(*case, 100 * seed + pass_)
                        if not passed:
                            ok = False
                            direct["errors"].append(f"{name} self-check differs: {detail}"[:300])
                            break
                    if not ok:
                        break
                direct[name] = ok
            except Exception as error:  # noqa: BLE001 - keep the reference path
                direct["errors"].append(f"{name}: {error!r}"[:300])

        if o.conv_split in ("fused", "tile", "tile_norm", "loop"):
            # With "loop" this checks the "fused" path, which serves prefills without a worklist (graph templates).
            normalize = o.conv_split == "tile_norm"
            if o.conv_split in ("tile", "tile_norm") and lean:
                def conv_launch(x, weight, states, cu, lens, idx, hi):
                    return L.conv_tile(x, weight, states, cu, lens, idx, hi, stream, fuse_l2norm=normalize)
            elif o.conv_split in ("tile", "tile_norm"):
                def conv_launch(x, weight, states, cu, lens, idx, hi):
                    return KT.causal_conv1d_split_tile_fn(x, weight, None, states, cu, lens, idx, hi, "silu", 16, 16, 48, 128,
                                                        block_t=32, block_n=128, num_warps=4, feat_first=False, fuse_l2norm=normalize)
            elif lean:
                def conv_launch(x, weight, states, cu, lens, idx, hi):
                    return L.conv_split(x, weight, states, cu, lens, idx, hi, stream)
            else:
                def conv_launch(x, weight, states, cu, lens, idx, hi):
                    return K3.causal_conv1d_split_fn(x, weight, None, states, cu, lens, idx, hi, "silu", 16, 16, 48, 128)
            attempt("conv_split", [([5, 64, 1, 300, 2], [3, 0, 4, 1, 2], [True, False, True, True, False]), ([1, 2, 1], [1, 2, 0], [True, True, False])],
                    lambda lens, slots, init, seed: K3.check_conv_split(conv_launch, device, dtype, lens, slots, init, seed, normalize=normalize))
            direct["conv_l2norm"] = normalize and direct["conv_split"]
        if o.conv_split == "loop":
            # Lane 3 turn 4: the loop kernel (L2 norm fused) must reproduce the reference conv + split + l2norm and the
            # complete conv state for every tokens-per-program value, with and without the worklist grid.
            def loop_launch(x, weight, states, cu, lens, idx, hi, block_t, use_worklist):
                weight_t = weight.t().contiguous()
                work = torch.from_numpy(KL.worklist_items(lens, block_t)).to(device) if use_worklist else None
                if lean:
                    return L.conv_loop(x, weight_t, states, cu, lens, idx, hi, work, block_t, stream, use_worklist=use_worklist)
                return KL.causal_conv1d_split_loop_fn(x, weight, None, states, cu, lens, idx, hi, "silu", 16, 16, 48, 128, block_t=block_t, block_n=128,
                                                      num_warps=KL.CONV_NUM_WARPS, feat_first=False, fuse_l2norm=True, sub_t=KL.CONV_SUB_T, worklist=work,
                                                      use_worklist=use_worklist, weight_t=weight_t)
            loop_cases = [([5, 64, 1, 300, 2], [3, 0, 4, 1, 2], [True, False, True, True, False], 32, True),
                          ([1, 2, 1], [1, 2, 0], [True, True, False], 32, True),
                          ([300, 700, 37], [1, 0, 2], [True, False, True], 64, True),
                          ([1100, 1300], [0, 1], [False, True], 128, True),
                          ([257, 64], [1, 0], [True, True], 64, False)]
            attempt("conv_loop", loop_cases,
                    lambda lens, slots, init, block_t, use_worklist, seed: K3.check_conv_split(
                        lambda *a: loop_launch(*a, block_t, use_worklist), device, dtype, lens, slots, init, seed, normalize=True))
        if o.kv_store in ("fused", "cuda", "cuda4"):
            if lean:
                def rope_launch(qg, k, v, qw, kw, rope, positions, locations, kr, vr, eps, qpp, kpp):
                    return L.rope_kvstore(qg, k, v, qw, kw, rope, positions, locations, kr, vr, eps, qpp, kpp, stream)
            else:
                def rope_launch(qg, k, v, qw, kw, rope, positions, locations, kr, vr, eps, qpp, kpp):
                    return K3.fused_qk_gemma_rmsnorm_rope_gate_kvstore(qg, k, v, qw, kw, rope, positions, locations, kr, vr, eps, 24, 4, 256, model.rotary_dim,
                                                                      q_per_prog=qpp, k_per_prog=kpp)
            grouped = self._rope_grouping(300)
            attempt("kv_store", [(3, 1, 1), (300,) + grouped],
                    lambda T, qpp, kpp, seed: K3.check_rope_kvstore(rope_launch, model._ops.store, device, dtype, T, qpp, kpp, seed))
        if o.kv_store in ("cuda", "cuda4"):
            # Lane 3 turn 3 ("cuda", one warp per head) or turn 4 ("cuda4", four heads per warp): the CUDA kernel
            # replaces the Triton fused store for prefills of >= 256 tokens. The first launch builds the extension
            # (persistent kernel cache); the check runs it on the three shapes.
            cuda_kernel = L.rope_kvstore_cuda4 if o.kv_store == "cuda4" else L.rope_kvstore_cuda

            def rope_cuda_launch(qg, k, v, qw, kw, rope, positions, locations, kr, vr, eps, qpp, kpp):
                return cuda_kernel(qg, k, v, qw, kw, rope, positions, locations, kr, vr, eps, stream)
            attempt("rope_cuda", [(3, 1, 1), (300, 1, 1), (1000, 1, 1)],
                    lambda T, qpp, kpp, seed: K3.check_rope_kvstore(rope_cuda_launch, model._ops.store, device, dtype, T, qpp, kpp, seed))
            direct["rope_kernel"] = o.kv_store
        if o.norm_rows == "token16":
            if lean:
                def norm_token_launch(x, w, z, eps, rows):
                    return L.rms_norm_gated_token(x, w, z, eps, stream)
            else:
                def norm_token_launch(x, w, z, eps, rows):
                    return T3.rms_norm_gated_token(x, w, z, eps)
            attempt("norm_token", [(256,), (700,), (4096,)], lambda T, seed: K3.check_norm_rows(norm_token_launch, device, dtype, T, None, seed))
        if o.gating == "blocked_exp":
            if lean:
                def gating_launch(A_log, a, b, dt_bias):
                    return L.gating_blocked_exp(A_log, a, b, dt_bias, stream)
            else:
                def gating_launch(A_log, a, b, dt_bias):
                    return K3.fused_gdn_gating_blocked(A_log, a, b, dt_bias, output_exp=True)
            attempt("gating", [(1,), (1000,)], lambda T, seed: K3.check_gating_exp(gating_launch, device, dtype, T, seed))
        if o.qk_norm == "joint":
            if lean:
                def norm_launch(q, k, eps):
                    return L.joint_l2norm(q, k, eps, stream)
            else:
                def norm_launch(q, k, eps):
                    return K3.joint_l2norm(q, k, eps)
            attempt("qk_norm", [(1,), (1000,)], lambda T, seed: K3.check_joint_l2norm(norm_launch, device, dtype, T, seed))
        direct["l2_prefetch"] = False
        if o.dense == "lt":
            try:
                from . import lane2_fastlt
                linear = lane2_fastlt.linear
                ok = True
                gdn = next(w for kind, w in zip(model.layer_types, model.layers) if kind != "full_attention")
                for weight, M in ((gdn["gate_up"], 1), (gdn["gate_up"], 256), (gdn["down"], 1024), (model.lm_head, 16)):
                    passed, detail = K3.check_dense(linear, weight, M, 7 + M)
                    if not passed:
                        ok = False
                        direct["errors"].append(f"dense self-check differs: {detail}"[:300])
                        break
                direct["dense"] = ok
                self._linear = linear if ok else F.linear
            except Exception as error:  # noqa: BLE001 - keep F.linear
                direct["errors"].append(f"dense: {error!r}"[:300])
                self._linear = F.linear
        if o.dense == "lt" and o.dense_preresolve and direct.get("dense"):
            self._preresolve_dense(direct)

    def _preresolve_dense(self, direct):
        """Resolve every row of Lane 2's cuBLASLt table on this runtime; route rows that do not resolve to F.linear.

        ``lane2_fastlt.dense_linear`` resolves a row's recorded algorithm attributes on the row's first use and raises
        when the current cuBLASLt heuristics no longer offer them, which would abort a timed forward on that shape.
        The bitwise self-check in ``_check_fused`` covers four rows; this method asks the extension's own resolver
        (``lt_algo_from_attrs``: heuristic lookup, no GEMM, no activation allocation) for all rows once, records the
        runtime version and the outcome in ``direct`` and, only when some rows fail, wraps the bound dense callable
        with a shape check that sends those rows to ``F.linear`` (the reference operation). With every row resolved
        the bound callable and the forward path are unchanged.
        """
        import json
        from pathlib import Path
        try:
            from . import lane2_fastlt
            ext = lane2_fastlt.load()
            table = json.loads(Path(lane2_fastlt.__file__).with_name("lane2_dense_table.json").read_text())
            resolve = ext.lt_algo_from_attrs  # the resolver dense_linear itself uses on a row's first use
            version = int(ext.lt_version())
            direct["dense_lt_version"] = version
            direct["dense_table_lt_version"] = DENSE_TABLE_LT_VERSION
            rows, unresolved = 0, []
            for shape, entries in table.items():
                N, K = map(int, shape.split(","))
                for entry in entries:
                    attrs = [int(a) for a in entry["attrs"]]
                    for M in range(int(entry["min_m"]), int(entry["max_m"]) + 1):
                        rows += 1
                        try:
                            resolve(M, N, K, attrs)
                        except Exception:  # noqa: BLE001 - this row would raise inside a forward
                            unresolved.append((M, N, K))
            direct["dense_rows"] = rows
            direct["dense_unresolved"] = unresolved
            if unresolved:
                blocked = frozenset(unresolved)
                table_linear = self._linear

                def guarded_linear(x, w, _table=table_linear, _blocked=blocked):
                    if x.dim() == 2 and (x.shape[0], w.shape[0], w.shape[1]) in _blocked:
                        return F.linear(x, w)
                    return _table(x, w)

                self._linear = guarded_linear
                direct["errors"].append(f"dense: {len(unresolved)} of {rows} table rows do not resolve on cuBLASLt {version} "
                                        f"(table recorded on {DENSE_TABLE_LT_VERSION}); those shapes use F.linear"[:300])
            direct["dense_preresolve"] = not unresolved
        except Exception as error:  # noqa: BLE001 - keep the bound callable; rows then resolve lazily on first use
            direct["dense_preresolve"] = False
            direct["errors"].append(f"dense_preresolve: {error!r}"[:300])

    def _rope_grouping(self, total):
        """(q heads, k heads) per RoPE program for a call of ``total`` tokens with the fused store."""
        if total >= 256 and self.options.rope_group != "none":
            return (2, 1) if self.options.rope_group == "q2k1" else (8, 4)
        return (1, 1)

    def _gdn_wrapper(self, meta, q, k, v, gate, beta, initial, final):
        model = self.model
        if self._gdn_cu32:
            try:
                out, _ = model._gdn_prefill(q=q, k=k, v=v, g=gate, beta=beta, scale=None, initial_state=initial, output_final_state=True,
                                            cu_seqlens=meta.query_start, use_qk_l2norm_in_kernel=False, output_state=final)
                return out
            except Exception:  # noqa: BLE001 - a wrapper rejecting int32 fails before launching; retry with int64
                self._gdn_cu32 = False
        cu64 = meta.query_start64 if meta.query_start64 is not None else meta.query_start.to(torch.int64)
        out, _ = model._gdn_prefill(q=q, k=k, v=v, g=gate, beta=beta, scale=None, initial_state=initial, output_final_state=True,
                                    cu_seqlens=cu64, use_qk_l2norm_in_kernel=False, output_state=final)
        return out

    # ---------------------------------------------------------------- forward
    def _body(self, cache, meta, is_decode, lens, batch, total, return_all_logits, *, graph_dense=False):
        """All 64 layers plus the vocabulary projection; reads/writes cache tensors only."""
        model = self.model
        eps = model.eps
        if self.direct is None:
            self._bind_direct()
        flags = self.direct or {}
        issue_lean = self.options.issue == "lean"
        norm, add_norm, ops = model._norm, (self._add_norm if issue_lean else model._add_norm), model._ops
        lean = self.lean
        linear = self._graph_linear if graph_dense and self._graph_linear is not None else self._linear
        # The stream cannot change inside one call; query it once instead of per launch.
        stream = torch.cuda.current_stream().cuda_stream if (lean and issue_lean) else None
        identity = (not is_decode) and self.options.state_path == "identity" and meta.identity_slots is True
        # Fused paths (Lane 3), each enabled only after its bitwise self-check passed.
        fused_store = flags.get("kv_store", False)
        q_per_prog, k_per_prog = self._rope_grouping(total) if fused_store else (1, 1)
        rope_cuda = (not is_decode) and total >= 256 and flags.get("rope_cuda", False)
        rope_cuda_launch = L.rope_kvstore_cuda4 if self.options.kv_store == "cuda4" else L.rope_kvstore_cuda
        conv_split = (not is_decode) and flags.get("conv_split", False)
        conv_tile = conv_split and self.options.conv_split in ("tile", "tile_norm")
        conv_l2norm = (not is_decode) and flags.get("conv_l2norm", False)
        # Lane 3 turn 4: loop conv (L2 norm fused) for prefills whose metadata carries this call's worklist.
        conv_loop = (not is_decode) and flags.get("conv_loop", False) and meta.conv_worklist is not None
        joint_norm = (not is_decode) and flags.get("qk_norm", False)
        gating_exp = (not is_decode) and flags.get("gating", False)
        norm_rows = None
        if not is_decode and self.options.norm_rows in ("prefill8", "token16") and K.calc_rows_per_block(total * 48, self.device) == 4:
            norm_rows = 8
        # Token-major gated norm (Lane 3 turn 3): eligibility is checked once per call; z is the [T, 48, 128]
        # view of the contiguous [T, 16384] qkvz projection in every GDN layer (stride 16384).
        norm_token = (not is_decode) and total >= 256 and flags.get("norm_token", False) and total * 16384 < 2 ** 31
        recurrent_bv = 32
        session = None
        if not is_decode:
            if issue_lean and self._gdn_direct is not None:
                gdn_direct, cu32 = self._gdn_direct, meta.query_start

                def gdn(q, k, v, gate, beta, initial, final):
                    return gdn_direct(q, k, v, gate, beta, initial, final, cu32)
            else:
                def gdn(q, k, v, gate, beta, initial, final):
                    return self._gdn_wrapper(meta, q, k, v, gate, beta, initial, final)
        hidden = F.embedding(meta.tokens, model.embedding)
        residual = None
        for i, (kind, w) in enumerate(zip(model.layer_types, model.layers)):
            if residual is None:
                residual = hidden
                hidden = norm(hidden, w["input_norm"], eps)
            else:
                add_norm(hidden, residual, w["input_norm"], eps)
            if kind == "full_attention":
                projected = linear(hidden, w["qkv"])
                if session is not None:
                    session.issue(i, "out")
                qg, k, v = projected.split([12288, 1024, 1024], dim=-1)
                kc, vc = cache.kv[i]
                if rope_cuda:
                    # Lane 3 turn 3/4: QK-norm, RoPE, gate pass-through and the K/V cache writes in one CUDA kernel (one warp per
                    # head, or four heads per warp with kv_store="cuda4").
                    q, gate = rope_cuda_launch(qg, k, v, w["q_norm"], w["k_norm"], model.rope, meta.positions, meta.locations, kc.view(-1, 1024), vc.view(-1, 1024),
                                               eps, stream)
                elif fused_store:
                    # QK-norm + RoPE with the rotated keys and the values written straight into the cache rows.
                    if lean:
                        q, gate = L.rope_kvstore(qg, k, v, w["q_norm"], w["k_norm"], model.rope, meta.positions, meta.locations, kc.view(-1, 1024), vc.view(-1, 1024),
                                                 eps, q_per_prog, k_per_prog, stream)
                    else:
                        q, gate = K3.fused_qk_gemma_rmsnorm_rope_gate_kvstore(qg, k, v, w["q_norm"], w["k_norm"], model.rope, meta.positions, meta.locations,
                                                                              kc.view(-1, 1024), vc.view(-1, 1024), eps, 24, 4, 256, model.rotary_dim,
                                                                              q_per_prog=q_per_prog, k_per_prog=k_per_prog)
                else:
                    if lean:
                        q, k, gate = L.rope_gate(qg, k, w["q_norm"], w["k_norm"], model.rope, meta.positions, eps, stream)
                    else:
                        q, k, gate = K.fused_qk_gemma_rmsnorm_rope_gate(qg, k, w["q_norm"], w["k_norm"], model.rope, meta.positions, eps, 24, 4, 256, model.rotary_dim, has_gate=True)
                    ops.store(k.view(-1, 1024), v, kc.view(-1, 1024), vc.view(-1, 1024), meta.locations, 4, kc.shape[0], 0)
                paged_kv = tuple(t.view(-1, self.page_size, 4, 256).permute(0, 2, 1, 3) for t in (kc, vc))
                attention_args = dict(query=q.view(-1, 24, 256), kv_cache=paged_kv, workspace_buffer=model.workspace, block_tables=meta.page_table,
                                      seq_lens=meta.sequence_lengths, bmm1_scale=256 ** -0.5, bmm2_scale=1.0, window_left=-1, sinks=None,
                                      skip_softmax_threshold_scale_factor=None, out_dtype=model.dtype)
                if is_decode:
                    out = model._decode_attention(**attention_args, max_seq_len=model.max_context, multi_ctas_kv_counter_buffer=model._attention_counter)
                else:
                    out = model._context_attention(**attention_args, max_q_len=max(lens), max_kv_len=model.max_context, batch_size=batch,
                                                   cum_seq_lens_q=meta.query_start, cum_seq_lens_kv=meta.kv_starts)
                out = out.view(-1, 6144)
                out = L.sigmoid_mul(out, gate.view(-1, 6144), stream) if lean else K.fused_sigmoid_mul(out, gate.view(-1, 6144), inplace=True)
                hidden = linear(out, w["out"])
            else:
                qkvz = linear(hidden, w["qkvz"])
                ba = linear(hidden, w["ba"])
                if session is not None:
                    session.issue(i, "out")
                if is_decode and lean:
                    mixed, z, b, a = L.conv_update(qkvz, ba, cache.conv[i], w["conv"], meta.state_indices, stream)
                    out = mixed.new_empty(batch, 1, 48, 128)
                    L.recurrent_decode(mixed, a, b, w["A_log"], w["dt_bias"], cache.recurrent[i], out, meta.state_indices, stream, bv=recurrent_bv)
                    z = z.reshape(-1, 128)
                elif is_decode:
                    mixed, z, b, a = K.fused_qkvzba_causal_conv1d_update_contiguous(qkvz, ba, cache.conv[i], w["conv"], None, meta.state_indices, qkv_dim=10240, v_dim=6144, num_v_heads=48, head_v_dim=128, activation="silu")
                    out = mixed.new_empty(batch, 1, 48, 128)
                    K.fused_recurrent_gated_delta_rule_packed_decode(mixed_qkv=mixed, a=a, b=b, A_log=w["A_log"], dt_bias=w["dt_bias"], scale=128 ** -0.5, initial_state=cache.recurrent[i], out=out, ssm_state_indices=meta.state_indices, use_qk_l2norm_in_kernel=True)
                    z = z.reshape(-1, 128)
                else:
                    mixed, z, b, a = K.qwen3_5_gdn_prefill_projection_views(qkvz, ba, 16, 48, 128, 128)
                    if conv_loop:
                        # Lane 3 turn 4: sub-tile loop conv writing L2-normalised q/k and v token-major; the state update is inside.
                        weight_t = self._conv_weight_t(i, w["conv"])
                        if lean:
                            q, k, v = L.conv_loop(mixed.T, weight_t, cache.conv[i], meta.query_start, lens, meta.state_indices, meta.has_initial,
                                                  meta.conv_worklist, meta.conv_block_t, stream)
                        else:
                            q, k, v = KL.causal_conv1d_split_loop_fn(mixed.T, w["conv"], None, cache.conv[i], meta.query_start, lens, meta.state_indices,
                                                                    meta.has_initial, "silu", 16, 16, 48, 128, block_t=meta.conv_block_t, block_n=128,
                                                                    num_warps=KL.CONV_NUM_WARPS, feat_first=False, fuse_l2norm=True, sub_t=KL.CONV_SUB_T,
                                                                    worklist=meta.conv_worklist, use_worklist=True, weight_t=weight_t)
                    elif conv_tile:
                        if lean:
                            q, k, v = L.conv_tile(mixed.T, w["conv"], cache.conv[i], meta.query_start, lens, meta.state_indices, meta.has_initial, stream,
                                                 fuse_l2norm=conv_l2norm)
                        else:
                            q, k, v = KT.causal_conv1d_split_tile_fn(mixed.T, w["conv"], None, cache.conv[i], meta.query_start, lens, meta.state_indices, meta.has_initial,
                                                                  "silu", 16, 16, 48, 128, block_t=32, block_n=128, num_warps=4, feat_first=False,
                                                                  fuse_l2norm=conv_l2norm)
                    elif conv_split:
                        # The conv1d writes token-major q/k/v directly (no channel-major intermediate, no split kernel).
                        if lean:
                            q, k, v = L.conv_split(mixed.T, w["conv"], cache.conv[i], meta.query_start, lens, meta.state_indices, meta.has_initial, stream)
                        else:
                            q, k, v = K3.causal_conv1d_split_fn(mixed.T, w["conv"], None, cache.conv[i], meta.query_start, lens, meta.state_indices, meta.has_initial, "silu", 16, 16, 48, 128)
                    elif lean:
                        mixed = L.causal_conv1d(mixed.T, w["conv"], cache.conv[i], meta.query_start, lens, meta.state_indices, meta.has_initial, stream).T
                        q, k, v = L.qkv_split(mixed, stream)
                    else:
                        mixed = K.causal_conv1d_fn(mixed.T, w["conv"], None, cache.conv[i], meta.query_start, lens, cache_indices=meta.state_indices, has_initial_state=meta.has_initial, activation="silu").T
                        q, k, v = K.fused_qkv_split_gdn_prefill(mixed, 16, 16, 48, 128, 128, 128)
                    # q/k/v are contiguous, so the upstream prepare step reduces to the two l2norms.
                    if conv_l2norm or conv_loop:
                        q, k, v = q[0], k[0], v[0]
                    elif joint_norm:
                        q, k = L.joint_l2norm(q[0], k[0], 1e-6, stream) if lean else K3.joint_l2norm(q[0], k[0], 1e-6)
                        v = v[0]
                    elif lean:
                        q, k, v = L.l2norm(q[0], stream=stream), L.l2norm(k[0], stream=stream), v[0]
                    else:
                        q, k, v = K.gdn_prefill_qkv_prepare_fwd(q[0], k[0], v[0])
                    if gating_exp:
                        # The gating kernel emits exp(g) itself (libdevice exp equals torch.exp bitwise).
                        g, beta = L.gating_blocked_exp(w["A_log"], a, b, w["dt_bias"], stream) if lean else K3.fused_gdn_gating_blocked(w["A_log"], a, b, w["dt_bias"], output_exp=True)
                        gate, beta0 = g[0], beta[0]
                    else:
                        g, beta = L.gdn_gating(w["A_log"], a, b, w["dt_bias"], stream) if lean else K.fused_gdn_gating(w["A_log"], a, b, w["dt_bias"])
                        gate, beta0 = torch.exp(g[0].float()), beta[0].float()
                    if identity:
                        # Active slots are exactly the cache's slots in order: the kernel reads the cache
                        # tensor directly and writes a fresh final-state tensor that replaces it.
                        initial = cache.recurrent[i]
                        final = torch.empty_like(initial)
                        out = gdn(q, k, v, gate, beta0, initial, final)
                        cache.recurrent[i] = final
                    else:
                        slots64 = meta.slots64 if meta.slots64 is not None else meta.state_indices.to(torch.int64)
                        initial = cache.recurrent[i][slots64].contiguous()
                        final = torch.empty_like(initial)
                        out = gdn(q, k, v, gate, beta0, initial, final)
                        cache.recurrent[i].index_copy_(0, slots64, final)
                if norm_token:
                    x_norm = out.reshape(-1, 128)
                    out = L.rms_norm_gated_token(x_norm, w["norm"], z, eps, stream) if lean else T3.rms_norm_gated_token(x_norm, w["norm"], z, eps)
                elif lean:
                    out = L.rms_norm_gated(out.reshape(-1, 128), w["norm"], z, eps, stream, norm_rows)
                elif norm_rows is not None:
                    out = K3.rms_norm_gated_rows(out.reshape(-1, 128), w["norm"], z, eps, norm_rows)
                else:
                    out = K.rms_norm_gated(x=out.reshape(-1, 128), weight=w["norm"], bias=None, z=z, eps=eps, norm_before_gate=True, is_rms_norm=True, activation="swish")
                hidden = linear(out.reshape(-1, 6144), w["out"])
            if session is not None:
                session.issue(i, "gate_up")
            add_norm(hidden, residual, w["post_norm"], eps)
            gate_up = linear(hidden, w["gate_up"])
            if session is not None:
                session.issue(i, "down")
            activated = torch.empty((total, 17408), dtype=model.dtype, device=self.device)
            ops.silu(gate_up, activated, "silu")
            hidden = linear(activated, w["down"])
            if session is not None:
                session.issue(i + 1, "in")
        add_norm(hidden, residual, model.final_norm, eps)
        if meta.last_indices is not None:
            # Prefill: keep each request's last position. Decode rows already are the last positions.
            hidden = hidden[meta.last_indices]
        logits = linear(hidden, model.lm_head)
        if session is not None:
            session.finish()
        return logits.float()

    @torch.inference_mode()
    def forward_step(self, input_ids, kv_cache, *, request_indices=None, return_all_logits=False):
        """Same contract as Qwen38.forward_step; see that docstring."""
        model = self.model
        rows, _ = K._token_batch(input_ids)
        cache = kv_cache
        if not isinstance(cache, HybridCache) or cache.owner != id(model):
            raise ValueError("kv_cache must have been created by this model")
        slots = list(range(len(cache.lengths))) if request_indices is None else list(request_indices)
        if len(slots) != len(rows) or len(set(slots)) != len(slots) or any(type(s) is not int or s < 0 or s >= len(cache.lengths) for s in slots):
            raise ValueError("request_indices must identify one distinct cache slot per input sequence")
        flat = [t for row in rows for t in row]
        if any(t < 0 or t >= self.vocab_size for t in flat):
            raise ValueError("Token ID outside model vocabulary")
        lens = [len(r) for r in rows]
        prefixes = [cache.lengths[s] for s in slots]
        lengths = [p + n for p, n in zip(prefixes, lens)]
        self._reserve(cache, max(lengths))
        is_decode = all(n == 1 and p > 0 for n, p in zip(lens, prefixes))
        batch = len(slots)
        total = len(flat)
        if is_decode and self.options.graphs == "decode":
            logits = self._decode_with_graph(cache, flat, slots, prefixes, lengths, batch)
            if logits is not None:
                for slot, length in zip(slots, lengths):
                    cache.lengths[slot] = length
                return logits
        if not is_decode:
            logits = self._prefill_with_graph(cache, flat, slots, lens, prefixes, lengths, batch, total, return_all_logits)
            if logits is not None:
                for slot, length in zip(slots, lengths):
                    cache.lengths[slot] = length
                return logits
        cu = [0]
        kv_cu = [0]
        for n, length in zip(lens, lengths):
            cu.append(cu[-1] + n)
            kv_cu.append(kv_cu[-1] + length)
        meta = self._metadata(flat, slots, lens, prefixes, lengths, cu, kv_cu, cache.capacity, is_decode, return_all_logits)
        meta.identity_slots = slots == list(range(len(cache.lengths)))
        logits = self._body(cache, meta, is_decode, lens, batch, total, return_all_logits)
        for slot, length in zip(slots, lengths):
            cache.lengths[slot] = length
        return logits

    # ----------------------------------------------------------------- graphs
    def _prefill_with_graph(self, cache, flat, slots, lens, prefixes, lengths, batch, total, return_all_logits):
        """Optional subclass hook after validation and growth; None selects ordinary prefill."""
        return None

    @staticmethod
    def _fingerprint(cache):
        return (tuple(t.data_ptr() for pair in cache.kv.values() for t in pair)
                + tuple(t.data_ptr() for t in cache.conv.values())
                + tuple(t.data_ptr() for t in cache.recurrent.values()))

    def _fill_static(self, entry, flat, slots, prefixes, lengths):
        ps = self.page_size
        batch = entry.batch
        pages = entry.capacity // ps
        slots_np = np.asarray(slots, dtype=np.int64)
        prefixes_np = np.asarray(prefixes, dtype=np.int64)
        lengths_np = np.asarray(lengths, dtype=np.int64)
        entry.event.synchronize()
        h64 = entry.host64.numpy()
        h64[0:batch] = flat
        h64[batch:2 * batch] = prefixes_np
        h64[2 * batch:3 * batch] = ps + slots_np * entry.capacity + prefixes_np
        h32 = entry.host32.numpy()
        h32[0:batch] = slots_np
        h32[batch:2 * batch] = lengths_np
        columns = np.arange(pages, dtype=np.int64)[None, :]
        pages_needed = ((lengths_np + ps - 1) // ps)[:, None]
        h32[2 * batch:] = np.where(columns < pages_needed, 1 + slots_np[:, None] * pages + columns, 0).reshape(-1)
        entry.dev64.copy_(entry.host64, non_blocking=True)
        entry.dev32.copy_(entry.host32, non_blocking=True)
        entry.event.record()

    def _decode_with_graph(self, cache, flat, slots, prefixes, lengths, batch):
        """Replay (or lazily capture) the decode graph for this cache/batch; None means run eagerly."""
        return self._decode_with_cache_graph(cache, flat, slots, prefixes, lengths, batch, self.options.graph_warm_calls)

    def _decode_with_cache_graph(self, cache, flat, slots, prefixes, lengths, batch, warm_calls):
        """Per-cache graphs: capture on the ``warm_calls``-th decode of one (cache, capacity, batch, addresses) signature."""
        key = (id(cache), cache.capacity, batch, self._fingerprint(cache))
        entry = self._graphs.get(key)
        if entry is not None and entry.cache_ref() is not cache:
            del self._graphs[key]
            entry = None
        if entry is None:
            record = self._graph_seen.get(key)
            # A freed cache's id() and tensor addresses can be reused by a new cache: the warm count and a
            # capture-failure mark belong to the cache object, so a dead or different referent restarts at one.
            seen = record[1] + 1 if record is not None and record[0]() is cache else 1
            if len(self._graph_seen) > 4096:
                self._graph_seen.clear()
            self._graph_seen[key] = (weakref.ref(cache), seen)
            if seen < warm_calls or seen < 0:
                self.graph_stats["eager_decodes"] += 1
                return None
            entry = self._capture(cache, key, flat, slots, prefixes, lengths, batch)
            if entry is None:
                return None
        else:
            self._fill_static(entry, flat, slots, prefixes, lengths)
        self._graph_tick += 1
        entry.tick = self._graph_tick
        entry.graph.replay()
        self.graph_stats["replays"] += 1
        return entry.output.clone()

    def _capture(self, cache, key, flat, slots, prefixes, lengths, batch):
        self._evict_graphs()
        entry = _GraphEntry(cache, batch, cache.capacity, self.device)
        self._fill_static(entry, flat, slots, prefixes, lengths)
        graph = torch.cuda.CUDAGraph()
        try:
            # Manual capture: torch.cuda.graph() would call torch.cuda.empty_cache() (and free the pinned
            # host cache) before every capture, which forces later eager calls back into cudaMalloc.
            torch.cuda.synchronize()
            if self.direct is None:
                self._bind_direct()  # self-checks must run eagerly, never inside a capture
            with torch.cuda.stream(self._capture_stream):
                graph.capture_begin(capture_error_mode="thread_local")
                try:
                    output = self._body(cache, entry.meta, True, [1] * batch, batch, batch, False, graph_dense=True)
                finally:
                    graph.capture_end()
            torch.cuda.synchronize()
        except Exception:  # noqa: BLE001 - fall back to eager execution for this signature
            self.graph_stats["capture_failures"] += 1
            self._graph_seen[key] = (weakref.ref(cache), -(1 << 30))
            torch.cuda.synchronize()
            return None
        entry.graph = graph
        entry.output = output
        self._graphs[key] = entry
        self.graph_stats["captures"] += 1
        return entry

    def _evict_graphs(self):
        dead = [key for key, entry in self._graphs.items() if entry.cache_ref() is None]
        for key in dead:
            del self._graphs[key]
        while len(self._graphs) >= self.options.max_graphs:
            oldest = min(self._graphs, key=lambda k: self._graphs[k].tick)
            del self._graphs[oldest]

    # --------------------------------------------------------------- generate
    @torch.inference_mode()
    def generate(self, input_ids, *, max_new_tokens=None, eos_token_ids=None, prefill_chunk_size=512):
        """Same contract as Qwen38.generate, executed through this engine."""
        model = self.model
        rows, single = K._token_batch(input_ids)
        if max_new_tokens is not None and (type(max_new_tokens) is not int or max_new_tokens < 0):
            raise ValueError("max_new_tokens must be a nonnegative integer or None")
        if type(prefill_chunk_size) is not int or prefill_chunk_size < 1:
            raise ValueError("prefill_chunk_size must be a positive integer")
        eos = model.eos_token_ids if eos_token_ids is None else ({eos_token_ids} if isinstance(eos_token_ids, int) else set(eos_token_ids))
        if not eos or any(type(t) is not int or t < 0 or t >= self.vocab_size for t in eos):
            raise ValueError("eos_token_ids must contain valid vocabulary IDs")
        output = [[] for _ in rows]
        if max_new_tokens == 0:
            return output[0] if single else output
        cache = model.new_cache(len(rows))
        if any(len(row) > cache.max_context for row in rows):
            raise ValueError("Input exceeds model context")
        last_logits = {}
        for offset in range(0, max(map(len, rows)), prefill_chunk_size):
            slots = [i for i, row in enumerate(rows) if offset < len(row)]
            chunks = [rows[i][offset:offset + prefill_chunk_size] for i in slots]
            logits = self.forward_step(chunks, cache, request_indices=slots)
            for j, slot in enumerate(slots):
                last_logits[slot] = logits[j]
        active = list(range(len(rows)))
        logits = torch.stack([last_logits[i] for i in active])
        # Decode steps repeat the same (cache, batch) signature many times, where graph replay pays off
        # (validated bit-exact by solution.probe_graphs); prefill chunks above stay eager.
        decoder = self if self.options.graphs == "decode" else get_engine(model, dataclasses.replace(self.options, graphs="decode"))
        while active:
            tokens = logits.argmax(-1).tolist()
            next_active, next_inputs = [], []
            for slot, token in zip(active, tokens):
                output[slot].append(token)
                if token in eos or (max_new_tokens is not None and len(output[slot]) >= max_new_tokens):
                    continue
                if cache.lengths[slot] >= cache.max_context:
                    raise RuntimeError(f"Request {slot} exhausted context without EOS; produced {len(output[slot])} tokens")
                next_active.append(slot)
                next_inputs.append([token])
            active = next_active
            if active:
                logits = decoder.forward_step(next_inputs, cache, request_indices=active)
        return output[0] if single else output


class _ModelEngineTables:
    """Keep engine tables alive with their model, without a global ownership root.

    A WeakKeyDictionary is insufficient: its strong values contain engines that
    hold the model, so the weak key never dies. The model owns this private table
    instead. Engines keep their normal strong model reference; when neither is
    held by the caller, Python can collect the model/table/engine cycle. Existing
    model weights, configuration and reference methods are unaffected.
    """

    _attribute = "_qwen38_solution_engines"

    def get(self, model, default=None):
        return vars(model).get(self._attribute, default)

    def __setitem__(self, model, table):
        vars(model)[self._attribute] = table

    def setdefault(self, model, default):
        return vars(model).setdefault(self._attribute, default)


_ENGINES = _ModelEngineTables()


def get_engine(model, options: EngineOptions | None = None) -> Engine:
    """Return the engine bound to ``model`` for these options (created on first use)."""
    options = options or EngineOptions()
    per_model = _ENGINES.get(model)
    if per_model is None:
        per_model = _ENGINES[model] = {}
    engine = per_model.get(options)
    if engine is None:
        engine = per_model[options] = Engine(model, options)
    return engine


def run(model, input_ids, kv_cache, **kwargs):
    """Candidate forward_step with the default engine options."""
    return get_engine(model).forward_step(input_ids, kv_cache, **kwargs)


def generate(model, input_ids, **kwargs):
    """Candidate generate with the default engine options."""
    return get_engine(model).generate(input_ids, **kwargs)
