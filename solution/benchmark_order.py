"""Deterministic benchmark orders with auditable position and predecessor counts.

An even number of arms takes N rows; an odd number takes 2*N rows. Within each
complete block every arm occupies every position equally and every distinct
ordered pair occurs equally often as adjacent arms. Boundaries between cases or
repetitions are outside this balance guarantee. Every arm still runs the whole
original case sequence on its own fresh cache.

This module has no inference or timing dependencies.
"""
from __future__ import annotations

import random
from collections import Counter


def balanced_rows(n):
    if not isinstance(n, int) or n < 1:
        raise ValueError("Need a positive integer number of arms")
    if n == 1:
        return [[0]]
    row = [0]
    for i in range(1, n):
        row.append((i + 1) // 2 if i % 2 else n - i // 2)
    rows = [[(v + offset) % n for v in row] for offset in range(n)]
    if n % 2:
        rows += [list(reversed(r)) for r in rows]
    return rows


def make_orders(labels, case_names, repeats, mode="alternate", seed=10010):
    """Return case -> repetition -> ordered arm labels, plus the block size."""
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("Arm labels must be nonempty and distinct")
    if len(set(case_names)) != len(case_names):
        raise ValueError("Case names must be distinct")
    if repeats < 1 or mode not in ("alternate", "fixed", "balanced"):
        raise ValueError("Need positive repeats and a known order mode")
    rows = balanced_rows(len(labels)) if mode == "balanced" else None
    block_size = len(rows) if rows else (2 if mode == "alternate" else 1)
    if mode == "balanced" and repeats % block_size:
        raise ValueError(f"Balanced order with {len(labels)} arms requires repeats divisible by {block_size}")
    rng = random.Random(seed)
    orders = {}
    for name in case_names:
        if rows is None:
            orders[name] = [list(reversed(labels)) if mode == "alternate" and r % 2 else list(labels)
                            for r in range(repeats)]
            continue
        orders[name] = []
        for _ in range(repeats // block_size):
            indices = list(range(block_size))
            rng.shuffle(indices)
            permutation = list(labels)
            rng.shuffle(permutation)
            orders[name].extend([[permutation[i] for i in rows[j]] for j in indices])
    return orders, block_size


def order_counts(labels, orders):
    positions = {label: [0] * len(labels) for label in labels}
    previous = {label: {p: 0 for p in labels if p != label} for label in labels}
    for row in orders:
        if Counter(row) != Counter(labels):
            raise ValueError("Every repetition must contain every arm exactly once")
        for i, label in enumerate(row):
            positions[label][i] += 1
            if i:
                previous[label][row[i - 1]] += 1
    return {"positions": positions, "preceding_arm_within_case": previous}
