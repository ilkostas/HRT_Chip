"""Random legal macro placement (rejection sampling) for initialization."""

from __future__ import annotations

import random
from typing import Any

from hrt_chip.geometry import placement_is_legal
from hrt_chip.models import MacroRect, PlacementCandidate


def random_legal_candidate(
    *,
    benchmark_id: str,
    bench_obj: Any,
    canvas_w: float,
    canvas_h: float,
    rng: random.Random,
    candidate_id: str,
    max_attempts_per_macro: int = 8000,
) -> PlacementCandidate | None:
    """
    Place movable macros one-by-one in random order; each position is uniform random
    subject to legality with already-placed macros.
    """
    n = int(bench_obj.num_macros)
    names = [str(bench_obj.macro_names[i]) for i in range(n)]
    sizes = bench_obj.macro_sizes
    centers = bench_obj.macro_positions
    fixed_t = bench_obj.macro_fixed
    fixed_mask = [bool(fixed_t[i].item() if hasattr(fixed_t[i], "item") else fixed_t[i]) for i in range(n)]

    macros: list[MacroRect | None] = [None] * n
    for i in range(n):
        w = float(sizes[i, 0])
        h = float(sizes[i, 1])
        cx = float(centers[i, 0])
        cy = float(centers[i, 1])
        if fixed_mask[i]:
            macros[i] = MacroRect(name=names[i], x=cx - w / 2.0, y=cy - h / 2.0, w=w, h=h)

    movable = [i for i in range(n) if not fixed_mask[i]]
    if not movable:
        mlist = [m for m in macros if m is not None]
        return PlacementCandidate(
            candidate_id=candidate_id,
            benchmark_id=benchmark_id,
            macros=mlist,  # type: ignore[arg-type]
            metadata={"stage": "initialized", "seed_family": "random_legal"},
        )

    rng.shuffle(movable)
    for idx in movable:
        w = float(sizes[idx, 0])
        h = float(sizes[idx, 1])
        if w > canvas_w or h > canvas_h:
            return None
        placed: list[MacroRect] = [m for m in macros if m is not None]
        ok = False
        for _ in range(max_attempts_per_macro):
            x = rng.uniform(0.0, max(0.0, canvas_w - w))
            y = rng.uniform(0.0, max(0.0, canvas_h - h))
            trial = MacroRect(name=names[idx], x=x, y=y, w=w, h=h)
            trial_list = placed + [trial]
            if placement_is_legal(trial_list, canvas_w=canvas_w, canvas_h=canvas_h):
                macros[idx] = trial
                ok = True
                break
        if not ok:
            return None

    mlist = [m for m in macros if m is not None]
    assert len(mlist) == n
    return PlacementCandidate(
        candidate_id=candidate_id,
        benchmark_id=benchmark_id,
        macros=mlist,  # type: ignore[arg-type]
        metadata={"stage": "initialized", "seed_family": "random_legal"},
    )
