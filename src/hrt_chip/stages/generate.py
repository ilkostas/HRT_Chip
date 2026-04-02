"""Stub candidate generation (deterministic under fixed seed)."""

from __future__ import annotations

import random
from typing import Sequence

from hrt_chip.models import MacroRect, PlacementCandidate


# Minimal stub netlist: two macros per benchmark for overlap/legalizer demos.
def _default_macro_names(benchmark_id: str) -> list[tuple[str, float, float]]:
    return [
        (f"{benchmark_id}_M0", 0.12, 0.08),
        (f"{benchmark_id}_M1", 0.10, 0.09),
    ]


def generate_candidates(
    *,
    benchmark_id: str,
    seed: int,
    num_candidates: int,
    macro_specs: Sequence[tuple[str, float, float]] | None = None,
) -> list[PlacementCandidate]:
    """
    Produce `num_candidates` placement hypotheses.

    Uses a seeded RNG so runs are reproducible when `seed` and `num_candidates` match.
    """
    rng = random.Random(seed)
    specs = list(macro_specs) if macro_specs is not None else _default_macro_names(benchmark_id)
    out: list[PlacementCandidate] = []
    for i in range(num_candidates):
        cid = f"cand_{i:04d}"
        macros: list[MacroRect] = []
        for name, w, h in specs:
            x = rng.uniform(0.0, 1.0 - w)
            y = rng.uniform(0.0, 1.0 - h)
            macros.append(MacroRect(name=name, x=x, y=y, w=w, h=h))
        out.append(
            PlacementCandidate(
                candidate_id=cid,
                benchmark_id=benchmark_id,
                macros=macros,
                metadata={"stage": "generated"},
            )
        )
    return out
