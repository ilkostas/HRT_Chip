"""Regression: hard_macro_count must not exceed list length during overlap checks."""

from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest

from hrt_chip.geometry import count_overlapping_pairs, placement_is_legal
from hrt_chip.initializers.random_legal import random_legal_candidate
from hrt_chip.models import MacroRect


def test_placement_is_legal_clamps_hard_macro_count_past_list_len() -> None:
    macros = [
        MacroRect("a", 0.0, 0.0, 0.2, 0.2),
        MacroRect("b", 0.5, 0.5, 0.2, 0.2),
    ]
    assert placement_is_legal(macros, hard_macro_count=10_000) is True


def test_placement_is_legal_clamped_still_detects_overlap() -> None:
    macros = [
        MacroRect("a", 0.0, 0.0, 0.5, 0.5),
        MacroRect("b", 0.2, 0.2, 0.5, 0.5),
    ]
    assert placement_is_legal(macros, hard_macro_count=10_000) is False


def test_count_overlapping_pairs_clamps_hard_macro_count() -> None:
    macros = [
        MacroRect("a", 0.0, 0.0, 0.2, 0.2),
        MacroRect("b", 0.5, 0.5, 0.2, 0.2),
    ]
    assert count_overlapping_pairs(macros, hard_macro_count=10_000) == 0


def test_random_legal_candidate_ignores_inflated_num_hard_macros() -> None:
    """Official benchmarks set num_hard_macros to full design size; incremental trial_list is shorter."""
    n = 3
    bench = SimpleNamespace(
        num_macros=n,
        macro_names=np.array([f"m{i}" for i in range(n)], dtype=object),
        macro_sizes=np.ones((n, 2), dtype=np.float64) * 0.08,
        macro_positions=np.array([[0.04, 0.04], [0.5, 0.5], [0.9, 0.9]], dtype=np.float64),
        macro_fixed=[False, False, False],
        num_hard_macros=9999,
    )
    out = random_legal_candidate(
        benchmark_id="ibm01",
        bench_obj=bench,
        canvas_w=1.0,
        canvas_h=1.0,
        rng=random.Random(0),
        candidate_id="t0",
        max_attempts_per_macro=20_000,
    )
    assert out is not None
    assert len(out.macros) == n
    assert placement_is_legal(out.macros)
