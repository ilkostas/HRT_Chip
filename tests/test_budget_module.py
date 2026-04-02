"""Wall-clock budget resolution."""

from __future__ import annotations

from pathlib import Path

from hrt_chip.budget import resolve_generation_budget
from hrt_chip.config import RunConfig


def test_budget_unlimited_returns_requested() -> None:
    sweep = ((0.33, 0.33, 0.34), (0.5, 0.25, 0.25))
    cfg = RunConfig(
        num_candidates=4,
        wall_clock_budget_seconds=None,
        evaluator_backend="stub",
        mixed_size_backend="stub",
        output_dir=Path("."),
    )
    s, k, meta = resolve_generation_budget(cfg, sweep)
    assert s == sweep
    assert k == 4
    assert meta["budget_limited"] is False


def test_tiny_budget_limits_total() -> None:
    sweep = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    cfg = RunConfig(
        num_candidates=10,
        wall_clock_budget_seconds=0.2,
        evaluator_backend="stub",
        mixed_size_backend="stub",
        output_dir=Path("."),
    )
    s, k, meta = resolve_generation_budget(cfg, sweep)
    assert meta["budget_limited"] is True
    assert k >= 1
    assert len(s) * k < len(sweep) * cfg.num_candidates
