"""Runtime budget manager."""

import time
from pathlib import Path

from hrt_chip.config import RunConfig
from hrt_chip.runtime_budget import RuntimeBudgetManager


def test_manager_none_when_no_wall_budget() -> None:
    cfg = RunConfig(
        benchmark_id="ibm01",
        output_dir=Path("."),
        wall_clock_budget_seconds=None,
    )
    assert RuntimeBudgetManager.from_config(cfg, start_perf=0.0) is None


def test_can_generate_with_large_budget() -> None:
    cfg = RunConfig(
        benchmark_id="ibm01",
        output_dir=Path("."),
        wall_clock_budget_seconds=3600.0,
        evaluator_backend="stub",
        mixed_size_backend="stub",
        num_candidates=2,
    )
    t0 = time.perf_counter()
    m = RuntimeBudgetManager.from_config(cfg, start_perf=t0)
    assert m is not None
    assert m.can_generate_next_sweep_vector(cfg, num_candidates_this_vector=2, already_generated_unprocessed=0)
