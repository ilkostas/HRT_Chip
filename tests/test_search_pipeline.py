"""Smoke tests for search-hybrid solver path."""

from __future__ import annotations

from pathlib import Path

from hrt_chip.config import RunConfig
from hrt_chip.pipeline import run_pipeline


def test_search_hybrid_stub_deterministic(tmp_path: Path) -> None:
    """Stub evaluator, no ICCAD: initializer falls back to stub generate; SA + eval are deterministic."""
    cfg = RunConfig(
        benchmark_id="ibm01",
        seed=11,
        output_dir=tmp_path,
        deterministic=True,
        mixed_size_backend="stub",
        evaluator_backend="stub",
        solver_backend="search_hybrid",
        wall_clock_budget_seconds=12.0,
        search_screen_seconds=2.0,
        search_max_iterations=4000,
        search_seeds_per_family=1,
        search_families=("benchmark_jitter",),
    )
    r1 = run_pipeline(cfg, run_id="00000000-0000-0000-0000-0000000000a1")
    r2 = run_pipeline(cfg, run_id="00000000-0000-0000-0000-0000000000a1")
    assert r1.get("solver_backend") == "search_hybrid"
    assert r1["best_candidate_id"] == r2["best_candidate_id"]
    assert r1["timing"].get("search_seconds") is not None
    assert len(r1["ranking"]) >= 1
    assert "search_telemetry" in r1["scoring_table"][0]


def test_legacy_unchanged(tmp_path: Path) -> None:
    cfg = RunConfig(
        benchmark_id="ibm01",
        seed=7,
        num_candidates=2,
        output_dir=tmp_path,
        deterministic=True,
        mixed_size_backend="stub",
        solver_backend="legacy",
    )
    r = run_pipeline(cfg, run_id="00000000-0000-0000-0000-0000000000a3")
    assert r.get("solver_backend", "legacy") == "legacy" or "guidance_sweep_resolved" in r
