"""Phase 5: IBM suite, gates, sweep aggregation."""

from __future__ import annotations

from pathlib import Path

import pytest

from hrt_chip.benchmarks import (
    AGGREGATE_REPLACE_PROXY,
    AGGREGATE_SA_PROXY,
    IBM_BENCHMARKS,
    BenchmarkRow,
    SweepReport,
    build_sweep_report,
    evaluate_gates,
)
from hrt_chip.benchmark_sweep import run_ibm_benchmark_sweep
from hrt_chip.config import RunConfig


def test_ibm_benchmarks_count_and_order() -> None:
    assert len(IBM_BENCHMARKS) == 17
    assert "ibm05" not in IBM_BENCHMARKS
    assert IBM_BENCHMARKS[0] == "ibm01"
    assert IBM_BENCHMARKS[-1] == "ibm18"


def test_gate_a_requires_all_legal() -> None:
    rows = [
        BenchmarkRow("ibm01", 1.0, True, 0, 1.0),
        BenchmarkRow("ibm02", 2.0, False, 1, 1.0),
    ]
    g = evaluate_gates(rows=rows)
    assert g.gate_a_legal_all is False
    assert g.legal_count == 1


def test_gate_b_c_mean_proxy() -> None:
    mean_ok = (AGGREGATE_SA_PROXY + AGGREGATE_REPLACE_PROXY) / 2  # between SA and RePlAce
    rows = [
        BenchmarkRow("ibm01", mean_ok, True, 0, 1.0),
    ]
    g = evaluate_gates(rows=rows)
    assert g.mean_proxy == pytest.approx(mean_ok)
    assert g.gate_b_beat_sa_aggregate is (mean_ok < AGGREGATE_SA_PROXY)
    assert g.gate_c_beat_replace_aggregate is (mean_ok < AGGREGATE_REPLACE_PROXY)


def test_build_sweep_report_json_shape() -> None:
    rows = [BenchmarkRow("ibm01", 1.5, True, 0, 0.5)]
    r = build_sweep_report(rows, evaluator_backend="stub", sweep_id="s1")
    d = r.to_dict()
    assert d["sweep_id"] == "s1"
    assert d["evaluator_backend"] == "stub"
    assert "gate_a_legal_all" in d
    assert len(d["rows"]) == 1


def test_run_ibm_benchmark_sweep_stub_subset(tmp_path: Path) -> None:
    base = RunConfig(
        benchmark_id="ibm01",
        seed=0,
        num_candidates=1,
        output_dir=tmp_path,
        evaluator_backend="stub",
        guidance_preset=None,
    )
    report, meta = run_ibm_benchmark_sweep(
        base,
        sweep_output_dir=tmp_path,
        sweep_id="test_sweep",
        benchmarks=("ibm01",),
    )
    assert isinstance(report, SweepReport)
    assert report.sweep_id == "test_sweep"
    assert len(report.rows) == 1
    assert report.rows[0].benchmark_id == "ibm01"
    assert report.rows[0].error is None
    assert (meta["sweep_root"] / "sweep_report.json").is_file()
    assert (meta["sweep_root"] / "ibm01").is_dir()
