"""IBM ICCAD04 benchmark suite and Phase 5 milestone gates (aggregate proxy vs baselines)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal

# Canonical tier-1 IBM suite (ibm05 does not exist in ICCAD04 used by the challenge).
IBM_BENCHMARKS: tuple[str, ...] = (
    "ibm01",
    "ibm02",
    "ibm03",
    "ibm04",
    "ibm06",
    "ibm07",
    "ibm08",
    "ibm09",
    "ibm10",
    "ibm11",
    "ibm12",
    "ibm13",
    "ibm14",
    "ibm15",
    "ibm16",
    "ibm17",
    "ibm18",
)

# Aggregate (AVG) proxy baselines from competition docs / official evaluate harness.
# Used for Gate B and Gate C against leaderboard-style averages.
AGGREGATE_SA_PROXY: float = 2.1251
AGGREGATE_REPLACE_PROXY: float = 1.4578

# Per-design baselines (competition table / official evaluate harness).
SA_BASELINE_BY_DESIGN: dict[str, float] = {
    "ibm01": 1.3166,
    "ibm02": 1.9072,
    "ibm03": 1.7401,
    "ibm04": 1.5037,
    "ibm06": 2.5057,
    "ibm07": 2.0229,
    "ibm08": 1.9239,
    "ibm09": 1.3875,
    "ibm10": 2.1108,
    "ibm11": 1.7111,
    "ibm12": 2.8261,
    "ibm13": 1.9141,
    "ibm14": 2.2750,
    "ibm15": 2.3000,
    "ibm16": 2.2337,
    "ibm17": 3.6726,
    "ibm18": 2.7755,
}

REPLACE_BASELINE_BY_DESIGN: dict[str, float] = {
    "ibm01": 0.9976,
    "ibm02": 1.8370,
    "ibm03": 1.3222,
    "ibm04": 1.3024,
    "ibm06": 1.6187,
    "ibm07": 1.4633,
    "ibm08": 1.4285,
    "ibm09": 1.1194,
    "ibm10": 1.5009,
    "ibm11": 1.1774,
    "ibm12": 1.7261,
    "ibm13": 1.3355,
    "ibm14": 1.5436,
    "ibm15": 1.5159,
    "ibm16": 1.4780,
    "ibm17": 1.6446,
    "ibm18": 1.7722,
}


def default_testcase_root() -> str:
    """Root containing ``<benchmark_id>/netlist.pb.txt`` (MacroPlacement ICCAD04 testcases)."""
    return os.environ.get(
        "HRT_CHIP_TESTCASE_ROOT",
        "external/MacroPlacement/Testcases/ICCAD04",
    )


@dataclass
class BenchmarkRow:
    """One design in a sweep report."""

    benchmark_id: str
    proxy_score: float | None
    legal: bool
    overlaps: int | None
    runtime_seconds: float
    error: str | None = None
    run_id: str | None = None


@dataclass
class GateStatus:
    """Phase 5 milestone gates."""

    gate_a_legal_all: bool
    """100% legal placements (no overlaps / evaluator-valid)."""

    gate_b_beat_sa_aggregate: bool
    """Mean proxy strictly better than aggregate SA baseline."""

    gate_c_beat_replace_aggregate: bool
    """Mean proxy strictly better than aggregate RePlAce baseline."""

    mean_proxy: float | None
    legal_count: int
    total_count: int


@dataclass
class SweepReport:
    """Full IBM17 sweep + gate evaluation."""

    rows: list[BenchmarkRow]
    gates: GateStatus
    mean_runtime_seconds: float
    total_runtime_seconds: float
    aggregate_sa_proxy: float = AGGREGATE_SA_PROXY
    aggregate_replace_proxy: float = AGGREGATE_REPLACE_PROXY
    evaluator_backend: Literal["stub", "official"] = "stub"
    sweep_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sweep_id": self.sweep_id,
            "evaluator_backend": self.evaluator_backend,
            "aggregate_sa_proxy": self.aggregate_sa_proxy,
            "aggregate_replace_proxy": self.aggregate_replace_proxy,
            "mean_proxy": self.gates.mean_proxy,
            "mean_runtime_seconds": self.mean_runtime_seconds,
            "total_runtime_seconds": self.total_runtime_seconds,
            "legal_count": self.gates.legal_count,
            "total_count": self.gates.total_count,
            "gate_a_legal_all": self.gates.gate_a_legal_all,
            "gate_b_beat_sa_aggregate": self.gates.gate_b_beat_sa_aggregate,
            "gate_c_beat_replace_aggregate": self.gates.gate_c_beat_replace_aggregate,
            "rows": [
                {
                    "benchmark_id": r.benchmark_id,
                    "proxy_score": r.proxy_score,
                    "legal": r.legal,
                    "overlaps": r.overlaps,
                    "runtime_seconds": r.runtime_seconds,
                    "error": r.error,
                    "run_id": r.run_id,
                }
                for r in self.rows
            ],
            **self.extra,
        }


def evaluate_gates(
    *,
    rows: list[BenchmarkRow],
    aggregate_sa: float = AGGREGATE_SA_PROXY,
    aggregate_replace: float = AGGREGATE_REPLACE_PROXY,
) -> GateStatus:
    """Compute Gate A/B/C from per-benchmark rows."""
    ok_rows = [r for r in rows if r.error is None and r.proxy_score is not None]
    legal_count = sum(1 for r in ok_rows if r.legal)
    total_count = len(rows)
    gate_a = legal_count == total_count and total_count > 0

    proxies = [
        float(r.proxy_score)
        for r in ok_rows
        if r.legal
        and r.proxy_score is not None
        and r.proxy_score != float("inf")
    ]
    if proxies:
        mean_proxy = sum(proxies) / len(proxies)
    else:
        mean_proxy = None

    gate_b = mean_proxy is not None and mean_proxy < aggregate_sa
    gate_c = mean_proxy is not None and mean_proxy < aggregate_replace

    return GateStatus(
        gate_a_legal_all=gate_a,
        gate_b_beat_sa_aggregate=gate_b,
        gate_c_beat_replace_aggregate=gate_c,
        mean_proxy=mean_proxy,
        legal_count=legal_count,
        total_count=total_count,
    )


def build_sweep_report(
    rows: list[BenchmarkRow],
    *,
    evaluator_backend: Literal["stub", "official"],
    sweep_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> SweepReport:
    """Aggregate timings and compute gates."""
    gates = evaluate_gates(rows=rows)
    rt = [r.runtime_seconds for r in rows]
    total_rt = sum(rt)
    mean_rt = total_rt / len(rt) if rt else 0.0
    return SweepReport(
        rows=rows,
        gates=gates,
        mean_runtime_seconds=mean_rt,
        total_runtime_seconds=total_rt,
        evaluator_backend=evaluator_backend,
        sweep_id=sweep_id,
        extra=dict(extra or {}),
    )
