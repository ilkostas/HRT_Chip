"""Phase 5: run all IBM benchmarks and aggregate gate status."""

from __future__ import annotations

import time
import traceback
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from hrt_chip.benchmarks import (
    AGGREGATE_REPLACE_PROXY,
    AGGREGATE_SA_PROXY,
    BenchmarkRow,
    IBM_BENCHMARKS,
    SweepReport,
    build_sweep_report,
)
from hrt_chip.config import RunConfig
from hrt_chip.io.artifacts import write_json
from hrt_chip.pipeline import run_pipeline


def run_ibm_benchmark_sweep(
    base_config: RunConfig,
    *,
    sweep_output_dir: Path,
    sweep_id: str | None = None,
    benchmarks: tuple[str, ...] | None = None,
) -> tuple[SweepReport, dict[str, Any]]:
    """
    Run ``run_pipeline`` once per benchmark; write per-benchmark artifacts under ``sweep_output_dir``.

    ``benchmarks`` defaults to all 17 IBM ids when ``None``; pass a subset for smoke / CI.

    Returns ``(SweepReport, extra_diagnostics)``.
    """
    sid = sweep_id or str(uuid.uuid4())
    sweep_root = sweep_output_dir / sid
    sweep_root.mkdir(parents=True, exist_ok=True)

    bench_list = benchmarks if benchmarks is not None else IBM_BENCHMARKS

    rows: list[BenchmarkRow] = []
    errors: list[dict[str, Any]] = []

    for bid in bench_list:
        run_dir = sweep_root / bid
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg = replace(base_config, benchmark_id=bid, output_dir=run_dir)
        t0 = time.perf_counter()
        try:
            results = run_pipeline(cfg)
            dt = time.perf_counter() - t0
            best_proxy = results.get("best_proxy_score")
            ranking = results.get("ranking") or []
            top = ranking[0] if ranking else {}
            legal = bool(top.get("legal", False))
            overlaps = (top.get("details") or {}).get("overlap_count")
            if overlaps is None and not legal:
                overlaps = 1
            elif overlaps is None:
                overlaps = 0
            rows.append(
                BenchmarkRow(
                    benchmark_id=bid,
                    proxy_score=float(best_proxy) if best_proxy is not None else None,
                    legal=legal,
                    overlaps=int(overlaps) if overlaps is not None else None,
                    runtime_seconds=dt,
                    error=None,
                    run_id=str((results.get("manifest") or {}).get("run_id", "")),
                )
            )
        except Exception as e:  # noqa: BLE001 — report row + continue
            dt = time.perf_counter() - t0
            errors.append({"benchmark_id": bid, "error": str(e), "traceback": traceback.format_exc()})
            rows.append(
                BenchmarkRow(
                    benchmark_id=bid,
                    proxy_score=None,
                    legal=False,
                    overlaps=None,
                    runtime_seconds=dt,
                    error=str(e),
                    run_id=None,
                )
            )

    report = build_sweep_report(
        rows,
        evaluator_backend=base_config.evaluator_backend,
        sweep_id=sid,
        extra={
            "sweep_output_dir": str(sweep_root),
            "errors": errors,
            "aggregate_sa_proxy": AGGREGATE_SA_PROXY,
            "aggregate_replace_proxy": AGGREGATE_REPLACE_PROXY,
        },
    )
    write_json(sweep_root / "sweep_report.json", report.to_dict())
    return report, {"sweep_root": sweep_root, "errors": errors}
