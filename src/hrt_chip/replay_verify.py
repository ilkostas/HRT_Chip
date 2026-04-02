"""Compare replayed pipeline output to a baseline run (Phase 6 deterministic verification)."""

from __future__ import annotations

import math
from typing import Any


def _proxy_equal(a: Any, b: Any) -> bool:
    if a == b:
        return True
    if isinstance(a, float) and isinstance(b, float):
        if math.isinf(a) and math.isinf(b) and (a > 0) == (b > 0):
            return True
        return math.isclose(a, b, rel_tol=0.0, abs_tol=1e-9)
    return False


def fingerprint_for_verification(results: dict[str, Any]) -> dict[str, Any]:
    """Stable subset of results used for regression / replay checks."""
    ranking = results.get("ranking") or []
    rows: list[dict[str, Any]] = []
    for r in ranking:
        rows.append(
            {
                "candidate_id": r.get("candidate_id"),
                "proxy_score": r.get("proxy_score"),
                "legal": r.get("legal"),
            }
        )
    return {
        "benchmark_id": results.get("benchmark_id"),
        "best_candidate_id": results.get("best_candidate_id"),
        "best_proxy_score": results.get("best_proxy_score"),
        "ranking_fingerprint": rows,
        "evaluator_backend": results.get("evaluator_backend"),
    }


def compare_replay_to_baseline(
    baseline: dict[str, Any],
    replay: dict[str, Any],
) -> dict[str, Any]:
    """
    Return a structured verification report. ``ok`` is True iff all checked fields match.
    """
    fb = fingerprint_for_verification(baseline)
    fr = fingerprint_for_verification(replay)
    mismatches: list[str] = []

    if fb.get("benchmark_id") != fr.get("benchmark_id"):
        mismatches.append(
            f"benchmark_id: baseline={fb.get('benchmark_id')!r} replay={fr.get('benchmark_id')!r}"
        )
    if fb.get("best_candidate_id") != fr.get("best_candidate_id"):
        mismatches.append(
            f"best_candidate_id: baseline={fb.get('best_candidate_id')!r} "
            f"replay={fr.get('best_candidate_id')!r}"
        )
    if not _proxy_equal(fb.get("best_proxy_score"), fr.get("best_proxy_score")):
        mismatches.append(
            f"best_proxy_score: baseline={fb.get('best_proxy_score')!r} "
            f"replay={fr.get('best_proxy_score')!r}"
        )

    br = fb.get("ranking_fingerprint") or []
    rr = fr.get("ranking_fingerprint") or []
    if len(br) != len(rr):
        mismatches.append(f"ranking length: baseline={len(br)} replay={len(rr)}")
    else:
        for i, (x, y) in enumerate(zip(br, rr, strict=True)):
            if x.get("candidate_id") != y.get("candidate_id"):
                mismatches.append(f"ranking[{i}].candidate_id: {x.get('candidate_id')!r} vs {y.get('candidate_id')!r}")
            if x.get("legal") != y.get("legal"):
                mismatches.append(f"ranking[{i}].legal: {x.get('legal')!r} vs {y.get('legal')!r}")
            if not _proxy_equal(x.get("proxy_score"), y.get("proxy_score")):
                mismatches.append(
                    f"ranking[{i}].proxy_score: {x.get('proxy_score')!r} vs {y.get('proxy_score')!r}"
                )

    return {
        "ok": len(mismatches) == 0,
        "mismatches": mismatches,
        "baseline_fingerprint": fb,
        "replay_fingerprint": fr,
    }
