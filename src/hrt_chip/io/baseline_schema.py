"""Versioned contract for pipeline results and sweep artifacts (competition hardening).

Use this module to validate outputs before comparing runs or merging branches.
"""

from __future__ import annotations

from typing import Any

# Bump when adding required top-level keys or changing semantic meaning of scores.
RESULTS_JSON_SCHEMA_VERSION = "2"

# Sweep reports written by benchmark_sweep.
SWEEP_REPORT_SCHEMA_VERSION = "1"

REQUIRED_RESULTS_TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {
        "results_schema_version",
        "manifest",
        "benchmark_id",
        "evaluator_backend",
        "ranking",
        "scoring_table",
        "best_candidate_id",
        "best_proxy_score",
    }
)


def validate_results_json_shape(results: dict[str, Any]) -> list[str]:
    """Return list of validation errors; empty means OK for structural checks."""
    errors: list[str] = []
    missing = REQUIRED_RESULTS_TOP_LEVEL_KEYS - set(results.keys())
    if missing:
        errors.append(f"missing_keys: {sorted(missing)}")
    ver = results.get("results_schema_version")
    if ver != RESULTS_JSON_SCHEMA_VERSION:
        errors.append(
            f"results_schema_version mismatch: expected {RESULTS_JSON_SCHEMA_VERSION!r}, got {ver!r}"
        )
    ranking = results.get("ranking")
    if not isinstance(ranking, list):
        errors.append("ranking must be a list")
    else:
        for i, row in enumerate(ranking):
            if not isinstance(row, dict):
                errors.append(f"ranking[{i}] must be dict")
                continue
            for k in ("candidate_id", "proxy_score", "legal"):
                if k not in row:
                    errors.append(f"ranking[{i}] missing {k!r}")
    st = results.get("scoring_table")
    if not isinstance(st, list):
        errors.append("scoring_table must be a list")
    return errors


def attach_results_schema_version(results: dict[str, Any]) -> dict[str, Any]:
    """Mutate and return results with schema version (idempotent if already set)."""
    results["results_schema_version"] = RESULTS_JSON_SCHEMA_VERSION
    return results
