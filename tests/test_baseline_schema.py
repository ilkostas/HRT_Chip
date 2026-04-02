"""results.json schema contract (competition hardening)."""

from __future__ import annotations

from hrt_chip.io.baseline_schema import (
    RESULTS_JSON_SCHEMA_VERSION,
    attach_results_schema_version,
    validate_results_json_shape,
)


def test_validate_results_ok_minimal() -> None:
    r = {
        "results_schema_version": RESULTS_JSON_SCHEMA_VERSION,
        "manifest": {"run_id": "x", "config": {}},
        "benchmark_id": "ibm01",
        "evaluator_backend": "stub",
        "ranking": [{"candidate_id": "a", "proxy_score": 1.0, "legal": True}],
        "scoring_table": [],
        "best_candidate_id": "a",
        "best_proxy_score": 1.0,
    }
    assert validate_results_json_shape(r) == []


def test_validate_results_missing_key() -> None:
    r = {
        "results_schema_version": RESULTS_JSON_SCHEMA_VERSION,
        "manifest": {},
        "benchmark_id": "ibm01",
    }
    errs = validate_results_json_shape(r)
    assert any("missing_keys" in e for e in errs)


def test_attach_results_schema_version() -> None:
    r: dict = {"benchmark_id": "ibm01"}
    attach_results_schema_version(r)
    assert r["results_schema_version"] == RESULTS_JSON_SCHEMA_VERSION
