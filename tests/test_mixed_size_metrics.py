"""Unit tests for mixed-size batch normalization and ranking keys."""

from __future__ import annotations

from hrt_chip import mixed_size_metrics as msm


def test_min_max_normalize_handles_missing() -> None:
    assert msm.min_max_normalize([None, 0.0, 2.0]) == [1.0, 0.0, 1.0]


def test_min_max_normalize_degenerate_range() -> None:
    assert msm.min_max_normalize([1.0, 1.0, None]) == [0.5, 0.5, 1.0]


def test_build_profiles_composite_order() -> None:
    rows = [
        {
            "candidate_id": "a",
            "legal": True,
            "ms_ok": True,
            "ms_extra": {"density_overflow": 1.0, "rudy_or_route_proxy": 1.0, "backend_runtime_seconds": 1.0},
        },
        {
            "candidate_id": "b",
            "legal": True,
            "ms_ok": True,
            "ms_extra": {"density_overflow": 0.0, "rudy_or_route_proxy": 0.0, "backend_runtime_seconds": 0.0},
        },
    ]
    profs = msm.build_mixed_size_profiles_for_candidates(rows)
    assert profs["b"]["composite_ppa"] < profs["a"]["composite_ppa"]


def test_ranking_key_ppa_priority_prefers_composite_over_proxy() -> None:
    hi_proxy_good_comp = {
        "candidate_id": "x",
        "legal": True,
        "proxy_score": 1.0,
        "mixed_size_profile": {"backend_ok": True, "composite_ppa": 0.9},
    }
    lo_proxy_better_comp = {
        "candidate_id": "y",
        "legal": True,
        "proxy_score": 5.0,
        "mixed_size_profile": {"backend_ok": True, "composite_ppa": 0.1},
    }
    assert msm.ranking_key_ppa_priority(lo_proxy_better_comp) < msm.ranking_key_ppa_priority(hi_proxy_good_comp)


def test_ranking_key_proxy_first_prefers_proxy() -> None:
    a = {
        "candidate_id": "a",
        "legal": True,
        "proxy_score": 1.0,
        "mixed_size_profile": {"composite_ppa": 0.99},
    }
    b = {
        "candidate_id": "b",
        "legal": True,
        "proxy_score": 2.0,
        "mixed_size_profile": {"composite_ppa": 0.01},
    }
    assert msm.ranking_key_proxy_first(a) < msm.ranking_key_proxy_first(b)
