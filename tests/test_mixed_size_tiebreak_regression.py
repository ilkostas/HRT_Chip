"""Mixed-size profile tie-break keys stay stable when profiles differ (proxy-first)."""

from hrt_chip import mixed_size_metrics as msm


def test_ranking_key_proxy_first_orders_by_proxy_then_profile() -> None:
    a = {
        "candidate_id": "a",
        "proxy_score": 1.0,
        "legal": True,
        "mixed_size_profile": {"composite_ppa": 0.5},
    }
    b = {
        "candidate_id": "b",
        "proxy_score": 1.0,
        "legal": True,
        "mixed_size_profile": {"composite_ppa": 0.6},
    }
    rows = sorted([a, b], key=msm.ranking_key_proxy_first)
    assert [r["candidate_id"] for r in rows] == ["a", "b"]


def test_ranking_with_empty_profile() -> None:
    row = {
        "candidate_id": "x",
        "proxy_score": 2.0,
        "legal": True,
        "mixed_size_profile": {},
    }
    k = msm.ranking_key_proxy_first(row)
    assert k[0] == 2.0
