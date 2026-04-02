"""Rank correlation helpers (Spearman, Kendall, mismatch quartiles)."""

from __future__ import annotations

from hrt_chip.rank_metrics import kendall_tau, spearman_rho, surrogate_good_proxy_bad_quartiles


def test_spearman_perfect_positive() -> None:
    x = [1.0, 2.0, 3.0, 4.0]
    y = [10.0, 20.0, 30.0, 40.0]
    r = spearman_rho(x, y)
    assert r is not None
    assert abs(r - 1.0) < 1e-9


def test_spearman_perfect_negative() -> None:
    x = [1.0, 2.0, 3.0, 4.0]
    y = [40.0, 30.0, 20.0, 10.0]
    r = spearman_rho(x, y)
    assert r is not None
    assert abs(r + 1.0) < 1e-9


def test_spearman_ties() -> None:
    x = [1.0, 1.0, 3.0]
    y = [2.0, 2.0, 6.0]
    r = spearman_rho(x, y)
    assert r is not None
    assert abs(r - 1.0) < 1e-9


def test_kendall_tau_monotone() -> None:
    x = [0.0, 1.0, 2.0, 3.0]
    y = [0.0, 1.0, 2.0, 3.0]
    t = kendall_tau(x, y)
    assert t is not None
    assert t > 0.99


def test_surrogate_good_proxy_bad_quartiles() -> None:
    ids = ["a", "b", "c", "d"]
    comp = [0.0, 1.0, 2.0, 100.0]  # a best surrogate
    prox = [100.0, 1.0, 2.0, 0.0]  # a worst proxy
    bad = surrogate_good_proxy_bad_quartiles(ids, comp, prox, quartile_size=1)
    assert len(bad) >= 1
    assert bad[0]["candidate_id"] == "a"
