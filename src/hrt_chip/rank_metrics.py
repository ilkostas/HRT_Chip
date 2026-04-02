"""Deterministic rank correlation helpers (Spearman ρ, Kendall τ) without SciPy."""

from __future__ import annotations

import math
from typing import Sequence


def _average_ranks(values: Sequence[float]) -> list[float]:
    """1-based average ranks with tie handling."""
    n = len(values)
    idx = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        v = values[idx[i]]
        while j + 1 < n and values[idx[j + 1]] == v:
            j += 1
        # positions i..j in sorted order get average rank (1-based)
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[idx[k]] = avg
        i = j + 1
    return ranks


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Spearman correlation; ``None`` if undefined (n<2, constant ranks)."""
    n = len(x)
    if n < 2 or len(y) != n:
        return None
    rx = _average_ranks(list(x))
    ry = _average_ranks(list(y))
    mx = sum(rx) / n
    my = sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    vx = sum((r - mx) ** 2 for r in rx)
    vy = sum((r - my) ** 2 for r in ry)
    if vx <= 0.0 or vy <= 0.0:
        return None
    return float(cov / math.sqrt(vx * vy))


def kendall_tau(x: Sequence[float], y: Sequence[float]) -> float | None:
    """
    Kendall's tau-b style denominator for ties; simple tau when no ties in x or y.

    For small candidate batches (Phase 3), O(n²) is acceptable.
    """
    n = len(x)
    if n < 2 or len(y) != n:
        return None

    concordant = 0
    discordant = 0
    tie_x = 0
    tie_y = 0
    tie_both = 0
    for i in range(n):
        for j in range(i + 1, n):
            cmp_x = x[i] - x[j]
            cmp_y = y[i] - y[j]
            if cmp_x == 0 and cmp_y == 0:
                tie_both += 1
                continue
            if cmp_x == 0:
                tie_x += 1
                continue
            if cmp_y == 0:
                tie_y += 1
                continue
            if (cmp_x > 0 and cmp_y > 0) or (cmp_x < 0 and cmp_y < 0):
                concordant += 1
            else:
                discordant += 1

    n0 = n * (n - 1) // 2
    denom = math.sqrt((n0 - tie_x) * (n0 - tie_y))
    if denom <= 0.0:
        return None
    return float((concordant - discordant) / denom)


def surrogate_good_proxy_bad_quartiles(
    candidate_ids: Sequence[str],
    composites: Sequence[float],
    proxies: Sequence[float],
    *,
    quartile_size: int | None = None,
) -> list[dict[str, float | str | int]]:
    """
    Candidates in the best surrogate quartile (lowest composite) and worst proxy quartile.

    ``quartile_size`` defaults to ``max(1, ceil(n/4))``.
    """
    n = len(candidate_ids)
    if n != len(composites) or n != len(proxies):
        return []
    if n == 0:
        return []

    k = quartile_size if quartile_size is not None else max(1, (n + 3) // 4)
    k = min(k, n)

    order_s = sorted(range(n), key=lambda i: composites[i])
    order_p = sorted(range(n), key=lambda i: proxies[i], reverse=True)

    best_s = set(order_s[:k])
    worst_p = set(order_p[:k])

    s_rank = {idx: r + 1 for r, idx in enumerate(order_s)}
    p_rank_worst = {idx: r + 1 for r, idx in enumerate(order_p)}

    out: list[dict[str, float | str | int]] = []
    for i in sorted(best_s & worst_p):
        sr = s_rank[i]
        pr = p_rank_worst[i]
        out.append(
            {
                "candidate_id": str(candidate_ids[i]),
                "surrogate_composite": float(composites[i]),
                "proxy_score": float(proxies[i]),
                "surrogate_rank_best_first": int(sr),
                "proxy_rank_worst_first": int(pr),
            }
        )
    return out
