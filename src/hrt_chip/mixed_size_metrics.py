"""Batch-normalized mixed-size placement metrics for PPA-priority ranking (lower composite is better)."""

from __future__ import annotations

import math
from typing import Any

# Default weights for composite PPA surrogate (density, route congestion proxy, runtime).
WEIGHT_DENSITY = 0.45
WEIGHT_CONGESTION = 0.40
WEIGHT_RUNTIME = 0.15


def _to_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def extract_mixed_size_raw(extra: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    """
    Return (density_overflow, route_congestion_proxy, backend_runtime_s) from backend ``extra`` dict.
    Supports dreamplace, estimate, and dreamplace_real field names.
    """
    density = extra.get("density_overflow")
    if density is None:
        density = extra.get("density_overflow_proxy")
    rudy = extra.get("rudy_or_route_proxy")
    if rudy is None:
        rudy = extra.get("rudy_density_variance")
    rt = extra.get("backend_runtime_seconds")
    if rt is None:
        rt = extra.get("docker_host_elapsed_seconds")
    if rt is None:
        rt = extra.get("host_wall_seconds")
    return _to_float(density), _to_float(rudy), _to_float(rt)


def min_max_normalize(values: list[float | None]) -> list[float]:
    """
    Min-max normalize finite values to [0, 1]. None / non-finite → 1.0 (worst).
    If no finite samples, all entries become 1.0. Degenerate range uses 0.5 for finite values.
    """
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if not finite:
        return [1.0] * len(values)

    lo, hi = min(finite), max(finite)
    out: list[float] = []
    for v in values:
        if v is None or not math.isfinite(v):
            out.append(1.0)
            continue
        if hi <= lo:
            out.append(0.5)
        else:
            out.append((v - lo) / (hi - lo))
    return out


def composite_ppa(
    norm_density: float,
    norm_congestion: float,
    norm_runtime: float,
    *,
    w_d: float = WEIGHT_DENSITY,
    w_c: float = WEIGHT_CONGESTION,
    w_t: float = WEIGHT_RUNTIME,
) -> float:
    """Weighted sum; lower is better."""
    return w_d * norm_density + w_c * norm_congestion + w_t * norm_runtime


def build_mixed_size_profiles_for_candidates(
    rows: list[dict[str, Any]],
    *,
    w_d: float = WEIGHT_DENSITY,
    w_c: float = WEIGHT_CONGESTION,
    w_t: float = WEIGHT_RUNTIME,
) -> dict[str, dict[str, Any]]:
    """
    Batch-normalize metrics across candidates and return per-candidate profiles keyed by ``candidate_id``.

    Each ``rows`` item: ``candidate_id``, ``legal`` (bool), ``ms_ok`` (bool), ``ms_extra`` (dict).
    """
    d_raw: list[float | None] = []
    r_raw: list[float | None] = []
    t_raw: list[float | None] = []
    for r in rows:
        ex = r.get("ms_extra") if isinstance(r.get("ms_extra"), dict) else {}
        if r.get("legal") is True and r.get("ms_ok") is True:
            dd, rr, tt = extract_mixed_size_raw(ex)
        else:
            dd, rr, tt = None, None, None
        d_raw.append(dd)
        r_raw.append(rr)
        t_raw.append(tt)

    nd = min_max_normalize(d_raw)
    nr = min_max_normalize(r_raw)
    nt = min_max_normalize(t_raw)

    out: dict[str, dict[str, Any]] = {}
    for i, r in enumerate(rows):
        cid = str(r["candidate_id"])
        comp = composite_ppa(nd[i], nr[i], nt[i], w_d=w_d, w_c=w_c, w_t=w_t)
        out[cid] = {
            "backend_ok": r.get("ms_ok") is True,
            "raw_density_overflow": d_raw[i],
            "raw_route_congestion_proxy": r_raw[i],
            "raw_backend_runtime_seconds": t_raw[i],
            "norm_density_overflow": nd[i],
            "norm_route_congestion_proxy": nr[i],
            "norm_backend_runtime_seconds": nt[i],
            "composite_ppa": comp,
            "weights": {"density": w_d, "congestion": w_c, "runtime": w_t},
        }
    return out


def ranking_key_proxy_first(row: dict[str, Any]) -> tuple[float, float, str]:
    """Primary: proxy (lower better). Tie-break: composite PPA (lower better), then id."""
    ps = row.get("proxy_score")
    fp = float(ps) if isinstance(ps, (int, float)) and math.isfinite(float(ps)) else float("inf")
    prof = row.get("mixed_size_profile") or {}
    comp = prof.get("composite_ppa")
    fc = float(comp) if comp is not None and math.isfinite(float(comp)) else float("inf")
    return (fp, fc, str(row["candidate_id"]))


def ranking_key_ppa_priority(row: dict[str, Any]) -> tuple[int, float, float, str]:
    """
    Tier 0: legal + mixed-size backend ok → sort by composite PPA then proxy.
    Tier 1: legal but backend not ok → sort by proxy then composite.
    Tier 2: illegal / infinite proxy → last tier, by proxy then id.
    """
    legal = row.get("legal") is True
    ps = row.get("proxy_score")
    fp = float(ps) if isinstance(ps, (int, float)) and math.isfinite(float(ps)) else float("inf")
    prof = row.get("mixed_size_profile") or {}
    ms_ok = prof.get("backend_ok") is True
    comp = prof.get("composite_ppa")
    fc = float(comp) if comp is not None and math.isfinite(float(comp)) else float("inf")
    cid = str(row["candidate_id"])
    if not legal:
        return (2, fp, fc, cid)
    if not ms_ok:
        return (1, fp, fc, cid)
    return (0, fc, fp, cid)
