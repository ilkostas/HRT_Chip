"""Inference-time objective guidance interfaces and deterministic surrogates (Phase 3).

Official Tier-1 selection uses the evaluator proxy only; these components are for
exploration metadata and future DDPM guidance hooks.

With a netlist ``Benchmark`` and **pin-group** fields (``net_pin_dx`` / ``net_pin_dy``),
HPWL is **exact weighted pin HPWL** and congestion uses **RUDY over pin bounding boxes**.
If the benchmark has nets but no pin offsets, objectives ``phi_hpwl`` / ``phi_congestion``
are left undefined (``nan`` → JSON ``null``) and ``surrogate_mode`` is ``netlist_pins_missing``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

from hrt_chip.geometry import count_overlapping_pairs, overlap_area
from hrt_chip.models import MacroRect, PlacementCandidate
from hrt_chip.netlist_surrogates import (
    benchmark_has_netlist,
    benchmark_has_pin_groups,
    hpwl_exact_pin_surrogate,
    rudy_congestion_pin_surrogate,
)


@dataclass(frozen=True)
class GuidanceWeights:
    """Inference-time weights (α, β, γ) for HPWL, congestion, and legality surrogates."""

    alpha_hpwl: float
    beta_congestion: float
    gamma_legality: float

    def to_tuple(self) -> tuple[float, float, float]:
        return (self.alpha_hpwl, self.beta_congestion, self.gamma_legality)

    def to_dict(self) -> dict[str, float]:
        return {
            "alpha_hpwl": self.alpha_hpwl,
            "beta_congestion": self.beta_congestion,
            "gamma_legality": self.gamma_legality,
        }


class ObjectiveSurrogate(Protocol):
    """Fast surrogate ϕ(placement) used for scoring tables and future guided sampling."""

    def score(self, macros: list[MacroRect], *, canvas_w: float = 1.0, canvas_h: float = 1.0) -> float:
        """Return a scalar cost (lower is better unless documented otherwise)."""


def hpwl_bbox_surrogate(macros: list[MacroRect], *, canvas_w: float = 1.0, canvas_h: float = 1.0) -> float:
    """Half-perimeter of the bounding box of macro centers (no netlist; cheap surrogate)."""
    if not macros:
        return 0.0
    cx = [m.x + 0.5 * m.w for m in macros]
    cy = [m.y + 0.5 * m.h for m in macros]
    min_x, max_x = min(cx), max(cx)
    min_y, max_y = min(cy), max(cy)
    w_bb = (max_x - min_x) / max(canvas_w, 1e-12)
    h_bb = (max_y - min_y) / max(canvas_h, 1e-12)
    return w_bb + h_bb


def congestion_grid_surrogate(
    macros: list[MacroRect],
    *,
    canvas_w: float = 1.0,
    canvas_h: float = 1.0,
    grid: int = 8,
) -> float:
    """Simple occupancy variance on a G×G grid (macro footprint; legacy stub)."""
    if grid < 2 or not macros:
        return 0.0
    cell_w = canvas_w / grid
    cell_h = canvas_h / grid
    occ = [0.0] * (grid * grid)
    for m in macros:
        ax1, ay1 = m.x, m.y
        ax2, ay2 = m.x + m.w, m.y + m.h
        i0 = max(0, int(ax1 / cell_w))
        i1 = min(grid - 1, int(ax2 / cell_w - 1e-12))
        j0 = max(0, int(ay1 / cell_h))
        j1 = min(grid - 1, int(ay2 / cell_h - 1e-12))
        for ii in range(i0, i1 + 1):
            for jj in range(j0, j1 + 1):
                occ[ii + jj * grid] += (m.w * m.h) / (cell_w * cell_h * grid * grid)
    mean = sum(occ) / len(occ)
    var = sum((x - mean) ** 2 for x in occ) / len(occ)
    return var


def smooth_overlap_penalty(macros: list[MacroRect]) -> float:
    """Sum of squared overlap areas between macro pairs (0 when fully legal)."""
    n = len(macros)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            oa = overlap_area(macros[i], macros[j])
            if oa > 0:
                total += oa * oa
    return total


def hard_overlap_pair_count(macros: list[MacroRect]) -> int:
    """Discrete count of overlapping macro pairs (constraint-style)."""
    return int(count_overlapping_pairs(macros))


def _float_for_json(x: float) -> float | None:
    if math.isnan(x) or math.isinf(x):
        return None
    return float(x)


@dataclass(frozen=True)
class ObjectiveComponents:
    """Surrogate breakdown for candidate scoring tables."""

    phi_hpwl: float
    phi_congestion: float
    phi_legality: float
    """Smooth overlap penalty (same as ``smooth_overlap_penalty``; kept for weight γ)."""
    hard_overlap_pairs: int
    smooth_overlap_penalty: float
    surrogate_mode: str = "bbox_grid_legacy"

    def to_dict(self) -> dict[str, float | str | int | None]:
        return {
            "phi_hpwl": _float_for_json(self.phi_hpwl),
            "phi_congestion": _float_for_json(self.phi_congestion),
            "phi_legality": _float_for_json(self.phi_legality),
            "hard_overlap_pairs": int(self.hard_overlap_pairs),
            "smooth_overlap_penalty": _float_for_json(self.smooth_overlap_penalty),
            "surrogate_mode": self.surrogate_mode,
        }


def composite_guidance_objective(
    objs: ObjectiveComponents,
    *,
    alpha_hpwl: float,
    beta_congestion: float,
    gamma_legality: float,
) -> float | None:
    """
    Weighted scalar surrogate matching inference weights (lower is better).

    Returns ``None`` when HPWL or congestion terms are unavailable (e.g. pin data missing).
    """
    if math.isnan(objs.phi_hpwl) or math.isnan(objs.phi_congestion):
        return None
    return (
        alpha_hpwl * objs.phi_hpwl
        + beta_congestion * objs.phi_congestion
        + gamma_legality * objs.phi_legality
    )


def compute_objective_components(
    macros: list[MacroRect],
    *,
    canvas_w: float = 1.0,
    canvas_h: float = 1.0,
    benchmark: Any | None = None,
) -> ObjectiveComponents:
    smooth = smooth_overlap_penalty(macros)
    hard = hard_overlap_pair_count(macros)

    if benchmark_has_pin_groups(benchmark):
        hpwl = hpwl_exact_pin_surrogate(macros, benchmark, canvas_w=canvas_w, canvas_h=canvas_h)
        cong = rudy_congestion_pin_surrogate(macros, benchmark, canvas_w=canvas_w, canvas_h=canvas_h)
        mode = "netlist_pin_rudy"
    elif benchmark_has_netlist(benchmark):
        hpwl = float("nan")
        cong = float("nan")
        mode = "netlist_pins_missing"
    else:
        hpwl = hpwl_bbox_surrogate(macros, canvas_w=canvas_w, canvas_h=canvas_h)
        cong = congestion_grid_surrogate(macros, canvas_w=canvas_w, canvas_h=canvas_h)
        mode = "bbox_grid_legacy"

    return ObjectiveComponents(
        phi_hpwl=hpwl,
        phi_congestion=cong,
        phi_legality=smooth,
        hard_overlap_pairs=hard,
        smooth_overlap_penalty=smooth,
        surrogate_mode=mode,
    )


def compute_objectives_for_candidate(
    candidate: PlacementCandidate,
    *,
    benchmark: Any | None = None,
    canvas_w: float = 1.0,
    canvas_h: float = 1.0,
) -> ObjectiveComponents:
    """Compute surrogates for one candidate's current macro geometry."""
    return compute_objective_components(
        candidate.macros,
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        benchmark=benchmark,
    )
