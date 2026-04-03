"""Fast objectives for SA (HPWL-first and proxy-aligned surrogate)."""

from __future__ import annotations

import math
from typing import Any, Literal

from hrt_chip.guidance import (
    compute_objective_components,
    congestion_grid_surrogate,
    hpwl_bbox_surrogate,
)
from hrt_chip.models import PlacementCandidate

EnergyMode = Literal["hpwl", "full"]


def placement_energy(
    candidate: PlacementCandidate,
    *,
    benchmark: Any | None,
    canvas_w: float,
    canvas_h: float,
    mode: EnergyMode,
) -> float:
    """
    Lower is better. Overlap penalized heavily via smooth overlap in components.
    Full mode: 1.0 * hpwl + 0.5 * density_proxy + 0.5 * congestion_proxy (surrogate alignment).
    """
    objs = compute_objective_components(
        candidate.macros,
        benchmark=benchmark,
        canvas_w=canvas_w,
        canvas_h=canvas_h,
    )
    hpwl = objs.phi_hpwl
    if not math.isfinite(hpwl):
        hpwl = hpwl_bbox_surrogate(candidate.macros, canvas_w=canvas_w, canvas_h=canvas_h)

    leg_pen = float(objs.smooth_overlap_penalty) + 1_000.0 * float(objs.hard_overlap_pairs)

    if mode == "hpwl":
        return float(hpwl + leg_pen)

    dens = congestion_grid_surrogate(candidate.macros, canvas_w=canvas_w, canvas_h=canvas_h, grid=16)
    cong = objs.phi_congestion
    if not math.isfinite(cong):
        cong = dens
    return float(hpwl + 0.5 * dens + 0.5 * float(cong) + leg_pen)


def schedule_mode(
    step_index: int,
    total_steps: int,
    schedule: str,
) -> EnergyMode:
    """First ~60% HPWL-only when ``hpwl_then_full``."""
    if schedule == "hpwl_only":
        return "hpwl"
    if schedule == "full_surrogate":
        return "full"
    if total_steps <= 0:
        return "full"
    cut = int(0.6 * total_steps)
    return "hpwl" if step_index < cut else "full"
