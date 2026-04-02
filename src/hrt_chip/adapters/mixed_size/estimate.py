"""Mixed-size handoff estimate: macro utilization + RUDY-style congestion proxy (no DREAMPlace binary).

A full DREAMPlace/hMETIS integration can replace this adapter via the same ``MixedSizeBackend`` contract.
"""

from __future__ import annotations

import time
from typing import Any

from hrt_chip.adapters.mixed_size.base import MixedSizeBackend, MixedSizeRequest, MixedSizeResult
from hrt_chip.geometry import placement_is_legal
from hrt_chip.netlist_surrogates import benchmark_has_netlist, rudy_congestion_surrogate
from hrt_chip.models import MacroRect


def _macro_area_fraction(macros: list[MacroRect], canvas_w: float, canvas_h: float) -> float:
    area = sum(max(0.0, m.w) * max(0.0, m.h) for m in macros)
    den = max(canvas_w * canvas_h, 1e-18)
    return float(area / den)


class MixedSizeEstimateBackend(MixedSizeBackend):
    """
    Fast analytical proxy after macro legalization.

    Uses optional ``benchmark`` on the request for RUDY congestion; otherwise returns
    utilization-only diagnostics.
    """

    def run(self, request: MixedSizeRequest) -> MixedSizeResult:
        t0 = time.perf_counter()
        macros = list(request.fixed_macros)
        if not placement_is_legal(macros, canvas_w=request.canvas_w, canvas_h=request.canvas_h):
            return MixedSizeResult(
                ok=False,
                message="rejected: illegal macro geometry for mixed-size estimate",
                extra={"benchmark_id": request.benchmark_id},
            )

        util = _macro_area_fraction(macros, request.canvas_w, request.canvas_h)
        bench = request.benchmark
        rudy_var = 0.0
        if benchmark_has_netlist(bench):
            rudy_var = float(
                rudy_congestion_surrogate(
                    macros,
                    bench,
                    canvas_w=request.canvas_w,
                    canvas_h=request.canvas_h,
                )
            )
        dt = time.perf_counter() - t0
        # Heuristic density overflow vs 100% canvas (macros only — std-cells absent here).
        density_overflow_proxy = max(0.0, util - 0.95)

        return MixedSizeResult(
            ok=True,
            message="estimate: analytical macro utilization + RUDY proxy (no std-cell placer)",
            extra={
                "benchmark_id": request.benchmark_id,
                "n_macros": len(macros),
                "macro_area_fraction": util,
                "rudy_density_variance": rudy_var,
                "density_overflow_proxy": density_overflow_proxy,
                "backend_runtime_seconds": dt,
                "backend": "mixed_size_estimate",
            },
        )
