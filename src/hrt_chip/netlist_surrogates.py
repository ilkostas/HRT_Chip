"""Netlist-aware differentiable-style surrogates (LogSumExp HPWL, RUDY congestion).

Compatible with ``macro_place`` ``Benchmark`` (``net_nodes``, ``net_weights``, ``macro_names``, …).
Falls back to geometric proxies when no benchmark is provided.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from hrt_chip.geometry import overlap_area
from hrt_chip.models import MacroRect


def _to_float(x: Any) -> float:
    if hasattr(x, "item"):
        return float(x.item())
    return float(x)


def _net_nodes_to_indices(nodes_t: Any) -> np.ndarray:
    if hasattr(nodes_t, "detach"):
        return nodes_t.detach().cpu().numpy().astype(np.int64).reshape(-1)
    return np.asarray(nodes_t, dtype=np.int64).reshape(-1)


def _macro_centers_from_candidate(
    macros: list[MacroRect],
    benchmark: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Build [N] center arrays in microns aligned with benchmark macro index order."""
    n = int(benchmark.num_macros)
    cx = np.zeros(n, dtype=np.float64)
    cy = np.zeros(n, dtype=np.float64)
    by_name = {m.name: m for m in macros}
    names: Sequence[str] = benchmark.macro_names
    pos = benchmark.macro_positions
    sizes = benchmark.macro_sizes
    for i in range(n):
        name = names[i]
        m = by_name.get(name)
        if m is not None:
            cx[i] = float(m.x) + float(m.w) / 2.0
            cy[i] = float(m.y) + float(m.h) / 2.0
        else:
            cx[i] = _to_float(pos[i, 0])
            cy[i] = _to_float(pos[i, 1])
    return cx, cy


def _logsumexp(a: np.ndarray) -> float:
    if a.size == 0:
        return float("-inf")
    m = float(np.max(a))
    if math.isinf(m) and m > 0:
        return m
    return m + math.log(float(np.sum(np.exp(a - m))) + 1e-30)


def hpwl_logsumexp_surrogate(
    macros: list[MacroRect],
    benchmark: Any,
    *,
    canvas_w: float,
    canvas_h: float,
    tau_um: float | None = None,
) -> float:
    """
    Smooth HPWL: sum_net w * (LSE_max(x) - LSE_min(x) + LSE_max(y) - LSE_min(y)) with temperature tau.
    Lower tau approaches true HPWL (nondifferentiable at ties).
    """
    if benchmark is None or not hasattr(benchmark, "net_nodes") or len(benchmark.net_nodes) == 0:
        return float("nan")

    tau = tau_um if tau_um is not None else max(canvas_w, canvas_h) * 0.02
    if tau <= 0:
        tau = 1e-6

    cx, cy = _macro_centers_from_candidate(macros, benchmark)
    weights = benchmark.net_weights
    net_nodes = benchmark.net_nodes

    total = 0.0
    for ni, nodes_t in enumerate(net_nodes):
        nodes_np = _net_nodes_to_indices(nodes_t)
        if nodes_np.size < 2:
            continue
        xs = cx[nodes_np]
        ys = cy[nodes_np]
        w = _to_float(weights[ni]) if ni < len(weights) else 1.0
        # LSE max ~ max; LSE(-x) relates to min
        lse_max_x = tau * _logsumexp(xs / tau)
        lse_min_x = -tau * _logsumexp(-xs / tau)
        lse_max_y = tau * _logsumexp(ys / tau)
        lse_min_y = -tau * _logsumexp(-ys / tau)
        total += w * (lse_max_x - lse_min_x + lse_max_y - lse_min_y)

    # Normalize by canvas half-perimeter for scale-free reporting
    norm = max(canvas_w + canvas_h, 1e-12)
    return float(total / norm)


def rudy_congestion_surrogate(
    macros: list[MacroRect],
    benchmark: Any,
    *,
    canvas_w: float,
    canvas_h: float,
    grid_rows: int | None = None,
    grid_cols: int | None = None,
) -> float:
    """
    RUDY-style routing demand: for each net, spread net weight uniformly over the net's bounding box
    of macro centers; accumulate per-cell density; return variance (congestion proxy).
    """
    if benchmark is None or not hasattr(benchmark, "net_nodes") or len(benchmark.net_nodes) == 0:
        return float("nan")

    gr = int(grid_rows) if grid_rows is not None else int(getattr(benchmark, "grid_rows", 32) or 32)
    gc = int(grid_cols) if grid_cols is not None else int(getattr(benchmark, "grid_cols", 32) or 32)
    gr = max(2, gr)
    gc = max(2, gc)

    cx, cy = _macro_centers_from_candidate(macros, benchmark)
    weights = benchmark.net_weights
    net_nodes = benchmark.net_nodes

    rudy = np.zeros((gr, gc), dtype=np.float64)
    cell_w = canvas_w / gc
    cell_h = canvas_h / gr

    eps = max(canvas_w, canvas_h) * 1e-9

    for ni, nodes_t in enumerate(net_nodes):
        nodes_np = _net_nodes_to_indices(nodes_t)
        if nodes_np.size < 2:
            continue
        xs = cx[nodes_np]
        ys = cy[nodes_np]
        min_x, max_x = float(np.min(xs)), float(np.max(xs))
        min_y, max_y = float(np.min(ys)), float(np.max(ys))
        bw = max(max_x - min_x, eps)
        bh = max(max_y - min_y, eps)
        bbox_area = bw * bh
        wn = _to_float(weights[ni]) if ni < len(weights) else 1.0
        dens = wn / bbox_area

        i0 = max(0, int(min_x / cell_w))
        i1 = min(gc - 1, int(max_x / cell_w - 1e-12))
        j0 = max(0, int(min_y / cell_h))
        j1 = min(gr - 1, int(max_y / cell_h - 1e-12))
        for jj in range(j0, j1 + 1):
            for ii in range(i0, i1 + 1):
                cell_x0, cell_x1 = ii * cell_w, (ii + 1) * cell_w
                cell_y0, cell_y1 = jj * cell_h, (jj + 1) * cell_h
                ix0 = max(min_x, cell_x0)
                ix1 = min(max_x, cell_x1)
                iy0 = max(min_y, cell_y0)
                iy1 = min(max_y, cell_y1)
                inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
                if inter > 0:
                    rudy[jj, ii] += dens * inter

    flat = rudy.reshape(-1)
    mean = float(np.mean(flat))
    var = float(np.mean((flat - mean) ** 2))
    return var


def legality_overlap_surrogate_np(macros: list[MacroRect]) -> float:
    n = len(macros)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            oa = overlap_area(macros[i], macros[j])
            if oa > 0:
                total += oa * oa
    return float(total)


def benchmark_has_netlist(benchmark: Any) -> bool:
    return (
        benchmark is not None
        and hasattr(benchmark, "net_nodes")
        and hasattr(benchmark, "num_nets")
        and int(getattr(benchmark, "num_nets", 0) or 0) > 0
        and len(getattr(benchmark, "net_nodes", []) or []) > 0
    )
