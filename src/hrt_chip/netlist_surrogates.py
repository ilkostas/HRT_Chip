"""Netlist-aware surrogates: exact pin-group HPWL, RUDY raster, optional LogSumExp HPWL.

Compatible with ``macro_place``-style ``Benchmark`` objects. **Pin-group objectives** require
per-net pin offsets ``net_pin_dx`` / ``net_pin_dy`` aligned with ``net_nodes`` (see
``benchmark_has_pin_groups``). When only macro-level ``net_nodes`` exist, use
``rudy_congestion_macro_surrogate`` / legacy paths; guidance uses ``netlist_pins_missing`` mode
rather than silently substituting pin geometry.
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


def _to_numpy_float1d(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(np.float64).reshape(-1)
    return np.asarray(x, dtype=np.float64).reshape(-1)


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


def benchmark_has_netlist(benchmark: Any) -> bool:
    return (
        benchmark is not None
        and hasattr(benchmark, "net_nodes")
        and hasattr(benchmark, "num_nets")
        and int(getattr(benchmark, "num_nets", 0) or 0) > 0
        and len(getattr(benchmark, "net_nodes", []) or []) > 0
    )


def benchmark_has_pin_groups(benchmark: Any) -> bool:
    """
    True when per-net pin offsets are present and aligned with ``net_nodes``.

    Expected layout (duck-typed, macro_place-compatible):

    - ``benchmark.net_pin_dx``: sequence, length == ``len(net_nodes)``.
    - ``benchmark.net_pin_dy``: same.
    - For each net index ``i``, ``net_pin_dx[i]`` and ``net_pin_dy[i]`` are 1D float arrays/tensors
      with the same length as ``net_nodes[i]`` (one offset pair per pin on that net).

    Offsets are in microns relative to the **macro center** of the corresponding macro in
    ``net_nodes[i]``.
    """
    if not benchmark_has_netlist(benchmark):
        return False
    if not hasattr(benchmark, "net_pin_dx") or not hasattr(benchmark, "net_pin_dy"):
        return False
    nodes = benchmark.net_nodes
    dx_list = benchmark.net_pin_dx
    dy_list = benchmark.net_pin_dy
    if len(dx_list) != len(nodes) or len(dy_list) != len(nodes):
        return False
    for i, nodes_t in enumerate(nodes):
        n_pins = int(_net_nodes_to_indices(nodes_t).size)
        if n_pins < 2:
            continue
        dxi = _to_numpy_float1d(dx_list[i])
        dyi = _to_numpy_float1d(dy_list[i])
        if dxi.size != n_pins or dyi.size != n_pins:
            return False
    return True


def _pin_xy_arrays_for_net(
    macros: list[MacroRect],
    benchmark: Any,
    net_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Absolute pin (x, y) in microns for one net."""
    nodes_t = benchmark.net_nodes[net_index]
    nodes_np = _net_nodes_to_indices(nodes_t)
    cx, cy = _macro_centers_from_candidate(macros, benchmark)
    dxi = _to_numpy_float1d(benchmark.net_pin_dx[net_index])
    dyi = _to_numpy_float1d(benchmark.net_pin_dy[net_index])
    xs = cx[nodes_np] + dxi
    ys = cy[nodes_np] + dyi
    return xs, ys


def hpwl_exact_pin_surrogate(
    macros: list[MacroRect],
    benchmark: Any,
    *,
    canvas_w: float,
    canvas_h: float,
) -> float:
    """
    Weighted exact HPWL over pin positions (half-perimeter of pin bbox per net).

    Returns ``nan`` if ``benchmark_has_pin_groups`` is false.
    """
    if not benchmark_has_pin_groups(benchmark):
        return float("nan")

    weights = benchmark.net_weights
    net_nodes = benchmark.net_nodes
    total = 0.0
    for ni, nodes_t in enumerate(net_nodes):
        nodes_np = _net_nodes_to_indices(nodes_t)
        if nodes_np.size < 2:
            continue
        xs, ys = _pin_xy_arrays_for_net(macros, benchmark, ni)
        wn = _to_float(weights[ni]) if ni < len(weights) else 1.0
        hp = (float(np.max(xs)) - float(np.min(xs))) + (float(np.max(ys)) - float(np.min(ys)))
        total += wn * hp
    norm = max(canvas_w + canvas_h, 1e-12)
    return float(total / norm)


def hpwl_logsumexp_surrogate(
    macros: list[MacroRect],
    benchmark: Any,
    *,
    canvas_w: float,
    canvas_h: float,
    tau_um: float | None = None,
) -> float:
    """
    Smooth HPWL over **macro centers** (optional training / differentiable hook).

    Uses LogSumExp smoothing; not used for Tier-1 surrogate tables when pin groups exist.
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
        lse_max_x = tau * _logsumexp(xs / tau)
        lse_min_x = -tau * _logsumexp(-xs / tau)
        lse_max_y = tau * _logsumexp(ys / tau)
        lse_min_y = -tau * _logsumexp(-ys / tau)
        total += w * (lse_max_x - lse_min_x + lse_max_y - lse_min_y)

    norm = max(canvas_w + canvas_h, 1e-12)
    return float(total / norm)


def rudy_congestion_macro_surrogate(
    macros: list[MacroRect],
    benchmark: Any,
    *,
    canvas_w: float,
    canvas_h: float,
    grid_rows: int | None = None,
    grid_cols: int | None = None,
) -> float:
    """
    RUDY-style demand from each net's bounding box of **macro centers**; cell density variance.
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
    return float(np.mean((flat - mean) ** 2))


def rudy_congestion_pin_surrogate(
    macros: list[MacroRect],
    benchmark: Any,
    *,
    canvas_w: float,
    canvas_h: float,
    grid_rows: int | None = None,
    grid_cols: int | None = None,
) -> float:
    """
    RUDY-style demand using each net's bounding box of **pin locations** (macro center + offset).

    Returns ``nan`` if pin groups are not available.
    """
    if not benchmark_has_pin_groups(benchmark):
        return float("nan")

    gr = int(grid_rows) if grid_rows is not None else int(getattr(benchmark, "grid_rows", 32) or 32)
    gc = int(grid_cols) if grid_cols is not None else int(getattr(benchmark, "grid_cols", 32) or 32)
    gr = max(2, gr)
    gc = max(2, gc)

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
        xs, ys = _pin_xy_arrays_for_net(macros, benchmark, ni)
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
    return float(np.mean((flat - mean) ** 2))


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
    RUDY congestion proxy: **pin-based** when ``benchmark_has_pin_groups``, else macro-center RUDY.

    Mixed-size estimate and other diagnostics use this convenience entry point.
    """
    if benchmark_has_pin_groups(benchmark):
        return rudy_congestion_pin_surrogate(
            macros,
            benchmark,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
        )
    return rudy_congestion_macro_surrogate(
        macros,
        benchmark,
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
    )


def legality_overlap_surrogate_np(macros: list[MacroRect]) -> float:
    n = len(macros)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            oa = overlap_area(macros[i], macros[j])
            if oa > 0:
                total += oa * oa
    return float(total)
