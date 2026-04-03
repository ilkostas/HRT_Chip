from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np

from submissions.force_directed_sa.types import Benchmark, Net, Node, PinRef


def _orient_offsets(orientation: str, x_off: float, y_off: float) -> tuple[float, float]:
    o = orientation.upper()
    if o in ("N", "-"):
        return x_off, y_off
    if o == "S":
        return -x_off, -y_off
    if o == "FN":
        return -x_off, y_off
    if o == "FS":
        return x_off, -y_off
    return x_off, y_off


def pin_position(node: Node, pref: PinRef) -> tuple[float, float]:
    ox, oy = _orient_offsets(node.orientation, pref.x_offset, pref.y_offset)
    return node.x + ox, node.y + oy


def net_weighted_hpwl(benchmark: Benchmark, net: Net) -> float:
    xs: list[float] = []
    ys: list[float] = []
    for pref in net.pin_refs:
        node = benchmark.nodes[pref.node_idx]
        x, y = pin_position(node, pref)
        xs.append(x)
        ys.append(y)
    hpwl = (max(xs) - min(xs)) + (max(ys) - min(ys))
    return net.weight * hpwl


def compute_hpwl_component(benchmark: Benchmark) -> float:
    if not benchmark.nets:
        return 0.0
    total_hpwl = sum(net_weighted_hpwl(benchmark, net) for net in benchmark.nets)
    norm = len(benchmark.nets) * (benchmark.canvas_width + benchmark.canvas_height)
    return total_hpwl / max(norm, 1e-9)


def _node_cell_overlap(
    node: Node,
    grid: np.ndarray,
    cell_w: float,
    cell_h: float,
    sign: float,
    *,
    override_xy: tuple[float, float] | None = None,
) -> None:
    if node.area <= 0.0:
        return
    cx, cy = override_xy if override_xy is not None else (node.x, node.y)
    x_lo = cx - node.width / 2.0
    x_hi = cx + node.width / 2.0
    y_lo = cy - node.height / 2.0
    y_hi = cy + node.height / 2.0
    rows, cols = grid.shape
    c0 = max(0, int(np.floor(x_lo / cell_w)))
    c1 = min(cols - 1, int(np.floor(x_hi / cell_w)))
    r0 = max(0, int(np.floor(y_lo / cell_h)))
    r1 = min(rows - 1, int(np.floor(y_hi / cell_h)))
    cell_area = cell_w * cell_h
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            ox0 = max(x_lo, c * cell_w)
            ox1 = min(x_hi, (c + 1) * cell_w)
            oy0 = max(y_lo, r * cell_h)
            oy1 = min(y_hi, (r + 1) * cell_h)
            overlap = max(0.0, ox1 - ox0) * max(0.0, oy1 - oy0)
            if overlap > 0.0:
                grid[r, c] += sign * (overlap / cell_area)


def compute_density_grid(benchmark: Benchmark) -> np.ndarray:
    grid = np.zeros((benchmark.grid_rows, benchmark.grid_cols), dtype=np.float64)
    cell_w = benchmark.canvas_width / benchmark.grid_cols
    cell_h = benchmark.canvas_height / benchmark.grid_rows
    for node in benchmark.nodes:
        _node_cell_overlap(node, grid, cell_w, cell_h, sign=1.0)
    return grid


def density_abu10_cost_from_grid(grid: np.ndarray) -> float:
    flat = np.sort(grid.reshape(-1))
    if flat.size == 0:
        return 0.0
    k = max(1, int(ceil(flat.size * 0.10)))
    raw_abu10 = float(np.mean(flat[-k:]))
    return 0.5 * raw_abu10


def _net_congestion_entries(benchmark: Benchmark, net: Net) -> list[tuple[int, int, float, float]]:
    cell_w = benchmark.canvas_width / benchmark.grid_cols
    cell_h = benchmark.canvas_height / benchmark.grid_rows
    xs: list[float] = []
    ys: list[float] = []
    for pref in net.pin_refs:
        node = benchmark.nodes[pref.node_idx]
        x, y = pin_position(node, pref)
        xs.append(x)
        ys.append(y)
    x0 = min(xs)
    x1 = max(xs)
    y0 = min(ys)
    y1 = max(ys)
    bbox_w = max(0.0, x1 - x0)
    bbox_h = max(0.0, y1 - y0)
    if bbox_w < 1e-12 and bbox_h < 1e-12:
        return []
    bbox_area = max(bbox_w * bbox_h, cell_w * cell_h)
    h_dem = net.weight * bbox_h / bbox_area
    v_dem = net.weight * bbox_w / bbox_area
    c0 = max(0, int(np.floor(x0 / cell_w)))
    c1 = min(benchmark.grid_cols - 1, int(np.floor(x1 / cell_w)))
    r0 = max(0, int(np.floor(y0 / cell_h)))
    r1 = min(benchmark.grid_rows - 1, int(np.floor(y1 / cell_h)))
    entries: list[tuple[int, int, float, float]] = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            entries.append((r, c, h_dem, v_dem))
    return entries


def compute_congestion_grids(benchmark: Benchmark) -> tuple[np.ndarray, np.ndarray]:
    h_grid = np.zeros((benchmark.grid_rows, benchmark.grid_cols), dtype=np.float64)
    v_grid = np.zeros((benchmark.grid_rows, benchmark.grid_cols), dtype=np.float64)
    for net in benchmark.nets:
        for r, c, h_dem, v_dem in _net_congestion_entries(benchmark, net):
            h_grid[r, c] += h_dem
            v_grid[r, c] += v_dem
    cell_w = benchmark.canvas_width / benchmark.grid_cols
    cell_h = benchmark.canvas_height / benchmark.grid_rows
    if benchmark.routes_h > 0.0:
        h_grid /= benchmark.routes_h * cell_h
    if benchmark.routes_v > 0.0:
        v_grid /= benchmark.routes_v * cell_w
    return h_grid, v_grid


def congestion_abu5_cost(h_grid: np.ndarray, v_grid: np.ndarray) -> float:
    flat = np.concatenate([h_grid.reshape(-1), v_grid.reshape(-1)])
    if flat.size == 0:
        return 0.0
    flat.sort()
    k = max(1, int(ceil(flat.size * 0.05)))
    return float(np.mean(flat[-k:]))


@dataclass
class SurrogateBreakdown:
    hpwl: float
    density: float
    congestion: float
    total: float


def compute_surrogate(benchmark: Benchmark) -> SurrogateBreakdown:
    hpwl = compute_hpwl_component(benchmark)
    d_grid = compute_density_grid(benchmark)
    density = density_abu10_cost_from_grid(d_grid)
    h_grid, v_grid = compute_congestion_grids(benchmark)
    congestion = congestion_abu5_cost(h_grid, v_grid)
    total = hpwl + 0.5 * density + 0.5 * congestion
    return SurrogateBreakdown(hpwl=hpwl, density=density, congestion=congestion, total=total)


class IncrementalSurrogate:
    """
    Incremental objective updates for SA.

    Updates only moved-node density occupancy and affected-net HPWL/congestion.
    """

    def __init__(self, benchmark: Benchmark) -> None:
        self.benchmark = benchmark
        self.cell_w = benchmark.canvas_width / benchmark.grid_cols
        self.cell_h = benchmark.canvas_height / benchmark.grid_rows
        self.net_hpwl = [net_weighted_hpwl(benchmark, n) for n in benchmark.nets]
        self.total_hpwl_raw = float(sum(self.net_hpwl))
        self.norm = max(len(benchmark.nets) * (benchmark.canvas_width + benchmark.canvas_height), 1e-9)
        self.density_grid = compute_density_grid(benchmark)
        self.h_grid = np.zeros((benchmark.grid_rows, benchmark.grid_cols), dtype=np.float64)
        self.v_grid = np.zeros((benchmark.grid_rows, benchmark.grid_cols), dtype=np.float64)
        self.net_contribs: list[list[tuple[int, int, float, float]]] = []
        for net in benchmark.nets:
            entries = _net_congestion_entries(benchmark, net)
            self.net_contribs.append(entries)
            for r, c, h_dem, v_dem in entries:
                self.h_grid[r, c] += h_dem
                self.v_grid[r, c] += v_dem
        if benchmark.routes_h > 0.0:
            self.h_grid /= benchmark.routes_h * self.cell_h
        if benchmark.routes_v > 0.0:
            self.v_grid /= benchmark.routes_v * self.cell_w
        self._pending_node_old_xy: tuple[int, float, float] | None = None

    def total(self) -> float:
        hpwl = self.total_hpwl_raw / self.norm
        density = density_abu10_cost_from_grid(self.density_grid)
        congestion = congestion_abu5_cost(self.h_grid, self.v_grid)
        return hpwl + 0.5 * density + 0.5 * congestion

    def apply_move(self, node_idx: int, new_x: float, new_y: float) -> float:
        node = self.benchmark.nodes[node_idx]
        old_x, old_y = node.x, node.y
        old_total = self.total()

        _node_cell_overlap(node, self.density_grid, self.cell_w, self.cell_h, sign=-1.0, override_xy=(old_x, old_y))
        node.x, node.y = new_x, new_y
        _node_cell_overlap(node, self.density_grid, self.cell_w, self.cell_h, sign=1.0, override_xy=(new_x, new_y))

        affected_nets = self.benchmark.node_to_nets[node_idx]
        if affected_nets:
            h_scale = self.benchmark.routes_h * self.cell_h if self.benchmark.routes_h > 0.0 else 1.0
            v_scale = self.benchmark.routes_v * self.cell_w if self.benchmark.routes_v > 0.0 else 1.0
            for net_idx in affected_nets:
                old_entries = self.net_contribs[net_idx]
                for r, c, h_dem, v_dem in old_entries:
                    self.h_grid[r, c] -= h_dem / h_scale
                    self.v_grid[r, c] -= v_dem / v_scale
                self.total_hpwl_raw -= self.net_hpwl[net_idx]
                net = self.benchmark.nets[net_idx]
                new_hpwl = net_weighted_hpwl(self.benchmark, net)
                self.net_hpwl[net_idx] = new_hpwl
                self.total_hpwl_raw += new_hpwl
                new_entries = _net_congestion_entries(self.benchmark, net)
                self.net_contribs[net_idx] = new_entries
                for r, c, h_dem, v_dem in new_entries:
                    self.h_grid[r, c] += h_dem / h_scale
                    self.v_grid[r, c] += v_dem / v_scale

        self._pending_node_old_xy = (node_idx, old_x, old_y)
        return self.total() - old_total

    def rollback_last_move(self) -> None:
        if self._pending_node_old_xy is None:
            return
        idx, old_x, old_y = self._pending_node_old_xy
        node = self.benchmark.nodes[idx]
        self.apply_move(idx, old_x, old_y)
        node.x, node.y = old_x, old_y
        self._pending_node_old_xy = None

