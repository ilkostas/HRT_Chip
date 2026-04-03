from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np

from submissions.force_directed_sa.surrogate import compute_density_grid, pin_position
from submissions.force_directed_sa.types import Benchmark


@dataclass
class ForceDirectedConfig:
    iterations: int = 250
    alpha: float = 1.0
    beta_initial: float = 0.1
    beta_final: float = 2.0
    initial_step_scale: float = 0.10


def _clamp_to_canvas(benchmark: Benchmark, node_idx: int, x: float, y: float) -> tuple[float, float]:
    node = benchmark.nodes[node_idx]
    x = min(max(x, node.width / 2.0), benchmark.canvas_width - node.width / 2.0)
    y = min(max(y, node.height / 2.0), benchmark.canvas_height - node.height / 2.0)
    return x, y


def _density_force(benchmark: Benchmark, density: np.ndarray, node_idx: int) -> tuple[float, float]:
    node = benchmark.nodes[node_idx]
    if node.width <= 0.0 or node.height <= 0.0:
        return 0.0, 0.0
    cell_w = benchmark.canvas_width / benchmark.grid_cols
    cell_h = benchmark.canvas_height / benchmark.grid_rows
    gx = min(benchmark.grid_cols - 1, max(0, int(node.x / cell_w)))
    gy = min(benchmark.grid_rows - 1, max(0, int(node.y / cell_h)))
    base = density[gy, gx]
    fx = 0.0
    fy = 0.0
    for ny in range(max(0, gy - 1), min(benchmark.grid_rows - 1, gy + 1) + 1):
        for nx in range(max(0, gx - 1), min(benchmark.grid_cols - 1, gx + 1) + 1):
            if nx == gx and ny == gy:
                continue
            diff = base - density[ny, nx]
            dx = float(nx - gx)
            dy = float(ny - gy)
            n = sqrt(dx * dx + dy * dy)
            if n <= 1e-12:
                continue
            fx += diff * (dx / n)
            fy += diff * (dy / n)
    return fx, fy


def _wirelength_force(benchmark: Benchmark, node_idx: int) -> tuple[float, float]:
    fx = 0.0
    fy = 0.0
    node = benchmark.nodes[node_idx]
    for net_idx in benchmark.node_to_nets[node_idx]:
        net = benchmark.nets[net_idx]
        sum_x = 0.0
        sum_y = 0.0
        cnt = 0
        for pref in net.pin_refs:
            if pref.node_idx == node_idx:
                continue
            other = benchmark.nodes[pref.node_idx]
            px, py = pin_position(other, pref)
            sum_x += px
            sum_y += py
            cnt += 1
        if cnt == 0:
            continue
        cx = sum_x / cnt
        cy = sum_y / cnt
        fx += net.weight * (cx - node.x)
        fy += net.weight * (cy - node.y)
    return fx, fy


def run_force_directed(benchmark: Benchmark, cfg: ForceDirectedConfig) -> None:
    if not benchmark.movable_indices:
        return
    diag = max(1.0, float(np.hypot(benchmark.canvas_width, benchmark.canvas_height)))
    for it in range(cfg.iterations):
        t = it / max(1, cfg.iterations - 1)
        beta = cfg.beta_initial + (cfg.beta_final - cfg.beta_initial) * (t**2)
        step = cfg.initial_step_scale * diag * (1.0 - t)
        density = compute_density_grid(benchmark)
        updates: dict[int, tuple[float, float]] = {}
        for node_idx in benchmark.movable_indices:
            wl_fx, wl_fy = _wirelength_force(benchmark, node_idx)
            den_fx, den_fy = _density_force(benchmark, density, node_idx)
            fx = cfg.alpha * wl_fx + beta * den_fx
            fy = cfg.alpha * wl_fy + beta * den_fy
            nx = benchmark.nodes[node_idx].x + step * fx
            ny = benchmark.nodes[node_idx].y + step * fy
            nx, ny = _clamp_to_canvas(benchmark, node_idx, nx, ny)
            updates[node_idx] = (nx, ny)
        for node_idx, (nx, ny) in updates.items():
            benchmark.nodes[node_idx].x = nx
            benchmark.nodes[node_idx].y = ny

