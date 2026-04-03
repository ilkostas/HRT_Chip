from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from submissions.force_directed_sa.spatial import SpatialHashIndex, node_bbox, node_bbox_xy
from submissions.force_directed_sa.surrogate import IncrementalSurrogate, pin_position
from submissions.force_directed_sa.types import Benchmark


@dataclass
class SAConfig:
    iterations: int = 150_000
    max_seconds: float = 120.0
    seed: int = 12345
    shift_prob: float = 0.70
    swap_prob: float = 0.20
    smart_prob: float = 0.10


def _build_index(benchmark: Benchmark) -> SpatialHashIndex:
    sizes = [0.5 * (benchmark.nodes[i].width + benchmark.nodes[i].height) for i in benchmark.movable_indices]
    cell_size = float(max(0.25, np.mean(sizes) if sizes else 1.0))
    index = SpatialHashIndex(benchmark.canvas_width, benchmark.canvas_height, cell_size)
    mov_set = set(benchmark.movable_indices)
    for idx, node in enumerate(benchmark.nodes):
        if idx not in mov_set and (node.node_type != "MACRO" or node.name.startswith("Grp_")):
            continue
        if node.width <= 0.0 or node.height <= 0.0:
            continue
        index.insert(idx, node_bbox(node))
    return index


def _in_canvas(benchmark: Benchmark, idx: int, x: float, y: float) -> bool:
    n = benchmark.nodes[idx]
    return (
        x - n.width / 2.0 >= 0.0
        and y - n.height / 2.0 >= 0.0
        and x + n.width / 2.0 <= benchmark.canvas_width
        and y + n.height / 2.0 <= benchmark.canvas_height
    )


def _random_shift(
    benchmark: Benchmark, rng: np.random.Generator, idx: int, scale: float
) -> tuple[float, float]:
    node = benchmark.nodes[idx]
    theta = rng.uniform(0.0, 2.0 * np.pi)
    radius = rng.uniform(0.0, scale)
    nx = node.x + radius * float(np.cos(theta))
    ny = node.y + radius * float(np.sin(theta))
    nx = min(max(nx, node.width / 2.0), benchmark.canvas_width - node.width / 2.0)
    ny = min(max(ny, node.height / 2.0), benchmark.canvas_height - node.height / 2.0)
    return nx, ny


def _smart_shift(
    benchmark: Benchmark, rng: np.random.Generator, idx: int, scale: float
) -> tuple[float, float]:
    adjs = benchmark.node_to_nets[idx]
    if not adjs:
        return _random_shift(benchmark, rng, idx, scale)
    best_net = max(adjs, key=lambda ni: len(benchmark.nets[ni].pin_refs))
    net = benchmark.nets[best_net]
    sum_x = 0.0
    sum_y = 0.0
    cnt = 0
    for pref in net.pin_refs:
        if pref.node_idx == idx:
            continue
        node = benchmark.nodes[pref.node_idx]
        px, py = pin_position(node, pref)
        sum_x += px
        sum_y += py
        cnt += 1
    if cnt == 0:
        return _random_shift(benchmark, rng, idx, scale)
    cx = sum_x / cnt
    cy = sum_y / cnt
    node = benchmark.nodes[idx]
    dx = cx - node.x
    dy = cy - node.y
    norm = max(1e-9, float(np.hypot(dx, dy)))
    step = min(scale, norm)
    nx = node.x + dx / norm * step
    ny = node.y + dy / norm * step
    nx = min(max(nx, node.width / 2.0), benchmark.canvas_width - node.width / 2.0)
    ny = min(max(ny, node.height / 2.0), benchmark.canvas_height - node.height / 2.0)
    return nx, ny


def _accept(delta: float, temp: float, rng: np.random.Generator) -> bool:
    if delta <= 0.0:
        return True
    if temp <= 1e-15:
        return False
    return rng.uniform(0.0, 1.0) < math.exp(-delta / temp)


def run_sa_refinement(benchmark: Benchmark, cfg: SAConfig) -> None:
    if not benchmark.movable_indices:
        return
    rng = np.random.default_rng(cfg.seed)
    index = _build_index(benchmark)
    obj = IncrementalSurrogate(benchmark)
    cur_cost = obj.total()
    t_init = 0.1 * max(cur_cost, 1e-6)
    t_final = 1e-6
    cooling = (t_final / t_init) ** (1.0 / max(cfg.iterations, 1))
    temp = t_init
    start = time.perf_counter()
    diag = float(np.hypot(benchmark.canvas_width, benchmark.canvas_height))

    mov = benchmark.movable_indices
    for it in range(cfg.iterations):
        if time.perf_counter() - start > cfg.max_seconds:
            break
        progress = it / max(1, cfg.iterations - 1)
        step_scale = (1.0 - progress) * 0.10 * diag + 0.01 * diag

        op_rand = rng.uniform(0.0, 1.0)
        if op_rand < cfg.shift_prob:
            idx = int(mov[rng.integers(0, len(mov))])
            node = benchmark.nodes[idx]
            old_x, old_y = node.x, node.y
            nx, ny = _random_shift(benchmark, rng, idx, step_scale)
            if not _in_canvas(benchmark, idx, nx, ny):
                temp *= cooling
                continue
            index.remove(idx)
            new_bbox = node_bbox_xy(node, nx, ny)
            if index.collides(new_bbox):
                index.insert(idx, node_bbox_xy(node, old_x, old_y))
                temp *= cooling
                continue
            index.insert(idx, new_bbox)
            delta = obj.apply_move(idx, nx, ny)
            if _accept(delta, temp, rng):
                cur_cost += delta
            else:
                obj.apply_move(idx, old_x, old_y)
                index.update(idx, node_bbox_xy(node, old_x, old_y))

        elif op_rand < cfg.shift_prob + cfg.swap_prob and len(mov) > 1:
            i = int(mov[rng.integers(0, len(mov))])
            j = i
            while j == i:
                j = int(mov[rng.integers(0, len(mov))])
            ni = benchmark.nodes[i]
            nj = benchmark.nodes[j]
            old_ix, old_iy = ni.x, ni.y
            old_jx, old_jy = nj.x, nj.y

            index.remove(i)
            index.remove(j)
            bbox_i_new = node_bbox_xy(ni, old_jx, old_jy)
            bbox_j_new = node_bbox_xy(nj, old_ix, old_iy)
            legal = (
                _in_canvas(benchmark, i, old_jx, old_jy)
                and _in_canvas(benchmark, j, old_ix, old_iy)
                and not index.collides(bbox_i_new)
                and not index.collides(bbox_j_new)
                and not (
                    bbox_i_new[2] <= bbox_j_new[0]
                    or bbox_j_new[2] <= bbox_i_new[0]
                    or bbox_i_new[3] <= bbox_j_new[1]
                    or bbox_j_new[3] <= bbox_i_new[1]
                )
            )
            if not legal:
                index.insert(i, node_bbox_xy(ni, old_ix, old_iy))
                index.insert(j, node_bbox_xy(nj, old_jx, old_jy))
                temp *= cooling
                continue
            index.insert(i, bbox_i_new)
            index.insert(j, bbox_j_new)
            delta_i = obj.apply_move(i, old_jx, old_jy)
            delta_j = obj.apply_move(j, old_ix, old_iy)
            delta = delta_i + delta_j
            if _accept(delta, temp, rng):
                cur_cost += delta
            else:
                obj.apply_move(j, old_jx, old_jy)
                obj.apply_move(i, old_ix, old_iy)
                index.update(i, node_bbox_xy(ni, old_ix, old_iy))
                index.update(j, node_bbox_xy(nj, old_jx, old_jy))

        else:
            idx = int(mov[rng.integers(0, len(mov))])
            node = benchmark.nodes[idx]
            old_x, old_y = node.x, node.y
            nx, ny = _smart_shift(benchmark, rng, idx, step_scale)
            if not _in_canvas(benchmark, idx, nx, ny):
                temp *= cooling
                continue
            index.remove(idx)
            new_bbox = node_bbox_xy(node, nx, ny)
            if index.collides(new_bbox):
                index.insert(idx, node_bbox_xy(node, old_x, old_y))
                temp *= cooling
                continue
            index.insert(idx, new_bbox)
            delta = obj.apply_move(idx, nx, ny)
            if _accept(delta, temp, rng):
                cur_cost += delta
            else:
                obj.apply_move(idx, old_x, old_y)
                index.update(idx, node_bbox_xy(node, old_x, old_y))

        temp *= cooling

