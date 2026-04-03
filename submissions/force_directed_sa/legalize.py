from __future__ import annotations

from math import inf

import numpy as np

from submissions.force_directed_sa.spatial import SpatialHashIndex, bboxes_overlap, node_bbox, node_bbox_xy
from submissions.force_directed_sa.types import Benchmark


def _in_canvas(benchmark: Benchmark, node_idx: int, x: float, y: float) -> bool:
    node = benchmark.nodes[node_idx]
    return (
        x - node.width / 2.0 >= 0.0
        and y - node.height / 2.0 >= 0.0
        and x + node.width / 2.0 <= benchmark.canvas_width
        and y + node.height / 2.0 <= benchmark.canvas_height
    )


def _build_legalizer_index(benchmark: Benchmark) -> SpatialHashIndex:
    mov_set = set(benchmark.movable_indices)
    avg_size = np.mean(
        [max(0.25, 0.5 * (benchmark.nodes[i].width + benchmark.nodes[i].height)) for i in benchmark.movable_indices]
    )
    idx = SpatialHashIndex(benchmark.canvas_width, benchmark.canvas_height, float(avg_size))
    for i, node in enumerate(benchmark.nodes):
        if i in mov_set:
            continue
        if node.node_type != "MACRO" or node.name.startswith("Grp_"):
            continue
        if node.width <= 0.0 or node.height <= 0.0:
            continue
        idx.insert(i, node_bbox(node))
    return idx


def legalize(benchmark: Benchmark) -> None:
    if not benchmark.movable_indices:
        return
    index = _build_legalizer_index(benchmark)
    ordered = sorted(benchmark.movable_indices, key=lambda i: benchmark.nodes[i].area, reverse=True)
    for idx in ordered:
        node = benchmark.nodes[idx]
        cur_bbox = node_bbox(node)
        if _in_canvas(benchmark, idx, node.x, node.y) and not index.collides(cur_bbox):
            index.insert(idx, cur_bbox)
            continue
        best_xy: tuple[float, float] | None = None
        best_dist = inf
        step = max(0.05, min(node.width, node.height) * 0.25)
        for radius in (1, 2, 4, 8, 16, 32, 64, 96, 128):
            r_step = radius * step
            offsets = np.arange(-r_step, r_step + step, step)
            for dx in offsets:
                for dy in offsets:
                    nx = node.x + float(dx)
                    ny = node.y + float(dy)
                    if not _in_canvas(benchmark, idx, nx, ny):
                        continue
                    bbox = node_bbox_xy(node, nx, ny)
                    if index.collides(bbox):
                        continue
                    d2 = float(dx * dx + dy * dy)
                    if d2 < best_dist:
                        best_dist = d2
                        best_xy = (nx, ny)
            if best_xy is not None:
                break
        if best_xy is None:
            # Fallback: full-canvas coarse scan to guarantee legalization completion.
            scan_step = max(0.05, min(node.width, node.height) * 0.5)
            xs = np.arange(node.width / 2.0, benchmark.canvas_width - node.width / 2.0 + scan_step, scan_step)
            ys = np.arange(node.height / 2.0, benchmark.canvas_height - node.height / 2.0 + scan_step, scan_step)
            for nx in xs:
                for ny in ys:
                    bbox = node_bbox_xy(node, float(nx), float(ny))
                    if not index.collides(bbox):
                        best_xy = (float(nx), float(ny))
                        break
                if best_xy is not None:
                    break
        if best_xy is None:
            raise RuntimeError(f"Failed to legalize macro {node.name}")
        node.x, node.y = best_xy
        index.insert(idx, node_bbox(node))
    _assert_zero_overlap(benchmark)


def _assert_zero_overlap(benchmark: Benchmark) -> None:
    mov_set = set(benchmark.movable_indices)
    fixed_hard = [
        i
        for i, n in enumerate(benchmark.nodes)
        if i not in mov_set and n.node_type == "MACRO" and not n.name.startswith("Grp_") and n.width > 0.0 and n.height > 0.0
    ]
    areas = list(mov_set) + fixed_hard
    avg = np.mean([0.5 * (benchmark.nodes[i].width + benchmark.nodes[i].height) for i in areas]) if areas else 1.0
    index = SpatialHashIndex(benchmark.canvas_width, benchmark.canvas_height, float(max(0.25, avg)))
    for idx in areas:
        bbox = node_bbox(benchmark.nodes[idx])
        if index.collides(bbox):
            raise AssertionError(f"Overlap remains after legalization at node index {idx}")
        index.insert(idx, bbox)

