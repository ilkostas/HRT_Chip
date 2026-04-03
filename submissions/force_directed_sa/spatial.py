from __future__ import annotations

from collections import defaultdict

from submissions.force_directed_sa.types import Benchmark, Node


def bboxes_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float], eps: float = 1e-9) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 <= bx0 + eps or bx1 <= ax0 + eps or ay1 <= by0 + eps or by1 <= ay0 + eps)


def node_bbox_xy(node: Node, x: float, y: float) -> tuple[float, float, float, float]:
    return (x - node.width / 2.0, y - node.height / 2.0, x + node.width / 2.0, y + node.height / 2.0)


def node_bbox(node: Node) -> tuple[float, float, float, float]:
    return node_bbox_xy(node, node.x, node.y)


class SpatialHashIndex:
    def __init__(self, canvas_w: float, canvas_h: float, cell_size: float) -> None:
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h
        self.cell_size = max(cell_size, 1e-3)
        self.cells: dict[tuple[int, int], set[int]] = defaultdict(set)
        self.bboxes: dict[int, tuple[float, float, float, float]] = {}

    def _cell_range(self, bbox: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = bbox
        c0 = int(max(0.0, x0) // self.cell_size)
        c1 = int(max(0.0, x1) // self.cell_size)
        r0 = int(max(0.0, y0) // self.cell_size)
        r1 = int(max(0.0, y1) // self.cell_size)
        return c0, c1, r0, r1

    def insert(self, idx: int, bbox: tuple[float, float, float, float]) -> None:
        self.bboxes[idx] = bbox
        c0, c1, r0, r1 = self._cell_range(bbox)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                self.cells[(r, c)].add(idx)

    def remove(self, idx: int) -> None:
        bbox = self.bboxes.pop(idx, None)
        if bbox is None:
            return
        c0, c1, r0, r1 = self._cell_range(bbox)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                bucket = self.cells.get((r, c))
                if bucket is None:
                    continue
                bucket.discard(idx)
                if not bucket:
                    self.cells.pop((r, c), None)

    def update(self, idx: int, bbox: tuple[float, float, float, float]) -> None:
        self.remove(idx)
        self.insert(idx, bbox)

    def collides(self, bbox: tuple[float, float, float, float], *, exclude: int | None = None) -> bool:
        c0, c1, r0, r1 = self._cell_range(bbox)
        checked: set[int] = set()
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                for idx in self.cells.get((r, c), ()):
                    if idx == exclude or idx in checked:
                        continue
                    checked.add(idx)
                    if bboxes_overlap(bbox, self.bboxes[idx]):
                        return True
        return False


def build_obstacle_index(benchmark: Benchmark) -> SpatialHashIndex:
    macros = [benchmark.nodes[i] for i in benchmark.movable_indices]
    if macros:
        avg_w = sum(n.width for n in macros) / max(1, len(macros))
        avg_h = sum(n.height for n in macros) / max(1, len(macros))
        cell_size = max(0.25, 0.5 * (avg_w + avg_h))
    else:
        cell_size = 1.0
    index = SpatialHashIndex(benchmark.canvas_width, benchmark.canvas_height, cell_size)
    for idx, node in enumerate(benchmark.nodes):
        if node.width <= 0.0 or node.height <= 0.0:
            continue
        index.insert(idx, node_bbox(node))
    return index

