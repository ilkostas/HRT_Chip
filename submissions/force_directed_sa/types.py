from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Node:
    name: str
    node_type: str
    width: float
    height: float
    x: float
    y: float
    orientation: str = "N"
    is_fixed: bool = False

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)


@dataclass(frozen=True)
class PinRef:
    node_idx: int
    x_offset: float = 0.0
    y_offset: float = 0.0


@dataclass
class Net:
    pin_refs: list[PinRef]
    weight: float = 1.0
    name: str = ""


@dataclass
class Benchmark:
    benchmark_id: str
    canvas_width: float
    canvas_height: float
    grid_cols: int
    grid_rows: int
    routes_h: float
    routes_v: float
    smooth_range: int
    overlap_threshold: float
    nodes: list[Node]
    nets: list[Net]
    movable_indices: list[int]
    node_to_nets: list[list[int]] = field(default_factory=list)

