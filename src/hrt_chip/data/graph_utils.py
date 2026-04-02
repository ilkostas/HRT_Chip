"""Graph construction helpers for PyG (Phase 4)."""

from __future__ import annotations

import math
import random

import torch


def complete_edge_index(num_nodes: int) -> torch.Tensor:
    """Undirected all-pairs edges as COO ``[2, E]`` (no self-loops)."""
    rows: list[int] = []
    cols: list[int] = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            rows.extend([i, j])
            cols.extend([j, i])
    if not rows:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor([rows, cols], dtype=torch.long)


def node_features_from_wh(wh: torch.Tensor) -> torch.Tensor:
    """``wh`` is [N, 2] widths/heights in canvas units (matches training)."""
    w = wh[:, 0:1]
    h = wh[:, 1:2]
    area = w * h
    return torch.cat([w, h, torch.log(w + 1e-8), torch.log(h + 1e-8), area], dim=1)


def node_features_from_wh_degree(wh: torch.Tensor, degree: torch.Tensor) -> torch.Tensor:
    """
    Extend WH features with a normalized spatial-graph degree column.

    ``degree`` is [N, 1] non-negative; normalized by max(1, max(degree)).
    """
    base = node_features_from_wh(wh)
    d = degree.to(dtype=torch.float32)
    if d.dim() == 1:
        d = d.unsqueeze(1)
    mx = float(torch.max(d).item()) if d.numel() else 0.0
    scale = max(mx, 1.0)
    d_norm = d / scale
    return torch.cat([base, d_norm], dim=1)


def spatial_neighbor_degrees(
    rng: random.Random,
    pos_unit: torch.Tensor,
    *,
    p0: float = 0.55,
    length_scale: float = 0.35,
) -> torch.Tensor:
    """
    Sample stochastic spatial edges in [0,1]^2 with P(connect) ~ p0 * exp(-dist / length_scale).

    Returns degree column [N, 1]. Guarantees at least one edge on a path if graph was empty.
    """
    n = int(pos_unit.shape[0])
    deg = torch.zeros(n, 1, dtype=torch.float32)
    if n < 2:
        return deg
    for i in range(n):
        for j in range(i + 1, n):
            dx = float(pos_unit[i, 0] - pos_unit[j, 0])
            dy = float(pos_unit[i, 1] - pos_unit[j, 1])
            d = math.hypot(dx, dy)
            prob = p0 * math.exp(-d / max(length_scale, 1e-6))
            if rng.random() < prob:
                deg[i, 0] += 1.0
                deg[j, 0] += 1.0
    if float(torch.sum(deg).item()) < 1.0:
        for i in range(n - 1):
            deg[i, 0] += 1.0
            deg[i + 1, 0] += 1.0
    return deg
