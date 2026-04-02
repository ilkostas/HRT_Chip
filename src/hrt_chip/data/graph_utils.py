"""Graph construction helpers for PyG (Phase 4)."""

from __future__ import annotations

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
