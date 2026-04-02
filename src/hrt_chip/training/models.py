"""ε_θ graph networks for macro placement diffusion (Phase 4)."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv, GCNConv

from hrt_chip.config import ModelArchitecture

Arch = ModelArchitecture | Literal["baseline_gnn", "res_gnn", "att_gnn"]


class EpsilonPlacementNet(nn.Module):
    """
    Predicts noise ε in R^{N×2} given noisy positions and discrete timesteps per node.

    ``forward(batch, x_noisy, t_node)`` — ``t_node`` is long [num_nodes] in [0, T-1].
    """

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_timesteps: int,
        architecture: Arch = "baseline_gnn",
        gat_heads: int = 4,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.num_timesteps = num_timesteps
        self.time_emb = nn.Embedding(num_timesteps, hidden_dim)
        self.input_dim = node_dim + 2 + hidden_dim

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        if architecture == "att_gnn":
            c_in = self.input_dim
            for i in range(num_layers):
                heads = gat_heads if i < num_layers - 1 else 1
                out_c = hidden_dim // heads if i < num_layers - 1 else hidden_dim
                concat = i < num_layers - 1
                self.layers.append(
                    GATConv(c_in, out_c, heads=heads, concat=concat, dropout=0.1)
                )
                c_in = hidden_dim
                self.norms.append(nn.LayerNorm(hidden_dim))
        elif architecture == "res_gnn":
            self.in_proj = nn.Linear(self.input_dim, hidden_dim)
            for _ in range(num_layers):
                self.layers.append(GCNConv(hidden_dim, hidden_dim))
                self.norms.append(nn.LayerNorm(hidden_dim))
        else:
            self.layers.append(GCNConv(self.input_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
            for _ in range(num_layers - 1):
                self.layers.append(GCNConv(hidden_dim, hidden_dim))
                self.norms.append(nn.LayerNorm(hidden_dim))

        self.out = nn.Linear(hidden_dim, 2)

    def forward(self, batch: Batch, x_noisy: torch.Tensor, t_node: torch.Tensor) -> torch.Tensor:
        te = self.time_emb(t_node)
        h = torch.cat([batch.x, x_noisy, te], dim=-1)

        if self.architecture == "att_gnn":
            for conv, ln in zip(self.layers, self.norms, strict=True):
                h = conv(h, batch.edge_index)
                h = ln(h)
                h = F.elu(h)
            return self.out(h)

        if self.architecture == "res_gnn":
            h = self.in_proj(h)
            for conv, ln in zip(self.layers, self.norms, strict=True):
                residual = h
                h = conv(h, batch.edge_index)
                h = ln(h)
                h = F.elu(h + residual)
            return self.out(h)

        for conv, ln in zip(self.layers, self.norms, strict=True):
            h = conv(h, batch.edge_index)
            h = ln(h)
            h = F.elu(h)
        return self.out(h)


def build_epsilon_model(
    *,
    node_dim: int,
    hidden_dim: int,
    num_layers: int,
    num_timesteps: int,
    architecture: Arch,
) -> EpsilonPlacementNet:
    return EpsilonPlacementNet(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_timesteps=num_timesteps,
        architecture=architecture,
    )
