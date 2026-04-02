"""Synthetic legal macro layouts for DDPM training (normalized centers in [-1, 1])."""

from __future__ import annotations

import math
import random
import uuid
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Data

from hrt_chip.config import SyntheticDatasetConfig
from hrt_chip.data.graph_utils import complete_edge_index as _complete_edge_index
from hrt_chip.io.artifacts import DatasetManifest, utc_now_iso
from hrt_chip.io.artifacts import write_json as write_json_atomic


def _lower_left_to_normalized_center(x: float, y: float, w: float, h: float) -> tuple[float, float]:
    """Unit canvas [0,1]^2 lower-left to normalized center [-1,1]^2."""
    cx = x + w / 2.0
    cy = y + h / 2.0
    return (2.0 * cx - 1.0, 2.0 * cy - 1.0)


def _legal_grid_layout(
    rng: random.Random,
    n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return (x0, macro_wh) where x0 is [N,2] centers in [-1,1], macro_wh is [N,2] widths/heights in (0,1] canvas units.
    Macros occupy a row-major grid with margin so pairwise overlap is zero.
    """
    cols = max(1, int(math.ceil(math.sqrt(n))))
    rows = max(1, int(math.ceil(n / cols)))
    cell_w = 1.0 / cols
    cell_h = 1.0 / rows
    margin = 0.05
    # Fit all macros with same max scale then jitter sizes slightly (still legal).
    base_w = cell_w * (1.0 - margin)
    base_h = cell_h * (1.0 - margin)
    xs: list[float] = []
    ys: list[float] = []
    ws: list[float] = []
    hs: list[float] = []
    for i in range(n):
        col = i % cols
        row = i // cols
        # shrink randomly but stay inside cell
        sw = rng.uniform(0.65, 1.0)
        sh = rng.uniform(0.65, 1.0)
        w = base_w * sw
        h = base_h * sh
        lx = col * cell_w + (cell_w - w) / 2.0
        ly = row * cell_h + (cell_h - h) / 2.0
        cx, cy = _lower_left_to_normalized_center(lx, ly, w, h)
        xs.append(cx)
        ys.append(cy)
        ws.append(w)
        hs.append(h)
    x0 = torch.tensor([[xs[i], ys[i]] for i in range(n)], dtype=torch.float32)
    wh = torch.tensor([[ws[i], hs[i]] for i in range(n)], dtype=torch.float32)
    return x0, wh


def _node_features(wh: torch.Tensor) -> torch.Tensor:
    from hrt_chip.data.graph_utils import node_features_from_wh

    return node_features_from_wh(wh)


def generate_synthetic_dataset(config: SyntheticDatasetConfig) -> Path:
    """
    Write shards under ``config.output_dir`` and ``dataset_manifest.json``.

    Each shard is a list of ``torch_geometric.data.Data`` pickles (torch.save).
    """
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(config.seed)
    torch.manual_seed(config.seed)

    dataset_id = str(uuid.uuid4())
    shard_files: list[str] = []
    shard_size = max(1, min(64, config.num_samples))

    sample_idx = 0
    shard: list[Data] = []
    shard_id = 0

    def flush_shard() -> None:
        nonlocal shard, shard_id
        if not shard:
            return
        name = f"shard_{shard_id:04d}.pt"
        path = out / name
        torch.save(shard, path)
        shard_files.append(name)
        shard = []
        shard_id += 1

    nmin = config.n_macros_min or 2
    nmax = config.n_macros_max or 8

    for _ in range(config.num_samples):
        n = rng.randint(nmin, nmax)
        x0, wh = _legal_grid_layout(rng, n)
        edge_index = _complete_edge_index(n)
        x = _node_features(wh)
        data = Data(
            x=x,
            edge_index=edge_index,
            pos=x0.clone(),
            macro_wh=wh,
            num_nodes=n,
        )
        data.layout_id = sample_idx  # type: ignore[attr-defined]
        shard.append(data)
        sample_idx += 1
        if len(shard) >= shard_size:
            flush_shard()
    flush_shard()

    manifest = DatasetManifest(
        dataset_id=dataset_id,
        dataset_version=config.dataset_version,
        schema_version=config.schema_version,
        corpus_version=config.corpus_version,
        seed=config.seed,
        num_samples=config.num_samples,
        n_macros_min=nmin,
        n_macros_max=nmax,
        created_at_utc=utc_now_iso(),
        data_dir=str(out.resolve()),
        shards=shard_files,
        notes="synthetic grid-packed legal layouts; pos stores x0 centers [-1,1]",
    )
    manifest.write_json(out / "dataset_manifest.json")

    # Convenience copy for training defaults
    write_json_atomic(out / "dataset_config_snapshot.json", config.to_dict())

    return out / "dataset_manifest.json"


def load_manifest(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))
