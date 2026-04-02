"""Load on-disk synthetic shards as a PyTorch / PyG dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data


class PlacementGraphDataset(TorchDataset):
    """Loads Phase 4 ``shard_*.pt`` lists of PyG ``Data`` graphs."""

    def __init__(self, root: Path | str, manifest_path: Path | str | None = None) -> None:
        self.root_path = Path(root)
        mp = Path(manifest_path) if manifest_path else self.root_path / "dataset_manifest.json"
        self.manifest: dict[str, Any] = json.loads(mp.read_text(encoding="utf-8"))
        self._graphs: list[Data] = []
        for name in self.manifest.get("shards", []):
            shard = torch.load(self.root_path / name, map_location="cpu", weights_only=False)
            if not isinstance(shard, list):
                raise TypeError(f"Expected list in shard {name}")
            self._graphs.extend(shard)

    def __len__(self) -> int:
        return len(self._graphs)

    def __getitem__(self, idx: int) -> Data:
        return self._graphs[idx]


def load_dataset_from_dir(path: Path | str) -> PlacementGraphDataset:
    return PlacementGraphDataset(path)
