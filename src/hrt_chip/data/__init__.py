"""Synthetic datasets and PyG adapters (Phase 4)."""

from hrt_chip.data.manifest import build_dataset_manifest
from hrt_chip.data.synthetic import generate_synthetic_dataset
from hrt_chip.data.pyg_dataset import PlacementGraphDataset, load_dataset_from_dir

__all__ = [
    "build_dataset_manifest",
    "generate_synthetic_dataset",
    "PlacementGraphDataset",
    "load_dataset_from_dir",
]
