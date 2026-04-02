"""Helpers to build dataset manifests."""

from __future__ import annotations

from pathlib import Path

from hrt_chip.config import SyntheticDatasetConfig
from hrt_chip.io.artifacts import DatasetManifest, utc_now_iso


def build_dataset_manifest(
    *,
    dataset_dir: Path,
    dataset_id: str,
    config: SyntheticDatasetConfig,
    shards: list[str],
) -> DatasetManifest:
    nmin = config.n_macros_min or 2
    nmax = config.n_macros_max or 8
    return DatasetManifest(
        dataset_id=dataset_id,
        dataset_version=config.dataset_version,
        schema_version=config.schema_version,
        corpus_version=config.corpus_version,
        seed=config.seed,
        num_samples=config.num_samples,
        n_macros_min=nmin,
        n_macros_max=nmax,
        created_at_utc=utc_now_iso(),
        data_dir=str(dataset_dir.resolve()),
        shards=shards,
        notes="",
    )
