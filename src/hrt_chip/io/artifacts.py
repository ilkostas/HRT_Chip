"""Structured run artifacts and manifest (reproducibility baseline)."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hrt_chip.config import RunConfig


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RunManifest:
    """Written alongside results for replay and auditing."""

    run_id: str
    created_at_utc: str
    config: dict[str, Any]
    deterministic_mode: bool
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def new_run_id() -> str:
    return str(uuid.uuid4())


def build_manifest(config: RunConfig, *, run_id: str | None = None, notes: str = "") -> RunManifest:
    rid = run_id or new_run_id()
    return RunManifest(
        run_id=rid,
        created_at_utc=utc_now_iso(),
        config=config.to_dict(),
        deterministic_mode=config.deterministic,
        notes=notes,
    )


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


@dataclass
class DatasetManifest:
    """Versioned metadata for a synthetic on-disk dataset (Phase 4)."""

    dataset_id: str
    dataset_version: str
    schema_version: str
    corpus_version: str
    seed: int
    num_samples: int
    n_macros_min: int
    n_macros_max: int
    created_at_utc: str
    data_dir: str
    shards: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: Path) -> None:
        write_json(path, self.to_dict())


@dataclass
class TrainingRunManifest:
    """Written next to checkpoints for replay and auditing."""

    train_run_id: str
    created_at_utc: str
    config: dict[str, Any]
    dataset_manifest_path: str
    dataset_version: str
    checkpoint_path: str
    metrics_path: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: Path) -> None:
        write_json(path, self.to_dict())


@dataclass
class PipelineArtifacts:
    """Paths for one pipeline run."""

    run_dir: Path
    manifest_path: Path = field(init=False)
    results_path: Path = field(init=False)
    candidates_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.manifest_path = self.run_dir / "manifest.json"
        self.results_path = self.run_dir / "results.json"
        self.candidates_dir = self.run_dir / "candidates"

    def ensure_dirs(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
