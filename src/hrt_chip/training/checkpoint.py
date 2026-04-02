"""Save / load training checkpoints for Phase 4 diffusion models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from hrt_chip.config import ModelArchitecture, TrainingConfig
from hrt_chip.training.models import EpsilonPlacementNet, build_epsilon_model
from hrt_chip.training.schedule import DiffusionSchedule


def save_checkpoint(
    path: Path,
    *,
    model: EpsilonPlacementNet,
    sched: DiffusionSchedule,
    training_config: TrainingConfig,
    dataset_manifest: dict[str, Any],
    node_feature_dim: int,
    metrics: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "num_timesteps": sched.num_timesteps,
        "training_config": training_config.to_dict(),
        "dataset_manifest": dataset_manifest,
        "model_architecture": model.architecture,
        "node_feature_dim": node_feature_dim,
        "metrics": metrics or {},
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    *,
    device: torch.device | None = None,
) -> tuple[EpsilonPlacementNet, DiffusionSchedule, dict[str, Any]]:
    dev = device or torch.device("cpu")
    payload = torch.load(path, map_location=dev, weights_only=False)
    tc = TrainingConfig.from_dict(payload["training_config"])
    dm = payload["dataset_manifest"]
    node_dim = int(payload.get("node_feature_dim") or 5)
    arch: ModelArchitecture = payload.get("model_architecture") or tc.model_architecture
    T = int(payload.get("num_timesteps") or tc.diffusion_steps)
    model = build_epsilon_model(
        node_dim=node_dim,
        hidden_dim=tc.hidden_dim,
        num_layers=tc.num_layers,
        num_timesteps=T,
        architecture=arch,
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(dev)
    model.eval()
    sched = DiffusionSchedule(T, device=dev)
    meta = {
        "training_config": tc,
        "dataset_manifest": dm,
        "metrics": payload.get("metrics") or {},
    }
    return model, sched, meta


def read_dataset_version_from_manifest(manifest: dict[str, Any]) -> str | None:
    return manifest.get("dataset_version")
