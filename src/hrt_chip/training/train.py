"""Training loop for ε-prediction DDPM on synthetic placement graphs."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from hrt_chip.config import TrainingConfig
from hrt_chip.data.pyg_dataset import PlacementGraphDataset
from hrt_chip.io.artifacts import TrainingRunManifest, utc_now_iso
from hrt_chip.io.artifacts import write_json as write_json_atomic
from hrt_chip.training.checkpoint import save_checkpoint
from hrt_chip.training.models import build_epsilon_model
from hrt_chip.training.schedule import DiffusionSchedule, q_sample


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def train_loop(config: TrainingConfig) -> dict[str, Any]:
    """Train DDPM ε-model; write checkpoint, metrics, and training manifest under ``config.output_dir``."""
    _set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_root = Path(config.dataset_dir)
    manifest_path = ds_root / "dataset_manifest.json"
    dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ds = PlacementGraphDataset(ds_root, manifest_path=manifest_path)
    if len(ds) == 0:
        raise ValueError(f"No samples in dataset at {ds_root}")

    sample0 = ds[0]
    node_dim = int(sample0.x.shape[1])
    T = config.diffusion_steps
    sched = DiffusionSchedule(T, device=device)

    model = build_epsilon_model(
        node_dim=node_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_timesteps=T,
        architecture=config.model_architecture,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=True)

    metrics_rows: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            batch = batch.to(device)
            x0 = batch.pos
            assert batch.batch is not None
            num_graphs = int(batch.batch.max().item()) + 1
            t = torch.randint(0, T, (num_graphs,), device=device)
            t_node = t[batch.batch]
            noise = torch.randn_like(x0)
            x_t = q_sample(x0, t_node, noise, sched)
            pred = model(batch, x_t, t_node)
            loss = F.mse_loss(pred, noise)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            n_batches += 1
        avg = total_loss / max(1, n_batches)
        metrics_rows.append({"epoch": float(epoch), "loss": avg})

    # Artifacts
    from hrt_chip.io.artifacts import new_run_id

    train_id = config.train_run_id or new_run_id()
    out_root = Path(config.output_dir) / train_id
    out_root.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_root / "checkpoint.pt"
    metrics_path = out_root / "metrics.json"

    model.eval()
    save_checkpoint(
        ckpt_path,
        model=model,
        sched=sched,
        training_config=config,
        dataset_manifest=dataset_manifest,
        node_feature_dim=node_dim,
        metrics={"epochs": metrics_rows},
    )
    write_json_atomic(metrics_path, {"epochs": metrics_rows})

    tr_manifest = TrainingRunManifest(
        train_run_id=train_id,
        created_at_utc=utc_now_iso(),
        config=config.to_dict(),
        dataset_manifest_path=str(manifest_path.resolve()),
        dataset_version=str(dataset_manifest.get("dataset_version", "")),
        checkpoint_path=str(ckpt_path.resolve()),
        metrics_path=str(metrics_path.resolve()),
        notes="Phase 4 DDPM epsilon training",
    )
    tr_manifest.write_json(out_root / "training_manifest.json")

    return {
        "train_run_id": train_id,
        "checkpoint_path": str(ckpt_path),
        "metrics_path": str(metrics_path),
        "training_manifest": str(out_root / "training_manifest.json"),
        "final_loss": metrics_rows[-1]["loss"] if metrics_rows else None,
    }
