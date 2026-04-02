"""Phase 4: synthetic data, training smoke, PyTorch sampler, artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from hrt_chip.config import RunConfig, SyntheticDatasetConfig, TrainingConfig
from hrt_chip.data.pyg_dataset import PlacementGraphDataset
from hrt_chip.data.synthetic import generate_synthetic_dataset
from hrt_chip.pipeline import run_pipeline
from hrt_chip.training.train import train_loop


def test_dataset_manifest_deterministic(tmp_path: Path) -> None:
    cfg = SyntheticDatasetConfig(
        output_dir=tmp_path / "ds_a",
        corpus_version="v1",
        seed=123,
        num_samples=5,
        dataset_version="test-1",
    )
    p1 = generate_synthetic_dataset(cfg)
    cfg2 = SyntheticDatasetConfig(
        output_dir=tmp_path / "ds_b",
        corpus_version="v1",
        seed=123,
        num_samples=5,
        dataset_version="test-1",
    )
    p2 = generate_synthetic_dataset(cfg2)
    m1 = json.loads(p1.read_text(encoding="utf-8"))
    m2 = json.loads(p2.read_text(encoding="utf-8"))
    assert m1["num_samples"] == m2["num_samples"] == 5
    assert m1["seed"] == m2["seed"] == 123
    assert m1["shards"] == m2["shards"]
    # dataset_id is random UUID per run — not compared


def test_train_smoke_and_checkpoint(tmp_path: Path) -> None:
    ds_dir = tmp_path / "ds"
    generate_synthetic_dataset(
        SyntheticDatasetConfig(
            output_dir=ds_dir,
            corpus_version="v1",
            seed=7,
            num_samples=4,
            dataset_version="t",
        )
    )
    tr = TrainingConfig(
        dataset_dir=ds_dir,
        output_dir=tmp_path / "tr",
        seed=1,
        epochs=1,
        batch_size=2,
        diffusion_steps=20,
        model_architecture="baseline_gnn",
        hidden_dim=32,
        num_layers=2,
        train_run_id="00000000-0000-0000-0000-0000000000aa",
    )
    out = train_loop(tr)
    ckpt = Path(out["checkpoint_path"])
    assert ckpt.is_file()
    man = json.loads((tmp_path / "tr" / "00000000-0000-0000-0000-0000000000aa" / "training_manifest.json").read_text())
    assert man["dataset_version"] == "t"


def test_run_with_pytorch_sampler(tmp_path: Path) -> None:
    ds_dir = tmp_path / "ds"
    generate_synthetic_dataset(
        SyntheticDatasetConfig(
            output_dir=ds_dir,
            corpus_version="v1",
            seed=11,
            num_samples=6,
            dataset_version="run-t",
        )
    )
    train_loop(
        TrainingConfig(
            dataset_dir=ds_dir,
            output_dir=tmp_path / "tr",
            seed=2,
            epochs=1,
            batch_size=2,
            diffusion_steps=10,
            model_architecture="res_gnn",
            hidden_dim=32,
            num_layers=2,
            train_run_id="00000000-0000-0000-0000-0000000000bb",
        )
    )
    ckpt = tmp_path / "tr" / "00000000-0000-0000-0000-0000000000bb" / "checkpoint.pt"
    cfg = RunConfig(
        benchmark_id="ibm01",
        seed=99,
        num_candidates=1,
        diffusion_steps=10,
        output_dir=tmp_path / "runs",
        deterministic=True,
        sampler_backend="pytorch_checkpoint",
        checkpoint_path=ckpt,
        training_dataset_version="run-t",
        model_architecture="res_gnn",
        mixed_size_backend="stub",
    )
    r = run_pipeline(cfg, run_id="00000000-0000-0000-0000-0000000000cc")
    sp = r.get("sampler_provenance") or {}
    assert sp.get("checkpoint_path")
    assert r.get("training_dataset_version") == "run-t" or sp.get("training_dataset_version") == "run-t"
    assert r.get("sampler_backend") == "pytorch_checkpoint"


def test_benchmark_like_curriculum_six_features(tmp_path: Path) -> None:
    ds_dir = tmp_path / "ds_bl"
    generate_synthetic_dataset(
        SyntheticDatasetConfig(
            output_dir=ds_dir,
            corpus_version="v1",
            seed=99,
            num_samples=3,
            dataset_version="bl",
            curriculum="benchmark_like",
        )
    )
    ds = PlacementGraphDataset(ds_dir)
    assert ds[0].x.shape[1] == 6


def test_run_with_pytorch_sampler_accelerated_steps(tmp_path: Path) -> None:
    ds_dir = tmp_path / "ds"
    generate_synthetic_dataset(
        SyntheticDatasetConfig(
            output_dir=ds_dir,
            corpus_version="v1",
            seed=11,
            num_samples=6,
            dataset_version="run-t",
        )
    )
    train_loop(
        TrainingConfig(
            dataset_dir=ds_dir,
            output_dir=tmp_path / "tr",
            seed=2,
            epochs=1,
            batch_size=2,
            diffusion_steps=20,
            model_architecture="res_gnn",
            hidden_dim=32,
            num_layers=2,
            train_run_id="00000000-0000-0000-0000-0000000000dd",
        )
    )
    ckpt = tmp_path / "tr" / "00000000-0000-0000-0000-0000000000dd" / "checkpoint.pt"
    cfg = RunConfig(
        benchmark_id="ibm01",
        seed=99,
        num_candidates=1,
        diffusion_steps=20,
        diffusion_inference_steps=5,
        output_dir=tmp_path / "runs",
        deterministic=True,
        sampler_backend="pytorch_checkpoint",
        checkpoint_path=ckpt,
        mixed_size_backend="stub",
    )
    r = run_pipeline(cfg, run_id="00000000-0000-0000-0000-0000000000ee")
    sp = r.get("sampler_provenance") or {}
    assert sp.get("diffusion_steps") == 5
