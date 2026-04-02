"""CLI entrypoint: `hrt-chip` or `python -m hrt_chip`."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, cast

import typer
from rich.console import Console
from rich.table import Table

from hrt_chip.config import RunConfig, SamplerBackend, SyntheticDatasetConfig, TrainingConfig
from hrt_chip.pipeline import replay_from_manifest, run_pipeline

app = typer.Typer(no_args_is_help=True, help="HRT macro placement pipeline.")
console = Console()


def _parse_guidance_weight_triples(values: list[str] | None) -> tuple[tuple[float, float, float], ...] | None:
    if not values:
        return None
    out: list[tuple[float, float, float]] = []
    for raw in values:
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) != 3:
            raise typer.BadParameter(
                f"Expected three comma-separated floats (alpha,beta,gamma), got {raw!r}"
            )
        try:
            a, b, g = float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError as e:
            raise typer.BadParameter(f"Invalid floats in {raw!r}") from e
        out.append((a, b, g))
    return tuple(out)


@app.command()
def run(
    benchmark: str = typer.Option("ibm01", "--benchmark", "-b", help="Benchmark id (e.g. ibm01)."),
    seed: int = typer.Option(42, "--seed", "-s", help="RNG seed for deterministic stub generation."),
    candidates: int = typer.Option(
        4,
        "--candidates",
        "-k",
        help="Number of placement candidates per guidance weight vector (Phase 3 sweep).",
    ),
    diffusion_steps: int = typer.Option(
        1000,
        "--diffusion-steps",
        help="DDPM timestep count (stub uses for provenance; PyTorch path in Phase 4).",
    ),
    output_dir: Path = typer.Option(
        Path("runs"), "--output-dir", "-o", help="Base directory for run artifacts."
    ),
    run_id: Optional[str] = typer.Option(
        None, "--run-id", help="Optional fixed run UUID (default: auto)."
    ),
    guidance_preset: Optional[str] = typer.Option(
        None,
        "--guidance-preset",
        help="Built-in multi-weight sweep: pareto3 (see docs). Ignored if --guidance-weight is set.",
    ),
    guidance_weight: Optional[list[str]] = typer.Option(
        None,
        "--guidance-weight",
        multiple=True,
        help=(
            "One triple alpha,beta,gamma (HPWL, congestion, legality surrogates). "
            "Repeat option for multiple vectors; overrides --guidance-preset."
        ),
    ),
    sampler_backend: str = typer.Option(
        "stub",
        "--sampler-backend",
        help="stub (default) or pytorch_checkpoint (Phase 4 trained DDPM).",
    ),
    checkpoint: Optional[Path] = typer.Option(
        None,
        "--checkpoint",
        help="Path to checkpoint.pt when --sampler-backend=pytorch_checkpoint.",
    ),
    training_dataset_version: Optional[str] = typer.Option(
        None,
        "--training-dataset-version",
        help="Optional dataset version string for audit (defaults from checkpoint manifest).",
    ),
    model_architecture: Optional[str] = typer.Option(
        None,
        "--model-architecture",
        help="Optional echo of train-time architecture (baseline_gnn, res_gnn, att_gnn).",
    ),
) -> None:
    """Run generate -> legalize -> evaluate (stub) and write artifacts under output_dir/<run_id>."""
    if sampler_backend == "pytorch_checkpoint" and checkpoint is None:
        raise typer.BadParameter("--checkpoint is required when --sampler-backend=pytorch_checkpoint")
    gw = _parse_guidance_weight_triples(guidance_weight)
    arch = cast(Optional[Literal["baseline_gnn", "res_gnn", "att_gnn"]], model_architecture)
    cfg = RunConfig(
        benchmark_id=benchmark,
        seed=seed,
        num_candidates=candidates,
        diffusion_steps=diffusion_steps,
        output_dir=output_dir,
        deterministic=True,
        guidance_preset=None if gw else guidance_preset,
        guidance_weights_sweep=gw,
        sampler_backend=cast(SamplerBackend, sampler_backend),
        checkpoint_path=checkpoint,
        training_dataset_version=training_dataset_version,
        model_architecture=arch,
    )
    results = run_pipeline(cfg, run_id=run_id)
    _print_summary(results, cfg)


@app.command("replay")
def replay_cmd(
    manifest: Path = typer.Argument(..., exists=True, help="Path to manifest.json from a prior run."),
) -> None:
    """Re-execute pipeline from a saved manifest (reproducibility check)."""
    results = replay_from_manifest(str(manifest))
    cfg = RunConfig.from_dict(results["manifest"]["config"])
    _print_summary(results, cfg)


@app.command("dataset-generate")
def dataset_generate_cmd(
    output_dir: Path = typer.Option(
        Path("data/synthetic/v1"),
        "--output-dir",
        "-o",
        help="Directory to write shards and dataset_manifest.json.",
    ),
    corpus: str = typer.Option(
        "v1",
        "--corpus",
        "-c",
        help="Synthetic corpus scale: v1 (smaller graphs) or v2 (larger).",
    ),
    seed: int = typer.Option(42, "--seed", "-s"),
    num_samples: int = typer.Option(256, "--num-samples", "-n"),
    dataset_version: str = typer.Option("1", "--dataset-version"),
) -> None:
    """Generate synthetic legal layouts and PyG shards (Phase 4)."""
    if corpus not in ("v1", "v2"):
        raise typer.BadParameter("--corpus must be v1 or v2")
    from hrt_chip.data.synthetic import generate_synthetic_dataset

    cfg = SyntheticDatasetConfig(
        output_dir=output_dir,
        corpus_version=cast(Literal["v1", "v2"], corpus),
        seed=seed,
        num_samples=num_samples,
        dataset_version=dataset_version,
    )
    manifest_path = generate_synthetic_dataset(cfg)
    console.print(f"[bold green]Dataset written[/bold green] manifest={manifest_path}")


@app.command("train")
def train_cmd(
    dataset_dir: Path = typer.Option(
        ...,
        "--dataset-dir",
        "-d",
        exists=True,
        file_okay=False,
        help="Directory containing dataset_manifest.json and shard_*.pt files.",
    ),
    output_dir: Path = typer.Option(Path("training_runs"), "--output-dir", "-o"),
    seed: int = typer.Option(42, "--seed", "-s"),
    epochs: int = typer.Option(10, "--epochs", "-e"),
    batch_size: int = typer.Option(8, "--batch-size", "-b"),
    learning_rate: float = typer.Option(1e-3, "--lr"),
    diffusion_steps: int = typer.Option(1000, "--diffusion-steps", help="DDPM timesteps T."),
    model_architecture: str = typer.Option(
        "baseline_gnn",
        "--model-architecture",
        "-m",
        help="baseline_gnn | res_gnn | att_gnn",
    ),
    hidden_dim: int = typer.Option(64, "--hidden-dim"),
    num_layers: int = typer.Option(3, "--num-layers"),
    train_run_id: Optional[str] = typer.Option(None, "--train-run-id"),
) -> None:
    """Train ε-prediction DDPM on a synthetic dataset (Phase 4)."""
    if model_architecture not in ("baseline_gnn", "res_gnn", "att_gnn"):
        raise typer.BadParameter("--model-architecture must be baseline_gnn, res_gnn, or att_gnn")
    from hrt_chip.training.train import train_loop

    cfg = TrainingConfig(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        diffusion_steps=diffusion_steps,
        model_architecture=cast(Literal["baseline_gnn", "res_gnn", "att_gnn"], model_architecture),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        train_run_id=train_run_id,
    )
    out = train_loop(cfg)
    console.print(f"[bold green]Training complete[/bold green] {out}")


def _print_summary(results: dict, cfg: RunConfig) -> None:
    mid = results["manifest"]["run_id"]
    console.print(f"[bold green]Run complete[/bold green] run_id={mid}")
    console.print(f"benchmark={cfg.benchmark_id} seed={cfg.seed} candidates={cfg.num_candidates}")
    console.print(f"sampler_backend={cfg.sampler_backend}")
    if cfg.checkpoint_path:
        console.print(f"checkpoint={cfg.checkpoint_path}")
    gsr = results.get("guidance_sweep_resolved")
    if gsr:
        console.print(f"guidance_sweep_resolved={gsr}")
    console.print(f"best_candidate_id={results.get('best_candidate_id')} "
                  f"best_proxy={results.get('best_proxy_score')}")

    table = Table(title="Candidate ranking (lower proxy is better; official proxy only)")
    table.add_column("Rank", justify="right")
    table.add_column("candidate_id")
    table.add_column("proxy_score", justify="right")
    table.add_column("legal")

    ranking = results.get("ranking") or []
    for i, row in enumerate(ranking, start=1):
        ps = row["proxy_score"]
        ps_s = "inf" if ps == float("inf") else f"{float(ps):.6f}"
        table.add_row(
            str(i),
            str(row["candidate_id"]),
            ps_s,
            str(row["legal"]),
        )
    console.print(table)
    out = cfg.output_dir / mid
    console.print(f"Artifacts: [cyan]{out}[/cyan]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
