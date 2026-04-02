"""CLI entrypoint: `hrt-chip` or `python -m hrt_chip`."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, cast

import typer
from rich.console import Console
from rich.table import Table

from hrt_chip.benchmark_sweep import run_ibm_benchmark_sweep
from hrt_chip.benchmarks import (
    AGGREGATE_REPLACE_PROXY,
    AGGREGATE_SA_PROXY,
    IBM_BENCHMARKS,
    REPLACE_BASELINE_BY_DESIGN,
    SA_BASELINE_BY_DESIGN,
)
from hrt_chip.config import (
    ArtifactRetentionMode,
    EvaluatorBackend,
    RunConfig,
    SamplerBackend,
    SyntheticDatasetConfig,
    TrainingConfig,
)
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
    evaluator: str = typer.Option(
        "stub",
        "--evaluator",
        "-e",
        help="stub (default) or official (macro_place + external/MacroPlacement testcases).",
    ),
    testcase_root: Optional[Path] = typer.Option(
        None,
        "--testcase-root",
        help="ICCAD04 testcase root (default: env HRT_CHIP_TESTCASE_ROOT or external/MacroPlacement/Testcases/ICCAD04).",
    ),
    deterministic: bool = typer.Option(
        True,
        "--deterministic/--no-deterministic",
        help="Seed RNGs for repeatable runs (default: on).",
    ),
    deterministic_verification: bool = typer.Option(
        False,
        "--deterministic-verification",
        help="Phase 6: strict PyTorch/cuDNN determinism (slower; use with replay --verify).",
    ),
    artifact_retention: str = typer.Option(
        "full",
        "--artifact-retention",
        help="Phase 6: full | compact | best_only (per-candidate JSON pruning after run).",
    ),
    artifact_retention_top_k: Optional[int] = typer.Option(
        None,
        "--artifact-retention-top-k",
        help="With compact retention, keep top-K candidate JSONs by proxy (omit = keep none).",
    ),
) -> None:
    """Run generate -> legalize -> evaluate (stub) and write artifacts under output_dir/<run_id>."""
    if sampler_backend == "pytorch_checkpoint" and checkpoint is None:
        raise typer.BadParameter("--checkpoint is required when --sampler-backend=pytorch_checkpoint")
    if evaluator not in ("stub", "official"):
        raise typer.BadParameter("--evaluator must be stub or official")
    if artifact_retention not in ("full", "compact", "best_only"):
        raise typer.BadParameter("--artifact-retention must be full, compact, or best_only")
    gw = _parse_guidance_weight_triples(guidance_weight)
    arch = cast(Optional[Literal["baseline_gnn", "res_gnn", "att_gnn"]], model_architecture)
    cfg = RunConfig(
        benchmark_id=benchmark,
        seed=seed,
        num_candidates=candidates,
        diffusion_steps=diffusion_steps,
        output_dir=output_dir,
        deterministic=deterministic,
        deterministic_verification=deterministic_verification,
        artifact_retention=cast(ArtifactRetentionMode, artifact_retention),
        artifact_retention_top_k=artifact_retention_top_k,
        guidance_preset=None if gw else guidance_preset,
        guidance_weights_sweep=gw,
        sampler_backend=cast(SamplerBackend, sampler_backend),
        checkpoint_path=checkpoint,
        training_dataset_version=training_dataset_version,
        model_architecture=arch,
        evaluator_backend=cast(EvaluatorBackend, evaluator),
        testcase_root=testcase_root,
    )
    results = run_pipeline(cfg, run_id=run_id)
    _print_summary(results, cfg)


@app.command("replay")
def replay_cmd(
    manifest: Path = typer.Argument(..., exists=True, help="Path to manifest.json from a prior run."),
    verify: bool = typer.Option(
        False,
        "--verify",
        "-v",
        help="Phase 6: compare replay to existing results.json; write replay_verification.json; exit 1 on mismatch.",
    ),
) -> None:
    """Re-execute pipeline from a saved manifest (reproducibility check)."""
    results = replay_from_manifest(str(manifest), verify=verify)
    cfg = RunConfig.from_dict(results["manifest"]["config"])
    _print_summary(results, cfg)
    rv = results.get("replay_verification")
    if verify and rv is not None:
        if rv.get("ok"):
            console.print("[bold green]Replay verification: PASS[/bold green]")
        else:
            console.print("[bold red]Replay verification: FAIL[/bold red]")
            for m in rv.get("mismatches") or []:
                console.print(f"  - {m}")
            raise typer.Exit(code=1)


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


@app.command("benchmark-sweep")
def benchmark_sweep_cmd(
    seed: int = typer.Option(42, "--seed", "-s"),
    candidates: int = typer.Option(
        4,
        "--candidates",
        "-k",
        help="Candidates per guidance weight (same as hrt-chip run).",
    ),
    diffusion_steps: int = typer.Option(1000, "--diffusion-steps"),
    output_dir: Path = typer.Option(
        Path("runs/sweeps"),
        "--output-dir",
        "-o",
        help="Base directory; sweep writes output_dir/<sweep_id>/...",
    ),
    sweep_id: Optional[str] = typer.Option(None, "--sweep-id", help="Optional sweep folder name (default: UUID)."),
    evaluator: str = typer.Option(
        "official",
        "--evaluator",
        "-e",
        help="official (default) or stub (CI / no testcase tree).",
    ),
    testcase_root: Optional[Path] = typer.Option(None, "--testcase-root"),
    guidance_preset: Optional[str] = typer.Option(
        None,
        "--guidance-preset",
        help="Built-in sweep: pareto3. Ignored if --guidance-weight is set.",
    ),
    guidance_weight: Optional[list[str]] = typer.Option(
        None,
        "--guidance-weight",
        multiple=True,
        help="Repeatable triple alpha,beta,gamma; overrides --guidance-preset.",
    ),
    sampler_backend: str = typer.Option("stub", "--sampler-backend"),
    checkpoint: Optional[Path] = typer.Option(None, "--checkpoint"),
    training_dataset_version: Optional[str] = typer.Option(None, "--training-dataset-version"),
    model_architecture: Optional[str] = typer.Option(None, "--model-architecture"),
    benchmark: Optional[list[str]] = typer.Option(
        None,
        "--benchmark",
        "-b",
        help="Restrict sweep to these benchmark ids (repeatable). Default: all 17 IBM designs.",
    ),
) -> None:
    """Run all 17 IBM benchmarks, print gate status, write sweep_report.json."""
    if evaluator not in ("stub", "official"):
        raise typer.BadParameter("--evaluator must be stub or official")
    if sampler_backend == "pytorch_checkpoint" and checkpoint is None:
        raise typer.BadParameter("--checkpoint is required when --sampler-backend=pytorch_checkpoint")
    gw = _parse_guidance_weight_triples(guidance_weight)
    arch = cast(Optional[Literal["baseline_gnn", "res_gnn", "att_gnn"]], model_architecture)
    base = RunConfig(
        benchmark_id="ibm01",
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
        evaluator_backend=cast(EvaluatorBackend, evaluator),
        testcase_root=testcase_root,
    )
    bench_tuple: tuple[str, ...] | None = None
    if benchmark:
        bench_tuple = tuple(benchmark)
        unknown = set(bench_tuple) - set(IBM_BENCHMARKS)
        if unknown:
            raise typer.BadParameter(f"Unknown benchmark id(s): {sorted(unknown)}")
    report, meta = run_ibm_benchmark_sweep(
        base,
        sweep_output_dir=output_dir,
        sweep_id=sweep_id,
        benchmarks=bench_tuple,
    )
    sweep_root = meta["sweep_root"]

    console.print(f"[bold green]Benchmark sweep[/bold green] sweep_id={report.sweep_id}")
    console.print(f"evaluator={evaluator} artifacts={sweep_root}")

    table = Table(
        title="IBM ICCAD04 sweep (proxy = official evaluator when available)",
    )
    table.add_column("Benchmark", justify="right")
    table.add_column("Proxy", justify="right")
    table.add_column("SA", justify="right")
    table.add_column("RePlAce", justify="right")
    table.add_column("vs SA", justify="right")
    table.add_column("vs RePlAce", justify="right")
    table.add_column("Overlaps", justify="right")
    table.add_column("Time s", justify="right")

    for row in report.rows:
        sa_b = SA_BASELINE_BY_DESIGN.get(row.benchmark_id)
        rep_b = REPLACE_BASELINE_BY_DESIGN.get(row.benchmark_id)
        ps = row.proxy_score
        ps_s = "—" if ps is None or row.error else f"{ps:.4f}"
        sa_s = f"{sa_b:.4f}" if sa_b is not None else "—"
        rep_s = f"{rep_b:.4f}" if rep_b is not None else "—"
        if ps is not None and sa_b is not None and sa_b != 0:
            vsa = f"{(sa_b - ps) / sa_b * 100:+.1f}%"
        else:
            vsa = "—"
        if ps is not None and rep_b is not None and rep_b != 0:
            vrep = f"{(rep_b - ps) / rep_b * 100:+.1f}%"
        else:
            vrep = "—"
        ov = "—" if row.overlaps is None else str(row.overlaps)
        if row.error:
            ps_s = "ERR"
            vsa = vrep = "—"
        table.add_row(
            row.benchmark_id,
            ps_s,
            sa_s,
            rep_s,
            vsa,
            vrep,
            ov,
            f"{row.runtime_seconds:.2f}",
        )

    g = report.gates
    mean_p = g.mean_proxy
    mp_s = "—" if mean_p is None else f"{mean_p:.4f}"
    if mean_p is not None:
        vsa_avg = f"{(AGGREGATE_SA_PROXY - mean_p) / AGGREGATE_SA_PROXY * 100:+.1f}%"
        vrep_avg = f"{(AGGREGATE_REPLACE_PROXY - mean_p) / AGGREGATE_REPLACE_PROXY * 100:+.1f}%"
    else:
        vsa_avg = vrep_avg = "—"
    table.add_row(
        "AVG",
        mp_s,
        f"{AGGREGATE_SA_PROXY:.4f}",
        f"{AGGREGATE_REPLACE_PROXY:.4f}",
        vsa_avg,
        vrep_avg,
        "—",
        f"{report.total_runtime_seconds:.2f}",
    )
    console.print(table)

    console.print(
        f"Gate A (100% legal): {'PASS' if g.gate_a_legal_all else 'FAIL'}  "
        f"({g.legal_count}/{g.total_count} legal)"
    )
    console.print(
        f"Gate B (avg proxy < SA {AGGREGATE_SA_PROXY}): "
        f"{'PASS' if g.gate_b_beat_sa_aggregate else 'FAIL'}"
    )
    console.print(
        f"Gate C (avg proxy < RePlAce {AGGREGATE_REPLACE_PROXY}): "
        f"{'PASS' if g.gate_c_beat_replace_aggregate else 'FAIL'}"
    )
    if meta.get("errors"):
        console.print("[yellow]Some benchmarks raised exceptions; see sweep_report.json[/yellow]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
