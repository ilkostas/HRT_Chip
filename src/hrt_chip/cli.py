"""CLI entrypoint: `hrt-chip` or `python -m hrt_chip`."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from hrt_chip.config import RunConfig
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
) -> None:
    """Run generate -> legalize -> evaluate (stub) and write artifacts under output_dir/<run_id>."""
    gw = _parse_guidance_weight_triples(guidance_weight)
    cfg = RunConfig(
        benchmark_id=benchmark,
        seed=seed,
        num_candidates=candidates,
        diffusion_steps=diffusion_steps,
        output_dir=output_dir,
        deterministic=True,
        guidance_preset=None if gw else guidance_preset,
        guidance_weights_sweep=gw,
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


def _print_summary(results: dict, cfg: RunConfig) -> None:
    mid = results["manifest"]["run_id"]
    console.print(f"[bold green]Run complete[/bold green] run_id={mid}")
    console.print(f"benchmark={cfg.benchmark_id} seed={cfg.seed} candidates={cfg.num_candidates}")
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
