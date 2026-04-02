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


@app.command()
def run(
    benchmark: str = typer.Option("ibm01", "--benchmark", "-b", help="Benchmark id (e.g. ibm01)."),
    seed: int = typer.Option(42, "--seed", "-s", help="RNG seed for deterministic stub generation."),
    candidates: int = typer.Option(
        4, "--candidates", "-k", help="Number of placement candidates to generate."
    ),
    output_dir: Path = typer.Option(
        Path("runs"), "--output-dir", "-o", help="Base directory for run artifacts."
    ),
    run_id: Optional[str] = typer.Option(
        None, "--run-id", help="Optional fixed run UUID (default: auto)."
    ),
) -> None:
    """Run generate -> legalize -> evaluate (stub) and write artifacts under output_dir/<run_id>."""
    cfg = RunConfig(
        benchmark_id=benchmark,
        seed=seed,
        num_candidates=candidates,
        output_dir=output_dir,
        deterministic=True,
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
    console.print(f"best_candidate_id={results.get('best_candidate_id')} "
                  f"best_proxy={results.get('best_proxy_score')}")

    table = Table(title="Candidate ranking (lower proxy is better)")
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
