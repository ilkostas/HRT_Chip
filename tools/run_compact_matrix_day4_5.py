#!/usr/bin/env python3
"""
Day 4-5 helper: run a compact E1-E4 performance matrix on DEV_SUBSET,
then write finalists only if they beat the incumbent mean proxy under
legality (Gate A) and a runtime sanity check.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _run(*, args: list[str], cwd: Path) -> None:
    subprocess.check_call(args, cwd=cwd)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pick_dev_benchmarks_from_incumbent(incumbent: dict[str, Any]) -> tuple[str, ...]:
    dev_report_path = Path(incumbent["evidence"]["dev"]["sweep_report_path"])
    dev_report = _load_json(dev_report_path)
    rows = dev_report.get("rows") or []
    ids = [str(r["benchmark_id"]) for r in rows if r.get("benchmark_id") is not None]
    # Keep stable order as given by the sweep.
    return tuple(ids)


def _sweep_run_root(output_dir: Path, sweep_id: str) -> Path:
    return output_dir / sweep_id


def _run_matrix_experiment(
    *,
    repo_root: Path,
    output_dir: Path,
    sweep_id: str,
    benchmark_ids: tuple[str, ...],
    competition_profile: dict[str, Any],
    sampler_mode: str,
    diffusion_inference_steps: int,
) -> dict[str, Any]:
    checkpoint = Path(competition_profile["checkpoint_path"])
    cmd = [
        sys.executable,
        "-m",
        "hrt_chip",
        "benchmark-sweep",
        "--evaluator",
        "official",
        "--sampler-backend",
        "pytorch_checkpoint",
        "--checkpoint",
        str(checkpoint),
        "--mixed-size-backend",
        str(competition_profile["mixed_size_backend"]),
        "--selection-policy",
        str(competition_profile["selection_policy"]),
        "--guidance-preset",
        str(competition_profile["guidance_preset"]),
        "--diffusion-steps",
        str(competition_profile["diffusion_steps"]),
        "--diffusion-inference-steps",
        str(diffusion_inference_steps),
        "--sampler-mode",
        sampler_mode,
        "--seed",
        str(competition_profile["seed"]),
        "--candidates",
        str(competition_profile["candidates_per_guidance_vector"]),
        "--sweep-id",
        sweep_id,
        "--output-dir",
        str(output_dir),
    ]
    for bid in benchmark_ids:
        cmd += ["--benchmark", bid]

    _run(args=cmd, cwd=repo_root)
    report_path = _sweep_run_root(output_dir, sweep_id) / "sweep_report.json"
    return _load_json(report_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incumbent-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/day4_5"))
    parser.add_argument("--runtime-multiplier", type=float, default=1.2)
    parser.add_argument(
        "--min-mean-proxy-delta",
        type=float,
        default=0.0,
        help="Require mean_proxy < incumbent_mean_proxy - min-mean-proxy-delta.",
    )
    parser.add_argument("--candidate-finalists", type=int, default=2)
    parser.add_argument(
        "--replay-verify",
        action="store_true",
        help="Optional: replay-verify the best finalist manifest for one benchmark.",
    )
    parser.add_argument(
        "--replay-verify-benchmark",
        type=str,
        default="ibm01",
        help="Benchmark id used when doing --replay-verify.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    incumbent = _load_json(args.incumbent_json)

    competition_profile = incumbent["competition_profile"]
    benchmark_ids = _pick_dev_benchmarks_from_incumbent(incumbent)

    incumbent_dev_report_path = Path(incumbent["evidence"]["dev"]["sweep_report_path"])
    incumbent_dev_report = _load_json(incumbent_dev_report_path)
    incumbent_mean_proxy = incumbent_dev_report.get("mean_proxy")
    incumbent_mean_runtime = incumbent_dev_report.get("mean_runtime_seconds")
    incumbent_gate_a = incumbent_dev_report.get("gate_a_legal_all")

    # Compact matrix (E1-E4) from docs/experiment-matrix.md.
    # Values are ordered by increasing expected compute cost.
    matrix: list[tuple[str, str, int]] = [
        ("E1", "ddpm_subsampled", 25),
        ("E2", "ddpm_subsampled", 50),
        ("E3", "ddpm_subsampled", 100),
        ("E4", "ddim", 50),
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    finalists: list[dict[str, Any]] = []
    per_exp: list[dict[str, Any]] = []

    for exp_id, sampler_mode, inf_steps in matrix:
        sweep_id = f"{incumbent['incumbent_id']}__day4_5__{exp_id}"
        report = _run_matrix_experiment(
            repo_root=repo_root,
            output_dir=args.output_dir,
            sweep_id=sweep_id,
            benchmark_ids=benchmark_ids,
            competition_profile=competition_profile,
            sampler_mode=sampler_mode,
            diffusion_inference_steps=inf_steps,
        )

        gate_a = report.get("gate_a_legal_all")
        mean_proxy = report.get("mean_proxy")
        mean_runtime = report.get("mean_runtime_seconds")
        passes_gate = bool(gate_a) is True

        # Beat incumbent.
        beat_incumbent = True
        if incumbent_mean_proxy is not None and mean_proxy is not None:
            beat_incumbent = float(mean_proxy) < float(incumbent_mean_proxy) - float(args.min_mean_proxy_delta)

        # Runtime sanity.
        runtime_ok = True
        if incumbent_mean_runtime is not None and mean_runtime is not None:
            runtime_ok = float(mean_runtime) <= float(incumbent_mean_runtime) * float(args.runtime_multiplier)

        exp_summary = {
            "exp_id": exp_id,
            "sampler_mode": sampler_mode,
            "diffusion_inference_steps": inf_steps,
            "sweep_id": sweep_id,
            "gate_a_legal_all": gate_a,
            "mean_proxy": mean_proxy,
            "mean_runtime_seconds": mean_runtime,
            "passes_gate_a": passes_gate,
            "beats_incumbent": beat_incumbent,
            "runtime_ok": runtime_ok,
            "eligible_for_finalists": passes_gate and beat_incumbent and runtime_ok,
        }
        per_exp.append(exp_summary)

        if exp_summary["eligible_for_finalists"]:
            finalists.append(exp_summary)

    # Pick top-2 by best mean proxy.
    finalists_sorted = sorted(
        finalists,
        key=lambda r: float("inf") if r.get("mean_proxy") is None else float(r["mean_proxy"]),
    )
    top_finalists = finalists_sorted[: max(1, int(args.candidate_finalists))]

    out = {
        "incumbent_summary": {
            "incumbent_id": incumbent["incumbent_id"],
            "incumbent_mean_proxy": incumbent_mean_proxy,
            "incumbent_mean_runtime_seconds": incumbent_mean_runtime,
            "incumbent_gate_a_legal_all": incumbent_gate_a,
        },
        "dev_benchmark_ids": list(benchmark_ids),
        "experiments": per_exp,
        "finalists": top_finalists,
    }

    out_path = args.output_dir / f"{incumbent['incumbent_id']}__day4_5_finalists.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote Day 4-5 finalists: {out_path}")

    if args.replay_verify and top_finalists:
        # Replay-verify best finalist on one benchmark for stronger confidence.
        best = top_finalists[0]
        sweep_id = str(best["sweep_id"])
        best_report_path = _sweep_run_root(args.output_dir, sweep_id) / "sweep_report.json"
        best_report = _load_json(best_report_path)
        rows = best_report.get("rows") or []
        target_row = next((r for r in rows if r.get("benchmark_id") == args.replay_verify_benchmark), None)
        if target_row is None:
            raise ValueError(f"best finalist sweep missing benchmark {args.replay_verify_benchmark}")
        run_id = target_row.get("run_id")
        if not run_id:
            raise ValueError("best finalist target row missing run_id")
        manifest = _sweep_run_root(args.output_dir, sweep_id) / args.replay_verify_benchmark / str(run_id) / "manifest.json"
        cmd = [sys.executable, "-m", "hrt_chip", "replay", str(manifest), "--verify"]
        _run(args=cmd, cwd=repo_root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

