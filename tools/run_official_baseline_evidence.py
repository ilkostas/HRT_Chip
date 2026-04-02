#!/usr/bin/env python3
"""
Day 2 helper: run official-evaluator baseline sweeps (smoke + dev) and write an
`incumbent.json` evidence record that points to manifests/results.

This script does not require the competition assets until you actually choose
`--evaluator official` (it relies on `hrt-chip benchmark-sweep`).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _run_benchmark_sweep(*, sweep_args: list[str], cwd: Path) -> None:
    cmd = [sys.executable, "-m", "hrt_chip", "benchmark-sweep", *sweep_args]
    subprocess.check_call(cmd, cwd=cwd)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_manifest_for_benchmark(sweep_root: Path, *, benchmark_id: str, row_run_id: str) -> Path:
    run_dir = sweep_root / benchmark_id / row_run_id
    manifest = run_dir / "manifest.json"
    if manifest.is_file():
        return manifest
    # Fallback: glob (in case the run_id folder naming differs).
    matches = list((sweep_root / benchmark_id).rglob("manifest.json"))
    if not matches:
        raise FileNotFoundError(f"manifest.json not found for {benchmark_id} under {sweep_root}")
    # Pick the one closest to expected run_id.
    matches.sort(key=lambda p: len(p.as_posix()))
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("runs/baselines"))
    parser.add_argument("--sweep-id", type=str, default="day2_official_baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidates", type=int, default=2)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--diffusion-inference-steps", type=int, default=50)
    parser.add_argument("--sampler-mode", type=str, default="ddpm_subsampled")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mixed-size-backend", type=str, default="estimate")
    parser.add_argument("--selection-policy", type=str, default="proxy_first")
    parser.add_argument("--guidance-preset", type=str, default="pareto3")
    parser.add_argument("--deterministic", action="store_true")

    parser.add_argument(
        "--smoke-benchmarks",
        type=str,
        default="ibm01,ibm06",
        help="Comma-separated list of benchmark ids for the smoke sweep.",
    )
    parser.add_argument(
        "--dev-benchmarks",
        type=str,
        default="ibm01,ibm03,ibm06,ibm12,ibm17",
        help="Comma-separated list of benchmark ids for the dev sweep.",
    )
    parser.add_argument(
        "--replay-verify-benchmark",
        type=str,
        default="ibm01",
        help="Which benchmark id to replay --verify for (one is enough for Day 2 evidence).",
    )
    parser.add_argument("--replay-verify", action="store_true", help="Run hrt-chip replay --verify for one manifest.")

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    smoke_ids = tuple(x.strip() for x in args.smoke_benchmarks.split(",") if x.strip())
    dev_ids = tuple(x.strip() for x in args.dev_benchmarks.split(",") if x.strip())

    # Sweep 1: smoke
    smoke_sweep_id = f"{args.sweep_id}__smoke"
    smoke_root_args = [
        "--evaluator",
        "official",
        "--sampler-backend",
        "pytorch_checkpoint",
        "--checkpoint",
        str(args.checkpoint),
        "--mixed-size-backend",
        args.mixed_size_backend,
        "--selection-policy",
        args.selection_policy,
        "--guidance-preset",
        args.guidance_preset,
        "--diffusion-steps",
        str(args.diffusion_steps),
        "--diffusion-inference-steps",
        str(args.diffusion_inference_steps),
        "--sampler-mode",
        args.sampler_mode,
        "--seed",
        str(args.seed),
        "--candidates",
        str(args.candidates),
        "--sweep-id",
        smoke_sweep_id,
        "--output-dir",
        str(args.output_dir),
    ]
    for bid in smoke_ids:
        smoke_root_args += ["--benchmark", bid]

    _run_benchmark_sweep(sweep_args=smoke_root_args, cwd=repo_root)
    smoke_root = args.output_dir / smoke_sweep_id
    smoke_report = _load_json(smoke_root / "sweep_report.json")

    # Sweep 2: dev
    dev_sweep_id = f"{args.sweep_id}__dev"
    dev_root_args = [
        "--evaluator",
        "official",
        "--sampler-backend",
        "pytorch_checkpoint",
        "--checkpoint",
        str(args.checkpoint),
        "--mixed-size-backend",
        args.mixed_size_backend,
        "--selection-policy",
        args.selection_policy,
        "--guidance-preset",
        args.guidance_preset,
        "--diffusion-steps",
        str(args.diffusion_steps),
        "--diffusion-inference-steps",
        str(args.diffusion_inference_steps),
        "--sampler-mode",
        args.sampler_mode,
        "--seed",
        str(args.seed),
        "--candidates",
        str(args.candidates),
        "--sweep-id",
        dev_sweep_id,
        "--output-dir",
        str(args.output_dir),
    ]
    for bid in dev_ids:
        dev_root_args += ["--benchmark", bid]

    _run_benchmark_sweep(sweep_args=dev_root_args, cwd=repo_root)
    dev_root = args.output_dir / dev_sweep_id
    dev_report = _load_json(dev_root / "sweep_report.json")

    # Pull manifest paths for evidence.
    dev_rows = dev_report.get("rows") or []
    replay_bid = args.replay_verify_benchmark
    replay_row = next((r for r in dev_rows if r.get("benchmark_id") == replay_bid), None)
    if replay_row is None:
        raise ValueError(f"replay benchmark {replay_bid!r} not found in dev sweep rows")
    replay_run_id = replay_row.get("run_id")
    if not replay_run_id:
        raise ValueError(f"replay benchmark {replay_bid!r} row missing run_id")
    replay_manifest = _find_manifest_for_benchmark(
        dev_root,
        benchmark_id=replay_bid,
        row_run_id=str(replay_run_id),
    )

    if args.replay_verify:
        cmd = [sys.executable, "-m", "hrt_chip", "replay", str(replay_manifest), "--verify"]
        subprocess.check_call(cmd, cwd=repo_root)

    incumbent = {
        "incumbent_id": args.sweep_id,
        "competition_profile": {
            "evaluator_backend": "official",
            "sampler_backend": "pytorch_checkpoint",
            "checkpoint_path": str(args.checkpoint),
            "guidance_preset": args.guidance_preset,
            "selection_policy": args.selection_policy,
            "mixed_size_backend": args.mixed_size_backend,
            "sampler_mode": args.sampler_mode,
            "diffusion_steps": args.diffusion_steps,
            "diffusion_inference_steps": args.diffusion_inference_steps,
            "seed": args.seed,
            "candidates_per_guidance_vector": args.candidates,
            "deterministic": bool(args.deterministic),
        },
        "evidence": {
            "smoke": {
                "sweep_id": smoke_sweep_id,
                "sweep_report_path": str(smoke_root / "sweep_report.json"),
            },
            "dev": {
                "sweep_id": dev_sweep_id,
                "sweep_report_path": str(dev_root / "sweep_report.json"),
            },
            "replay_verify": {
                "benchmark_id": replay_bid,
                "manifest_path": str(replay_manifest),
                "replay_verification_path": str(replay_manifest.parent / "replay_verification.json"),
            },
        },
        "dev_gate_summary": {
            "gate_a_legal_all": dev_report.get("gate_a_legal_all"),
            "gate_b_beat_sa_aggregate": dev_report.get("gate_b_beat_sa_aggregate"),
            "gate_c_beat_replace_aggregate": dev_report.get("gate_c_beat_replace_aggregate"),
            "mean_proxy": dev_report.get("mean_proxy"),
            "legal_count": dev_report.get("legal_count"),
            "total_count": dev_report.get("total_count"),
        },
    }

    incumbent_path = args.output_dir / f"{args.sweep_id}__incumbent.json"
    incumbent_path.write_text(json.dumps(incumbent, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote incumbent evidence: {incumbent_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

