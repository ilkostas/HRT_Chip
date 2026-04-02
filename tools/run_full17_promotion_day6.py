#!/usr/bin/env python3
"""
Day 6 helper: run FULL17 promotion sweeps for Day 4-5 finalists,
pick the winner by lowest mean proxy among Gate-A-legal configs,
and replay-verify the winner/backup on one benchmark.
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


def _sweep_root(output_dir: Path, sweep_id: str) -> Path:
    return output_dir / sweep_id


def _find_manifest_for_benchmark(sweep_root: Path, *, benchmark_id: str, run_id: str) -> Path:
    manifest = sweep_root / benchmark_id / run_id / "manifest.json"
    if manifest.is_file():
        return manifest
    raise FileNotFoundError(f"manifest.json not found for {benchmark_id} run_id={run_id} under {sweep_root}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incumbent-json", type=Path, required=True)
    parser.add_argument("--day4-5-finalists-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/day6"))
    parser.add_argument("--replay-verify-benchmark", type=str, default="ibm01")
    parser.add_argument("--candidate-backup-count", type=int, default=2)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    incumbent = _load_json(args.incumbent_json)
    competition_profile = incumbent["competition_profile"]
    finalists_payload = _load_json(args.day4_5_finalists_json)
    finalists = finalists_payload.get("finalists") or []

    checkpoint = Path(competition_profile["checkpoint_path"])
    guidance_preset = str(competition_profile["guidance_preset"])
    mixed_size_backend = str(competition_profile["mixed_size_backend"])
    selection_policy = str(competition_profile["selection_policy"])
    diffusion_steps = str(competition_profile["diffusion_steps"])
    seed = str(competition_profile["seed"])
    candidates = str(competition_profile["candidates_per_guidance_vector"])
    deterministic = bool(competition_profile.get("deterministic", True))

    if not finalists:
        raise ValueError("No Day 4-5 finalists provided")

    full17_results: list[dict[str, Any]] = []
    for f in finalists:
        exp_id = str(f["exp_id"])
        sampler_mode = str(f["sampler_mode"])
        inf_steps = int(f["diffusion_inference_steps"])
        sweep_id = f"{incumbent['incumbent_id']}__day6__{exp_id}"

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
            mixed_size_backend,
            "--selection-policy",
            selection_policy,
            "--guidance-preset",
            guidance_preset,
            "--diffusion-steps",
            diffusion_steps,
            "--diffusion-inference-steps",
            str(inf_steps),
            "--sampler-mode",
            sampler_mode,
            "--seed",
            seed,
            "--candidates",
            candidates,
            "--sweep-id",
            sweep_id,
            "--output-dir",
            str(args.output_dir),
        ]
        # FULL17 is the default when --benchmark is not specified.
        _run(args=cmd, cwd=repo_root)
        report = _load_json(_sweep_root(args.output_dir, sweep_id) / "sweep_report.json")
        full17_results.append(
            {
                "exp_id": exp_id,
                "sampler_mode": sampler_mode,
                "diffusion_inference_steps": inf_steps,
                "sweep_id": sweep_id,
                "gate_a_legal_all": report.get("gate_a_legal_all"),
                "mean_proxy": report.get("mean_proxy"),
                "mean_runtime_seconds": report.get("mean_runtime_seconds"),
                "report_path": str(_sweep_root(args.output_dir, sweep_id) / "sweep_report.json"),
            }
        )

    # Gate-A eligible finalists only.
    eligible = [r for r in full17_results if r.get("gate_a_legal_all") is True]
    if not eligible:
        raise ValueError("No Day 4-5 finalists satisfied Gate A (17/17 legal) on FULL17")

    eligible_sorted = sorted(
        eligible,
        key=lambda r: float("inf") if r.get("mean_proxy") is None else float(r["mean_proxy"]),
    )
    winner = eligible_sorted[0]
    backup = eligible_sorted[1] if len(eligible_sorted) > 1 else eligible_sorted[0]

    def replay_verify_for(exp: dict[str, Any], benchmark_id: str) -> dict[str, Any]:
        sweep_id = str(exp["sweep_id"])
        sweep_root = _sweep_root(args.output_dir, sweep_id)
        report = _load_json(sweep_root / "sweep_report.json")
        rows = report.get("rows") or []
        row = next((r for r in rows if r.get("benchmark_id") == benchmark_id), None)
        if row is None:
            raise ValueError(f"FULL17 sweep {sweep_id} missing benchmark {benchmark_id}")
        run_id = row.get("run_id")
        if not run_id:
            raise ValueError("row missing run_id")
        manifest = _find_manifest_for_benchmark(sweep_root, benchmark_id=benchmark_id, run_id=str(run_id))
        cmd = [sys.executable, "-m", "hrt_chip", "replay", str(manifest), "--verify"]
        _run(args=cmd, cwd=repo_root)
        rv_path = manifest.parent / "replay_verification.json"
        rv = _load_json(rv_path) if rv_path.is_file() else {"ok": None, "note": "missing replay_verification.json"}
        return {"benchmark_id": benchmark_id, "manifest_path": str(manifest), "replay_verification": rv}

    verification = {
        "winner": replay_verify_for(winner, args.replay_verify_benchmark),
        "backup": replay_verify_for(backup, args.replay_verify_benchmark)
        if backup is not winner
        else {"skipped": True},
    }

    out = {
        "incumbent_id": incumbent["incumbent_id"],
        "winner": winner,
        "backup": backup if backup is not winner else None,
        "full17_results": full17_results,
        "replay_verification": verification,
        "competition_profile_echo": {
            "checkpoint_path": str(checkpoint),
            "guidance_preset": guidance_preset,
            "mixed_size_backend": mixed_size_backend,
            "selection_policy": selection_policy,
            "diffusion_steps": diffusion_steps,
            "seed": seed,
            "candidates_per_guidance_vector": candidates,
            "deterministic": deterministic,
        },
    }
    out_path = args.output_dir / f"{incumbent['incumbent_id']}__day6_full17_finalist_lock.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote Day 6 finalist lock: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

