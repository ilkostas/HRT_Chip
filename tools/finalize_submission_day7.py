#!/usr/bin/env python3
"""
Day 7 helper: verify final go/no-go conditions and assemble a compact evidence pack.

Inputs:
  - Day 6 lock JSON produced by tools/run_full17_promotion_day6.py

Outputs:
  - evidence pack directory containing:
      sweep_report.json (winner)
      manifest.json + results.json + best-candidate JSON (winner)
      replay_verification.json (winner)
      (optional) backup equivalents
      evidence_manifest.json (index of copied artifacts)
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_if_exists(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(f"Missing required file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return _load_json(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--day6-lock-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/day7"))
    parser.add_argument("--copy-backup", action="store_true")
    args = parser.parse_args()

    lock = _load_json(args.day6_lock_json)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    winner = lock.get("winner")
    if not isinstance(winner, dict):
        raise ValueError("Day 6 lock JSON missing winner object")

    if winner.get("gate_a_legal_all") is not True:
        raise ValueError("Winner does not satisfy Gate A (17/17 legal) on FULL17")

    inc_id = str(lock.get("incumbent_id") or "unknown_incumbent")
    pack_root = output_dir / f"{inc_id}__evidence_pack"
    if pack_root.exists():
        shutil.rmtree(pack_root)
    pack_root.mkdir(parents=True)

    replay_verification = lock.get("replay_verification") or {}
    winner_rv = replay_verification.get("winner") or {}
    rv_ok = None
    # replay_verification.json comes from replay --verify; it uses `ok`.
    if isinstance(winner_rv.get("replay_verification"), dict):
        rv_ok = winner_rv["replay_verification"].get("ok")

    if rv_ok is not True:
        raise ValueError("Replay verification for the winner is not PASS (expected ok=true)")

    winner_manifest_path = Path(winner_rv.get("manifest_path"))
    if not winner_manifest_path.is_file():
        raise FileNotFoundError(f"Winner manifest missing: {winner_manifest_path}")

    winner_results_path = winner_manifest_path.parent / "results.json"
    winner_sweep_id = str(winner.get("sweep_id"))
    winner_sweep_root = Path(lock.get("full17_results")[0]["report_path"]).parents[1]  # best-effort; overwritten below

    # Locate winner sweep_root based on winner sweep id and report_path in full17_results.
    sweep_report_path = None
    for r in lock.get("full17_results") or []:
        if isinstance(r, dict) and str(r.get("sweep_id")) == winner_sweep_id:
            sweep_report_path = Path(r["report_path"])
            break
    if sweep_report_path is None:
        raise ValueError("Could not locate winner sweep_report.json path in full17_results")
    winner_sweep_root = sweep_report_path.parent

    # Copy core evidence.
    _copy_if_exists(sweep_report_path, pack_root / "winner" / "sweep_report.json")
    _copy_if_exists(winner_manifest_path, pack_root / "winner" / "manifest.json")
    _copy_if_exists(winner_results_path, pack_root / "winner" / "results.json")

    winner_best_candidate_id = None
    winner_results = _load_json(winner_results_path)
    winner_best_candidate_id = winner_results.get("best_candidate_id")
    if not winner_best_candidate_id:
        raise ValueError("Winner results.json missing best_candidate_id")

    winner_candidate_path = winner_manifest_path.parent / "candidates" / f"{winner_best_candidate_id}.json"
    _copy_if_exists(winner_candidate_path, pack_root / "winner" / "candidates" / f"{winner_best_candidate_id}.json")

    # Replay verification file.
    winner_rv_path = winner_manifest_path.parent / "replay_verification.json"
    _copy_if_exists(winner_rv_path, pack_root / "winner" / "replay_verification.json")

    if args.copy_backup:
        backup = lock.get("backup") or None
        if isinstance(backup, dict):
            backup_manifest_path = None
            if isinstance(replay_verification.get("backup"), dict):
                if "manifest_path" in replay_verification["backup"]:
                    backup_manifest_path = Path(replay_verification["backup"]["manifest_path"])
            if backup_manifest_path is not None and backup_manifest_path.is_file():
                _copy_if_exists(backup_manifest_path.parent / "results.json", pack_root / "backup" / "results.json")
                _copy_if_exists(backup_manifest_path, pack_root / "backup" / "manifest.json")
                _copy_if_exists(backup_manifest_path.parent / "replay_verification.json", pack_root / "backup" / "replay_verification.json")

    # Write an evidence index file.
    evidence_manifest = {
        "incumbent_id": inc_id,
        "pack_root": str(pack_root.resolve()),
        "winner": {
            "sweep_id": winner_sweep_id,
            "best_candidate_id": winner_best_candidate_id,
            "evidence": {
                "sweep_report.json": str((pack_root / "winner" / "sweep_report.json").resolve()),
                "manifest.json": str((pack_root / "winner" / "manifest.json").resolve()),
                "results.json": str((pack_root / "winner" / "results.json").resolve()),
                "candidate_json": str(
                    (pack_root / "winner" / "candidates" / f"{winner_best_candidate_id}.json").resolve()
                ),
                "replay_verification.json": str((pack_root / "winner" / "replay_verification.json").resolve()),
            },
        },
        "go_no_go": {
            "gate_a_legal_all": winner.get("gate_a_legal_all"),
            "replay_verification_ok": rv_ok,
        },
    }
    (pack_root / "evidence_manifest.json").write_text(
        json.dumps(evidence_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(f"Wrote evidence pack: {pack_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

