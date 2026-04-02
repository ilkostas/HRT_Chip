#!/usr/bin/env python3
"""CI helper: run a minimal stub benchmark sweep and assert Gate A (100% legal)."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        sweep_root = tdir / "sw"
        trends = tdir / "trends.jsonl"
        cmd = [
            sys.executable,
            "-m",
            "hrt_chip",
            "benchmark-sweep",
            "--evaluator",
            "stub",
            "--benchmark",
            "ibm01",
            "--benchmark",
            "ibm02",
            "--candidates",
            "1",
            "--output-dir",
            str(sweep_root),
            "--sweep-id",
            "ci_regression",
            "--mixed-size-backend",
            "stub",
            "--trends-log-path",
            str(trends),
        ]
        subprocess.check_call(cmd, cwd=repo_root)
        report_path = sweep_root / "ci_regression" / "sweep_report.json"
        data = json.loads(report_path.read_text(encoding="utf-8"))
        if not data.get("gate_a_legal_all"):
            print("FAIL: Gate A not satisfied", file=sys.stderr)
            print(json.dumps(data, indent=2), file=sys.stderr)
            return 1
        if not trends.is_file():
            print("FAIL: trends log missing", file=sys.stderr)
            return 1
    print("OK: stub sweep Gate A + trends log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
