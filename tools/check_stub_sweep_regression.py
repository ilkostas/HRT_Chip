#!/usr/bin/env python3
"""CI helper: run a minimal stub benchmark sweep and assert Gate A + trend record quality."""

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
        lines = [ln for ln in trends.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            print("FAIL: trends log empty", file=sys.stderr)
            return 1
        last = json.loads(lines[-1])
        required = (
            "sweep_id",
            "gate_a_legal_all",
            "mean_proxy",
            "total_runtime_seconds",
            "recorded_at_utc",
        )
        for k in required:
            if k not in last:
                print(f"FAIL: trends line missing key {k!r}", file=sys.stderr)
                return 1
        if not last.get("gate_a_legal_all"):
            print("FAIL: trend line Gate A not true", file=sys.stderr)
            return 1
        mp = last.get("mean_proxy")
        if not isinstance(mp, (int, float)):
            print("FAIL: mean_proxy not numeric in trends", file=sys.stderr)
            return 1
        # Loose stub sanity: deterministic stub should stay in a stable band across runs.
        if float(mp) > 1e6:
            print(f"FAIL: mean_proxy absurdly large: {mp}", file=sys.stderr)
            return 1
        # Per-row timing from pipeline results
        rows = data.get("rows") or []
        if len(rows) < 2:
            print("FAIL: expected at least two benchmark rows", file=sys.stderr)
            return 1
        for row in rows:
            if row.get("error"):
                continue
            tim = row.get("timing")
            if not isinstance(tim, dict) or "generation_seconds" not in tim:
                print("FAIL: row missing timing.generation_seconds", file=sys.stderr)
                print(json.dumps(row, indent=2), file=sys.stderr)
                return 1
    print("OK: stub sweep Gate A + trends integrity + timing rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
