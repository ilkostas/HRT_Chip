#!/usr/bin/env python3
"""Compare two sweep_report.json files (mean proxy and per-row deltas)."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: ab_compare_runs.py sweep_report_a.json sweep_report_b.json", file=sys.stderr)
        return 2
    a = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    b = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    ma = a.get("mean_proxy")
    mb = b.get("mean_proxy")
    print(f"mean_proxy A={ma} B={mb}")
    rows_a = {r["benchmark_id"]: r for r in (a.get("rows") or [])}
    rows_b = {r["benchmark_id"]: r for r in (b.get("rows") or [])}
    for bid in sorted(set(rows_a) | set(rows_b)):
        pa = rows_a.get(bid, {}).get("proxy_score")
        pb = rows_b.get(bid, {}).get("proxy_score")
        print(f"  {bid}: A={pa} B={pb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
