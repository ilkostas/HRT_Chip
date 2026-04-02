"""Append-only sweep history for Gate A/B/C regression tracking."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from hrt_chip.io.artifacts import utc_now_iso


def default_trends_path() -> Path:
    return Path("runs/trends/sweep_history.jsonl")


def append_sweep_trend(
    report_dict: dict[str, Any],
    *,
    log_path: Path | str | None = None,
) -> Path:
    """
    Append one JSON line with sweep summary (gates, mean proxy, evaluator backend).

    Safe to call from multiple processes only with external file locking (not provided).
    """
    path = Path(log_path) if log_path else default_trends_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "recorded_at_utc": utc_now_iso(),
        **report_dict,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def load_recent_trends(log_path: Path | str | None = None, *, limit: int = 50) -> list[dict[str, Any]]:
    path = Path(log_path) if log_path else default_trends_path()
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    out: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def summarize_trends(
    rows: list[dict[str, Any]],
    *,
    baseline_sweep_id: str | None = None,
) -> dict[str, Any]:
    """
    Build aggregate stats for operator review: gate pass rates over recent rows,
    optional mean_proxy delta vs a baseline sweep id.
    """
    if not rows:
        return {"n_rows": 0, "message": "no trend rows"}

    base_proxy: float | None = None
    if baseline_sweep_id:
        for r in rows:
            if r.get("sweep_id") == baseline_sweep_id:
                mp = r.get("mean_proxy")
                if isinstance(mp, (int, float)) and math.isfinite(float(mp)):
                    base_proxy = float(mp)
                break

    gate_a_rate = sum(1 for r in rows if r.get("gate_a_legal_all")) / len(rows)
    gate_b_rate = sum(1 for r in rows if r.get("gate_b_beat_sa_aggregate")) / len(rows)
    gate_c_rate = sum(1 for r in rows if r.get("gate_c_beat_replace_aggregate")) / len(rows)

    last = rows[-1]
    last_proxy = last.get("mean_proxy")
    delta: float | None = None
    if base_proxy is not None and isinstance(last_proxy, (int, float)) and math.isfinite(float(last_proxy)):
        delta = float(last_proxy) - base_proxy

    return {
        "n_rows": len(rows),
        "baseline_sweep_id": baseline_sweep_id,
        "baseline_mean_proxy": base_proxy,
        "gate_a_pass_rate": gate_a_rate,
        "gate_b_pass_rate": gate_b_rate,
        "gate_c_pass_rate": gate_c_rate,
        "last_sweep_id": last.get("sweep_id"),
        "last_mean_proxy": last_proxy,
        "last_mean_proxy_delta_vs_baseline": delta,
        "last_total_runtime_seconds": last.get("total_runtime_seconds"),
    }
