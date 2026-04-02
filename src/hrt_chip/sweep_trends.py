"""Append-only sweep history for Gate A/B/C regression tracking."""

from __future__ import annotations

import json
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
