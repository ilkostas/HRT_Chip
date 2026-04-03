from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from submissions.force_directed_sa.parser import load_benchmark
from submissions.force_directed_sa.placer import SolverConfig, run_solver, write_plc


def _benchmarks(testcases_root: Path) -> list[str]:
    found = [p.name for p in testcases_root.iterdir() if p.is_dir() and p.name.startswith("ibm")]
    return sorted(found)


def run_sweep(testcases_root: Path, output_root: Path, log_path: Path, seed: int = 12345) -> dict[str, object]:
    cfg = SolverConfig()
    cfg.seed = seed
    cfg.sa.seed = seed
    results: dict[str, dict[str, float | str | None]] = {}
    total = 0.0
    benches = _benchmarks(testcases_root)
    for bench in benches:
        bench_dir = testcases_root / bench
        solved, score = run_solver(bench_dir, cfg)
        out_plc = output_root / f"{bench}.plc"
        write_plc(solved, bench_dir / "initial.plc", out_plc)
        results[bench] = {
            "surrogate_total": score.total,
            "surrogate_hpwl": score.hpwl,
            "surrogate_density": score.density,
            "surrogate_congestion": score.congestion,
            "official_proxy": None,
            "output_plc": str(out_plc),
        }
        total += score.total
    avg = total / max(1, len(benches))
    payload: dict[str, object] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parameters": {"seed": seed},
        "results": results,
        "average_surrogate": avg,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        (log_path.read_text(encoding="utf-8") if log_path.exists() else "")
        + json.dumps(payload, ensure_ascii=True)
        + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> int:
    p = argparse.ArgumentParser(description="Run force-directed+SA sweep and append experiments.jsonl")
    p.add_argument(
        "--testcases-root",
        default=str(Path("external") / "MacroPlacement" / "Testcases" / "ICCAD04"),
    )
    p.add_argument("--output-root", default=str(Path("runs") / "classical"))
    p.add_argument("--log-path", default=str(Path("submissions") / "force_directed_sa" / "experiments.jsonl"))
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args()
    payload = run_sweep(Path(args.testcases_root), Path(args.output_root), Path(args.log_path), seed=args.seed)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

