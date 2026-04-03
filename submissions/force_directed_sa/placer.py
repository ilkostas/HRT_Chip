from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from submissions.force_directed_sa.force_directed import ForceDirectedConfig, run_force_directed
from submissions.force_directed_sa.legalize import legalize
from submissions.force_directed_sa.orient import run_orientation_pass
from submissions.force_directed_sa.parser import load_benchmark
from submissions.force_directed_sa.sa_refine import SAConfig, run_sa_refinement
from submissions.force_directed_sa.surrogate import SurrogateBreakdown, compute_surrogate
from submissions.force_directed_sa.types import Benchmark


@dataclass
class SolverConfig:
    seed: int = 12345
    force: ForceDirectedConfig = field(default_factory=ForceDirectedConfig)
    sa: SAConfig = field(default_factory=SAConfig)
    orientation_enabled: bool = True


def run_solver(benchmark_dir: Path, config: SolverConfig | None = None) -> tuple[Benchmark, SurrogateBreakdown]:
    cfg = config or SolverConfig()
    benchmark = load_benchmark(benchmark_dir)
    run_force_directed(benchmark, cfg.force)
    legalize(benchmark)
    run_sa_refinement(benchmark, cfg.sa)
    run_orientation_pass(benchmark, enabled=cfg.orientation_enabled)
    score = compute_surrogate(benchmark)
    return benchmark, score


def write_plc(benchmark: Benchmark, template_plc: Path, out_plc: Path) -> None:
    lines = template_plc.read_text(encoding="utf-8").splitlines()
    updated: list[str] = []
    node_idx = 0
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5 and parts[0].isdigit():
            if node_idx >= len(benchmark.nodes):
                raise ValueError("PLS node rows exceed parsed benchmark nodes.")
            node = benchmark.nodes[node_idx]
            orient = node.orientation or "N"
            fixed = 1 if node.is_fixed else 0
            updated.append(f"{parts[0]} {node.x:.6f} {node.y:.6f} {orient} {fixed}")
            node_idx += 1
        else:
            updated.append(line)
    out_plc.parent.mkdir(parents=True, exist_ok=True)
    out_plc.write_text("\n".join(updated) + "\n", encoding="utf-8")


def _default_testcases_root() -> Path:
    return Path("external") / "MacroPlacement" / "Testcases" / "ICCAD04"


def run_one_cli(args: argparse.Namespace) -> int:
    root = Path(args.testcases_root)
    bench_dir = root / args.benchmark
    if not bench_dir.exists():
        raise FileNotFoundError(f"Benchmark directory not found: {bench_dir}")
    cfg = SolverConfig()
    cfg.seed = args.seed
    cfg.sa.seed = args.seed
    benchmark, score = run_solver(bench_dir, cfg)
    out_plc = Path(args.output) if args.output else Path("runs") / "classical" / f"{args.benchmark}.plc"
    write_plc(benchmark, bench_dir / "initial.plc", out_plc)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmark": args.benchmark,
        "score": asdict(score),
        "output_plc": str(out_plc),
    }
    print(json.dumps(payload, indent=2))
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Standalone force-directed + SA macro placer")
    p.add_argument("-b", "--benchmark", required=True, help="Benchmark ID (e.g. ibm01)")
    p.add_argument(
        "--testcases-root",
        default=str(_default_testcases_root()),
        help="Path containing ibmXX benchmark directories",
    )
    p.add_argument("--output", default="", help="Output PLC path (default runs/classical/<bench>.plc)")
    p.add_argument("--seed", type=int, default=12345, help="Random seed")
    return p


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    return run_one_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())

