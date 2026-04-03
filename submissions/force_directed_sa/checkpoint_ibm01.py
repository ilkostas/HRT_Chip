from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from submissions.force_directed_sa.parser import load_benchmark
from submissions.force_directed_sa.surrogate import compute_surrogate

_FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def parse_plc_reference_metrics(plc_path: Path) -> dict[str, float]:
    refs: dict[str, float] = {}
    for line in plc_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# Wirelength cost"):
            refs["wirelength_cost"] = float(_FLOAT_RE.findall(line)[0])
        elif line.startswith("# Congestion cost"):
            refs["congestion_cost"] = float(_FLOAT_RE.findall(line)[0])
        elif line.startswith("# Density cost"):
            refs["density_cost"] = float(_FLOAT_RE.findall(line)[0])
        elif line.startswith("# Wirelength :"):
            refs["wirelength_raw"] = float(_FLOAT_RE.findall(line)[0])
    return refs


def run_checkpoint(testcases_root: Path) -> dict[str, object]:
    bench_dir = testcases_root / "ibm01"
    benchmark = load_benchmark(bench_dir)
    surrogate = compute_surrogate(benchmark)
    refs = parse_plc_reference_metrics(bench_dir / "initial.plc")
    report: dict[str, object] = {
        "benchmark": "ibm01",
        "surrogate": {
            "hpwl": surrogate.hpwl,
            "density": surrogate.density,
            "congestion": surrogate.congestion,
            "total": surrogate.total,
        },
        "reference_from_initial_plc_header": refs,
        "note": (
            "Use this immediate checkpoint before optimizer work. "
            "If component scales differ materially, fix parser/normalization first."
        ),
    }
    return report


def main() -> int:
    p = argparse.ArgumentParser(description="Immediate ibm01 surrogate-vs-reference checkpoint")
    p.add_argument(
        "--testcases-root",
        default=str(Path("external") / "MacroPlacement" / "Testcases" / "ICCAD04"),
    )
    args = p.parse_args()
    report = run_checkpoint(Path(args.testcases_root))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

