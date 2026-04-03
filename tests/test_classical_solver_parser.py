from __future__ import annotations

from pathlib import Path

from submissions.force_directed_sa.checkpoint_ibm01 import parse_plc_reference_metrics
from submissions.force_directed_sa.parser import load_benchmark
from submissions.force_directed_sa.surrogate import compute_surrogate


def _root() -> Path:
    return Path("external") / "MacroPlacement" / "Testcases" / "ICCAD04"


def test_parse_all_17_ibm_benchmarks() -> None:
    root = _root()
    benches = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("ibm"))
    assert len(benches) == 17
    for bench in benches:
        b = load_benchmark(root / bench)
        assert b.grid_cols > 0 and b.grid_rows > 0
        assert b.canvas_width > 0.0 and b.canvas_height > 0.0
        assert len(b.nodes) > 0
        assert len(b.nets) > 0
        assert len(b.movable_indices) > 0


def test_ibm01_initial_surrogate_scales_not_grossly_wrong() -> None:
    root = _root()
    b = load_benchmark(root / "ibm01")
    s = compute_surrogate(b)
    refs = parse_plc_reference_metrics(root / "ibm01" / "initial.plc")
    assert s.hpwl > 0.0 and s.density >= 0.0 and s.congestion >= 0.0
    wl_ref = refs.get("wirelength_cost", 0.0)
    assert wl_ref > 0.0
    ratio = s.hpwl / wl_ref
    # Conservative bound: catches obvious parser/normalization bugs without enforcing exact formula matching.
    assert 0.2 <= ratio <= 5.0

