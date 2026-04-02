"""Load ICCAD04 benchmark geometry from the official MacroPlacement + macro_place stack (optional)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


def macro_place_available() -> bool:
    try:
        import macro_place  # noqa: F401

        return True
    except ImportError:
        return False


def load_benchmark_and_plc(benchmark_id: str, testcase_root: Path) -> tuple[Any, Any]:
    """
    Load (Benchmark, PlacementCost) via official ``macro_place.loader``.

    Raises:
        FileNotFoundError: testcase directory or netlist missing.
        ImportError: macro_place not installed.
    """
    from macro_place.loader import load_benchmark_from_dir

    bench_dir = testcase_root / benchmark_id
    if not bench_dir.is_dir():
        raise FileNotFoundError(
            f"Benchmark directory not found: {bench_dir}. "
            "Initialize submodules: git submodule update --init external/MacroPlacement"
        )
    return load_benchmark_from_dir(str(bench_dir))


def load_macro_specs_and_canvas(
    benchmark_id: str,
    testcase_root: Path,
) -> tuple[tuple[tuple[str, float, float], ...], float, float]:
    """
    Return ``((name, w, h), ...)`` in microns and canvas (width, height) for pipeline generation/legalization.

    Raises on missing deps or testcases (same as ``load_benchmark_and_plc``).
    """
    benchmark, _plc = load_benchmark_and_plc(benchmark_id, testcase_root)
    specs: list[tuple[str, float, float]] = []
    for i in range(benchmark.num_macros):
        specs.append(
            (
                benchmark.macro_names[i],
                float(benchmark.macro_sizes[i, 0].item()),
                float(benchmark.macro_sizes[i, 1].item()),
            )
        )
    return (
        tuple(specs),
        float(benchmark.canvas_width),
        float(benchmark.canvas_height),
    )


def load_full_benchmark(
    benchmark_id: str,
    testcase_root: Path,
) -> tuple[Any, Any, tuple[tuple[str, float, float], ...], float, float]:
    """
    Single load for pipeline + evaluator: (benchmark, plc, macro_specs, canvas_w, canvas_h).
    """
    benchmark, plc = load_benchmark_and_plc(benchmark_id, testcase_root)
    specs: list[tuple[str, float, float]] = []
    for i in range(benchmark.num_macros):
        specs.append(
            (
                benchmark.macro_names[i],
                float(benchmark.macro_sizes[i, 0].item()),
                float(benchmark.macro_sizes[i, 1].item()),
            )
        )
    return (
        benchmark,
        plc,
        tuple(specs),
        float(benchmark.canvas_width),
        float(benchmark.canvas_height),
    )


def restore_fixed_macro_positions(candidate: Any, benchmark: Any) -> None:
    """
    After legalization, reset fixed macros to initial benchmark centers (lower-left rects).

    Mutates ``candidate.macros`` in place.
    """
    from hrt_chip.models import MacroRect

    by_name = {m.name: m for m in candidate.macros}
    for i in range(benchmark.num_macros):
        fix = benchmark.macro_fixed[i]
        fixed_val = fix.item() if hasattr(fix, "item") else bool(fix)
        if not bool(fixed_val):
            continue
        name = benchmark.macro_names[i]
        w = float(benchmark.macro_sizes[i, 0].item())
        h = float(benchmark.macro_sizes[i, 1].item())
        cx = float(benchmark.macro_positions[i, 0].item())
        cy = float(benchmark.macro_positions[i, 1].item())
        x_ll = cx - w / 2.0
        y_ll = cy - h / 2.0
        m = by_name.get(name)
        if m is None:
            candidate.macros.append(MacroRect(name=name, x=x_ll, y=y_ll, w=w, h=h))
        else:
            m.x, m.y = x_ll, y_ll
            m.w, m.h = w, h
