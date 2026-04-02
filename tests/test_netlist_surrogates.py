"""Netlist-aware surrogates vs mock Benchmark layout."""

from __future__ import annotations

import math

import torch

from hrt_chip.models import MacroRect
from hrt_chip.netlist_surrogates import (
    benchmark_has_netlist,
    benchmark_has_pin_groups,
    hpwl_exact_pin_surrogate,
    hpwl_logsumexp_surrogate,
    rudy_congestion_macro_surrogate,
    rudy_congestion_pin_surrogate,
    rudy_congestion_surrogate,
)


def _fake_benchmark(*, with_pins: bool) -> object:
    base: dict[str, object] = {
        "num_macros": 2,
        "macro_names": ["m0", "m1"],
        "macro_positions": torch.tensor([[0.0, 0.0], [10.0, 10.0]]),
        "macro_sizes": torch.tensor([[2.0, 2.0], [2.0, 2.0]]),
        "num_nets": 1,
        "net_nodes": [torch.tensor([0, 1])],
        "net_weights": torch.tensor([1.0]),
        "grid_rows": 8,
        "grid_cols": 8,
    }
    if with_pins:
        # Offsets relative to macro center; zeros => pins at centers.
        base["net_pin_dx"] = [torch.tensor([0.0, 0.0])]
        base["net_pin_dy"] = [torch.tensor([0.0, 0.0])]
    return type("B", (), base)()


def test_benchmark_has_netlist() -> None:
    b = _fake_benchmark(with_pins=False)
    assert benchmark_has_netlist(b) is True
    assert benchmark_has_netlist(None) is False


def test_benchmark_has_pin_groups() -> None:
    assert benchmark_has_pin_groups(_fake_benchmark(with_pins=True)) is True
    assert benchmark_has_pin_groups(_fake_benchmark(with_pins=False)) is False


def test_hpwl_rudy_finite_logsumexp_and_macro_rudy() -> None:
    b = _fake_benchmark(with_pins=False)
    macros = [
        MacroRect("m0", 0.0, 0.0, 2.0, 2.0),
        MacroRect("m1", 8.0, 8.0, 2.0, 2.0),
    ]
    hpwl = hpwl_logsumexp_surrogate(macros, b, canvas_w=20.0, canvas_h=20.0)
    rudy = rudy_congestion_macro_surrogate(macros, b, canvas_w=20.0, canvas_h=20.0)
    assert hpwl == hpwl and hpwl < 1e6
    assert rudy == rudy and rudy >= 0.0


def test_exact_pin_hpwl_zero_offsets_matches_geometry() -> None:
    """Centers (1,1) and (9,9) => HPWL half-perim = 8+8 = 16; norm canvas 40 => 0.4."""
    b = _fake_benchmark(with_pins=True)
    macros = [
        MacroRect("m0", 0.0, 0.0, 2.0, 2.0),
        MacroRect("m1", 8.0, 8.0, 2.0, 2.0),
    ]
    hpwl = hpwl_exact_pin_surrogate(macros, b, canvas_w=20.0, canvas_h=20.0)
    assert abs(hpwl - 0.4) < 1e-6


def test_exact_pin_hpwl_with_offsets() -> None:
    b = type(
        "B",
        (),
        {
            "num_macros": 2,
            "macro_names": ["m0", "m1"],
            "macro_positions": torch.tensor([[0.0, 0.0], [10.0, 10.0]]),
            "macro_sizes": torch.tensor([[2.0, 2.0], [2.0, 2.0]]),
            "num_nets": 1,
            "net_nodes": [torch.tensor([0, 1])],
            "net_weights": torch.tensor([1.0]),
            "grid_rows": 8,
            "grid_cols": 8,
            "net_pin_dx": [torch.tensor([1.0, -1.0])],
            "net_pin_dy": [torch.tensor([0.0, 0.0])],
        },
    )()
    macros = [
        MacroRect("m0", 0.0, 0.0, 2.0, 2.0),
        MacroRect("m1", 8.0, 8.0, 2.0, 2.0),
    ]
    # m0 center 1,1 + (1,0) => 2,1 ; m1 center 9,9 + (-1,0) => 8,9 => HPWL = 6+8 = 14
    hpwl = hpwl_exact_pin_surrogate(macros, b, canvas_w=20.0, canvas_h=20.0)
    assert abs(hpwl - 14.0 / 40.0) < 1e-6


def test_pin_rudy_finite() -> None:
    b = _fake_benchmark(with_pins=True)
    macros = [
        MacroRect("m0", 0.0, 0.0, 2.0, 2.0),
        MacroRect("m1", 8.0, 8.0, 2.0, 2.0),
    ]
    r = rudy_congestion_pin_surrogate(macros, b, canvas_w=20.0, canvas_h=20.0)
    assert r == r and r >= 0.0


def test_rudy_dispatcher_uses_pins_when_present() -> None:
    b = _fake_benchmark(with_pins=True)
    macros = [
        MacroRect("m0", 0.0, 0.0, 2.0, 2.0),
        MacroRect("m1", 8.0, 8.0, 2.0, 2.0),
    ]
    r_pin = rudy_congestion_surrogate(macros, b, canvas_w=20.0, canvas_h=20.0)
    r_macro = rudy_congestion_macro_surrogate(macros, b, canvas_w=20.0, canvas_h=20.0)
    assert r_pin == r_pin
    # Pin and macro RUDY can differ when offsets are zero; here they should match.
    assert abs(r_pin - r_macro) < 1e-9


def test_exact_hpwl_nan_without_pins() -> None:
    b = _fake_benchmark(with_pins=False)
    macros = [
        MacroRect("m0", 0.0, 0.0, 2.0, 2.0),
        MacroRect("m1", 8.0, 8.0, 2.0, 2.0),
    ]
    assert math.isnan(hpwl_exact_pin_surrogate(macros, b, canvas_w=20.0, canvas_h=20.0))
