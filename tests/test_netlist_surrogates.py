"""Netlist-aware surrogates vs mock Benchmark layout."""

from __future__ import annotations

import torch

from hrt_chip.models import MacroRect
from hrt_chip.netlist_surrogates import (
    benchmark_has_netlist,
    hpwl_logsumexp_surrogate,
    rudy_congestion_surrogate,
)


def _fake_benchmark() -> object:
    return type(
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
        },
    )()


def test_benchmark_has_netlist() -> None:
    b = _fake_benchmark()
    assert benchmark_has_netlist(b) is True
    assert benchmark_has_netlist(None) is False


def test_hpwl_rudy_finite() -> None:
    b = _fake_benchmark()
    macros = [
        MacroRect("m0", 0.0, 0.0, 2.0, 2.0),
        MacroRect("m1", 8.0, 8.0, 2.0, 2.0),
    ]
    hpwl = hpwl_logsumexp_surrogate(macros, b, canvas_w=20.0, canvas_h=20.0)
    rudy = rudy_congestion_surrogate(macros, b, canvas_w=20.0, canvas_h=20.0)
    assert hpwl == hpwl and hpwl < 1e6
    assert rudy == rudy and rudy >= 0.0
