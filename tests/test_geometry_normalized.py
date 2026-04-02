"""Unit tests for normalized center mapping."""

from __future__ import annotations

import pytest

from hrt_chip.geometry import normalized_center_to_lower_left


def test_normalized_center_maps_corners() -> None:
    # Center at (-1,-1) maps to lower-left corner of canvas for tiny macro
    x, y = normalized_center_to_lower_left(-1.0, -1.0, 0.0, 0.0)
    assert x == pytest.approx(0.0)
    assert y == pytest.approx(0.0)

    x, y = normalized_center_to_lower_left(1.0, 1.0, 0.0, 0.0, canvas_w=1.0, canvas_h=1.0)
    assert x == pytest.approx(1.0)
    assert y == pytest.approx(1.0)


def test_normalized_center_centers_macro() -> None:
    x, y = normalized_center_to_lower_left(0.0, 0.0, 0.2, 0.2)
    assert x == pytest.approx(0.4)
    assert y == pytest.approx(0.4)
