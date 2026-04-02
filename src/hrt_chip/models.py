"""Core data shapes for placement candidates (Phase 0 stubs)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MacroRect:
    """Axis-aligned rectangle for a macro instance (normalized stub coordinates)."""

    name: str
    x: float
    y: float
    w: float
    h: float

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "x": self.x, "y": self.y, "w": self.w, "h": self.h}


@dataclass
class PlacementCandidate:
    """
    One placement hypothesis before/after legalization.

    After `legalize_candidate`, metadata may include:
    ``legal``, ``legality_status``, ``legalization_passes``, ``resolved_pair_ops``,
    ``remaining_overlapping_pairs``, ``stage``, and optionally ``mixed_size``.
    """

    candidate_id: str
    benchmark_id: str
    macros: list[MacroRect] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "benchmark_id": self.benchmark_id,
            "macros": [m.to_dict() for m in self.macros],
            "metadata": dict(self.metadata),
        }
