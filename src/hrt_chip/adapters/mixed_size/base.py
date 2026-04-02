"""Contract for mixed-size placement handoff (macros fixed, std-cell region backend)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from hrt_chip.models import MacroRect, PlacementCandidate


@dataclass
class MixedSizeRequest:
    """Inputs for standard-cell clustering/placement after macro legalization."""

    benchmark_id: str
    fixed_macros: list[MacroRect]
    seed: int
    backend_options: dict[str, Any] = field(default_factory=dict)


@dataclass
class MixedSizeResult:
    """Opaque handle for proxy evaluation after mixed-size stage (Phase 0 stub)."""

    ok: bool
    message: str
    extra: dict[str, Any] = field(default_factory=dict)


class MixedSizeBackend(ABC):
    """
    Bridge to DREAMPlace, hMETIS, or competition-provided clustering/placement.

    Phase 0: no-op / stub. Phase 1+ may invoke real tools via `external/`.
    """

    @abstractmethod
    def run(self, request: MixedSizeRequest) -> MixedSizeResult:
        """Run backend on fixed macro geometry; return status for evaluator handoff."""

    def attach_to_candidate(self, candidate: PlacementCandidate, result: MixedSizeResult) -> None:
        """Merge backend output into candidate metadata for evaluator."""
        candidate.metadata["mixed_size"] = {
            "ok": result.ok,
            "message": result.message,
            "extra": dict(result.extra),
        }
