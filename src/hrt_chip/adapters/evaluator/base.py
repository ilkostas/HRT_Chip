"""Contract for official proxy / tier-1 evaluator integration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from hrt_chip.models import PlacementCandidate


@dataclass
class EvaluationResult:
    """Result of evaluating one legalized candidate."""

    candidate_id: str
    proxy_score: float
    legal: bool
    details: dict[str, Any]


class EvaluatorAdapter(ABC):
    """Adapter to the competition evaluator (local mock or official binary/API)."""

    @abstractmethod
    def evaluate(
        self,
        candidate: PlacementCandidate,
        *,
        benchmark_id: str,
        run_context: dict[str, Any],
    ) -> EvaluationResult:
        """Return official-proxy-aligned score for tier-1 selection."""
