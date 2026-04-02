"""Evaluate stage: invoke evaluator adapter and collect results."""

from __future__ import annotations

from typing import Any

from hrt_chip.adapters.evaluator.base import EvaluatorAdapter, EvaluationResult
from hrt_chip.models import PlacementCandidate


def evaluate_candidate(
    candidate: PlacementCandidate,
    evaluator: EvaluatorAdapter,
    *,
    benchmark_id: str,
    run_context: dict[str, Any],
) -> EvaluationResult:
    return evaluator.evaluate(candidate, benchmark_id=benchmark_id, run_context=run_context)
