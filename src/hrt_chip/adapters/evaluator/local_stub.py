"""Deterministic local evaluator stub for Phase 0 (no external binary)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from hrt_chip.adapters.evaluator.base import EvaluationResult, EvaluatorAdapter
from hrt_chip.models import PlacementCandidate


def _stable_float_from_payload(payload: str, seed: int) -> float:
    """Deterministic pseudo-proxy in [0, 1) for ranking demos."""
    h = hashlib.sha256(f"{seed}:{payload}".encode()).hexdigest()
    return int(h[:16], 16) / float(1 << 64)


class LocalStubEvaluator(EvaluatorAdapter):
    """
    Mock proxy: lower is better (matches typical wirelength-style cost).

    Integrate the official challenge evaluator behind `EvaluatorAdapter` later;
    see docs/integration-notes.md.
    """

    def evaluate(
        self,
        candidate: PlacementCandidate,
        *,
        benchmark_id: str,
        run_context: dict[str, Any],
    ) -> EvaluationResult:
        seed = int(run_context.get("seed", 0))
        payload = json.dumps(candidate.to_dict(), sort_keys=True)
        proxy = _stable_float_from_payload(payload, seed) * 1e6
        legal = bool(candidate.metadata.get("legal", True))
        if not legal:
            proxy = float("inf")
        return EvaluationResult(
            candidate_id=candidate.candidate_id,
            proxy_score=proxy,
            legal=legal,
            details={"stub": True, "benchmark_id": benchmark_id},
        )
