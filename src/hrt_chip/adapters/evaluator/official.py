"""Official TILOS / macro_place proxy evaluator (requires macro_place + MacroPlacement testcases)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from hrt_chip.adapters.evaluator.base import EvaluationResult, EvaluatorAdapter
from hrt_chip.models import PlacementCandidate


class OfficialMacroPlacementEvaluator(EvaluatorAdapter):
    """
    Tier-1 proxy via ``macro_place.objective.compute_proxy_cost`` and validation utilities.

    Install: clone ``partcl-macro-place-challenge``, ``pip install -e .``, and initialize
    ``external/MacroPlacement`` testcases.
    """

    def __init__(self, testcase_root: Path | None = None) -> None:
        from hrt_chip.benchmarks import default_testcase_root

        self.testcase_root = Path(testcase_root or default_testcase_root())
        self._cache: dict[str, tuple[Any, Any]] = {}

    def _bundle(self, benchmark_id: str) -> tuple[Any, Any]:
        if benchmark_id not in self._cache:
            from hrt_chip.official_benchmark import load_benchmark_and_plc

            self._cache[benchmark_id] = load_benchmark_and_plc(benchmark_id, self.testcase_root)
        return self._cache[benchmark_id]

    def prime(self, benchmark_id: str, benchmark: Any, plc: Any) -> None:
        """Avoid a second disk load when the pipeline already loaded the testcase."""
        self._cache[benchmark_id] = (benchmark, plc)

    def evaluate(
        self,
        candidate: PlacementCandidate,
        *,
        benchmark_id: str,
        run_context: dict[str, Any],
    ) -> EvaluationResult:
        from macro_place.objective import compute_proxy_cost
        from macro_place.utils import validate_placement

        benchmark, plc = self._bundle(benchmark_id)
        placement = _placement_tensor_from_candidate(candidate, benchmark)
        costs = compute_proxy_cost(placement, benchmark, plc)
        valid, violations = validate_placement(placement, benchmark, check_overlaps=True)
        overlaps = int(costs.get("overlap_count", 0))
        legal = bool(valid) and overlaps == 0
        proxy = float(costs["proxy_cost"]) if legal else float("inf")
        return EvaluationResult(
            candidate_id=candidate.candidate_id,
            proxy_score=proxy,
            legal=legal,
            details={
                "official": True,
                "benchmark_id": benchmark_id,
                "wirelength_cost": float(costs.get("wirelength_cost", 0.0)),
                "density_cost": float(costs.get("density_cost", 0.0)),
                "congestion_cost": float(costs.get("congestion_cost", 0.0)),
                "overlap_count": overlaps,
                "validate_ok": valid,
                "violations": violations[:10],
            },
        )


def _placement_tensor_from_candidate(candidate: PlacementCandidate, benchmark: Any) -> torch.Tensor:
    """Build [num_macros, 2] center tensor in microns; fixed macros follow benchmark initial centers."""
    by_name = {m.name: m for m in candidate.macros}
    out = benchmark.macro_positions.clone()
    for i in range(benchmark.num_macros):
        fix = benchmark.macro_fixed[i]
        if bool(fix.item() if hasattr(fix, "item") else fix):
            continue
        name = benchmark.macro_names[i]
        m = by_name.get(name)
        if m is None:
            continue
        cx = float(m.x) + float(m.w) / 2.0
        cy = float(m.y) + float(m.h) / 2.0
        out[i, 0] = cx
        out[i, 1] = cy
    return out
