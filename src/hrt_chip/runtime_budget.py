"""Runtime wall-clock budget tracking and adaptive decisions during pipeline execution."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Literal

from hrt_chip.budget import _estimated_seconds_per_candidate, estimated_generation_seconds_per_candidate
from hrt_chip.config import RunConfig

StageName = Literal["generation", "legalization", "mixed_size", "evaluation", "reserve"]


def _fractions_from_run_config(config: RunConfig) -> StageBudgetFractions:
    d = config.runtime_budget_stage_fractions
    if not d:
        return StageBudgetFractions()
    return StageBudgetFractions(
        generation=float(d.get("generation", 0.28)),
        legalization=float(d.get("legalization", 0.12)),
        mixed_size=float(d.get("mixed_size", 0.28)),
        evaluation=float(d.get("evaluation", 0.22)),
        reserve=float(d.get("reserve", 0.10)),
    )


@dataclass
class StageBudgetFractions:
    """Fractions of total wall-clock budget reserved per stage (must sum to <= 1.0)."""

    generation: float = 0.28
    legalization: float = 0.12
    mixed_size: float = 0.28
    evaluation: float = 0.22
    reserve: float = 0.10

    def __post_init__(self) -> None:
        s = (
            self.generation
            + self.legalization
            + self.mixed_size
            + self.evaluation
            + self.reserve
        )
        if s > 1.0001:
            raise ValueError(f"Stage budget fractions sum to {s}, must be <= 1.0")


@dataclass
class RuntimeBudgetManager:
    """
    Tracks elapsed time, per-stage spend, and EMA of per-candidate pipeline cost.

    Used when ``wall_clock_budget_seconds`` is set to avoid exhausting the budget
    during generation when evaluation is still pending.
    """

    wall_clock_budget_seconds: float
    start_perf: float
    fractions: StageBudgetFractions = field(default_factory=StageBudgetFractions)
    initial_per_candidate_estimate: float = 1.0
    ema_alpha: float = 0.35

    _spent: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    _ema_per_candidate: float | None = None
    _n_ema_samples: int = 0

    def __post_init__(self) -> None:
        self._ema_per_candidate = self.initial_per_candidate_estimate

    @classmethod
    def from_config(
        cls,
        config: RunConfig,
        *,
        start_perf: float,
    ) -> RuntimeBudgetManager | None:
        if config.wall_clock_budget_seconds is None:
            return None
        b = float(config.wall_clock_budget_seconds)
        if b <= 0:
            return None
        per = _estimated_seconds_per_candidate(config)
        fr = _fractions_from_run_config(config)
        return cls(
            wall_clock_budget_seconds=b,
            start_perf=start_perf,
            fractions=fr,
            initial_per_candidate_estimate=max(per, 0.05),
        )

    def elapsed(self) -> float:
        return time.perf_counter() - self.start_perf

    def remaining_wall(self) -> float:
        return max(0.0, self.wall_clock_budget_seconds - self.elapsed())

    def allocated_for_stage(self, stage: StageName) -> float:
        f = getattr(self.fractions, stage)
        return self.wall_clock_budget_seconds * f

    def remaining_for_stage(self, stage: StageName) -> float:
        return max(0.0, self.allocated_for_stage(stage) - self._spent[stage])

    def record(self, stage: str, seconds: float) -> None:
        self._spent[stage] += max(0.0, seconds)

    def observe_candidate_post_generation(self, seconds: float) -> None:
        """Update EMA for legalize + mixed-size + official eval for one candidate (seconds)."""
        if self._ema_per_candidate is None:
            self._ema_per_candidate = seconds
        else:
            a = self.ema_alpha
            self._ema_per_candidate = a * seconds + (1.0 - a) * self._ema_per_candidate
        self._n_ema_samples += 1

    def estimated_seconds_for_candidates(self, n: int) -> float:
        ema = self._ema_per_candidate or self.initial_per_candidate_estimate
        return max(0.0, ema * n)

    def can_generate_next_sweep_vector(
        self,
        config: RunConfig,
        *,
        num_candidates_this_vector: int,
        already_generated_unprocessed: int,
    ) -> bool:
        """
        Return False if generating another batch would likely exceed the wall budget
        before finishing legalization / mixed-size / eval for all pending work.
        """
        rem = self.remaining_wall()
        gen_per = estimated_generation_seconds_per_candidate(config)
        total_pending = already_generated_unprocessed + num_candidates_this_vector
        gen_cost = gen_per * num_candidates_this_vector
        pipe_per = self._ema_per_candidate or max(
            0.05, self.initial_per_candidate_estimate - gen_per
        )
        pipe_est = pipe_per * total_pending
        slack = 0.05 * self.wall_clock_budget_seconds
        return rem >= gen_cost + pipe_est + slack

    def recommended_diffusion_inference_steps(
        self,
        *,
        base_steps: int | None,
        training_timesteps: int,
        min_steps: int = 8,
    ) -> int | None:
        """
        Under time pressure, reduce effective inference steps for later batches.

        Returns None to mean \"use full training_timesteps\" (ddpm_full / unset cap).
        """
        rem = self.remaining_wall()
        frac = rem / max(self.wall_clock_budget_seconds, 1e-9)
        if frac > 0.35:
            return base_steps
        if base_steps is None:
            # Accelerate full trajectory when budget tight: cap to ~half
            return max(min_steps, training_timesteps // 2)
        return max(min_steps, int(base_steps * max(0.5, frac * 2.0)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "wall_clock_budget_seconds": self.wall_clock_budget_seconds,
            "elapsed_seconds": self.elapsed(),
            "remaining_seconds": self.remaining_wall(),
            "spent_by_stage": dict(self._spent),
            "ema_seconds_per_candidate_pipeline": self._ema_per_candidate,
            "ema_candidate_samples": self._n_ema_samples,
            "stage_fractions": {
                "generation": self.fractions.generation,
                "legalization": self.fractions.legalization,
                "mixed_size": self.fractions.mixed_size,
                "evaluation": self.fractions.evaluation,
                "reserve": self.fractions.reserve,
            },
        }
