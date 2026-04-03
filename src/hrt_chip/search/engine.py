"""Screening, time allocation, and SA orchestration for search-hybrid runs."""

from __future__ import annotations

import random
from typing import Any, Callable

from hrt_chip.config import RunConfig
from hrt_chip.models import PlacementCandidate
from hrt_chip.search.objective import placement_energy
from hrt_chip.search.sa import run_simulated_annealing


def _total_search_seconds(config: RunConfig) -> float:
    if config.wall_clock_budget_seconds is not None:
        return max(30.0, float(config.wall_clock_budget_seconds) * 0.82)
    return 300.0


def run_search_on_seeds(
    seeds: list[PlacementCandidate],
    config: RunConfig,
    *,
    benchmark: Any | None,
    canvas_w: float,
    canvas_h: float,
    hard_macro_count: int | None,
    fixed_mask: list[bool] | None,
    official_energy_fn: Callable[[PlacementCandidate], float | None] | None,
) -> list[tuple[PlacementCandidate, dict[str, Any]]]:
    """
    Screen each seed with short SA, then refine top-K seeds with remaining budget.

    Returns list of (candidate, combined_telemetry) for each seed (refined where applicable).
    """
    if not seeds:
        return []

    hard_n = hard_macro_count if hard_macro_count is not None else len(seeds[0].macros)
    rng_master = random.Random(config.seed + 900_000)
    total = _total_search_seconds(config)
    n = len(seeds)
    default_screen = min(120.0, max(15.0, total * 0.22 / max(1, n)))
    screen_t = float(config.search_screen_seconds) if config.search_screen_seconds is not None else default_screen
    screen_t = max(5.0, screen_t)

    screened: list[tuple[PlacementCandidate, dict[str, Any], float]] = []

    for i, cand in enumerate(seeds):
        # Work on a shallow copy of macro positions via candidate passed to SA (in-place)
        sub_rng = random.Random(rng_master.randint(0, 2**30))
        use_official = (
            config.search_official_eval_every_n_steps is not None
            and config.search_official_eval_every_n_steps > 0
            and official_energy_fn is not None
        )
        tele = run_simulated_annealing(
            cand,
            benchmark=benchmark,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            hard_n=hard_n,
            fixed_mask=fixed_mask,
            rng=sub_rng,
            time_limit_seconds=screen_t,
            max_iterations=config.search_max_iterations,
            cooling_rate=config.search_sa_cooling_rate,
            min_temperature=config.search_sa_min_temperature,
            initial_temperature_scale=config.search_sa_initial_temperature_scale,
            max_shift_fraction=config.search_max_shift_fraction,
            objective_schedule=config.search_objective_schedule,
            adaptive_operators=config.search_adaptive_operators,
            enable_net_aware=config.search_enable_net_aware,
            enable_swap=config.search_enable_swap,
            enable_cluster=config.search_enable_cluster,
            official_eval_interval=config.search_official_eval_every_n_steps if use_official else None,
            official_energy_fn=official_energy_fn if use_official else None,
        )
        e = placement_energy(
            cand,
            benchmark=benchmark,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            mode="full",
        )
        screened.append((cand, tele, e))

    screened.sort(key=lambda x: x[2])
    top_k = max(1, min(int(config.search_refine_top_k), len(screened)))
    refine_budget = max(0.0, total - screen_t * n)
    refine_t = refine_budget / max(1, top_k)

    out: list[tuple[PlacementCandidate, dict[str, Any]]] = []
    for rank, (cand, tele_screen, _) in enumerate(screened):
        meta: dict[str, Any] = {
            "phase": "screen",
            "screen_rank": rank,
            "screen_energy": tele_screen.get("best_energy"),
            **tele_screen,
        }
        if rank < top_k and refine_t >= 1.0:
            sub_rng = random.Random(rng_master.randint(0, 2**30))
            use_official = (
                config.search_official_eval_every_n_steps is not None
                and config.search_official_eval_every_n_steps > 0
                and official_energy_fn is not None
            )
            tele_r = run_simulated_annealing(
                cand,
                benchmark=benchmark,
                canvas_w=canvas_w,
                canvas_h=canvas_h,
                hard_n=hard_n,
                fixed_mask=fixed_mask,
                rng=sub_rng,
                time_limit_seconds=refine_t,
                max_iterations=config.search_max_iterations,
                cooling_rate=config.search_sa_cooling_rate,
                min_temperature=config.search_sa_min_temperature,
                initial_temperature_scale=config.search_sa_initial_temperature_scale,
                max_shift_fraction=config.search_max_shift_fraction * 0.6,
                objective_schedule=config.search_objective_schedule,
                adaptive_operators=config.search_adaptive_operators,
                enable_net_aware=config.search_enable_net_aware,
                enable_swap=config.search_enable_swap,
                enable_cluster=config.search_enable_cluster,
                official_eval_interval=config.search_official_eval_every_n_steps if use_official else None,
                official_energy_fn=official_energy_fn if use_official else None,
            )
            meta["phase"] = "screen+refine"
            meta["refine_telemetry"] = tele_r
        cand.metadata["search"] = meta
        out.append((cand, meta))

    return out
