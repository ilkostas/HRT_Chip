"""Wall-clock budget resolution for candidate generation and guidance sweeps."""

from __future__ import annotations

from typing import Any

from hrt_chip.config import RunConfig


def estimated_generation_seconds_per_candidate(config: RunConfig) -> float:
    """Rough CPU/GPU time for diffusion generation only (per candidate)."""
    if config.sampler_backend == "pytorch_checkpoint":
        steps = max(1, int(config.diffusion_steps))
        inferred = config.diffusion_inference_steps
        eff = steps if inferred is None else min(steps, max(1, int(inferred)))
        return max(0.02, eff * 0.008)
    return 0.05


def _estimated_seconds_per_candidate(config: RunConfig) -> float:
    """Rough lower bound for scheduling; tune with profiling on official + mixed-size."""
    base = 0.5 if config.evaluator_backend == "stub" else 12.0
    base += estimated_generation_seconds_per_candidate(config)
    if config.mixed_size_backend == "estimate":
        base += 0.2
    elif config.mixed_size_backend == "stub":
        base += 0.02
    elif config.mixed_size_backend == "dreamplace":
        # Docker pull/run variance; budget uses ~half the configured timeout as a rough upper bound.
        base += 10.0 + 0.55 * float(max(1, int(config.dreamplace_docker_timeout_seconds)))
    elif config.mixed_size_backend == "dreamplace_real":
        base += 15.0 + 0.55 * float(max(1, int(config.dreamplace_real_docker_timeout_seconds)))
    return max(base, 0.1)


def resolve_generation_budget(
    config: RunConfig,
    sweep: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[tuple[float, float, float], ...], int, dict[str, Any]]:
    """
    Optionally shrink (sweep, num_candidates) to fit ``wall_clock_budget_seconds``.

    Prefers keeping the full sweep and reducing K; if insufficient, truncates sweep prefix.
    """
    if config.wall_clock_budget_seconds is None:
        return sweep, config.num_candidates, {"budget_limited": False}

    budget = float(config.wall_clock_budget_seconds)
    per = _estimated_seconds_per_candidate(config)
    reserve = min(budget * 0.12, max(0.0, budget - 1e-9))
    alloc = max(0.0, budget - reserve)
    max_total = max(1, int(alloc / per)) if alloc >= 1e-12 else 1

    n_vec = len(sweep)
    k = config.num_candidates
    desired = n_vec * k

    if desired <= max_total:
        return sweep, k, {
            "budget_limited": False,
            "wall_clock_budget_seconds": budget,
            "estimated_seconds_per_candidate": per,
            "max_total_candidates_under_budget": max_total,
        }

    k2 = max(1, max_total // n_vec)
    if k2 >= 1:
        return sweep, k2, {
            "budget_limited": True,
            "wall_clock_budget_seconds": budget,
            "estimated_seconds_per_candidate": per,
            "requested_num_candidates": k,
            "resolved_num_candidates": k2,
            "guidance_vectors": n_vec,
        }

    new_n = max(1, max_total // max(k, 1))
    new_sweep = tuple(sweep[:new_n])
    return new_sweep, k, {
        "budget_limited": True,
        "wall_clock_budget_seconds": budget,
        "estimated_seconds_per_candidate": per,
        "requested_num_candidates": k,
        "resolved_num_candidates": k,
        "requested_guidance_vectors": n_vec,
        "resolved_guidance_vectors": new_n,
    }
