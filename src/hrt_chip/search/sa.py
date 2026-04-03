"""Simulated annealing core with adaptive operator weights."""

from __future__ import annotations

import math
import random
import time
from typing import Any, Callable

from hrt_chip.models import PlacementCandidate
from hrt_chip.search.objective import placement_energy, schedule_mode
from hrt_chip.search.operators import (
    OperatorKind,
    propose_cluster,
    propose_net_aware,
    propose_shift,
    propose_swap,
)


def run_simulated_annealing(
    candidate: PlacementCandidate,
    *,
    benchmark: Any | None,
    canvas_w: float,
    canvas_h: float,
    hard_n: int,
    fixed_mask: list[bool] | None,
    rng: random.Random,
    time_limit_seconds: float,
    max_iterations: int,
    cooling_rate: float,
    min_temperature: float,
    initial_temperature_scale: float,
    max_shift_fraction: float,
    objective_schedule: str,
    adaptive_operators: bool,
    enable_net_aware: bool,
    enable_swap: bool,
    enable_cluster: bool,
    official_eval_interval: int | None = None,
    official_energy_fn: Callable[[PlacementCandidate], float | None] | None = None,
) -> dict[str, Any]:
    """
    In-place refinement of ``candidate``. Returns telemetry dict.
    """
    t0 = time.perf_counter()
    deadline = t0 + max(0.0, time_limit_seconds)

    macros = candidate.macros
    total_schedule_steps = max_iterations
    temp0 = max(
        min_temperature,
        abs(placement_energy(candidate, benchmark=benchmark, canvas_w=canvas_w, canvas_h=canvas_h, mode="hpwl"))
        * initial_temperature_scale,
    )
    temp = float(temp0)
    cur_energy = placement_energy(
        candidate,
        benchmark=benchmark,
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        mode=schedule_mode(0, total_schedule_steps, objective_schedule),
    )
    best_energy = cur_energy
    best_xy = [(m.x, m.y) for m in macros]

    op_weights: dict[OperatorKind, float] = {
        OperatorKind.SHIFT: 1.0,
        OperatorKind.NET_AWARE: 1.0,
        OperatorKind.SWAP: 0.8,
        OperatorKind.CLUSTER: 0.6,
    }
    op_accepted: dict[str, int] = {k.value: 0 for k in OperatorKind}
    op_improved: dict[str, int] = {k.value: 0 for k in OperatorKind}
    iterations = 0
    accepted_moves = 0
    improving_moves = 0
    illegal_or_fail = 0
    last_official_anchor: float | None = None

    def pick_operator() -> OperatorKind:
        choices: list[OperatorKind] = [OperatorKind.SHIFT]
        if enable_net_aware:
            choices.append(OperatorKind.NET_AWARE)
        if enable_swap:
            choices.append(OperatorKind.SWAP)
        if enable_cluster:
            choices.append(OperatorKind.CLUSTER)
        if not adaptive_operators:
            return rng.choice(choices)
        wts = [max(0.05, op_weights[k]) for k in choices]
        s = sum(wts)
        r = rng.random() * s
        acc = 0.0
        for k, w in zip(choices, wts, strict=True):
            acc += w
            if r <= acc:
                return k
        return choices[-1]

    while iterations < max_iterations and time.perf_counter() < deadline:
        mode = schedule_mode(iterations, total_schedule_steps, objective_schedule)
        max_span = max(canvas_w, canvas_h) * max_shift_fraction * max(0.05, min(1.0, temp / max(temp0, 1e-12)))

        snapshot = [(m.x, m.y) for m in macros]
        kind = pick_operator()
        prop = None
        if kind == OperatorKind.SHIFT:
            prop = propose_shift(
                candidate,
                rng,
                hard_n=hard_n,
                fixed_mask=fixed_mask,
                canvas_w=canvas_w,
                canvas_h=canvas_h,
                max_span=max_span,
            )
        elif kind == OperatorKind.NET_AWARE:
            prop = propose_net_aware(
                candidate,
                rng,
                benchmark,
                hard_n=hard_n,
                fixed_mask=fixed_mask,
                canvas_w=canvas_w,
                canvas_h=canvas_h,
                max_span=max_span,
            )
        elif kind == OperatorKind.SWAP:
            prop = propose_swap(
                candidate,
                rng,
                hard_n=hard_n,
                fixed_mask=fixed_mask,
                canvas_w=canvas_w,
                canvas_h=canvas_h,
            )
        else:
            prop = propose_cluster(
                candidate,
                rng,
                benchmark,
                hard_n=hard_n,
                fixed_mask=fixed_mask,
                canvas_w=canvas_w,
                canvas_h=canvas_h,
                max_span=max_span,
            )

        if prop is None:
            illegal_or_fail += 1
            temp *= cooling_rate
            iterations += 1
            continue

        new_energy = placement_energy(
            candidate,
            benchmark=benchmark,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            mode=mode,
        )
        delta = new_energy - cur_energy
        accept = delta <= 0.0
        if not accept:
            z = max(-60.0, min(0.0, -delta / max(temp, 1e-12)))
            accept = rng.random() < math.exp(z)

        if accept:
            accepted_moves += 1
            op_accepted[prop.kind.value] += 1
            if delta < 0:
                improving_moves += 1
                op_improved[prop.kind.value] += 1
                if adaptive_operators:
                    op_weights[prop.kind] *= 1.05
            cur_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_xy = [(m.x, m.y) for m in macros]
        else:
            for i, m in enumerate(macros):
                m.x, m.y = snapshot[i]

        temp = max(min_temperature, temp * cooling_rate)
        iterations += 1

        if (
            official_eval_interval
            and official_energy_fn is not None
            and iterations % official_eval_interval == 0
        ):
            pe = official_energy_fn(candidate)
            if pe is not None and math.isfinite(pe):
                last_official_anchor = float(pe)
                if pe < best_energy:
                    best_energy = float(pe)
                    best_xy = [(m.x, m.y) for m in macros]

    for i, m in enumerate(macros):
        m.x, m.y = best_xy[i]

    elapsed = time.perf_counter() - t0
    return {
        "iterations": iterations,
        "accepted_moves": accepted_moves,
        "improving_moves": improving_moves,
        "illegal_or_failed_proposals": illegal_or_fail,
        "best_energy": best_energy,
        "temperature_start": temp0,
        "temperature_end": temp,
        "operator_accepted": op_accepted,
        "operator_improved": op_improved,
        "operator_weights_final": {k.value: float(v) for k, v in op_weights.items()},
        "elapsed_seconds": elapsed,
        "last_official_anchor_energy": last_official_anchor,
    }
