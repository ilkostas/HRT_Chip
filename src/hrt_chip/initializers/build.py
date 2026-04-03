"""Build multi-family seed candidates for search-hybrid pipeline."""

from __future__ import annotations

import random
from typing import Any

from hrt_chip.diffusion import DiffusionSampler
from hrt_chip.models import MacroRect, PlacementCandidate
from hrt_chip.initializers.random_legal import random_legal_candidate
from hrt_chip.stages.generate import derive_sweep_seed, generate_candidates
from hrt_chip.config import RunConfig, resolved_guidance_sweep


def _jitter_official(
    config: RunConfig,
    bench_obj: Any,
    canvas_w: float,
    canvas_h: float,
    sweep: tuple[tuple[float, float, float], ...],
    seed_sub: int,
    cand_suffix: str,
) -> PlacementCandidate:
    """Deterministic jitter around benchmark macro centers (official benchmarks)."""
    a, b, g = sweep[0]
    rng = random.Random(config.seed + 10_000 * seed_sub)
    centers = bench_obj.macro_positions
    sizes = bench_obj.macro_sizes
    names = bench_obj.macro_names
    fixed_mask_local = [bool(x) for x in bench_obj.macro_fixed.tolist()]
    jitter_x = 0.01 * float(canvas_w)
    jitter_y = 0.01 * float(canvas_h)
    macros = []
    for mi in range(int(bench_obj.num_macros)):
        cx = float(centers[mi, 0])
        cy = float(centers[mi, 1])
        if not fixed_mask_local[mi]:
            cx += rng.uniform(-jitter_x, jitter_x)
            cy += rng.uniform(-jitter_y, jitter_y)
        w = float(sizes[mi, 0])
        h = float(sizes[mi, 1])
        macros.append(MacroRect(name=str(names[mi]), x=cx - w / 2.0, y=cy - h / 2.0, w=w, h=h))
    return PlacementCandidate(
        candidate_id=f"seed_benchjit_{cand_suffix}",
        benchmark_id=config.benchmark_id,
        macros=macros,
        metadata={
            "stage": "initialized",
            "seed_family": "benchmark_jitter",
            "guidance": {
                "sweep_index": 0,
                "alpha_hpwl": float(a),
                "beta_congestion": float(b),
                "gamma_legality": float(g),
                "weights": [float(a), float(b), float(g)],
            },
        },
    )


def build_seed_candidates(
    config: RunConfig,
    *,
    bench_obj: Any | None,
    macro_specs_arg: Any,
    canvas_w: float,
    canvas_h: float,
    sampler: DiffusionSampler | None,
) -> list[PlacementCandidate]:
    """
    Produce seed placements for each enabled family × seeds_per_family.

    Requires ``bench_obj`` for official ``benchmark_jitter`` / ``random_legal`` / ``analytical_push``.
    Stub evaluator runs fall back to diffusion-only or stub generate.
    """
    families = config.search_families
    k = max(1, int(config.search_seeds_per_family))
    out: list[PlacementCandidate] = []
    sweep = resolved_guidance_sweep(
        guidance_preset=config.guidance_preset,
        guidance_weights_sweep=config.guidance_weights_sweep,
    )

    fam_idx = 0
    for family in families:
        for j in range(k):
            suffix = f"{fam_idx:02d}_{j:02d}"
            fam_idx += 1
            sub_seed = config.seed + 17_000 * fam_idx + j * 31

            if family == "benchmark_jitter":
                if bench_obj is None:
                    continue
                out.append(
                    _jitter_official(config, bench_obj, canvas_w, canvas_h, sweep, fam_idx * 100 + j, suffix)
                )
            elif family == "random_legal":
                if bench_obj is None:
                    continue
                rng = random.Random(sub_seed)
                rc = random_legal_candidate(
                    benchmark_id=config.benchmark_id,
                    bench_obj=bench_obj,
                    canvas_w=canvas_w,
                    canvas_h=canvas_h,
                    rng=rng,
                    candidate_id=f"seed_randleg_{suffix}",
                )
                if rc is not None:
                    out.append(rc)
            elif family == "diffusion":
                ss = derive_sweep_seed(config.seed, j, sweep[0])
                chunk = generate_candidates(
                    benchmark_id=config.benchmark_id,
                    seed=ss,
                    num_candidates=1,
                    macro_specs=macro_specs_arg,
                    canvas_w=canvas_w,
                    canvas_h=canvas_h,
                    diffusion_steps=config.diffusion_steps,
                    guidance_sweep=(sweep[0],),
                    sampler=sampler,
                    guidance_sweep_index_offset=0,
                )
                for c in chunk:
                    c.candidate_id = f"seed_diff_{suffix}"
                    c.metadata["seed_family"] = "diffusion"
                    c.metadata["stage"] = "initialized"
                    out.append(c)
            elif family == "analytical_push":
                if bench_obj is None:
                    continue
                base = _jitter_official(
                    config, bench_obj, canvas_w, canvas_h, sweep, fam_idx * 100 + j, suffix + "_ap"
                )
                # Light push: nudge each movable macro toward chip center (reduces spread drift).
                rng = random.Random(sub_seed)
                cx0 = 0.5 * canvas_w
                cy0 = 0.5 * canvas_h
                alpha = 0.02 * (0.5 + rng.random())
                fixed_mask_local = [bool(x) for x in bench_obj.macro_fixed.tolist()]
                for mi, m in enumerate(base.macros):
                    if mi < len(fixed_mask_local) and not fixed_mask_local[mi]:
                        mcx = m.x + m.w / 2.0
                        mcy = m.y + m.h / 2.0
                        mcx += alpha * (cx0 - mcx)
                        mcy += alpha * (cy0 - mcy)
                        m.x = mcx - m.w / 2.0
                        m.y = mcy - m.h / 2.0
                        m.x = max(0.0, min(m.x, canvas_w - m.w))
                        m.y = max(0.0, min(m.y, canvas_h - m.h))
                base.candidate_id = f"seed_analytical_{suffix}"
                base.metadata["seed_family"] = "analytical_push"
                out.append(base)

    if not out and bench_obj is not None:
        out.append(
            _jitter_official(config, bench_obj, canvas_w, canvas_h, sweep, 0, "fallback"),
        )
    if not out:
        chunk = generate_candidates(
            benchmark_id=config.benchmark_id,
            seed=config.seed,
            num_candidates=max(1, k),
            macro_specs=macro_specs_arg,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            diffusion_steps=config.diffusion_steps,
            guidance_sweep=(sweep[0],),
            sampler=sampler,
            guidance_sweep_index_offset=0,
        )
        for c in chunk:
            c.candidate_id = f"seed_stub_{c.candidate_id}"
            c.metadata["seed_family"] = "stub_generate"
            c.metadata["stage"] = "initialized"
            out.append(c)
    return out
