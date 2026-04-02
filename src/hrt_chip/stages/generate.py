"""Candidate generation via diffusion sampler contract (Phase 2 stub, Phase 3 sweep)."""

from __future__ import annotations

import hashlib
from typing import Sequence

from hrt_chip.diffusion import (
    COORD_SPACE_NORMALIZED,
    DeterministicDDPMStubSampler,
    DiffusionSampleRequest,
    DiffusionSampler,
    GuidanceContext,
    MacroSpec,
)
from hrt_chip.geometry import normalized_center_to_lower_left
from hrt_chip.models import MacroRect, PlacementCandidate


def _default_macro_specs(benchmark_id: str) -> tuple[MacroSpec, ...]:
    return (
        MacroSpec(name=f"{benchmark_id}_M0", w=0.12, h=0.08),
        MacroSpec(name=f"{benchmark_id}_M1", w=0.10, h=0.09),
    )


def derive_sweep_seed(base_seed: int, sweep_index: int, weights: tuple[float, float, float]) -> int:
    """
    Deterministic per-sweep RNG seed.

    Sweep index 0 uses ``base_seed`` unchanged so single-vector runs match Phase 2 behavior.
    """
    if sweep_index == 0:
        return base_seed
    payload = f"{base_seed}:{sweep_index}:{weights[0]:.12f}:{weights[1]:.12f}:{weights[2]:.12f}"
    h = hashlib.sha256(payload.encode()).hexdigest()
    return int(h[:12], 16) % (2**31)


def generate_candidates(
    *,
    benchmark_id: str,
    seed: int,
    num_candidates: int,
    macro_specs: Sequence[tuple[str, float, float]] | None = None,
    diffusion_steps: int = 1000,
    sampler: DiffusionSampler | None = None,
    guidance_sweep: Sequence[tuple[float, float, float]] | None = None,
) -> list[PlacementCandidate]:
    """
    Produce ``num_candidates`` placement hypotheses per weight vector using the diffusion sampler.

    ``guidance_sweep`` is a sequence of (α_hpwl, β_congestion, γ_legality) triples; each triple
    gets one ``sample_batch`` call. Total candidates = len(guidance_sweep) * num_candidates.

    Normalized centers in [-1, 1] are converted to unit-canvas lower-left
    coordinates for ``MacroRect`` (Phase 1 legalizer / geometry).
    """
    smp = sampler or DeterministicDDPMStubSampler()
    if macro_specs is not None:
        specs = tuple(MacroSpec(name=n, w=w, h=h) for n, w, h in macro_specs)
    else:
        specs = _default_macro_specs(benchmark_id)

    sweep = (
        tuple(guidance_sweep)
        if guidance_sweep is not None
        else ((1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),)
    )

    out: list[PlacementCandidate] = []
    for si, (a, b, g) in enumerate(sweep):
        sub_seed = derive_sweep_seed(seed, si, (a, b, g))
        gctx = GuidanceContext(
            sweep_index=si,
            alpha_hpwl=a,
            beta_congestion=b,
            gamma_legality=g,
        )
        req = DiffusionSampleRequest(
            benchmark_id=benchmark_id,
            seed=sub_seed,
            num_candidates=num_candidates,
            macro_specs=specs,
            coord_space=COORD_SPACE_NORMALIZED,
            diffusion_steps=diffusion_steps,
            guidance=gctx,
        )
        batch = smp.sample_batch(req)
        prov = batch.provenance.to_dict()

        for cs in batch.candidates:
            macros: list[MacroRect] = []
            normalized_centers: list[dict[str, float | str]] = []
            for center, spec in zip(cs.centers, specs, strict=True):
                normalized_centers.append(center.to_dict())
                x, y = normalized_center_to_lower_left(center.cx, center.cy, spec.w, spec.h)
                macros.append(MacroRect(name=spec.name, x=x, y=y, w=spec.w, h=spec.h))
            # Multi-sweep: prefix for global uniqueness; single-sweep: legacy ids (Phase 2 compat).
            cand_id = cs.candidate_id if len(sweep) == 1 else f"s{si:02d}_{cs.candidate_id}"
            out.append(
                PlacementCandidate(
                    candidate_id=cand_id,
                    benchmark_id=benchmark_id,
                    macros=macros,
                    metadata={
                        "stage": "generated",
                        "sampler": prov,
                        "normalized_centers": normalized_centers,
                        "guidance": {
                            "sweep_index": si,
                            "alpha_hpwl": a,
                            "beta_congestion": b,
                            "gamma_legality": g,
                            "weights": [a, b, g],
                        },
                    },
                )
            )
    return out
