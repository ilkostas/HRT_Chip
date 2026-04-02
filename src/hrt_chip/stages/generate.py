"""Candidate generation via diffusion sampler contract (Phase 2 stub)."""

from __future__ import annotations

from typing import Sequence

from hrt_chip.diffusion import (
    COORD_SPACE_NORMALIZED,
    DeterministicDDPMStubSampler,
    DiffusionSampleRequest,
    DiffusionSampler,
    MacroSpec,
)
from hrt_chip.geometry import normalized_center_to_lower_left
from hrt_chip.models import MacroRect, PlacementCandidate


def _default_macro_specs(benchmark_id: str) -> tuple[MacroSpec, ...]:
    return (
        MacroSpec(name=f"{benchmark_id}_M0", w=0.12, h=0.08),
        MacroSpec(name=f"{benchmark_id}_M1", w=0.10, h=0.09),
    )


def generate_candidates(
    *,
    benchmark_id: str,
    seed: int,
    num_candidates: int,
    macro_specs: Sequence[tuple[str, float, float]] | None = None,
    diffusion_steps: int = 1000,
    sampler: DiffusionSampler | None = None,
) -> list[PlacementCandidate]:
    """
    Produce ``num_candidates`` placement hypotheses using the diffusion sampler.

    Normalized centers in [-1, 1] are converted to unit-canvas lower-left
    coordinates for ``MacroRect`` (Phase 1 legalizer / geometry).
    """
    smp = sampler or DeterministicDDPMStubSampler()
    if macro_specs is not None:
        specs = tuple(MacroSpec(name=n, w=w, h=h) for n, w, h in macro_specs)
    else:
        specs = _default_macro_specs(benchmark_id)

    req = DiffusionSampleRequest(
        benchmark_id=benchmark_id,
        seed=seed,
        num_candidates=num_candidates,
        macro_specs=specs,
        coord_space=COORD_SPACE_NORMALIZED,
        diffusion_steps=diffusion_steps,
    )
    batch = smp.sample_batch(req)
    prov = batch.provenance.to_dict()

    out: list[PlacementCandidate] = []
    for cs in batch.candidates:
        macros: list[MacroRect] = []
        normalized_centers: list[dict[str, float | str]] = []
        for center, spec in zip(cs.centers, specs, strict=True):
            normalized_centers.append(center.to_dict())
            x, y = normalized_center_to_lower_left(center.cx, center.cy, spec.w, spec.h)
            macros.append(MacroRect(name=spec.name, x=x, y=y, w=spec.w, h=spec.h))
        out.append(
            PlacementCandidate(
                candidate_id=cs.candidate_id,
                benchmark_id=benchmark_id,
                macros=macros,
                metadata={
                    "stage": "generated",
                    "sampler": prov,
                    "normalized_centers": normalized_centers,
                },
            )
        )
    return out
