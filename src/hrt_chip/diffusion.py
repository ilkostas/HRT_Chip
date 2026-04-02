"""Diffusion inference contracts and deterministic DDPM stub sampler."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

COORD_SPACE_NORMALIZED = "normalized_centers_-1_1"


@dataclass(frozen=True)
class MacroSpec:
    """Macro geometry and identity used by generation contracts."""

    name: str
    w: float
    h: float


@dataclass(frozen=True)
class DiffusionSampleRequest:
    """
    Simultaneous placement request for all macros in one batch.

    The sampler must generate coordinates for the full macro set for each candidate.
    """

    benchmark_id: str
    seed: int
    num_candidates: int
    macro_specs: tuple[MacroSpec, ...]
    coord_space: str = COORD_SPACE_NORMALIZED
    diffusion_steps: int = 1000


@dataclass(frozen=True)
class MacroCenter:
    """One macro center in normalized coordinate space."""

    name: str
    cx: float
    cy: float

    def to_dict(self) -> dict[str, float | str]:
        return {"name": self.name, "cx": self.cx, "cy": self.cy}


@dataclass(frozen=True)
class CandidateSample:
    """Sampler output for a single candidate."""

    candidate_id: str
    centers: tuple[MacroCenter, ...]


@dataclass(frozen=True)
class SamplerProvenance:
    """Traceability fields persisted in manifest/results/candidate metadata."""

    sampler_name: str
    model_stub: str
    generation_mode: str
    coord_space: str
    seed: int
    num_candidates: int
    diffusion_steps: int

    def to_dict(self) -> dict[str, str | int]:
        return {
            "sampler_name": self.sampler_name,
            "model_stub": self.model_stub,
            "generation_mode": self.generation_mode,
            "coord_space": self.coord_space,
            "seed": self.seed,
            "num_candidates": self.num_candidates,
            "diffusion_steps": self.diffusion_steps,
        }


@dataclass(frozen=True)
class SampleBatch:
    """Batched candidate output with sampler provenance."""

    candidates: tuple[CandidateSample, ...]
    provenance: SamplerProvenance


class DiffusionSampler(Protocol):
    """Diffusion sampler interface: batch-in, full-layout batch-out."""

    def sample_batch(self, request: DiffusionSampleRequest) -> SampleBatch:
        """Return simultaneous full-macro coordinates for every candidate."""


class DeterministicDDPMStubSampler:
    """
    Deterministic sampler for Phase 2.

    Emits normalized centers in [-1, 1] with a simultaneous-batch API (no
    sequential macro-by-macro placement).
    """

    sampler_name = "ddpm_stub_sampler"
    model_stub = "ddpm_model_stub_v1"
    generation_mode = "simultaneous_diffusion"

    def sample_batch(self, request: DiffusionSampleRequest) -> SampleBatch:
        rng = random.Random(request.seed)
        out: list[CandidateSample] = []
        for idx in range(request.num_candidates):
            candidate_id = f"cand_{idx:04d}"
            centers: list[MacroCenter] = []
            for spec in request.macro_specs:
                centers.append(
                    MacroCenter(
                        name=spec.name,
                        cx=rng.uniform(-1.0, 1.0),
                        cy=rng.uniform(-1.0, 1.0),
                    )
                )
            out.append(CandidateSample(candidate_id=candidate_id, centers=tuple(centers)))

        provenance = SamplerProvenance(
            sampler_name=self.sampler_name,
            model_stub=self.model_stub,
            generation_mode=self.generation_mode,
            coord_space=request.coord_space,
            seed=request.seed,
            num_candidates=request.num_candidates,
            diffusion_steps=request.diffusion_steps,
        )
        return SampleBatch(candidates=tuple(out), provenance=provenance)
