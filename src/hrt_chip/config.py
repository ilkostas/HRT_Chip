"""Run configuration and reproducibility snapshot types."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Phase 3: preset from docs/step3-multi-objective-proxy-to-ppa.md (inference-time diversity).
GUIDANCE_PRESET_PARETO3: tuple[tuple[float, float, float], ...] = (
    (0.8, 0.1, 0.1),
    (0.2, 0.7, 0.1),
    (0.4, 0.4, 0.2),
)

DEFAULT_GUIDANCE_SINGLE: tuple[tuple[float, float, float], ...] = (
    (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
)


def resolved_guidance_sweep(
    *,
    guidance_preset: str | None,
    guidance_weights_sweep: tuple[tuple[float, float, float], ...] | None,
) -> tuple[tuple[float, float, float], ...]:
    """Resolve (α, β, γ) vectors: explicit sweep overrides preset; else preset or balanced single vector."""
    if guidance_weights_sweep is not None and len(guidance_weights_sweep) > 0:
        return guidance_weights_sweep
    if guidance_preset == "pareto3":
        return GUIDANCE_PRESET_PARETO3
    return DEFAULT_GUIDANCE_SINGLE


@dataclass
class RunConfig:
    """User-facing configuration for a single pipeline run."""

    benchmark_id: str = "ibm01"
    seed: int = 42
    num_candidates: int = 4
    diffusion_steps: int = 1000
    output_dir: Path = field(default_factory=lambda: Path("runs"))
    deterministic: bool = True
    # Phase 3: multi-weight sweep — K candidates per weight vector (total = len(sweep) * num_candidates).
    guidance_preset: str | None = None
    """Use built-in weight sets: ``pareto3`` or ``None`` for default single balanced vector."""

    guidance_weights_sweep: tuple[tuple[float, float, float], ...] | None = None
    """Explicit list of (α_hpwl, β_congestion, γ_legality); overrides ``guidance_preset`` when set."""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["output_dir"] = str(self.output_dir)
        if self.guidance_weights_sweep is not None:
            d["guidance_weights_sweep"] = [list(t) for t in self.guidance_weights_sweep]
        else:
            d["guidance_weights_sweep"] = None
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunConfig:
        out = dict(data)
        if "output_dir" in out:
            out["output_dir"] = Path(out["output_dir"])
        gws = out.get("guidance_weights_sweep")
        if gws is not None:
            out["guidance_weights_sweep"] = tuple(tuple(float(x) for x in row) for row in gws)
        return cls(**{k: v for k, v in out.items() if k in cls.__dataclass_fields__})
