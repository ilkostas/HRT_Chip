"""Run configuration and reproducibility snapshot types."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

# Phase 3: preset from docs/step3-multi-objective-proxy-to-ppa.md (inference-time diversity).
GUIDANCE_PRESET_PARETO3: tuple[tuple[float, float, float], ...] = (
    (0.8, 0.1, 0.1),
    (0.2, 0.7, 0.1),
    (0.4, 0.4, 0.2),
)

DEFAULT_GUIDANCE_SINGLE: tuple[tuple[float, float, float], ...] = (
    (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
)

# Phase 4: synthetic corpus scale (v1 smaller, v2 larger).
SyntheticCorpusVersion = Literal["v1", "v2"]

# Phase 4: model architecture selector for training / checkpoint.
ModelArchitecture = Literal["baseline_gnn", "res_gnn", "att_gnn"]

# Phase 4: pipeline sampler backend.
SamplerBackend = Literal["stub", "pytorch_checkpoint"]

# Phase 5: tier-1 evaluator (stub for dev; official requires macro_place + MacroPlacement testcases).
EvaluatorBackend = Literal["stub", "official"]

# Phase 6: how many per-candidate JSON files to keep on disk after a run.
ArtifactRetentionMode = Literal["full", "compact", "best_only"]


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
class SyntheticDatasetConfig:
    """Configuration for synthetic layout dataset generation (Phase 4)."""

    output_dir: Path = field(default_factory=lambda: Path("data/synthetic"))
    corpus_version: SyntheticCorpusVersion = "v1"
    seed: int = 42
    num_samples: int = 256
    """Number of labeled layouts to emit."""
    dataset_version: str = "1"
    """Logical version string stored in the dataset manifest."""
    schema_version: str = "hrt_synthetic_pyg_v1"
    """Schema id for loaders."""

    # v1: fewer macros; v2: larger graphs (paper-style progression).
    n_macros_min: int | None = None
    n_macros_max: int | None = None

    def __post_init__(self) -> None:
        if self.n_macros_min is None or self.n_macros_max is None:
            if self.corpus_version == "v1":
                self.n_macros_min = 2
                self.n_macros_max = 8
            else:
                self.n_macros_min = 8
                self.n_macros_max = 32

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["output_dir"] = str(self.output_dir)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SyntheticDatasetConfig:
        out = dict(data)
        out["output_dir"] = Path(out.get("output_dir", "data/synthetic"))
        return cls(**{k: v for k, v in out.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainingConfig:
    """Phase 4 DDPM ε-prediction training on synthetic PyG graphs."""

    dataset_dir: Path = field(default_factory=lambda: Path("data/synthetic/default"))
    output_dir: Path = field(default_factory=lambda: Path("training_runs"))
    seed: int = 42
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-3
    diffusion_steps: int = 1000
    model_architecture: ModelArchitecture = "baseline_gnn"
    hidden_dim: int = 64
    num_layers: int = 3
    train_run_id: str | None = None
    """Optional fixed id for the training artifact directory."""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["dataset_dir"] = str(self.dataset_dir)
        d["output_dir"] = str(self.output_dir)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingConfig:
        out = dict(data)
        if "dataset_dir" in out:
            out["dataset_dir"] = Path(out["dataset_dir"])
        if "output_dir" in out:
            out["output_dir"] = Path(out["output_dir"])
        return cls(**{k: v for k, v in out.items() if k in cls.__dataclass_fields__})


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

    # Phase 4: optional trained diffusion sampler.
    sampler_backend: SamplerBackend = "stub"
    checkpoint_path: Path | None = None
    """Path to training checkpoint (.pt) when ``sampler_backend`` is ``pytorch_checkpoint``."""
    training_dataset_version: str | None = None
    """Dataset version string from manifest (audit trail)."""
    model_architecture: ModelArchitecture | None = None
    """Architecture recorded at train time; optional echo for manifests."""

    evaluator_backend: EvaluatorBackend = "stub"
    """``stub``: deterministic hash proxy; ``official``: macro_place + TILOS PlacementCost."""

    testcase_root: Path | None = None
    """Directory containing ``<benchmark_id>/netlist.pb.txt`` (ICCAD04). Defaults via env / benchmarks.default_testcase_root."""

    # Phase 6: reproducibility / disk policy.
    deterministic_verification: bool = False
    """If True, apply strict PyTorch/cuDNN determinism toggles (may reduce performance)."""

    artifact_retention: ArtifactRetentionMode = "full"
    """``full``: keep all candidate JSONs; ``compact``: drop per-candidate files (summary stays in results.json); ``best_only``: keep only the selected best candidate file."""

    artifact_retention_top_k: int | None = None
    """When ``artifact_retention`` is ``compact``, optionally keep the top-K candidate JSONs by proxy (lowest first). ``None`` means keep none."""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["output_dir"] = str(self.output_dir)
        if self.guidance_weights_sweep is not None:
            d["guidance_weights_sweep"] = [list(t) for t in self.guidance_weights_sweep]
        else:
            d["guidance_weights_sweep"] = None
        if self.checkpoint_path is not None:
            d["checkpoint_path"] = str(self.checkpoint_path)
        if self.testcase_root is not None:
            d["testcase_root"] = str(self.testcase_root)
        else:
            d["testcase_root"] = None
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunConfig:
        out = dict(data)
        if "output_dir" in out:
            out["output_dir"] = Path(out["output_dir"])
        gws = out.get("guidance_weights_sweep")
        if gws is not None:
            out["guidance_weights_sweep"] = tuple(tuple(float(x) for x in row) for row in gws)
        cp = out.get("checkpoint_path")
        if cp is not None:
            out["checkpoint_path"] = Path(cp)
        # Defaults for new Phase 4 keys when replaying old manifests.
        out.setdefault("sampler_backend", "stub")
        out.setdefault("checkpoint_path", None)
        out.setdefault("training_dataset_version", None)
        out.setdefault("model_architecture", None)
        out.setdefault("evaluator_backend", "stub")
        tr = out.get("testcase_root")
        if tr is not None:
            out["testcase_root"] = Path(tr)
        else:
            out["testcase_root"] = None
        # Phase 6 defaults for older manifests.
        out.setdefault("deterministic_verification", False)
        out.setdefault("artifact_retention", "full")
        out.setdefault("artifact_retention_top_k", None)
        return cls(**{k: v for k, v in out.items() if k in cls.__dataclass_fields__})
