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

# PyTorch reverse diffusion: full DDPM, subsampled DDPM steps, or DDIM jumps.
SamplerMode = Literal["ddpm_full", "ddpm_subsampled", "ddim"]

# Phase 5: tier-1 evaluator (stub for dev; official requires macro_place + MacroPlacement testcases).
EvaluatorBackend = Literal["stub", "official"]

# Phase 6: how many per-candidate JSON files to keep on disk after a run.
ArtifactRetentionMode = Literal["full", "compact", "best_only"]

# Mixed-size handoff: stub, analytical estimate, Docker analytical proxy, or real toolchain image.
MixedSizeBackendName = Literal["stub", "estimate", "dreamplace", "dreamplace_real"]

# Final candidate ordering within a run (default preserves proxy-first Tier-1 style).
SelectionPolicy = Literal["proxy_first", "ppa_priority"]

# Hybrid pivot: legacy diffusion path vs search-centric solver.
SolverBackend = Literal["legacy", "search_hybrid"]

# Search objective schedule inside SA (HPWL-first then full surrogate).
SearchObjectiveSchedule = Literal["hpwl_only", "hpwl_then_full", "full_surrogate"]

# Synthetic dataset curriculum (Phase 4+).
SyntheticCurriculum = Literal["grid_v1", "benchmark_like"]


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
    """Schema id for loaders; benchmark_like curriculum bumps to v2 in the written manifest."""

    # v1: fewer macros; v2: larger graphs (paper-style progression).
    n_macros_min: int | None = None
    n_macros_max: int | None = None

    curriculum: SyntheticCurriculum = "grid_v1"
    """``grid_v1``: legacy packed grid; ``benchmark_like``: heavy-tail sizes + spatial net sampling."""

    def __post_init__(self) -> None:
        if self.curriculum not in ("grid_v1", "benchmark_like"):
            raise ValueError(f"curriculum must be grid_v1 or benchmark_like, got {self.curriculum!r}")
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
        out.setdefault("curriculum", "grid_v1")
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

    wall_clock_budget_seconds: float | None = None
    """If set, shrink guidance sweep / per-vector candidate count to fit this wall-clock budget."""

    mixed_size_backend: MixedSizeBackendName = "estimate"
    """``stub``: no-op; ``estimate``: utilization + RUDY; ``dreamplace`` / ``dreamplace_real``: Docker ``/work`` flow."""

    selection_policy: SelectionPolicy = "proxy_first"
    """``proxy_first``: rank by evaluator proxy (tie-break mixed-size composite). ``ppa_priority``: rank legal+ms-ok by composite first."""

    dreamplace_docker_image: str = "hrt-chip-dreamplace:local"
    """Image for ``mixed_size_backend=dreamplace`` (override env ``HRT_DREAMPLACE_IMAGE`` at runtime in adapter)."""

    dreamplace_real_docker_image: str = "hrt-chip-dreamplace-real:local"
    """Image for ``mixed_size_backend=dreamplace_real`` (env ``HRT_DREAMPLACE_REAL_IMAGE``)."""

    dreamplace_docker_timeout_seconds: int = 300
    """Per-candidate ``docker run`` timeout."""

    dreamplace_real_docker_timeout_seconds: int = 900
    """Per-candidate timeout for ``dreamplace_real`` (env ``HRT_DREAMPLACE_REAL_TIMEOUT``)."""

    dreamplace_docker_retries: int = 0
    """Retry count on transient Docker failures (0 = no retry)."""

    dreamplace_mount_testcase: bool = True
    """Mount ICCAD04 testcase dir read-only at ``/testcase`` in the container when using dreamplace backend."""

    dreamplace_docker_extra_args: str | None = None
    """Extra ``docker run`` args (quoted space-separated), e.g. ``-v /path/to/hmetis:/tools/hmetis:ro``."""

    dreamplace_docker_executable: str = "docker"
    """Docker CLI executable (``docker`` or full path; on Windows often ``docker`` from PATH)."""

    diffusion_inference_steps: int | None = None
    """Optional cap on reverse-diffusion steps for ``pytorch_checkpoint`` sampler (accelerated sampling)."""

    sampler_mode: SamplerMode = "ddpm_subsampled"
    """``ddpm_full``: full ``T`` ancestral steps; ``ddpm_subsampled``: fewer DDPM steps; ``ddim``: deterministic DDIM jumps."""

    diffusion_reverse_schedule: str | None = None
    """
    Optional explicit reverse timestep list for pytorch sampler, high→low, comma-separated
    (e.g. ``1000,500,250,0``). Must be valid indices in ``[0, T-1]``; last step should include ``0``.
    When set, overrides uniform subsampling for ``ddpm_subsampled`` / ``ddim``.
    """

    ddim_eta: float = 0.0
    """DDIM stochasticity (0 = deterministic). Reserved for future; current implementation uses ``eta=0``."""

    runtime_budget_stage_fractions: dict[str, float] | None = None
    """Optional overrides for generation/legalization/mixed_size/evaluation/reserve fractions (sum ≤ 1)."""

    pre_eval_rejection_enabled: bool = False
    """If True, skip expensive evaluator for obviously bad candidates (overlap / surrogate gates)."""

    pre_eval_max_hard_overlap_pairs: int | None = None
    """Skip official eval when ``hard_overlap_pairs`` exceeds this after legalization (None = no gate)."""

    pre_eval_surrogate_composite_max: float | None = None
    """Skip eval when surrogate composite is finite and above this threshold (None = no gate)."""

    experiment_tag: str | None = None
    """Optional label stored in sweep metadata / trends."""

    experiment_notes: str | None = None
    """Free-form notes stored in sweep_report extra."""

    trends_log_path: str | None = None
    """Append-only JSONL path for sweep trend lines (default: runs/trends/sweep_history.jsonl)."""

    # --- Hybrid search-centric solver (see docs/hybrid-search-solver.md) ---
    solver_backend: SolverBackend = "legacy"
    """``legacy``: generate → legalize (default, CI/replay). ``search_hybrid``: initialize → SA → evaluate."""

    search_families: tuple[str, ...] = ("benchmark_jitter", "random_legal")
    """
    Seed sources: ``benchmark_jitter``, ``random_legal``, ``diffusion`` (uses sampler when available).
    Optional: ``analytical_push`` (light force-directed nudge from current positions).
    """

    search_seeds_per_family: int = 2
    """Number of deterministic seeds to draw per enabled family."""

    search_screen_seconds: float | None = None
    """Short SA per seed for ranking; default ~25% of search budget when wall_clock_budget_seconds set."""

    search_refine_top_k: int = 3
    """After screening, allocate remaining search time to this many best seeds."""

    search_sa_cooling_rate: float = 0.995
    search_sa_min_temperature: float = 1e-6
    search_sa_initial_temperature_scale: float = 0.05
    search_max_shift_fraction: float = 0.3
    search_max_iterations: int = 500_000
    search_objective_schedule: SearchObjectiveSchedule = "hpwl_then_full"
    """``hpwl_then_full``: first ~60% of steps HPWL-only, then full surrogate. ``hpwl_only`` / ``full_surrogate``."""

    search_adaptive_operators: bool = True
    """Bias operator choice toward accepted improving moves."""

    search_surrogate_min_spearman: float | None = 0.90
    """If running batch calibration, below this threshold use official-eval anchoring instead of trusting surrogate."""

    search_official_eval_every_n_steps: int | None = None
    """When surrogate is untrusted, call official evaluator every N SA steps (expensive; None = disabled)."""

    search_enable_net_aware: bool = True
    search_enable_swap: bool = True
    search_enable_cluster: bool = True

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
        d["wall_clock_budget_seconds"] = self.wall_clock_budget_seconds
        d["mixed_size_backend"] = self.mixed_size_backend
        d["selection_policy"] = self.selection_policy
        d["diffusion_inference_steps"] = self.diffusion_inference_steps
        d["sampler_mode"] = self.sampler_mode
        d["diffusion_reverse_schedule"] = self.diffusion_reverse_schedule
        d["ddim_eta"] = self.ddim_eta
        d["runtime_budget_stage_fractions"] = self.runtime_budget_stage_fractions
        d["pre_eval_rejection_enabled"] = self.pre_eval_rejection_enabled
        d["pre_eval_max_hard_overlap_pairs"] = self.pre_eval_max_hard_overlap_pairs
        d["pre_eval_surrogate_composite_max"] = self.pre_eval_surrogate_composite_max
        d["experiment_tag"] = self.experiment_tag
        d["experiment_notes"] = self.experiment_notes
        d["trends_log_path"] = self.trends_log_path
        d["dreamplace_docker_image"] = self.dreamplace_docker_image
        d["dreamplace_real_docker_image"] = self.dreamplace_real_docker_image
        d["dreamplace_docker_timeout_seconds"] = self.dreamplace_docker_timeout_seconds
        d["dreamplace_real_docker_timeout_seconds"] = self.dreamplace_real_docker_timeout_seconds
        d["dreamplace_docker_retries"] = self.dreamplace_docker_retries
        d["dreamplace_mount_testcase"] = self.dreamplace_mount_testcase
        d["dreamplace_docker_extra_args"] = self.dreamplace_docker_extra_args
        d["dreamplace_docker_executable"] = self.dreamplace_docker_executable
        d["solver_backend"] = self.solver_backend
        d["search_families"] = list(self.search_families)
        d["search_seeds_per_family"] = self.search_seeds_per_family
        d["search_screen_seconds"] = self.search_screen_seconds
        d["search_refine_top_k"] = self.search_refine_top_k
        d["search_sa_cooling_rate"] = self.search_sa_cooling_rate
        d["search_sa_min_temperature"] = self.search_sa_min_temperature
        d["search_sa_initial_temperature_scale"] = self.search_sa_initial_temperature_scale
        d["search_max_shift_fraction"] = self.search_max_shift_fraction
        d["search_max_iterations"] = self.search_max_iterations
        d["search_objective_schedule"] = self.search_objective_schedule
        d["search_adaptive_operators"] = self.search_adaptive_operators
        d["search_surrogate_min_spearman"] = self.search_surrogate_min_spearman
        d["search_official_eval_every_n_steps"] = self.search_official_eval_every_n_steps
        d["search_enable_net_aware"] = self.search_enable_net_aware
        d["search_enable_swap"] = self.search_enable_swap
        d["search_enable_cluster"] = self.search_enable_cluster
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
        out.setdefault("wall_clock_budget_seconds", None)
        out.setdefault("mixed_size_backend", "estimate")
        out.setdefault("selection_policy", "proxy_first")
        out.setdefault("diffusion_inference_steps", None)
        out.setdefault("sampler_mode", "ddpm_subsampled")
        out.setdefault("diffusion_reverse_schedule", None)
        out.setdefault("ddim_eta", 0.0)
        out.setdefault("runtime_budget_stage_fractions", None)
        out.setdefault("pre_eval_rejection_enabled", False)
        out.setdefault("pre_eval_max_hard_overlap_pairs", None)
        out.setdefault("pre_eval_surrogate_composite_max", None)
        out.setdefault("experiment_tag", None)
        out.setdefault("experiment_notes", None)
        out.setdefault("trends_log_path", None)
        out.setdefault("dreamplace_docker_image", "hrt-chip-dreamplace:local")
        out.setdefault("dreamplace_real_docker_image", "hrt-chip-dreamplace-real:local")
        out.setdefault("dreamplace_docker_timeout_seconds", 300)
        out.setdefault("dreamplace_real_docker_timeout_seconds", 900)
        out.setdefault("dreamplace_docker_retries", 0)
        out.setdefault("dreamplace_mount_testcase", True)
        out.setdefault("dreamplace_docker_extra_args", None)
        out.setdefault("dreamplace_docker_executable", "docker")
        out.setdefault("solver_backend", "legacy")
        sf = out.get("search_families")
        if isinstance(sf, list):
            out["search_families"] = tuple(str(x) for x in sf)
        else:
            out.setdefault("search_families", ("benchmark_jitter", "random_legal"))
        out.setdefault("search_seeds_per_family", 2)
        out.setdefault("search_screen_seconds", None)
        out.setdefault("search_refine_top_k", 3)
        out.setdefault("search_sa_cooling_rate", 0.995)
        out.setdefault("search_sa_min_temperature", 1e-6)
        out.setdefault("search_sa_initial_temperature_scale", 0.05)
        out.setdefault("search_max_shift_fraction", 0.3)
        out.setdefault("search_max_iterations", 500_000)
        out.setdefault("search_objective_schedule", "hpwl_then_full")
        out.setdefault("search_adaptive_operators", True)
        out.setdefault("search_surrogate_min_spearman", 0.90)
        out.setdefault("search_official_eval_every_n_steps", None)
        out.setdefault("search_enable_net_aware", True)
        out.setdefault("search_enable_swap", True)
        out.setdefault("search_enable_cluster", True)
        return cls(**{k: v for k, v in out.items() if k in cls.__dataclass_fields__})
