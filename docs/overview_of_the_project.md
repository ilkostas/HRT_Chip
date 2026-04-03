# Overview of the HRT_Chip Project: A Codebase-Level Monograph

This document is a long-form, repository-centric description of **HRT_Chip** (`hrt-chip`): what problem it addresses, how the implementation is structured, which design decisions are “locked in,” and how the code tries to turn research intent into a reproducible engineering system. It is meant to complement—not replace—the shorter design notes under `docs/` (for example `proposed-solution-overview.md` and `implementation-roadmap.md`) by walking the reader through the **actual code paths**, naming conventions, and operational workflows.

---

## 1. Purpose and positioning

### 1.1 What this software is

**HRT_Chip** is a **Python package and command-line tool** for an end-to-end **macro placement** pipeline aimed at the **Partcl × HRT** challenge context described in the repository README. The pipeline’s advertised shape is:

**generate → legalize → mixed-size handoff → evaluate**

That sequence is not merely documentation: it is implemented as a single orchestrated path in `src/hrt_chip/pipeline.py` (`run_pipeline`), with each stage wired to explicit adapters and artifact writers.

The project’s **default stance** is **diffusion-first**: the core generative abstraction is a diffusion-style sampler that proposes **all macro centers simultaneously** (a batch API), rather than a sequential RL policy that places one macro at a time. The repository deliberately avoids maintaining a parallel “RL placement track” as a competing first-class implementation path; research direction is consolidated around diffusion and supporting infrastructure (see `docs/proposed-solution-overview.md`).

### 1.2 What problem macro placement solves (at the level this code cares about)

At a high level, **macro placement** chooses **where fixed rectangular blocks (“macros”)** sit on a **chip canvas** so that downstream objectives (wirelength, routability, congestion proxies, and ultimately timing/area in a full flow) are acceptable.

This codebase focuses on the **placement phase** and on **Tier-1 proxy scoring** as exposed by the official challenge stack when configured. Concretely:

- Macros are represented as **axis-aligned rectangles** (`MacroRect` in `src/hrt_chip/models.py`): lower-left corner `(x, y)` plus width and height.
- **Legality** in the geometric sense used here means: **no pairwise overlaps** among the relevant macro subset (with optional handling of “hard” macro counts and fixed macros), and **macros stay inside the canvas**.
- **Official evaluation** (when available) calls into `macro_place` APIs for **proxy cost** and **placement validation** (`src/hrt_chip/adapters/evaluator/official.py`).

The code does **not** attempt to hide the fact that real benchmarks can contain **hundreds or thousands of macros** and **fixed macros**: those concerns appear explicitly in pipeline branches (for example, initialization strategies when the stub sampler would otherwise make greedy legalization impractically slow).

---

## 2. Engineering worldview: how this repository “thinks”

This section is intentionally subjective: it describes the **implicit engineering philosophy** visible in the structure of the code.

### 2.1 Adapters everywhere: isolate unstable externals

A recurring pattern is: **pure-ish core logic** (geometry, scheduling, data shapes) surrounded by **adapters** that translate to/from external systems (official evaluator, Docker-backed DREAMPlace, PyTorch checkpoints).

Benefits baked into this design:

- **Local development without external assets**: a deterministic stub evaluator can rank candidates without ICCAD04 testcases or GPU clusters.
- **CI realism without full heavyweight dependencies**: the default continuous integration path runs stub evaluation and a replay verification smoke (`README.md`, `.github/workflows/ci.yml`).
- **Gradual integration**: you can turn on “official” or “dreamplace_real” when your environment is ready, without rewriting the pipeline core.

### 2.2 Reproducibility as a first-class output

The project treats a run as something that should be **auditable after the fact**. A run writes:

- `manifest.json`: configuration snapshot and metadata (`src/hrt_chip/io/artifacts.py`, `build_manifest`).
- `results.json`: structured ranking, scoring tables, timing, and alignment diagnostics (`pipeline.run_pipeline`).
- Optional `replay_verification.json` when replaying with verification (`replay_from_manifest` + `src/hrt_chip/replay_verify.py`).

This is not “nice to have” documentation—it is how the authors intend to prevent silent regressions when sampler logic, legalization, or selection policy changes.

### 2.3 Separation between “training objective” and “inference steering”

The documentation set makes a key research distinction (see `docs/proposed-solution-overview.md`):

- **Training** (Phase 4) targets a standard diffusion noise-prediction objective on **synthetic** layouts (implemented under `src/hrt_chip/training/` and `src/hrt_chip/data/`).
- **Inference-time multi-objective exploration** uses weight triples **(α, β, γ)** attached to generation requests as **guidance metadata** and surrogate diagnostics (`src/hrt_chip/diffusion.py`, `src/hrt_chip/guidance.py`, `src/hrt_chip/stages/generate.py`).

The codebase preserves this separation carefully: guidance fields are recorded in provenance even when a sampler ignores them for true gradient steering (the stub sampler uses weights only to bias random placement slightly; the PyTorch sampler records guidance but the core DDPM/DDIM reverse process is ε-prediction driven).

### 2.4 Legality: continuous generation, discrete guarantee

The research docs explain why MaskPlace-style discrete masking does not map cleanly onto simultaneous diffusion. The code’s practical response is:

- Accept that raw samples may be **illegal** in continuous space.
- Run a **deterministic greedy legalizer** (`src/hrt_chip/stages/legalize.py`) to pursue **zero overlap**.
- Assert consistency between **metadata** and **geometry checks** in the pipeline (`placement_is_legal` vs `candidate.metadata["legal"]`).

This is a deliberate engineering compromise: **optimize in continuous space**, but **never claim success without a hard geometric check** before you trust downstream stages.

---

## 3. End-to-end architecture: the pipeline as the spine

The orchestrator is `run_pipeline` in `src/hrt_chip/pipeline.py`. At a high level, it performs:

1. **Determinism hooks**: `apply_pipeline_determinism(config)` (`src/hrt_chip/deterministic_runtime.py`) when configured.
2. **Benchmark loading (optional official path)**: if `evaluator_backend == "official"`, load benchmark objects and macro specs from disk via `official_benchmark.load_full_benchmark`.
3. **Adapter construction**: evaluator (`LocalStubEvaluator` vs `OfficialMacroPlacementEvaluator`) and mixed-size backend (stub/estimate/Docker variants).
4. **Artifact directory setup**: `PipelineArtifacts` under `output_dir/run_id`.
5. **Guidance sweep resolution**: `resolved_guidance_sweep` (`src/hrt_chip/config.py`) turns CLI presets like `pareto3` or explicit triples into a concrete list of `(α, β, γ)` vectors.
6. **Budget resolution**: `resolve_generation_budget` (`src/hrt_chip/budget.py`) may shrink candidate counts or truncate sweeps to meet wall-clock budgets.
7. **Outer loop over guidance vectors**: for each weight triple, generate candidates, legalize, mixed-size, compute surrogate objectives, evaluate, accumulate scoring rows.
8. **Ranking and selection**: sort evaluations by a selection policy (`proxy_first` vs `ppa_priority`) using helpers in `src/hrt_chip/mixed_size_metrics.py`.
9. **Diagnostics**: surrogate vs official proxy alignment statistics (Spearman/Kendall and mismatch summaries) via `src/hrt_chip/rank_metrics.py`.
10. **Write `results.json`**, apply retention policy, return a Python dict mirroring JSON.

The CLI entrypoint (`src/hrt_chip/cli.py`, Typer) exposes these knobs for human and automated runs (`hrt-chip run`, `hrt-chip replay`, `hrt-chip benchmark-sweep`, plus dataset/training commands for Phase 4).

---

## 4. Stage 1: Generation (diffusion contracts)

### 4.1 Contract: `DiffusionSampler` and `DiffusionSampleRequest`

`src/hrt_chip/diffusion.py` defines the abstract contract:

- **`DiffusionSampleRequest`**: specifies benchmark id, RNG seed, number of candidates, macro specs, coordinate space label, diffusion step count, optional guidance context `(α, β, γ)`, optional objective bias placeholder, and optional PyTorch overrides (inference step count, sampler mode, reverse timestep indices).
- **`DiffusionSampler`**: a protocol with `sample_batch(request) -> SampleBatch`.
- **`SampleBatch`**: contains multiple `CandidateSample` items plus **`SamplerProvenance`** for traceability.

**Coordinate convention**: the canonical coordinate tag is `normalized_centers_-1_1`. Macro centers are conceptualized in **normalized** \([-1, 1]^2\) space; conversion to physical lower-left placement uses `normalized_center_to_lower_left` in `src/hrt_chip/geometry.py`.

### 4.2 Stub sampler: `DeterministicDDPMStubSampler`

The stub sampler is the **Phase 2** implementation: it is not a neural network. It generates random normalized centers per macro and candidate, with a **deterministic mapping** from seeds and optional **guidance biases** (small shifts) to explore different regions of space.

Purpose in the project:

- Provide a **fully local** generation path for tests and CI.
- Validate pipeline plumbing and artifact schemas without GPU training.

### 4.3 PyTorch sampler: `PyTorchDDPMSampler`

`src/hrt_chip/adapters/diffusion/pytorch_sampler.py` implements the same `sample_batch` API using:

- A trained ε-model checkpoint (`load_checkpoint` in `src/hrt_chip/training/checkpoint.py`).
- TorchGeometric `Batch` objects with **complete graph** edges (`complete_edge_index` in `src/hrt_chip/data/graph_utils.py`) and node features derived from macro width/height.
- Reverse diffusion loops using schedules in `src/hrt_chip/training/schedule.py` (DDPM full/subsampled, DDIM).

This is the bridge from **Phase 4 training** to **Phase 2/3 inference contracts**: the outer pipeline does not “know” PyTorch; it only depends on the `DiffusionSampler` protocol.

### 4.4 `generate_candidates` stage wrapper

`src/hrt_chip/stages/generate.py` converts sampler output into **`PlacementCandidate`** objects:

- Iterates guidance vectors (often length 1, but may be multiple).
- Derives per-sweep seeds via `derive_sweep_seed` so multi-sweep behavior is stable and Phase-2-compatible for sweep index 0.
- Converts normalized centers to **`MacroRect`** footprints using canvas dimensions.

---

## 5. Stage 2: Legalization (hard constraints via algorithmic repair)

### 5.1 What `legalize_candidate` does

`src/hrt_chip/stages/legalize.py` implements a **deterministic greedy** strategy:

- Macros are processed in a fixed nested-loop order; overlaps trigger **minimum L1-style separation moves** (`min_l1_separation_move_b` and related geometry helpers in `src/hrt_chip/geometry.py`).
- There is a **`hard_macro_count`** concept: only the first *N* macros participate in overlap checks in some modes, matching benchmark semantics where additional entries may exist for data plumbing reasons.
- **`fixed_mask`** support ensures the legalizer does not try to “move” fixed macros; the official pipeline later restores fixed macro positions (`restore_fixed_macro_positions` in `src/hrt_chip/official_benchmark.py`).

### 5.2 Geometry primitives

`src/hrt_chip/geometry.py` provides the **computational geometry backbone**:

- Overlap area and overlap predicates with epsilon tolerances.
- Canvas containment checks.
- Pairwise overlap counting, with optional hard-macro limits.
- Normalized center mapping to lower-left coordinates.

These functions are used both by the legalizer and by **objective surrogates** (overlap penalties).

### 5.3 Official benchmark special case: why the pipeline sometimes bypasses naive random generation

`pipeline.py` includes an important pragmatic branch: when running **official** benchmarks with the **stub sampler**, pure random normalized placements can create extremely hard legalization instances for large macro counts. In that configuration, the pipeline may **seed candidates near the benchmark’s provided macro centers** with small deterministic jitter for movable macros.

This is a **performance and stability** measure for “smoke” and evidence runs, not a statement that the final competition solver should always warm-start from the input placement. It encodes a lesson: **algorithmic worst cases matter** when your legalizer is greedy and \(O(n^2)\) in the overlap scan.

---

## 6. Stage 3: Mixed-size handoff (proxy-ready downstream placement)

Macros are only part of a mixed-size design; standard cells exist conceptually in the full flow. The mixed-size stage is abstracted behind `MixedSizeBackend` (`src/hrt_chip/adapters/mixed_size/base.py`) and implemented by:

- **`LocalStubMixedSizeBackend`**: no-op / minimal metadata for fast tests.
- **`MixedSizeEstimateBackend`**: analytical-ish proxies (utilization + RUDY-like quantities) described in integration docs; good default for local iteration (`src/hrt_chip/adapters/mixed_size/estimate.py`).
- **`DreamPlaceDockerBackend`**: Docker integration for DREAMPlace-oriented workflows, including a “real” variant (`dreamplace_real`) (`src/hrt_chip/adapters/mixed_size/dreamplace_docker.py`).

The pipeline constructs a `MixedSizeRequest` with fixed macro rectangles and benchmark context, runs the backend, and attaches results into candidate metadata. Illegal macro placements skip mixed-size work and record why.

**Selection policies** (`RunConfig.selection_policy`) allow the final ranking to emphasize:

- **`proxy_first`**: official proxy is primary; mixed-size composite can break ties.
- **`ppa_priority`**: emphasize mixed-size normalized metrics for legal candidates with successful backend runs (`src/hrt_chip/mixed_size_metrics.py`).

---

## 7. Stage 4: Evaluation (Tier-1 proxy and stubs)

### 7.1 Evaluator adapter contract

`src/hrt_chip/adapters/evaluator/base.py` defines `EvaluatorAdapter.evaluate(...) -> EvaluationResult`.

### 7.2 Local stub evaluator

`LocalStubEvaluator` (`src/hrt_chip/adapters/evaluator/local_stub.py`) produces a **deterministic pseudo-proxy** from a hash of the candidate JSON and the run seed. Illegal candidates receive infinite proxy cost.

This is ideal for:

- Unit tests and CI where you want stable ordering without external dependencies.

### 7.3 Official evaluator

`OfficialMacroPlacementEvaluator` (`src/hrt_chip/adapters/evaluator/official.py`) calls:

- `macro_place.objective.compute_proxy_cost`
- `macro_place.utils.validate_placement`

…and returns proxy cost, component breakdowns, and legality information. It can be **primed** with already-loaded benchmark objects to avoid duplicate IO.

---

## 8. Guidance, surrogates, and “Phase 3” metadata

Even when the neural sampler does not yet implement full gradient-based guided diffusion, the project tracks **inference weights** and **cheap surrogate objectives** for analysis and future steering.

### 8.1 Weight triples \((\alpha, \beta, \gamma)\)

`config.py` defines default presets (including `pareto3`) and resolution rules: explicit sweeps override presets.

### 8.2 Surrogate objectives

`src/hrt_chip/guidance.py` computes `ObjectiveComponents`:

- **HPWL / congestion** surrogates may be **netlist-aware** when benchmark pin group data exists (`netlist_surrogates.py`), else may be absent/`nan` with a documented surrogate mode.
- **Legality surrogate** includes smooth overlap penalties and discrete overlap counts.

`composite_guidance_objective` combines surrogates using the same weights as the guidance sweep, producing a single scalar for **diagnostics** (not the official score).

### 8.3 Surrogate vs official alignment

`pipeline.py` computes rank correlations and “surrogate good / proxy bad” style mismatch summaries (`rank_metrics.py`). This directly supports research iteration: it tells you when your cheap objective is a misleading compass relative to Tier-1 proxy.

---

## 9. Phase 4: Synthetic data and training (implemented, optional at runtime)

Training is organized under `src/hrt_chip/training/` and `src/hrt_chip/data/`:

- **Synthetic dataset generation** (`src/hrt_chip/data/synthetic.py`, manifests via `src/hrt_chip/data/manifest.py`).
- **PyG dataset loading** (`src/hrt_chip/data/pyg_dataset.py`).
- **Model architectures** (`EpsilonPlacementNet` in `src/hrt_chip/training/models.py`): GCN/GAT-style backbones with time embedding.
- **Training loop** (`src/hrt_chip/training/train.py`, schedules in `src/hrt_chip/training/schedule.py`, checkpoints in `src/hrt_chip/training/checkpoint.py`).

The intent matches the research docs: learn a generative model offline on synthetic graphs, then evaluate in the real pipeline via checkpoint sampling.

---

## 10. Phase 5: Benchmark harness and milestone framing

`src/hrt_chip/benchmarks.py` centralizes:

- The canonical list of **17 IBM ICCAD04 IDs** used by the challenge framing.
- Aggregate and per-design **baseline proxy references** (SA and RePlAce) used as **milestone gates** in sweep reports.

`src/hrt_chip/benchmark_sweep.py` orchestrates multi-benchmark runs; the CLI exposes `hrt-chip benchmark-sweep`.

This phase is as much **project management instrumentation** as it is code: it turns research progress into comparable tables across benchmarks.

---

## 11. Phase 6: Determinism, replay, retention, and CI discipline

### 11.1 Deterministic verification mode

`RunConfig.deterministic_verification` triggers stricter PyTorch/cuDNN determinism toggles (`deterministic_runtime.py`). This exists because **floating-point and GPU nondeterminism** can otherwise make replay checks flaky.

### 11.2 Replay verification

`hrt-chip replay path/to/manifest.json --verify` loads the saved `RunConfig`, reruns the pipeline, compares a **fingerprint** of ranking outputs (`replay_verify.py`), and writes `replay_verification.json`.

### 11.3 Artifact retention policies

Because candidate JSON files can be large, `artifact_retention` supports `full`, `compact`, and `best_only` modes with optional `top_k` retention (`io/artifacts.py`, used at the end of `run_pipeline`).

### 11.4 Continuous integration

GitHub Actions runs `uv sync --group dev`, `pytest`, a CLI smoke (`hrt-chip run` + `replay --verify`), and a stub sweep regression script (`tools/check_stub_sweep_regression.py`).

---

## 12. Runtime budgets and operational safety

`src/hrt_chip/runtime_budget.py` and `src/hrt_chip/budget.py` implement **approximate scheduling** to avoid accidentally launching runs that cannot finish under a wall-clock limit: they may reduce candidates per sweep vector or truncate sweep vectors, recording the resolution in `results.json`.

Separately, Docker backends use configurable timeouts (`dreamplace_docker_timeout_seconds`, etc.), acknowledging that containerized toolchains are often the dominant tail latency.

---

## 13. How to run the project (tooling)

The repository is explicitly **`uv`-managed** (`pyproject.toml`). Typical commands:

- Install: `uv sync --group dev`
- Tests: `uv run pytest`
- CLI: `uv run hrt-chip ...`

Entrypoint: `hrt-chip = hrt_chip.cli:main`.

Dependencies include **Typer** and **Rich** for CLI UX, **NumPy**, **PyTorch**, and **Torch Geometric** for the neural diffusion path.

---

## 14. Testing philosophy (what the tests imply)

The `tests/` suite is organized around phased guardrails:

- **Pipeline smoke tests** ensure the skeleton runs end-to-end.
- **Geometry tests** protect normalization and legality predicates.
- **Phase 3 tests** protect guidance and surrogate behavior.
- **Phase 4/5/6 tests** protect training schedules, benchmark gates, determinism/replay, and schema stability (`test_baseline_schema.py`).

This mirrors the roadmap: the project’s definition of “done” is not only algorithms on paper, but **locked behaviors** under automated checks.

---

## 15. Known limitations (honest system view)

The README lists major limitations; the codebase makes them tangible:

- **Mixed-size “production” quality** depends on backend availability (Docker images, environment).
- **Scaling training data and model quality** is ongoing; synthetic distributions may not match benchmark structure.
- **Tier-2 OpenROAD-oriented handoff** is not fully productized here.

These are not footnotes—they shape why adapters and stubs exist.

---

## 16. How to read this repository (a suggested path)

If you are onboarding into the code for the first time, this reading order matches the runtime spine:

1. `README.md` — commands and capabilities.
2. `src/hrt_chip/cli.py` — what operators can do.
3. `src/hrt_chip/config.py` — what is configurable and what defaults mean.
4. `src/hrt_chip/pipeline.py` — the real story of orchestration.
5. `src/hrt_chip/diffusion.py` + `src/hrt_chip/stages/generate.py` — generation contracts.
6. `src/hrt_chip/geometry.py` + `src/hrt_chip/stages/legalize.py` — hard constraints.
7. `src/hrt_chip/guidance.py` + `src/hrt_chip/adapters/evaluator/*` — objectives and scoring.
8. `src/hrt_chip/training/*` + `src/hrt_chip/adapters/diffusion/pytorch_sampler.py` — learned sampler path.
9. `docs/implementation-roadmap.md` — which phases are marked done vs planned at the documentation level.

---

## 17. Closing remarks: what “success” means in this codebase

For this project, “success” is not a single number in a leaderboard—it is a **repeatable pipeline** that:

- proposes placements in a diffusion-shaped **simultaneous** interface,
- repairs illegality with explicit algorithmic guarantees checked in geometry,
- evaluates candidates with swappable **official or stub** scoring,
- records enough metadata to **replay and verify** decisions later,
- and scales operationally via **budgeting**, **retention**, and **multi-benchmark sweeps**.

The code is written to make those goals **measurable**, not merely aspirational.

---

## 18. Related documentation in this repository

- `docs/proposed-solution-overview.md` — research framing (diffusion-first, masking discussion, proxy-to-PPA narrative).
- `docs/implementation-roadmap.md` — phased build plan with explicit file pointers and “done” statuses.
- `docs/integration-notes.md` — external evaluator and Docker backend contracts (see README pointer).
- `docs/baseline-artifacts.md` — artifact schema expectations for baseline comparisons.
- `docs/step2-position-masking.md`, `docs/step3-multi-objective-proxy-to-ppa.md` — deeper dives referenced throughout the roadmap.

This overview file (`docs/overview_of_the_project.md`) is intended to be the **widest** lens: it ties those documents to **modules, functions, and execution order** as implemented in the tree today.
