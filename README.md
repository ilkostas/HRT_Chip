# HRT_Chip

Docs-first research repository for the Partcl x Hudson River Trading (HRT) macro placement challenge.  
This project currently defines a diffusion-centered methodology for macro placement under strict competition constraints, with emphasis on legality, proxy-score optimization, and practical bridge-to-PPA reasoning.

## Project Thesis

Macro placement is a constrained, high-dimensional optimization problem where seemingly local moves alter global routing, density, and timing behavior. This repository commits to a diffusion-first, three-part strategy:

1. Generate candidate macro coordinates with a simultaneous diffusion process.
2. Enforce hard legality through overlap-aware guidance plus explicit post-processing legalization.
3. Explore multi-objective trade-offs at inference time and select finalists with the official proxy evaluator.

The docs in this repo are written to make the reasoning explicit: what is optimized, when it is optimized, which guarantees are strict, and where assumptions remain open until implementation and benchmark validation.

## Problem Context and Hard Constraints

The target task is macro placement for the Partcl/HRT challenge on ICCAD04 IBM benchmarks (tier-1 proxy ranking) with top entries validated via OpenROAD on NG45 designs (tier-2 PPA metrics).

### Competition-Critical Constraints

- **Hard legality**: final placements must have zero overlaps under the official checker.
- **Runtime cap**: end-to-end algorithm runtime is limited to one hour per benchmark.
- **Two-tier evaluation**:
  - Tier 1: proxy-cost ranking across IBM benchmarks.
  - Tier 2: top proxy entries are measured on WNS, TNS, and Area in OpenROAD.
- **Generalization requirement**: no benchmark-specific hardcoding.

These constraints shape all architectural decisions in this repository.

## Methodology (Current Design Direction)

The proposed system is documented as a three-step pipeline.

### Step 1: Diffusion Core Generator

The primary generator is a DDPM-style diffusion model that predicts continuous macro coordinates jointly rather than placing macros one-by-one. This repository does not maintain a parallel RL core; resources are focused on one diffusion implementation path plus a strong legalizer and evaluator loop.

Key ideas:

- Coordinates are modeled in normalized continuous space and denoised over timesteps.
- Netlist structure is represented with graph-aware neural components (GNN/attention concepts in docs).
- Training focuses on denoising objectives; placement-quality shaping occurs during inference via guidance.

### Step 2: Legality Strategy

This repository explicitly distinguishes two legality paradigms:

- **Sequential RL masking** (MaskPlace-like): legality-by-construction on a placement order and discrete grid.
- **Simultaneous diffusion**: no equivalent per-step autoregressive mask; legality is promoted using continuous overlap penalties and then completed by legalization.

Design implication: in this diffusion-first project, a dedicated legalization stage is not optional; it is the mechanism that converts high-legality samples into strict zero-overlap placements required by the competition.

### Step 3: Multi-Objective Proxy-to-PPA Bridge

The docs treat wirelength, congestion, and legality as distinct potentials during guided sampling and frame inference as a Pareto-style exploration over weight settings. Final selection is then performed with the exact official proxy formula after legalization.

Methodological principle:

- Use inference-time diversity to generate robust candidates.
- Use official proxy for submission selection to maximize tier-1 advancement probability.
- Treat tier-2 PPA outcomes as downstream validation rather than a directly optimized in-loop objective inside the benchmark-time loop.

## Why This Architecture

The chosen direction is based on constraint-method alignment:

- **Need for parallel candidate generation** under runtime limits favors simultaneous generative sampling.
- **Need for hard legality** requires an explicit legalizer when generation is continuous.
- **Need to survive tier-1 ranking** favors strict, evaluator-aligned candidate selection using official proxy definitions.

In other words, this architecture separates concerns deliberately:

- generation (diversity and structure),
- legality (hard feasibility),
- ranking (competition metric alignment).

## Evaluation Logic and Decision Flow

The repository's analytical stance is that optimization-time surrogates and competition-time metrics should be separated but connected by a clear selection policy.

### Surrogate vs Official Metrics

- **During sampling/guidance**: use fast differentiable surrogates (for tractable optimization).
- **After legalization**: compute official proxy for all candidates and choose by competition metric.
- **For tier-2 expectations**: track hypotheses relating proxy choices to possible PPA outcomes, but do not assume monotonicity without measured evidence.

### Intended End-to-End Selection Loop

1. Generate multiple placement candidates with varied objective weights.
2. Legalize each candidate to zero overlap.
3. Evaluate all legalized candidates with official proxy definitions.
4. Keep the single best proxy candidate for submission/evaluation.

### Runtime and Budget Strategy

- Hard cap is one hour per benchmark end-to-end.
- Use GPU-parallel diffusion sampling for candidate diversity.
- Spend remaining wall-clock budget on legalization + official proxy evaluation across the candidate batch.

## Repository Map

**Code (Phase 0–2):**

- [`src/hrt_chip/`](src/hrt_chip/) — package: CLI, pipeline, stages, [`diffusion.py`](src/hrt_chip/diffusion.py) (sampler contract + DDPM stub), adapters, artifacts.
- [`docs/integration-notes.md`](docs/integration-notes.md) — evaluator and mixed-size integration contracts.
- [`external/`](external/) — reserved for official challenge submodule (see integration notes).

**Documentation:** The key documents are:

- [`docs/proposed-solution-overview.md`](docs/proposed-solution-overview.md)  
  High-level synthesis of the full approach.
- [`docs/step1-diffusion-model.md`](docs/step1-diffusion-model.md)  
  Detailed rationale for diffusion-based placement generation.
- [`docs/step2-position-masking.md`](docs/step2-position-masking.md)  
  Legality discussion: sequential masking vs diffusion guidance + legalization.
- [`docs/step3-multi-objective-proxy-to-ppa.md`](docs/step3-multi-objective-proxy-to-ppa.md)  
  Multi-objective inference strategy and proxy-to-PPA reasoning.
- [`docs/macro-placement-competition.md`](docs/macro-placement-competition.md)  
  Practical competition constraints and targets.
- [`docs/README of Competition.md`](docs/README%20of%20Competition.md)  
  Full challenge statement, rules, prizes, and baseline context.
- [`docs/implementation-roadmap.md`](docs/implementation-roadmap.md)  
  Concrete phase-by-phase execution order for implementation.

## Current Project Status

Phase 0 scaffolding, **Phase 1 legality baseline**, **Phase 2 diffusion inference skeleton**, **Phase 3 guided objectives / Pareto-style batch selection**, **Phase 4 synthetic data + PyTorch/PyG DDPM training**, **Phase 5 benchmark harness + milestone gates**, and **Phase 6 reproducibility / regression controls** are in place: a **`uv`-managed Python package** with a CLI that runs **generate → legalize → mixed-size → evaluate**. Generation uses a **DDPM sampler interface** with a **deterministic stub** (default) or a **trained PyTorch checkpoint** (`--sampler-backend pytorch_checkpoint --checkpoint …`) that emits normalized centers in **`[-1, 1]`** (stored per candidate as `metadata["normalized_centers"]`), mapped to the physical or unit canvas for [`MacroRect`](src/hrt_chip/models.py) before the greedy legalizer. **Phase 3** adds optional **multi-weight inference sweeps** (`--guidance-preset pareto3` or repeated `--guidance-weight a,b,c`), per-candidate **surrogate objective** fields (HPWL/congestion/legality stubs in [`src/hrt_chip/guidance.py`](src/hrt_chip/guidance.py)), and a **`scoring_table`** in `results.json`; the **best candidate is always the argmin of the evaluator proxy** (stub or official). **Phase 5** adds **`hrt-chip benchmark-sweep`** over all 17 IBM designs (`ibm01`–`ibm04`, `ibm06`–`ibm18`), aggregate proxy vs **SA (2.1251)** and **RePlAce (1.4578)** baselines, and **Gate A/B/C** reporting; sweep artifacts include `sweep_report.json`. Structured artifacts include `manifest.json`, `results.json` (with `sampler_provenance`, `guidance_sweep_resolved`, `scoring_table`, optional `checkpoint_path` / `training_dataset_version`), and per-candidate JSON (optional pruning via **artifact retention**). **Phase 6** adds centralized RNG seeding (`deterministic_runtime`), optional **`--deterministic-verification`** (stricter PyTorch/cuDNN), **`hrt-chip replay … --verify`** (writes `replay_verification.json`), and **GitHub Actions** CI (pytest + stub replay + small benchmark subset). Illegal placements skip mixed-size handoff and receive infinite proxy from the evaluator.

### What Exists

- Architecture rationale and method documentation.
- Constraint analysis and evaluation framing.
- Cross-step conceptual pipeline.
- Runnable package [`src/hrt_chip/`](src/hrt_chip/) with CLI `hrt-chip` and module entrypoint `python -m hrt_chip`.
- Diffusion sampler contract + stub ([`src/hrt_chip/diffusion.py`](src/hrt_chip/diffusion.py)), batched candidate generation, and guardrail tests ([`tests/test_diffusion_guardrails.py`](tests/test_diffusion_guardrails.py)).
- Phase 3 guidance sweep, surrogate objectives, scoring table, strict proxy selection ([`src/hrt_chip/guidance.py`](src/hrt_chip/guidance.py), [`tests/test_phase3_guidance.py`](tests/test_phase3_guidance.py)).
- Adapter contracts for evaluator and mixed-size backend ([`docs/integration-notes.md`](docs/integration-notes.md)).
- Phase 5 IBM sweep + gates: [`src/hrt_chip/benchmarks.py`](src/hrt_chip/benchmarks.py), [`src/hrt_chip/benchmark_sweep.py`](src/hrt_chip/benchmark_sweep.py), [`src/hrt_chip/adapters/evaluator/official.py`](src/hrt_chip/adapters/evaluator/official.py); tests in [`tests/test_phase5_benchmarks.py`](tests/test_phase5_benchmarks.py).
- Phase 6 reproducibility: [`src/hrt_chip/deterministic_runtime.py`](src/hrt_chip/deterministic_runtime.py), [`src/hrt_chip/replay_verify.py`](src/hrt_chip/replay_verify.py), retention helpers in [`src/hrt_chip/io/artifacts.py`](src/hrt_chip/io/artifacts.py); tests in [`tests/test_phase6.py`](tests/test_phase6.py); workflow [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

### What Is Still Stubbed / Planned

- Further scaling of synthetic data and model quality toward competition benchmarks (beyond Phase 4 scaffolding).
- Differentiable DDPM guidance on ε-prediction / x̂₀ (Phase 4+; Phase 3 provides sweep + surrogate scoring + selection policy).
- DREAMPlace/hMETIS wiring behind mixed-size adapter ([`docs/integration-notes.md`](docs/integration-notes.md)).

## Environment and How to Run

**Dependency manager:** [`uv`](https://docs.astral.sh/uv/). Install dependencies and sync the lockfile:

```bash
uv sync
# include dev deps (pytest) for local testing / CI parity
uv sync --group dev
```

Run the end-to-end stub pipeline (example: benchmark `ibm01`, 4 candidates, fixed seed):

```bash
uv run hrt-chip run --benchmark ibm01 --seed 42 --candidates 4 --output-dir runs
```

**Evaluator backend:** default is **`--evaluator stub`** (deterministic hash proxy). For the **official** tier-1 proxy (TILOS `PlacementCost` via `macro_place`), install the [Partcl macro-place challenge](https://github.com/partcleda/partcl-macro-place-challenge) package (`pip install -e .` from a clone), initialize **`external/MacroPlacement`**, then run with `--evaluator official` and optionally `--testcase-root path/to/ICCAD04`.

Optional: `--diffusion-steps` (default `1000`) is recorded in the manifest and sampler provenance for forward compatibility with the real DDPM schedule.

**Phase 3 — guidance sweep** (multiple weight vectors; `num_candidates` is per vector):

```bash
uv run hrt-chip run --benchmark ibm01 --seed 42 --candidates 2 --guidance-preset pareto3 --output-dir runs
```

Custom weights (repeat `--guidance-weight` for each `(α, β, γ)` triple; overrides preset):

```bash
uv run hrt-chip run --benchmark ibm01 --guidance-weight 0.8,0.1,0.1 --guidance-weight 0.2,0.7,0.1 --candidates 2
```

`results.json` includes `guidance_sweep_resolved`, `scoring_table` (surrogate + proxy), and `ranking` / `best_candidate_id` by official proxy only.

**Phase 4 — synthetic dataset, training, trained sampler inference:**

```bash
# Generate synthetic layouts (v1 = smaller graphs, v2 = larger); writes dataset_manifest.json + shards
uv run hrt-chip dataset-generate --output-dir data/synthetic/v1 --corpus v1 --num-samples 256

# Train ε-prediction DDPM (writes checkpoint.pt + training_manifest.json under training_runs/<id>/)
uv run hrt-chip train --dataset-dir data/synthetic/v1 --epochs 10 --model-architecture baseline_gnn

# Run pipeline using a trained checkpoint
uv run hrt-chip run --benchmark ibm01 --sampler-backend pytorch_checkpoint --checkpoint training_runs/<id>/checkpoint.pt
```

**Phase 5 — full IBM suite (17 benchmarks) and milestone gates:**

```bash
# Full sweep; default --evaluator official (requires macro_place + MacroPlacement testcases)
uv run hrt-chip benchmark-sweep --output-dir runs/sweeps --seed 42 --candidates 2

# Fast local / CI without official testcase tree
uv run hrt-chip benchmark-sweep --evaluator stub --output-dir runs/sweeps_stub

# Subset of benchmarks (repeat --benchmark); gates use only rows in this sweep
uv run hrt-chip benchmark-sweep --evaluator stub --benchmark ibm01 --benchmark ibm02 --candidates 1
```

Writes `runs/sweeps/<sweep_id>/sweep_report.json` and per-benchmark run dirs. Prints **Gate A** (100% legal), **Gate B** (mean proxy lower than aggregate SA **2.1251**), **Gate C** (mean proxy lower than aggregate RePlAce **1.4578**).

Equivalent:

```bash
uv run python -m hrt_chip run --benchmark ibm01
```

Artifacts are written under `runs/<run_id>/`: `manifest.json`, `results.json`, and `candidates/*.json`.

Re-run from a saved manifest (reproducibility):

```bash
uv run hrt-chip replay runs/<run_id>/manifest.json
```

**Phase 6 — deterministic verification and artifact retention**

```bash
# Strict PyTorch/cuDNN determinism (optional; slower)
uv run hrt-chip run --benchmark ibm01 --seed 42 --deterministic-verification --output-dir runs

# After a run, replay and assert ranking/proxy/best match prior results.json (exit 1 on mismatch)
uv run hrt-chip replay runs/<run_id>/manifest.json --verify

# Prune per-candidate JSON on disk (full ranking remains in results.json)
uv run hrt-chip run --benchmark ibm01 --artifact-retention compact --output-dir runs
uv run hrt-chip run --benchmark ibm01 --artifact-retention compact --artifact-retention-top-k 2 --output-dir runs
uv run hrt-chip run --benchmark ibm01 --artifact-retention best_only --output-dir runs
```

**CI:** on push/PR, `.github/workflows/ci.yml` runs `uv sync --group dev`, `pytest`, stub `run` + `replay --verify`, and a two-benchmark stub sweep.

### AWS / local secrets (when using cloud evaluators)

If your workflow assumes an AWS role and env files (typical for this team):

1. Assume role: `source env-assume-role.sh`
2. Load variables: `source env.sh`

(On Windows, use Git Bash/WSL or translate these to your shell.)

### Official challenge repo submodule (optional)

Reserve directory: [`external/`](external/). When you add the official challenge repository:

```bash
git submodule add <challenge-repo-url> external/challenge
```

Details: [`docs/integration-notes.md`](docs/integration-notes.md).

### Reproducibility Controls (Project Requirement)

This project treats reproducibility as mandatory:

- Lock random seeds for training and inference (stub generation uses `--seed`; pipeline calls [`deterministic_runtime`](src/hrt_chip/deterministic_runtime.py) when `deterministic` is on).
- Each run writes `manifest.json` with config snapshot, run id, UTC timestamp, and Phase 6 `deterministic_verification` flag.
- `hrt-chip replay` re-executes from a saved manifest; **`--verify`** compares to the previous `results.json` and writes `replay_verification.json`.
- Optional **`--artifact-retention`** reduces on-disk candidate JSON while keeping full scores in `results.json`.
- CI runs pytest plus end-to-end stub checks (see `.github/workflows/ci.yml`).

## Next Milestones (Suggested)

1. ~~Create baseline project scaffold (`pyproject.toml`, package layout, CLI entrypoint) using `uv`.~~
2. ~~Harden legality checker + greedy legalizer with explicit zero-overlap assertions (Phase 1).~~
3. ~~Diffusion inference skeleton: sampler contract, batched stub, provenance, guardrail tests (Phase 2).~~
4. ~~Guided objectives + strict proxy selection policy (Phase 3).~~
5. ~~Synthetic data generation, PyTorch/PyG DDPM training, and checkpoint-based sampler inference (Phase 4).~~
6. ~~Experiment harness for 17 IBM benchmark sweeps, gate reporting, and `sweep_report.json` (Phase 5).~~
7. ~~Reproducibility: deterministic verification, replay `--verify`, artifact retention, CI smoke (Phase 6).~~
8. Add NG45-oriented handoff format/export path for downstream validation.

## References

Primary references and context are documented in:

- [`docs/README of Competition.md`](docs/README%20of%20Competition.md)
- [`docs/macro-placement-competition.md`](docs/macro-placement-competition.md)
- [`docs/step1-diffusion-model.md`](docs/step1-diffusion-model.md)
- [`docs/step2-position-masking.md`](docs/step2-position-masking.md)
- [`docs/step3-multi-objective-proxy-to-ppa.md`](docs/step3-multi-objective-proxy-to-ppa.md)

For academic sources and baseline comparisons, see the links collected in the competition docs above.

