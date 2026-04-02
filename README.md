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

This repository is currently docs-centric. The key documents are:

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

## Current Project Status

This is a **research/design documentation phase** repository.

### What Exists

- Architecture rationale and method documentation.
- Constraint analysis and evaluation framing.
- Cross-step conceptual pipeline.

### What Does Not Yet Exist

- Executable training/inference code.
- Official benchmark runner implementation in this repo.
- Reproducible experiment scripts and result artifacts.

The README intentionally avoids claiming implementation completeness.

## Environment and Reproducibility Notes

No active Python project scaffold is present yet in the repository root (for example, no `pyproject.toml` or `uv.lock` at this stage). As a result:

- There is no canonical install command yet.
- There are no runnable project commands yet.
- Setup guidance in this README is intentionally deferred until code scaffolding is added.

When implementation begins, the environment section should be updated with:

- dependency manager choice (`uv`),
- exact environment bootstrapping steps,
- benchmark execution and evaluation commands.
- a single end-to-end entrypoint command for generate -> legalize -> evaluate.

### Reproducibility Controls (Project Requirement)

This project treats reproducibility as mandatory:

- lock random seeds for training and inference;
- support deterministic compute mode for debugging and result verification;
- snapshot run configs for every experiment;
- log artifacts/metrics needed to reproduce benchmark outcomes and candidate selection.

## Next Milestones (Suggested)

1. Create baseline project scaffold (`pyproject.toml`, package layout, CLI entrypoint) using `uv`.
2. Implement legality checker + greedy legalizer with explicit zero-overlap assertions.
3. Implement diffusion sampling prototype with deterministic/debug reproducibility controls.
4. Add batch candidate evaluator using official proxy-compatible interfaces and best-proxy auto-selection.
5. Add experiment harness for 17 IBM benchmark sweeps and structured result logging.
6. Add NG45-oriented handoff format/export path for downstream validation.

## References

Primary references and context are documented in:

- [`docs/README of Competition.md`](docs/README%20of%20Competition.md)
- [`docs/macro-placement-competition.md`](docs/macro-placement-competition.md)
- [`docs/step1-diffusion-model.md`](docs/step1-diffusion-model.md)
- [`docs/step2-position-masking.md`](docs/step2-position-masking.md)
- [`docs/step3-multi-objective-proxy-to-ppa.md`](docs/step3-multi-objective-proxy-to-ppa.md)

For academic sources and baseline comparisons, see the links collected in the competition docs above.

