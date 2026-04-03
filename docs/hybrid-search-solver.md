# Hybrid search-centric solver (canonical contract)

This document locks the **production** macro-placement path for competition runs. The legacy diffusion-primary path remains available for regression and comparison.

## Canonical pipeline

```text
initialize (multi-family seeds) → legalize (repair only if needed) → local search (SA, legality-enforced moves) → mixed-size handoff → official proxy evaluate → rank (proxy_first)
```

- **Initialize:** Build diverse legal or near-legal starting placements from configured families (benchmark jitter, optional diffusion/stub generation, random legal).
- **Local search:** Simulated annealing optimizes a fast objective (HPWL-first early, then full surrogate aligned with proxy weights). Moves that create overlaps are rejected. Adaptive operator selection biases toward operators that produce accepted improving moves.
- **Evaluate:** Tier-1 selection uses **official proxy cost** only (`selection_policy=proxy_first`).

## Legacy pipeline (frozen default for CI)

```text
generate (diffusion contract) → greedy legalize → mixed-size → evaluate
```

Set `solver_backend=legacy` (default) to preserve existing behavior and replay manifests.

## Diffusion role

Diffusion is **one initializer source** when `search_families` includes `diffusion`. It is not the primary optimizer. Training/sampler architecture work is deprioritized unless it improves downstream search outcomes.

## Configuration

- `solver_backend`: `legacy` | `search_hybrid`
- Search and seed options are documented in `RunConfig` (`src/hrt_chip/config.py`) and CLI (`hrt-chip run --help`).

## Week-1 evidence gates (operational)

See competition runbooks and sweep reports. Targets from the pivot plan:

- Aggregate proxy **&lt; 1.9** on the agreed Week-1 benchmark suite, **or** within **30% of RePlAce** on at least **5** benchmarks.
- Do not treat marginal improvements (e.g. 2.1 vs 2.21 greedy baseline) as sufficient signal.

## Feature freeze (Week 5+)

After Week 5, avoid new algorithm features; prioritize experiment throughput, parameter tuning, and submission selection.
