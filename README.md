# HRT_Chip

**Diffusion-first macro placement** for the Partcl × Hudson River Trading (HRT) challenge: a **`uv`-managed Python package** with a runnable **`hrt-chip` CLI**, end-to-end pipeline (**generate → legalize → mixed-size → evaluate**), optional **official ICCAD04 proxy** evaluation, **IBM benchmark sweeps** with milestone gates, and **Phase 6** reproducibility (manifests, replay verification, CI).

Research rationale and competition constraints live in [`docs/`](docs/); this README focuses on **what is implemented today** and how to run it.

## Quickstart

**Requirements:** Python **3.11+** ([`pyproject.toml`](pyproject.toml)). Use **[uv](https://docs.astral.sh/uv/)** — dependencies are lockfile-managed ([`uv.lock`](uv.lock)).

```bash
uv sync --group dev          # install + dev deps (pytest; matches CI)
uv run pytest                # unit + integration tests

# End-to-end stub pipeline (no external testcase tree)
uv run hrt-chip run --benchmark ibm01 --seed 42 --candidates 4 --output-dir runs

# Same as: uv run python -m hrt_chip run --benchmark ibm01 ...
```

**CI parity** (see [`.github/workflows/ci.yml`](.github/workflows/ci.yml)):

```bash
SMOKE_ROOT=/tmp/hrt_chip_smoke   # or $env:TEMP\hrt_chip_smoke on PowerShell
mkdir -p "$SMOKE_ROOT"           # PowerShell: New-Item -ItemType Directory -Force
uv run hrt-chip run --benchmark ibm01 --seed 1 --candidates 2 \
  --output-dir "$SMOKE_ROOT" --run-id phase6-smoke-run
uv run hrt-chip replay "$SMOKE_ROOT/phase6-smoke-run/manifest.json" --verify

uv run hrt-chip benchmark-sweep --evaluator stub --benchmark ibm01 --benchmark ibm02 \
  --candidates 1 --output-dir /tmp/hrt_sweep_smoke --sweep-id ci_subset
```

## Project thesis (short)

1. Generate candidate macro coordinates with a **joint** diffusion-style process (stub or trained checkpoint).
2. Enforce **hard legality** via overlap-aware signals plus a **greedy legalizer** (continuous geometry).
3. Explore multi-objective trade-offs at inference (guidance weight sweep + surrogate **scoring table**); **Tier-1 selection** is always **argmin official proxy** after legalization.

Full methodology: [`docs/proposed-solution-overview.md`](docs/proposed-solution-overview.md), [`docs/step1-diffusion-model.md`](docs/step1-diffusion-model.md), [`docs/step2-position-masking.md`](docs/step2-position-masking.md), [`docs/step3-multi-objective-proxy-to-ppa.md`](docs/step3-multi-objective-proxy-to-ppa.md).

## Competition context (hard constraints)

- **Zero overlap** under the official checker; illegal runs get infinite proxy in-pipeline.
- **~1 hour** end-to-end wall time per benchmark (competition target).
- **Tier 1:** proxy cost on IBM ICCAD04-style designs; **Tier 2:** OpenROAD PPA on NG45 for top entries.
- **No benchmark-specific hardcoding** in the competition spirit.

Practical rules summary: [`docs/macro-placement-competition.md`](docs/macro-placement-competition.md), [`docs/README of Competition.md`](docs/README%20of%20Competition.md).

## What is implemented (Phases 0–6)

| Area | Summary |
|------|---------|
| **Pipeline** | `run_pipeline` in [`src/hrt_chip/pipeline.py`](src/hrt_chip/pipeline.py): generate → legalize → mixed-size stub → evaluate; **best_candidate_id = argmin(proxy_score)**. |
| **Sampling** | `DiffusionSampler` + `DeterministicDDPMStubSampler`; normalized centers **`[-1, 1]`**; optional **`pytorch_checkpoint`** ([`adapters/diffusion/pytorch_sampler.py`](src/hrt_chip/adapters/diffusion/pytorch_sampler.py)). |
| **Legality** | Geometry + greedy legalizer ([`geometry.py`](src/hrt_chip/geometry.py), [`stages/legalize.py`](src/hrt_chip/stages/legalize.py)). |
| **Guidance (Phase 3)** | Presets / `--guidance-weight` sweep, surrogates + `scoring_table` in `results.json` ([`guidance.py`](src/hrt_chip/guidance.py)). |
| **Data & training (Phase 4)** | `dataset-generate`, `train` (ε-prediction DDPM), manifests + checkpoints ([`data/`](src/hrt_chip/data/), [`training/`](src/hrt_chip/training/)). |
| **Benchmarks (Phase 5)** | `benchmark-sweep` over **17 IBM** designs, `sweep_report.json`, Gate A/B/C vs aggregate SA / RePlAce ([`benchmarks.py`](src/hrt_chip/benchmarks.py), [`benchmark_sweep.py`](src/hrt_chip/benchmark_sweep.py)). |
| **Repro (Phase 6)** | Deterministic seeding, `--deterministic-verification`, `replay --verify` → `replay_verification.json`, artifact retention ([`deterministic_runtime.py`](src/hrt_chip/deterministic_runtime.py), [`replay_verify.py`](src/hrt_chip/replay_verify.py), [`io/artifacts.py`](src/hrt_chip/io/artifacts.py)). |

**Adapters:** evaluator **stub** (default) or **official** ([`adapters/evaluator/`](src/hrt_chip/adapters/evaluator/)); mixed-size **stub** only today ([`adapters/mixed_size/`](src/hrt_chip/adapters/mixed_size/)). Integration details: [`docs/integration-notes.md`](docs/integration-notes.md).

## CLI commands

All entrypoints: **`hrt-chip`** ([`src/hrt_chip/cli.py`](src/hrt_chip/cli.py)) or **`python -m hrt_chip`**.

| Command | Purpose |
|---------|---------|
| `run` | Full pipeline for one benchmark; writes `runs/<run_id>/manifest.json`, `results.json`, `candidates/*.json`. |
| `replay` | Re-run from `manifest.json`; `--verify` compares to prior `results.json`. |
| `dataset-generate` | Synthetic layouts + PyG shards + `dataset_manifest.json`. |
| `train` | DDPM training; writes `checkpoint.pt` + training manifest under `training_runs/<id>/`. |
| `benchmark-sweep` | All 17 IBM benchmarks (or `--benchmark` subset); default `--evaluator official` (use **stub** for CI/local without testcases). |

### `run` — common flags

- **`--sampler-backend stub`** (default) or **`pytorch_checkpoint`** + **`--checkpoint path/to/checkpoint.pt`**
- **`--evaluator stub`** (default) or **`official`** + testcase tree (see below)
- **`--guidance-preset pareto3`** or repeat **`--guidance-weight α,β,γ`** (HPWL, congestion, legality surrogates); `num_candidates` is **per** weight vector
- **Phase 6:** `--deterministic-verification`, `--artifact-retention full|compact|best_only`, `--artifact-retention-top-k K`

```bash
uv run hrt-chip run --benchmark ibm01 --seed 42 --candidates 2 \
  --guidance-preset pareto3 --output-dir runs

uv run hrt-chip run --benchmark ibm01 --guidance-weight 0.8,0.1,0.1 --guidance-weight 0.2,0.7,0.1 \
  --candidates 2 --output-dir runs
```

### Phase 4 — dataset, train, inference

```bash
uv run hrt-chip dataset-generate --output-dir data/synthetic/v1 --corpus v1 --num-samples 256

uv run hrt-chip train --dataset-dir data/synthetic/v1 --epochs 10 --model-architecture baseline_gnn

uv run hrt-chip run --benchmark ibm01 --sampler-backend pytorch_checkpoint \
  --checkpoint training_runs/<id>/checkpoint.pt
```

Architectures: `baseline_gnn`, `res_gnn`, `att_gnn`.

### Phase 5 — benchmark sweep

```bash
# Full sweep; needs official stack + ICCAD04 testcases (default --evaluator official)
uv run hrt-chip benchmark-sweep --output-dir runs/sweeps --seed 42 --candidates 2

# Fast local / CI (stub proxy)
uv run hrt-chip benchmark-sweep --evaluator stub --output-dir runs/sweeps_stub

# Subset + gates computed only on included rows
uv run hrt-chip benchmark-sweep --evaluator stub --benchmark ibm01 --benchmark ibm02 --candidates 1
```

Artifacts: `runs/sweeps/<sweep_id>/sweep_report.json` plus per-benchmark run directories. Console reports **Gate A** (100% legal), **Gate B** (mean proxy vs aggregate SA **2.1251**), **Gate C** (vs RePlAce **1.4578**).

### Phase 6 — verification & retention

```bash
uv run hrt-chip run --benchmark ibm01 --seed 42 --deterministic-verification --output-dir runs
uv run hrt-chip replay runs/<run_id>/manifest.json --verify

uv run hrt-chip run --benchmark ibm01 --artifact-retention compact --output-dir runs
uv run hrt-chip run --benchmark ibm01 --artifact-retention compact --artifact-retention-top-k 2 --output-dir runs
uv run hrt-chip run --benchmark ibm01 --artifact-retention best_only --output-dir runs
```

`--diffusion-steps` (default `1000`) is recorded for provenance; stub uses it for metadata compatibility with a real DDPM schedule.

### Competition hardening (surrogates, budget, trends)

- **`results.json` schema:** `results_schema_version` and required keys are documented in [`docs/baseline-artifacts.md`](docs/baseline-artifacts.md); validate with [`src/hrt_chip/io/baseline_schema.py`](src/hrt_chip/io/baseline_schema.py).
- **Surrogates:** with `--evaluator official`, scoring-table HPWL/congestion use **LogSumExp net HPWL** and **RUDY-style** demand when a `macro_place` `Benchmark` is loaded; stub runs keep bbox/grid fallbacks.
- **Mixed-size:** default **`--mixed-size-backend estimate`** (macro utilization + RUDY proxy + timing); use **`stub`** for no-op CI/smoke.
- **Budget:** optional **`--wall-clock-budget-seconds`** shrinks per-vector candidate count or the guidance sweep (see [`src/hrt_chip/budget.py`](src/hrt_chip/budget.py)).
- **Accelerated DDPM inference:** **`--diffusion-inference-steps`** (with `--sampler-backend pytorch_checkpoint`) subsamples reverse timesteps.
- **Sweep trends:** each **`benchmark-sweep`** appends a JSON line to **`runs/trends/sweep_history.jsonl`** (override with **`--trends-log-path`**).
- **Synthetic curriculum:** **`hrt-chip dataset-generate --curriculum benchmark_like`** uses heavy-tail sizes, random legal packing, and **6-D** node features (WH + spatial degree); default remains **`grid_v1`**.

## Official evaluator (optional)

For **Tier-1 proxy** via TILOS / `macro_place` (same stack as the challenge `evaluate` flow):

1. Clone and install [partcl-macro-place-challenge](https://github.com/partcleda/partcl-macro-place-challenge): `pip install -e .` or `uv pip install -e .`.
2. Initialize testcases: e.g. `git submodule update --init external/MacroPlacement` so ICCAD04 lives under `external/MacroPlacement/Testcases/ICCAD04/`.
3. Run: `uv run hrt-chip run --benchmark ibm01 --evaluator official` (optional `--testcase-root` or env **`HRT_CHIP_TESTCASE_ROOT`**).

See [`docs/integration-notes.md`](docs/integration-notes.md) for contracts and defaults (`default_testcase_root()` in [`benchmarks.py`](src/hrt_chip/benchmarks.py)).

## Repository map

| Path | Role |
|------|------|
| [`src/hrt_chip/`](src/hrt_chip/) | Package: CLI, config, pipeline, stages (`generate`, `legalize`, `evaluate`), diffusion, guidance, geometry, benchmarks, training, data, I/O artifacts. |
| [`src/hrt_chip/adapters/`](src/hrt_chip/adapters/) | Evaluator (stub / official), diffusion (PyTorch), mixed-size (stub / estimate). |
| [`tests/`](tests/) | Pytest: smoke, diffusion guardrails, geometry, Phase 3–6. |
| [`docs/`](docs/) | Design, roadmap, integration, competition notes. |
| [`external/`](external/) | Reserved for **MacroPlacement** / challenge assets (see [`external/.gitkeep`](external/.gitkeep)); optional git submodule. |

**Implementation order / phase checklist:** [`docs/implementation-roadmap.md`](docs/implementation-roadmap.md).

## Verification

- **Tests:** `uv run pytest` — [`tests/test_pipeline_smoke.py`](tests/test_pipeline_smoke.py), [`tests/test_diffusion_guardrails.py`](tests/test_diffusion_guardrails.py), [`tests/test_geometry_normalized.py`](tests/test_geometry_normalized.py), [`tests/test_phase3_guidance.py`](tests/test_phase3_guidance.py), [`tests/test_phase4.py`](tests/test_phase4.py), [`tests/test_phase5_benchmarks.py`](tests/test_phase5_benchmarks.py), [`tests/test_phase6.py`](tests/test_phase6.py).
- **CI:** [`.github/workflows/ci.yml`](.github/workflows/ci.yml) — `uv sync --group dev`, `pytest`, stub `run` + `replay --verify`, [`tools/check_stub_sweep_regression.py`](tools/check_stub_sweep_regression.py) (Gate A + trends JSONL).

## Environment notes

### AWS / secrets (not in repo)

If your workflow uses AWS role assumption and env files:

1. `source env-assume-role.sh`
2. `source env.sh`

These scripts are **local** to your machine (not committed). On **Windows**, use Git Bash, WSL, or equivalent.

### Optional challenge submodule

```bash
git submodule add <challenge-repo-url> external/challenge
```

Details: [`docs/integration-notes.md`](docs/integration-notes.md).

## Reproducibility (project expectation)

- Lock seeds via `--seed` and `RunConfig.deterministic` (default on for CLI runs).
- Each run writes **`manifest.json`** (config snapshot, run id, UTC time, `deterministic_verification` when set).
- **`hrt-chip replay`** re-executes from manifest; **`--verify`** fingerprints ranking/proxy/best vs prior **`results.json`** and writes **`replay_verification.json`**.
- **`--artifact-retention`** reduces `candidates/*.json` on disk; **`results.json`** keeps full ranking and `scoring_table`.

## Roadmap — what is still open

- **Mixed-size / standard cells:** wire a real placer (e.g. DREAMPlace / hMETIS) behind [`MixedSizeBackend`](src/hrt_chip/adapters/mixed_size/base.py); **estimate** backend is analytical-only.
- **Model & data:** scale synthetic data and training quality toward strong competition proxies; **differentiable DDPM guidance** on ε / x̂₀ (Phase 3 today is sweep + surrogates + proxy selection only).
- **Handoff / Tier 2:** NG45-oriented export or validation path for downstream OpenROAD-style flows.

Completed baseline milestones (scaffold through Phase 6 harness + CI) are tracked in [`docs/implementation-roadmap.md`](docs/implementation-roadmap.md).

## References

- [`docs/README of Competition.md`](docs/README%20of%20Competition.md)
- [`docs/macro-placement-competition.md`](docs/macro-placement-competition.md)
- [`docs/step1-diffusion-model.md`](docs/step1-diffusion-model.md)
- [`docs/step2-position-masking.md`](docs/step2-position-masking.md)
- [`docs/step3-multi-objective-proxy-to-ppa.md`](docs/step3-multi-objective-proxy-to-ppa.md)
