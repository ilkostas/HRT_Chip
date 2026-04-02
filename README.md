# HRT_Chip

Diffusion-first macro placement pipeline for the Partcl x HRT challenge.

## Snapshot

- `hrt-chip` CLI runs an end-to-end flow: generate -> legalize -> mixed-size -> evaluate.
- Default local path is fully runnable with `uv` + stub evaluator (no testcase tree required).
- Optional official evaluator path supports ICCAD04 proxy scoring with challenge assets installed.
- Benchmark harness supports all 17 IBM circuits and milestone gate reporting.
- Reproducibility is built in: manifests, replay verification, deterministic controls, CI smoke.

Research and competition context live in [`docs/`](docs/); this README focuses on running and developing the implemented system.

## Quickstart

Requirements:

- Python 3.11+ (see [`pyproject.toml`](pyproject.toml))
- [`uv`](https://docs.astral.sh/uv/) dependency manager

```bash
# Install package + development dependencies (matches CI)
uv sync --group dev

# Run test suite
uv run pytest

# First pipeline run (stub evaluator, no external testcase tree)
uv run hrt-chip run --benchmark ibm01 --seed 42 --candidates 4 --output-dir runs
```

### Environment bootstrap (AWS / secrets)

If your local workflow requires role assumption and env loading:

```bash
source env-assume-role.sh
source env.sh
```

On Windows, run those commands in Git Bash, WSL, or equivalent POSIX shell.

## Core workflows

### 1) Single benchmark run

```bash
uv run hrt-chip run --benchmark ibm01 --seed 42 --candidates 2 --output-dir runs
```

Useful flags:

- `--evaluator stub|official`
- `--mixed-size-backend stub|estimate|dreamplace|dreamplace_real`
- `--selection-policy proxy_first|ppa_priority`
- `--guidance-preset pareto3` or repeated `--guidance-weight a,b,c`

### 2) Replay + verify determinism

```bash
uv run hrt-chip replay runs/<run_id>/manifest.json --verify
```

### 3) Sweep benchmarks

```bash
# Local/CI smoke path (no external challenge testcase setup required)
uv run hrt-chip benchmark-sweep --evaluator stub --benchmark ibm01 --benchmark ibm02 --candidates 1 --output-dir runs/sweeps_stub

# Full sweep (defaults to official evaluator; requires official setup)
uv run hrt-chip benchmark-sweep --output-dir runs/sweeps --seed 42 --candidates 2
```

## Backends and integration

Evaluator backends:

- `stub`: default local path, quick iteration, CI-compatible.
- `official`: uses challenge `macro_place` stack and ICCAD04 testcase assets.

Mixed-size backends:

- `estimate`: default analytical proxy backend.
- `stub`: no-op backend for smoke tests.
- `dreamplace` / `dreamplace_real`: Docker-backed integration path.

For detailed contracts, environment variables, Docker image build scripts, and troubleshooting, see [`docs/integration-notes.md`](docs/integration-notes.md).

## Artifacts and reproducibility

Each run writes artifacts under `runs/<run_id>/`, including:

- `manifest.json`: config snapshot, run metadata, reproducibility controls.
- `results.json`: ranking table, scores, and stage-level outputs.
- `candidates/*.json`: candidate-level artifacts (retention policy dependent).
- `replay_verification.json`: produced by `replay --verify`.

Common reproducibility controls:

```bash
uv run hrt-chip run --benchmark ibm01 --seed 42 --deterministic-verification --output-dir runs
uv run hrt-chip run --benchmark ibm01 --artifact-retention compact --artifact-retention-top-k 2 --output-dir runs
```

Schema details for baseline artifacts are documented in [`docs/baseline-artifacts.md`](docs/baseline-artifacts.md).

## Official evaluator setup (optional)

To run with `--evaluator official`:

1. Clone and install [`partcl-macro-place-challenge`](https://github.com/partcleda/partcl-macro-place-challenge) (for `macro_place` imports).
2. Initialize testcase assets so ICCAD04 data is available.
3. Run:

```bash
uv run hrt-chip run --benchmark ibm01 --evaluator official
```

You can also pass `--testcase-root` or set `HRT_CHIP_TESTCASE_ROOT`.

## Repository map

- [`src/hrt_chip/`](src/hrt_chip/): package code (CLI, pipeline, adapters, stages, training, I/O).
- [`tests/`](tests/): smoke, geometry, guidance, phase 4/5/6 verification.
- [`docs/`](docs/): design docs, roadmap, competition notes, integration contracts.
- [`external/`](external/): reserved path for optional challenge assets/submodules.

## Documentation index

- Implementation status and phased build order: [`docs/implementation-roadmap.md`](docs/implementation-roadmap.md)
- Diffusion-first approach overview: [`docs/proposed-solution-overview.md`](docs/proposed-solution-overview.md)
- Legality strategy: [`docs/step2-position-masking.md`](docs/step2-position-masking.md)
- Multi-objective proxy/selection details: [`docs/step3-multi-objective-proxy-to-ppa.md`](docs/step3-multi-objective-proxy-to-ppa.md)
- Competition rule context: [`docs/macro-placement-competition.md`](docs/macro-placement-competition.md), [`docs/README of Competition.md`](docs/README%20of%20Competition.md)

## Current limitations

- Production mixed-size flow is still backend-dependent (`dreamplace_real` path requires a production image and environment).
- Model/data quality scaling remains active work beyond baseline synthetic-training support.
- Tier-2 handoff validation (NG45/OpenROAD-oriented downstream flow) is not fully productized in this repo.
