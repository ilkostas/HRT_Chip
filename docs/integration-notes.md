# Integration Notes (Phase 0–6)

This document describes how the executable pipeline connects to external competition assets and backends. Phase 0–3 run **fully locally** using stub adapters for evaluator, mixed-size, and diffusion sampling; no submodule checkout is required to execute `hrt-chip run` with **`--evaluator stub`**. **Phase 5** adds an optional **official** evaluator path and IBM sweep (`hrt-chip benchmark-sweep`).

## Guidance sweep and objectives (Phase 3)

- **Config:** [`src/hrt_chip/config.py`](../src/hrt_chip/config.py) — `guidance_preset` (e.g. `pareto3`) or explicit `guidance_weights_sweep`; `resolved_guidance_sweep()` yields the `(α, β, γ)` list persisted in `results.json` as `guidance_sweep_resolved`.
- **Generation:** [`src/hrt_chip/stages/generate.py`](../src/hrt_chip/stages/generate.py) — one `sample_batch` per weight vector; optional `DiffusionSampleRequest.guidance` in [`src/hrt_chip/diffusion.py`](../src/hrt_chip/diffusion.py) (stub applies deterministic coordinate bias for diversity).
- **Surrogates:** [`src/hrt_chip/guidance.py`](../src/hrt_chip/guidance.py) — fast HPWL-bbox, grid congestion, and overlap surrogates for `scoring_table` only; **Tier-1 selection uses evaluator proxy** (`run_pipeline` asserts `best_candidate_id == argmin(proxy_score)`).

## Diffusion sampler (Phase 2)

- **Contract:** [`src/hrt_chip/diffusion.py`](../src/hrt_chip/diffusion.py) — `DiffusionSampler.sample_batch(DiffusionSampleRequest) -> SampleBatch`. The request includes the **full macro set**; each candidate in the batch carries **all** macro centers in normalized **`[-1, 1]`** space (`coord_space: normalized_centers_-1_1`).
- **Stub:** `DeterministicDDPMStubSampler` — deterministic RNG layout for development; records provenance (`sampler_name`, `model_stub`, `generation_mode`, `diffusion_steps`) in `PlacementCandidate.metadata["sampler"]` and in `results.json` as `sampler_provenance`.
- **Integration (Phase 4+):** Implement a PyTorch-backed sampler that satisfies `DiffusionSampler`, load trained ε-prediction weights, and swap the default in `generate_candidates(..., sampler=...)`. Keep the adapter boundary so `run_pipeline` stays unchanged.

## Evaluator adapter

- **Contract:** [`src/hrt_chip/adapters/evaluator/base.py`](../src/hrt_chip/adapters/evaluator/base.py) — implement `EvaluatorAdapter.evaluate(...)`.
- **Stub:** [`src/hrt_chip/adapters/evaluator/local_stub.py`](../src/hrt_chip/adapters/evaluator/local_stub.py) — deterministic pseudo-proxy for development and CI.
- **Official (Phase 5):** [`src/hrt_chip/adapters/evaluator/official.py`](../src/hrt_chip/adapters/evaluator/official.py) — `OfficialMacroPlacementEvaluator` calls `macro_place.objective.compute_proxy_cost` and `macro_place.utils.validate_placement` (same stack as `uv run evaluate` in the [Partcl challenge repo](https://github.com/partcleda/partcl-macro-place-challenge)). Requires:
  1. Install the challenge package: clone the repo and `pip install -e .` (or `uv pip install -e .`) so `macro_place` imports.
  2. Initialize TILOS testcases: `git submodule update --init external/MacroPlacement` (IBM designs under `external/MacroPlacement/Testcases/ICCAD04/<benchmark_id>/`).
- **Config / CLI:** `RunConfig.evaluator_backend` is `stub` or `official`; `RunConfig.testcase_root` overrides the default ICCAD04 root. Environment variable **`HRT_CHIP_TESTCASE_ROOT`** also sets the default root (see [`src/hrt_chip/benchmarks.py`](../src/hrt_chip/benchmarks.py) `default_testcase_root()`).
- **Pipeline behavior with `official`:** [`src/hrt_chip/pipeline.py`](../src/hrt_chip/pipeline.py) loads full macro geometry from disk ([`src/hrt_chip/official_benchmark.py`](../src/hrt_chip/official_benchmark.py)), maps normalized centers to the **physical** canvas for `MacroRect`, legalizes on that canvas, restores **fixed** macro positions from the benchmark, then evaluates with the official proxy.

## Phase 5 benchmark sweep

- **CLI:** `hrt-chip benchmark-sweep` — runs all 17 IBM benchmarks ([`src/hrt_chip/benchmarks.py`](../src/hrt_chip/benchmarks.py) `IBM_BENCHMARKS`), writes `sweep_report.json`, prints Gate A/B/C vs aggregate SA (**2.1251**) and RePlAce (**1.4578**).
- **Implementation:** [`src/hrt_chip/benchmark_sweep.py`](../src/hrt_chip/benchmark_sweep.py) orchestrates repeated `run_pipeline` calls; default `--evaluator official` for milestone tracking; use `--evaluator stub` for quick smoke tests without testcases.

## Mixed-size / standard-cell backend

- **Contract:** [`src/hrt_chip/adapters/mixed_size/base.py`](../src/hrt_chip/adapters/mixed_size/base.py) — `MixedSizeBackend.run(MixedSizeRequest)` after macro legalization.
- **Stub:** [`src/hrt_chip/adapters/mixed_size/local_stub.py`](../src/hrt_chip/adapters/mixed_size/local_stub.py) — no-op success.
- **Future:** Wire DREAMPlace, hMETIS, or competition-provided clustering/placement under `external/` and pass results via `PlacementCandidate.metadata["mixed_size"]` for the evaluator handoff.

## Official challenge repository (`external/`)

- Add the official Partcl/HRT challenge repository as a **git submodule** under [`external/`](../external/) when you are ready to pin a specific evaluator revision.
- Until then, [`external/.gitkeep`](../external/.gitkeep) reserves the directory; the codebase does not import submodule paths at install time.
- See [README.md](../README.md) for the intended submodule workflow.

## AWS / environment

For runs that call cloud-hosted evaluators or data:

1. Assume role: `source env-assume-role.sh` (bash; use your platform equivalent on Windows).
2. Load secrets and endpoints: `source env.sh`.

These scripts are local to your machine and are not committed to this repository.

## Phase 6 — reproducibility and regression controls

- **Determinism:** [`src/hrt_chip/deterministic_runtime.py`](../src/hrt_chip/deterministic_runtime.py) seeds `random` / NumPy / PyTorch when `RunConfig.deterministic` is true; optional **`deterministic_verification`** tightens cuDNN and CUDA behavior (may be slower).
- **Manifest:** [`src/hrt_chip/io/artifacts.py`](../src/hrt_chip/io/artifacts.py) `RunManifest` includes `deterministic_verification` alongside `deterministic_mode`.
- **Replay verify:** [`hrt-chip replay`](../src/hrt_chip/cli.py) `--verify` loads baseline `results.json` next to the manifest, re-runs the pipeline, compares fingerprints ([`src/hrt_chip/replay_verify.py`](../src/hrt_chip/replay_verify.py)), writes `replay_verification.json`, exits non-zero on mismatch.
- **Retention:** After each run, per-candidate JSON under `candidates/` can be pruned via `artifact_retention` (`full` | `compact` | `best_only`) and optional `artifact_retention_top_k` for compact mode; `results.json` always keeps full `ranking` / `scoring_table`.
- **CI:** [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs `pytest`, `hrt-chip run` + `replay --verify`, and a stub **`benchmark-sweep`** over a small `--benchmark` subset.
