# Integration Notes (Phase 0–6)

This document describes how the executable pipeline connects to external competition assets and backends. Phase 0–3 run **fully locally** using stub adapters for evaluator, mixed-size, and diffusion sampling; no submodule checkout is required to execute `hrt-chip run` with **`--evaluator stub`**. **Phase 5** adds an optional **official** evaluator path and IBM sweep (`hrt-chip benchmark-sweep`).

## Guidance sweep and objectives (Phase 3)

- **Config:** [`src/hrt_chip/config.py`](../src/hrt_chip/config.py) — `guidance_preset` (e.g. `pareto3`) or explicit `guidance_weights_sweep`; `resolved_guidance_sweep()` yields the `(α, β, γ)` list persisted in `results.json` as `guidance_sweep_resolved`.
- **Generation:** [`src/hrt_chip/stages/generate.py`](../src/hrt_chip/stages/generate.py) — one `sample_batch` per weight vector; optional `DiffusionSampleRequest.guidance` in [`src/hrt_chip/diffusion.py`](../src/hrt_chip/diffusion.py) (stub applies deterministic coordinate bias for diversity).
- **Surrogates:** [`src/hrt_chip/guidance.py`](../src/hrt_chip/guidance.py) + [`src/hrt_chip/netlist_surrogates.py`](../src/hrt_chip/netlist_surrogates.py) — when an official `Benchmark` exposes **pin groups** (`net_pin_dx` / `net_pin_dy` aligned with `net_nodes`), the scoring table uses **exact weighted pin HPWL** and **RUDY over pin bounding boxes**; legality reports **hard overlap pair count** plus **smooth squared-overlap penalty**. If the benchmark has nets but no pin offsets, `phi_hpwl` / `phi_congestion` are omitted in JSON (`null`) and `surrogate_mode` is `netlist_pins_missing` (no silent fallback to macro-center pin pretending). Stub runs (no benchmark) keep bbox + macro occupancy-variance surrogates. **Default final ordering** is `RunConfig.selection_policy=proxy_first` (proxy primary, mixed-size **composite_ppa** tie-break). Opt-in **`ppa_priority`** ranks legal candidates with `mixed_size.ok` by batch-normalized mixed-size metrics first, then proxy. [`src/hrt_chip/pipeline.py`](../src/hrt_chip/pipeline.py) adds `surrogate_proxy_alignment` (Spearman / Kendall vs weighted surrogate composite) for diagnostics.
- **Future DDPM guidance hook:** [`src/hrt_chip/diffusion.py`](../src/hrt_chip/diffusion.py) `GuidanceObjectiveBias` + optional `DiffusionSampleRequest.objective_bias` (wired through [`generate_candidates`](../src/hrt_chip/stages/generate.py); no-op in current samplers).

## Diffusion sampler (Phase 2)

- **Contract:** [`src/hrt_chip/diffusion.py`](../src/hrt_chip/diffusion.py) — `DiffusionSampler.sample_batch(DiffusionSampleRequest) -> SampleBatch`. The request includes the **full macro set**; each candidate in the batch carries **all** macro centers in normalized **`[-1, 1]`** space (`coord_space: normalized_centers_-1_1`). Optional request fields: `diffusion_inference_steps`, `sampler_mode`, `reverse_timestep_indices` (PyTorch path).
- **Stub:** `DeterministicDDPMStubSampler` — deterministic RNG layout for development; records provenance (`sampler_name`, `model_stub`, `generation_mode`, `diffusion_steps`) in `PlacementCandidate.metadata["sampler"]` and in `results.json` as `sampler_provenance`.
- **PyTorch (Phase 4+):** [`src/hrt_chip/adapters/diffusion/pytorch_sampler.py`](../src/hrt_chip/adapters/diffusion/pytorch_sampler.py) — `RunConfig.sampler_mode`: **`ddpm_full`** (full `T` ancestral steps), **`ddpm_subsampled`** (fewer DDPM steps via [`subsampled_reverse_timesteps`](../src/hrt_chip/training/schedule.py)), **`ddim`** (deterministic DDIM jumps between schedule indices). Optional **`diffusion_reverse_schedule`** string (comma-separated indices, e.g. `999,500,0`) overrides uniform spacing. CLI: `--sampler-mode`, `--diffusion-reverse-schedule`, `--diffusion-inference-steps`.

## Runtime budget and pre-eval (orchestration)

- **Static caps:** [`src/hrt_chip/budget.py`](../src/hrt_chip/budget.py) — `resolve_generation_budget` shrinks guidance sweep / `num_candidates` when `wall_clock_budget_seconds` is set (pre-run).
- **Runtime manager:** [`src/hrt_chip/runtime_budget.py`](../src/hrt_chip/runtime_budget.py) — optional per-stage fraction tracking and **early stop** of further guidance vectors if remaining wall time is insufficient for pending legalization / mixed-size / eval (pipeline generates **one sweep vector at a time**). Adaptive **fewer diffusion inference steps** under time pressure (`recommended_diffusion_inference_steps`).
- **Pre-eval skipping:** `RunConfig.pre_eval_rejection_enabled` — skips the expensive evaluator for illegal macro placements, excessive **hard overlap pairs**, or bad **surrogate composite** when thresholds are set (`pre_eval_max_hard_overlap_pairs`, `pre_eval_surrogate_composite_max`). Skips are recorded under `evaluations[].details` and `pre_eval_rejection` in `results.json`.

## Trend dashboard (CLI)

- **`hrt-chip trends-report`** — loads recent JSONL lines (default `runs/trends/sweep_history.jsonl`) and prints gate pass rates and a table of recent sweeps. Optional `--baseline-sweep-id` for mean-proxy delta vs a reference sweep.

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
- **Estimate (default in CLI):** [`src/hrt_chip/adapters/mixed_size/estimate.py`](../src/hrt_chip/adapters/mixed_size/estimate.py) — macro area utilization, RUDY variance proxy from the loaded `Benchmark` nets, and backend runtime in `metadata["mixed_size"]["extra"]`. No standard-cell placement binary.
- **DreamPlace Docker (`mixed_size_backend=dreamplace`):**
  - **Adapter:** [`src/hrt_chip/adapters/mixed_size/dreamplace_docker.py`](../src/hrt_chip/adapters/mixed_size/dreamplace_docker.py) — writes per-candidate dirs under `runs/<run_id>/mixed_size/<candidate_id>/`, runs `docker run` with that directory mounted at **`/work`**, optional ICCAD04 root at **`/testcase:ro`** ([`runner.py`](../src/hrt_chip/adapters/mixed_size/runner.py)).
  - **JSON contract:** [`contracts.py`](../src/hrt_chip/adapters/mixed_size/contracts.py) — host writes `input.json` (`hrt_mixed_size_input_v1`); container must write `output.json` (`hrt_mixed_size_output_v1`) with at least `ok`, `message`, and metrics such as `density_overflow`, `rudy_or_route_proxy`, `backend_runtime_seconds`, `hmetis_invoked`, `dreamplace_invoked`, `placement_mode`.
  - **Default image:** build from [`docker/Dockerfile.dreamplace`](../docker/Dockerfile.dreamplace) (`scripts/build_dreamplace_image.ps1` or `.sh`) → tag **`hrt-chip-dreamplace:local`**. The bundled [`docker/dreamplace_flow/run_flow.py`](../docker/dreamplace_flow/run_flow.py) is an **analytical proxy** (deterministic metrics from macro geometry). Replace the image with a CUDA DREAMPlace + hMETIS toolchain while keeping the same `/work` paths.
  - **Config / CLI:** `RunConfig.mixed_size_backend`, `dreamplace_docker_image`, `dreamplace_docker_timeout_seconds`, `dreamplace_docker_retries`, `dreamplace_mount_testcase`, `dreamplace_docker_extra_args`, `dreamplace_docker_executable`. Env: **`HRT_DREAMPLACE_IMAGE`**, **`HRT_DREAMPLACE_TIMEOUT`**, **`HRT_DOCKER_EXECUTABLE`**.
- **DreamPlace real (`mixed_size_backend=dreamplace_real`):**
  - Same adapter class with variant **`REAL_DOCKER_VARIANT`** — writes optional **`flow: mixed_size_real`** into `input.json` so the container can run a mixed-size / std-cell branch (see [`run_flow.py`](../docker/dreamplace_flow/run_flow.py)).
  - **Separate defaults:** `dreamplace_real_docker_image` (default **`hrt-chip-dreamplace-real:local`**), `dreamplace_real_docker_timeout_seconds` (default **900**). Env: **`HRT_DREAMPLACE_REAL_IMAGE`**, **`HRT_DREAMPLACE_REAL_TIMEOUT`**. Reuses `dreamplace_docker_retries`, mounts, and `dreamplace_docker_extra_args` for extra tool bind-mounts.
  - Build a second image tagged `hrt-chip-dreamplace-real:local` (your CUDA DREAMPlace+hMETIS Dockerfile) or point the env var at an existing image; until then you can point **`HRT_DREAMPLACE_REAL_IMAGE`** at the analytical image for contract tests.
- **Mixed-size metrics in `results.json`:** batch min–max normalization and `composite_ppa` per candidate live under `evaluations[].mixed_size_profile`, `ranking[].mixed_size_profile`, `scoring_table[].mixed_size_profile`, and `candidates/*.json` → `metadata.mixed_size.profile`. Weights are defined in [`src/hrt_chip/mixed_size_metrics.py`](../src/hrt_chip/mixed_size_metrics.py).
  - **Failure behavior:** On Docker missing, timeout, non-zero exit, or bad `output.json`, `mixed_size.ok` is **false**; the pipeline still evaluates the candidate (illegal geometry still yields infinite proxy as before).
  - **CI:** workflows keep **`--mixed-size-backend stub`** so runners do not require Docker.

### Docker Desktop (Windows) operator checklist

1. Enable **WSL2** backend (recommended) or ensure the drive with the repo is **shared** with Docker.
2. From repo root: `.\scripts\build_dreamplace_image.ps1` (or set `HRT_DREAMPLACE_IMAGE_TAG` for a custom tag).
3. Smoke the container alone: create a folder with a valid `input.json` (see `tests/test_mixed_size_dreamplace_docker.py` payload shape) or use `scripts/run_dreamplace_flow.ps1 -WorkDir <path>`.
4. Run `uv run hrt-chip run ... --mixed-size-backend dreamplace`. If testcase mount causes issues, use **`--no-dreamplace-mount-testcase`** (official evaluator still loads macros from disk on the host).
5. **Optional integration test:** `HRT_DREAMPLACE_INTEGRATION=1 uv run pytest tests/test_mixed_size_dreamplace_docker.py::test_dreamplace_integration_smoke` (requires image present).

### `dreamplace_docker_extra_args` (POSIX shlex)

Extra args are parsed with **`shlex.split(..., posix=True)`**. On Windows, prefer forward slashes in paths passed inside `-v` mounts, or set `dreamplace_docker_executable` to a known-good `docker.exe` path.

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
