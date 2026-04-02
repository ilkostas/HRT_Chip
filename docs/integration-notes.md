# Integration Notes (Phase 0–3)

This document describes how the executable pipeline connects to external competition assets and backends. Phase 0–3 run **fully locally** using stub adapters for evaluator, mixed-size, and diffusion sampling; no submodule checkout is required to execute `hrt-chip run`.

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
- **Stub:** [`src/hrt_chip/adapters/evaluator/local_stub.py`](../src/hrt_chip/adapters/evaluator/local_stub.py) — deterministic pseudo-proxy for development.
- **Integration:** Replace `LocalStubEvaluator` with a thin wrapper around the official challenge evaluator binary or API when available. Keep the adapter boundary so the pipeline (`run_pipeline`) stays unchanged.

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
