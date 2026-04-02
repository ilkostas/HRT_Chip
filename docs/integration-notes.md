# Integration Notes (Phase 0)

This document describes how the executable pipeline connects to external competition assets and backends. Phase 0 runs **fully locally** using stub adapters; no submodule checkout is required to execute `hrt-chip run`.

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
- Until then, [`external/.gitkeep`](../external/.gitkeep) reserves the directory; the codebase does not import submodule paths at Phase 0.
- See [README.md](../README.md) for the intended submodule workflow.

## AWS / environment

For runs that call cloud-hosted evaluators or data:

1. Assume role: `source env-assume-role.sh` (bash; use your platform equivalent on Windows).
2. Load secrets and endpoints: `source env.sh`.

These scripts are local to your machine and are not committed to this repository.
