# Competition Run Profile (Claimed Results)

This document defines the **only** run configuration you are allowed to use when you claim performance or competitiveness.
It is designed to make runs comparable (same evaluator, same candidate-selection policy, same determinism controls).

## What “competition-grade” means

- Official Tier-1 metric path: `evaluator_backend=official` (TILOS MacroPlacement via `macro_place`).
- Hard legality: final candidates must be continuous-legal (no overlaps under the evaluator).
- Reproducibility: identical config + seed must be replayable via `hrt-chip replay --verify`.
- Selection: Tier-1 claimed results use `selection_policy=proxy_first` (proxy primary, mixed-size composite tie-break).

## Required knobs (Tier 1 / Proxy Cost claims)

Use these parameters for every run you intend to count toward “beating SA/RePlAce”:

| Knob | Value | Where |
|---|---|---|
| `--evaluator` | `official` | `hrt-chip run`, `hrt-chip benchmark-sweep` |
| `--sampler-backend` | `pytorch_checkpoint` | `hrt-chip run` (and for sweeps) |
| `--checkpoint` | path to your trained `checkpoint.pt` | `--checkpoint <...>` |
| `--guidance-preset` | `pareto3` (or explicit `--guidance-weight` triples) | Phase 3 objective sweep |
| `--selection-policy` | `proxy_first` | Ensures official proxy is the primary ranking key |
| `--deterministic` | enabled (default is on) | `RunConfig.deterministic=True` |
| `--artifact-retention` | `best_only` or `compact` (pick once, keep consistent) | controls disk usage only |
| `--diffusion-inference-steps` | pick a fixed value for the whole week | ensures comparable sampling cost/quality |
| `--mixed-size-backend` | **Required for comparability.** Recommended: `estimate`. Nice-to-have: `dreamplace_real` for deeper mixed-size exploration. | affects tie-breaks only under `proxy_first` |

## Tier-1 vs Tier-2 Objectives

**Required:** Tier-1 “claimed” competitiveness in this repository is driven by the **official proxy cost** (TILOS MacroPlacement `proxy_cost`) and your candidate ranking under `selection_policy=proxy_first`.

**Required (Grand Prize condition):** Tier-2 outcomes depend on the **OpenROAD** flow (WNS/TNS/Area) and are evaluated by the competition infrastructure/judges.

**Recommended:** Treat local proxy optimization as necessary but not sufficient; if you have a way to run OpenROAD validation externally on public NG45, do it only after you lock by proxy.

### Recommended “starting point” values (edit only with care)

- `--mixed-size-backend estimate`
- `--candidates 2` (per guidance vector)
- `--diffusion-inference-steps 50`
- `--diffusion-steps 1000`
- `--guidance-preset pareto3`
- `--artifact-retention best_only`

If you later change any of these values, you must treat the new settings as a new experiment family in your matrix (see `docs/experiment-matrix.md`).

## Nice-to-have (optional, but may improve submission quality)

- Increase candidate count beyond `--candidates 2` (only if you can keep runtime comfortably under 1 hour).
- Try `--mixed-size-backend dreamplace_real` as a separate exploration track (do not overwrite your “competition profile” evidence).
- Use `--deterministic-verification` only for replay audits of promoted finalists (slower).

## CLI templates (copy exactly)

### Smoke run (2 benchmarks)

Run a small official sweep to confirm the pipeline works end-to-end and produces legal outputs.

```bash
source env-assume-role.sh
source env.sh

uv sync --group dev

uv run hrt-chip benchmark-sweep \
  --evaluator official \
  --sampler-backend pytorch_checkpoint \
  --checkpoint <path/to/checkpoint.pt> \
  --mixed-size-backend estimate \
  --selection-policy proxy_first \
  --guidance-preset pareto3 \
  --diffusion-inference-steps 50 \
  --candidates 2 \
  --benchmark ibm01 --benchmark ibm06 \
  --output-dir runs/sweeps
```

### Full run (one benchmark, for deep inspection)

```bash
source env-assume-role.sh
source env.sh

uv run hrt-chip run \
  --benchmark ibm01 \
  --seed 42 \
  --evaluator official \
  --sampler-backend pytorch_checkpoint \
  --checkpoint <path/to/checkpoint.pt> \
  --mixed-size-backend estimate \
  --selection-policy proxy_first \
  --guidance-preset pareto3 \
  --diffusion-inference-steps 50 \
  --candidates 4 \
  --artifact-retention best_only \
  --deterministic \
  --output-dir runs
```

## Official evaluator prerequisites (must be true before Day 2/3 runs)

The `official` evaluator path expects:

1. The challenge package `partcl-macro-place-challenge` installed so `macro_place` imports.
2. ICCAD04 testcases available under one of:
   - **Recommended:** set `HRT_CHIP_TESTCASE_ROOT` to the **ICCAD04 root** inside your cloned `macro-place-challenge-2026-main` checkout (the folder that contains `ibm01/`, `ibm02/`, ... and `netlist.pb.txt`).
   - `external/MacroPlacement/Testcases/ICCAD04/` (only if present under this repo).
   - the env var `HRT_CHIP_TESTCASE_ROOT` (alternative), or
   - `--testcase-root <path>`.
3. (Optional) submodule checkout if your repo uses a submodule layout:
   - `git submodule update --init external/MacroPlacement`

If these prerequisites are not available locally, you can still run the pipeline with `--evaluator stub`, but those runs are **not** admissible as competition evidence.

## Determinism policy

- Use `--deterministic` for every official run.
- Only use `--deterministic-verification` when you are performing a replay audit for a promoted finalist.

## Rules you must not violate

Before running any official evidence, review: [`docs/failure-modes.md`](docs/failure-modes.md).

