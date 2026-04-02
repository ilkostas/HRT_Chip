# Baseline artifacts and schema

This document locks the **structural contract** for reproducibility and regression tracking.

## `results.json`

- **Schema version**: `results_schema_version` (current: `2`), defined in [`src/hrt_chip/io/baseline_schema.py`](../src/hrt_chip/io/baseline_schema.py).
- **Required top-level keys**: `manifest`, `benchmark_id`, `evaluator_backend`, `ranking`, `scoring_table`, `best_candidate_id`, `best_proxy_score`, plus `results_schema_version`.
- **Optional (competition hardening)**:
  - `timing`: per-stage seconds — `generation_seconds`, `legalization_seconds`, `mixed_size_seconds`, `evaluation_seconds`, `legalize_mixed_size_eval_seconds` (sum of the three post-gen stages), `total_pipeline_seconds`.
  - `runtime_budget`: when `wall_clock_budget_seconds` is set, includes elapsed/remaining and per-stage spend (see [`src/hrt_chip/runtime_budget.py`](../src/hrt_chip/runtime_budget.py)).
  - `sweep_vectors_used` / `sweep_vectors_requested` / `generation_stopped_early`: guidance vectors actually generated vs resolved sweep (early stop under runtime budget).
  - `pre_eval_rejection`: optional cheap filters that skip the official evaluator (`skipped_eval_count`).
  - `sampler_mode`, `diffusion_reverse_schedule`, `experiment_tag`, `experiment_notes`.
  - `budget_resolution` (not `budget`): resolved sweep/candidate caps when a wall-clock budget is active (see [`src/hrt_chip/budget.py`](../src/hrt_chip/budget.py)).
  - `testcase_root`, `canvas_width`, `canvas_height`, guidance/sampler fields as before.
  - `selection_policy`, `selection_rationale`, per-row `mixed_size_profile` (batch-normalized mixed-size metrics + `composite_ppa`).

Validation (Python):

```python
from hrt_chip.io.baseline_schema import validate_results_json_shape, RESULTS_JSON_SCHEMA_VERSION
errs = validate_results_json_shape(results)
assert not errs, errs
```

## `sweep_report.json`

- Written under `runs/sweeps/<sweep_id>/sweep_report.json`.
- Includes gate fields (`gate_a_legal_all`, `gate_b_beat_sa_aggregate`, `gate_c_beat_replace_aggregate`) and per-row proxies.
- Each `rows[]` entry may include **`timing`** (copy of the benchmark run’s `results.json` `timing` object: generation / legalization / mixed-size / evaluation breakdown).
- `extra` may include `experiment_tag`, `experiment_notes` when set on `RunConfig`.
- Trend history is appended to `runs/trends/sweep_history.jsonl` (one JSON object per line). Use **`hrt-chip trends-report`** to summarize recent lines.

## Regression policy

CI runs a **stub** sweep subset and asserts **Gate A** (all legal), validates the **trends JSONL** last line schema, and checks **per-row timing** keys (`tools/check_stub_sweep_regression.py`). Official proxy mean thresholds are not hard-failed in CI (environment-dependent); stub `mean_proxy` sanity is loosely bounded.
