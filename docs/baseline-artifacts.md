# Baseline artifacts and schema

This document locks the **structural contract** for reproducibility and regression tracking.

## `results.json`

- **Schema version**: `results_schema_version` (current: `2`), defined in [`src/hrt_chip/io/baseline_schema.py`](../src/hrt_chip/io/baseline_schema.py).
- **Required top-level keys**: `manifest`, `benchmark_id`, `evaluator_backend`, `ranking`, `scoring_table`, `best_candidate_id`, `best_proxy_score`, plus `results_schema_version`.
- **Optional (competition hardening)**:
  - `timing`: stage and total wall-clock seconds.
  - `budget`: resolved sweep/candidate caps when a wall-clock budget is active.
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
- Trend history is appended to `runs/trends/sweep_history.jsonl` (one JSON object per line).

## Regression policy

CI runs a **stub** sweep subset and asserts **Gate A** (all legal) for that subset. Official proxy thresholds are not enforced in CI (environment-dependent).
