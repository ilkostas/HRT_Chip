# A/B protocol for solver changes

Use this when comparing `legacy` vs `search_hybrid` or two search parameter sets.

## Protocol

1. **Fixed factors:** Same `--evaluator`, `--benchmark` set, `--seed`, `--solver-backend` (when comparing legacy vs search, vary only solver and required search flags).
2. **Repeats:** At least **3 seeds** per benchmark for promotion decisions; for a quick check, 1 seed is acceptable only for smoke.
3. **Metrics:** Record `best_proxy_score` from `results.json` and per-benchmark rows in `sweep_report.json`.
4. **Promotion:** Promote a change if mean proxy improves and no benchmark regresses beyond an agreed tolerance (e.g. worst-case **+5%** proxy vs baseline branch for that design).

## Automation

- Run two sweeps to separate directories; compare `sweep_report.json` means with `tools/ab_compare_runs.py`.

Example (stub smoke):

```bash
uv run hrt-chip benchmark-sweep -e stub -b ibm01 -b ibm02 --candidates 1 --output-dir runs/ab_a --sweep-id branch_a
uv run hrt-chip benchmark-sweep -e stub -b ibm01 -b ibm02 --candidates 1 --output-dir runs/ab_b --sweep-id branch_b --solver-backend search_hybrid
uv run python tools/ab_compare_runs.py runs/ab_a/branch_a/sweep_report.json runs/ab_b/branch_b/sweep_report.json
```
