# Experiment Matrix (Days 4-5)

This matrix defines the **compute-constrained** experiments you run during Days 4-5.
The intent is to maximize win probability within 7 days by only exploring settings that can plausibly change the official proxy ranking.

## Inputs you must provide

- A trained checkpoint file: `checkpoint.pt`
- Official evaluator assets (or temporarily run with `--evaluator stub` but do not use stub results as evidence).

## Benchmark subsets (use the same subset every time)

Pick these once on Day 1 and do not change them during Days 4-5:

- `SMOKE_SUBSET` (2 benchmarks): `ibm01`, `ibm06` (example)
- `DEV_SUBSET` (4-6 benchmarks): `ibm01`, `ibm03`, `ibm06`, `ibm12`, `ibm17` (example)
- `FULL17` (promotion day): all `ibm01–ibm18` in `src/hrt_chip/benchmarks.py`

If you have a reason to prefer a different dev subset, change it on Day 1 and document the change in `docs/submission-runbook.md`.

## Fixed settings (do not vary)

These must stay constant while you test sampler/guidance variations:

- `--evaluator official`
- `--sampler-backend pytorch_checkpoint`
- `--checkpoint <same checkpoint>`
- `--mixed-size-backend estimate` (fast; affects tie-breaks under `proxy_first` only)
- `--selection-policy proxy_first` (so ranking is driven by official proxy)
- `--guidance-preset pareto3` (or the explicit triple list if you are overriding)
- `--deterministic` enabled
- `--candidates` small enough to fit runtime budget (start with `2`)
- `--artifact-retention best_only` (disk control)

## Variables to explore (keep the matrix compact)

For a small, high-signal search, vary only:

1. Sampler mode + effective reverse step count
2. Guidance weights (stick to `pareto3` for week-wide comparability; only override if you have evidence)

### Candidate matrix (recommended)

Run this set on the `DEV_SUBSET`:

| Exp ID | Sampler mode | `--diffusion-inference-steps` | Guidance |
|---|---|---:|---|
| E1 | `ddpm_subsampled` | 25 | `pareto3` |
| E2 | `ddpm_subsampled` | 50 | `pareto3` |
| E3 | `ddpm_subsampled` | 100 | `pareto3` |
| E4 | `ddim` | 50 | `pareto3` |

All four experiments should use the same `--candidates 2`.

If runtime blows up in E3, drop E3 first.
If legality ever drops below 100% on smoke, reduce `--diffusion-inference-steps` next.

## Promotion criteria (how you decide what to run on Day 6)

Promotion happens only when all of the following are true:

- Smoke subset:
  - `Gate A` is satisfied on the subset you treat as “smoke” (all legal)
- Dev subset:
  - official mean proxy is strictly better (lower) than incumbent
  - you did not introduce instability (e.g., legality failures or huge runtime variance)
- Candidate selection is consistent:
  - `proxy_first` is used for the tie-break logic

If two experiments are within measurement noise, promote both as top-1/top-2 finalists and confirm them on `FULL17` on Day 6.

## Evidence you must save per experiment

For each run or sweep, preserve:

- sweep output JSON (`sweep_report.json`) when using `benchmark-sweep`
- run `manifest.json`
- run `results.json`
- if you change your mindset about a “best” finalist, rerun with `hrt-chip replay ... --verify` at least once

