# Release / Submission Readiness Checklist (Go/No-Go)

This checklist is used to decide whether your current configuration is “submission-ready”.
It is intentionally strict to avoid false progress from non-comparable runs.

## A) Competition profile compliance

All of the following must be true:

- You ran with `--evaluator official` (not `stub`) for any performance claim you intend to use.
- You used `--selection-policy proxy_first` for claimed Tier-1 proxy results.
- You used deterministic mode: `--deterministic` enabled.
- Guidance policy is fixed for the competition family:
  - either `--guidance-preset pareto3`
  - or explicit repeated `--guidance-weight` triples.
- You used fixed sampling policy:
  - same `--sampler-backend` and `--checkpoint`
  - same `--sampler-mode`
  - same `--diffusion-inference-steps`

## B) Evidence completeness (artifacts exist and are consistent)

For the winner run (the one you will claim), the directory must contain:

- `manifest.json`
- `results.json`
- `candidates/`
- a replay verification file if you ran replay verification:
  - `replay_verification.json`

For sweeps:

- `sweep_report.json`
- trends line appended to your configured JSONL log (if you use `benchmark-sweep` with trends logging)

## C) Legality requirement (hard gate)

Your final elected winner must satisfy:

- `Gate A` (17/17 legal on FULL17) is satisfied for the config you claim.
- No exceptions occurred during official evaluation.

If any benchmark is illegal:

- the run is not submission-ready;
- you may still use it for debugging, but not for claims.

## D) Competitiveness requirement (soft gate, but required for win)

On FULL17 with your chosen competition profile:

- Mean proxy beats the competition baselines:
  - aggregate SA proxy: `2.1251`
  - aggregate RePlAce proxy: `1.4578`
- If you are close to threshold, you must rerun confirmation passes to rule out noise:
  - same seed policy, same config

## E) Reproducibility requirement (mandatory for finalists)

Replay verification must pass:

- `hrt-chip replay <manifest.json> --verify` returns PASS for the winner.
- Fingerprints / comparisons show no mismatches.

## F) Runtime requirement (mandatory for competition practicality)

Your final run must meet runtime assumptions:

- The measured runtime per benchmark is compatible with the 1-hour macro placement constraint.
- If runtime is near the limit, reduce only the *performance-search width* (candidates / inference steps) and rerun.

## What “No-Go” looks like

Failing any of these is an immediate stop:

- Official evaluator wasn’t used for claimed numbers
- Winner fails Gate A (not 17/17 legal)
- Replay verification fails
- Evidence artifacts are missing or inconsistent with the config you claim

