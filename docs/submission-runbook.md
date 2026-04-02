# Submission Runbook (7-Day Rehearsal)

This runbook is the step-by-step execution checklist for the 7-day “winning plan”.
It is written so you can run Day 1/2/4/5/6/7 with minimal decisions mid-flight.

## 0) Conventions

- Always store outputs under `runs/` (or `--output-dir runs`).
- Always keep a single “competition profile” and never mix in dev defaults for claimed results.
- If any run fails, stop and fix the failure mode before continuing the matrix.

## 1) Preflight (must pass before any official-evaluator run)

### 1.1 Environment scripts

1. `source env-assume-role.sh`
2. `source env.sh`

### 1.2 Python / dependencies

- `uv sync --group dev`
- Confirm CLI:
  - `uv run hrt-chip --help` (should show `run`, `replay`, `benchmark-sweep`, etc.)

### 1.3 Official evaluator prerequisites

For `--evaluator official`, you must have:

1. Challenge evaluator package installed so `macro_place` imports.
2. ICCAD04 testcases available under:
   - `external/MacroPlacement/Testcases/ICCAD04/`, or
   - `HRT_CHIP_TESTCASE_ROOT`, or
   - via `--testcase-root`.

If prerequisites are missing, you may run with `--evaluator stub` for debugging, but you must treat those results as *non-competitive evidence*.

### 1.4 Docker / mixed-size readiness (only if you use `dreamplace*`)

If you choose `--mixed-size-backend dreamplace` or `dreamplace_real`:

- Docker must be able to run the container and produce the expected `output.json` contract.
- Ensure the repo path is mounted correctly into the container.

## 2) Day 1: Freeze + baseline sanity

1. Decide your competition run profile and freeze it (see `docs/competition-profile.md`).
2. Define the benchmark subsets and freeze them (see `docs/experiment-matrix.md`).
3. Run the **smoke** official sweep:
   - 2 benchmarks
   - small candidate count
   - record `sweep_report.json`, and the manifests/results for the best run(s).
4. Pick the incumbent config (best by legality first, then mean proxy, then runtime).

## 3) Day 2: Incumbent evidence and replay verification

For the incumbent candidate, do:

1. Produce official baseline sweeps (smoke + dev) and write an `incumbent.json` evidence record.
   - Use the provided helper script:

```bash
python tools/run_official_baseline_evidence.py \
  --checkpoint <path/to/checkpoint.pt> \
  --output-dir runs/baselines \
  --sweep-id day2_incumbent_v1 \
  --seed 42 \
  --candidates 2 \
  --diffusion-steps 1000 \
  --diffusion-inference-steps 50 \
  --sampler-mode ddpm_subsampled \
  --mixed-size-backend estimate \
  --selection-policy proxy_first \
  --guidance-preset pareto3 \
  --smoke-benchmarks ibm01,ibm06 \
  --dev-benchmarks ibm01,ibm03,ibm06,ibm12,ibm17 \
  --replay-verify-benchmark ibm01 \
  --replay-verify
```

2. Inspect the evidence it produces:
   - `runs/baselines/<sweep-id>__smoke/sweep_report.json`
   - `runs/baselines/<sweep-id>__dev/sweep_report.json`
   - `runs/baselines/<sweep-id>__incumbent.json`

3. (Optional) For deeper auditing, replay-verify another benchmark run manifest.

If you instead prefer the manual approach, it is:

1. One `hrt-chip run` using the exact incumbent config and fixed seed.
2. One replay verification:

```bash
uv run hrt-chip replay <path/to/manifest.json> --verify
```

If verification fails, do not proceed with Days 4-5. Fix reproducibility first.

## 4) Days 4-5: Performance search matrix

Using the exact fixed settings from `docs/experiment-matrix.md`, run the compact matrix:

- Use the helper script (recommended, avoids manual bookkeeping):

```bash
python tools/run_compact_matrix_day4_5.py \
  --incumbent-json <path/to/runs/baselines/<day2_sweep-id>__incumbent.json> \
  --output-dir runs/day4_5 \
  --runtime-multiplier 1.2 \
  --candidate-finalists 2
```

- Run each experiment ID (E1-E4) on the DEV subset.
- For each sweep:
  - ensure `Gate A` is not broken on the smoke subset you use for filtering
  - record mean proxy and legality rate

After each experiment, update the incumbent immediately:

- New incumbent replaces the old one only if it improves mean proxy without harming legality.

Stop exploring configurations that:

- do not improve proxy ranking relative to the incumbent,
- introduce legality regressions,
- exceed runtime budget.

## 5) Day 6: Full-17 promotion and finalist lock

1. Run FULL17 promotion sweeps using the helper script (locks the winner/backup
   and performs replay verification on one benchmark):

```bash
python tools/run_full17_promotion_day6.py \
  --incumbent-json <path/to/runs/baselines/...__incumbent.json> \
  --day4-5-finalists-json <path/to/runs/day4_5/...__day4_5_finalists.json> \
  --output-dir runs/day6 \
  --replay-verify-benchmark ibm01
```

2. The script runs official `benchmark-sweep` on FULL17 for each Day 4-5 finalist,
   then selects:
   - only Gate-A-legal configs (`gate_a_legal_all=True`)
   - lowest `mean_proxy` as the winner
   - second-best as backup
3. Replay verification output is written under the Day 6 lock JSON:
   - `...__day6_full17_finalist_lock.json`

If you prefer manual execution, the equivalent steps are:

1. Run `benchmark-sweep` on FULL17 for the top 2 finalists from Days 4-5.
2. For the top finalist:
   - rerun the same config at fixed seed/control at least once (confirmation run).
3. Select the final winner using:
   - 17/17 legal
   - best aggregate proxy
   - replay confidence (at least one replay verify)

## 6) Day 7: Freeze and evidence pack

Run the Day 7 finalizer (recommended; it enforces Gate A + replay PASS and assembles a compact evidence pack):

```bash
python tools/finalize_submission_day7.py \
  --day6-lock-json <path/to/runs/day6/...__day6_full17_finalist_lock.json> \
  --output-dir runs/day7 \
  --copy-backup
```

What it validates:

- `winner.gate_a_legal_all == true`
- `replay_verification.winner.replay_verification.ok == true`

What it assembles:

1. Final full17 sweep artifacts (`sweep_report.json`)
2. The winner run artifacts:
   - `manifest.json`
   - `results.json`
   - `candidates/<winner>.json` (or retained candidate JSONs based on retention mode)
3. Replay verification artifact:
   - `replay_verification.json` (from `replay --verify`)
4. The exact competition command(s) you ran, copied from this runbook.

### Evidence acceptance checklist (final)

- Winner run: 17/17 legal under official evaluator.
- Replay verification: PASS for the winner.
- Runtime: consistent with your competition-time assumptions.

