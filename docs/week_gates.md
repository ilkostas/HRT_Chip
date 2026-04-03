# Weekly milestone gates (operational)

Use these as go/no-go checks alongside `hrt-chip benchmark-sweep` and saved `sweep_report.json`.

## Week 1

- Search-hybrid path runs on all target benchmarks (`--solver-backend search_hybrid`, official evaluator when available).
- Legality: Gate A (100% legal placements) holds.
- Proxy: aim for aggregate proxy **below 1.9** on the agreed suite, **or** within **30% of RePlAce** on at least **five** designs (see `docs/hybrid-search-solver.md`).
- Do not treat tiny aggregate improvements as success if the search engine shows no clear margin over strong baselines.

## Weeks 2–4

- Track per-benchmark proxy components (wirelength, density, congestion) and outliers vs `SA_BASELINE_BY_DESIGN` / `REPLACE_BASELINE_BY_DESIGN` in `src/hrt_chip/benchmarks.py`.
- Promote changes only when mean improves with controlled variance (see `tools/ab_protocol.md`).

## Week 5 (feature freeze)

- Stop adding new algorithm features; focus on tuning, regression sweeps, and replay verification for promoted configs.

## Weeks 6–7

- Experiment throughput and submission selection only; optional OpenROAD/Tier-2 sanity on finalist DEFs if infrastructure allows.
