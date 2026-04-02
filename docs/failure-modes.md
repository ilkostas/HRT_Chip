# Do Not Violate (and Avoid)

This document centralizes the rules and “anti-patterns” that can invalidate results or waste the week.
Treat this as a checklist: if something below applies to your run, stop and correct it.

## Do Not Violate (Required)

These are hard competition constraints.

- Overlaps in resulting placements: final evaluated placement must have **zero overlaps** (otherwise disqualified).
- Modifying evaluation functions: the official evaluator must be used **exactly as provided** (do not patch or replace logic).
- Hardcoding solutions: never special-case a benchmark ID to “cheat” its expected structure.
- Using external/proprietary placement tools: do not depend on tools judges won’t allow; keep everything open-source.
- Exceeding runtime limits: end-to-end runtime for the macro placement algorithm must fit the competition timeout budget.

## Proxy Anti-patterns (Recommended)

Proxy optimization is necessary but not sufficient for winning.

- Proxy overfitting: improvements in proxy cost (wirelength/density/congestion) can come at the expense of timing realism that OpenROAD will measure.
- Champion selection drift: do not select “winner” configs based on local proxy experiments that differ in evaluator backend, seeds, determinism, or selection policy.
- Missing contract fields: if the mixed-size backend fails or returns incomplete metrics, tie-break ranking may become unstable.

## Practical mitigation rules

- Always use the competition run profile (same evaluator/sampler/checkpoint/guidance/selection policy) for comparisons.
- Enforce Gate A early (smoke legality) before spending compute on Days 4–6.
- Use replay verification for any configuration you intend to promote.

