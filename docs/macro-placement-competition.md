# Macro Placement Competition (Partcl × HRT)

You are tasked with developing a next-generation macro placement algorithm for a competition hosted by Partcl and Hudson River Trading (HRT).

Specifically, you need to build a system capable of positioning large fixed-size blocks (such as SRAMs, IPs, and analog macros) on a chip floorplan. This is an incredibly difficult physical design challenge that requires your algorithm to navigate a massive search space (roughly 10^800 possible placements) while balancing conflicting objectives like routing congestion, timing, power delivery, and area constraints.

To successfully complete this task, your algorithm must meet the following criteria:

## Performance targets

Your primary goal is to outperform the existing Simulated Annealing (SA) and RePlAce baselines. Submissions are first evaluated on a **Proxy Cost** (wirelength, density, congestion) across the **public IBM ICCAD04 benchmarks** (17 designs: **ibm01–ibm04, ibm06–ibm18**). Top submissions are then tested through a full **OpenROAD** flow on **NG45** designs to measure place-and-route outcomes: **WNS**, **TNS**, and **Area**.

**Proxy cost (TILOS MacroPlacement evaluator, used as-is):**

`Proxy Cost = 1.0 × Wirelength + 0.5 × Density + 0.5 × Congestion`

**Aggregate baselines (average proxy across all IBM benchmarks):** SA **2.1251**, RePlAce **1.4578**. Per-benchmark SA and RePlAce numbers are in the competition materials and [`README of Competition.md`](README%20of%20Competition.md).

**Tier 2 cutoff:** The **top 7** submissions by proxy score advance to OpenROAD validation on NG45 (including **1–2 hidden** NG45 designs to limit overfitting). Public NG45 examples include **ariane133**, **ariane136**, **mempool_tile**, **nvdla**.

## Algorithmic freedom

You are allowed to use any algorithmic approach to build your solution. This includes reinforcement learning, graph neural networks (GNNs), differentiable optimization, classical heuristics, or hybrid searches. You can also use any framework, such as PyTorch, TensorFlow, JAX, or pure Python/C++.

**Training:** You may train and tune on the **public** IBM benchmark data for Tier 1; Tier 2 adds hidden NG45 stress tests.

## Hard constraints

- Your algorithm must produce placements with **absolutely zero overlaps** between the blocks (enforced by the evaluator).
- You **cannot hardcode** solutions for specific benchmarks; it must be a **general** algorithm.
- You must use the **TILOS MacroPlacement evaluator exactly as provided** — it is the official validator and leaderboard metric source; do not swap or modify evaluation functions.

## Submission

- Submissions go through a **Google Form**; you must grant judges access so they can **clone and run** your repository.
- Exact placement I/O formats and API details are defined in the **official challenge repository** (see [`README of Competition.md`](README%20of%20Competition.md) Quick Start).

## Runtime limits

- **1 hour end-to-end per benchmark** (hard timeout; treat as total wall-clock for the macro placement step unless the organizers specify otherwise).
- Reference evaluation hardware: **AMD EPYC 9655P** (16 cores, **100 GB** RAM) and **NVIDIA RTX 6000 Ada** (**48 GB** GPU).

## Open-source requirement

If your algorithm is a winning submission, the implementation must be **open-sourced** under **Apache 2.0** or **GPL**.

## Prize

If your creation successfully surpasses the baselines on the required metrics and is validated on hidden designs, you will be eligible for a **$20,000 Grand Prize**.

## More detail

Full rules, prizes, submission link, benchmark table, FAQ, and contacts: **[`README of Competition.md`](README%20of%20Competition.md)**. Questions: **contact@partcl.com** or the challenge GitHub.
