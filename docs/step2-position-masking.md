# Step 2: Legality — Sequential Masking (RL) vs Diffusion Guidance + Legalization

To enforce **absolutely zero overlaps** between macros, the right mechanism depends on **how** placements are generated.

- **Sequential RL placers (e.g. MaskPlace)** can use **position masking** on a discrete grid and get **0% overlap by construction** (on that grid).
- **Simultaneous diffusion** places **all** macros in one trajectory; **MaskPlace-style masking does not apply** (no ordering of “previous” macros). Legality is pushed via a **continuous legality potential** during sampling, then completed with **post-processing legalization**.

This document separates those two worlds. It also records **competition-aligned facts**: any algorithmic approach is allowed (sequential RL, GNNs, heuristics, hybrids). Earlier guidance favored **diffusion** for sample efficiency and wirelength on IBM benchmarks—you are **not** required to use diffusion, and **MaskPlace-style sequential RL remains a valid choice**.

---

## Competition scope, scorer, and hard legality

**Rules:** The competition permits **any** algorithmic approach (sequential RL, GNNs, classical heuristics, hybrids, etc.).

**Official overlap check:** Submissions are scored with the **TILOS MacroPlacement** evaluator using the **exact continuous geometry** of macros. **Absolutely zero overlaps** is a **strict hard constraint**: any overlap fails evaluation.

**Implication for your stack:** A **grid** or other discrete approximation inside a neural net is only an internal stepping stone. The **final** coordinates must be **fully legal in continuous space**—typically via a **legalization** pass (or equivalent) that satisfies the continuous checker.

---

## 1. Position masking (MaskPlace-style) — sequential RL only

Neural nets output **continuous** values. If overlap avoidance is left entirely to learning, mistakes are inevitable—especially on **heterogeneous** macros (ICCAD04-style designs can span **~33×** size ratios).

**MaskPlace** and related **RL** methods place macros **one at a time**. At each step *t*, only macro *t* is chosen; the canvas state reflects **M₁:ₜ₋₁**. That yields a well-defined **feasible set** for the next anchor.

### Macro geometry (ICCAD04 / this competition)

For the **IBM ICCAD04** benchmarks used in this competition, macros are **axis-aligned rectangles**. **Minkowski-style** forbidden regions for the **current** macro’s anchor can therefore be computed **exactly** in grid space—no general polygon approximation is required for rotation or arbitrary shapes.

### Discretize the canvas

Map the die to a **2D grid**. **Grid size *N* is not fixed** by the competition; it is a **hyperparameter** balancing accuracy vs. speed and what your architecture expects.

**Reference points from the literature:**

- **MaskPlace** uses a **fixed high-resolution** grid (**224 × 224**), mapping macro footprint to cells via **ceiling** of scaled width/height (**⌈wN/W⌉ × ⌈hN/H⌉** when anchoring on an *N × N* canvas over physical *W × H*).
- **AlphaChip** treats grid choice partly as a **bin-packing** problem to limit wasted space, caps the maximum grid at **128 × 128**, and reports that an **average** of about **30 × 30** rows/columns works well in practice.

Pick *N* for **your** network and runtime budget—not a single mandated value.

### Position mask

For the **current** macro, build **M ∈ {0,1}^{N×N}** (same resolution as the grid):

- **1:** anchor at that cell is **legal** (no overlap with **already placed** macros, no out-of-bounds footprint).  
- **0:** forbidden.

The mask is **recomputed every step** because it depends on **M₁:ₜ₋₁**.

### Efficient updates — **O(V)**

For each already-placed macro, expand to a **forbidden region** for the **current** macro’s anchor (Minkowski-style expansion by the current macro’s width/height in grid space), zero those cells in **M**. Complexity scales with **V** placed macros, not a naive **N²** scan against all geometry from scratch.

### Applying the mask

If the policy outputs a distribution **P** over grid cells (logits or probabilities):

1. **P′ = P ⊙ M** (zero forbidden mass).  
2. Sample or argmax from **P′**.

On the **grid**, illegality is **impossible** for the sampled discrete anchor. The **official** evaluation still uses **continuous** geometry—ensure the final exported centers (after any snap or refine) satisfy the continuous checker.

**Guarantee by construction** applies to this **sequential, masked** process on the grid—not to joint continuous diffusion over all centers at once.

---

## 2. Why simultaneous diffusion cannot use that mask

A **diffusion** model for placement generates **all macro centers together** over *T* denoising steps. There is **no** step *t* where “only macro *i* moves while others are fixed placements on a discrete grid” in the same way as autoregressive RL.

Therefore you **cannot** maintain MaskPlace’s **per-step mask against M₁:ₜ₋₁** inside the diffusion sampler: there is **no** such sequence **during** generation.

---

## 3. Diffusion: legality potential + backwards universal guidance

For **simultaneous diffusion**, legality is encouraged with a **continuous legality potential** whose **gradient** steers sampling. The intended mechanism is **backwards universal guidance**: at each denoising step, use the model’s prediction of the **fully denoised** coordinates **x̂₀**, compute gradients of scalar potentials w.r.t. those coordinates, and inject them into the sampling update. You **do not** train a **separate critic network** for this—the guidance comes from **differentiable potentials** on **x̂₀**.

**Potentials (typical stack):**

- **Legality potential:** built from **pairwise** rectangular geometry; one form penalizes **squared signed distance**—e.g. **∑_{i,j} min(0, d_ij(x))²** over macro pairs, where *d_ij* is the signed gap between rectangles (negative when overlapping).
- **HPWL potential:** a differentiable wirelength surrogate (see [`step1-diffusion-model.md`](./step1-diffusion-model.md) §4).

**Dynamic weights:** Gradients are combined using **Lagrangian multipliers** adjusted via **interleaved gradient-style** updates so the trade-off between overlap and HPWL can adapt across circuits—avoid fixing a single weight for all benchmarks.

### Coordinate parameterization (unambiguous “gradient on coordinates”)

Align with **Step 1**: the diffusion model operates on an array of **2D centers** for movable objects, **normalized to chip boundaries in [−1, 1]** end-to-end. “Guidance applies to coordinates” means: compute **∂ϕ / ∂x̂₀** on those **continuous [−1, 1]** tensors and inject the resulting direction into the reverse diffusion step (see Step 1 for full DDPM / noise-prediction context).

### Paper vs competition target legality

Diffusion papers may report **high** but **not perfect** legality during sampling (e.g. **~0.997** on IBM benchmarks in *Chip Placement with Diffusion Models*)—that illustrates how strong **soft** guidance can be. For **this competition**, that number is **background only**. Your submission must achieve **100%** legality (**0%** overlap) under the **official continuous** evaluator; soft guidance alone is **not** sufficient—you still need **legalization** (§4) unless some other part of your pipeline guarantees exact legality.

---

## 4. Post-processing legalization (diffusion-oriented pipeline)

For a **hard** zero-overlap rule under continuous checking, a **diffusion** (or any soft-guided) pipeline should include a **legalization** pass after sampling:

- Remove **any** residual overlap (snap, nudge, push, or other moves).  
- After legality is restored, prefer moves that **minimize harm** to the competition **Proxy Cost**.

**Competition constraints on the legalizer:** There are **no** special limits on the **number** of moves, greedy vs. other strategies, or specific forbidden maneuvers **during legalization** itself. The binding limit is **end-to-end wall time**: the full pipeline from loading the benchmark to writing the **final legalized** placement must finish within **1 hour** on the provided hardware (e.g. **AMD EPYC + NVIDIA RTX 6000 Ada** per competition setup).

**Stated priority when fixing overlaps:** **Legality first** (disqualification if any overlap remains). When resolving overlaps, prioritize **minimizing degradation** of the official **Proxy Cost**:

**1.0 × Wirelength + 0.5 × Density + 0.5 × Congestion**

This is standard when continuous guidance leaves **small** residual overlap risk; post-process legalization is the price of **not** using sequential discrete masking inside a simultaneous diffusion loop.

---

## 5. Hybrids and architectural recommendation

The competition **welcomes hybrid** methods. Conceptually, you can combine grids, snapping, ordering, or multiple generators—provided the **final** placement is **continuously** legal.

**Architectural note:** For **macro** placement specifically, it is often clearest to commit to **simultaneous diffusion + post-legalization**, then hand **fixed macro** positions to an analytical placer such as **DREAMPlace** for standard cells (see Step 1 overview). **Weaving MaskPlace-style sequential grid masking directly into the parallel, joint coordinate updates** of a diffusion sampler is **computationally conflicting** (different invariants per step). If you need masking’s guarantees, a **dedicated sequential RL (or ordered) branch** is usually cleaner than forcing autoregressive masks inside a joint diffusion inner loop.

If coordinates are continuous but you use a **grid** for some checks, you might **snap** to anchors and look up feasibility—that is **not** the same as MaskPlace’s **autoregressive mask chain**. Any hybrid “mask” must be **consistent** with **joint** updates or with an **explicit ordering** and sampler design.

---

## See also

- [`step1-diffusion-model.md`](./step1-diffusion-model.md) — **[−1, 1]** normalized centers, DDPM, **backwards universal guidance** stack, synthetic training.  
- [`step3-multi-objective-proxy-to-ppa.md`](./step3-multi-objective-proxy-to-ppa.md) — HPWL vs legality weights, Pareto search.  
- [`proposed-solution-overview.md`](./proposed-solution-overview.md) — updated three-pillar overview (diffusion + legality/legalize + inference MOO).  
- [`macro-placement-competition.md`](./macro-placement-competition.md) — zero overlap, runtime, metrics.
