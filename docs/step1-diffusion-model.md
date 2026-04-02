# Step 1: Diffusion Model (Core Generator) — Detailed Explanation

To build a diffusion model for chip placement, you shift perspective away from traditional decision-making algorithms and toward **generative AI**.

**Primary methodology reference:** *Chip Placement with Diffusion Models* (Lee et al., 2024). Align the implementation with that paper’s architecture, training, and guidance; the HPWL / congestion figures in §6 are **evaluated results** from that work on ICCAD04 (IBM), not illustrative ballparks.

### Implementation status (this repository)

**Phase 2 (current code)** wires a **sampler interface** and **deterministic DDPM stub** only: no PyTorch, no neural forward pass, no ResGNN/AttGNN yet. The stub emits macro centers in **`[-1, 1]`**, which the pipeline maps to unit-canvas `MacroRect` lower-left coordinates for the Phase 1 legalizer (see [`src/hrt_chip/diffusion.py`](../src/hrt_chip/diffusion.py), [`src/hrt_chip/stages/generate.py`](../src/hrt_chip/stages/generate.py)). **Phase 4** is the intended milestone for synthetic data, PyTorch training, and real reverse diffusion sampling; **Phase 3** adds guidance objectives at inference time.

## Reinforcement learning vs. diffusion

In standard **reinforcement learning (RL)**, placement is modeled as a **Markov decision process (MDP)**. The RL agent places **one macro at a time**, sequentially. If the agent commits a suboptimal placement early, it cannot easily backtrack, which can cause **cascading errors**. Because placement is sequential, evaluating the final layout is expensive, so training and inference tend to be **slow**.

A **diffusion model** inverts that pattern: instead of sequential step-by-step placement, the problem is treated as a **continuous generation task** where **all macros are placed simultaneously**.

The sections below describe how this works, the concrete neural blocks, training, guidance, and how this ties to the competition **Proxy Cost**.

---

## 1. How the diffusion process works for placement

Classical image diffusion (e.g., DALL·E, Midjourney) generates **pixels**. Here, the diffusion model generates an array of **continuous 2D coordinates** in a bounded range representing the **center** of every movable macro on the chip canvas.

### Coordinate space (normalized end-to-end)

Inside the network, macro centers live in a **continuous normalized** range (e.g. **`[-1, 1]`**) **end-to-end**. To handle **varying die sizes and aspect ratios** across benchmarks (the 17 IBM ICCAD04 circuits used in Tier 1), **preprocess** by mapping all coordinates to the respective **chip boundaries**, and **only convert back to absolute die coordinates** when exporting (e.g. to DREAMPlace or DEF).

### Forward process (adding noise)

Take a **good** layout—e.g., optimized and non-overlapping—and **gradually add Gaussian noise** to the **(x, y)** coordinates of **every movable macro** over many steps until the positions are **fully scattered** and effectively random over the canvas.

### Reverse process (denoising)

The model is trained to **reverse** this corruption. At **inference** time, sampling **starts from pure noise** (random coordinates for all macros) and applies a neural network over a fixed number of timesteps (**T = 1000**, **cosine noise schedule**) to **denoise** incrementally, moving all coordinates **together** toward a structured, high-quality layout.

Because denoising is **parallel over all macros** in each step, the **neural forward passes** for 1000 steps are on the order of **~2–21 minutes** depending on design scale and implementation details. The competition **1 hour per benchmark** wall clock is the hard budget: use headroom to **multi-sample** (e.g. **16** layouts in parallel on **48 GB VRAM**), run a fast Step 2 legalizer on each candidate, then score with the **official** Proxy Cost and keep the best.

---

## 2. Proposed neural architecture (ResGNN + AttGNN)

To predict noise and reconstruct coordinates, the network must encode **circuit context**. A netlist is naturally a **graph**: **nodes** are macros, **edges** are wires.

**Do not** use a plain stack of **GCN** or **GAT** alone. Use **interleaved** blocks as in the paper:

- **ResGNN block** — local **message passing** along netlist edges (heavily connected macros share information; coherent relative positions).
- **AttGNN block** — **global self-attention** over all macros (long-range coupling; mitigates GNN **oversmoothing** and captures pairs that are **topologically distant** but should be **spatially close** on the 2D canvas).

Alternate **ResGNN** and **AttGNN** through the depth of the network.

### Inputs

- Current **noisy coordinates** for all macros (normalized; **movable** macros only are updated by predicted noise—see below).
- Current **diffusion timestep** *t*.
- **Per-macro features**: width, height, connection count, **macro type**, and a **fixed vs movable** flag.

**Fixed macros and I/O:** ICCAD04 includes **fixed I/O** and sometimes **fixed macros**. Pass them into the model for **context**, but **do not** add the predicted noise to their coordinates—their positions stay fixed.

### Output

The network predicts **ε** (noise) to subtract under the **DDPM** formulation (see §3). Sampling updates **only movable** macro coordinates accordingly.

---

## 3. Training the model (synthetic data, ε-prediction, MSE)

A major hurdle in ML for EDA is **limited open placement data**. Following Lee et al., train the diffusion core **on synthetic data only**; the **training loss** is **noise prediction (ε)** with **mean squared error (MSE)** in the standard **DDPM** setup—**no** RL or distillation on top of that core loss.

### Synthetic data that matches real statistics

**Random rectangles alone are not enough** for strong zero-shot transfer to ICCAD04. Match **statistics**:

- **Object sizes:** use a **clipped exponential** distribution over sizes to approximate the large **spread** of real macros (on the order of **~33×** size variation in real designs).
- **Wiring:** connect macros with **exponentially decaying** edge probability vs. **physical separation** so closer blocks tend to share more wires.

### Two-stage scale (paper)

The paper uses a **two-stage** pre-training corpus: an early dataset with **smaller** circuits and a later dataset (**“v2”**) with **larger** circuits, with component counts on the order of **~1000**, closer to **clustered** real circuits. Generating on the order of **~100,000** synthetic golden layouts is a reasonable target; generating layouts is comparatively cheap vs. training.

The model learns to **reconstruct** these layouts from noise. **Full competition Proxy Cost** is **not** the training loss; **HPWL / legality** enter at **inference** via **guidance** (§4).

---

## 4. Guidance in this repository (inference only; fixed per-candidate weights)

A vanilla diffusion model produces **plausible** layouts from its training distribution. In this repository, the `(α, β, γ)` guidance weights are **fixed exploration weights** per weight-vector / candidate, not dynamically tuned Lagrange multipliers during reverse diffusion.

### What guidance does (and does not) do here

- `(α, β, γ)` are fixed per weight-vector / candidate in this repo; they are not dynamically updated during sampling.
- Candidate generation runs the diffusion sampler once per weight-vector; the deterministic diffusion stub uses these weights for deterministic diversity shifts, while the `pytorch_checkpoint` sampler records guidance for provenance but does not apply gradient-based steering inside the reverse diffusion loop.

In this repository, these potentials are computed after sampling as **cheap surrogates** for diagnostics / optional pre-evaluation rejection; they are not injected as gradients during reverse diffusion.

- **HPWL:** computed as a surrogate objective for candidate diagnostics (not per-step gradient guidance in this repo).
- **Legality:** computed as an overlap-style surrogate objective for candidate diagnostics (not per-step gradient guidance in this repo).

These surrogates are combined into the composite surrogate objective using the fixed `(α, β, γ)` weights stored on each candidate; the reverse diffusion loop does not apply gradient-based updates from these potentials.

### Dynamic Lagrangian multipliers (research concept; not implemented here)

Some diffusion placement papers adapt Lagrange multipliers λ with interleaved gradient-style updates during sampling; this repository does not currently implement that dynamic scheduling.

### Official metric **after** sampling

The **competition** Proxy Cost (e.g. **1.0×WL + 0.5×Density + 0.5×Congestion**) is used to **rank final candidates**: generate a **batch** of placements, run each through the **exact** evaluator, and select the **best**. Soft guidance yields layouts that are **highly** legal (e.g. **~99.7%**) but **rarely** perfectly overlap-free; **Step 2** (greedy legalizer) snaps to **100%** legal positions—**not** position masking **inside** the diffusion loop.

---

## 5. Integration with Step 2 and downstream

- **Step 1** is **runnable standalone**: train on synthetic data, sample normalized coordinates, export to die space.
- **No** **position masking** inside the diffusion loop (that is a different paradigm, e.g. MaskPlace-style RL). Diffusion outputs **unconstrained** continuous positions with **soft** legality pressure.
- **Pipeline:** **Diffusion** → **fast greedy legalizer** (Step 2) → **DREAMPlace** with macros fixed (standard cells) → **DEF** for judge-side OpenROAD. **OpenROAD** is **not** required inside your **1 hour** execution; judges run it on the submitted DEF.

---

## 6. Benchmarks and reported results (ICCAD04 / IBM)

**Validation alignment:** use the **17 IBM ICCAD04** benchmarks (**ibm01–ibm04, ibm06–ibm18**): standard **mixed-size** circuits with on the order of **246–537 macros** plus large numbers of standard cells.

---

## 7. Reproducibility controls (required)

For this project, reproducibility is not optional:

- Lock random seeds for data generation, training, and sampling.
- Persist the exact run config (model, optimizer, guidance weights, dataset version, benchmark id).
- Save candidate-level artifacts and scores so final selection can be replayed exactly.
- Provide a deterministic compute mode for debugging/verification, even if normal competition runs use faster non-deterministic kernels.

From **Chip Placement with Diffusion Models** (*macro-only* IBM table):

| Method | Average HPWL | Routing congestion (proxy, from paper) |
|--------|----------------|----------------------------------------|
| MaskPlace (RL) | 8.72 × 10⁵ | 345 |
| Diffusion (GNN + attention as in paper) | 2.49 × 10⁵ | 196 |

That is **over 50%** lower average HPWL than the MaskPlace baseline in that comparison. For **mixed-size** flows, the same paper reports example full-flow numbers with DREAMPlace for standard cells—re-verify on the **exact** benchmark list and evaluation flow for your submission.

**Hardware note:** the competition environment includes **NVIDIA RTX 6000 Ada (48 GB VRAM)**—memory is **not** the binding constraint for **T = 1000** and moderate batch sampling.

---

## See also

- [`step2-position-masking.md`](./step2-position-masking.md) — diffusion + greedy legalization vs MaskPlace-style sequential masking.  
- [`step3-multi-objective-proxy-to-ppa.md`](./step3-multi-objective-proxy-to-ppa.md) — proxy vs PPA, Pareto frontier, guided weights.  
- [`proposed-solution-overview.md`](./proposed-solution-overview.md) — high-level design (diffusion, legalization, objectives).  
- [`macro-placement-competition.md`](./macro-placement-competition.md) — competition rules and metrics.
