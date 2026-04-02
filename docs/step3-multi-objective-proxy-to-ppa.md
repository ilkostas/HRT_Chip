# Step 3: Proxy-to-PPA Bridge (Flexible Multi-Objective Optimization) — Detailed Explanation

The competition first scores **Proxy Cost** (wirelength, density, congestion), but the **Grand Prize** path depends on **OpenROAD** outcomes: **Worst Negative Slack (WNS)**, **Total Negative Slack (TNS)**, and **Area**. This step bridges **proxy optimization** and **real PPA** by treating placement as **multi-objective**: you do not commit to a single scalar goal; you **explore** trade-offs at **inference time** and then **select** layouts using the **official proxy** so you advance to the judges’ timing tier.

Research consistently shows a **gap** between intermediate objectives (e.g. macro **Half-Perimeter Wirelength**, HPWL) and end-to-end **timing**. Optimizing **only** wirelength does **not** guarantee better slack on critical paths. The fix is to produce a **Pareto frontier**—many strong floorplans with different balances of wirelength and congestion—then **rank the full batch** with the competition proxy after legalization.

**Architecture lock-in:** The **sole** core engine is a **Denoising Diffusion Probabilistic Model (DDPM)**. Multi-objective behavior is achieved **only** via **inference-time** weights during sampling. Do **not** run **RL**, **MOPPO**, or preference-conditioned critics in parallel; the diffusion model places all objects **simultaneously** in one shot.

---

## 1. Independent cost functions (“potentials”)

Implement **standalone**, **fast** surrogates. In diffusion sampling these act as **potential functions** ϕ(·) whose gradients steer the state.

### Wirelength potential — ϕ_hpwl

Use **Half-Perimeter Wirelength (HPWL)**: for each net, bounding box over connected **pins**; sum (or weighted sum) over nets. Differentiable approximations (e.g. LogSumExp over pin coordinates) are common for gradient-based guidance.

**Implementation note (this repo):** the pipeline scoring table uses **exact pin HPWL** when the loaded `Benchmark` provides per-net pin offsets (`net_pin_dx`, `net_pin_dy` parallel to `net_nodes`). Without those fields, netlist-aware HPWL/congestion scalars are left undefined (`null` in `results.json`) so surrogates are not silently misaligned with true pin geometry.

### Congestion potential — ϕ_congestion

**Prefer RUDY** (Rectangular Uniform wire Density): deterministic, fast, and well correlated with routing congestion without a global router. RUDY builds wire density from net bounding boxes; smooth it efficiently (e.g. **5×1** 2D convolutional filters in horizontal/vertical passes). You may still emphasize **hotspots**, e.g. the **top 10%** most congested grid cells, to match competition congestion proxies.

### Legality potential — ϕ_legality (not the competition “Density” term)

**ϕ_legality** is an **overlap-style** penalty: e.g. **squared signed distance** or squared overlap area between rectangular macros. It is a **continuous, differentiable** force that pushes macros apart during the diffusion loop.

The competition’s official **“Density”** proxy is evaluated **only** on the **final discrete layout** after your **greedy legalizer** snaps macros to **exact** zero overlap. So: overlap potential is an **internal surrogate** to reach **high** legality during sampling (e.g. **~99.7%** legality before snap); **legalization** produces the layout on which **Density** (and the full proxy) is computed.

**Lagrange-style** balancing between HPWL and **ϕ_legality** during denoising is common in the diffusion placement literature; treat legality as a **constraint** tightened via **interleaved** gradient steps (see §2).

---

## 2. DDPM sampling and guidance on denoised coordinates (not on ε)

**Sampler:** Use a **DDPM** with the network trained for **ε-prediction** (noise) under **MSE** over a fixed horizon (e.g. **T = 1000** timesteps). See [`step1-diffusion-model.md`](./step1-diffusion-model.md).

**No MORL head:** You do **not** train a multi-objective RL critic or a family of policies. **(α, β, γ)** are **inference-only** knobs.

**Where gradients attach:** Guidance does **not** modify the noise target directly. At reverse step **t**, derive the predicted **fully denoised** macro coordinates **x̂₀** from the current noisy state (standard DDPM algebra from the ε-prediction). Compute:

**∇ϕ_hpwl(x̂₀)**, **∇ϕ_congestion(x̂₀)**, **∇ϕ_legality(x̂₀)**,

then form a combined guidance force **g(x_t)** (or equivalent injection into the coordinate update) from **α**, **β**, **γ** and add it to the diffusion step so the **continuous coordinates** are steered. Keep **sign conventions** (minimize vs. maximize) and scaling **consistent** with your implementation.

**Guided update (conceptual):**

**Δx_guided = Δx_model + α ∇ϕ_hpwl + β ∇ϕ_congestion + γ ∇ϕ_legality**

**Dynamic legality:** Optimize **γ** (or a Lagrange multiplier schedule) during sampling so overlap is driven down while HPWL/congestion trade off—**interleaved gradient descent** alongside the reverse process.

**RL / MOPPO:** Not used. Changing weights does **not** require retraining the DDPM; only the inference-time **(α, β, γ)** sweep changes.

---

## 3. Pareto frontier (zero retraining) and batch size **K**

Because **(α, β, γ)** apply only at **generation** time, one **fixed** checkpoint can emit **many** diverse layouts.

### Definition of “Pareto” for this stack

The frontier is explored by sweeping **discrete weight vectors** over **ϕ_hpwl** and **ϕ_congestion** surrogates. **ϕ_legality** is treated as a **strict constraint**, enforced via the **Lagrangian / dynamic weight** mechanism above so layouts are **fully legal** or **~99.7%** legal before snapping. The **Pareto** interpretation: **non-dominated trade-offs** between **wirelength** and **routing congestion** among **legal** (or pre-legalize) candidates—not a sweep that mirrors the official **1.0 / 0.5 / 0.5** proxy coefficients.

### Tuning **(α, β, γ)** vs. the official proxy

**(α, β, γ)** are **independent knobs** to **span diversity** (some runs HPWL-heavy, others congestion-heavy, others balanced). They are **not** required to match **1.0×Wirelength + 0.5×Density + 0.5×Congestion** during sampling. After **greedy legalization** and any required standard-cell placement, **rank every candidate** with the **exact** official proxy formula.

### Parallel trajectories on one GPU

**K** is limited by **single-machine, single-GPU** inference (e.g. **16–32** parallel diffusion trajectories depending on VRAM), **and** by the **1-hour wall-clock** budget per benchmark (see §4). Launch **K** trajectories with **different** weight vectors, e.g.:

- **[0.8, 0.1, 0.1]** — prioritize HPWL.  
- **[0.2, 0.7, 0.1]** — spread macros to reduce congestion.  
- **[0.4, 0.4, 0.2]** — balanced.  

Research does not prescribe a universal **K**; choose **K** from GPU memory and the time needed for DDPM (often **~2–21 minutes** per run depending on circuit scale), legalizer, and downstream steps.

### Practical one-hour budget policy

Given the one-hour hard timeout, use diffusion for candidate generation early in the run and reserve substantial wall clock for legalization + scoring over the full candidate set.

### OpenROAD during your run

**Zero** full OpenROAD (or multi-hour commercial P&R) runs fit in the **1-hour** inference budget. Organizers run the full flow **on their side** for Grand Prize evaluation (see §4).

---

## 4. Final selection pipeline and two-tier competition rules

**Wall-clock budget:** **60 minutes per benchmark design** from loading the netlist to writing the **final** layout. Budget time for: batched DDPM, **greedy legalizer**, standard-cell placement, and proxy scoring of all candidates.

**Step A — Greedy legalizer.** Snap macros to **0%** overlap so the official **Density** and other discrete proxy terms apply to a legal result.

**Step B — Proxy scoring (competition formula).** Rank **all** legalized candidates with the **exact** official proxy, e.g.:

**Proxy = 1.0 × Wirelength + 0.5 × Density + 0.5 × Congestion**

(Use the **exact** coefficients and definitions from the current competition specification.)

**Step C — What to submit.** Evaluation is **two-tiered**:

1. **Tier 1 — Proxy ranking:** Your submission is judged on **proxy cost** first. **Only** designs that place **high enough** on the proxy (e.g. **top 7** in the competition’s stated rule set) proceed.  
2. **Tier 2 — OpenROAD WNS/TNS:** Run **only** on the organizers’ side for those qualifiers.

Therefore, for **your** pipeline, the **default rule** is to **trust the proxy completely** for choosing what to submit: among your Pareto batch, submit the candidate with the **best (lowest) official proxy score**. You **cannot** use local OpenROAD timing to pick the winner without risking Tier 1; optimizing for a timing guess that hurts the proxy can **remove** you from the pool that ever gets WNS/TNS measured.

This turns the placer from a **single-guess** optimizer into a **search** over **HPWL–congestion** trade-offs, with **Tier 1** survival determined solely by the **legalized** official proxy.

## 5. Reproducibility controls (required)

Run-to-run reproducibility is mandatory for this project:

- fixed seeds for training and inference;
- persisted config snapshots per run;
- candidate-level artifact and score logging;
- deterministic execution mode for verification and regression debugging.

---

## See also

- [`step1-diffusion-model.md`](./step1-diffusion-model.md) — DDPM core, ε-prediction, and reverse sampling.  
- [`step2-position-masking.md`](./step2-position-masking.md) — diffusion legality vs masking; greedy legalization.  
- [`proposed-solution-overview.md`](./proposed-solution-overview.md) — overview (diffusion, legality/legalize, inference MOO).  
- [`macro-placement-competition.md`](./macro-placement-competition.md) — metrics, runtime, proxy vs Grand Prize criteria.
