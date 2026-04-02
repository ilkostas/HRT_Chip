# Proposed Solution Overview

This document summarizes our proposed approach to the macro placement competition: a framework built around **diffusion-based generation**, **legality via continuous guidance plus legalization** (not MaskPlace-style masking at inference), and **flexible multi-objective optimization at inference time**.

Project decision: this repository follows a **diffusion-first only** implementation strategy (no parallel RL training track).

## 1. The core generator: diffusion models

By using a diffusion model as the foundational engine, we bypass the slow, sequential trial-and-error process of standard reinforcement learning (RL). Instead of placing one block at a time, a diffusion model generates the **2D coordinates for all macros simultaneously**.

The model operates in **continuous space**: it denoises an array of coordinates representing the **center** of each macro, **normalized to the chip boundaries within [−1, 1]**. Those coordinates are **node features** in a **graph** representation of the netlist; the backbone uses an **interleaved graph neural network (GNN) and attention (Transformer)** architecture, producing a vector of size **2 × num_macros** per diffusion timestep.

### Reported HPWL (macro-only and mixed-size)

The **“50%+ HPWL vs RL”** comparison comes from *Chip Placement with Diffusion Models*. On **macro-only** ICCAD04 / ISPD-style IBM benchmarks, reported average HPWL was **2.49 × 10⁵** for the diffusion approach vs **8.72 × 10⁵** for MaskPlace-class RL. The same work reports that performance extends under **mixed-size** flows: e.g. fixed diffusion macro placements with **DREAMPlace** for standard cells yielded **8.00 × 10⁶** on ibm04 vs **10.4 × 10⁶** (MaskPlace) and **10.1 × 10⁶** (ChiPFormer).

### Training vs inference

The diffusion model is trained **offline on synthetic layouts only**, with the **standard noise-prediction loss (MSE)**—**not** on real circuits or proxy costs. **Proxy objectives (e.g. HPWL, legality)** enter **at inference time** via fixed exploration weights `(α, β, γ)` attached to each generated candidate (e.g. `pareto3`) and used for surrogate diagnostics; strict legality is enforced by greedy legalization + the official evaluator. There is **no RL or distillation** on top of the core diffusion training.

Runtime figures in the literature (e.g. **~21.2 minutes average** on ISPD circuits for the neural placement pass) measure **forward passes and placement execution**, **not** a full OpenROAD place-and-route evaluation.

## 2. Legality: not MaskPlace-style masking with diffusion

The competition requires **absolutely zero overlaps** between mixed-size macros.

**MaskPlace-style position masking** (discrete grid, binary feasibility mask per step, multiply logits by the mask) **requires sequential placement**: the mask at step *t* depends on macros **M₁:ₜ₋₁** already placed. **Simultaneous diffusion** places **all** macros together—there is **no** such sequence, so **that** masking mechanism **does not apply** to the diffusion sampler.

- **Sequential RL (e.g. MaskPlace):** can achieve **0% overlap by construction** on a discretized canvas (e.g. **224 × 224**), with **O(V)** mask updates over previously placed macros.
- **Diffusion:** uses a continuous **legality potential** (e.g. squared overlap between macro shapes) as a **guidance force** during sampling. Reported results reach **high** legality (e.g. **~0.997** on IBM benchmarks) but **do not** natively guarantee **1.0**.

Therefore, a **diffusion-based pipeline must include a fast, greedy post-processing legalization** step to snap placements to **100%** legal positions before submission, as is standard for satisfying the hard constraint.

## 3. The proxy-to-PPA bridge: multi-objective optimization at inference

Submissions are scored first on **Proxy Cost** (wirelength, density, congestion); top results are validated with **OpenROAD**: **WNS**, **TNS**, and **Area**.

For this diffusion-first stack, there is **no** separate critic/policy for objectives; use **inference-time weights** `(α, β, γ)` as fixed per-candidate exploration vectors. Hard legality is enforced by legalization + the official evaluator, and final ranking uses the official Proxy Cost—this is **hand-tuned or searched**, not a trained MORL head on the diffusion net.

**Pareto exploration:** sweep a **discrete set** of preference weights at inference (literature often illustrates on the order of **~10–15** distinct vectors). In this project, candidate selection is explicit: choose the single layout with the **lowest official Proxy Cost** after legalization.

## 4. Second stage: mixed-size and Grand Prize metrics

The diffusion literature pipeline for **mixed-size** designs is **not** macro-only end-to-end:

1. **Partition** standard cells (e.g. **hMETIS**).
2. **Diffusion** places **macros and standard-cell clusters**.
3. **Fix** macro locations; run a **traditional analytical placer** (e.g. **DREAMPlace**) for standard cells—often **under a minute**.

Plan this **second stage** explicitly if targeting full-flow **WNS / TNS / Area**, along with **legalization** for any residual overlap from continuous guidance.

## Summary

We combine **parallel diffusion placement** (synthetic-trained, MSE objective; proxy and legality via **guidance at inference**), **continuous legality potential + mandatory legalization** for **zero overlap** (not MaskPlace-style simultaneous masking), and **inference-time multi-objective weighting / Pareto search** to align proxy optimization with downstream PPA—within the competition runtime budget for the **placement** phase, with **OpenROAD** validation budgeted separately.

## Reproducibility requirement

All experiments and benchmark runs must include:

- fixed seeds,
- saved run configs,
- candidate-level score logs,
- deterministic verification mode for regression debugging.

## Related documentation

- [`step1-diffusion-model.md`](./step1-diffusion-model.md) — diffusion core (Step 1).  
- [`step2-position-masking.md`](./step2-position-masking.md) — sequential RL masking vs diffusion legality + legalization (Step 2).  
- [`step3-multi-objective-proxy-to-ppa.md`](./step3-multi-objective-proxy-to-ppa.md) — proxy-to-PPA bridge and Pareto search (Step 3).
