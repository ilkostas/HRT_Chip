# HRT / Partcl Macro Placement Challenge — Engineering Guidelines

**Purpose:** Decision document for the dedicated engineer. Every question below has a clear answer, rationale, and implementation consequence. No ambiguity — just build what it says.

## 1. Strategy & Scope

### 1.1 Primary Solver — Classical, Not Diffusion

**Decision:** Pivot. The diffusion pipeline (hrt-chip) is a research vehicle. For this contest, the competitive path is force-directed analytical placement + simulated annealing refinement as the primary solver.

**Rationale:**

- RePlAce (which is essentially force-directed) dominates every single benchmark at 1.4578 avg proxy cost. SA is at 2.1251. The gap is enormous.
- Diffusion-based placement is unproven at this scale, requires GPU training on synthetic data, and your hardware can't support the iteration loop.
- The leaderboard's only human entry (Will @ Partcl, 1.5338) is ~5% behind RePlAce — this is the target zone. A well-tuned force-directed + SA hybrid can land there.
- Force-directed runs in seconds per benchmark on any CPU. SA refinement adds minutes. This fits your hardware.

**What this means for the engineer:**

- Ignore `src/hrt_chip/training/`, `src/hrt_chip/adapters/diffusion/`, and the PyTorch sampler entirely for contest purposes.
- Build a standalone Python module (details in §4) that plugs into the official evaluate CLI.
- The hrt-chip repo infrastructure (artifacts, replay, CI) can optionally wrap the new solver later, but don't block on integration.

### 1.2 Success Metric

**Decision:** Beat SA baseline (2.1251 avg) convincingly. Stretch goal: approach RePlAce (1.4578 avg). A placement in the 1.5–1.8 range with zero overlaps on all 17 benchmarks is a strong CV entry and potentially a leaderboard placement.

**Rationale:**

- Beating SA is achievable with a force-directed approach even before SA refinement.
- Matching RePlAce requires getting density and congestion right, not just wirelength — this is the hard part.
- Even placing in the top 7 by proxy cost earns Tier 2 evaluation (OpenROAD flow), which is huge for credibility.

### 1.3 Timeline

**Decision:** Assume 3–4 engineering weeks before May 21, 2026 deadline.

**Recommended phasing:**

- **Week 1:** Parser + surrogate scorer + force-directed solver (get numbers on all 17 benchmarks)
- **Week 2:** SA refinement pass + orientation search + parameter tuning against surrogate
- **Week 3:** Calibrate surrogate against official eval (one session on fast machine), tune weights, fix any systematic bias
- **Week 4:** Final runs on all benchmarks with official evaluator, submission packaging, write-up

### 1.4 Deliverable Shape

**Decision:** Standalone submission file that conforms to the challenge's evaluate CLI interface. Separate from the hrt-chip repo.

**Concretely:** A single Python file (or small package) at `submissions/force_directed_sa.py` in the partcl-macro-place-challenge repo that the official harness calls. It receives benchmark info and returns macro positions. No dependency on hrt-chip at runtime.

**Optionally:** After the contest, backport the solver as a new backend in hrt-chip (new adapter under `src/hrt_chip/adapters/solver/classical.py`) for the write-up.

---

## 2. What You Have vs. What to Build

### 2.1 Official Stack — Run It Once, Not in the Loop

**Decision:** Assume the official evaluator (partcl-macro-place-challenge + external/MacroPlacement testcases) can be run locally but is too slow for iterative development. Use it only for:

- Initial calibration (5–10 placements on ibm01 to measure surrogate correlation)
- Final validation (all 17 benchmarks, one session)

If even single evaluations are too slow: Use GitHub Codespaces (free tier: 4-core, 16GB RAM, 120 hrs/month). Clone the challenge repo there, run evaluations remotely. Or Google Colab.

**Practical consequence:** The engineer's dev loop is: edit algorithm → run surrogate on all 17 benchmarks (should take &lt;60s total) → check scores → iterate. No official evaluator in the loop.

### 2.2 Surrogate Goals — HPWL + Density. Skip Congestion Initially.

**Decision:** The fast surrogate implements two of three proxy cost components:

1. **Wirelength (HPWL-based)** — highest priority, most correlated with final score
2. **Density (grid-based ABU10)** — second priority, prevents macro clustering
3. **Congestion (RUDY)** — defer to Week 2 or 3. It's the most complex to implement and the least marginal improvement to surrogate correlation.

**Rationale:**

- The official proxy cost is 1.0 × WL + 0.5 × Density + 0.5 × Congestion.
- Wirelength dominates (weight 1.0 vs 0.5 for the others). Getting HPWL right matters most.
- Density is straightforward to compute and catches the failure mode where force-directed converges to a clump.
- Congestion (RUDY + ABU5) requires routing demand estimation per grid cell with smoothing. It's worth adding eventually but not blocking on it.

**Surrogate formula for dev iterations:**

```
surrogate_cost = 1.0 × normalized_hpwl + 0.5 × density_abu10
```

**Add congestion term when ready:**

```
surrogate_cost = 1.0 × normalized_hpwl + 0.5 × density_abu10 + 0.5 × congestion_abu5
```

### 2.3 Calibration Discipline — Yes, One Structured Session

**Decision:** After Week 1, run the following calibration protocol:

1. Take ibm01 (smallest benchmark, 246 macros).
2. Generate 20 placements: your solver's output + 19 random perturbations of it (jitter each macro position by ±1–5% of canvas).
3. Score all 20 with both your surrogate AND the official evaluator.
4. Compute Spearman rank correlation between surrogate scores and official proxy costs.
5. If Spearman &lt; 0.85: investigate which component is off (compare WL/density/congestion breakdowns).
6. Adjust surrogate weights or fix bugs until Spearman ≥ 0.90.

This is a one-time activity (2–3 hours on a fast machine). The existing hrt-chip `rank_metrics.py` already computes Spearman/Kendall — reuse it if convenient.

### 2.4 Parser — Write Your Own, Lightweight

**Decision:** Write a purpose-built parser for `netlist.pb.txt` that extracts exactly what the solver needs. Don't depend on `load_full_benchmark` from hrt-chip or the official macro_place Python package.

**Rationale:**

- Independence from heavy dependencies = faster iteration.
- You only need: node list (name, type, width, height, x, y, fixed), pin offsets per node, net list (which pins connect), grid parameters from `.plc` header comments.
- The protobuf text format is regular enough for regex/line-by-line parsing.

**What to extract from `.plc` comments:**

| Comment pattern | Meaning |
|-----------------|--------|
| `#[PLACEMENT GRID] Col: 45, Row: 41` | `grid_cols`, `grid_rows` |
| `#[ROUTES PER MICRON] Hor: 65.96, Ver: 106.96` | `routes_h`, `routes_v` (for congestion) |
| `#[CONGESTION SMOOTH RANGE] Smooth Range: 2` | `smooth_range` |
| `#[OVERLAP THRESHOLD] Threshold: 0.0040` | `overlap_threshold` |

**What to extract from `netlist.pb.txt`:**

- Every `node { ... }` block: name, type (MACRO / STDCELL / PORT), width/height, x/y
- Within each node: `attr { ... }` blocks for pin offsets (`x_offset`, `y_offset`)
- Every `edge { ... }` block: the net, which node pins it connects, weight

---

## 3. Problem Details — Implementation-Affecting Decisions

### 3.1 Movable Degrees of Freedom

**Decision:** Only hard macros move. Soft macros (standard cell clusters) and ports are fixed at their initial positions throughout optimization.

**Orientation search:** Yes, but as a final greedy pass only, not in the inner optimization loop.

**Implementation:**

- After force-directed + SA converges to final positions, run one pass over all hard macros.
- For each macro, try all valid orientations (N, S, FN, FS — which flip pin offsets).
- Keep the orientation that gives lowest HPWL contribution for that macro's connected nets.
- This is O(macros × orientations × avg_pins_per_macro) — very fast, typically shaves 1–3% off wirelength.

**Note:** Check whether the official ICCAD04 benchmarks support orientation. Some may have all macros fixed to N. If the `.plc` file lists orientations, they're changeable. If the evaluator ignores orientation, skip this.

### 3.2 Legality — Legalize Once at the End

**Decision:** Allow overlaps during force-directed optimization. Legalize once after force-directed converges, before SA refinement.

**Rationale:**

- Force-directed methods naturally produce approximate overlap reduction through the density spreading force, but won't guarantee zero overlap.
- Legalizing after every step would destroy the gradient signal and make force-directed useless.
- The SA refinement phase operates on legalized positions and must maintain zero overlap (reject any SA move that creates overlap).

**Legalization strategy:**

- Sort macros by area (largest first).
- For each macro, find the nearest legal position (no overlap with already-placed macros, within canvas bounds).
- "Nearest" = minimize L2 displacement from force-directed position.
- This is a greedy one-pass legalizer. It works because force-directed already gets positions approximately right.

**Critical:** After legalization, assert zero overlaps. If any remain (shouldn't happen with a correct legalizer), flag and debug.

### 3.3 Canvas and Grid

**Decision:** Parse grid dimensions from `.plc` comments every time. Never hardcode.

**Implementation:** The grid dimensions vary per benchmark. Parse them from the `.plc` file header:

```python
# Parse from .plc header comments
grid_cols = int(re.search(r'Col:\s*(\d+)', plc_header).group(1))
grid_rows = int(re.search(r'Row:\s*(\d+)', plc_header).group(1))
cell_width = canvas_width / grid_cols
cell_height = canvas_height / grid_rows
```

Canvas dimensions come from the `netlist.pb.txt` or `.plc` comments as well.

### 3.4 Benchmark Coverage

**Decision:** Optimize for all 17 IBM benchmarks equally from day one.

**Rationale:**

- The contest ranks by average across all benchmarks. Doing well on 10 and poorly on 7 loses to a competitor who is mediocre on all 17.
- The benchmarks scale from 246 to 537 macros with 42–53% utilization — the algorithm should handle this range without per-benchmark tuning.
- Run all 17 in every surrogate evaluation. With a fast surrogate, this should take &lt;60s total.

**Development order:**

1. Debug on ibm01 (smallest, fastest).
2. Validate on ibm01 + ibm09 + ibm17 (small / medium / large).
3. Final tuning on all 17.

---

## 4. Algorithm Specification

This is the core section for the engineer. Build exactly this.

### 4.1 Architecture Overview

```
Input: netlist.pb.txt + initial .plc
  │
  ├─ Parse benchmark (§4.2)
  ├─ Initialize positions (§4.3)
  │
  ├─ Stage 1: Force-Directed Placement (§4.4)
  │   └─ ~200-500 iterations, <10s per benchmark
  │
  ├─ Stage 2: Greedy Legalization (§4.5)
  │   └─ One pass, <1s per benchmark
  │
  ├─ Stage 3: SA Refinement (§4.6)
  │   └─ ~50K-200K iterations, 30-120s per benchmark
  │
  ├─ Stage 4: Orientation Search (§4.7)
  │   └─ One pass, <1s per benchmark
  │
  └─ Output: updated .plc file
```

**Total runtime target:** &lt;5 minutes per benchmark (well under 1-hour limit).

### 4.2 Benchmark Parser

Parse `netlist.pb.txt` into:

```python
@dataclass
class Node:
    name: str
    node_type: str        # "MACRO", "STDCELL", "PORT"
    width: float
    height: float
    x: float              # center x
    y: float              # center y
    is_fixed: bool        # soft macros + ports = True
    pins: List[Pin]       # (x_offset, y_offset) relative to center

@dataclass
class Net:
    pin_indices: List[Tuple[int, int]]  # (node_idx, pin_idx)
    weight: float

@dataclass
class Benchmark:
    canvas_width: float
    canvas_height: float
    grid_cols: int
    grid_rows: int
    nodes: List[Node]
    nets: List[Net]
    movable_indices: List[int]  # indices of hard macros that move
```

**Pin coordinates for HPWL:** For a macro with center `(cx, cy)` and a pin with offset `(px, py)`, the pin's absolute position is `(cx + px, cy + py)`. When the macro moves, pin positions move with it.

**Important parsing detail:** In the protobuf text format, `attr` blocks inside nodes define pin offsets. The `input` and `output` entries within `attr` specify pin metadata. Net (`edge`) blocks list which node indices and pin indices connect. Parse carefully — the indexing matters.

### 4.3 Initialization

**Decision:** Use the benchmark's provided initial placement as the starting point.

**Rationale:**

- The ICCAD04 benchmarks come with an initial hand-crafted placement.
- Starting from there vs. random reduces force-directed iterations needed by 3–5×.
- This is NOT "hardcoding" — you're using it as an initialization; the algorithm optimizes from there.

### 4.4 Force-Directed Placement (Stage 1)

This is the core placement engine. The idea: macros are pulled toward their connected nets' centers (wirelength force) and pushed away from dense regions (density force).

Per iteration, for each movable macro `i`:

**Wirelength force (attraction to net centers):**

```python
f_wl = (0, 0)
for each net n connected to macro i:
    # Compute weighted center of all OTHER pins on net n
    cx, cy = weighted_center_of_net_n_excluding_macro_i
    # Force pulls macro toward this center
    dx = cx - macro_i.x
    dy = cy - macro_i.y
    f_wl += net_weight * (dx, dy)
```

**Density force (repulsion from dense regions):**

```python
f_den = (0, 0)
# Find grid cell(s) that macro i occupies
gx, gy = grid_cell_of(macro_i.x, macro_i.y)
# Compute density gradient: push toward less dense neighbors
for each neighboring grid cell (nx, ny):
    density_diff = density[gx][gy] - density[nx][ny]
    direction = normalize((nx - gx, ny - gy))
    f_den += density_diff * direction
```

**Position update:**

```python
# Step size decreases over iterations (annealing)
step = initial_step * (1 - iteration / max_iterations)
macro_i.x += step * (alpha * f_wl.x + beta * f_den.x)
macro_i.y += step * (alpha * f_wl.y + beta * f_den.y)

# Clamp to canvas bounds (account for macro width/height)
macro_i.x = clamp(macro_i.x, w/2, canvas_width - w/2)
macro_i.y = clamp(macro_i.y, h/2, canvas_height - h/2)
```

**Key parameters to tune:**

- `alpha` (wirelength weight): start at 1.0
- `beta` (density weight): start at 0.5, increase over iterations (annealing schedule)
- `initial_step`: 0.1 × canvas_diagonal
- `max_iterations`: 200–500

**Density weight schedule:** `beta = beta_initial + (beta_final - beta_initial) * (iter / max_iter)^2`

- `beta_initial = 0.1`, `beta_final = 2.0` (density becomes dominant toward convergence)

**Optimization for speed:** Use NumPy arrays. Store all macro positions as a `(N, 2)` array. Vectorize net center computation. Density grid is a 2D array updated each iteration.

### 4.5 Greedy Legalization (Stage 2)

After force-directed, remove any remaining overlaps:

```python
def legalize(macros, canvas_w, canvas_h):
    # Sort by area descending — place biggest first
    sorted_indices = sorted(movable_indices, key=lambda i: macros[i].area, reverse=True)
    placed = []  # list of (x, y, w, h) already legalized

    for idx in sorted_indices:
        m = macros[idx]
        # Try current position first
        if not overlaps_any(m.x, m.y, m.w, m.h, placed, canvas_w, canvas_h):
            placed.append((m.x, m.y, m.w, m.h))
            continue

        # Search nearby positions in expanding radius
        best_pos = None
        best_dist = float('inf')
        for radius in [1, 2, 4, 8, 16, 32, 64]:
            step = min(m.w, m.h) * 0.25
            for dx in np.arange(-radius * step, radius * step + step, step):
                for dy in np.arange(-radius * step, radius * step + step, step):
                    nx, ny = m.x + dx, m.y + dy
                    if in_canvas(nx, ny, m.w, m.h, canvas_w, canvas_h):
                        if not overlaps_any(nx, ny, m.w, m.h, placed, canvas_w, canvas_h):
                            dist = dx*dx + dy*dy
                            if dist < best_dist:
                                best_dist = dist
                                best_pos = (nx, ny)
            if best_pos is not None:
                break

        macros[idx].x, macros[idx].y = best_pos
        placed.append((best_pos[0], best_pos[1], m.w, m.h))

    assert count_overlaps(macros) == 0, "Legalization failed"
```

**Performance note:** For 500+ macros, the naive overlap check is O(N²) per macro. Use a spatial index (simple grid hash) to make it O(N) amortized.

### 4.6 SA Refinement (Stage 3)

Starting from the legalized placement, refine with simulated annealing.

**Move operators** (choose one randomly each step):

1. **Single-macro shift (70% probability):** Pick a random movable macro, shift by a random vector with magnitude proportional to temperature. Accept only if no overlaps created.
2. **Two-macro swap (20% probability):** Swap positions of two randomly chosen movable macros of similar size. Check both for overlaps.
3. **Single-macro shift toward net center (10% probability):** Pick a random macro, move it toward the weighted center of its most-connected net. This is a "smart" move that biases toward HPWL reduction.

**Acceptance criterion (standard Metropolis):**

```python
delta = new_surrogate_cost - old_surrogate_cost
if delta < 0:
    accept  # always accept improvements
elif random() < exp(-delta / temperature):
    accept  # sometimes accept worsening moves
else:
    reject
```

**Cooling schedule:**

```python
T_initial = 0.1 * current_surrogate_cost  # ~10% of starting cost
T_final = 1e-6
cooling_rate = (T_final / T_initial) ** (1 / num_iterations)
# Geometric: T *= cooling_rate each step
```

**Critical optimization — incremental cost update:**

Do NOT recompute full HPWL + density for every SA move. Instead:

- When you move macro `i`, only recompute HPWL for the nets connected to macro `i`.
- Only update the density grid cells that macro `i` enters/leaves.

This turns each SA step from O(nets + grid) to O(nets_of_macro_i + ~4 grid cells).

**SA iterations:** 100K–500K depending on benchmark size. Use wall-clock budget: run SA for up to 120 seconds per benchmark.

### 4.7 Orientation Search (Stage 4)

Final greedy pass:

```python
for each movable macro i:
    best_orient = current_orientation
    best_hpwl = hpwl_contribution(macro_i)
    for orient in ["N", "S", "FN", "FS"]:
        set_orientation(macro_i, orient)  # flips pin offsets
        h = hpwl_contribution(macro_i)
        if h < best_hpwl:
            best_hpwl = h
            best_orient = orient
    set_orientation(macro_i, best_orient)
```

**Pin offset flipping rules:**

| Orientation | Transform |
|-------------|-----------|
| N | `(px, py)`: default |
| S | `(-px, -py)`: 180° rotation |
| FN | `(-px, py)`: flip horizontal |
| FS | `(px, -py)`: flip vertical |

---

## 5. Surrogate Scoring Implementation

### 5.1 HPWL (Wirelength)

```python
def compute_hpwl(nodes, nets):
    total_hpwl = 0.0
    for net in nets:
        xs, ys = [], []
        for (node_idx, pin_idx) in net.pin_indices:
            node = nodes[node_idx]
            px, py = node.pins[pin_idx]
            xs.append(node.x + px)
            ys.append(node.y + py)
        total_hpwl += net.weight * ((max(xs) - min(xs)) + (max(ys) - min(ys)))
    return total_hpwl
```

**Normalized HPWL** (to match official proxy wirelength cost):

```python
# Normalization: divide by (num_nets * (canvas_diag))
# The exact normalization matches the plc_client formula:
# proxy_wl = sum_over_nets(weight * (bbox_w + bbox_h)) / (num_nets * (canvas_w + canvas_h))
normalized_hpwl = total_hpwl / (num_nets * (canvas_width + canvas_height))
```

Verify this normalization during calibration (§2.3). The TILOS documentation says wirelength cost is typically between 0 and 1.

### 5.2 Density (ABU10)

```python
def compute_density(nodes, grid_cols, grid_rows, canvas_w, canvas_h):
    cell_w = canvas_w / grid_cols
    cell_h = canvas_h / grid_rows
    grid = np.zeros((grid_rows, grid_cols))

    for node in nodes:
        # Compute overlap of this node's bounding box with each grid cell it touches
        x_lo = node.x - node.width / 2
        x_hi = node.x + node.width / 2
        y_lo = node.y - node.height / 2
        y_hi = node.y + node.height / 2

        col_lo = max(0, int(x_lo / cell_w))
        col_hi = min(grid_cols - 1, int(x_hi / cell_w))
        row_lo = max(0, int(y_lo / cell_h))
        row_hi = min(grid_rows - 1, int(y_hi / cell_h))

        for r in range(row_lo, row_hi + 1):
            for c in range(col_lo, col_hi + 1):
                # Overlap area between node bbox and grid cell
                ox_lo = max(x_lo, c * cell_w)
                ox_hi = min(x_hi, (c + 1) * cell_w)
                oy_lo = max(y_lo, r * cell_h)
                oy_hi = min(y_hi, (r + 1) * cell_h)
                overlap_area = max(0, ox_hi - ox_lo) * max(0, oy_hi - oy_lo)
                grid[r][c] += overlap_area / (cell_w * cell_h)

    # ABU10: average of top 10% densest cells
    flat = grid.flatten()
    flat.sort()
    top_10_pct = flat[int(len(flat) * 0.9):]
    density_cost = 0.5 * np.mean(top_10_pct)  # the 0.5 is part of the density formula itself
    return density_cost
```

**Important:** The 0.5 multiplier is INTERNAL to the density cost formula (per TILOS documentation). It is applied before the global 0.5 × Density weight in the proxy cost formula. So the effective weight on `density_abu10_raw` is 0.5 * 0.5 = 0.25 in the final proxy cost. Verify during calibration.

### 5.3 Congestion (RUDY + ABU5) — Implement in Week 2

```python
def compute_congestion(nodes, nets, grid_cols, grid_rows, canvas_w, canvas_h,
                        routes_h, routes_v, smooth_range):
    cell_w = canvas_w / grid_cols
    cell_h = canvas_h / grid_rows
    h_congestion = np.zeros((grid_rows, grid_cols))
    v_congestion = np.zeros((grid_rows, grid_cols))

    for net in nets:
        # Compute pin positions
        xs, ys = [], []
        for (node_idx, pin_idx) in net.pin_indices:
            node = nodes[node_idx]
            px, py = node.pins[pin_idx]
            xs.append(node.x + px)
            ys.append(node.y + py)

        bbox_w = max(xs) - min(xs)
        bbox_h = max(ys) - min(ys)
        if bbox_w < 1e-9 and bbox_h < 1e-9:
            continue

        # RUDY: uniform demand in bounding box
        # Horizontal demand = bbox_h / (bbox_area), vertical = bbox_w / (bbox_area)
        num_pins = len(xs)
        bbox_area = max(bbox_w * bbox_h, cell_w * cell_h)  # floor at one cell
        h_demand = net.weight * bbox_h / bbox_area
        v_demand = net.weight * bbox_w / bbox_area

        # Distribute across grid cells in bounding box
        col_lo = max(0, int(min(xs) / cell_w))
        col_hi = min(grid_cols - 1, int(max(xs) / cell_w))
        row_lo = max(0, int(min(ys) / cell_h))
        row_hi = min(grid_rows - 1, int(max(ys) / cell_h))

        for r in range(row_lo, row_hi + 1):
            for c in range(col_lo, col_hi + 1):
                h_congestion[r][c] += h_demand
                v_congestion[r][c] += v_demand

    # Optional: smooth with smooth_range
    # (Apply uniform box filter of size 2*smooth_range+1)

    # Normalize by routing capacity
    h_congestion /= (routes_h * cell_h)
    v_congestion /= (routes_v * cell_w)

    # ABU5: concatenate h and v, take average of top 5%
    all_congestion = np.concatenate([h_congestion.flatten(), v_congestion.flatten()])
    all_congestion.sort()
    top_5_pct = all_congestion[int(len(all_congestion) * 0.95):]
    congestion_cost = np.mean(top_5_pct)
    return congestion_cost
```

**Note:** This is an approximation. The exact TILOS congestion formula has nuances around smoothing and normalization. Calibrate against official evaluator.

---

## 6. Compute & Workflow

### 6.1 Machine Profile

**Assumption:** CPU-only, limited RAM, slow single-threaded performance. No usable GPU.

**Consequence:**

- All computation in NumPy (vectorized) on CPU. No PyTorch, no GPU.
- Surrogate eval over all 17 benchmarks must complete in &lt;60 seconds.
- Per-benchmark solver runtime target: &lt;5 minutes.
- Total wall time for full run over 17 benchmarks: &lt;90 minutes.

### 6.2 Iteration Budget

- **Dev loop:** Algorithm change → surrogate eval over 17 benchmarks → inspect table. Target: &lt;60s for the full sweep. This allows 50+ iterations per hour of engineering time.
- **Nightly run:** After each day's changes, kick off a full run overnight with more SA iterations (500K per benchmark). Review results in the morning.

### 6.3 Official Evaluation

**Decision:** Use GitHub Codespaces for official evaluator runs.

**Setup** (one-time, ~15 min):

1. Create a Codespace on the partcl-macro-place-challenge repo.
2. `git submodule update --init external/MacroPlacement`
3. `uv sync`
4. Verify: `uv run evaluate submissions/examples/greedy_row_placer.py -b ibm01`

**Workflow:**

1. Push your solver + saved placements to a branch.
2. Open Codespace, pull branch.
3. Run `uv run evaluate your_solver.py --all`
4. Record results. Close Codespace when done (preserves free hours).

---

## 7. Team & Process

### 7.1 Solo vs. Team

**Assumption:** You + one dedicated engineer. You own strategy and write-up. Engineer owns implementation and final runs.

### 7.2 Experiment Tracking

**Decision:** Lightweight. A JSON file per run with:

```json
{
    "timestamp": "2026-04-10T14:30:00",
    "git_hash": "abc1234",
    "parameters": {"alpha": 1.0, "beta_final": 2.0, "sa_iters": 200000},
    "results": {
        "ibm01": {"surrogate": 1.23, "official": null},
        "ibm02": {"surrogate": 1.45, "official": null},
        "...": {},
        "average": 1.67
    }
}
```

Append to a JSONL file (`experiments.jsonl`). Review with a simple script that prints a leaderboard of your own runs. Don't over-engineer this — a text file you grep is fine.

---

## 8. Clarifications on Cost Function Details

### 8.1 Density Formula

**Answer:** The 0.5 × average_of_top_10% is the standalone density cost as computed internally by the `plc_client`. This is the value that then gets multiplied by the global weight of 0.5 in the 1.0×WL + 0.5×D + 0.5×C formula.

**So the full chain is:**

1. Compute grid cell densities (area ratios, can exceed 1.0 if overlap)
2. Sort densities, take average of top 10% → `raw_density`
3. Multiply by 0.5 → `density_cost` (this is what the plc_client reports)
4. In proxy cost formula: 0.5 × `density_cost`

**Effective contribution:** 0.5 × 0.5 × raw_density_abu10 = 0.25 × raw_density_abu10

Verify during calibration. If the official evaluator returns a density component and it doesn't match your surrogate, check this double-0.5 factor.

### 8.2 Congestion (RUDY + ABU5)

**Decision:** Start with deliberately rough RUDY (uniform demand in HPWL bounding box, as specified in §5.3). For the first two weeks, you can even omit congestion entirely from the surrogate. Add it in Week 2–3 and recalibrate.

**Rationale:** Wirelength minimization tends to implicitly reduce congestion (shorter nets = less routing demand). Density spreading also helps. You'll get 80% of the way there without an explicit congestion term. The congestion surrogate is a refinement for squeezing out the last few percent.

---

## 9. File Structure

```
submissions/
└── force_directed_sa/
    ├── placer.py              # Main entry point (conforms to challenge evaluate CLI)
    ├── parser.py              # Benchmark parser (netlist.pb.txt + .plc)
    ├── force_directed.py      # Stage 1: Force-directed engine
    ├── legalize.py            # Stage 2: Greedy legalizer
    ├── sa_refine.py           # Stage 3: Simulated annealing
    ├── orient.py              # Stage 4: Orientation search
    ├── surrogate.py           # Fast HPWL + density + congestion scoring
    ├── experiments.jsonl      # Experiment tracking log
    └── README.md              # Write-up for CV / submission
```

Or if single-file is required by the challenge harness, merge into one `placer.py` with clear section comments.

---

## 10. Definition of Done (per week)

### Week 1

- Parser reads all 17 IBM benchmarks correctly (node count, net count match expected)
- HPWL surrogate produces scores for all 17 benchmarks
- Force-directed solver runs on all 17 benchmarks, produces placements with &lt;5% overlap area
- Greedy legalizer produces zero-overlap placements on all 17 benchmarks
- Surrogate scores are in plausible range (compare to baseline table in README)

### Week 2

- SA refinement improves surrogate score by ≥5% on average
- Density term added to surrogate
- Orientation search pass implemented
- All 17 benchmarks run end-to-end in &lt;90 minutes total

### Week 3

- Calibration session completed: Spearman ≥ 0.85 between surrogate and official proxy
- Official evaluator numbers on all 17 benchmarks
- Average proxy cost &lt; SA baseline (2.1251)
- Parameter tuning based on official results

### Week 4

- Final official eval: target avg proxy cost &lt; 1.85
- Zero overlaps confirmed on all 17 benchmarks
- Submission packaged and tested on clean clone
- README / write-up completed
- Submitted via Google Form before May 21

---

## Appendix A: Quick Reference — Baseline Scores to Beat

| Target | Avg proxy cost | Notes |
|--------|----------------|-------|
| Greedy Row (demo) | 2.2109 | Trivial baseline, must beat easily |
| SA Baseline | 2.1251 | Minimum viable target |
| Will (Partcl) | 1.5338 | Current leaderboard #1 |
| RePlAce Baseline | 1.4578 | Gold standard, stretch goal |

---

## Appendix B: IBM Benchmark Quick Stats

| Benchmark | Macros | Nets | Canvas | SA | RePlAce |
|-----------|--------|------|--------|-----|---------|
| ibm01 | 246 | 7,269 | 22.9×23.0 | 1.317 | 0.998 |
| ibm09 | 369 | 11,463 | 28.1×28.5 | 1.388 | 1.119 |
| ibm17 | 517 | 15,741 | 33.3×33.7 | 3.673 | 1.645 |

(Full table in challenge README.)

---

## Appendix C: Common Pitfalls

- **Forgetting pin offsets:** HPWL must use pin positions, not macro centers. A macro center at (10, 10) with a pin offset at (2, -1) means pin is at (12, 9). This matters enormously.
- **Density includes ALL nodes:** Not just hard macros — soft macros and standard cell clusters contribute to grid density too. They're fixed but they take area.
- **Canvas bounds:** Macro centers aren't constrained to canvas; macro EDGES are. A macro of width 5 at x=2 has its left edge at x=-0.5, which is illegal. Clamp: x ≥ width/2 and x ≤ canvas_width - width/2.
- **Net weight:** Some nets have weight &gt; 1 (especially nets to soft macro pins with weight factors). Multiply HPWL by net weight.
- **Overlap check precision:** Use a small epsilon (1e-6) for overlap checks. Exact floating-point equality is fragile.
- **SA acceptance too aggressive:** If temperature starts too high, SA will accept terrible moves and waste iterations undoing them. Start T at ~10% of initial cost.
