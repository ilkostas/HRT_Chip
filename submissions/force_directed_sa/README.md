# Force-Directed + SA Standalone Solver

Standalone classical macro placement flow:

1. Parse `netlist.pb.txt` + `initial.plc`
2. Force-directed global placement
3. Greedy legalization (spatial-hash accelerated)
4. SA local refinement (incremental objective updates)
5. Optional orientation sweep (`N/S/FN/FS`)
6. Write final `.plc`

## Quick Commands

- Immediate ibm01 normalization checkpoint:
  - `uv run python -m submissions.force_directed_sa.checkpoint_ibm01`
- Run one benchmark:
  - `uv run python -m submissions.force_directed_sa.placer -b ibm01`
- Run all benchmarks and append JSONL experiment row:
  - `uv run python -m submissions.force_directed_sa.runner`

## Files

- `parser.py`: benchmark parser
- `surrogate.py`: HPWL + ABU10 density + RUDY ABU5 congestion
- `force_directed.py`: stage 1 solver
- `legalize.py`: stage 2 legalizer
- `sa_refine.py`: stage 3 simulated annealing
- `orient.py`: stage 4 orientation refinement
- `placer.py`: end-to-end single benchmark entrypoint
- `runner.py`: all-17 sweep runner + experiment tracking
- `checkpoint_ibm01.py`: early surrogate normalization checkpoint

