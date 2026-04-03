from __future__ import annotations

from submissions.force_directed_sa.surrogate import net_weighted_hpwl
from submissions.force_directed_sa.types import Benchmark

ORIENTS = ("N", "S", "FN", "FS")


def supports_orientation_search(benchmark: Benchmark) -> bool:
    mov = set(benchmark.movable_indices)
    for net in benchmark.nets:
        for pref in net.pin_refs:
            if pref.node_idx in mov and (abs(pref.x_offset) > 1e-12 or abs(pref.y_offset) > 1e-12):
                return True
    return False


def _local_hpwl_sum(benchmark: Benchmark, node_idx: int) -> float:
    return sum(net_weighted_hpwl(benchmark, benchmark.nets[ni]) for ni in benchmark.node_to_nets[node_idx])


def run_orientation_pass(benchmark: Benchmark, *, enabled: bool = True) -> bool:
    if not enabled:
        return False
    if not supports_orientation_search(benchmark):
        return False
    for idx in benchmark.movable_indices:
        node = benchmark.nodes[idx]
        best_o = node.orientation
        best_val = _local_hpwl_sum(benchmark, idx)
        for orient in ORIENTS:
            node.orientation = orient
            val = _local_hpwl_sum(benchmark, idx)
            if val < best_val:
                best_val = val
                best_o = orient
        node.orientation = best_o
    return True

