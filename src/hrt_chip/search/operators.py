"""SA move operators: shift, net-aware shift, swap, small cluster move."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from hrt_chip.geometry import clamp_macro_to_canvas, rects_overlap
from hrt_chip.models import MacroRect, PlacementCandidate


class OperatorKind(str, Enum):
    SHIFT = "shift"
    NET_AWARE = "net_aware"
    SWAP = "swap"
    CLUSTER = "cluster"


@dataclass(frozen=True)
class MoveProposal:
    kind: OperatorKind
    """Indices touched (macro indices in ``candidate.macros``)."""
    indices: tuple[int, ...]


def _movable_indices(candidate: PlacementCandidate, fixed_mask: list[bool] | None, hard_n: int) -> list[int]:
    fm = fixed_mask if fixed_mask is not None else [False] * len(candidate.macros)
    return [i for i in range(min(hard_n, len(candidate.macros))) if not fm[i]]


def _overlap_with_others(
    macros: list[MacroRect],
    idx: int,
    *,
    hard_n: int,
) -> bool:
    m = macros[idx]
    for j in range(hard_n):
        if j == idx:
            continue
        if rects_overlap(m, macros[j]):
            return True
    return False


def propose_shift(
    candidate: PlacementCandidate,
    rng: random.Random,
    *,
    hard_n: int,
    fixed_mask: list[bool] | None,
    canvas_w: float,
    canvas_h: float,
    max_span: float,
) -> MoveProposal | None:
    movable = _movable_indices(candidate, fixed_mask, hard_n)
    if not movable:
        return None
    idx = rng.choice(movable)
    m = candidate.macros[idx]
    old_x, old_y = m.x, m.y
    m.x += rng.uniform(-max_span, max_span)
    m.y += rng.uniform(-max_span, max_span)
    clamp_macro_to_canvas(m, canvas_w=canvas_w, canvas_h=canvas_h)
    if _overlap_with_others(candidate.macros, idx, hard_n=hard_n):
        m.x, m.y = old_x, old_y
        return None
    return MoveProposal(OperatorKind.SHIFT, (idx,))


def _net_center_for_macro(benchmark: Any, macros: list[MacroRect], macro_idx: int) -> tuple[float, float] | None:
    if benchmark is None or not hasattr(benchmark, "net_nodes"):
        return None
    cx_acc: list[float] = []
    cy_acc: list[float] = []
    for net in benchmark.net_nodes:
        nodes = net.detach().cpu().numpy().astype(np.int64).reshape(-1) if hasattr(net, "detach") else np.asarray(net, dtype=np.int64).reshape(-1)
        if macro_idx not in set(nodes.tolist()):
            continue
        for ni in nodes.tolist():
            if 0 <= ni < len(macros):
                mm = macros[ni]
                cx_acc.append(mm.x + mm.w / 2.0)
                cy_acc.append(mm.y + mm.h / 2.0)
    if not cx_acc:
        return None
    return (float(np.mean(cx_acc)), float(np.mean(cy_acc)))


def propose_net_aware(
    candidate: PlacementCandidate,
    rng: random.Random,
    benchmark: Any | None,
    *,
    hard_n: int,
    fixed_mask: list[bool] | None,
    canvas_w: float,
    canvas_h: float,
    max_span: float,
) -> MoveProposal | None:
    movable = _movable_indices(candidate, fixed_mask, hard_n)
    if not movable or benchmark is None:
        return None
    idx = rng.choice(movable)
    center = _net_center_for_macro(benchmark, candidate.macros, idx)
    if center is None:
        return None
    m = candidate.macros[idx]
    mcx = m.x + m.w / 2.0
    mcy = m.y + m.h / 2.0
    tx, ty = center
    step = max_span * 0.5
    old_x, old_y = m.x, m.y
    dist = math.hypot(tx - mcx, ty - mcy)
    if dist < 1e-12:
        return None
    mcx += step * (tx - mcx) / dist
    mcy += step * (ty - mcy) / dist
    m.x = mcx - m.w / 2.0
    m.y = mcy - m.h / 2.0
    clamp_macro_to_canvas(m, canvas_w=canvas_w, canvas_h=canvas_h)
    if _overlap_with_others(candidate.macros, idx, hard_n=hard_n):
        m.x, m.y = old_x, old_y
        return None
    return MoveProposal(OperatorKind.NET_AWARE, (idx,))


def propose_swap(
    candidate: PlacementCandidate,
    rng: random.Random,
    *,
    hard_n: int,
    fixed_mask: list[bool] | None,
    canvas_w: float,
    canvas_h: float,
) -> MoveProposal | None:
    movable = _movable_indices(candidate, fixed_mask, hard_n)
    if len(movable) < 2:
        return None
    i, j = rng.sample(movable, 2)
    a, b = candidate.macros[i], candidate.macros[j]
    if abs(a.w - b.w) > 1e-6 * max(a.w, b.w) or abs(a.h - b.h) > 1e-6 * max(a.h, b.h):
        return None
    ax, ay = a.x, a.y
    bx, by = b.x, b.y
    a.x, a.y = bx, by
    b.x, b.y = ax, ay
    clamp_macro_to_canvas(a, canvas_w=canvas_w, canvas_h=canvas_h)
    clamp_macro_to_canvas(b, canvas_w=canvas_w, canvas_h=canvas_h)
    if _overlap_with_others(candidate.macros, i, hard_n=hard_n) or _overlap_with_others(
        candidate.macros, j, hard_n=hard_n
    ):
        a.x, a.y = ax, ay
        b.x, b.y = bx, by
        return None
    return MoveProposal(OperatorKind.SWAP, (i, j))


def propose_cluster(
    candidate: PlacementCandidate,
    rng: random.Random,
    benchmark: Any | None,
    *,
    hard_n: int,
    fixed_mask: list[bool] | None,
    canvas_w: float,
    canvas_h: float,
    max_span: float,
) -> MoveProposal | None:
    movable = _movable_indices(candidate, fixed_mask, hard_n)
    if len(movable) < 2 or benchmark is None:
        return None
    # Pick a net with >=2 movable macros
    candidates_net: list[tuple[int, int]] = []
    for ni, net in enumerate(benchmark.net_nodes):
        nodes = net.detach().cpu().numpy().astype(np.int64).reshape(-1) if hasattr(net, "detach") else np.asarray(net, dtype=np.int64).reshape(-1)
        ids = [int(x) for x in nodes.tolist() if int(x) in movable]
        if len(ids) >= 2:
            candidates_net.append((ids[0], ids[1]))
    if not candidates_net:
        return None
    i, j = rng.choice(candidates_net)
    dx = rng.uniform(-max_span, max_span)
    dy = rng.uniform(-max_span, max_span)
    mi, mj = candidate.macros[i], candidate.macros[j]
    oix, oiy, ojx, ojy = mi.x, mi.y, mj.x, mj.y
    mi.x += dx
    mi.y += dy
    mj.x += dx
    mj.y += dy
    clamp_macro_to_canvas(mi, canvas_w=canvas_w, canvas_h=canvas_h)
    clamp_macro_to_canvas(mj, canvas_w=canvas_w, canvas_h=canvas_h)
    if _overlap_with_others(candidate.macros, i, hard_n=hard_n) or _overlap_with_others(
        candidate.macros, j, hard_n=hard_n
    ):
        mi.x, mi.y, mj.x, mj.y = oix, oiy, ojx, ojy
        return None
    return MoveProposal(OperatorKind.CLUSTER, (i, j))
