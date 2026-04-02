"""Greedy overlap removal with deterministic ordering (Phase 1)."""

from __future__ import annotations

from hrt_chip.geometry import (
    clamp_macro_to_canvas,
    count_overlapping_pairs,
    min_l1_separation_move_b,
    placement_is_legal,
    rects_overlap,
)
from hrt_chip.models import MacroRect, PlacementCandidate


def _fallback_separate_pair(a: MacroRect, b: MacroRect, *, canvas_w: float, canvas_h: float) -> None:
    """Push b minimally along +x then +y to clear overlap (legacy greedy)."""
    if not rects_overlap(a, b):
        return
    ax2 = a.x + a.w
    dx = max(0.0, ax2 - b.x)
    b.x += dx
    clamp_macro_to_canvas(b, canvas_w=canvas_w, canvas_h=canvas_h)
    if not rects_overlap(a, b):
        return
    ay2 = a.y + a.h
    dy = max(0.0, ay2 - b.y)
    b.y += dy
    clamp_macro_to_canvas(b, canvas_w=canvas_w, canvas_h=canvas_h)


def legalize_candidate(
    candidate: PlacementCandidate,
    *,
    max_passes: int = 256,
    canvas_w: float = 1.0,
    canvas_h: float = 1.0,
) -> PlacementCandidate:
    """
    Deterministic greedy legalization: fixed macro order; always adjust the higher-index macro.

    Sets candidate.metadata: legal, legality_status, legalization_passes, resolved_pair_ops,
    remaining_overlapping_pairs, stage.
    """
    macros = candidate.macros
    for m in macros:
        clamp_macro_to_canvas(m, canvas_w=canvas_w, canvas_h=canvas_h)

    resolved_ops = 0
    passes_used = 0

    for p in range(max_passes):
        if placement_is_legal(macros, canvas_w=canvas_w, canvas_h=canvas_h):
            break
        passes_used = p + 1

        changed = False
        for i in range(len(macros)):
            for j in range(i + 1, len(macros)):
                if not rects_overlap(macros[i], macros[j]):
                    continue
                a, b = macros[i], macros[j]
                move = min_l1_separation_move_b(
                    a, b, canvas_w=canvas_w, canvas_h=canvas_h
                )
                if move is not None:
                    dx, dy = move
                    b.x += dx
                    b.y += dy
                    clamp_macro_to_canvas(b, canvas_w=canvas_w, canvas_h=canvas_h)
                    resolved_ops += 1
                    changed = True
                else:
                    _fallback_separate_pair(a, b, canvas_w=canvas_w, canvas_h=canvas_h)
                    if not rects_overlap(a, b):
                        resolved_ops += 1
                        changed = True

        if not changed:
            break

    remaining = count_overlapping_pairs(macros)
    legal = placement_is_legal(macros, canvas_w=canvas_w, canvas_h=canvas_h)
    candidate.metadata["legal"] = legal
    candidate.metadata["legality_status"] = "legal" if legal else "illegal"
    candidate.metadata["legalization_passes"] = passes_used
    candidate.metadata["resolved_pair_ops"] = resolved_ops
    candidate.metadata["remaining_overlapping_pairs"] = remaining
    candidate.metadata["stage"] = "legalized"
    return candidate
