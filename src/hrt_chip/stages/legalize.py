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
    hard_macro_count: int | None = None,
    fixed_mask: list[bool] | None = None,
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

    hard_n = hard_macro_count if hard_macro_count is not None else len(macros)
    fixed_mask_local = fixed_mask if fixed_mask is not None else [False] * len(macros)

    for p in range(max_passes):
        if placement_is_legal(
            macros,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            hard_macro_count=hard_macro_count,
        ):
            break
        passes_used = p + 1

        changed = False
        for i in range(hard_n):
            for j in range(i + 1, hard_n):
                if not rects_overlap(macros[i], macros[j]):
                    continue
                a, b = macros[i], macros[j]
                a_fixed = bool(fixed_mask_local[i])
                b_fixed = bool(fixed_mask_local[j])

                # Move only non-fixed macros; fixed macros will be restored post-legalization,
                # so moving them here would just undo progress.
                moved = False
                if not b_fixed:
                    move = min_l1_separation_move_b(
                        a, b, canvas_w=canvas_w, canvas_h=canvas_h
                    )
                    if move is not None:
                        dx, dy = move
                        b.x += dx
                        b.y += dy
                        clamp_macro_to_canvas(b, canvas_w=canvas_w, canvas_h=canvas_h)
                        moved = True
                if not moved and (b_fixed and not a_fixed):
                    # b is fixed, try moving a instead (deterministically).
                    move = min_l1_separation_move_b(
                        b, a, canvas_w=canvas_w, canvas_h=canvas_h
                    )
                    if move is not None:
                        dx, dy = move
                        a.x += dx
                        a.y += dy
                        clamp_macro_to_canvas(a, canvas_w=canvas_w, canvas_h=canvas_h)
                        moved = True

                if moved:
                    resolved_ops += 1
                    changed = True
                else:
                    if not b_fixed:
                        _fallback_separate_pair(a, b, canvas_w=canvas_w, canvas_h=canvas_h)
                        if not rects_overlap(a, b):
                            resolved_ops += 1
                            changed = True
                    elif not a_fixed:
                        _fallback_separate_pair(b, a, canvas_w=canvas_w, canvas_h=canvas_h)
                        if not rects_overlap(a, b):
                            resolved_ops += 1
                            changed = True

        if not changed:
            break

    remaining = count_overlapping_pairs(macros)
    remaining = count_overlapping_pairs(macros, hard_macro_count=hard_macro_count)
    legal = placement_is_legal(
        macros,
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        hard_macro_count=hard_macro_count,
    )
    candidate.metadata["legal"] = legal
    candidate.metadata["legality_status"] = "legal" if legal else "illegal"
    candidate.metadata["legalization_passes"] = passes_used
    candidate.metadata["resolved_pair_ops"] = resolved_ops
    candidate.metadata["remaining_overlapping_pairs"] = remaining
    candidate.metadata["stage"] = "legalized"
    return candidate
