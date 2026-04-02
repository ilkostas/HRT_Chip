"""Axis-aligned rectangle geometry for macro legality (Phase 1)."""

from __future__ import annotations

from hrt_chip.models import MacroRect

# Tolerance for overlap detection (continuous geometry; evaluator-scale may differ).
OVERLAP_EPS = 1e-12


def overlap_area(a: MacroRect, b: MacroRect) -> float:
    """Area of intersection of two axis-aligned rectangles."""
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx2, by2 = b.x + b.w, b.y + b.h
    ix = max(0.0, min(ax2, bx2) - max(a.x, b.x))
    iy = max(0.0, min(ay2, by2) - max(a.y, b.y))
    return ix * iy


def rects_overlap(a: MacroRect, b: MacroRect, *, eps: float = OVERLAP_EPS) -> bool:
    return overlap_area(a, b) > eps


def macro_in_canvas(
    m: MacroRect,
    *,
    canvas_w: float = 1.0,
    canvas_h: float = 1.0,
    eps: float = OVERLAP_EPS,
) -> bool:
    """True if the macro lies fully inside [0, canvas_w] x [0, canvas_h]."""
    return (
        m.x >= -eps
        and m.y >= -eps
        and m.x + m.w <= canvas_w + eps
        and m.y + m.h <= canvas_h + eps
    )


def count_overlapping_pairs(macros: list[MacroRect], *, eps: float = OVERLAP_EPS) -> int:
    n = 0
    for i in range(len(macros)):
        for j in range(i + 1, len(macros)):
            if overlap_area(macros[i], macros[j]) > eps:
                n += 1
    return n


def placement_is_legal(
    macros: list[MacroRect],
    *,
    canvas_w: float = 1.0,
    canvas_h: float = 1.0,
    eps: float = OVERLAP_EPS,
) -> bool:
    """Zero pairwise overlap and all macros inside the canvas."""
    if not all(macro_in_canvas(m, canvas_w=canvas_w, canvas_h=canvas_h, eps=eps) for m in macros):
        return False
    for i in range(len(macros)):
        for j in range(i + 1, len(macros)):
            if overlap_area(macros[i], macros[j]) > eps:
                return False
    return True


def normalized_center_to_lower_left(
    cx: float,
    cy: float,
    w: float,
    h: float,
    *,
    canvas_w: float = 1.0,
    canvas_h: float = 1.0,
) -> tuple[float, float]:
    """
    Map a macro center in [-1, 1]² to lower-left (x, y) on the unit canvas [0, canvas_w] × [0, canvas_h].

    Linear map per axis: center = (c + 1) / 2 * extent, then x = center_x - w/2, y = center_y - h/2.
    """
    center_x = (cx + 1.0) * 0.5 * canvas_w
    center_y = (cy + 1.0) * 0.5 * canvas_h
    return (center_x - w / 2.0, center_y - h / 2.0)


def clamp_macro_to_canvas(
    m: MacroRect,
    *,
    canvas_w: float = 1.0,
    canvas_h: float = 1.0,
) -> None:
    """In-place clamp so the macro fits inside the canvas (may still overlap others)."""
    m.x = max(0.0, min(m.x, canvas_w - m.w))
    m.y = max(0.0, min(m.y, canvas_h - m.h))


def _overlap_with_rect_at(a: MacroRect, b: MacroRect, bx: float, by: float) -> bool:
    """Whether b placed at (bx, by) overlaps a."""
    tmp = MacroRect(name=b.name, x=bx, y=by, w=b.w, h=b.h)
    return rects_overlap(a, tmp)


def min_l1_separation_move_b(
    a: MacroRect,
    b: MacroRect,
    *,
    canvas_w: float = 1.0,
    canvas_h: float = 1.0,
    eps: float = OVERLAP_EPS,
) -> tuple[float, float] | None:
    """
    Minimum L1 translation (dx, dy) to apply to b so b does not overlap a and stays in-canvas.

    Considers axis-aligned snaps: b entirely to the right/left/above/below a.
    Deterministic tie-break: smaller dx, then smaller dy, then smaller |dx|+|dy| already implied.
    """
    candidates: list[tuple[float, float, float]] = []  # (manhattan, dx, dy)

    # To the right of a: b.x >= a.x + a.w
    nx = a.x + a.w
    if nx + b.w <= canvas_w + eps:
        if not _overlap_with_rect_at(a, b, nx, b.y):
            dx, dy = nx - b.x, 0.0
            candidates.append((abs(dx) + abs(dy), dx, dy))

    # To the left of a: b.x + b.w <= a.x
    nx = a.x - b.w
    if nx >= -eps:
        if not _overlap_with_rect_at(a, b, nx, b.y):
            dx, dy = nx - b.x, 0.0
            candidates.append((abs(dx) + abs(dy), dx, dy))

    # Above a: b.y >= a.y + a.h
    ny = a.y + a.h
    if ny + b.h <= canvas_h + eps:
        if not _overlap_with_rect_at(a, b, b.x, ny):
            dx, dy = 0.0, ny - b.y
            candidates.append((abs(dx) + abs(dy), dx, dy))

    # Below a: b.y + b.h <= a.y
    ny = a.y - b.h
    if ny >= -eps:
        if not _overlap_with_rect_at(a, b, b.x, ny):
            dx, dy = 0.0, ny - b.y
            candidates.append((abs(dx) + abs(dy), dx, dy))

    if not candidates:
        return None

    candidates.sort(key=lambda t: (t[0], t[1], t[2]))
    _, dx, dy = candidates[0]
    return (dx, dy)
