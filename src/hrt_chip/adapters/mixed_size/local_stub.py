"""No-op mixed-size backend for Phase 0 / Phase 1 handoff validation."""

from __future__ import annotations

from hrt_chip.adapters.mixed_size.base import MixedSizeBackend, MixedSizeRequest, MixedSizeResult
from hrt_chip.geometry import placement_is_legal


class LocalStubMixedSizeBackend(MixedSizeBackend):
    def run(self, request: MixedSizeRequest) -> MixedSizeResult:
        macros = list(request.fixed_macros)
        if not placement_is_legal(macros):
            return MixedSizeResult(
                ok=False,
                message="rejected: fixed macros must be pairwise non-overlapping and in-bounds",
                extra={
                    "benchmark_id": request.benchmark_id,
                    "n_macros": len(macros),
                },
            )
        return MixedSizeResult(
            ok=True,
            message="stub: mixed-size backend not invoked",
            extra={"benchmark_id": request.benchmark_id, "n_macros": len(macros)},
        )
