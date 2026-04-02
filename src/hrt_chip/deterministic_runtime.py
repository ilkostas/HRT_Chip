"""Centralized RNG and backend toggles for reproducible pipeline runs (Phase 6)."""

from __future__ import annotations

import os
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hrt_chip.config import RunConfig


def apply_pipeline_determinism(config: RunConfig) -> None:
    """
    Apply seeds and (optionally) strict PyTorch determinism before pipeline work.

    - ``deterministic=True``: seed Python ``random`` and NumPy (if installed);
      set ``torch.manual_seed`` on CPU for consistent PyTorch sampler streams.
    - ``deterministic_verification=True`` (implies stricter checks): also disable
      cuDNN benchmarking, enable deterministic algorithms where supported, and
      seed all CUDA devices.

    If ``deterministic`` is False, this is a no-op (non-reproducible run).
    """
    if not config.deterministic:
        return

    seed = int(config.seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if not config.deterministic_verification:
        return

    # Stricter verification mode: reduce nondeterminism from CUDA/cuDNN.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (TypeError, RuntimeError):
        # Older PyTorch or unsupported ops: warn_only path or API mismatch.
        pass
