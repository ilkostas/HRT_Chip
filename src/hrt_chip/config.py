"""Run configuration and reproducibility snapshot types."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunConfig:
    """User-facing configuration for a single pipeline run."""

    benchmark_id: str = "ibm01"
    seed: int = 42
    num_candidates: int = 4
    output_dir: Path = field(default_factory=lambda: Path("runs"))
    deterministic: bool = True

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["output_dir"] = str(self.output_dir)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunConfig:
        out = dict(data)
        if "output_dir" in out:
            out["output_dir"] = Path(out["output_dir"])
        return cls(**{k: v for k, v in out.items() if k in cls.__dataclass_fields__})
