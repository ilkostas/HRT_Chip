"""Smoke tests for deterministic stub pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from hrt_chip.config import RunConfig
from hrt_chip.geometry import placement_is_legal
from hrt_chip.models import MacroRect, PlacementCandidate
from hrt_chip.pipeline import run_pipeline
from hrt_chip.stages.legalize import legalize_candidate


def test_run_pipeline_deterministic(tmp_path: Path) -> None:
    cfg = RunConfig(
        benchmark_id="ibm01",
        seed=7,
        num_candidates=3,
        output_dir=tmp_path,
        deterministic=True,
    )
    r1 = run_pipeline(cfg, run_id="00000000-0000-0000-0000-000000000001")
    r2 = run_pipeline(cfg, run_id="00000000-0000-0000-0000-000000000001")
    assert r1["best_candidate_id"] == r2["best_candidate_id"]
    assert r1["best_proxy_score"] == r2["best_proxy_score"]
    assert len(r1["ranking"]) == 3
    for row in r1["ranking"]:
        assert row["legal"] is True
        assert row["proxy_score"] != float("inf")


def test_legalize_removes_overlap_stub() -> None:
    c = PlacementCandidate(
        candidate_id="t",
        benchmark_id="ibm01",
        macros=[
            MacroRect("a", 0.0, 0.0, 0.5, 0.5),
            MacroRect("b", 0.2, 0.2, 0.5, 0.5),
        ],
    )
    legalize_candidate(c)
    assert c.metadata.get("legal") is True
    assert c.metadata.get("legality_status") == "legal"
    assert placement_is_legal(c.macros)


def test_legalize_clamps_to_canvas() -> None:
    c = PlacementCandidate(
        candidate_id="t",
        benchmark_id="ibm01",
        macros=[
            MacroRect("a", -0.5, 2.0, 0.2, 0.2),
        ],
    )
    legalize_candidate(c)
    assert c.macros[0].x == pytest.approx(0.0)
    assert c.macros[0].y == pytest.approx(0.8)
    assert c.metadata.get("legal") is True


def test_legalize_unsatisfiable_pair_stays_illegal() -> None:
    """Two full-chip macros cannot be separated in a unit canvas."""
    c = PlacementCandidate(
        candidate_id="t",
        benchmark_id="ibm01",
        macros=[
            MacroRect("a", 0.0, 0.0, 1.0, 1.0),
            MacroRect("b", 0.0, 0.0, 1.0, 1.0),
        ],
    )
    legalize_candidate(c, max_passes=64)
    assert c.metadata.get("legal") is False
    assert c.metadata.get("legality_status") == "illegal"
    assert not placement_is_legal(c.macros)


def test_pipeline_skips_mixed_size_when_illegal(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_gen(**kwargs: object) -> list[PlacementCandidate]:
        return [
            PlacementCandidate(
                candidate_id="cand_0000",
                benchmark_id="ibm01",
                macros=[
                    MacroRect("a", 0.0, 0.0, 1.0, 1.0),
                    MacroRect("b", 0.0, 0.0, 1.0, 1.0),
                ],
                metadata={"stage": "generated"},
            ),
        ]

    monkeypatch.setattr("hrt_chip.pipeline.generate_candidates", fake_gen)
    cfg = RunConfig(
        benchmark_id="ibm01",
        seed=1,
        num_candidates=1,
        output_dir=tmp_path,
        deterministic=True,
    )
    r = run_pipeline(cfg, run_id="00000000-0000-0000-0000-000000000099")
    assert len(r["ranking"]) == 1
    row = r["ranking"][0]
    assert row["legal"] is False
    assert row["proxy_score"] == float("inf")

    import json

    cand_path = tmp_path / r["manifest"]["run_id"] / "candidates" / "cand_0000.json"
    data = json.loads(cand_path.read_text(encoding="utf-8"))
    assert data["metadata"]["mixed_size"]["ok"] is False
    assert "skipped" in data["metadata"]["mixed_size"]["message"]
