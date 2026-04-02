"""Phase 6: reproducibility, replay verification, artifact retention."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hrt_chip.config import RunConfig
from hrt_chip.io.artifacts import apply_candidate_retention, PipelineArtifacts
from hrt_chip.pipeline import replay_from_manifest, run_pipeline
from hrt_chip.replay_verify import compare_replay_to_baseline, fingerprint_for_verification


def test_fingerprint_shape() -> None:
    r = {
        "benchmark_id": "ibm01",
        "best_candidate_id": "cand_0000",
        "best_proxy_score": 1.5,
        "evaluator_backend": "stub",
        "ranking": [
            {"candidate_id": "cand_0000", "proxy_score": 1.5, "legal": True},
        ],
    }
    fp = fingerprint_for_verification(r)
    assert fp["best_candidate_id"] == "cand_0000"
    assert len(fp["ranking_fingerprint"]) == 1


def test_compare_replay_ok() -> None:
    base = {
        "benchmark_id": "ibm01",
        "best_candidate_id": "a",
        "best_proxy_score": 3.25,
        "evaluator_backend": "stub",
        "ranking": [
            {"candidate_id": "a", "proxy_score": 3.25, "legal": True},
            {"candidate_id": "b", "proxy_score": 4.0, "legal": True},
        ],
    }
    rep = compare_replay_to_baseline(base, json.loads(json.dumps(base)))
    assert rep["ok"] is True
    assert rep["mismatches"] == []


def test_compare_replay_detects_mismatch() -> None:
    base = {
        "benchmark_id": "ibm01",
        "best_candidate_id": "a",
        "best_proxy_score": 1.0,
        "evaluator_backend": "stub",
        "ranking": [{"candidate_id": "a", "proxy_score": 1.0, "legal": True}],
    }
    bad = {**base, "best_candidate_id": "b"}
    rep = compare_replay_to_baseline(base, bad)
    assert rep["ok"] is False
    assert any("best_candidate_id" in m for m in rep["mismatches"])


def test_apply_candidate_retention_compact(tmp_path: Path) -> None:
    run_dir = tmp_path / "r1"
    cand_dir = run_dir / "candidates"
    cand_dir.mkdir(parents=True)
    for name in ("a.json", "b.json", "c.json"):
        (cand_dir / name).write_text("{}", encoding="utf-8")
    art = PipelineArtifacts(run_dir=run_dir)
    results = {
        "ranking": [
            {"candidate_id": "a", "proxy_score": 1.0, "legal": True},
            {"candidate_id": "b", "proxy_score": 2.0, "legal": True},
            {"candidate_id": "c", "proxy_score": 3.0, "legal": True},
        ],
        "best_candidate_id": "a",
    }
    apply_candidate_retention(art, results, mode="compact", top_k=None)
    assert list(cand_dir.glob("*.json")) == []


def test_apply_candidate_retention_best_only(tmp_path: Path) -> None:
    run_dir = tmp_path / "r2"
    cand_dir = run_dir / "candidates"
    cand_dir.mkdir(parents=True)
    (cand_dir / "win.json").write_text("{}", encoding="utf-8")
    (cand_dir / "lose.json").write_text("{}", encoding="utf-8")
    art = PipelineArtifacts(run_dir=run_dir)
    results = {
        "ranking": [
            {"candidate_id": "win", "proxy_score": 0.1, "legal": True},
            {"candidate_id": "lose", "proxy_score": 9.0, "legal": True},
        ],
        "best_candidate_id": "win",
    }
    apply_candidate_retention(art, results, mode="best_only")
    assert (cand_dir / "win.json").is_file()
    assert not (cand_dir / "lose.json").exists()


def test_apply_candidate_retention_compact_top_k(tmp_path: Path) -> None:
    run_dir = tmp_path / "r3"
    cand_dir = run_dir / "candidates"
    cand_dir.mkdir(parents=True)
    for cid in ("x", "y", "z"):
        (cand_dir / f"{cid}.json").write_text("{}", encoding="utf-8")
    art = PipelineArtifacts(run_dir=run_dir)
    results = {
        "ranking": [
            {"candidate_id": "x", "proxy_score": 1.0, "legal": True},
            {"candidate_id": "y", "proxy_score": 2.0, "legal": True},
            {"candidate_id": "z", "proxy_score": 3.0, "legal": True},
        ],
        "best_candidate_id": "x",
    }
    apply_candidate_retention(art, results, mode="compact", top_k=2)
    assert (cand_dir / "x.json").is_file()
    assert (cand_dir / "y.json").is_file()
    assert not (cand_dir / "z.json").exists()


def test_replay_from_manifest_verify_passes(tmp_path: Path) -> None:
    rid = "00000000-0000-0000-0000-0000000000f6"
    cfg = RunConfig(
        benchmark_id="ibm01",
        seed=11,
        num_candidates=2,
        output_dir=tmp_path,
        deterministic=True,
        evaluator_backend="stub",
    )
    run_pipeline(cfg, run_id=rid)
    manifest = tmp_path / rid / "manifest.json"
    out = replay_from_manifest(str(manifest), verify=True)
    assert out.get("replay_verification", {}).get("ok") is True
    assert (tmp_path / rid / "replay_verification.json").is_file()


def test_run_pipeline_artifact_retention_compact(tmp_path: Path) -> None:
    rid = "00000000-0000-0000-0000-0000000000f7"
    cfg = RunConfig(
        benchmark_id="ibm01",
        seed=3,
        num_candidates=2,
        output_dir=tmp_path,
        artifact_retention="compact",
        evaluator_backend="stub",
    )
    run_pipeline(cfg, run_id=rid)
    cdir = tmp_path / rid / "candidates"
    assert cdir.is_dir()
    assert list(cdir.glob("*.json")) == []


def test_deterministic_runtime_skipped_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from hrt_chip import deterministic_runtime

    called: dict[str, bool] = {"seed": False}

    def fake_seed(_: int) -> None:
        called["seed"] = True

    monkeypatch.setattr("random.seed", fake_seed)
    cfg = RunConfig(deterministic=False, seed=99)
    deterministic_runtime.apply_pipeline_determinism(cfg)
    assert called["seed"] is False
