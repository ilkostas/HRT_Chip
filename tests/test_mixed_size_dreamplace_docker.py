"""Unit tests for DreamPlace Docker mixed-size backend (mocked docker)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from hrt_chip.adapters.mixed_size.base import MixedSizeRequest
from hrt_chip.adapters.mixed_size.contracts import INPUT_JSON_NAME, OUTPUT_JSON_NAME, OUTPUT_SCHEMA_V1
from hrt_chip.adapters.mixed_size.dreamplace_docker import (
    REAL_DOCKER_VARIANT,
    DreamPlaceDockerBackend,
)
from hrt_chip.adapters.mixed_size.runner import build_input_payload, parse_extra_docker_args, write_input_json
from hrt_chip.config import RunConfig
from hrt_chip.models import MacroRect


def _legal_macros() -> list[MacroRect]:
    return [
        MacroRect("m0", 0.1, 0.1, 0.1, 0.1),
        MacroRect("m1", 0.5, 0.5, 0.1, 0.1),
    ]


def test_parse_extra_docker_args_empty() -> None:
    assert parse_extra_docker_args(None) == []
    assert parse_extra_docker_args("  ") == []


def test_parse_extra_docker_args_splits() -> None:
    assert parse_extra_docker_args("-e FOO=bar -v /a:/b:ro") == ["-e", "FOO=bar", "-v", "/a:/b:ro"]


def test_build_input_payload_optional_flow(tmp_path: Path) -> None:
    macros = _legal_macros()
    p = build_input_payload(
        benchmark_id="ibm01",
        candidate_id="c1",
        seed=1,
        canvas_w=1.0,
        canvas_h=1.0,
        macros=macros,
        flow="mixed_size_real",
    )
    assert p.get("flow") == "mixed_size_real"


def test_build_input_payload_roundtrip_keys(tmp_path: Path) -> None:
    macros = _legal_macros()
    p = build_input_payload(
        benchmark_id="ibm01",
        candidate_id="cand_0001",
        seed=7,
        canvas_w=1.0,
        canvas_h=1.0,
        macros=macros,
    )
    write_input_json(tmp_path, p)
    loaded = json.loads((tmp_path / INPUT_JSON_NAME).read_text(encoding="utf-8"))
    assert loaded["benchmark_id"] == "ibm01"
    assert loaded["candidate_id"] == "cand_0001"
    assert len(loaded["macros"]) == 2


def test_dreamplace_backend_docker_command_construction(tmp_path: Path) -> None:
    work = tmp_path / "mixed_size"
    work.mkdir()
    cand_dir = work / "cand_0001"
    cand_dir.mkdir()

    captured: list[list[str]] = []

    def fake_run(cmd, capture_output, text, timeout, check):  # type: ignore[no-untyped-def]
        captured.append(list(cmd))
        assert "-v" in cmd
        i = cmd.index("-v")
        assert str(cand_dir.resolve()) in cmd[i + 1]
        assert "/work" in cmd[i + 1]
        assert cmd[-1] == "hrt-chip-dreamplace:local"
        out = {
            "schema": OUTPUT_SCHEMA_V1,
            "ok": True,
            "message": "ok",
            "backend_runtime_seconds": 0.01,
            "density_overflow": 0.1,
            "rudy_or_route_proxy": 0.2,
            "hmetis_invoked": True,
            "dreamplace_invoked": False,
            "placement_mode": "analytical_proxy",
        }
        (cand_dir / OUTPUT_JSON_NAME).write_text(json.dumps(out), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    cfg = RunConfig(
        mixed_size_backend="dreamplace",
        dreamplace_docker_image="hrt-chip-dreamplace:local",
        dreamplace_docker_timeout_seconds=120,
        dreamplace_mount_testcase=False,
        dreamplace_docker_extra_args="--memory 512m",
    )
    backend = DreamPlaceDockerBackend(cfg)
    req = MixedSizeRequest(
        benchmark_id="ibm01",
        fixed_macros=_legal_macros(),
        seed=1,
        candidate_id="cand_0001",
        work_dir_host=work,
        testcase_root_host=None,
        canvas_w=1.0,
        canvas_h=1.0,
    )
    with patch("hrt_chip.adapters.mixed_size.runner.subprocess.run", side_effect=fake_run):
        res = backend.run(req)
    assert res.ok is True
    assert res.extra.get("backend") == "dreamplace"
    assert res.extra.get("density_overflow") == 0.1
    assert res.extra.get("rudy_or_route_proxy") == 0.2
    assert "--memory" in captured[0]
    assert "512m" in captured[0]


def test_dreamplace_backend_timeout(tmp_path: Path) -> None:
    work = tmp_path / "mixed_size"
    work.mkdir()

    def boom(cmd, capture_output, text, timeout, check):  # type: ignore[no-untyped-def]
        raise subprocess.TimeoutExpired(cmd, timeout)

    cfg = RunConfig(mixed_size_backend="dreamplace", dreamplace_docker_timeout_seconds=5)
    backend = DreamPlaceDockerBackend(cfg)
    req = MixedSizeRequest(
        benchmark_id="ibm01",
        fixed_macros=_legal_macros(),
        seed=1,
        candidate_id="c1",
        work_dir_host=work,
        canvas_w=1.0,
        canvas_h=1.0,
    )
    with patch("hrt_chip.adapters.mixed_size.runner.subprocess.run", side_effect=boom):
        res = backend.run(req)
    assert res.ok is False
    assert "timeout" in res.message.lower()


def test_dreamplace_backend_missing_output_json(tmp_path: Path) -> None:
    work = tmp_path / "mixed_size"
    work.mkdir()

    def ok_no_output(cmd, capture_output, text, timeout, check):  # type: ignore[no-untyped-def]
        return subprocess.CompletedProcess(cmd, 0, "", "")

    cfg = RunConfig(mixed_size_backend="dreamplace")
    backend = DreamPlaceDockerBackend(cfg)
    req = MixedSizeRequest(
        benchmark_id="ibm01",
        fixed_macros=_legal_macros(),
        seed=1,
        candidate_id="c1",
        work_dir_host=work,
        canvas_w=1.0,
        canvas_h=1.0,
    )
    with patch("hrt_chip.adapters.mixed_size.runner.subprocess.run", side_effect=ok_no_output):
        res = backend.run(req)
    assert res.ok is False
    assert "output.json" in res.message


def test_dreamplace_real_backend_sets_backend_and_flow(tmp_path: Path) -> None:
    work = tmp_path / "mixed_size"
    work.mkdir()
    cand_dir = work / "c1"
    cand_dir.mkdir()

    def fake_run(cmd, capture_output, text, timeout, check):  # type: ignore[no-untyped-def]
        assert "hrt-chip-dreamplace-real:local" in cmd
        out = {
            "schema": OUTPUT_SCHEMA_V1,
            "ok": True,
            "message": "ok",
            "density_overflow": 0.0,
            "rudy_or_route_proxy": 0.0,
        }
        (cand_dir / OUTPUT_JSON_NAME).write_text(json.dumps(out), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    cfg = RunConfig(
        mixed_size_backend="dreamplace_real",
        dreamplace_real_docker_image="hrt-chip-dreamplace-real:local",
        dreamplace_mount_testcase=False,
    )
    backend = DreamPlaceDockerBackend(cfg, variant=REAL_DOCKER_VARIANT)
    req = MixedSizeRequest(
        benchmark_id="ibm01",
        fixed_macros=_legal_macros(),
        seed=1,
        candidate_id="c1",
        work_dir_host=work,
        canvas_w=1.0,
        canvas_h=1.0,
    )
    with patch("hrt_chip.adapters.mixed_size.runner.subprocess.run", side_effect=fake_run):
        res = backend.run(req)
    assert res.ok is True
    assert res.extra.get("backend") == "dreamplace_real"
    inp = json.loads((cand_dir / INPUT_JSON_NAME).read_text(encoding="utf-8"))
    assert inp.get("flow") == "mixed_size_real"


def test_dreamplace_backend_retries(tmp_path: Path) -> None:
    work = tmp_path / "mixed_size"
    work.mkdir()
    calls = {"n": 0}

    def flaky(cmd, capture_output, text, timeout, check):  # type: ignore[no-untyped-def]
        sub = work / "c1"
        sub.mkdir(parents=True, exist_ok=True)
        calls["n"] += 1
        if calls["n"] == 1:
            return subprocess.CompletedProcess(cmd, 1, "", "ephemeral")
        out = {
            "schema": OUTPUT_SCHEMA_V1,
            "ok": True,
            "message": "ok",
            "density_overflow": 0.0,
            "rudy_or_route_proxy": 0.0,
        }
        (sub / OUTPUT_JSON_NAME).write_text(json.dumps(out), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    cfg = RunConfig(mixed_size_backend="dreamplace", dreamplace_docker_retries=1)
    backend = DreamPlaceDockerBackend(cfg)
    req = MixedSizeRequest(
        benchmark_id="ibm01",
        fixed_macros=_legal_macros(),
        seed=1,
        candidate_id="c1",
        work_dir_host=work,
        canvas_w=1.0,
        canvas_h=1.0,
    )
    with patch("hrt_chip.adapters.mixed_size.runner.subprocess.run", side_effect=flaky):
        res = backend.run(req)
    assert res.ok is True
    assert calls["n"] == 2


@pytest.mark.skipif(
    __import__("os").environ.get("HRT_DREAMPLACE_INTEGRATION") != "1",
    reason="Set HRT_DREAMPLACE_INTEGRATION=1 and build hrt-chip-dreamplace:local to run",
)
def test_dreamplace_integration_smoke(tmp_path: Path) -> None:
    """Optional: docker run with real image (local operator verification)."""
    from hrt_chip.pipeline import run_pipeline

    cfg = RunConfig(
        benchmark_id="ibm01",
        seed=3,
        num_candidates=1,
        output_dir=tmp_path,
        evaluator_backend="stub",
        mixed_size_backend="dreamplace",
        dreamplace_docker_timeout_seconds=120,
        dreamplace_mount_testcase=False,
    )
    out = run_pipeline(cfg, run_id="integration-dreamplace")
    ranking = out.get("ranking") or []
    assert len(ranking) == 1
    rid = out["manifest"]["run_id"]
    cpath = tmp_path / rid / "candidates" / f"{ranking[0]['candidate_id']}.json"
    data = json.loads(cpath.read_text(encoding="utf-8"))
    ms = data["metadata"]["mixed_size"]
    assert ms["ok"] is True
    assert ms["extra"].get("backend") == "dreamplace"
