"""Docker runner for mixed-size placement flow (bind-mount workdir + optional testcase)."""

from __future__ import annotations

import json
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hrt_chip.adapters.mixed_size.contracts import (
    INPUT_JSON_NAME,
    INPUT_SCHEMA_V1,
    OUTPUT_JSON_NAME,
)
from hrt_chip.models import MacroRect


def safe_candidate_subdir(candidate_id: str) -> str:
    out = "".join(c if c.isalnum() or c in "._-" else "_" for c in candidate_id)
    return out[:200] if out else "candidate"


def build_input_payload(
    *,
    benchmark_id: str,
    candidate_id: str | None,
    seed: int,
    canvas_w: float,
    canvas_h: float,
    macros: list[MacroRect],
    flow: str | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "schema": INPUT_SCHEMA_V1,
        "benchmark_id": benchmark_id,
        "candidate_id": candidate_id,
        "seed": seed,
        "canvas": {"width": float(canvas_w), "height": float(canvas_h)},
        "macros": [m.to_dict() for m in macros],
    }
    if flow:
        out["flow"] = flow
    return out


def write_input_json(work_dir: Path, payload: dict[str, Any]) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    p = work_dir / INPUT_JSON_NAME
    p.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return p


@dataclass
class DockerRunResult:
    returncode: int
    stdout: str
    stderr: str
    elapsed_seconds: float


def run_dreamplace_container(
    *,
    docker_exe: str,
    image: str,
    work_dir: Path,
    testcase_root: Path | None,
    mount_testcase: bool,
    timeout_seconds: int,
    extra_docker_args: list[str] | None = None,
) -> DockerRunResult:
    """
    Run ``docker run --rm`` with ``/work`` bound to ``work_dir``.

    Mounts ``testcase_root`` at ``/testcase:ro`` when ``mount_testcase`` and path is set.
    """
    work_abs = str(work_dir.resolve())
    cmd: list[str] = [
        docker_exe,
        "run",
        "--rm",
        "-v",
        f"{work_abs}:/work",
    ]
    if mount_testcase and testcase_root is not None and testcase_root.is_dir():
        tr = str(testcase_root.resolve())
        cmd.extend(["-v", f"{tr}:/testcase:ro"])
    if extra_docker_args:
        cmd.extend(extra_docker_args)
    cmd.append(image)

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    dt = time.perf_counter() - t0
    return DockerRunResult(
        returncode=int(proc.returncode),
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
        elapsed_seconds=dt,
    )


def parse_extra_docker_args(raw: str | None) -> list[str]:
    if not raw or not raw.strip():
        return []
    return shlex.split(raw, posix=True)


def read_output_json(work_dir: Path) -> dict[str, Any] | None:
    p = work_dir / OUTPUT_JSON_NAME
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
