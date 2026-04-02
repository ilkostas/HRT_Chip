"""Docker-based mixed-size backend (DREAMPlace/hMETIS flow image).

The default image runs an analytical placement proxy inside the container. Operators can
replace the image with a full DREAMPlace build; the same ``/work`` contract applies.

``dreamplace_real`` uses a separate image/timeout env vars for a production toolchain image.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hrt_chip.adapters.mixed_size.base import MixedSizeBackend, MixedSizeRequest, MixedSizeResult
from hrt_chip.adapters.mixed_size.contracts import DOCKER_LOG_NAME, OUTPUT_SCHEMA_V1
from hrt_chip.adapters.mixed_size.runner import (
    build_input_payload,
    DockerRunResult,
    parse_extra_docker_args,
    read_output_json,
    run_dreamplace_container,
    safe_candidate_subdir,
    write_input_json,
)
from hrt_chip.config import RunConfig
from hrt_chip.geometry import placement_is_legal
from hrt_chip.io.artifacts import write_json


@dataclass(frozen=True)
class DreamPlaceDockerVariant:
    """Per-backend Docker settings (analytical vs real mixed-size toolchain)."""

    backend_key: str
    image_env: str
    image_attr: str
    timeout_env: str
    timeout_attr: str
    flow: str | None = None


DEFAULT_DOCKER_VARIANT = DreamPlaceDockerVariant(
    backend_key="dreamplace",
    image_env="HRT_DREAMPLACE_IMAGE",
    image_attr="dreamplace_docker_image",
    timeout_env="HRT_DREAMPLACE_TIMEOUT",
    timeout_attr="dreamplace_docker_timeout_seconds",
    flow=None,
)

REAL_DOCKER_VARIANT = DreamPlaceDockerVariant(
    backend_key="dreamplace_real",
    image_env="HRT_DREAMPLACE_REAL_IMAGE",
    image_attr="dreamplace_real_docker_image",
    timeout_env="HRT_DREAMPLACE_REAL_TIMEOUT",
    timeout_attr="dreamplace_real_docker_timeout_seconds",
    flow="mixed_size_real",
)


class DreamPlaceDockerBackend(MixedSizeBackend):
    """Invoke Docker image with ``/work`` input/output contract."""

    def __init__(self, config: RunConfig, *, variant: DreamPlaceDockerVariant | None = None) -> None:
        self._cfg = config
        self._variant = variant or DEFAULT_DOCKER_VARIANT

    def run(self, request: MixedSizeRequest) -> MixedSizeResult:
        t0 = time.perf_counter()
        v = self._variant
        macros = list(request.fixed_macros)
        if not placement_is_legal(macros, canvas_w=request.canvas_w, canvas_h=request.canvas_h):
            return MixedSizeResult(
                ok=False,
                message=f"rejected: illegal macro geometry for {v.backend_key} docker backend",
                extra={"benchmark_id": request.benchmark_id, "backend": v.backend_key},
            )
        if request.work_dir_host is None:
            return MixedSizeResult(
                ok=False,
                message=f"rejected: work_dir_host is required for {v.backend_key} backend",
                extra={"benchmark_id": request.benchmark_id, "backend": v.backend_key},
            )

        work_dir = Path(request.work_dir_host)
        cid = request.candidate_id or "candidate"
        sub = safe_candidate_subdir(cid)
        cand_work = work_dir / sub
        cand_work.mkdir(parents=True, exist_ok=True)

        payload = build_input_payload(
            benchmark_id=request.benchmark_id,
            candidate_id=request.candidate_id,
            seed=request.seed,
            canvas_w=request.canvas_w,
            canvas_h=request.canvas_h,
            macros=macros,
            flow=v.flow,
        )
        write_input_json(cand_work, payload)

        image = os.environ.get(v.image_env, getattr(self._cfg, v.image_attr))
        docker_exe = os.environ.get("HRT_DOCKER_EXECUTABLE", self._cfg.dreamplace_docker_executable)
        timeout = int(os.environ.get(v.timeout_env, str(getattr(self._cfg, v.timeout_attr))))
        retries = max(0, int(self._cfg.dreamplace_docker_retries))
        extra = parse_extra_docker_args(self._cfg.dreamplace_docker_extra_args)

        last_dr = None
        attempt = 0
        while attempt <= retries:
            try:
                last_dr = run_dreamplace_container(
                    docker_exe=docker_exe,
                    image=image,
                    work_dir=cand_work,
                    testcase_root=request.testcase_root_host,
                    mount_testcase=bool(self._cfg.dreamplace_mount_testcase),
                    timeout_seconds=timeout,
                    extra_docker_args=extra if extra else None,
                )
            except FileNotFoundError:
                return MixedSizeResult(
                    ok=False,
                    message="docker executable not found (install Docker Desktop / add to PATH)",
                    extra={
                        "benchmark_id": request.benchmark_id,
                        "backend": v.backend_key,
                        "docker_exe": docker_exe,
                    },
                )
            except subprocess.TimeoutExpired as e:
                # Allow transient timeouts to be retried when the operator configured retries.
                if attempt >= retries:
                    return MixedSizeResult(
                        ok=False,
                        message=f"docker run exceeded timeout ({timeout}s)",
                        extra={
                            "benchmark_id": request.benchmark_id,
                            "backend": v.backend_key,
                            "image": image,
                            "work_dir": str(cand_work.resolve()),
                            "timeout_seconds": timeout,
                        },
                    )

                stdout = getattr(e, "stdout", None) or getattr(e, "output", None) or ""
                stderr = getattr(e, "stderr", None) or ""
                last_dr = DockerRunResult(
                    returncode=124,
                    stdout=str(stdout),
                    stderr=str(stderr),
                    elapsed_seconds=float(timeout),
                )

            log_path = cand_work / DOCKER_LOG_NAME
            log_body = (
                f"returncode={last_dr.returncode}\n"
                f"elapsed_seconds={last_dr.elapsed_seconds:.4f}\n"
                f"--- stdout ---\n{last_dr.stdout}\n--- stderr ---\n{last_dr.stderr}\n"
            )
            log_path.write_text(log_body, encoding="utf-8")

            if last_dr.returncode == 0:
                break
            attempt += 1
            if attempt > retries:
                return MixedSizeResult(
                    ok=False,
                    message=f"docker run failed (code {last_dr.returncode})",
                    extra={
                        "benchmark_id": request.benchmark_id,
                        "backend": v.backend_key,
                        "image": image,
                        "returncode": last_dr.returncode,
                        "docker_log": str(log_path.resolve()),
                        "stderr_excerpt": (last_dr.stderr or "")[:2000],
                    },
                )

        assert last_dr is not None
        out = read_output_json(cand_work)
        if out is None:
            return MixedSizeResult(
                ok=False,
                message="missing or invalid output.json from container",
                extra={
                    "benchmark_id": request.benchmark_id,
                    "backend": v.backend_key,
                    "docker_log": str((cand_work / DOCKER_LOG_NAME).resolve()),
                    "returncode": last_dr.returncode,
                },
            )

        if out.get("schema") != OUTPUT_SCHEMA_V1:
            return MixedSizeResult(
                ok=False,
                message=f"unexpected output schema: {out.get('schema')!r}",
                extra={"benchmark_id": request.benchmark_id, "backend": v.backend_key},
            )

        # Minimal output contract validation so unexpected container outputs do
        # not silently degrade downstream ranking.
        missing: list[str] = []
        if "ok" not in out:
            missing.append("ok")
        if "message" not in out:
            missing.append("message")
        if out.get("density_overflow") is None and out.get("density_overflow_proxy") is None:
            missing.append("density_overflow")
        if out.get("rudy_or_route_proxy") is None and out.get("rudy_density_variance") is None:
            missing.append("rudy_or_route_proxy")
        if missing:
            return MixedSizeResult(
                ok=False,
                message=f"invalid output contract (missing keys: {missing})",
                extra={"benchmark_id": request.benchmark_id, "backend": v.backend_key},
            )

        ok = bool(out.get("ok", False))
        density = out.get("density_overflow", out.get("density_overflow_proxy"))
        rudy = out.get("rudy_or_route_proxy")
        extra_out: dict[str, Any] = {
            "benchmark_id": request.benchmark_id,
            "backend": v.backend_key,
            "image": image,
            "candidate_id": request.candidate_id,
            "output_dir": str(cand_work.resolve()),
            "status": "ok" if ok else "failed",
            "log_path": str((cand_work / DOCKER_LOG_NAME).resolve()),
            "docker_log": str((cand_work / DOCKER_LOG_NAME).resolve()),
            "container_returncode": last_dr.returncode,
            "docker_host_elapsed_seconds": last_dr.elapsed_seconds,
            "density_overflow": density,
            "rudy_or_route_proxy": rudy,
            "hmetis_invoked": out.get("hmetis_invoked"),
            "dreamplace_invoked": out.get("dreamplace_invoked"),
            "placement_mode": out.get("placement_mode"),
        }
        for k in ("backend_runtime_seconds", "notes"):
            if k in out:
                extra_out[k] = out[k]

        msg = str(
            out.get("message") or (f"{v.backend_key} docker: ok" if ok else f"{v.backend_key} docker: failed")
        )
        total_dt = time.perf_counter() - t0
        extra_out["host_wall_seconds"] = total_dt
        write_json(cand_work / "host_summary.json", {"parsed_output": out, "extra": extra_out})
        return MixedSizeResult(ok=ok, message=msg, extra=extra_out)
