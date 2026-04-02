"""Orchestrate generate -> legalize -> mixed-size -> evaluate."""

from __future__ import annotations

from typing import Any

from hrt_chip.adapters.evaluator.base import EvaluationResult, EvaluatorAdapter
from hrt_chip.adapters.evaluator.local_stub import LocalStubEvaluator
from hrt_chip.adapters.mixed_size.base import MixedSizeBackend, MixedSizeRequest
from hrt_chip.adapters.mixed_size.local_stub import LocalStubMixedSizeBackend
from hrt_chip.config import RunConfig
from hrt_chip.geometry import placement_is_legal
from hrt_chip.io.artifacts import PipelineArtifacts, build_manifest, write_json
from hrt_chip.stages.evaluate import evaluate_candidate
from hrt_chip.stages.generate import generate_candidates
from hrt_chip.stages.legalize import legalize_candidate


def run_pipeline(
    config: RunConfig,
    *,
    evaluator: EvaluatorAdapter | None = None,
    mixed_size: MixedSizeBackend | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """
    Full pipeline: generate -> legalize -> mixed-size -> evaluate.

    Illegal macro placements skip mixed-size handoff and receive infinite proxy from the evaluator.

    Returns structured dict suitable for JSON serialization and CLI display.
    """
    ev = evaluator or LocalStubEvaluator()
    ms = mixed_size or LocalStubMixedSizeBackend()

    manifest = build_manifest(config, run_id=run_id)
    artifacts = PipelineArtifacts(run_dir=config.output_dir / manifest.run_id)
    artifacts.ensure_dirs()
    manifest.write_json(artifacts.manifest_path)

    run_context: dict[str, Any] = {
        "seed": config.seed,
        "deterministic": config.deterministic,
        "run_id": manifest.run_id,
    }

    raw = generate_candidates(
        benchmark_id=config.benchmark_id,
        seed=config.seed,
        num_candidates=config.num_candidates,
    )

    evaluations: list[dict[str, Any]] = []
    best: EvaluationResult | None = None
    best_candidate_id: str | None = None

    for cand in raw:
        legalize_candidate(cand)
        legal_flag = cand.metadata.get("legal") is True
        geom_ok = placement_is_legal(cand.macros)
        assert legal_flag == geom_ok, (
            "legality metadata must match geometry check (legal flag vs placement_is_legal)"
        )
        if legal_flag:
            req = MixedSizeRequest(
                benchmark_id=config.benchmark_id,
                fixed_macros=list(cand.macros),
                seed=config.seed,
            )
            ms_result = ms.run(req)
            ms.attach_to_candidate(cand, ms_result)
        else:
            cand.metadata["mixed_size"] = {
                "ok": False,
                "message": "skipped: macro placement not legal (overlaps or out of bounds)",
                "extra": {"benchmark_id": config.benchmark_id, "n_macros": len(cand.macros)},
            }

        er = evaluate_candidate(
            cand,
            ev,
            benchmark_id=config.benchmark_id,
            run_context=run_context,
        )
        row = {
            "candidate_id": er.candidate_id,
            "proxy_score": er.proxy_score,
            "legal": er.legal,
            "details": er.details,
        }
        evaluations.append(row)

        cand_path = artifacts.candidates_dir / f"{cand.candidate_id}.json"
        write_json(cand_path, cand.to_dict())

        if best is None or er.proxy_score < best.proxy_score:
            best = er
            best_candidate_id = er.candidate_id

    results: dict[str, Any] = {
        "manifest": manifest.to_dict(),
        "benchmark_id": config.benchmark_id,
        "ranking": sorted(evaluations, key=lambda r: r["proxy_score"]),
        "best_candidate_id": best_candidate_id,
        "best_proxy_score": best.proxy_score if best else None,
        "evaluations": evaluations,
    }
    write_json(artifacts.results_path, results)
    return results


def replay_from_manifest(manifest_path: str) -> dict[str, Any]:
    """
    Re-run pipeline from a saved manifest.json (same config snapshot).

    Intended for reproducibility checks; requires manifest written by a prior run.
    """
    from pathlib import Path

    p = Path(manifest_path)
    data = __import__("json").loads(p.read_text(encoding="utf-8"))
    cfg = RunConfig.from_dict(data["config"])
    return run_pipeline(cfg, run_id=data["run_id"])
