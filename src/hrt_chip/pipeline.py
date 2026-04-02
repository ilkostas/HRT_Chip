"""Orchestrate generate -> legalize -> mixed-size -> evaluate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hrt_chip.adapters.evaluator.base import EvaluationResult, EvaluatorAdapter
from hrt_chip.adapters.evaluator.local_stub import LocalStubEvaluator
from hrt_chip.adapters.mixed_size.base import MixedSizeBackend, MixedSizeRequest
from hrt_chip.adapters.mixed_size.local_stub import LocalStubMixedSizeBackend
from hrt_chip.benchmarks import default_testcase_root
from hrt_chip.config import RunConfig, resolved_guidance_sweep
from hrt_chip.deterministic_runtime import apply_pipeline_determinism
from hrt_chip.diffusion import DiffusionSampler
from hrt_chip.geometry import placement_is_legal
from hrt_chip.guidance import compute_objectives_for_candidate
from hrt_chip.io.artifacts import (
    PipelineArtifacts,
    apply_candidate_retention,
    build_manifest,
    write_json,
)
from hrt_chip.replay_verify import compare_replay_to_baseline
from hrt_chip.stages.evaluate import evaluate_candidate
from hrt_chip.stages.generate import generate_candidates
from hrt_chip.stages.legalize import legalize_candidate


def _resolve_evaluator(
    config: RunConfig,
    *,
    prime: tuple[str, Any, Any] | None = None,
) -> EvaluatorAdapter:
    if config.evaluator_backend == "official":
        from hrt_chip.adapters.evaluator.official import OfficialMacroPlacementEvaluator
        from hrt_chip.benchmarks import default_testcase_root

        root = config.testcase_root or Path(default_testcase_root())
        ev = OfficialMacroPlacementEvaluator(testcase_root=root)
        if prime is not None:
            bid, bench, plc = prime
            ev.prime(bid, bench, plc)
        return ev
    return LocalStubEvaluator()


def _resolve_sampler(config: RunConfig, sampler: DiffusionSampler | None) -> DiffusionSampler | None:
    if sampler is not None:
        return sampler
    if config.sampler_backend == "pytorch_checkpoint":
        from hrt_chip.adapters.diffusion.pytorch_sampler import build_pytorch_sampler

        return build_pytorch_sampler(config)
    return None


def run_pipeline(
    config: RunConfig,
    *,
    evaluator: EvaluatorAdapter | None = None,
    mixed_size: MixedSizeBackend | None = None,
    sampler: DiffusionSampler | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """
    Full pipeline: generate -> legalize -> mixed-size -> evaluate.

    Illegal macro placements skip mixed-size handoff and receive infinite proxy from the evaluator.

    Final best candidate is always the argmin of official proxy score (Tier-1 selection).

    Returns structured dict suitable for JSON serialization and CLI display.
    """
    apply_pipeline_determinism(config)

    testcase_path = config.testcase_root or Path(default_testcase_root())
    bench_obj: Any | None = None
    macro_specs_arg: Any = None
    canvas_w, canvas_h = 1.0, 1.0
    prime_bundle: tuple[str, Any, Any] | None = None

    if config.evaluator_backend == "official":
        from hrt_chip.official_benchmark import load_full_benchmark

        bench_obj, plc_obj, macro_specs_arg, canvas_w, canvas_h = load_full_benchmark(
            config.benchmark_id, Path(testcase_path)
        )
        prime_bundle = (config.benchmark_id, bench_obj, plc_obj)

    ev = evaluator or _resolve_evaluator(config, prime=prime_bundle)
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

    sweep = resolved_guidance_sweep(
        guidance_preset=config.guidance_preset,
        guidance_weights_sweep=config.guidance_weights_sweep,
    )
    smp = _resolve_sampler(config, sampler)
    raw = generate_candidates(
        benchmark_id=config.benchmark_id,
        seed=config.seed,
        num_candidates=config.num_candidates,
        macro_specs=macro_specs_arg,
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        diffusion_steps=config.diffusion_steps,
        guidance_sweep=sweep,
        sampler=smp,
    )

    evaluations: list[dict[str, Any]] = []
    scoring_table: list[dict[str, Any]] = []
    best: EvaluationResult | None = None
    best_candidate_id: str | None = None

    for cand in raw:
        legalize_candidate(cand, canvas_w=canvas_w, canvas_h=canvas_h)
        if bench_obj is not None:
            from hrt_chip.official_benchmark import restore_fixed_macro_positions

            restore_fixed_macro_positions(cand, bench_obj)
        legal_flag = cand.metadata.get("legal") is True
        geom_ok = placement_is_legal(cand.macros, canvas_w=canvas_w, canvas_h=canvas_h)
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

        objs = compute_objectives_for_candidate(cand)
        guidance_meta = cand.metadata.get("guidance")
        if not isinstance(guidance_meta, dict):
            guidance_meta = {}

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

        scoring_table.append(
            {
                "candidate_id": er.candidate_id,
                "proxy_score": er.proxy_score,
                "legal": er.legal,
                "guidance": dict(guidance_meta),
                "surrogate_objectives": objs.to_dict(),
            }
        )

        cand_path = artifacts.candidates_dir / f"{cand.candidate_id}.json"
        write_json(cand_path, cand.to_dict())

        if best is None or er.proxy_score < best.proxy_score:
            best = er
            best_candidate_id = er.candidate_id

    ranking = sorted(evaluations, key=lambda r: r["proxy_score"])
    if ranking:
        assert best_candidate_id == ranking[0]["candidate_id"], (
            "best_candidate_id must be argmin(proxy_score)"
        )
        assert best is not None
        assert ranking[0]["proxy_score"] == best.proxy_score

    sampler_provenance: dict[str, Any] | None = None
    if raw:
        sp = raw[0].metadata.get("sampler")
        if isinstance(sp, dict):
            sampler_provenance = dict(sp)

    results: dict[str, Any] = {
        "manifest": manifest.to_dict(),
        "benchmark_id": config.benchmark_id,
        "evaluator_backend": config.evaluator_backend,
        "testcase_root": str(testcase_path),
        "canvas_width": canvas_w,
        "canvas_height": canvas_h,
        "guidance_sweep_resolved": [list(t) for t in sweep],
        "sampler_provenance": sampler_provenance,
        "sampler_backend": config.sampler_backend,
        "checkpoint_path": str(config.checkpoint_path) if config.checkpoint_path else None,
        "training_dataset_version": config.training_dataset_version
        or (sampler_provenance or {}).get("training_dataset_version"),
        "ranking": ranking,
        "scoring_table": scoring_table,
        "best_candidate_id": best_candidate_id,
        "best_proxy_score": best.proxy_score if best else None,
        "evaluations": evaluations,
    }
    write_json(artifacts.results_path, results)

    apply_candidate_retention(
        artifacts,
        results,
        mode=config.artifact_retention,
        top_k=config.artifact_retention_top_k,
    )

    return results


def replay_from_manifest(
    manifest_path: str,
    *,
    verify: bool = False,
) -> dict[str, Any]:
    """
    Re-run pipeline from a saved manifest.json (same config snapshot).

    Intended for reproducibility checks; requires manifest written by a prior run.

    If ``verify`` is True, compares the new run to ``results.json`` beside the manifest
    before it is overwritten (load baseline first), then writes ``replay_verification.json``.
    """
    import json
    from pathlib import Path

    p = Path(manifest_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    cfg = RunConfig.from_dict(data["config"])
    run_dir = p.parent
    baseline_results: dict[str, Any] | None = None
    if verify:
        baseline_path = run_dir / "results.json"
        if baseline_path.is_file():
            baseline_results = json.loads(baseline_path.read_text(encoding="utf-8"))

    results = run_pipeline(cfg, run_id=data["run_id"])

    if verify:
        report: dict[str, Any]
        if baseline_results is None:
            report = {
                "ok": False,
                "mismatches": [f"Missing baseline file: {run_dir / 'results.json'}"],
                "baseline_fingerprint": None,
                "replay_fingerprint": None,
            }
        else:
            report = compare_replay_to_baseline(baseline_results, results)
        from hrt_chip.io.artifacts import utc_now_iso

        report["verified_at_utc"] = utc_now_iso()
        report["manifest_path"] = str(p.resolve())
        artifacts = PipelineArtifacts(run_dir=run_dir)
        write_json(artifacts.replay_verification_path, report)
        results = {**results, "replay_verification": report}

    return results
