"""Orchestrate generate -> legalize -> mixed-size -> evaluate."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

from hrt_chip.adapters.evaluator.base import EvaluatorAdapter, EvaluationResult
from hrt_chip.adapters.evaluator.local_stub import LocalStubEvaluator
from hrt_chip.adapters.mixed_size.base import MixedSizeBackend, MixedSizeRequest
from hrt_chip.adapters.mixed_size.dreamplace_docker import (
    REAL_DOCKER_VARIANT,
    DreamPlaceDockerBackend,
)
from hrt_chip.adapters.mixed_size.estimate import MixedSizeEstimateBackend
from hrt_chip.adapters.mixed_size.local_stub import LocalStubMixedSizeBackend
from hrt_chip.benchmarks import default_testcase_root
from hrt_chip.budget import resolve_generation_budget
from hrt_chip.config import RunConfig, resolved_guidance_sweep
from hrt_chip.deterministic_runtime import apply_pipeline_determinism
from hrt_chip.diffusion import DiffusionSampler
from hrt_chip.models import PlacementCandidate
from hrt_chip.geometry import placement_is_legal
from hrt_chip.guidance import composite_guidance_objective, compute_objectives_for_candidate
from hrt_chip import mixed_size_metrics as msm
from hrt_chip.rank_metrics import kendall_tau, spearman_rho, surrogate_good_proxy_bad_quartiles
from hrt_chip.io.artifacts import (
    PipelineArtifacts,
    apply_candidate_retention,
    build_manifest,
    write_json,
)
from hrt_chip.io.baseline_schema import attach_results_schema_version
from hrt_chip.replay_verify import compare_replay_to_baseline
from hrt_chip.runtime_budget import RuntimeBudgetManager
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


def _resolve_mixed_size_backend(config: RunConfig) -> MixedSizeBackend:
    if config.mixed_size_backend == "stub":
        return LocalStubMixedSizeBackend()
    if config.mixed_size_backend == "dreamplace":
        return DreamPlaceDockerBackend(config)
    if config.mixed_size_backend == "dreamplace_real":
        return DreamPlaceDockerBackend(config, variant=REAL_DOCKER_VARIANT)
    return MixedSizeEstimateBackend()


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

    Default ``selection_policy`` is ``proxy_first`` (proxy primary, mixed-size composite tie-break).
    ``ppa_priority`` ranks legal candidates with successful mixed-size backend by composite placement
    metrics first, then proxy.

    Returns structured dict suitable for JSON serialization and CLI display.
    """
    apply_pipeline_determinism(config)
    t_pipeline0 = time.perf_counter()

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
    ms = mixed_size or _resolve_mixed_size_backend(config)

    manifest = build_manifest(config, run_id=run_id)
    artifacts = PipelineArtifacts(run_dir=config.output_dir / manifest.run_id)
    artifacts.ensure_dirs()
    manifest.write_json(artifacts.manifest_path)

    mixed_size_work_root: Path | None = None
    if config.mixed_size_backend in ("dreamplace", "dreamplace_real"):
        mixed_size_work_root = artifacts.run_dir / "mixed_size"
        mixed_size_work_root.mkdir(parents=True, exist_ok=True)

    run_context: dict[str, Any] = {
        "seed": config.seed,
        "deterministic": config.deterministic,
        "run_id": manifest.run_id,
    }

    sweep_requested = resolved_guidance_sweep(
        guidance_preset=config.guidance_preset,
        guidance_weights_sweep=config.guidance_weights_sweep,
    )
    t_budget0 = time.perf_counter()
    sweep, num_candidates_eff, budget_meta = resolve_generation_budget(config, sweep_requested)
    budget_resolution_seconds = time.perf_counter() - t_budget0

    smp = _resolve_sampler(config, sampler)
    runtime_budget = RuntimeBudgetManager.from_config(config, start_perf=t_pipeline0)

    generation_seconds = 0.0
    legalization_seconds = 0.0
    mixed_size_seconds = 0.0
    evaluation_seconds = 0.0

    raw: list[PlacementCandidate] = []
    evaluations: list[dict[str, Any]] = []
    scoring_table: list[dict[str, Any]] = []
    alignment_scratch: list[dict[str, Any]] = []

    pre_eval_skipped = 0
    sweep_vectors_used = 0

    def _should_continue_sweep(si: int) -> bool:
        if runtime_budget is None:
            return True
        return runtime_budget.can_generate_next_sweep_vector(
            config,
            num_candidates_this_vector=num_candidates_eff,
            already_generated_unprocessed=0,
        )

    for si, _weights in enumerate(sweep):
        if not _should_continue_sweep(si):
            break
        inf_override: int | None = None
        mode_override: str | None = None
        if config.sampler_backend == "pytorch_checkpoint":
            mode_override = config.sampler_mode
            if runtime_budget is not None:
                inf_override = runtime_budget.recommended_diffusion_inference_steps(
                    base_steps=config.diffusion_inference_steps,
                    training_timesteps=config.diffusion_steps,
                )
            else:
                inf_override = config.diffusion_inference_steps
        t_gen0 = time.perf_counter()
        if config.evaluator_backend == "official" and config.sampler_backend == "stub" and bench_obj is not None:
            # The stub sampler generates random macro centers.
            # For official benchmarks with hundreds/thousands of macros, that makes
            # the greedy O(n^2) legalizer prohibitively slow.
            #
            # For a smoke / evidence pipeline, initialize candidates from the benchmark's
            # provided macro centers so legalize_candidate converges quickly.
            from hrt_chip.models import MacroRect
            import random

            a, b, g = sweep[si]
            eff_si = si  # generate() passes a single-weight vector per outer loop
            centers = bench_obj.macro_positions  # [num_macros, 2] (centers, physical units)
            sizes = bench_obj.macro_sizes  # [num_macros, 2] (w, h)
            names = bench_obj.macro_names
            fixed_mask_local = [bool(x) for x in bench_obj.macro_fixed.tolist()]
            # Small deterministic perturbation for movable macros only.
            # Fixed macros must remain at benchmark locations (later restored after legalization).
            jitter_x = 0.01 * float(canvas_w)
            jitter_y = 0.01 * float(canvas_h)
            chunk: list[PlacementCandidate] = []
            for idx in range(num_candidates_eff):
                rng = random.Random(config.seed + 10_000 * eff_si + idx)
                base_id = f"cand_{idx:04d}"
                cand_id = base_id if si == 0 else f"s{eff_si:02d}_{base_id}"
                macros = []
                for mi in range(int(bench_obj.num_macros)):
                    cx = float(centers[mi, 0])
                    cy = float(centers[mi, 1])
                    if not fixed_mask_local[mi]:
                        cx += rng.uniform(-jitter_x, jitter_x)
                        cy += rng.uniform(-jitter_y, jitter_y)
                    w = float(sizes[mi, 0])
                    h = float(sizes[mi, 1])
                    macros.append(MacroRect(name=str(names[mi]), x=cx - w / 2.0, y=cy - h / 2.0, w=w, h=h))

                chunk.append(
                    PlacementCandidate(
                        candidate_id=cand_id,
                        benchmark_id=config.benchmark_id,
                        macros=macros,
                        metadata={
                            "stage": "generated",
                            "guidance": {
                                "sweep_index": eff_si,
                                "alpha_hpwl": float(a),
                                "beta_congestion": float(b),
                                "gamma_legality": float(g),
                                "weights": [float(a), float(b), float(g)],
                            },
                        },
                    )
                )
        else:
            chunk = generate_candidates(
                benchmark_id=config.benchmark_id,
                seed=config.seed,
                num_candidates=num_candidates_eff,
                macro_specs=macro_specs_arg,
                canvas_w=canvas_w,
                canvas_h=canvas_h,
                diffusion_steps=config.diffusion_steps,
                guidance_sweep=(sweep[si],),
                sampler=smp,
                should_continue_sweep=None,
                guidance_sweep_index_offset=si,
                diffusion_inference_steps_override=inf_override,
                sampler_mode_override=mode_override,
            )
        generation_seconds += time.perf_counter() - t_gen0
        sweep_vectors_used += 1
        if runtime_budget is not None:
            runtime_budget.record("generation", time.perf_counter() - t_gen0)

        hard_macro_count_opt = int(getattr(bench_obj, "num_hard_macros", 0)) if bench_obj is not None else None
        if hard_macro_count_opt is not None and hard_macro_count_opt <= 0:
            hard_macro_count_opt = None
        fixed_mask_opt = None
        if bench_obj is not None and hasattr(bench_obj, "macro_fixed"):
            # Benchmarks store fixed flags as a tensor; convert once per benchmark run.
            fixed_mask_opt = [bool(x) for x in bench_obj.macro_fixed.tolist()]

        for cand in chunk:
            t_pipe0 = time.perf_counter()
            t_le = time.perf_counter()
            legalize_candidate(
                cand,
                canvas_w=canvas_w,
                canvas_h=canvas_h,
                hard_macro_count=hard_macro_count_opt,
                fixed_mask=fixed_mask_opt,
            )
            if bench_obj is not None:
                from hrt_chip.official_benchmark import restore_fixed_macro_positions

                restore_fixed_macro_positions(cand, bench_obj)
            legal_flag = cand.metadata.get("legal") is True
            geom_ok = placement_is_legal(
                cand.macros,
                canvas_w=canvas_w,
                canvas_h=canvas_h,
                hard_macro_count=hard_macro_count_opt,
            )
            assert legal_flag == geom_ok, (
                "legality metadata must match geometry check (legal flag vs placement_is_legal)"
            )
            legalization_seconds += time.perf_counter() - t_le
            if runtime_budget is not None:
                runtime_budget.record("legalization", time.perf_counter() - t_le)

            t_ms0 = time.perf_counter()
            if legal_flag:
                req = MixedSizeRequest(
                    benchmark_id=config.benchmark_id,
                    fixed_macros=list(cand.macros),
                    seed=config.seed,
                    benchmark=bench_obj,
                    canvas_w=canvas_w,
                    canvas_h=canvas_h,
                    candidate_id=cand.candidate_id,
                    work_dir_host=mixed_size_work_root,
                    testcase_root_host=Path(testcase_path),
                )
                ms_result = ms.run(req)
                ms.attach_to_candidate(cand, ms_result)
            else:
                cand.metadata["mixed_size"] = {
                    "ok": False,
                    "message": "skipped: macro placement not legal (overlaps or out of bounds)",
                    "extra": {"benchmark_id": config.benchmark_id, "n_macros": len(cand.macros)},
                }
            mixed_size_seconds += time.perf_counter() - t_ms0
            if runtime_budget is not None:
                runtime_budget.record("mixed_size", time.perf_counter() - t_ms0)

            objs = compute_objectives_for_candidate(
                cand,
                benchmark=bench_obj,
                canvas_w=canvas_w,
                canvas_h=canvas_h,
            )
            guidance_meta = cand.metadata.get("guidance")
            if not isinstance(guidance_meta, dict):
                guidance_meta = {}

            a_w = float(guidance_meta.get("alpha_hpwl", 1.0 / 3.0))
            b_w = float(guidance_meta.get("beta_congestion", 1.0 / 3.0))
            g_w = float(guidance_meta.get("gamma_legality", 1.0 / 3.0))
            comp = composite_guidance_objective(
                objs, alpha_hpwl=a_w, beta_congestion=b_w, gamma_legality=g_w
            )

            skip_eval = False
            skip_reason: str | None = None
            if config.pre_eval_rejection_enabled:
                if not legal_flag:
                    skip_eval = True
                    skip_reason = "illegal_macro"
                else:
                    max_ov = config.pre_eval_max_hard_overlap_pairs
                    if max_ov is not None and objs.hard_overlap_pairs > max_ov:
                        skip_eval = True
                        skip_reason = "hard_overlap_pairs"
                    max_c = config.pre_eval_surrogate_composite_max
                    if not skip_eval and max_c is not None and comp is not None and comp > max_c:
                        skip_eval = True
                        skip_reason = "surrogate_composite"

            t_ev0 = time.perf_counter()
            if skip_eval:
                pre_eval_skipped += 1
                er = EvaluationResult(
                    candidate_id=cand.candidate_id,
                    proxy_score=float("inf"),
                    legal=legal_flag,
                    details={
                        "eval_skipped": True,
                        "eval_skip_reason": skip_reason,
                        "surrogate_objectives": objs.to_dict(),
                    },
                )
            else:
                er = evaluate_candidate(
                    cand,
                    ev,
                    benchmark_id=config.benchmark_id,
                    run_context=run_context,
                )
            evaluation_seconds += time.perf_counter() - t_ev0
            if runtime_budget is not None:
                runtime_budget.record("evaluation", time.perf_counter() - t_ev0)
                runtime_budget.observe_candidate_post_generation(time.perf_counter() - t_pipe0)

            row = {
                "candidate_id": er.candidate_id,
                "proxy_score": er.proxy_score,
                "legal": er.legal,
                "details": dict(er.details),
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

            alignment_scratch.append(
                {
                    "candidate_id": er.candidate_id,
                    "proxy_score": er.proxy_score,
                    "legal": er.legal,
                    "composite": comp,
                }
            )
            raw.append(cand)

    ms_rows: list[dict[str, Any]] = []
    for cand, ev in zip(raw, evaluations, strict=True):
        ms_meta = cand.metadata.get("mixed_size") or {}
        extra = ms_meta.get("extra") if isinstance(ms_meta.get("extra"), dict) else {}
        ms_rows.append(
            {
                "candidate_id": cand.candidate_id,
                "legal": ev["legal"],
                "ms_ok": ms_meta.get("ok") is True,
                "ms_extra": extra,
            }
        )
    profiles = msm.build_mixed_size_profiles_for_candidates(ms_rows)

    for ev, cand in zip(evaluations, raw, strict=True):
        prof = profiles.get(ev["candidate_id"], {})
        ev["mixed_size_profile"] = prof
        ev["details"]["mixed_size_profile"] = prof

    for st, cand in zip(scoring_table, raw, strict=True):
        st["mixed_size_profile"] = profiles.get(st["candidate_id"], {})

    for cand in raw:
        prof = profiles.get(cand.candidate_id, {})
        ms_slot = cand.metadata.setdefault("mixed_size", {})
        if isinstance(ms_slot, dict):
            ms_slot["profile"] = prof
        write_json(artifacts.candidates_dir / f"{cand.candidate_id}.json", cand.to_dict())

    policy = config.selection_policy
    key_fn = msm.ranking_key_proxy_first if policy == "proxy_first" else msm.ranking_key_ppa_priority
    ranking = sorted(evaluations, key=key_fn)
    best_candidate_id: str | None = ranking[0]["candidate_id"] if ranking else None
    best_proxy_score: float | None = ranking[0]["proxy_score"] if ranking else None

    if ranking and policy == "proxy_first":
        proxy_sorted = sorted(evaluations, key=msm.ranking_key_proxy_first)
        assert [r["candidate_id"] for r in ranking] == [r["candidate_id"] for r in proxy_sorted], (
            "proxy_first ranking must match stable proxy-first sort"
        )

    filtered_ids: list[str] = []
    filtered_composites: list[float] = []
    filtered_proxies: list[float] = []
    for row in alignment_scratch:
        if not row["legal"]:
            continue
        ps = row["proxy_score"]
        if not isinstance(ps, (int, float)) or not math.isfinite(float(ps)):
            continue
        if row["composite"] is None:
            continue
        filtered_ids.append(str(row["candidate_id"]))
        filtered_composites.append(float(row["composite"]))
        filtered_proxies.append(float(ps))

    spear = spearman_rho(filtered_composites, filtered_proxies)
    kend = kendall_tau(filtered_composites, filtered_proxies)
    mismatch = surrogate_good_proxy_bad_quartiles(filtered_ids, filtered_composites, filtered_proxies)

    surrogate_proxy_alignment: dict[str, Any] = {
        "n_candidates_total": len(alignment_scratch),
        "n_used_for_correlation": len(filtered_ids),
        "spearman_rho": spear,
        "kendall_tau": kend,
        "surrogate_good_proxy_bad": mismatch,
    }

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
        "guidance_sweep_requested": [list(t) for t in sweep_requested],
        "guidance_sweep_resolved": [list(t) for t in sweep],
        "sampler_provenance": sampler_provenance,
        "sampler_backend": config.sampler_backend,
        "checkpoint_path": str(config.checkpoint_path) if config.checkpoint_path else None,
        "training_dataset_version": config.training_dataset_version
        or (sampler_provenance or {}).get("training_dataset_version"),
        "ranking": ranking,
        "scoring_table": scoring_table,
        "surrogate_proxy_alignment": surrogate_proxy_alignment,
        "best_candidate_id": best_candidate_id,
        "best_proxy_score": best_proxy_score,
        "selection_policy": config.selection_policy,
        "selection_rationale": {
            "policy": config.selection_policy,
            "ranking_key": "proxy_first" if policy == "proxy_first" else "ppa_priority",
            "mixed_size_weights": {
                "density": msm.WEIGHT_DENSITY,
                "congestion": msm.WEIGHT_CONGESTION,
                "runtime": msm.WEIGHT_RUNTIME,
            },
        },
        "evaluations": evaluations,
        "mixed_size_backend": config.mixed_size_backend,
        "budget_resolution": {
            **budget_meta,
            "budget_resolution_seconds": budget_resolution_seconds,
            "requested_num_candidates": config.num_candidates,
            "resolved_num_candidates": num_candidates_eff,
        },
        "runtime_budget": runtime_budget.to_dict() if runtime_budget is not None else None,
        "sweep_vectors_used": sweep_vectors_used,
        "sweep_vectors_requested": len(sweep),
        "generation_stopped_early": sweep_vectors_used < len(sweep),
        "pre_eval_rejection": {
            "enabled": config.pre_eval_rejection_enabled,
            "skipped_eval_count": pre_eval_skipped,
        },
        "sampler_mode": config.sampler_mode,
        "diffusion_reverse_schedule": config.diffusion_reverse_schedule,
        "experiment_tag": config.experiment_tag,
        "experiment_notes": config.experiment_notes,
        "timing": {
            "generation_seconds": generation_seconds,
            "legalization_seconds": legalization_seconds,
            "mixed_size_seconds": mixed_size_seconds,
            "evaluation_seconds": evaluation_seconds,
            "legalize_mixed_size_eval_seconds": legalization_seconds
            + mixed_size_seconds
            + evaluation_seconds,
            "total_pipeline_seconds": time.perf_counter() - t_pipeline0,
        },
    }
    attach_results_schema_version(results)
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
