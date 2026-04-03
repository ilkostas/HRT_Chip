"""Search-hybrid pipeline: initialize → search (SA) → legalize touch-up → mixed-size → evaluate."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

from hrt_chip.adapters.evaluator.base import EvaluationResult, EvaluatorAdapter
from hrt_chip.adapters.mixed_size.base import MixedSizeBackend, MixedSizeRequest
from hrt_chip.adapters.mixed_size.dreamplace_docker import REAL_DOCKER_VARIANT, DreamPlaceDockerBackend
from hrt_chip.adapters.mixed_size.estimate import MixedSizeEstimateBackend
from hrt_chip.adapters.mixed_size.local_stub import LocalStubMixedSizeBackend
from hrt_chip.benchmarks import default_testcase_root
from hrt_chip.config import RunConfig
from hrt_chip.deterministic_runtime import apply_pipeline_determinism
from hrt_chip.diffusion import DiffusionSampler
from hrt_chip.geometry import placement_is_legal
from hrt_chip.guidance import composite_guidance_objective, compute_objectives_for_candidate
from hrt_chip import mixed_size_metrics as msm
from hrt_chip.models import PlacementCandidate
from hrt_chip.rank_metrics import kendall_tau, spearman_rho, surrogate_good_proxy_bad_quartiles
from hrt_chip.io.artifacts import PipelineArtifacts, apply_candidate_retention, build_manifest, write_json
from hrt_chip.io.baseline_schema import attach_results_schema_version
from hrt_chip.initializers.build import build_seed_candidates
from hrt_chip.official_benchmark import restore_fixed_macro_positions
from hrt_chip.search.engine import run_search_on_seeds
from hrt_chip.search.objective import placement_energy
from hrt_chip.stages.evaluate import evaluate_candidate
from hrt_chip.stages.legalize import legalize_candidate


def _resolve_evaluator(
    config: RunConfig,
    *,
    prime: tuple[str, Any, Any] | None = None,
) -> EvaluatorAdapter:
    if config.evaluator_backend == "official":
        from hrt_chip.adapters.evaluator.official import OfficialMacroPlacementEvaluator
        from hrt_chip.benchmarks import default_testcase_root as dtr

        root = config.testcase_root or Path(dtr())
        ev = OfficialMacroPlacementEvaluator(testcase_root=root)
        if prime is not None:
            bid, bench, plc = prime
            ev.prime(bid, bench, plc)
        return ev
    from hrt_chip.adapters.evaluator.local_stub import LocalStubEvaluator

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


def run_search_pipeline(
    config: RunConfig,
    *,
    evaluator: EvaluatorAdapter | None = None,
    mixed_size: MixedSizeBackend | None = None,
    sampler: DiffusionSampler | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
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
    smp = _resolve_sampler(config, sampler)

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
        "solver_backend": "search_hybrid",
    }

    t_gen0 = time.perf_counter()
    seeds = build_seed_candidates(
        config,
        bench_obj=bench_obj,
        macro_specs_arg=macro_specs_arg,
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        sampler=smp,
    )
    generation_seconds = time.perf_counter() - t_gen0

    hard_macro_count_opt = int(getattr(bench_obj, "num_hard_macros", 0)) if bench_obj is not None else None
    if hard_macro_count_opt is not None and hard_macro_count_opt <= 0:
        hard_macro_count_opt = None
    fixed_mask_opt = None
    if bench_obj is not None and hasattr(bench_obj, "macro_fixed"):
        fixed_mask_opt = [bool(x) for x in bench_obj.macro_fixed.tolist()]

    legalized_seeds: list[PlacementCandidate] = []
    t_le_total = 0.0
    for cand in seeds:
        t_le = time.perf_counter()
        legalize_candidate(
            cand,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            hard_macro_count=hard_macro_count_opt,
            fixed_mask=fixed_mask_opt,
        )
        if bench_obj is not None:
            restore_fixed_macro_positions(cand, bench_obj)
        t_le_total += time.perf_counter() - t_le
        if cand.metadata.get("legal") is True:
            legalized_seeds.append(cand)

    if not legalized_seeds:
        legalized_seeds = seeds  # attempt search anyway; evaluator will penalize illegal

    def official_energy_fn(c: PlacementCandidate) -> float | None:
        er = evaluate_candidate(
            c,
            ev,
            benchmark_id=config.benchmark_id,
            run_context=run_context,
        )
        ps = er.proxy_score
        if isinstance(ps, (int, float)) and math.isfinite(float(ps)):
            return float(ps)
        return None

    official_fn = official_energy_fn if config.evaluator_backend == "official" else None

    t_search0 = time.perf_counter()
    searched = run_search_on_seeds(
        legalized_seeds,
        config,
        benchmark=bench_obj,
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        hard_macro_count=hard_macro_count_opt,
        fixed_mask=fixed_mask_opt,
        official_energy_fn=official_fn,
    )
    search_seconds = time.perf_counter() - t_search0

    raw: list[PlacementCandidate] = []
    evaluations: list[dict[str, Any]] = []
    scoring_table: list[dict[str, Any]] = []
    alignment_scratch: list[dict[str, Any]] = []
    search_energies: list[float] = []
    pre_eval_skipped = 0

    legalization_seconds = t_le_total
    mixed_size_seconds = 0.0
    evaluation_seconds = 0.0

    for cand, search_meta in searched:
        cand.metadata["stage"] = "search_refined"
        t_le = time.perf_counter()
        legalize_candidate(
            cand,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            hard_macro_count=hard_macro_count_opt,
            fixed_mask=fixed_mask_opt,
        )
        if bench_obj is not None:
            restore_fixed_macro_positions(cand, bench_obj)
        legalization_seconds += time.perf_counter() - t_le
        legal_flag = cand.metadata.get("legal") is True
        geom_ok = placement_is_legal(
            cand.macros,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            hard_macro_count=hard_macro_count_opt,
        )
        assert legal_flag == geom_ok

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

        objs = compute_objectives_for_candidate(
            cand,
            benchmark=bench_obj,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
        )
        se = placement_energy(
            cand,
            benchmark=bench_obj,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            mode="full",
        )
        search_energies.append(se)

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
                "search_telemetry": search_meta,
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

    # Search energy vs official proxy alignment (same as surrogate alignment)
    filtered_ids: list[str] = []
    filtered_energy: list[float] = []
    filtered_proxies: list[float] = []
    for i, row in enumerate(alignment_scratch):
        if not row["legal"]:
            continue
        ps = row["proxy_score"]
        if not isinstance(ps, (int, float)) or not math.isfinite(float(ps)):
            continue
        if i < len(search_energies):
            filtered_ids.append(str(row["candidate_id"]))
            filtered_energy.append(float(search_energies[i]))
            filtered_proxies.append(float(ps))

    spear_se = spearman_rho(filtered_energy, filtered_proxies)
    kend_se = kendall_tau(filtered_energy, filtered_proxies)
    mismatch_se = surrogate_good_proxy_bad_quartiles(filtered_ids, filtered_energy, filtered_proxies)

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
        assert [r["candidate_id"] for r in ranking] == [r["candidate_id"] for r in proxy_sorted]

    filtered_ids2: list[str] = []
    filtered_composites: list[float] = []
    filtered_proxies2: list[float] = []
    for row in alignment_scratch:
        if not row["legal"]:
            continue
        ps = row["proxy_score"]
        if not isinstance(ps, (int, float)) or not math.isfinite(float(ps)):
            continue
        if row["composite"] is None:
            continue
        filtered_ids2.append(str(row["candidate_id"]))
        filtered_composites.append(float(row["composite"]))
        filtered_proxies2.append(float(ps))

    spear = spearman_rho(filtered_composites, filtered_proxies2)
    kend = kendall_tau(filtered_composites, filtered_proxies2)
    mismatch = surrogate_good_proxy_bad_quartiles(filtered_ids2, filtered_composites, filtered_proxies2)

    surrogate_proxy_alignment: dict[str, Any] = {
        "n_candidates_total": len(alignment_scratch),
        "n_used_for_correlation": len(filtered_ids2),
        "spearman_rho": spear,
        "kendall_tau": kend,
        "surrogate_good_proxy_bad": mismatch,
    }

    search_alignment: dict[str, Any] = {
        "n_used_for_correlation": len(filtered_ids),
        "spearman_rho": spear_se,
        "kendall_tau": kend_se,
        "search_energy_vs_proxy_bad": mismatch_se,
        "note": "placement_energy(full) vs official proxy",
    }

    sampler_provenance: dict[str, Any] | None = {"solver": "search_hybrid", "seed_count": len(seeds)}

    results: dict[str, Any] = {
        "manifest": manifest.to_dict(),
        "benchmark_id": config.benchmark_id,
        "evaluator_backend": config.evaluator_backend,
        "testcase_root": str(testcase_path),
        "canvas_width": canvas_w,
        "canvas_height": canvas_h,
        "solver_backend": "search_hybrid",
        "search_solver_config": {
            "search_families": list(config.search_families),
            "search_seeds_per_family": config.search_seeds_per_family,
            "search_objective_schedule": config.search_objective_schedule,
            "search_adaptive_operators": config.search_adaptive_operators,
            "search_surrogate_min_spearman": config.search_surrogate_min_spearman,
            "search_official_eval_every_n_steps": config.search_official_eval_every_n_steps,
        },
        "guidance_sweep_requested": [],
        "guidance_sweep_resolved": [],
        "sampler_provenance": sampler_provenance,
        "sampler_backend": config.sampler_backend,
        "checkpoint_path": str(config.checkpoint_path) if config.checkpoint_path else None,
        "training_dataset_version": config.training_dataset_version,
        "ranking": ranking,
        "scoring_table": scoring_table,
        "surrogate_proxy_alignment": surrogate_proxy_alignment,
        "search_surrogate_proxy_alignment": search_alignment,
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
            "budget_limited": False,
            "budget_resolution_seconds": 0.0,
            "requested_num_candidates": config.num_candidates,
            "resolved_num_candidates": len(raw),
        },
        "runtime_budget": None,
        "sweep_vectors_used": 1,
        "sweep_vectors_requested": 1,
        "generation_stopped_early": False,
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
            "search_seconds": search_seconds,
            "mixed_size_seconds": mixed_size_seconds,
            "evaluation_seconds": evaluation_seconds,
            "legalize_mixed_size_eval_seconds": legalization_seconds + mixed_size_seconds + evaluation_seconds,
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
