"""Phase 3: guidance sweep, scoring table, strict official-proxy selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from hrt_chip.config import (
    GUIDANCE_PRESET_PARETO3,
    RunConfig,
    resolved_guidance_sweep,
)
from hrt_chip.diffusion import DiffusionSampleRequest
from hrt_chip import mixed_size_metrics as msm
from hrt_chip.pipeline import run_pipeline
from hrt_chip.stages import generate as generate_mod
from hrt_chip.stages.generate import derive_sweep_seed, generate_candidates


def test_resolved_guidance_sweep_explicit_overrides_preset() -> None:
    custom = ((1.0, 0.0, 0.0),)
    r = resolved_guidance_sweep(guidance_preset="pareto3", guidance_weights_sweep=custom)
    assert r == custom


def test_resolved_guidance_sweep_pareto3() -> None:
    r = resolved_guidance_sweep(guidance_preset="pareto3", guidance_weights_sweep=None)
    assert r == GUIDANCE_PRESET_PARETO3


def test_derive_sweep_seed_index_zero_matches_base() -> None:
    assert derive_sweep_seed(42, 0, (0.1, 0.2, 0.7)) == 42


def test_multi_sweep_generates_prefixed_candidate_ids() -> None:
    sweep = ((0.8, 0.1, 0.1), (0.2, 0.7, 0.1))
    cands = generate_candidates(
        benchmark_id="ibm01",
        seed=1,
        num_candidates=2,
        diffusion_steps=100,
        guidance_sweep=sweep,
    )
    assert len(cands) == 4
    ids = [c.candidate_id for c in cands]
    assert "s00_cand_0000" in ids
    assert "s01_cand_0001" in ids
    for c in cands:
        g = c.metadata.get("guidance")
        assert isinstance(g, dict)
        assert "sweep_index" in g
        assert "alpha_hpwl" in g


def test_sample_batch_once_per_weight_vector() -> None:
    from hrt_chip.diffusion import DeterministicDDPMStubSampler

    calls: list[DiffusionSampleRequest] = []

    class SpySampler(DeterministicDDPMStubSampler):
        def sample_batch(self, request: DiffusionSampleRequest):  # type: ignore[override]
            calls.append(request)
            return super().sample_batch(request)

    sweep = ((0.8, 0.1, 0.1), (0.2, 0.7, 0.1), (0.4, 0.4, 0.2))
    generate_candidates(
        benchmark_id="ibm01",
        seed=5,
        num_candidates=2,
        sampler=SpySampler(),
        guidance_sweep=sweep,
    )
    assert len(calls) == 3
    for i, req in enumerate(calls):
        assert req.guidance is not None
        assert req.guidance.sweep_index == i


def test_pipeline_scoring_table_and_argmin(tmp_path: Path) -> None:
    cfg = RunConfig(
        benchmark_id="ibm01",
        seed=3,
        num_candidates=2,
        guidance_preset="pareto3",
        output_dir=tmp_path,
        deterministic=True,
    )
    r = run_pipeline(cfg, run_id="00000000-0000-0000-0000-000000000020")
    assert "scoring_table" in r
    st = r["scoring_table"]
    assert len(st) == 6  # 3 weights * 2 candidates
    for row in st:
        assert "candidate_id" in row
        assert "proxy_score" in row
        assert "legal" in row
        assert "guidance" in row
        assert "surrogate_objectives" in row
        so = row["surrogate_objectives"]
        assert "phi_hpwl" in so and "phi_congestion" in so and "phi_legality" in so
        assert "hard_overlap_pairs" in so and "smooth_overlap_penalty" in so
        assert "surrogate_mode" in so
        assert "mixed_size_profile" in row

    spa = r["surrogate_proxy_alignment"]
    assert spa["n_candidates_total"] == 6
    assert "spearman_rho" in spa and "kendall_tau" in spa
    assert "surrogate_good_proxy_bad" in spa

    ranking = r["ranking"]
    assert ranking == sorted(ranking, key=msm.ranking_key_proxy_first)
    assert r["best_candidate_id"] == ranking[0]["candidate_id"]
    assert r["best_proxy_score"] == ranking[0]["proxy_score"]
    assert r.get("selection_policy") == "proxy_first"
    assert r.get("guidance_sweep_resolved") == [list(t) for t in GUIDANCE_PRESET_PARETO3]


def test_ppa_priority_selects_better_mixed_size_over_proxy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """With equal proxy ordering concern, ppa_priority picks lower composite mixed-size score."""

    from hrt_chip.adapters.evaluator.base import EvaluationResult, EvaluatorAdapter
    from hrt_chip.adapters.mixed_size.base import MixedSizeBackend, MixedSizeRequest, MixedSizeResult
    from hrt_chip.models import MacroRect, PlacementCandidate

    class SeqMs(MixedSizeBackend):
        def __init__(self) -> None:
            self._n = 0
            self._extras = (
                {
                    "density_overflow": 10.0,
                    "rudy_or_route_proxy": 10.0,
                    "backend_runtime_seconds": 10.0,
                },
                {
                    "density_overflow": 0.01,
                    "rudy_or_route_proxy": 0.01,
                    "backend_runtime_seconds": 0.01,
                },
            )

        def run(self, request: MixedSizeRequest) -> MixedSizeResult:
            ex = dict(self._extras[min(self._n, len(self._extras) - 1)])
            self._n += 1
            return MixedSizeResult(ok=True, message="ok", extra=ex)

    class ProxyByOrder(EvaluatorAdapter):
        def evaluate(self, candidate, *, benchmark_id, run_context):  # type: ignore[no-untyped-def]
            is_first = candidate.candidate_id.endswith("0000")
            p = 1.0 if is_first else 2.0
            return EvaluationResult(candidate.candidate_id, p, True, {})

    def fake_gen(**kwargs: object) -> list[PlacementCandidate]:
        return [
            PlacementCandidate(
                "s00_cand_0000",
                "ibm01",
                [
                    MacroRect("ibm01_M0", 0.0, 0.0, 0.12, 0.08),
                    MacroRect("ibm01_M1", 0.5, 0.5, 0.10, 0.09),
                ],
                metadata={"stage": "generated", "sampler": {}, "guidance": {"sweep_index": 0}},
            ),
            PlacementCandidate(
                "s00_cand_0001",
                "ibm01",
                [
                    MacroRect("ibm01_M0", 0.02, 0.02, 0.12, 0.08),
                    MacroRect("ibm01_M1", 0.52, 0.52, 0.10, 0.09),
                ],
                metadata={"stage": "generated", "sampler": {}, "guidance": {"sweep_index": 0}},
            ),
        ]

    monkeypatch.setattr("hrt_chip.pipeline.generate_candidates", fake_gen)
    rid0 = "00000000-0000-0000-0000-0000000000a1"
    rid1 = "00000000-0000-0000-0000-0000000000a2"
    cfg_pf = RunConfig(
        benchmark_id="ibm01",
        seed=1,
        num_candidates=2,
        output_dir=tmp_path,
        deterministic=True,
        selection_policy="proxy_first",
    )
    r_pf = run_pipeline(cfg_pf, mixed_size=SeqMs(), evaluator=ProxyByOrder(), run_id=rid0)
    assert r_pf["best_candidate_id"] == "s00_cand_0000"

    cfg_pp = RunConfig(
        benchmark_id="ibm01",
        seed=1,
        num_candidates=2,
        output_dir=tmp_path,
        deterministic=True,
        selection_policy="ppa_priority",
    )
    r_pp = run_pipeline(cfg_pp, mixed_size=SeqMs(), evaluator=ProxyByOrder(), run_id=rid1)
    assert r_pp["best_candidate_id"] == "s00_cand_0001"
    assert r_pp["ranking"] == sorted(r_pp["ranking"], key=msm.ranking_key_ppa_priority)


def test_weights_do_not_override_proxy_selection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Official proxy is authoritative; surrogate objectives are metadata only."""

    from hrt_chip.adapters.evaluator.local_stub import LocalStubEvaluator
    from hrt_chip.models import MacroRect, PlacementCandidate

    def fake_gen(**kwargs: object) -> list[PlacementCandidate]:
        return [
            PlacementCandidate(
                candidate_id="cand_a",
                benchmark_id="ibm01",
                macros=[
                    MacroRect("ibm01_M0", 0.0, 0.0, 0.12, 0.08),
                    MacroRect("ibm01_M1", 0.5, 0.5, 0.10, 0.09),
                ],
                metadata={"stage": "generated", "sampler": {}, "guidance": {"sweep_index": 0}},
            ),
            PlacementCandidate(
                candidate_id="cand_b",
                benchmark_id="ibm01",
                macros=[
                    MacroRect("ibm01_M0", 0.01, 0.01, 0.12, 0.08),
                    MacroRect("ibm01_M1", 0.51, 0.51, 0.10, 0.09),
                ],
                metadata={"stage": "generated", "sampler": {}, "guidance": {"sweep_index": 1}},
            ),
        ]

    class FixedProxyEvaluator(LocalStubEvaluator):
        def evaluate(self, candidate, *, benchmark_id, run_context):  # type: ignore[no-untyped-def]
            from hrt_chip.adapters.evaluator.base import EvaluationResult

            # cand_b wins on official proxy regardless of surrogate objectives
            proxy = 10.0 if candidate.candidate_id == "cand_a" else 1.0
            return EvaluationResult(
                candidate_id=candidate.candidate_id,
                proxy_score=proxy,
                legal=True,
                details={"stub": True},
            )

    monkeypatch.setattr("hrt_chip.pipeline.generate_candidates", fake_gen)
    cfg = RunConfig(
        benchmark_id="ibm01",
        seed=1,
        num_candidates=2,
        output_dir=tmp_path,
        deterministic=True,
    )
    r = run_pipeline(cfg, evaluator=FixedProxyEvaluator(), run_id="00000000-0000-0000-0000-000000000021")
    assert r["best_candidate_id"] == "cand_b"
    assert r["best_proxy_score"] == 1.0


def test_no_sequential_rl_strings_in_generate() -> None:
    import inspect

    src = inspect.getsource(generate_mod.generate_candidates)
    assert "sample_batch" in src
    assert "sequential" not in src.lower()
    assert "maskplace" not in src.lower()
