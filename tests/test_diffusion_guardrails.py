"""Phase 2: diffusion sampler contract and no-sequential-RL guardrails."""

from __future__ import annotations

from pathlib import Path

from hrt_chip.config import RunConfig
from hrt_chip.diffusion import DeterministicDDPMStubSampler, DiffusionSampleRequest, MacroSpec
from hrt_chip.pipeline import run_pipeline
from hrt_chip.stages import generate as generate_mod
from hrt_chip.stages.generate import generate_candidates


def test_stub_sampler_outputs_normalized_centers_and_full_macro_set() -> None:
    specs = (
        MacroSpec("a", 0.1, 0.1),
        MacroSpec("b", 0.2, 0.2),
    )
    req = DiffusionSampleRequest(
        benchmark_id="ibm01",
        seed=123,
        num_candidates=5,
        macro_specs=specs,
        diffusion_steps=1000,
    )
    batch = DeterministicDDPMStubSampler().sample_batch(req)
    assert len(batch.candidates) == 5
    for cand in batch.candidates:
        assert len(cand.centers) == 2
        for mc in cand.centers:
            assert -1.0 <= mc.cx <= 1.0
            assert -1.0 <= mc.cy <= 1.0
    assert batch.provenance.generation_mode == "simultaneous_diffusion"
    assert batch.provenance.coord_space == "normalized_centers_-1_1"


def test_generate_candidates_attaches_sampler_provenance() -> None:
    cands = generate_candidates(
        benchmark_id="ibm01",
        seed=99,
        num_candidates=2,
        diffusion_steps=500,
    )
    assert len(cands) == 2
    for c in cands:
        s = c.metadata.get("sampler")
        assert isinstance(s, dict)
        assert s["sampler_name"] == "ddpm_stub_sampler"
        assert s["generation_mode"] == "simultaneous_diffusion"
        assert s["diffusion_steps"] == 500
        nc = c.metadata.get("normalized_centers")
        assert isinstance(nc, list)
        assert len(nc) == 2


def test_pipeline_includes_sampler_provenance(tmp_path: Path) -> None:
    cfg = RunConfig(
        benchmark_id="ibm01",
        seed=1,
        num_candidates=2,
        diffusion_steps=1000,
        output_dir=tmp_path,
        deterministic=True,
    )
    r = run_pipeline(cfg, run_id="00000000-0000-0000-0000-000000000010")
    sp = r.get("sampler_provenance")
    assert isinstance(sp, dict)
    assert sp.get("generation_mode") == "simultaneous_diffusion"


def test_sample_batch_called_once_per_generate_batch() -> None:
    calls: list[DiffusionSampleRequest] = []

    class SpySampler(DeterministicDDPMStubSampler):
        def sample_batch(self, request: DiffusionSampleRequest):  # type: ignore[override]
            calls.append(request)
            return super().sample_batch(request)

    generate_candidates(benchmark_id="ibm01", seed=5, num_candidates=8, sampler=SpySampler())
    assert len(calls) == 1
    assert calls[0].num_candidates == 8
    assert len(calls[0].macro_specs) == 2


def test_no_sequential_placement_module_in_generate() -> None:
    import inspect

    src = inspect.getsource(generate_mod.generate_candidates)
    assert "sample_batch" in src
    assert "sequential" not in src.lower()
    assert "maskplace" not in src.lower()
    assert "mdp" not in src.lower()
