"""PyTorch + PyG DDPM sampler implementing ``DiffusionSampler`` (Phase 4)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Batch, Data

from hrt_chip.config import RunConfig, TrainingConfig
from hrt_chip.data.graph_utils import complete_edge_index, node_features_from_wh
from hrt_chip.diffusion import (
    COORD_SPACE_NORMALIZED,
    CandidateSample,
    DiffusionSampleRequest,
    MacroCenter,
    SampleBatch,
    SamplerProvenance,
)
from hrt_chip.training.checkpoint import load_checkpoint, read_dataset_version_from_manifest
from hrt_chip.training.schedule import (
    build_reverse_timestep_list,
    ddim_step_eps,
    p_sample_step_eps,
)


class PyTorchDDPMSampler:
    """Loads a trained ε-model checkpoint and runs reverse diffusion (DDPM and/or DDIM)."""

    sampler_name = "pytorch_ddpm_sampler"
    generation_mode = "simultaneous_diffusion"

    def __init__(
        self,
        checkpoint_path: Path | str,
        *,
        device: torch.device | None = None,
        training_dataset_version: str | None = None,
        model_architecture: str | None = None,
        diffusion_inference_steps: int | None = None,
        sampler_mode: str = "ddpm_subsampled",
        diffusion_reverse_schedule: str | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = dev
        self.model, self.sched, self.meta = load_checkpoint(self.checkpoint_path, device=dev)
        dm = self.meta.get("dataset_manifest") or {}
        self.training_dataset_version = training_dataset_version or read_dataset_version_from_manifest(
            dm
        )
        tc = self.meta.get("training_config")
        if isinstance(tc, TrainingConfig):
            tr = tc
        elif isinstance(tc, dict):
            tr = TrainingConfig.from_dict(tc)
        else:
            tr = None
        if tr is not None:
            self.model_architecture = model_architecture or str(tr.model_architecture)
        else:
            self.model_architecture = model_architecture or "baseline_gnn"
        self.diffusion_inference_steps = diffusion_inference_steps
        self.sampler_mode = sampler_mode
        self.diffusion_reverse_schedule = diffusion_reverse_schedule

    @torch.no_grad()
    def sample_batch(self, request: DiffusionSampleRequest) -> SampleBatch:
        specs = request.macro_specs
        n = len(specs)
        wh = torch.tensor([[s.w, s.h] for s in specs], dtype=torch.float32, device=self.device)
        x_attr = node_features_from_wh(wh)
        edge_index = complete_edge_index(n).to(self.device)

        mode = (request.sampler_mode or self.sampler_mode).lower()
        if mode not in ("ddpm_full", "ddpm_subsampled", "ddim"):
            mode = "ddpm_subsampled"

        inf_steps = request.diffusion_inference_steps
        if inf_steps is None:
            inf_steps = self.diffusion_inference_steps

        T = self.sched.num_timesteps
        rev_steps, sched_desc = build_reverse_timestep_list(
            num_timesteps=T,
            sampler_mode=mode,
            num_inference_steps=inf_steps,
            custom_indices=request.reverse_timestep_indices,
            custom_schedule_str=self.diffusion_reverse_schedule,
        )
        eff_visits = len(rev_steps)

        out_candidates: list[CandidateSample] = []
        for idx in range(request.num_candidates):
            g = torch.Generator(device=self.device)
            g.manual_seed(int(request.seed) + idx * 1_000_003)

            data = Data(
                x=x_attr,
                edge_index=edge_index,
                pos=torch.zeros(n, 2, device=self.device),
                num_nodes=n,
            )
            batch = Batch.from_data_list([data])

            x = torch.randn(n, 2, generator=g, device=self.device, dtype=torch.float32)

            if mode == "ddim":
                for i in range(len(rev_steps) - 1):
                    t_cur = rev_steps[i]
                    t_next = rev_steps[i + 1]
                    t_node = torch.full((n,), t_cur, device=self.device, dtype=torch.long)
                    x = ddim_step_eps(self.model, x, t_cur, t_next, self.sched, batch, t_node)
            else:
                for t_scalar in rev_steps:
                    t_node = torch.full((n,), t_scalar, device=self.device, dtype=torch.long)
                    x = p_sample_step_eps(self.model, x, t_scalar, self.sched, batch, t_node)

            centers = tuple(
                MacroCenter(name=specs[i].name, cx=float(x[i, 0].item()), cy=float(x[i, 1].item()))
                for i in range(n)
            )
            cand_id = f"cand_{idx:04d}"
            out_candidates.append(CandidateSample(candidate_id=cand_id, centers=centers))

        guidance_dict = None
        if request.guidance is not None:
            gu = request.guidance
            guidance_dict = {
                "sweep_index": gu.sweep_index,
                "alpha_hpwl": gu.alpha_hpwl,
                "beta_congestion": gu.beta_congestion,
                "gamma_legality": gu.gamma_legality,
            }

        objective_bias_dict: dict[str, Any] | None = None
        if request.objective_bias is not None:
            objective_bias_dict = request.objective_bias.to_dict()

        prov = SamplerProvenance(
            sampler_name=self.sampler_name,
            model_stub=f"pytorch_checkpoint:{self.checkpoint_path.name}",
            generation_mode=self.generation_mode,
            coord_space=COORD_SPACE_NORMALIZED,
            seed=request.seed,
            num_candidates=request.num_candidates,
            diffusion_steps=eff_visits,
            guidance=guidance_dict,
            objective_bias=objective_bias_dict,
            checkpoint_path=str(self.checkpoint_path.resolve()),
            training_dataset_version=self.training_dataset_version,
            model_architecture=self.model_architecture,
            sampler_mode=mode,
            reverse_schedule_description=sched_desc,
        )
        return SampleBatch(candidates=tuple(out_candidates), provenance=prov)


def build_pytorch_sampler(config: RunConfig) -> PyTorchDDPMSampler:
    if config.checkpoint_path is None:
        raise ValueError("checkpoint_path is required for pytorch_checkpoint sampler backend")
    return PyTorchDDPMSampler(
        config.checkpoint_path,
        training_dataset_version=config.training_dataset_version,
        model_architecture=config.model_architecture,
        diffusion_inference_steps=config.diffusion_inference_steps,
        sampler_mode=config.sampler_mode,
        diffusion_reverse_schedule=config.diffusion_reverse_schedule,
    )
