"""Linear beta schedule for DDPM (training + sampling), ε-prediction parameterization."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class DiffusionSchedule:
    """DDPM β schedule with precomputed coefficients (Ho et al., ε prediction)."""

    def __init__(self, num_timesteps: int, device: torch.device | None = None) -> None:
        self.num_timesteps = num_timesteps
        self.device = device or torch.device("cpu")
        betas = torch.linspace(1e-4, 0.02, num_timesteps, dtype=torch.float32, device=self.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)


def extract(a: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Index ``a`` by per-row timesteps ``t`` to match ``x`` shape [N, ...]."""
    out = a.gather(0, t)
    return out.reshape(t.shape[0], *((1,) * (x.dim() - 1)))


def q_sample(
    x0: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor,
    sched: DiffusionSchedule,
) -> torch.Tensor:
    """x_t = sqrt(α̅_t) x_0 + sqrt(1-α̅_t) ε ; ``t`` is per-row (per node)."""
    s1 = extract(sched.sqrt_alphas_cumprod, t, x0)
    s2 = extract(sched.sqrt_one_minus_alphas_cumprod, t, x0)
    return s1 * x0 + s2 * noise


@torch.no_grad()
def p_sample_step_eps(
    eps_model: torch.nn.Module,
    x: torch.Tensor,
    t_scalar: int,
    sched: DiffusionSchedule,
    batch: object,
    t_node: torch.Tensor,
) -> torch.Tensor:
    """
    One DDPM reverse step using ε_θ. ``t_scalar`` in [0, T-1]; ``t_node`` is long [num_nodes] filled with t_scalar.

    After loop, call with t_scalar=0 and use mean output as final x_0.
    """
    t = torch.full((x.shape[0],), t_scalar, device=x.device, dtype=torch.long)
    betas_t = extract(sched.betas, t, x)
    sqrt_one_minus_alphas_cumprod_t = extract(sched.sqrt_one_minus_alphas_cumprod, t, x)
    sqrt_recip_alphas_t = extract(sched.sqrt_recip_alphas, t, x)
    eps = eps_model(batch, x, t_node)
    model_mean = sqrt_recip_alphas_t * (x - betas_t * eps / sqrt_one_minus_alphas_cumprod_t)

    if t_scalar == 0:
        return model_mean

    posterior_variance_t = extract(sched.posterior_variance, t, x)
    noise = torch.randn_like(x)
    return model_mean + torch.sqrt(posterior_variance_t) * noise
