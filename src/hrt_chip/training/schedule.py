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


def parse_custom_reverse_timesteps(*, num_timesteps: int, schedule_str: str) -> list[int]:
    """
    Parse comma-separated timestep indices (e.g. ``1000,500,250,0`` for ``T=1000`` → indices ``999,499,...``).

    Accepts values in ``[0, num_timesteps]`` where ``num_timesteps`` is treated as ``T`` (maps ``T`` → index ``T-1``).
    Returns strictly decreasing indices ending at 0.
    """
    T = int(num_timesteps)
    parts = [p.strip() for p in schedule_str.split(",") if p.strip()]
    if not parts:
        raise ValueError("empty diffusion_reverse_schedule")
    raw: list[int] = []
    for p in parts:
        v = int(float(p))
        if v < 0:
            raise ValueError(f"invalid timestep {v} in schedule")
        if v > T:
            raise ValueError(f"timestep {v} exceeds training timesteps T={T}")
        idx = T - 1 if v == T else v
        if not (0 <= idx < T):
            raise ValueError(f"resolved index {idx} out of range for T={T}")
        raw.append(idx)
    out: list[int] = []
    seen: set[int] = set()
    for t in sorted(raw, reverse=True):
        if t not in seen:
            seen.add(t)
            out.append(t)
    if not out:
        raise ValueError("no timesteps after parse")
    if out[-1] != 0:
        out.append(0)
        out = sorted(set(out), reverse=True)
    return out


def build_reverse_timestep_list(
    *,
    num_timesteps: int,
    sampler_mode: str,
    num_inference_steps: int | None,
    custom_indices: tuple[int, ...] | None,
    custom_schedule_str: str | None,
) -> tuple[list[int], str]:
    """
    Monotonic decreasing indices for reverse diffusion.

    Returns ``(timesteps_high_to_low, description)``.
    """
    T = int(num_timesteps)
    if custom_indices is not None and len(custom_indices) > 0:
        seq = list(custom_indices)
    elif custom_schedule_str is not None and custom_schedule_str.strip():
        seq = parse_custom_reverse_timesteps(num_timesteps=T, schedule_str=custom_schedule_str)
    elif sampler_mode == "ddpm_full":
        seq = list(range(T - 1, -1, -1))
        return seq, f"ddpm_full:T={T}"
    else:
        seq = subsampled_reverse_timesteps(num_timesteps=T, num_inference_steps=num_inference_steps)
        return seq, f"{sampler_mode}:inference_steps={num_inference_steps}"
    # custom list path
    out: list[int] = []
    seen: set[int] = set()
    for t in sorted(seq, reverse=True):
        if 0 <= t < T and t not in seen:
            seen.add(t)
            out.append(t)
    if 0 not in out:
        out.append(0)
        out = sorted(set(out), reverse=True)
    return out, f"custom:{','.join(str(x) for x in out)}"


@torch.no_grad()
def subsampled_reverse_timesteps(*, num_timesteps: int, num_inference_steps: int | None) -> list[int]:
    """
    Monotonic decreasing timestep indices for accelerated sampling.

    When ``num_inference_steps`` is None or >= T, returns full ``[T-1, ..., 0]``.
    Always ends at 0 so the final denoising mean is applied at t=0.
    """
    T = int(num_timesteps)
    if T <= 0:
        return []
    if num_inference_steps is None or int(num_inference_steps) <= 0 or int(num_inference_steps) >= T:
        return list(range(T - 1, -1, -1))
    ns = max(2, min(int(num_inference_steps), T))
    raw = [int(round(i * (T - 1) / max(ns - 1, 1))) for i in range(ns)]
    if 0 not in raw:
        raw.append(0)
    out: list[int] = []
    seen: set[int] = set()
    for t in sorted(raw, reverse=True):
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


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


@torch.no_grad()
def ddim_step_eps(
    eps_model: torch.nn.Module,
    x: torch.Tensor,
    t_cur: int,
    t_next: int,
    sched: DiffusionSchedule,
    batch: object,
    t_node: torch.Tensor,
) -> torch.Tensor:
    """
    One DDIM update (η=0) from timestep ``t_cur`` to ``t_next`` (``t_next < t_cur``), ε-prediction model.

    ``t_node`` must be filled with ``t_cur`` for each row (per-node timestep tensor).
    """
    eps = eps_model(batch, x, t_node)
    ab_cur = sched.alphas_cumprod[t_cur].to(device=x.device, dtype=x.dtype)
    ab_next = sched.alphas_cumprod[t_next].to(device=x.device, dtype=x.dtype)
    pred_x0 = (x - torch.sqrt(1.0 - ab_cur) * eps) / torch.sqrt(ab_cur + 1e-8)
    return torch.sqrt(ab_next) * pred_x0 + torch.sqrt(1.0 - ab_next) * eps
