"""Custom reverse schedules and timestep list builder."""

from hrt_chip.training.schedule import (
    build_reverse_timestep_list,
    parse_custom_reverse_timesteps,
    subsampled_reverse_timesteps,
)


def test_parse_custom_maps_t_to_index() -> None:
    seq = parse_custom_reverse_timesteps(num_timesteps=1000, schedule_str="1000,500,0")
    assert seq[0] == 999
    assert 499 in seq or 500 in seq  # rounded spacing
    assert seq[-1] == 0


def test_build_reverse_subsampled() -> None:
    seq, desc = build_reverse_timestep_list(
        num_timesteps=100,
        sampler_mode="ddpm_subsampled",
        num_inference_steps=10,
        custom_indices=None,
        custom_schedule_str=None,
    )
    assert len(seq) <= 100
    assert seq[-1] == 0
    assert "inference_steps" in desc


def test_build_reverse_ddim_uses_same_list() -> None:
    seq, _ = build_reverse_timestep_list(
        num_timesteps=50,
        sampler_mode="ddim",
        num_inference_steps=5,
        custom_indices=None,
        custom_schedule_str=None,
    )
    assert seq[-1] == 0
    raw = subsampled_reverse_timesteps(num_timesteps=50, num_inference_steps=5)
    assert seq == raw
