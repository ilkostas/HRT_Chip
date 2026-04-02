"""Accelerated reverse diffusion timestep schedules."""

from __future__ import annotations

from hrt_chip.training.schedule import subsampled_reverse_timesteps


def test_subsampled_full_when_none() -> None:
    full = list(range(99, -1, -1))
    got = subsampled_reverse_timesteps(num_timesteps=100, num_inference_steps=None)
    assert got == full


def test_subsampled_shorter() -> None:
    got = subsampled_reverse_timesteps(num_timesteps=100, num_inference_steps=10)
    assert got[0] == 99
    assert got[-1] == 0
    assert len(got) < 100
    assert len(got) >= 2
