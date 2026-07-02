"""Tests for the MPC excitation loop."""

from __future__ import annotations

import numpy as np
import pytest

from ur5e_sim.core.layout import DofLayout
from ur5e_sim.identification.mpc.config import MPCConfig, PlannerConfig
from ur5e_sim.identification.mpc.loop import MPCLoop, _slice_trajectory
from ur5e_sim.trajectories.base import TrajectorySample

from .conftest import load_identification_scene


def _load_and_reset():
    return load_identification_scene()


def _short_config(loaded) -> MPCConfig:
    q0 = DofLayout.from_model(loaded.model).arm(loaded.data.qpos).copy()
    return MPCConfig(
        max_mpc_steps=2,
        planner=PlannerConfig(n_restarts=2, max_iter_per_start=20),
        q0=q0,
    )


def test_mpc_loop_runs():
    loaded = _load_and_reset()
    mpc = MPCLoop(_short_config(loaded), loaded.model, loaded.data)
    result = mpc.run()

    assert result is not None
    assert len(result.steps) <= 2
    assert result.final_estimation.phi.shape == (10,)
    assert np.isfinite(result.final_condition_number)
    assert result.total_samples > 0


def test_state_continuity():
    loaded = _load_and_reset()
    mpc = MPCLoop(_short_config(loaded), loaded.model, loaded.data)
    result = mpc.run()

    if len(result.steps) >= 2:
        np.testing.assert_allclose(result.steps[1].q_start, result.steps[0].q_end, atol=1e-9)


def test_slice_trajectory():
    fps = 100.0
    duration = 3.0
    n = int(round(duration * fps)) + 1
    time = np.linspace(0.0, duration, n)
    position = np.zeros((n, 6))
    traj = TrajectorySample(
        time=time,
        position=position,
        velocity=position.copy(),
        acceleration=position.copy(),
    )

    sliced = _slice_trajectory(traj, 0.0, 1.5)
    assert sliced.time[0] == 0.0
    assert abs(sliced.time[-1] - 1.5) < 1e-6
    assert sliced.position.shape[0] == sliced.time.shape[0]
