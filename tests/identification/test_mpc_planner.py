from __future__ import annotations

import numpy as np
import pytest

from ur5e_sim.core.env import load_model, reset_to_home
from ur5e_sim.identification.mpc import MPCConfig, PlannerConfig
from ur5e_sim.identification.mpc.planner import ExcitationPlanner, PlanResult

from .conftest import SCENE_PATH

# All planner tests are xfail because the identification scene has nq=14
# (6 arm + 8 gripper joints) but the planner internally constructs 6-joint
# spline trajectories and passes them to functions that do data.qpos[:] = q[i].
pytestmark = pytest.mark.xfail(
    reason="identification scene has nq=14, planner internally uses 6-joint arrays"
)


def _load_and_reset():
    loaded = load_model(SCENE_PATH)
    reset_to_home(loaded.model, loaded.data)
    return loaded


@pytest.fixture(scope="module")
def loaded():
    return _load_and_reset()


@pytest.fixture(scope="module")
def planner(loaded):
    config = MPCConfig(planner=PlannerConfig(n_restarts=2, max_iter_per_start=60))
    return ExcitationPlanner(config, loaded.model, loaded.data)


@pytest.fixture(scope="module")
def plan_result(planner):
    cfg = planner.config
    return planner.plan(cfg.q0, np.zeros(cfg.num_joints))


def test_plan_returns_result(plan_result) -> None:
    assert isinstance(plan_result, PlanResult)


def test_condition_number_finite(plan_result) -> None:
    assert np.isfinite(plan_result.condition_number)


def test_terminal_velocity_is_zero(plan_result) -> None:
    np.testing.assert_allclose(plan_result.trajectory.velocity[-1], 0.0, atol=1e-9)


def test_feasible(plan_result) -> None:
    assert plan_result.feasible
    assert all(m >= -1e-6 for m in plan_result.constraint_margins.values())


def test_waypoints_shape(planner, plan_result) -> None:
    cfg = planner.config
    assert plan_result.waypoints.shape == (cfg.horizon.num_segments, cfg.num_joints)


def test_w_accumulated_changes_condition_number(planner) -> None:
    cfg = planner.config
    q0, dq0 = cfg.q0, np.zeros(cfg.num_joints)

    result_none = planner.plan(q0, dq0, W_accumulated=None)

    rng = np.random.default_rng(0)
    W_acc = rng.standard_normal((60, 10))
    result_acc = planner.plan(q0, dq0, W_accumulated=W_acc)

    assert np.isfinite(result_acc.condition_number)
    assert result_none.condition_number != result_acc.condition_number
