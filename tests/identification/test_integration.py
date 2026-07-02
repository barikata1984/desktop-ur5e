"""Integration tests exercising the full pipeline end-to-end."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from ur5e_sim.identification import (  # noqa: E402
    BatchLeastSquares,
    BatchLSConfig,
    body_inertial_parameters_from_model,
    ExcitationOptimizer,
    JointLimits,
    OptimizationResult,
    OptimizerConfig,
    PlaybackConfig,
    TrajectoryPlayback,
    WorkspaceConstraintConfig,
    _TrajectoryCache,
    compute_stacked_body_regressor,
    load_optimization_result,
    make_joint_acceleration_constraint,
    make_joint_position_constraint,
    make_joint_velocity_constraint,
    result_to_trajectory,
    save_optimization_result,
)
from ur5e_sim.trajectories import (  # noqa: E402
    WindowedFourierTrajectory,
    WindowedFourierTrajectoryConfig,
)
from ur5e_sim.trajectories.base import TrajectorySample  # noqa: E402

from .conftest import (  # noqa: E402
    arm_to_full_qpos,
    arm_to_full_qvel,
    load_identification_scene,
)

NUM_JOINTS = 6
Q0_ARM = np.array([np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0])
BODY_NAME = "payload_box_mount"


def _load_scene():
    return load_identification_scene()


def _pad_sample(sample, nq: int, nv: int) -> TrajectorySample:
    """Pad a 6-joint trajectory sample to full model dimensions."""
    return TrajectorySample(
        time=sample.time,
        position=arm_to_full_qpos(sample.position, nq),
        velocity=arm_to_full_qvel(sample.velocity, nv),
        acceleration=arm_to_full_qvel(sample.acceleration, nv),
    )


def test_optimization_and_validation_roundtrip() -> None:
    """Run a minimal optimization, save/load result, validate it."""
    loaded = _load_scene()
    cfg = OptimizerConfig(
        num_joints=NUM_JOINTS,
        num_harmonics=2,
        base_freq=0.2,
        duration=2.0,
        fps=50.0,
        q0=Q0_ARM,
        subsample_factor=10,
        n_monte_carlo=1,
        max_iter_per_start=3,
        seed=42,
        body_name=BODY_NAME,
        site_name="ft300s_ft_sensor",
    )
    opt = ExcitationOptimizer(cfg, loaded.model, loaded.data)
    result = opt.optimize()

    assert isinstance(result, OptimizationResult)
    assert np.isfinite(result.condition_number)
    assert result.condition_number > 0
    assert result.condition_number < 1e6

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "result.json"
        save_optimization_result(result, path)
        loaded_result = load_optimization_result(path)

    np.testing.assert_allclose(loaded_result.x_opt, result.x_opt)
    assert loaded_result.condition_number == pytest.approx(result.condition_number)
    np.testing.assert_allclose(loaded_result.a_opt, result.a_opt)
    np.testing.assert_allclose(loaded_result.b_opt, result.b_opt)
    np.testing.assert_allclose(loaded_result.q0, result.q0)

    sample = result_to_trajectory(loaded_result)
    expected_steps = int(cfg.duration * cfg.fps) + 1
    assert sample.position.shape == (expected_steps, NUM_JOINTS)
    assert sample.velocity.shape == (expected_steps, NUM_JOINTS)
    assert sample.acceleration.shape == (expected_steps, NUM_JOINTS)


def test_playback_and_estimation_pipeline() -> None:
    """Playback a trajectory and run estimation.

    On the assembled (build_ur5e_model) model the force/torque sensors are
    prefixed ``ft300s_*`` while PlaybackConfig still defaults to the unprefixed
    sensor names, so playback finds no usable FT sensor and falls back to the
    analytic wrench of the payload body alone (``payload_box_mount``). The
    estimated mass therefore matches the payload body's own mass, not the full
    FT-sensor subtree mass. (Reconciling the sensor names is a later stage.)
    """
    loaded = _load_scene()
    nq, nv = loaded.model.nq, loaded.model.nv
    duration = 1.0
    fps = 50.0

    rng = np.random.default_rng(42)
    coefficients = {
        "a": rng.uniform(-0.05, 0.05, size=(6, 2)).tolist(),
        "b": rng.uniform(-0.05, 0.05, size=(6, 2)).tolist(),
    }
    traj_config = WindowedFourierTrajectoryConfig(
        duration=duration,
        fps=fps,
        num_joints=NUM_JOINTS,
        num_harmonics=2,
        base_freq=0.2,
        coefficients=coefficients,
        q0=Q0_ARM,
    )
    traj_arm = WindowedFourierTrajectory(traj_config).sample()
    traj = _pad_sample(traj_arm, nq, nv)

    playback_cfg = PlaybackConfig(
        body_name=BODY_NAME,
        site_name="attachment_site",
        ft_site_name="ft300s_ft_sensor",
    )
    playback = TrajectoryPlayback(loaded.model, loaded.data, playback_cfg)
    buffer = playback.execute(traj)

    expected_steps = int(duration * fps) + 1
    assert len(buffer) == expected_steps

    arrays = buffer.to_arrays()
    assert arrays["q"].shape == (expected_steps, nq)
    assert arrays["wrench"].shape == (expected_steps, 6)

    regressor = compute_stacked_body_regressor(
        loaded.model,
        loaded.data,
        arrays["q"],
        arrays["dq"],
        arrays["ddq"],
        BODY_NAME,
        subsample_factor=5,
        site_name="ft300s_ft_sensor",
    )
    wrench_stacked = arrays["wrench"][::5].reshape(-1)

    estimator = BatchLeastSquares(BatchLSConfig())
    result = estimator.estimate(regressor, wrench_stacked)

    expected_mass = body_inertial_parameters_from_model(loaded.model, BODY_NAME).mass
    assert result.mass == pytest.approx(expected_mass, rel=0.1)


def test_constraints_accept_small_reject_large() -> None:
    """Small-amplitude trajectory passes all constraints, large fails."""
    joint_limits = JointLimits()

    num_harmonics = 2
    cache_small = _TrajectoryCache(
        num_joints=NUM_JOINTS,
        num_harmonics=num_harmonics,
        base_freq=0.2,
        duration=2.0,
        fps=50.0,
        q0=Q0_ARM,
    )

    rng = np.random.default_rng(42)
    n = NUM_JOINTS * num_harmonics
    x_small = rng.uniform(-0.01, 0.01, size=2 * n)

    pos_fn = make_joint_position_constraint(cache_small, joint_limits)
    vel_fn = make_joint_velocity_constraint(cache_small, joint_limits)
    acc_fn = make_joint_acceleration_constraint(cache_small, joint_limits)

    assert pos_fn(x_small) >= 0
    assert vel_fn(x_small) >= 0
    assert acc_fn(x_small) >= 0

    cache_large = _TrajectoryCache(
        num_joints=NUM_JOINTS,
        num_harmonics=num_harmonics,
        base_freq=0.2,
        duration=2.0,
        fps=50.0,
        q0=Q0_ARM,
    )
    x_large = rng.uniform(-100.0, 100.0, size=2 * n)

    pos_fn_l = make_joint_position_constraint(cache_large, joint_limits)
    vel_fn_l = make_joint_velocity_constraint(cache_large, joint_limits)
    acc_fn_l = make_joint_acceleration_constraint(cache_large, joint_limits)

    margins = [
        pos_fn_l(x_large),
        vel_fn_l(x_large),
        acc_fn_l(x_large),
    ]
    assert any(m < 0 for m in margins), (
        "At least one constraint should be violated for large amplitude"
    )


def test_json_io_preserves_all_fields() -> None:
    """Save and load optimization result, verify all fields match."""
    cfg = OptimizerConfig(
        num_joints=NUM_JOINTS,
        num_harmonics=3,
        base_freq=0.15,
        duration=5.0,
        fps=100.0,
        q0=Q0_ARM,
    )
    rng = np.random.default_rng(99)
    n = NUM_JOINTS * 3
    x_opt = rng.uniform(-0.1, 0.1, size=2 * n)
    a_opt = x_opt[:n].reshape(NUM_JOINTS, 3)
    b_opt = x_opt[n:].reshape(NUM_JOINTS, 3)

    original = OptimizationResult(
        x_opt=x_opt,
        condition_number=123.456,
        a_opt=a_opt,
        b_opt=b_opt,
        q0=Q0_ARM.copy(),
        config=cfg,
        n_evaluations=500,
        wall_time=12.34,
        n_restarts=20,
        best_start_index=7,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_result.json"
        save_optimization_result(original, path)
        loaded = load_optimization_result(path)

    np.testing.assert_allclose(loaded.x_opt, original.x_opt)
    np.testing.assert_allclose(loaded.a_opt, original.a_opt)
    np.testing.assert_allclose(loaded.b_opt, original.b_opt)
    np.testing.assert_allclose(loaded.q0, original.q0)
    assert loaded.condition_number == pytest.approx(original.condition_number)
    assert loaded.n_evaluations == original.n_evaluations
    assert loaded.wall_time == pytest.approx(original.wall_time)
    assert loaded.n_restarts == original.n_restarts
    assert loaded.best_start_index == original.best_start_index
    assert loaded.config.num_joints == original.config.num_joints
    assert loaded.config.num_harmonics == original.config.num_harmonics
    assert loaded.config.base_freq == pytest.approx(original.config.base_freq)
    assert loaded.config.duration == pytest.approx(original.config.duration)
    assert loaded.config.fps == pytest.approx(original.config.fps)
    assert loaded.config.body_name == original.config.body_name
    assert loaded.config.site_name == original.config.site_name


def test_json_io_preserves_previously_defaulted_fields() -> None:
    """Fields that used to silently reset to defaults survive save/load.

    Regression test for the JSON round-trip only persisting a subset of
    OptimizerConfig: objective_type, with_ft_offset, joint_limits, and
    workspace_config previously reverted to their dataclass defaults after a
    load, even when the original config set them to non-default values.
    """
    custom_joint_limits = JointLimits(
        q_min=np.full(6, -1.5),
        q_max=np.full(6, 1.5),
        dq_max=np.full(6, 2.0),
        ddq_max=np.full(6, 5.0),
    )
    custom_workspace_config = WorkspaceConstraintConfig(
        max_displacement=0.3,
        box_lower=np.array([-0.2, -0.2, 0.0]),
        box_upper=np.array([0.2, 0.2, 0.4]),
        safety_margin=0.02,
    )
    cfg = OptimizerConfig(
        num_joints=NUM_JOINTS,
        num_harmonics=3,
        base_freq=0.15,
        duration=5.0,
        fps=100.0,
        q0=Q0_ARM,
        objective_type="condition_number",
        with_ft_offset=True,
        joint_limits=custom_joint_limits,
        workspace_config=custom_workspace_config,
    )
    n = NUM_JOINTS * 3
    rng = np.random.default_rng(7)
    x_opt = rng.uniform(-0.1, 0.1, size=2 * n)
    original = OptimizationResult(
        x_opt=x_opt,
        condition_number=42.0,
        a_opt=x_opt[:n].reshape(NUM_JOINTS, 3),
        b_opt=x_opt[n:].reshape(NUM_JOINTS, 3),
        q0=Q0_ARM.copy(),
        config=cfg,
        n_evaluations=100,
        wall_time=1.23,
        n_restarts=5,
        best_start_index=2,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_result.json"
        save_optimization_result(original, path)
        loaded = load_optimization_result(path)

    assert loaded.config.objective_type == "condition_number"
    assert loaded.config.with_ft_offset is True
    assert loaded.config.joint_limits is not None
    np.testing.assert_allclose(loaded.config.joint_limits.q_min, custom_joint_limits.q_min)
    np.testing.assert_allclose(loaded.config.joint_limits.q_max, custom_joint_limits.q_max)
    np.testing.assert_allclose(loaded.config.joint_limits.dq_max, custom_joint_limits.dq_max)
    np.testing.assert_allclose(loaded.config.joint_limits.ddq_max, custom_joint_limits.ddq_max)
    assert loaded.config.workspace_config is not None
    assert loaded.config.workspace_config.max_displacement == pytest.approx(0.3)
    np.testing.assert_allclose(
        loaded.config.workspace_config.box_lower, custom_workspace_config.box_lower
    )
    np.testing.assert_allclose(
        loaded.config.workspace_config.box_upper, custom_workspace_config.box_upper
    )
    assert loaded.config.workspace_config.safety_margin == pytest.approx(0.02)
