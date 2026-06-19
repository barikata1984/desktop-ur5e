import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from ur5e_sim.core.env import load_model, reset_to_home  # noqa: E402
from ur5e_sim.identification import (  # noqa: E402
    body_inertial_parameters_from_model,
    compute_condition_number,
    compute_stacked_body_regressor,
    compute_wrench_from_parameters,
    sample_body_regressor,
    set_model_state,
)
from ur5e_sim.trajectories import (  # noqa: E402
    WindowedFourierTrajectory,
    WindowedFourierTrajectoryConfig,
)

from .conftest import SCENE_PATH, arm_to_full_qpos, arm_to_full_qvel  # noqa: E402

PAYLOAD_BODY_NAME = "payload_box_mount"
Q0_ARM = np.array(
    [
        np.pi / 2,
        -np.pi / 2,
        np.pi / 2,
        -np.pi / 2,
        -np.pi / 2,
        0.0,
    ]
)


def _load_payload_scene():
    loaded = load_model(SCENE_PATH)
    reset_to_home(loaded.model, loaded.data)
    return loaded


def _trajectory_arm(scale: float = 0.1):
    """Generate a 6-joint windowed Fourier trajectory sample (arm joints only)."""
    generator = np.random.default_rng(42)
    coefficients = {
        "a": generator.uniform(-scale, scale, size=(6, 2)).tolist(),
        "b": generator.uniform(-scale, scale, size=(6, 2)).tolist(),
    }
    return WindowedFourierTrajectory(
        WindowedFourierTrajectoryConfig(
            duration=4.0,
            fps=50.0,
            num_joints=6,
            num_harmonics=2,
            base_freq=0.2,
            coefficients=coefficients,
            q0=Q0_ARM,
        )
    ).sample()


def _pad_sample(sample, nq: int, nv: int):
    """Pad a 6-joint trajectory sample to full model dimensions."""
    from ur5e_sim.trajectories.base import TrajectorySample

    return TrajectorySample(
        time=sample.time,
        position=arm_to_full_qpos(sample.position, nq),
        velocity=arm_to_full_qvel(sample.velocity, nv),
        acceleration=arm_to_full_qvel(sample.acceleration, nv),
    )


def test_body_inertial_parameters_extract_expected_payload_values() -> None:
    loaded = _load_payload_scene()
    parameters = body_inertial_parameters_from_model(loaded.model, PAYLOAD_BODY_NAME)

    # Values reflect the new identification scene payload geometry:
    # payload_blue_body (0.4 kg) + payload_red_body (0.6 kg) = 1.0 kg total.
    np.testing.assert_allclose(parameters.mass, 1.0)
    np.testing.assert_allclose(parameters.first_moments, np.array([0.0, 0.0, 0.19]))
    np.testing.assert_allclose(
        parameters.inertia_matrix,
        np.array(
            [
                [0.04816667, 0.0, 0.0],
                [0.0, 0.04416667, 0.0],
                [0.0, 0.0, 0.00566667],
            ]
        ),
        atol=1e-4,
    )


def test_sample_body_regressor_has_expected_shape() -> None:
    loaded = _load_payload_scene()
    sample = sample_body_regressor(loaded.model, loaded.data, PAYLOAD_BODY_NAME)

    assert sample.regressor.shape == (6, 10)
    assert sample.kinematics.body_name == PAYLOAD_BODY_NAME


def test_static_pose_regressor_predicts_gravity_wrench() -> None:
    loaded = _load_payload_scene()
    parameters = body_inertial_parameters_from_model(loaded.model, PAYLOAD_BODY_NAME)
    set_model_state(
        loaded.model,
        loaded.data,
        loaded.data.qpos.copy(),
        np.zeros(loaded.model.nv),
        np.zeros(loaded.model.nv),
    )
    sample = sample_body_regressor(loaded.model, loaded.data, PAYLOAD_BODY_NAME)
    wrench = compute_wrench_from_parameters(sample.regressor, parameters)

    assert wrench.shape == (6,)
    # Static gravity reaction: |force| = m*g = 1.0 * 9.81 = 9.81 N.
    np.testing.assert_allclose(np.linalg.norm(wrench[3:]), 9.81, atol=1e-3)


def test_stacked_body_regressor_shape_and_condition_number_change_with_motion() -> None:
    loaded = _load_payload_scene()
    nq, nv = loaded.model.nq, loaded.model.nv

    q0_full = arm_to_full_qpos(Q0_ARM, nq)
    static_q = np.tile(q0_full, (21, 1))
    static_dq = np.zeros((21, nv))
    static_ddq = np.zeros((21, nv))

    static_regressor = compute_stacked_body_regressor(
        loaded.model,
        loaded.data,
        static_q,
        static_dq,
        static_ddq,
        PAYLOAD_BODY_NAME,
    )

    dynamic_arm = _trajectory_arm(scale=0.2)
    dynamic = _pad_sample(dynamic_arm, nq, nv)
    dynamic_regressor = compute_stacked_body_regressor(
        loaded.model,
        loaded.data,
        dynamic.position,
        dynamic.velocity,
        dynamic.acceleration,
        PAYLOAD_BODY_NAME,
        subsample_factor=10,
    )

    assert static_regressor.shape == (21 * 6, 10)
    assert dynamic_regressor.shape[1] == 10

    static_condition = compute_condition_number(static_regressor)
    dynamic_condition = compute_condition_number(dynamic_regressor)
    static_rank = np.linalg.matrix_rank(static_regressor)
    dynamic_rank = np.linalg.matrix_rank(dynamic_regressor)

    assert np.isinf(static_condition) or static_condition > 1e8
    assert np.isfinite(dynamic_condition)
    assert dynamic_rank == 10
    assert static_rank < dynamic_rank


def test_set_model_state_preserves_qacc() -> None:
    loaded = _load_payload_scene()
    rng = np.random.default_rng(99)
    qacc_desired = rng.uniform(-5, 5, loaded.model.nv)
    q0_full = arm_to_full_qpos(Q0_ARM, loaded.model.nq)
    set_model_state(
        loaded.model,
        loaded.data,
        q0_full,
        rng.uniform(-1, 1, loaded.model.nv),
        qacc_desired,
    )
    np.testing.assert_allclose(loaded.data.qacc, qacc_desired)


def test_dynamic_trajectory_regressor_predicts_wrench() -> None:
    loaded = _load_payload_scene()
    nq, nv = loaded.model.nq, loaded.model.nv
    parameters = body_inertial_parameters_from_model(loaded.model, PAYLOAD_BODY_NAME)

    dynamic_arm = _trajectory_arm(scale=0.3)
    dynamic = _pad_sample(dynamic_arm, nq, nv)
    idx = dynamic.position.shape[0] // 2
    set_model_state(
        loaded.model,
        loaded.data,
        dynamic.position[idx],
        dynamic.velocity[idx],
        dynamic.acceleration[idx],
    )
    sample = sample_body_regressor(loaded.model, loaded.data, PAYLOAD_BODY_NAME)
    wrench = compute_wrench_from_parameters(sample.regressor, parameters)

    assert wrench.shape == (6,)
    assert np.all(np.isfinite(wrench))
    assert np.linalg.norm(wrench[:3]) > 0, "torque should be non-zero during motion"


def test_set_model_state_validates_shapes() -> None:
    loaded = _load_payload_scene()
    with pytest.raises(ValueError):
        set_model_state(
            loaded.model,
            loaded.data,
            np.zeros(loaded.model.nq + 1),
        )


# --- FT sensor offset augmentation tests ---


def _stacked_kwargs(loaded, sample_arm):
    """Build kwargs for compute_stacked_body_regressor with padded trajectories."""
    nq, nv = loaded.model.nq, loaded.model.nv
    padded = _pad_sample(sample_arm, nq, nv)
    return dict(
        model=loaded.model,
        data=loaded.data,
        q=padded.position,
        dq=padded.velocity,
        ddq=padded.acceleration,
        body_name=PAYLOAD_BODY_NAME,
        subsample_factor=10,
    )


def test_stacked_regressor_with_ft_offset_shape() -> None:
    loaded = _load_payload_scene()
    kwargs = _stacked_kwargs(loaded, _trajectory_arm(scale=0.2))
    regressor = compute_stacked_body_regressor(**kwargs, with_ft_offset=True)
    n_samples = regressor.shape[0] // 6
    assert regressor.shape == (n_samples * 6, 16)


def test_stacked_regressor_ft_offset_identity_block() -> None:
    loaded = _load_payload_scene()
    kwargs = _stacked_kwargs(loaded, _trajectory_arm(scale=0.2))
    regressor = compute_stacked_body_regressor(**kwargs, with_ft_offset=True)
    n_samples = regressor.shape[0] // 6
    expected_identity = np.tile(np.eye(6, dtype=np.float64), (n_samples, 1))
    np.testing.assert_array_equal(regressor[:, :6], expected_identity)


def test_stacked_regressor_ft_offset_preserves_physics() -> None:
    loaded = _load_payload_scene()
    kwargs = _stacked_kwargs(loaded, _trajectory_arm(scale=0.2))
    regressor_base = compute_stacked_body_regressor(**kwargs, with_ft_offset=False)
    regressor_ext = compute_stacked_body_regressor(**kwargs, with_ft_offset=True)
    np.testing.assert_array_equal(regressor_ext[:, 6:], regressor_base)


def test_condition_number_column_scale() -> None:
    loaded = _load_payload_scene()
    kwargs = _stacked_kwargs(loaded, _trajectory_arm(scale=0.2))
    regressor = compute_stacked_body_regressor(**kwargs, with_ft_offset=True)
    cond_raw = compute_condition_number(regressor, column_scale=False)
    cond_scaled = compute_condition_number(regressor, column_scale=True)
    assert np.isfinite(cond_scaled)
    assert np.isfinite(cond_raw)
    assert cond_scaled <= cond_raw


def test_ft_offset_default_backward_compatible() -> None:
    loaded = _load_payload_scene()
    kwargs = _stacked_kwargs(loaded, _trajectory_arm(scale=0.2))
    regressor = compute_stacked_body_regressor(**kwargs)
    assert regressor.shape[1] == 10
