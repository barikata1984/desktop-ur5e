import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from ur5e_sim.identification import (  # noqa: E402
    DataBuffer,
    PlaybackConfig,
    SensorSample,
    TrajectoryPlayback,
)
from ur5e_sim.trajectories.base import TrajectorySample  # noqa: E402

from .conftest import (  # noqa: E402
    arm_to_full_qpos,
    arm_to_full_qvel,
    load_identification_scene,
)

Q0_ARM = np.array([np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0])


def _load_payload_scene():
    return load_identification_scene()


def _make_short_trajectory(n_steps: int = 5, nq: int = 14, nv: int = 14) -> TrajectorySample:
    """Create a short trajectory around Q0 with small perturbations, padded to full model dims."""
    rng = np.random.default_rng(123)
    dt = 0.02
    time = np.linspace(0.0, dt * (n_steps - 1), n_steps)
    # Generate arm-only perturbations and pad
    arm_pos = np.tile(Q0_ARM, (n_steps, 1)) + rng.uniform(-0.01, 0.01, (n_steps, 6))
    arm_vel = rng.uniform(-0.1, 0.1, (n_steps, 6))
    arm_acc = rng.uniform(-1.0, 1.0, (n_steps, 6))
    position = arm_to_full_qpos(arm_pos, nq)
    velocity = arm_to_full_qvel(arm_vel, nv)
    acceleration = arm_to_full_qvel(arm_acc, nv)
    return TrajectorySample(
        time=time,
        position=position,
        velocity=velocity,
        acceleration=acceleration,
    )


def _make_sensor_sample(t: float = 0.0) -> SensorSample:
    return SensorSample(
        timestamp=t,
        q=Q0_ARM.copy(),
        dq=np.zeros(6),
        ddq=np.zeros(6),
        ee_position=np.array([0.1, 0.2, 0.3]),
        ee_rotation=np.eye(3),
        wrench=np.zeros(6),
    )


def test_sensor_sample_creation() -> None:
    sample = _make_sensor_sample(1.5)
    assert sample.timestamp == 1.5
    assert sample.q.shape == (6,)
    assert sample.dq.shape == (6,)
    assert sample.ddq.shape == (6,)
    assert sample.ee_position.shape == (3,)
    assert sample.ee_rotation.shape == (3, 3)
    assert sample.wrench is not None
    assert sample.wrench.shape == (6,)


def test_sensor_sample_with_none_wrench() -> None:
    sample = SensorSample(
        timestamp=0.0,
        q=np.zeros(6),
        dq=np.zeros(6),
        ddq=np.zeros(6),
        ee_position=np.zeros(3),
        ee_rotation=np.eye(3),
        wrench=None,
    )
    assert sample.wrench is None


def test_data_buffer_append_and_len() -> None:
    buf = DataBuffer()
    assert len(buf) == 0
    buf.append(_make_sensor_sample(0.0))
    assert len(buf) == 1
    buf.append(_make_sensor_sample(0.1))
    assert len(buf) == 2


def test_data_buffer_to_arrays_shapes() -> None:
    buf = DataBuffer()
    n = 7
    for i in range(n):
        buf.append(_make_sensor_sample(float(i) * 0.01))

    arrays = buf.to_arrays()
    assert arrays["timestamp"].shape == (n,)
    assert arrays["q"].shape == (n, 6)
    assert arrays["dq"].shape == (n, 6)
    assert arrays["ddq"].shape == (n, 6)
    assert arrays["ee_position"].shape == (n, 3)
    assert arrays["ee_rotation"].shape == (n, 3, 3)
    assert arrays["wrench"].shape == (n, 6)


def test_data_buffer_empty_to_arrays() -> None:
    buf = DataBuffer()
    arrays = buf.to_arrays()
    assert arrays["timestamp"].shape == (0,)
    assert arrays["q"].shape == (0, 6)


def test_playback_config_defaults() -> None:
    cfg = PlaybackConfig()
    assert cfg.dt == 0.002
    assert cfg.use_pd_control is False
    assert cfg.kp is None
    assert cfg.kd is None
    assert cfg.noise_std_q == 0.0
    assert cfg.noise_std_dq == 0.0
    assert cfg.noise_std_wrench == 0.0


def test_trajectory_playback_open_loop() -> None:
    loaded = _load_payload_scene()
    nq, nv = loaded.model.nq, loaded.model.nv
    cfg = PlaybackConfig()
    playback = TrajectoryPlayback(loaded.model, loaded.data, cfg)

    traj = _make_short_trajectory(5, nq, nv)
    buffer = playback.execute(traj)

    assert len(buffer) == 5
    arrays = buffer.to_arrays()
    # The playback returns n_joints columns (= traj.position.shape[1] = nq)
    assert arrays["q"].shape == (5, nq)
    assert arrays["wrench"].shape == (5, 6)
    assert arrays["ee_position"].shape == (5, 3)
    assert arrays["ee_rotation"].shape == (5, 3, 3)

    # In open-loop mode, measured q should match desired q
    np.testing.assert_allclose(arrays["q"], traj.position, atol=1e-12)
    np.testing.assert_allclose(arrays["dq"], traj.velocity, atol=1e-12)


def test_trajectory_playback_with_noise() -> None:
    loaded = _load_payload_scene()
    nq, nv = loaded.model.nq, loaded.model.nv
    cfg = PlaybackConfig(
        noise_std_q=0.01,
        noise_std_dq=0.01,
        noise_std_wrench=0.1,
    )
    playback = TrajectoryPlayback(loaded.model, loaded.data, cfg)

    traj = _make_short_trajectory(5, nq, nv)
    rng = np.random.default_rng(99)
    buffer = playback.execute(traj, rng=rng)

    arrays = buffer.to_arrays()
    # With noise, measurements should differ from desired
    q_diff = np.max(np.abs(arrays["q"] - traj.position))
    assert q_diff > 1e-6, "noise should perturb measurements"


def test_compute_tracking_error_keys() -> None:
    loaded = _load_payload_scene()
    nq, nv = loaded.model.nq, loaded.model.nv
    cfg = PlaybackConfig()
    playback = TrajectoryPlayback(loaded.model, loaded.data, cfg)

    traj = _make_short_trajectory(5, nq, nv)
    buffer = playback.execute(traj)
    errors = playback.compute_tracking_error(traj, buffer)

    expected_keys = {
        "max_position_error",
        "max_velocity_error",
        "rms_position_error",
        "rms_velocity_error",
    }
    assert set(errors.keys()) == expected_keys
    # Open-loop should have zero tracking error
    assert errors["max_position_error"] < 1e-12
    assert errors["rms_position_error"] < 1e-12


def test_compute_tracking_error_with_noise() -> None:
    loaded = _load_payload_scene()
    nq, nv = loaded.model.nq, loaded.model.nv
    cfg = PlaybackConfig(
        noise_std_q=0.05,
    )
    playback = TrajectoryPlayback(loaded.model, loaded.data, cfg)

    traj = _make_short_trajectory(10, nq, nv)
    rng = np.random.default_rng(42)
    buffer = playback.execute(traj, rng=rng)
    errors = playback.compute_tracking_error(traj, buffer)

    # Noisy measurements should show tracking error
    assert errors["max_position_error"] > 1e-4
    assert errors["rms_position_error"] > 1e-4
