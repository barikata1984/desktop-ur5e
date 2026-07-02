"""Trajectory playback and measurement on MuJoCo."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import mujoco
import numpy as np

from ur5e_sim.core import names
from ur5e_sim.core.layout import DofLayout
from ur5e_sim.core.sensors import FTSensor
from ur5e_sim.core.types import get_site_frame
from ur5e_sim.trajectories.base import TrajectorySample

from .data_buffer import DataBuffer, SensorSample
from .regressor import (
    body_inertial_parameters_from_model,
    compute_wrench_from_parameters,
    sample_body_regressor,
)
from .sampling import set_model_state


@dataclass
class PlaybackConfig:
    """Configuration for trajectory playback."""

    dt: float = 0.002
    use_pd_control: bool = False
    kp: np.ndarray | None = None
    kd: np.ndarray | None = None
    noise_std_q: float = 0.0
    noise_std_dq: float = 0.0
    noise_std_wrench: float = 0.0
    # Seconds to hold the initial target before recording so the arm settles
    # into its gravity-loaded equilibrium (PD-servo mode only).
    settle_time: float = 1.0
    # "auto": use the FT sensor when it resolves, else fall back to the analytic
    #   rigid-body regressor wrench (warns on fallback instead of failing silently).
    # "sensor": require the FT sensor to resolve; raise ValueError if it does not.
    # "analytic": always use the rigid-body regressor wrench, skipping FT sensor
    #   resolution entirely.
    wrench_source: Literal["auto", "sensor", "analytic"] = "auto"


class TrajectoryPlayback:
    """Execute a trajectory on a MuJoCo model and collect data."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        config: PlaybackConfig,
    ) -> None:
        self._model = model
        self._data = data
        self._config = config
        self._layout = DofLayout.from_model(model)

    def execute(
        self,
        trajectory: TrajectorySample,
        rng: np.random.Generator | None = None,
    ) -> DataBuffer:
        """Play back a trajectory and collect sensor data.

        Args:
            trajectory: Desired trajectory with time, position,
                velocity, and acceleration arrays.
            rng: Random generator for measurement noise.

        Returns:
            DataBuffer with one SensorSample per timestep.
        """
        cfg = self._config
        model = self._model
        data = self._data
        n_joints = trajectory.position.shape[1]

        ft_sensor = None
        use_ft_sensor = False
        if cfg.wrench_source in ("auto", "sensor"):
            try:
                ft_sensor = FTSensor(model, names.FT_FORCE_SENSOR, names.FT_TORQUE_SENSOR)
                use_ft_sensor = True
            except ValueError:
                if cfg.wrench_source == "sensor":
                    raise
                warnings.warn(
                    f"FT sensors {names.FT_FORCE_SENSOR!r}/{names.FT_TORQUE_SENSOR!r} not "
                    "found in model; falling back to the analytic rigid-body regressor wrench.",
                    stacklevel=2,
                )

        # The analytic-fallback wrench needs the payload's rigid-body inertia.
        # The FT-sensor path does not, so only resolve it when needed -- this lets
        # the playback run against models (e.g. an articulated gripper) where
        # names.PAYLOAD_BODY does not name a single rigid body.
        params = None
        if not use_ft_sensor:
            params = body_inertial_parameters_from_model(model, names.PAYLOAD_BODY)

        # Pad trajectory to model DOFs when it covers only a subset (e.g. 6 arm
        # joints on a 14-DOF model with gripper). Extra DOFs are held at zero.
        layout = self._layout
        nq, nv = model.nq, model.nv
        if n_joints < nq:
            trajectory = TrajectorySample(
                time=trajectory.time,
                position=layout.to_full_qpos(trajectory.position),
                velocity=layout.to_full_qvel(trajectory.velocity),
                acceleration=layout.to_full_qvel(trajectory.acceleration),
            )

        buffer = DataBuffer()
        n_steps = len(trajectory.time)

        # Number of physics substeps per trajectory step so that one trajectory
        # interval (its dt) is fully integrated by the simulator. The MJCF
        # actuators are position-velocity servos, so ctrl receives target angles
        # and the built-in servo provides the tracking torque.
        if n_steps > 1:
            traj_dt = float(trajectory.time[1] - trajectory.time[0])
            n_substeps = max(1, round(traj_dt / model.opt.timestep))
        else:
            n_substeps = 1

        # Settle the arm into its gravity-loaded equilibrium before recording:
        # hold the initial target on the servos and integrate so the transient
        # from the static (qacc=0) reset decays. Without this, the first frames
        # carry a large acceleration spike unrelated to the desired trajectory.
        if cfg.use_pd_control and cfg.settle_time > 0.0:
            q_start = trajectory.position[0]
            layout.set_arm_ctrl(data, layout.arm(q_start))
            n_settle = max(1, round(cfg.settle_time / model.opt.timestep))
            for _ in range(n_settle):
                mujoco.mj_step(model, data)

        for i in range(n_steps):
            t = float(trajectory.time[i])
            q_des = trajectory.position[i]
            dq_des = trajectory.velocity[i]
            ddq_des = trajectory.acceleration[i]

            if cfg.use_pd_control:
                # Servo-tracking mode: feed the target angle to the built-in
                # position-velocity servos and integrate the full trajectory dt.
                layout.set_arm_ctrl(data, layout.arm(q_des))
                for _ in range(n_substeps):
                    mujoco.mj_step(model, data)

                q_meas = np.array(data.qpos[:nq], dtype=np.float64)
                dq_meas = np.array(data.qvel[:nv], dtype=np.float64)
                ddq_meas = np.array(data.qacc[:nv], dtype=np.float64)
            else:
                # Open-loop mode: set state directly. The FT sensor reads the
                # interaction force (cfrc_int), computed at the acceleration stage.
                # mj_forward would overwrite qacc with the forward-dynamics solution
                # before computing cfrc_int, so the sensor would measure the force for
                # an arbitrary qacc instead of the commanded ddq_des. mj_inverse keeps
                # the input qacc and still populates cfrc_int (via mj_rnePostConstraint),
                # so the FT sensor reports the wrench corresponding to ddq_des.
                set_model_state(model, data, q_des, dq_des, ddq_des)
                if use_ft_sensor:
                    mujoco.mj_inverse(model, data)
                q_meas = q_des.copy()
                dq_meas = dq_des.copy()
                ddq_meas = ddq_des.copy()

            # Get EE pose
            frame = get_site_frame(model, data, names.EE_SITE)
            if frame is not None:
                ee_pos = np.array(frame.position, dtype=np.float64)
                ee_rot = np.array(frame.rotation, dtype=np.float64)
            else:
                ee_pos = np.zeros(3, dtype=np.float64)
                ee_rot = np.eye(3, dtype=np.float64)

            if use_ft_sensor:
                wrench = ft_sensor.read(model, data)
            else:
                # Fallback: analytic rigid-body regressor wrench ([torque; force]),
                # evaluated about the FT sensor site (names.FT_SITE), not the
                # EE-pose site (names.EE_SITE).
                reg_sample = sample_body_regressor(model, data, names.PAYLOAD_BODY, names.FT_SITE)
                wrench = compute_wrench_from_parameters(reg_sample.regressor, params)

            # Add measurement noise
            if rng is not None:
                if cfg.noise_std_q > 0:
                    q_meas = q_meas + rng.normal(0, cfg.noise_std_q, n_joints)
                if cfg.noise_std_dq > 0:
                    dq_meas = dq_meas + rng.normal(0, cfg.noise_std_dq, n_joints)
                if cfg.noise_std_wrench > 0:
                    wrench = wrench + rng.normal(0, cfg.noise_std_wrench, 6)

            sample = SensorSample(
                timestamp=t,
                q=q_meas,
                dq=dq_meas,
                ddq=ddq_meas,
                ee_position=ee_pos,
                ee_rotation=ee_rot,
                wrench=wrench,
            )
            buffer.append(sample)

        return buffer

    def compute_tracking_error(
        self,
        trajectory: TrajectorySample,
        buffer: DataBuffer,
    ) -> dict[str, float]:
        """Compare desired vs actual positions/velocities.

        Returns dict with max_position_error,
        max_velocity_error, rms_position_error,
        rms_velocity_error.
        """
        arrays = buffer.to_arrays()
        q_actual = arrays["q"]
        dq_actual = arrays["dq"]

        n = min(len(trajectory.time), len(buffer))
        q_des = trajectory.position[:n]
        dq_des = trajectory.velocity[:n]
        q_act = q_actual[:n]
        dq_act = dq_actual[:n]

        pos_err = np.abs(q_des - q_act)
        vel_err = np.abs(dq_des - dq_act)

        return {
            "max_position_error": float(np.max(pos_err)),
            "max_velocity_error": float(np.max(vel_err)),
            "rms_position_error": float(np.sqrt(np.mean(pos_err**2))),
            "rms_velocity_error": float(np.sqrt(np.mean(vel_err**2))),
        }
