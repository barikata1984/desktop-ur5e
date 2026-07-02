from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from ur5e_sim.core import names
from ur5e_sim.identification.collision import CollisionConfig
from ur5e_sim.identification.constraints import JointLimits, build_trajectory_from_params
from ur5e_sim.identification.optimizer import OptimizationResult, OptimizerConfig
from ur5e_sim.identification.workspace import EeVelocityConfig, WorkspaceConstraintConfig
from ur5e_sim.trajectories.base import TrajectorySample


def _joint_limits_to_dict(joint_limits: JointLimits) -> dict:
    d = dataclasses.asdict(joint_limits)
    for key in ("q_min", "q_max", "dq_max", "ddq_max"):
        d[key] = np.asarray(d[key]).tolist()
    return d


def _joint_limits_from_dict(d: dict | None) -> JointLimits | None:
    if d is None:
        return None
    return JointLimits(
        q_min=np.asarray(d["q_min"], dtype=np.float64),
        q_max=np.asarray(d["q_max"], dtype=np.float64),
        dq_max=np.asarray(d["dq_max"], dtype=np.float64),
        ddq_max=np.asarray(d["ddq_max"], dtype=np.float64),
    )


def _workspace_config_to_dict(cfg: WorkspaceConstraintConfig) -> dict:
    d = dataclasses.asdict(cfg)
    for key in ("box_lower", "box_upper"):
        if d[key] is not None:
            d[key] = np.asarray(d[key]).tolist()
    return d


def _workspace_config_from_dict(d: dict | None) -> WorkspaceConstraintConfig | None:
    if d is None:
        return None
    box_lower = np.asarray(d["box_lower"], dtype=np.float64) if d["box_lower"] is not None else None
    box_upper = np.asarray(d["box_upper"], dtype=np.float64) if d["box_upper"] is not None else None
    return WorkspaceConstraintConfig(
        max_displacement=d["max_displacement"],
        box_lower=box_lower,
        box_upper=box_upper,
        safety_margin=d["safety_margin"],
    )


def _collision_config_from_dict(d: dict | None) -> CollisionConfig | None:
    if d is None:
        return None
    return CollisionConfig(**d)


def _ee_velocity_config_from_dict(d: dict | None) -> EeVelocityConfig | None:
    if d is None:
        return None
    return EeVelocityConfig(**d)


def save_optimization_result(
    result: OptimizationResult,
    path: str | Path,
) -> None:
    """Save an OptimizationResult to a JSON file."""
    cfg = result.config
    payload = {
        "x_opt": result.x_opt.tolist(),
        "condition_number": result.condition_number,
        "feasible": result.feasible,
        "constraint_margins": result.constraint_margins,
        "trajectory_stats": result.trajectory_stats,
        "restart_history": result.restart_history,
        "a_opt": result.a_opt.tolist(),
        "b_opt": result.b_opt.tolist(),
        "q0": result.q0.tolist(),
        "n_evaluations": result.n_evaluations,
        "wall_time": result.wall_time,
        "n_restarts": result.n_restarts,
        "best_start_index": result.best_start_index,
        "config": {
            "num_joints": cfg.num_joints,
            "num_harmonics": cfg.num_harmonics,
            "base_freq": cfg.base_freq,
            "duration": cfg.duration,
            "fps": cfg.fps,
            "q0": np.asarray(cfg.q0).tolist(),
            "subsample_factor": cfg.subsample_factor,
            "n_monte_carlo": cfg.n_monte_carlo,
            "max_iter_per_start": cfg.max_iter_per_start,
            "optimizer_method": cfg.optimizer_method,
            "ftol": cfg.ftol,
            "seed": cfg.seed,
            "body_name": cfg.body_name,
            "site_name": cfg.site_name,
            "payload_xml": cfg.payload_xml,
            "objective_type": cfg.objective_type,
            "joint_limits": (
                _joint_limits_to_dict(cfg.joint_limits) if cfg.joint_limits is not None else None
            ),
            "workspace_config": (
                _workspace_config_to_dict(cfg.workspace_config)
                if cfg.workspace_config is not None
                else None
            ),
            "payload_workspace_config": (
                _workspace_config_to_dict(cfg.payload_workspace_config)
                if cfg.payload_workspace_config is not None
                else None
            ),
            "collision_config": (
                dataclasses.asdict(cfg.collision_config)
                if cfg.collision_config is not None
                else None
            ),
            "ee_velocity_config": (
                dataclasses.asdict(cfg.ee_velocity_config)
                if cfg.ee_velocity_config is not None
                else None
            ),
            "enable_velocity_constraint": cfg.enable_velocity_constraint,
            "enable_acceleration_constraint": cfg.enable_acceleration_constraint,
            "use_fourier_bounds": cfg.use_fourier_bounds,
            "with_ft_offset": cfg.with_ft_offset,
            "ft_offset_column_scale": cfg.ft_offset_column_scale,
            "n_workers": cfg.n_workers,
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_optimization_result(path: str | Path) -> OptimizationResult:
    """Load an OptimizationResult from a JSON file."""
    with open(Path(path)) as f:
        payload = json.load(f)

    cfg_dict = payload["config"]
    config = OptimizerConfig(
        num_joints=cfg_dict["num_joints"],
        num_harmonics=cfg_dict["num_harmonics"],
        base_freq=cfg_dict["base_freq"],
        duration=cfg_dict["duration"],
        fps=cfg_dict["fps"],
        q0=np.array(cfg_dict["q0"], dtype=np.float64),
        subsample_factor=cfg_dict["subsample_factor"],
        n_monte_carlo=cfg_dict["n_monte_carlo"],
        max_iter_per_start=cfg_dict["max_iter_per_start"],
        optimizer_method=cfg_dict["optimizer_method"],
        ftol=cfg_dict["ftol"],
        seed=cfg_dict["seed"],
        body_name=cfg_dict["body_name"],
        site_name=cfg_dict["site_name"],
        payload_xml=cfg_dict.get("payload_xml", OptimizerConfig.payload_xml),
        objective_type=cfg_dict.get("objective_type", OptimizerConfig.objective_type),
        joint_limits=_joint_limits_from_dict(cfg_dict.get("joint_limits")),
        workspace_config=_workspace_config_from_dict(cfg_dict.get("workspace_config")),
        payload_workspace_config=_workspace_config_from_dict(
            cfg_dict.get("payload_workspace_config")
        ),
        collision_config=_collision_config_from_dict(cfg_dict.get("collision_config")),
        ee_velocity_config=_ee_velocity_config_from_dict(cfg_dict.get("ee_velocity_config")),
        enable_velocity_constraint=cfg_dict.get(
            "enable_velocity_constraint", OptimizerConfig.enable_velocity_constraint
        ),
        enable_acceleration_constraint=cfg_dict.get(
            "enable_acceleration_constraint", OptimizerConfig.enable_acceleration_constraint
        ),
        use_fourier_bounds=cfg_dict.get("use_fourier_bounds", OptimizerConfig.use_fourier_bounds),
        with_ft_offset=cfg_dict.get("with_ft_offset", OptimizerConfig.with_ft_offset),
        ft_offset_column_scale=cfg_dict.get(
            "ft_offset_column_scale", OptimizerConfig.ft_offset_column_scale
        ),
        n_workers=cfg_dict.get("n_workers", OptimizerConfig.n_workers),
    )

    x_opt = np.array(payload["x_opt"], dtype=np.float64)
    a_opt = np.array(payload["a_opt"], dtype=np.float64).reshape(
        config.num_joints, config.num_harmonics
    )
    b_opt = np.array(payload["b_opt"], dtype=np.float64).reshape(
        config.num_joints, config.num_harmonics
    )

    return OptimizationResult(
        x_opt=x_opt,
        condition_number=float(payload["condition_number"]),
        a_opt=a_opt,
        b_opt=b_opt,
        q0=np.array(payload["q0"], dtype=np.float64),
        config=config,
        n_evaluations=int(payload["n_evaluations"]),
        wall_time=float(payload["wall_time"]),
        n_restarts=int(payload["n_restarts"]),
        best_start_index=int(payload["best_start_index"]),
        restart_history=payload.get("restart_history", []),
        feasible=payload.get("feasible", False),
        constraint_margins=payload.get("constraint_margins", {}),
        trajectory_stats=payload.get("trajectory_stats", {}),
    )


def result_to_trajectory(
    result: OptimizationResult,
    fps: float | None = None,
) -> TrajectorySample:
    """Reconstruct a full trajectory from an OptimizationResult.

    Args:
        result: Optimization result containing Fourier coefficients.
        fps: Override sampling rate. If None, uses the optimization fps.
    """
    cfg = result.config
    output_fps = fps if fps is not None else cfg.fps
    return build_trajectory_from_params(
        result.x_opt,
        cfg.num_joints,
        cfg.num_harmonics,
        cfg.base_freq,
        cfg.duration,
        output_fps,
        result.q0,
    )


# UR5e joint names in URDF order
_UR5E_JOINT_NAMES: list[str] = list(names.ARM_ACTUATORS)


def save_trajectory_json(
    trajectory: TrajectorySample,
    path: str | Path,
    *,
    condition_number: float | None = None,
    source: str | None = None,
) -> None:
    """Save a sampled trajectory as a self-contained JSON for real robot playback.

    The output JSON contains all joint positions, velocities, and accelerations
    at each timestep, so it can be loaded by a ROS node or similar system
    without requiring the Fourier trajectory generation code.

    Args:
        trajectory: Sampled trajectory with time, position, velocity, acceleration.
        path: Output file path.
        condition_number: Condition number from optimization (for metadata).
        source: Source file name (for metadata).
    """
    n_steps = len(trajectory.time)
    n_joints = trajectory.position.shape[1]
    duration = float(trajectory.time[-1] - trajectory.time[0])
    dt = float(trajectory.time[1] - trajectory.time[0]) if n_steps > 1 else 0.0
    fps = 1.0 / dt if dt > 0.0 else 0.0

    metadata: dict = {
        "description": "Sampled excitation trajectory for UR5e",
        "num_joints": n_joints,
        "num_steps": n_steps,
        "duration": duration,
        "fps": fps,
        "dt": dt,
        "joint_names": _UR5E_JOINT_NAMES[:n_joints],
    }
    if condition_number is not None:
        metadata["condition_number"] = condition_number
    if source is not None:
        metadata["source"] = source

    waypoints: list[dict] = []
    for i in range(n_steps):
        waypoints.append(
            {
                "t": round(float(trajectory.time[i]), 6),
                "q": [round(float(v), 8) for v in trajectory.position[i]],
                "dq": [round(float(v), 8) for v in trajectory.velocity[i]],
                "ddq": [round(float(v), 8) for v in trajectory.acceleration[i]],
            }
        )

    payload = {"metadata": metadata, "trajectory": waypoints}

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
