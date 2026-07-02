"""CLI: optimize an excitation trajectory for inertial parameter identification.

Uses tyro for CLI, with YAML defaults loaded from configs/identification_default.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import tyro
import yaml

from ur5e_sim.core.env import get_workspace_bounds
from ur5e_sim.core.layout import DofLayout
from ur5e_sim.core.model_builder import build_ur5e_model
from ur5e_sim.identification.collision import CollisionConfig
from ur5e_sim.identification.io import (
    result_to_trajectory,
    save_optimization_result,
    save_trajectory_json,
)
from ur5e_sim.identification.optimizer import (
    EarlyStopConfig,
    ExcitationOptimizer,
    OptimizerConfig,
    WandbConfig,
)
from ur5e_sim.identification.workspace import EeVelocityConfig, WorkspaceConstraintConfig

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "identification_default.yaml"


@dataclass
class OptimizeExcitationConfig:
    """Configuration for excitation trajectory optimization."""

    model: str = ""  # Unused: model is now built by build_ur5e_model()
    num_harmonics: int = 5
    base_freq: float = 0.2
    duration: float = 5.0
    fps: float = 100.0
    subsample_factor: int = 5
    n_monte_carlo: int = 20
    max_iter: int = 200
    seed: int = 42
    max_displacement: float = 0.0
    enable_collision: bool = True
    enable_payload_workspace: bool = True
    ee_max_linear_velocity: float = 0.0
    dq_max: float = 1.5
    ddq_max: float = 0.0
    use_fourier_bounds: bool = False
    with_ft_offset: bool = False
    ft_offset_column_scale: bool = True
    n_workers: int = 1
    objective: str = "condition_number"
    output: str = "results/excitation_result.json"
    trajectory_output: str = "results/excitation_trajectory.json"
    trajectory_fps: float = 0.0
    wandb: bool = True
    wandb_project: str = "ur5e-excitation"
    wandb_run_name: str | None = None
    early_stop: bool = False
    early_stop_patience: int = 5
    early_stop_target_cond: float = 0.0


def _load_yaml_defaults() -> dict:
    if _DEFAULT_CONFIG.exists():
        with open(_DEFAULT_CONFIG) as f:
            data = yaml.safe_load(f) or {}
        return data
    return {}


def _build_config() -> OptimizeExcitationConfig:
    yaml_data = _load_yaml_defaults()
    kwargs: dict = {}

    # Map YAML sections to flat dataclass fields
    if "model" in yaml_data and "path" in yaml_data["model"]:
        kwargs["model"] = yaml_data["model"]["path"]

    traj = yaml_data.get("trajectory", {})
    for key in ("num_harmonics", "base_freq", "duration", "fps"):
        if key in traj:
            kwargs[key] = traj[key]

    opt = yaml_data.get("optimizer", {})
    mapping = {
        "subsample_factor": "subsample_factor",
        "n_monte_carlo": "n_monte_carlo",
        "max_iter_per_start": "max_iter",
        "seed": "seed",
        "objective": "objective",
    }
    for yaml_key, config_key in mapping.items():
        if yaml_key in opt:
            kwargs[config_key] = opt[yaml_key]

    jl = yaml_data.get("joint_limits", {})
    if "dq_max" in jl:
        kwargs["dq_max"] = jl["dq_max"]
    if "ddq_max" in jl:
        kwargs["ddq_max"] = jl["ddq_max"]

    ee = yaml_data.get("ee_velocity", {})
    if "max_linear_velocity" in ee:
        kwargs["ee_max_linear_velocity"] = ee["max_linear_velocity"]

    out = yaml_data.get("output", {})
    if "optimization_result" in out:
        kwargs["output"] = out["optimization_result"]

    default = OptimizeExcitationConfig(**kwargs)
    return tyro.cli(OptimizeExcitationConfig, default=default)


_PAYLOAD_XML = "scenes/objects/payload_flat.xml"


def main() -> None:
    config = _build_config()

    model, data = build_ur5e_model(payload_xml=_PAYLOAD_XML)
    layout = DofLayout.from_model(model)
    q0 = layout.arm(data.qpos).copy()

    workspace_config: WorkspaceConstraintConfig | None = None
    if config.max_displacement > 0:
        workspace_config = WorkspaceConstraintConfig(max_displacement=config.max_displacement)

    collision_config: CollisionConfig | None = None
    if config.enable_collision:
        collision_config = CollisionConfig()

    payload_workspace_config: WorkspaceConstraintConfig | None = None
    if config.enable_payload_workspace:
        try:
            box_lower, box_upper = get_workspace_bounds(model, data)
            print(f"  payload workspace bounds: {box_lower} .. {box_upper}")
            payload_workspace_config = WorkspaceConstraintConfig(
                box_lower=box_lower, box_upper=box_upper
            )
        except RuntimeError:
            print(
                "WARNING: workspace region geom not found — workspace constraint disabled",
                flush=True,
            )

    ee_velocity_config: EeVelocityConfig | None = None
    if config.ee_max_linear_velocity > 0:
        ee_velocity_config = EeVelocityConfig(max_linear_velocity=config.ee_max_linear_velocity)

    joint_limits: JointLimits | None = None
    has_vel = config.dq_max > 0
    has_acc = config.ddq_max > 0
    if has_vel or has_acc:
        from ur5e_sim.identification.constraints import JointLimits

        kwargs_jl: dict = {}
        if has_vel:
            kwargs_jl["dq_max"] = np.full(6, config.dq_max)
        if has_acc:
            kwargs_jl["ddq_max"] = np.full(6, config.ddq_max)
        joint_limits = JointLimits(**kwargs_jl)

    use_fourier_bounds = config.use_fourier_bounds and (has_vel or has_acc)
    enable_vel_constraint = has_vel and not use_fourier_bounds

    opt_config = OptimizerConfig(
        num_joints=6,
        num_harmonics=config.num_harmonics,
        base_freq=config.base_freq,
        duration=config.duration,
        fps=config.fps,
        q0=q0,
        subsample_factor=config.subsample_factor,
        n_monte_carlo=config.n_monte_carlo,
        max_iter_per_start=config.max_iter,
        objective_type=config.objective,
        seed=config.seed,
        joint_limits=joint_limits,
        workspace_config=workspace_config,
        collision_config=collision_config,
        payload_workspace_config=payload_workspace_config,
        ee_velocity_config=ee_velocity_config,
        enable_velocity_constraint=enable_vel_constraint,
        enable_acceleration_constraint=has_acc,
        use_fourier_bounds=use_fourier_bounds,
        with_ft_offset=config.with_ft_offset,
        ft_offset_column_scale=config.ft_offset_column_scale,
        n_workers=config.n_workers,
        payload_xml=_PAYLOAD_XML,
    )

    optimizer = ExcitationOptimizer(config=opt_config, model=model, data=data)

    wandb_cfg = WandbConfig(
        enabled=config.wandb,
        project=config.wandb_project,
        run_name=config.wandb_run_name,
    )
    early_stop_cfg = EarlyStopConfig(
        enabled=config.early_stop,
        patience=config.early_stop_patience,
        target_cond=config.early_stop_target_cond,
    )

    print("Starting excitation trajectory optimization...", flush=True)
    print(f"  objective={config.objective}", flush=True)
    print(f"  harmonics={config.num_harmonics}, duration={config.duration}s", flush=True)
    print(f"  monte-carlo restarts={config.n_monte_carlo}", flush=True)
    print(f"  max_iter_per_start={config.max_iter}", flush=True)
    if config.n_workers > 1:
        print(f"  parallel workers={config.n_workers}", flush=True)
    if config.wandb:
        print(f"  wandb: project={config.wandb_project}", flush=True)
    if config.ee_max_linear_velocity > 0:
        print(f"  EE velocity limit: {config.ee_max_linear_velocity} m/s", flush=True)
    if has_vel:
        print(f"  joint velocity limit: {config.dq_max} rad/s (all joints)", flush=True)
    if has_acc:
        print(f"  joint acceleration limit: {config.ddq_max} rad/s^2 (all joints)", flush=True)
    if use_fourier_bounds:
        print(
            "  Fourier coefficient bounds: ENABLED (velocity constraint via box bounds)", flush=True
        )
    if config.with_ft_offset:
        scale_str = "column-scaled" if config.ft_offset_column_scale else "unscaled"
        print(f"  FT sensor offset estimation: ENABLED (16 params, {scale_str})", flush=True)
    if config.early_stop:
        msg = f"  early stopping: patience={config.early_stop_patience}"
        if config.early_stop_target_cond > 0:
            msg += f", target_cond={config.early_stop_target_cond}"
        print(msg, flush=True)
    result = optimizer.optimize(wandb_config=wandb_cfg, early_stop_config=early_stop_cfg)

    output_path = Path(config.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_optimization_result(result, output_path)

    print("\nOptimization complete:")
    print(f"  Condition number: {result.condition_number:.4f}")
    print(f"  Feasible: {result.feasible}")
    print(f"  Wall time: {result.wall_time:.1f}s")
    print(f"  Total evaluations: {result.n_evaluations}")
    print(f"  Best start index: {result.best_start_index}")
    if result.constraint_margins:
        violated = {k: v for k, v in result.constraint_margins.items() if v < 0}
        satisfied = {k: v for k, v in result.constraint_margins.items() if v >= 0}
        print("  Constraint margins:")
        for name, margin in satisfied.items():
            print(f"    {name}: {margin:.6f}  [OK]")
        if violated:
            print(f"  VIOLATED constraints ({len(violated)}/{len(result.constraint_margins)}):")
            for name, margin in violated.items():
                print(f"    {name}: {margin:.6f}  (violation = {-margin:.6f})")
    if result.trajectory_stats:
        print("  Trajectory stats:")
        for key, val in result.trajectory_stats.items():
            if isinstance(val, list):
                formatted = ", ".join(f"{v:.4f}" for v in val)
                print(f"    {key}: [{formatted}]")
            else:
                print(f"    {key}: {val:.4f}")
    print(f"  Output: {output_path}")

    traj_fps = config.trajectory_fps if config.trajectory_fps > 0 else None
    trajectory = result_to_trajectory(result, fps=traj_fps)
    traj_path = Path(config.trajectory_output)
    save_trajectory_json(
        trajectory,
        traj_path,
        condition_number=result.condition_number,
        source=output_path.name,
    )
    effective_fps = traj_fps if traj_fps else config.fps
    print(f"  Trajectory JSON: {traj_path} ({effective_fps:.0f} Hz, {len(trajectory.time)} steps)")


if __name__ == "__main__":
    main()
