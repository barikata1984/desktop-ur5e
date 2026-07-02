from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ur5e_sim.identification.constraints import JointLimits


@dataclass
class HorizonConfig:
    """Planning horizon parameters."""

    duration: float = 3.0  # T_h [s]
    num_segments: int = 4  # quintic spline segments
    fps: float = 100.0  # trajectory sampling rate
    subsample_factor: int = 10  # regressor subsampling for cost eval


@dataclass
class PlannerConfig:
    """Single-horizon optimization parameters."""

    n_restarts: int = 8  # multi-start count
    max_iter_per_start: int = 100  # SLSQP iterations per start
    ftol: float = 1e-6
    method: str = "SLSQP"
    seed: int = 42
    waypoint_perturbation: float = 0.3  # random init scale [rad]


@dataclass
class MPCConfig:
    """Top-level MPC loop configuration."""

    horizon: HorizonConfig = field(default_factory=HorizonConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)

    # MPC loop
    replan_period: float = 1.5  # T_replan [s]: execute before replanning
    max_mpc_steps: int = 10
    convergence_threshold: float = 0.01  # relative cond improvement

    # Robot
    num_joints: int = 6
    # Initial position. Callers should derive this from the model's "home"
    # keyframe (see build_ur5e_model()) rather than relying on a config default.
    q0: np.ndarray | None = None
    joint_limits: JointLimits = field(default_factory=JointLimits)

    n_inertial_params: int = 10  # inertial parameter count for RTLS

    # Execution
    use_pd_control: bool = False
    noise_std_wrench: float = 0.0

    def __post_init__(self) -> None:
        if self.replan_period > self.horizon.duration:
            raise ValueError(
                f"replan_period ({self.replan_period}) must be "
                f"<= horizon.duration ({self.horizon.duration})"
            )
