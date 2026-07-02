from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ur5e_sim.core import names
from ur5e_sim.identification.constraints import JointLimits
from ur5e_sim.identification.optimizer import UR5E_HOME_QPOS


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
    q0: np.ndarray | None = None  # initial position (None -> UR5E_HOME_QPOS)
    joint_limits: JointLimits = field(default_factory=JointLimits)

    # MuJoCo identifiers, canonical to models from build_ur5e_model().
    body_name: str = names.PAYLOAD_BODY
    site_name: str = names.EE_SITE  # EE pose frame (for playback)
    ft_site_name: str = names.FT_SITE  # FT sensor site (for regressor sampling)
    n_inertial_params: int = 10  # inertial parameter count for RTLS

    # Execution
    use_pd_control: bool = False
    noise_std_wrench: float = 0.0

    def __post_init__(self) -> None:
        if self.q0 is None:
            self.q0 = UR5E_HOME_QPOS.copy()
        if self.replan_period > self.horizon.duration:
            raise ValueError(
                f"replan_period ({self.replan_period}) must be "
                f"<= horizon.duration ({self.horizon.duration})"
            )
