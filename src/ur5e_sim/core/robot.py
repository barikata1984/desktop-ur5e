from __future__ import annotations

import numpy as np

from ur5e_sim.core.env import SimEnv
from ur5e_sim.core.types import FramePose, get_site_frame


class UR5eRobot:
    """Interface to the UR5e arm within a SimEnv."""

    def __init__(self, env: SimEnv, n_arm_joints: int = 6):
        self.env = env
        self.n = n_arm_joints

    def joint_positions(self) -> np.ndarray:
        return self.env.data.qpos[: self.n].copy()

    def joint_velocities(self) -> np.ndarray:
        return self.env.data.qvel[: self.n].copy()

    def joint_accelerations(self) -> np.ndarray:
        return self.env.data.qacc[: self.n].copy()

    def set_ctrl(self, ctrl: np.ndarray) -> None:
        self.env.data.ctrl[: self.n] = ctrl

    def ee_pose(self, site_name: str = "attachment_site") -> FramePose | None:
        return get_site_frame(self.env.model, self.env.data, site_name)
