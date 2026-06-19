"""Shared fixtures and helpers for identification tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE_PATH = str(REPO_ROOT / "scenes" / "tasks" / "identification.xml")

# The identification scene has 14 joints (6 arm + 8 gripper).
# Most identification tests only care about the 6 arm joints.
NUM_ARM_JOINTS = 6
NUM_GRIPPER_JOINTS = 8
NUM_TOTAL_JOINTS = NUM_ARM_JOINTS + NUM_GRIPPER_JOINTS

# Home arm joint positions (gripper joints are zero)
Q0_ARM = np.array([np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0])


def arm_to_full_qpos(q_arm: np.ndarray, nq: int = NUM_TOTAL_JOINTS) -> np.ndarray:
    """Pad a 6-element arm qpos to full model qpos (zeros for gripper joints)."""
    if q_arm.ndim == 1:
        full = np.zeros(nq)
        full[:NUM_ARM_JOINTS] = q_arm
        return full
    # 2D: (N, 6) -> (N, nq)
    n = q_arm.shape[0]
    full = np.zeros((n, nq))
    full[:, :NUM_ARM_JOINTS] = q_arm
    return full


def arm_to_full_qvel(dq_arm: np.ndarray, nv: int = NUM_TOTAL_JOINTS) -> np.ndarray:
    """Pad a 6-element arm qvel to full model qvel (zeros for gripper joints)."""
    if dq_arm.ndim == 1:
        full = np.zeros(nv)
        full[:NUM_ARM_JOINTS] = dq_arm
        return full
    n = dq_arm.shape[0]
    full = np.zeros((n, nv))
    full[:, :NUM_ARM_JOINTS] = dq_arm
    return full
