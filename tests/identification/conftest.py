"""Shared fixtures and helpers for identification tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from ur5e_sim.core.model_builder import build_ur5e_model

# Identification scene: the box payload attached to the end effector, assembled
# programmatically via build_ur5e_model() (the canonical construction path).
PAYLOAD_XML = "scenes/objects/payload_box.xml"


def load_identification_scene() -> SimpleNamespace:
    """Build the identification model (box payload) reset to its home keyframe.

    Returns an object with ``.model`` and ``.data`` attributes. build_ur5e_model
    already resets data to the "home" keyframe and calls mj_forward.
    """
    model, data = build_ur5e_model(payload_xml=PAYLOAD_XML)
    return SimpleNamespace(model=model, data=data)


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
