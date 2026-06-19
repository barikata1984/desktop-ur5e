"""Slider-specific kinematics helpers for the pushing task.

IK functions (damped_pinv, get_jacobian, get_jacobian6, orientation_error) live
in ``ur5e_sim.core.ik`` and should be imported from there. This module contains
only the slider-specific coordinate transforms.
"""

from __future__ import annotations

import mujoco
import numpy as np


def slider_pose_from_data(d: mujoco.MjData, slider_body_id: int) -> tuple[np.ndarray, float]:
    """World position and yaw angle (theta) of the slider body."""
    pos = d.xpos[slider_body_id].copy()
    quat = d.xquat[slider_body_id].copy()
    w, qx, qy, qz = quat
    theta = np.arctan2(2 * (w * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    return pos, theta


def pusher_in_slider_body(
    tip_world: np.ndarray, slider_pos: np.ndarray, theta: float
) -> np.ndarray:
    """Pusher tip position expressed in the slider body frame (px, py)."""
    dx = tip_world[0] - slider_pos[0]
    dy = tip_world[1] - slider_pos[1]
    ct, st = np.cos(theta), np.sin(theta)
    px = ct * dx + st * dy
    py = -st * dx + ct * dy
    return np.array([px, py])
