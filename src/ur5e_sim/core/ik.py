from __future__ import annotations

import mujoco
import numpy as np

# Desired tool0 (attachment_site) orientation: +x -> world +x, +z -> world -z
# (straight down), so the closed gripper is a vertical pusher with its finger
# axis perpendicular to the push direction. Held by the IK layer; the MPC is
# unchanged and purely 2D.
R_TOOL0_DES = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
ORI_GAIN = 2.0


def damped_pinv(J: np.ndarray, damping: float) -> np.ndarray:
    """Damped (Levenberg) pseudo-inverse of a Jacobian."""
    JJT = J @ J.T
    return J.T @ np.linalg.inv(JJT + damping**2 * np.eye(JJT.shape[0]))


def get_jacobian(m: mujoco.MjModel, d: mujoco.MjData, site_id: int) -> np.ndarray:
    """3xN_arm translational Jacobian of a site (arm joint columns only)."""
    jacp = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jacp, None, site_id)
    return jacp[:, :6]


def orientation_error(d: mujoco.MjData, site_id: int) -> np.ndarray:
    """World-frame axis-angle rotation driving the site toward R_TOOL0_DES."""
    quat = np.zeros(4)
    r_err = R_TOOL0_DES @ d.site_xmat[site_id].reshape(3, 3).T
    mujoco.mju_mat2Quat(quat, r_err.flatten())
    n = np.linalg.norm(quat[1:])
    return quat[1:] / n * 2 * np.arctan2(n, quat[0]) if n > 1e-9 else np.zeros(3)


def get_jacobian6(m: mujoco.MjModel, d: mujoco.MjData, pos_site: int, ori_site: int) -> np.ndarray:
    """Stacked 6x6 arm Jacobian: translation of pos_site + rotation of ori_site."""
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jacp, None, pos_site)
    mujoco.mj_jacSite(m, d, None, jacr, ori_site)
    return np.vstack([jacp[:, :6], jacr[:, :6]])
