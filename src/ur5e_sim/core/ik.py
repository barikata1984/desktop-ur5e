from __future__ import annotations

import mujoco
import numpy as np

ORI_GAIN = 2.0
GRIPPER_CLOSED_CTRL = 255


def damped_pinv(J: np.ndarray, damping: float) -> np.ndarray:
    """Damped (Levenberg) pseudo-inverse of a Jacobian."""
    JJT = J @ J.T
    return J.T @ np.linalg.inv(JJT + damping**2 * np.eye(JJT.shape[0]))


def get_jacobian(m: mujoco.MjModel, d: mujoco.MjData, site_id: int) -> np.ndarray:
    """3xN_arm translational Jacobian of a site (arm joint columns only)."""
    jacp = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jacp, None, site_id)
    return jacp[:, :6]


def orientation_error(
    d: mujoco.MjData, site_id: int, R_des: np.ndarray | None = None
) -> np.ndarray:
    """World-frame axis-angle rotation driving the site toward R_des."""
    if R_des is None:
        R_des = np.eye(3)
    quat = np.zeros(4)
    r_err = R_des @ d.site_xmat[site_id].reshape(3, 3).T
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


def solve_ik(
    m: mujoco.MjModel,
    d: mujoco.MjData,
    tip_site_id: int,
    ori_site_id: int,
    target_pos: np.ndarray,
    R_des: np.ndarray,
    q_init: np.ndarray,
    gripper_qpos: np.ndarray,
    max_iter: int = 2000,
    pos_tol: float = 5e-4,
    ori_tol: float = 1e-3,
    gain: float = 1.0,
    max_step: float = 0.05,
    damping: float = 1e-3,
    verbose: bool = False,
) -> tuple[np.ndarray, float, float]:
    """6-DOF damped-pseudoinverse IK: position + orientation control.

    Returns (arm_qpos, pos_err, ori_err).
    """
    d.qpos[:6] = q_init.copy()
    d.qpos[6:14] = gripper_qpos.copy()
    d.ctrl[:6] = q_init.copy()
    d.ctrl[6] = GRIPPER_CLOSED_CTRL
    mujoco.mj_forward(m, d)

    for i in range(max_iter):
        tip = d.site_xpos[tip_site_id].copy()
        perr = target_pos - tip
        oerr = orientation_error(d, ori_site_id, R_des)
        pdist = np.linalg.norm(perr)
        odist = np.linalg.norm(oerr)
        if pdist < pos_tol and odist < ori_tol:
            if verbose:
                print(f"  IK converged at iter {i}, pos_err={pdist:.6f}m ori_err={odist:.6f}")
            break

        pstep = perr * gain
        pn = np.linalg.norm(pstep)
        if pn > max_step:
            pstep *= max_step / pn
        ostep = np.clip(oerr * gain, -0.1, 0.1)

        J = get_jacobian6(m, d, tip_site_id, ori_site_id)
        dq = damped_pinv(J, damping) @ np.concatenate([pstep, ostep])
        d.qpos[:6] += dq
        d.qpos[6:14] = gripper_qpos.copy()
        d.ctrl[:6] = d.qpos[:6].copy()
        d.ctrl[6] = GRIPPER_CLOSED_CTRL
        mujoco.mj_forward(m, d)

    tip_final = d.site_xpos[tip_site_id].copy()
    pos_err = np.linalg.norm(target_pos - tip_final)
    ori_err = np.linalg.norm(orientation_error(d, ori_site_id, R_des))
    return d.qpos[:6].copy(), pos_err, ori_err
