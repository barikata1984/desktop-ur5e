"""GUI viewer: animate UR5e — gripper tip Z traces a 60deg-apex cone around world Z."""

from __future__ import annotations

import json
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from ur5e_sim.core.ik import GRIPPER_CLOSED_CTRL, solve_ik
from ur5e_sim.pushing.keyframe import close_gripper_sim
from ur5e_sim.pushing.kinematics import R_TOOL0_DES

DURATION = 20.0
FPS = 100.0
N_STEPS = int(DURATION * FPS)
AXIS_LEN = 0.08
AXIS_COLORS = [
    [1, 0, 0, 1],
    [0, 1, 0, 1],
    [0, 0, 1, 1],
]


def Ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz_world(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def draw_frame(viewer: mujoco.viewer.Handle, pos: np.ndarray, mat: np.ndarray) -> None:
    viewer.user_scn.ngeom = 0
    for ax in range(3):
        g = viewer.user_scn.geoms[ax]
        mujoco.mjv_initGeom(
            g,
            type=mujoco.mjtGeom.mjGEOM_ARROW,
            size=np.zeros(3),
            pos=np.zeros(3),
            mat=np.eye(3).flatten(),
            rgba=np.array(AXIS_COLORS[ax], dtype=np.float32),
        )
        end = pos + mat[:, ax] * AXIS_LEN
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.005, pos, end)
    viewer.user_scn.ngeom = 3


def main() -> None:
    m = mujoco.MjModel.from_xml_path("scenes/tasks/trajectory_design.xml")
    d = mujoco.MjData(m)

    tip_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "pinch")
    ori_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")

    gripper_qpos = close_gripper_sim(m, d)

    target_pos = np.array([0.0, 0.75, 0.477])
    q_init = np.array([1.1803, -1.6653, 2.4917, -2.3972, -1.5708, -0.3905])

    # Step 1: solve flange-down IK → get site transforms
    print("Solving flange-down reference IK...")
    mujoco.mj_resetData(m, d)
    q_ref, _, _ = solve_ik(m, d, tip_id, ori_id, target_pos, R_TOOL0_DES, q_init, gripper_qpos)

    R_world_att_ref = d.site_xmat[ori_id].reshape(3, 3).copy()
    R_world_pinch_ref = d.site_xmat[tip_id].reshape(3, 3).copy()
    R_att_pinch = R_world_att_ref.T @ R_world_pinch_ref

    # Step 2: tilt +30deg around pinch local Y → starting pose
    print("Solving tilted starting pose (+30deg around pinch Y)...")
    R_pinch_tilted = R_world_pinch_ref @ Ry(np.deg2rad(30))
    R_att_tilted = R_pinch_tilted @ R_att_pinch.T

    mujoco.mj_resetData(m, d)
    q_start, err, _ = solve_ik(
        m,
        d,
        tip_id,
        ori_id,
        target_pos,
        R_att_tilted,
        q_ref,
        gripper_qpos,
    )
    print(f"  Starting pose err: {err * 1000:.1f}mm")

    R_world_att_start = d.site_xmat[ori_id].reshape(3, 3).copy()

    # Step 3: sweep 360deg around world Z (at work surface center = tip position)
    # Pinch Z axis is tilted 30deg from world Z → traces 60deg apex cone
    angles = np.linspace(0, 2 * np.pi, N_STEPS, endpoint=False)
    q_trajectory = np.zeros((N_STEPS, 6))
    q_prev = q_start.copy()

    print(f"Pre-computing {N_STEPS} IK solutions (world Z sweep)...")
    for idx, angle in enumerate(angles):
        R_att_new = Rz_world(angle) @ R_world_att_start
        mujoco.mj_resetData(m, d)
        q, err, _ = solve_ik(m, d, tip_id, ori_id, target_pos, R_att_new, q_prev, gripper_qpos)
        q_trajectory[idx] = q
        q_prev = q
        if idx % 12 == 0:
            print(f"  {idx}/{N_STEPS} (err={err * 1000:.1f}mm)")

    # Apply wrist_3 +90deg offset to all steps
    q_trajectory[:, 5] += np.pi / 2

    # Compute dq/ddq via finite differences
    dt = 1.0 / FPS
    dq_trajectory = np.gradient(q_trajectory, dt, axis=0)
    ddq_trajectory = np.gradient(dq_trajectory, dt, axis=0)

    # Export trajectory JSON
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
    traj_data = {
        "metadata": {
            "description": "Cone sweep trajectory: gripper tip Z traces 60deg-apex cone around world Z",
            "scene": "scenes/tasks/trajectory_design.xml",
            "num_joints": 6,
            "num_steps": N_STEPS,
            "duration": DURATION,
            "fps": FPS,
            "dt": round(dt, 6),
            "joint_names": joint_names,
            "tip_position": target_pos.tolist(),
            "tilt_deg": 30.0,
            "wrist_3_offset_deg": 90.0,
        },
        "trajectory": [
            {
                "t": round(i * dt, 6),
                "q": q_trajectory[i].tolist(),
                "dq": dq_trajectory[i].tolist(),
                "ddq": ddq_trajectory[i].tolist(),
            }
            for i in range(N_STEPS)
        ],
    }

    out_path = Path("results/cone_sweep_traj.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(traj_data, indent=2))
    print(f"Trajectory saved to {out_path}")

    print("Launching viewer...")

    # Set initial pose
    mujoco.mj_resetData(m, d)
    d.qpos[:6] = q_trajectory[0]
    d.qpos[6:14] = gripper_qpos
    d.ctrl[:6] = q_trajectory[0]
    d.ctrl[6] = GRIPPER_CLOSED_CTRL
    mujoco.mj_forward(m, d)

    viewer_fps = 30
    skip = max(1, int(FPS / viewer_fps))

    with mujoco.viewer.launch_passive(m, d) as viewer:
        idx = 0
        while viewer.is_running():
            d.qpos[:6] = q_trajectory[idx]
            d.qpos[6:14] = gripper_qpos
            d.ctrl[:6] = q_trajectory[idx]
            d.ctrl[6] = GRIPPER_CLOSED_CTRL
            mujoco.mj_forward(m, d)

            tip_pos = d.site_xpos[tip_id].copy()
            tip_mat = d.site_xmat[tip_id].reshape(3, 3).copy()
            draw_frame(viewer, tip_pos, tip_mat)

            viewer.sync()
            time.sleep(skip / FPS)
            idx = (idx + skip) % N_STEPS


if __name__ == "__main__":
    main()
