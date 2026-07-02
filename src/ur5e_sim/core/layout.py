"""Validated index spaces (arm joints, gripper joints, actuators) of a UR5e model."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from ur5e_sim.core import names


def _contiguous_slice(addrs: list[int], label: str) -> slice:
    """Build a slice from a list of addresses, raising if they are not contiguous."""
    sorted_addrs = sorted(addrs)
    start = sorted_addrs[0]
    expected = list(range(start, start + len(sorted_addrs)))
    if sorted_addrs != expected:
        raise ValueError(f"{label} are not contiguous: {sorted_addrs}")
    return slice(start, start + len(sorted_addrs))


@dataclass(frozen=True)
class DofLayout:
    """Validated index spaces of a UR5e model: arm joints, gripper joints, actuators."""

    n_arm: int
    arm_qpos: slice
    arm_dof: slice
    gripper_qpos: slice | None
    gripper_dof: slice | None
    arm_ctrl: slice
    gripper_ctrl: int | None
    nq: int
    nv: int
    nu: int

    @classmethod
    def from_model(cls, model: mujoco.MjModel) -> "DofLayout":
        """Derive index spaces from ``names.ARM_JOINTS``/``names.ARM_ACTUATORS``.

        Raises:
            ValueError: If arm joints/actuators are missing, or their qpos/dof/id
                addresses are not contiguous.
        """
        arm_qpos_addrs: list[int] = []
        arm_dof_addrs: list[int] = []
        missing_joints = []
        for joint_name in names.ARM_JOINTS:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if jid < 0:
                missing_joints.append(joint_name)
                continue
            arm_qpos_addrs.append(int(model.jnt_qposadr[jid]))
            arm_dof_addrs.append(int(model.jnt_dofadr[jid]))
        if missing_joints:
            raise ValueError(f"Arm joints not found in model: {missing_joints}")

        arm_qpos = _contiguous_slice(arm_qpos_addrs, "arm joint qpos addresses")
        arm_dof = _contiguous_slice(arm_dof_addrs, "arm joint dof addresses")

        arm_ctrl_ids: list[int] = []
        missing_actuators = []
        for actuator_name in names.ARM_ACTUATORS:
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            if aid < 0:
                missing_actuators.append(actuator_name)
                continue
            arm_ctrl_ids.append(aid)
        if missing_actuators:
            raise ValueError(f"Arm actuators not found in model: {missing_actuators}")
        arm_ctrl = _contiguous_slice(arm_ctrl_ids, "arm actuator ids")

        gripper_ctrl_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, names.GRIPPER_ACTUATOR
        )
        gripper_ctrl = gripper_ctrl_id if gripper_ctrl_id >= 0 else None

        gripper_qpos_addrs: list[int] = []
        gripper_dof_addrs: list[int] = []
        for jid in range(model.njnt):
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if joint_name is None or not joint_name.startswith("gripper_"):
                continue
            if model.jnt_type[jid] not in (
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            ):
                continue
            gripper_qpos_addrs.append(int(model.jnt_qposadr[jid]))
            gripper_dof_addrs.append(int(model.jnt_dofadr[jid]))

        if gripper_qpos_addrs:
            gripper_qpos = _contiguous_slice(gripper_qpos_addrs, "gripper joint qpos addresses")
            gripper_dof = _contiguous_slice(gripper_dof_addrs, "gripper joint dof addresses")
        else:
            gripper_qpos = None
            gripper_dof = None

        return cls(
            n_arm=len(names.ARM_JOINTS),
            arm_qpos=arm_qpos,
            arm_dof=arm_dof,
            gripper_qpos=gripper_qpos,
            gripper_dof=gripper_dof,
            arm_ctrl=arm_ctrl,
            gripper_ctrl=gripper_ctrl,
            nq=int(model.nq),
            nv=int(model.nv),
            nu=int(model.nu),
        )

    def arm(self, full: np.ndarray) -> np.ndarray:
        """View of the arm block of a full qpos/qvel/qacc-width array (1D or 2D last axis)."""
        width = full.shape[-1]
        if width == self.nq:
            index = self.arm_qpos
        elif width == self.nv:
            index = self.arm_dof
        else:
            raise ValueError(f"Array width {width} matches neither nq={self.nq} nor nv={self.nv}")
        return full[..., index]

    def _expand(
        self, arm_values: np.ndarray, index: slice, width: int, base: np.ndarray | None
    ) -> np.ndarray:
        arm_values = np.asarray(arm_values, dtype=float)
        if arm_values.shape[-1] != self.n_arm:
            raise ValueError(f"Expected last axis of size {self.n_arm}, got {arm_values.shape[-1]}")
        out_shape = arm_values.shape[:-1] + (width,)
        if base is None:
            out = np.zeros(out_shape, dtype=float)
        else:
            out = np.broadcast_to(np.asarray(base, dtype=float), out_shape).copy()
        out[..., index] = arm_values
        return out

    def to_full_qpos(self, q_arm: np.ndarray, base: np.ndarray | None = None) -> np.ndarray:
        """Expand arm-width row(s) to nq width; non-arm entries from base or zero."""
        return self._expand(q_arm, self.arm_qpos, self.nq, base)

    def to_full_qvel(self, v_arm: np.ndarray, base: np.ndarray | None = None) -> np.ndarray:
        """Expand arm-width row(s) to nv width; non-arm entries from base or zero."""
        return self._expand(v_arm, self.arm_dof, self.nv, base)

    def set_arm_ctrl(self, data: mujoco.MjData, values: np.ndarray) -> None:
        """Write arm actuator commands; leaves gripper ctrl untouched."""
        values = np.asarray(values, dtype=float)
        if values.shape != (self.n_arm,):
            raise ValueError(f"Expected shape ({self.n_arm},), got {values.shape}")
        data.ctrl[self.arm_ctrl] = values

    def hold_gripper_ctrl(self, data: mujoco.MjData, value: float) -> None:
        """Write the gripper actuator command."""
        if self.gripper_ctrl is None:
            raise ValueError("Model has no gripper actuator")
        data.ctrl[self.gripper_ctrl] = value
