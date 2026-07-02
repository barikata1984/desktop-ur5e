"""Tests for ur5e_sim.core.names constants and ur5e_sim.core.layout.DofLayout."""

import mujoco
import numpy as np
import pytest

from ur5e_sim.core import names
from ur5e_sim.core.layout import DofLayout
from ur5e_sim.core.model_builder import build_ur5e_model
from ur5e_sim.pushing.scene import build_push_model


@pytest.fixture(scope="module")
def identification_model() -> mujoco.MjModel:
    model, _ = build_ur5e_model(payload_xml="scenes/objects/payload_box.xml")
    return model


@pytest.fixture(scope="module")
def push_model() -> mujoco.MjModel:
    model, _ = build_push_model()
    return model


def _resolve(model: mujoco.MjModel, objtype: mujoco.mjtObj, name: str) -> int:
    return mujoco.mj_name2id(model, objtype, name)


class TestNamesResolveInIdentificationModel:
    def test_bodies(self, identification_model: mujoco.MjModel) -> None:
        for body_name in (
            names.PAYLOAD_BODY,
            names.GRIPPER_MOUNT_BODY,
            names.FT300S_MOUNT_BODY,
            names.BASE_BODY,
        ):
            assert _resolve(identification_model, mujoco.mjtObj.mjOBJ_BODY, body_name) >= 0

        for link_name in names.UR5E_LINK_BODIES:
            assert _resolve(identification_model, mujoco.mjtObj.mjOBJ_BODY, link_name) >= 0

    def test_sites(self, identification_model: mujoco.MjModel) -> None:
        for site_name in (names.EE_SITE, names.FT_SITE, names.PINCH_SITE):
            assert _resolve(identification_model, mujoco.mjtObj.mjOBJ_SITE, site_name) >= 0

    def test_sensors(self, identification_model: mujoco.MjModel) -> None:
        for sensor_name in (names.FT_FORCE_SENSOR, names.FT_TORQUE_SENSOR):
            assert _resolve(identification_model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name) >= 0

    def test_geoms(self, identification_model: mujoco.MjModel) -> None:
        for geom_name in (names.PAYLOAD_GEOM, names.WORKSPACE_GEOM, *names.GRIPPER_PAD_GEOMS):
            assert _resolve(identification_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name) >= 0

    def test_cameras(self, identification_model: mujoco.MjModel) -> None:
        for camera_name in (names.SIDE_CAMERA, names.TOP_CAMERA):
            assert _resolve(identification_model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name) >= 0

    def test_keyframes(self, identification_model: mujoco.MjModel) -> None:
        assert _resolve(identification_model, mujoco.mjtObj.mjOBJ_KEY, names.HOME_KEYFRAME) >= 0

    def test_joints_and_actuators(self, identification_model: mujoco.MjModel) -> None:
        for joint_name in names.ARM_JOINTS:
            assert _resolve(identification_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name) >= 0
        for actuator_name in (*names.ARM_ACTUATORS, names.GRIPPER_ACTUATOR):
            assert _resolve(identification_model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name) >= 0


class TestNamesResolveInPushModel:
    def test_slider(self, push_model: mujoco.MjModel) -> None:
        assert _resolve(push_model, mujoco.mjtObj.mjOBJ_BODY, names.SLIDER_BODY) >= 0
        assert _resolve(push_model, mujoco.mjtObj.mjOBJ_SITE, names.SLIDER_CENTER_SITE) >= 0
        assert _resolve(push_model, mujoco.mjtObj.mjOBJ_GEOM, names.SLIDER_GEOM) >= 0

    def test_cameras(self, push_model: mujoco.MjModel) -> None:
        for camera_name in (names.SIDE_CAMERA, names.TOP_CAMERA):
            assert _resolve(push_model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name) >= 0

    def test_keyframes(self, push_model: mujoco.MjModel) -> None:
        assert _resolve(push_model, mujoco.mjtObj.mjOBJ_KEY, names.HOME_KEYFRAME) >= 0
        assert _resolve(push_model, mujoco.mjtObj.mjOBJ_KEY, names.READY_KEYFRAME) >= 0

    def test_workspace_geom(self, push_model: mujoco.MjModel) -> None:
        assert _resolve(push_model, mujoco.mjtObj.mjOBJ_GEOM, names.WORKSPACE_GEOM) >= 0

    def test_gripper_actuator_present_no_ft300s(self, push_model: mujoco.MjModel) -> None:
        assert _resolve(push_model, mujoco.mjtObj.mjOBJ_ACTUATOR, names.GRIPPER_ACTUATOR) >= 0


class TestDofLayoutIdentificationModel:
    def test_slices(self, identification_model: mujoco.MjModel) -> None:
        layout = DofLayout.from_model(identification_model)
        assert layout.arm_qpos == slice(0, 6)
        assert layout.arm_dof == slice(0, 6)
        assert layout.gripper_qpos == slice(6, 14)
        assert layout.gripper_dof == slice(6, 14)
        assert layout.arm_ctrl == slice(0, 6)
        assert layout.gripper_ctrl == 6
        assert layout.n_arm == 6
        assert layout.nq == 14
        assert layout.nv == 14
        assert layout.nu == 7


class TestDofLayoutPushModel:
    def test_slices(self, push_model: mujoco.MjModel) -> None:
        layout = DofLayout.from_model(push_model)
        assert layout.arm_qpos == slice(0, 6)
        assert layout.arm_dof == slice(0, 6)
        assert layout.arm_ctrl == slice(0, 6)
        # Gripper block unaffected by the trailing slider freejoint.
        assert layout.gripper_qpos == slice(6, 14)
        assert layout.gripper_dof == slice(6, 14)
        assert layout.gripper_ctrl == 6
        assert layout.nq == 21
        assert layout.nv == 20
        assert layout.nu == 7


class TestDofLayoutExpansion:
    def test_to_full_qpos_1d_zero_base(self, identification_model: mujoco.MjModel) -> None:
        layout = DofLayout.from_model(identification_model)
        q_arm = np.arange(6, dtype=float)
        full = layout.to_full_qpos(q_arm)
        assert full.shape == (14,)
        assert np.array_equal(full[:6], q_arm)
        assert np.all(full[6:] == 0.0)

    def test_to_full_qpos_1d_explicit_base(self, identification_model: mujoco.MjModel) -> None:
        layout = DofLayout.from_model(identification_model)
        q_arm = np.arange(6, dtype=float)
        base = np.full(14, 9.0)
        full = layout.to_full_qpos(q_arm, base=base)
        assert np.array_equal(full[:6], q_arm)
        assert np.all(full[6:] == 9.0)

    def test_to_full_qpos_2d(self, identification_model: mujoco.MjModel) -> None:
        layout = DofLayout.from_model(identification_model)
        q_arm = np.tile(np.arange(6, dtype=float), (3, 1))  # (T=3, 6)
        full = layout.to_full_qpos(q_arm)
        assert full.shape == (3, 14)
        assert np.array_equal(full[:, :6], q_arm)
        assert np.all(full[:, 6:] == 0.0)

    def test_to_full_qvel_1d(self, identification_model: mujoco.MjModel) -> None:
        layout = DofLayout.from_model(identification_model)
        v_arm = np.arange(6, dtype=float) * 0.1
        full = layout.to_full_qvel(v_arm)
        assert full.shape == (14,)
        assert np.array_equal(full[:6], v_arm)

    def test_round_trip_1d(self, identification_model: mujoco.MjModel) -> None:
        layout = DofLayout.from_model(identification_model)
        q_arm = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
        full = layout.to_full_qpos(q_arm)
        assert np.array_equal(layout.arm(full), q_arm)

    def test_round_trip_2d(self, identification_model: mujoco.MjModel) -> None:
        layout = DofLayout.from_model(identification_model)
        q_arm = np.arange(18, dtype=float).reshape(3, 6)
        full = layout.to_full_qpos(q_arm)
        assert np.array_equal(layout.arm(full), q_arm)

    def test_arm_rejects_mismatched_width(self, identification_model: mujoco.MjModel) -> None:
        layout = DofLayout.from_model(identification_model)
        with pytest.raises(ValueError):
            layout.arm(np.zeros(5))


class TestDofLayoutCtrl:
    def test_set_arm_ctrl(self, identification_model: mujoco.MjModel) -> None:
        layout = DofLayout.from_model(identification_model)
        data = mujoco.MjData(identification_model)
        data.ctrl[:] = -1.0
        layout.set_arm_ctrl(data, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        assert np.array_equal(data.ctrl[:6], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        assert data.ctrl[6] == -1.0  # gripper untouched

    def test_hold_gripper_ctrl(self, identification_model: mujoco.MjModel) -> None:
        layout = DofLayout.from_model(identification_model)
        data = mujoco.MjData(identification_model)
        data.ctrl[:] = -1.0
        layout.hold_gripper_ctrl(data, 255.0)
        assert data.ctrl[6] == 255.0
        assert np.all(data.ctrl[:6] == -1.0)  # arm untouched

    def test_hold_gripper_ctrl_raises_without_gripper(self) -> None:
        layout = DofLayout(
            n_arm=6,
            arm_qpos=slice(0, 6),
            arm_dof=slice(0, 6),
            gripper_qpos=None,
            gripper_dof=None,
            arm_ctrl=slice(0, 6),
            gripper_ctrl=None,
            nq=6,
            nv=6,
            nu=6,
        )
        with pytest.raises(ValueError):
            layout.hold_gripper_ctrl(
                mujoco.MjData(mujoco.MjModel.from_xml_string("<mujoco/>")), 0.0
            )


class TestDofLayoutNoArmJoints:
    def test_raises_value_error(self) -> None:
        xml = """
        <mujoco>
          <worldbody>
            <body name="dummy">
              <geom type="sphere" size="0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        with pytest.raises(ValueError):
            DofLayout.from_model(model)
