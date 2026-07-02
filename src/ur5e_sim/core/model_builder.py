"""Compose a UR5e + FT300s + 2F-85 gripper model programmatically via MjSpec.attach().

MjSpec.attach() places the child model's root body at the parent's site frame,
overriding pos/quat on the child root.  After attachment, child element names
are prefixed (e.g. ``ft300s_tool_output``, ``gripper_right_driver_joint``),
and can be looked up with ``spec.site(prefixed_name)``.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from ur5e_sim.core import names

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

_ARM_HOME_QPOS = [1.031643, -1.461450, 2.562062, -4.242204, -1.031607, 0.000245]
_GRIPPER_HOME_QPOS = [0.0] * 8  # 8 gripper joints (drivers, couplers, spring links, followers)
_ARM_HOME_CTRL = [1.031643, -1.461450, 2.562062, -4.242204, -1.031607, 0.000245]
_GRIPPER_HOME_CTRL = [0.0]  # single actuator


def build_ur5e_model(
    ur5e_xml: str = "assets/mujoco_menagerie/universal_robots_ur5e/ur5e.xml",
    ft300s_xml: str = "scenes/common/sensors/ft300s.xml",
    gripper_xml: str = "assets/mujoco_menagerie/robotiq_2f85/2f85.xml",
    payload_xml: str | None = None,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Build a UR5e + FT300s + 2F-85 model with environment, compile and return.

    Args:
        ur5e_xml: Path to UR5e MJCF relative to project root.
        ft300s_xml: Path to FT300s MJCF relative to project root.
        gripper_xml: Path to 2F-85 gripper MJCF relative to project root.
        payload_xml: Optional path to a payload MJCF relative to project root.
            When provided, the payload is attached to the ``ft300s_ft_sensor``
            site so that its mass is measured by the F/T sensor.

    Returns:
        Compiled ``(MjModel, MjData)`` tuple with the robot reset to its home
        keyframe.
    """
    spec = build_spec(
        ur5e_xml=ur5e_xml,
        ft300s_xml=ft300s_xml,
        gripper_xml=gripper_xml,
        payload_xml=payload_xml,
    )
    model = spec.compile()
    data = mujoco.MjData(model)

    # Reset to home keyframe
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, names.HOME_KEYFRAME)
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    return model, data


def build_spec(
    ur5e_xml: str = "assets/mujoco_menagerie/universal_robots_ur5e/ur5e.xml",
    ft300s_xml: str | None = "scenes/common/sensors/ft300s.xml",
    gripper_xml: str = "assets/mujoco_menagerie/robotiq_2f85/2f85.xml",
    payload_xml: str | None = None,
    slider_xml: str | None = None,
    contact_excludes: list[tuple[str, str]] | None = None,
    contact_pairs: list[dict] | None = None,
    extra_cameras: list[dict] | None = None,
    extra_keyframes: list[dict] | None = None,
) -> mujoco.MjSpec:
    """Assemble the full MjSpec without compiling.

    Args:
        ur5e_xml: Path to UR5e MJCF relative to project root.
        ft300s_xml: Path to FT300s MJCF relative to project root.
            Pass ``None`` to skip FT300s and attach the gripper directly.
        gripper_xml: Path to 2F-85 gripper MJCF relative to project root.
        payload_xml: Optional payload MJCF (attached at gripper pinch site).
        slider_xml: Optional slider MJCF relative to project root.
        contact_excludes: List of ``(body1, body2)`` pairs to exclude from
            collision detection.
        contact_pairs: List of dicts specifying contact pairs. Each dict must
            have ``geom1`` and ``geom2``; optional keys: ``condim``,
            ``friction``, ``solref``, ``solimp``.
        extra_cameras: List of camera dicts with ``name``, ``pos``, and
            optional ``mode``, ``targetbody``, ``fovy``.
        extra_keyframes: List of keyframe dicts with ``name``, ``qpos``,
            ``ctrl``.
    """
    # --- 1. Load UR5e ---
    spec = mujoco.MjSpec.from_file(str(_PROJECT_ROOT / ur5e_xml))

    # --- 2. FT300s + gripper attachment ---
    if ft300s_xml is not None:
        # Adjust attachment_site to match real FT300s mounting offset
        spec.site("attachment_site").pos = [0.0, 0.094, 0.0]
        ft300s = mujoco.MjSpec.from_file(str(_PROJECT_ROOT / ft300s_xml))
        spec.attach(ft300s, prefix="ft300s_", site=spec.site("attachment_site"))
        gripper = mujoco.MjSpec.from_file(str(_PROJECT_ROOT / gripper_xml))
        spec.attach(gripper, prefix="gripper_", site=spec.site("ft300s_tool_output"))
        spec.body("gripper_base_mount").pos = [0.0, 0.0, 0.004]
    else:
        # No FT300s: attach gripper directly to the UR5e attachment_site
        gripper = mujoco.MjSpec.from_file(str(_PROJECT_ROOT / gripper_xml))
        spec.attach(gripper, prefix="gripper_", site=spec.site("attachment_site"))

    # --- 3. Optional payload at gripper pinch site ---
    if payload_xml is not None:
        payload = mujoco.MjSpec.from_file(str(_PROJECT_ROOT / payload_xml))
        spec.attach(payload, prefix="payload_", site=spec.site("gripper_pinch"))

    # --- 4. Optional slider ---
    if slider_xml is not None:
        slider_site = spec.worldbody.add_site()
        slider_site.name = "slider_attach"
        slider = mujoco.MjSpec.from_file(str(_PROJECT_ROOT / slider_xml))
        spec.attach(slider, prefix="", site=slider_site)

    # --- 5. Attach environment (floor, table, lighting) ---
    env_site = spec.worldbody.add_site()
    env_site.name = "env_attach"
    env = mujoco.MjSpec.from_file(str(_PROJECT_ROOT / "scenes" / "common" / "environment.xml"))
    spec.attach(env, prefix="env_", site=env_site)

    # --- 6. Visual settings (not inherited from attached environment.xml) ---
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1440
    spec.visual.global_.azimuth = 120
    spec.visual.global_.elevation = -20

    # --- 7. Position UR5e base on table ---
    # Menagerie ur5e.xml has quat="0 0 0 -1" (180° about z) on the base body.
    # Override to identity to match the project convention (y-axis toward workspace).
    base = spec.body(names.BASE_BODY)
    base.pos = [0.0, 0.1, 0.3]
    base.quat = [1.0, 0.0, 0.0, 0.0]

    # --- 7. Workspace region (visual-only, read by optimizer at runtime) ---
    ws_body = spec.worldbody.add_body()
    ws_body.name = "workspace_region"
    ws_body.pos = [0.0, 0.65, 0.636]
    ws_geom = ws_body.add_geom()
    ws_geom.name = names.WORKSPACE_GEOM
    ws_geom.type = mujoco.mjtGeom.mjGEOM_BOX
    ws_geom.size = [0.25, 0.35, 0.275]
    ws_geom.contype = 0
    ws_geom.conaffinity = 0
    ws_geom.group = 4
    ws_geom.rgba = [0.2, 0.7, 0.9, 0.18]

    # --- 8. Update home keyframe for the extended qpos/ctrl ---
    _update_home_keyframe(spec)

    # --- 9. Extra cameras ---
    if extra_cameras is not None:
        for cam_spec in extra_cameras:
            cam = spec.worldbody.add_camera()
            cam.name = cam_spec["name"]
            cam.mode = cam_spec.get("mode", mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODYCOM)
            if "targetbody" in cam_spec:
                cam.targetbody = cam_spec["targetbody"]
            cam.pos = cam_spec["pos"]
            if "fovy" in cam_spec:
                cam.fovy = cam_spec["fovy"]

    # --- 10. Contact excludes ---
    if contact_excludes is not None:
        for body1, body2 in contact_excludes:
            exclude = spec.add_exclude()
            exclude.bodyname1 = body1
            exclude.bodyname2 = body2

    # --- 11. Contact pairs ---
    if contact_pairs is not None:
        for pair_spec in contact_pairs:
            pair = spec.add_pair()
            pair.geomname1 = pair_spec["geom1"]
            pair.geomname2 = pair_spec["geom2"]
            if "condim" in pair_spec:
                pair.condim = pair_spec["condim"]
            if "friction" in pair_spec:
                pair.friction = np.array(pair_spec["friction"])
            if "solref" in pair_spec:
                pair.solref = np.array(pair_spec["solref"])
            if "solimp" in pair_spec:
                pair.solimp = np.array(pair_spec["solimp"])

    # --- 12. Extra keyframes ---
    if extra_keyframes is not None:
        for kf in extra_keyframes:
            key = spec.add_key()
            key.name = kf["name"]
            key.qpos = np.array(kf["qpos"])
            key.ctrl = np.array(kf["ctrl"])

    return spec


def _update_home_keyframe(spec: mujoco.MjSpec) -> None:
    """Set the home keyframe qpos and ctrl for arm + gripper joints."""
    try:
        key = spec.key(names.HOME_KEYFRAME)
    except Exception:
        key = spec.add_key()
        key.name = names.HOME_KEYFRAME

    key.qpos = np.array(_ARM_HOME_QPOS + _GRIPPER_HOME_QPOS)
    key.ctrl = np.array(_ARM_HOME_CTRL + _GRIPPER_HOME_CTRL)
