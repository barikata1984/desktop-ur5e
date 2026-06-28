"""Build the push-task scene programmatically via MjSpec.

Composes UR5e + 2F-85 gripper (no FT300s) + slider using MjSpec.attach().
Contact exclusions and pairs reproduce the XML-based ``scenes/tasks/push.xml``.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

# "ready" keyframe values (from scenes/tasks/push.xml).
# qpos layout: 6 arm + 8 gripper (closed) + 7 slider freejoint = 21.
_READY_QPOS = [
    # arm
    1.1803,
    -1.6653,
    2.4917,
    -2.3972,
    -1.5708,
    -0.3905,
    # gripper (closed)
    0.782171,
    0.000500,
    0.780958,
    -0.757842,
    0.782169,
    0.000503,
    0.780896,
    -0.758075,
    # slider freejoint (pos xyz + quat wxyz)
    0.0,
    0.5,
    0.342,
    1.0,
    0.0,
    0.0,
    0.0,
]
# ctrl layout: 6 arm + 1 gripper actuator (255 = closed) = 7.
_READY_CTRL = [1.1803, -1.6708, 2.4842, -2.4020, -1.5708, -0.3905, 255.0]

# Bodies needing collision exclusions against both env surfaces.
_EXCLUDE_BODIES = [
    "base",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
    "gripper_base_mount",
    "gripper_base",
    "gripper_right_driver",
    "gripper_right_coupler",
    "gripper_right_spring_link",
    "gripper_right_follower",
    "gripper_right_pad",
    "gripper_left_driver",
    "gripper_left_coupler",
    "gripper_left_spring_link",
    "gripper_left_follower",
    "gripper_left_pad",
]
_ENV_SURFACES = ["env_table", "env_work_surface"]


def build_push_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Build UR5e + 2F-85 (no FT300s) + slider for the push task."""
    spec = _build_push_spec()
    model = spec.compile()
    data = mujoco.MjData(model)

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    return model, data


def _build_push_spec() -> mujoco.MjSpec:
    """Assemble the push-task MjSpec without compiling."""
    spec = mujoco.MjSpec.from_file(
        str(_PROJECT_ROOT / "assets/mujoco_menagerie/universal_robots_ur5e/ur5e.xml")
    )

    # Attach gripper directly to wrist_3_link (no FT300s for push task)
    gripper = mujoco.MjSpec.from_file(
        str(_PROJECT_ROOT / "assets/mujoco_menagerie/robotiq_2f85/2f85.xml")
    )
    spec.attach(gripper, prefix="gripper_", site=spec.site("attachment_site"))

    # Attach environment (floor, table, lighting)
    env_site = spec.worldbody.add_site()
    env_site.name = "env_attach"
    env = mujoco.MjSpec.from_file(str(_PROJECT_ROOT / "scenes/common/environment.xml"))
    spec.attach(env, prefix="env_", site=env_site)

    # Attach slider to worldbody
    slider_site = spec.worldbody.add_site()
    slider_site.name = "slider_attach"
    slider = mujoco.MjSpec.from_file(str(_PROJECT_ROOT / "scenes/objects/slider.xml"))
    spec.attach(slider, prefix="", site=slider_site)

    # Visual settings (not inherited from attached environment.xml)
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1440
    spec.visual.global_.azimuth = 120
    spec.visual.global_.elevation = -20

    # Position UR5e base on table
    base = spec.body("base")
    base.pos = [0.0, 0.1, 0.3]
    base.quat = [1.0, 0.0, 0.0, 0.0]

    # Workspace region (visual-only, read by optimizer at runtime)
    ws_body = spec.worldbody.add_body()
    ws_body.name = "workspace_region"
    ws_body.pos = [0.0, 0.65, 0.636]
    ws_geom = ws_body.add_geom()
    ws_geom.name = "workspace_region_geom"
    ws_geom.type = mujoco.mjtGeom.mjGEOM_BOX
    ws_geom.size = [0.25, 0.35, 0.275]
    ws_geom.contype = 0
    ws_geom.conaffinity = 0
    ws_geom.group = 4
    ws_geom.rgba = [0.2, 0.7, 0.9, 0.18]

    # Slider tracking camera
    cam = spec.worldbody.add_camera()
    cam.name = "slider_track"
    cam.mode = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODYCOM
    cam.targetbody = "slider"
    cam.pos = [0.35, 0.5, 0.65]
    cam.fovy = 50.0

    # Contact configuration
    _add_contact_excludes(spec)
    _add_contact_pairs(spec)

    # Update the menagerie home keyframe to match the extended qpos/ctrl layout
    _update_home_keyframe(spec)

    # Push-task "ready" keyframe
    key = spec.add_key()
    key.name = "ready"
    key.qpos = np.array(_READY_QPOS)
    key.ctrl = np.array(_READY_CTRL)

    return spec


def _update_home_keyframe(spec: mujoco.MjSpec) -> None:
    """Extend the menagerie home keyframe to cover gripper + slider dofs."""
    try:
        key = spec.key("home")
    except Exception:
        return
    arm_qpos = [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
    gripper_qpos = [0.0] * 8
    slider_qpos = [0.0, 0.5, 0.342, 1.0, 0.0, 0.0, 0.0]
    key.qpos = np.array(arm_qpos + gripper_qpos + slider_qpos)
    key.ctrl = np.array(arm_qpos + [0.0])


def _add_contact_excludes(spec: mujoco.MjSpec) -> None:
    """Add collision exclusions for robot/gripper bodies vs table/work_surface."""
    for body in _EXCLUDE_BODIES:
        for surface in _ENV_SURFACES:
            exc = spec.add_exclude()
            exc.bodyname1 = body
            exc.bodyname2 = surface


def _add_contact_pairs(spec: mujoco.MjSpec) -> None:
    """Add explicit contact pairs with tuned friction parameters."""
    # Slider vs table/work_surface: mu=0.35, condim=4 (Hogan 2016)
    for env_geom in ("env_table_surface", "env_work_surface_geom"):
        pair = spec.add_pair()
        pair.geomname1 = "slider_geom"
        pair.geomname2 = env_geom
        pair.condim = 4
        pair.friction = [0.35, 0.35, 0.005, 0.001, 0.001]
        pair.solref = [0.02, 1.0]
        pair.solimp = [0.9, 0.95, 0.001, 0.5, 2.0]

    # Gripper pads vs slider: mu=0.3, condim=4 (Hogan 2016)
    for pad_geom in (
        "gripper_right_pad1",
        "gripper_right_pad2",
        "gripper_left_pad1",
        "gripper_left_pad2",
    ):
        pair = spec.add_pair()
        pair.geomname1 = pad_geom
        pair.geomname2 = "slider_geom"
        pair.condim = 4
        pair.friction = [0.3, 0.3, 0.005, 0.001, 0.001]
        pair.solref = [0.004, 1.0]
        pair.solimp = [0.95, 0.99, 0.001, 0.5, 2.0]
