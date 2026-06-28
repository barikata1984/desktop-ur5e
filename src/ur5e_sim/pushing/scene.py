"""Push-task scene builder: UR5e + 2F-85 (no FT300s) + slider.

Delegates to :func:`ur5e_sim.core.model_builder.build_spec` for robot and
environment assembly, adding push-specific contact specs, the slider-tracking
camera, and the ``ready`` keyframe.
"""

from __future__ import annotations

import mujoco
import numpy as np

from ur5e_sim.core.model_builder import build_spec

# Ready keyframe: arm at vertical-pusher pose, gripper closed, slider at start.
# qpos: 6 arm + 8 gripper + 7 slider_freejoint = 21
_READY_QPOS = [
    # Arm
    1.1803,
    -1.6653,
    2.4917,
    -2.3972,
    -1.5708,
    -0.3905,
    # Gripper (closed 4-bar linkage)
    0.782171,
    0.000500,
    0.780958,
    -0.757842,
    0.782169,
    0.000503,
    0.780896,
    -0.758075,
    # Slider freejoint (pos xyz + quat wxyz)
    0.0,
    0.5,
    0.342,
    1.0,
    0.0,
    0.0,
    0.0,
]
# ctrl: 6 arm + 1 gripper = 7
_READY_CTRL = [
    1.1803,
    -1.6708,
    2.4842,
    -2.4020,
    -1.5708,
    -0.3905,
    255.0,
]

# Bodies whose collision capsules may clip through the table / work surface
# during normal operation. Names reflect the MjSpec prefix scheme:
# UR5e bodies have no prefix; gripper bodies are prefixed ``gripper_``.
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

# Environment surface bodies (prefixed ``env_`` by build_spec).
_ENV_SURFACES = ["env_table", "env_work_surface"]


def build_push_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Build UR5e + 2F-85 (no FT300s) + slider for the push task."""
    contact_excludes = [(body, surface) for body in _EXCLUDE_BODIES for surface in _ENV_SURFACES]
    contact_pairs = [
        # Slider vs table/work_surface: mu=0.35, condim=4 (Hogan 2016)
        *[
            {
                "geom1": "slider_geom",
                "geom2": env_geom,
                "condim": 4,
                "friction": [0.35, 0.35, 0.005, 0.001, 0.001],
                "solref": [0.02, 1.0],
                "solimp": [0.9, 0.95, 0.001, 0.5, 2.0],
            }
            for env_geom in ("env_table_surface", "env_work_surface_geom")
        ],
        # Gripper pads vs slider: mu=0.3, condim=4 (Hogan 2016)
        *[
            {
                "geom1": pad_geom,
                "geom2": "slider_geom",
                "condim": 4,
                "friction": [0.3, 0.3, 0.005, 0.001, 0.001],
                "solref": [0.004, 1.0],
                "solimp": [0.95, 0.99, 0.001, 0.5, 2.0],
            }
            for pad_geom in (
                "gripper_right_pad1",
                "gripper_right_pad2",
                "gripper_left_pad1",
                "gripper_left_pad2",
            )
        ],
    ]
    extra_cameras = [
        {
            "name": "slider_track",
            "mode": mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODYCOM,
            "targetbody": "slider",
            "pos": [0.35, 0.5, 0.65],
            "fovy": 50.0,
        },
    ]
    extra_keyframes = [
        {
            "name": "ready",
            "qpos": _READY_QPOS,
            "ctrl": _READY_CTRL,
        },
    ]

    spec = build_spec(
        ft300s_xml=None,
        slider_xml="scenes/objects/slider.xml",
        contact_excludes=contact_excludes,
        contact_pairs=contact_pairs,
        extra_cameras=extra_cameras,
        extra_keyframes=extra_keyframes,
    )
    model = spec.compile()
    data = mujoco.MjData(model)

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    return model, data
