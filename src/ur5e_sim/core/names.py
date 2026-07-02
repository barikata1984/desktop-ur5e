"""Canonical MuJoCo object names shared across the identification and push scenes."""

from __future__ import annotations

# --- Bodies ---
PAYLOAD_BODY = "payload_box_mount"
GRIPPER_MOUNT_BODY = "ft300s_gripper_mount"
FT300S_MOUNT_BODY = "ft300s_mount"
BASE_BODY = "base"
SLIDER_BODY = "slider"

# --- Sites ---
EE_SITE = "attachment_site"
FT_SITE = "ft300s_ft_sensor"
PINCH_SITE = "gripper_pinch"
SLIDER_CENTER_SITE = "slider_center"

# --- Sensors ---
FT_FORCE_SENSOR = "ft300s_ft_force"
FT_TORQUE_SENSOR = "ft300s_ft_torque"

# --- Geoms ---
PAYLOAD_GEOM = "payload_box_red"
WORKSPACE_GEOM = "workspace_region_geom"
SLIDER_GEOM = "slider_geom"
GRIPPER_PAD_GEOMS = (
    "gripper_right_pad1",
    "gripper_right_pad2",
    "gripper_left_pad1",
    "gripper_left_pad2",
)

# --- Cameras (push model) ---
SIDE_CAMERA = "env_side"
TOP_CAMERA = "env_top_down"

# --- Keyframes ---
HOME_KEYFRAME = "home"
READY_KEYFRAME = "ready"

# --- Joint/actuator names (path-invariant, from ur5e.xml base spec) ---
ARM_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
ARM_ACTUATORS = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")
GRIPPER_ACTUATOR = "gripper_fingers_actuator"
UR5E_LINK_BODIES = (
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
)
