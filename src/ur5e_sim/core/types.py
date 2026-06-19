from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class FramePose:
    name: str
    position: tuple[float, float, float]
    rotation: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]


def _rotation_tuple(
    matrix: np.ndarray,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    return tuple(tuple(float(value) for value in row) for row in matrix)  # type: ignore[return-value]


def _named_object_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int | None:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id == -1:
        return None
    return int(object_id)


def get_body_frame(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> FramePose | None:
    body_id = _named_object_id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id is None:
        return None

    position = tuple(float(value) for value in data.xpos[body_id])
    rotation = np.array(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
    return FramePose(name=body_name, position=position, rotation=_rotation_tuple(rotation))


def get_site_frame(model: mujoco.MjModel, data: mujoco.MjData, site_name: str) -> FramePose | None:
    site_id = _named_object_id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id is None:
        return None

    position = tuple(float(value) for value in data.site_xpos[site_id])
    rotation = np.array(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
    return FramePose(name=site_name, position=position, rotation=_rotation_tuple(rotation))


def get_site_body_name(model: mujoco.MjModel, site_name: str) -> str | None:
    site_id = _named_object_id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id is None:
        return None

    body_id = int(model.site_bodyid[site_id])
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
