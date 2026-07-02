from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco

from ur5e_sim.core import names


@dataclass
class LoadedModel:
    model: mujoco.MjModel
    data: mujoco.MjData
    model_path: Path


def load_model(model_path: str | Path) -> LoadedModel:
    resolved_path = Path(model_path).expanduser().resolve()
    model = mujoco.MjModel.from_xml_path(str(resolved_path))
    data = mujoco.MjData(model)
    return LoadedModel(model=model, data=data, model_path=resolved_path)


def get_named_object_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int | None:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id == -1:
        return None
    return int(object_id)


import numpy as np


def get_workspace_bounds(
    model: mujoco.MjModel, data: mujoco.MjData
) -> tuple[np.ndarray, np.ndarray]:
    """Return (box_lower, box_upper) from workspace_region_geom."""
    geom_id = get_named_object_id(model, mujoco.mjtObj.mjOBJ_GEOM, names.WORKSPACE_GEOM)
    if geom_id is None:
        raise RuntimeError("workspace_region_geom not found in model")
    body_id = model.geom_bodyid[geom_id]
    mujoco.mj_kinematics(model, data)
    center = data.xpos[body_id].copy()
    half = model.geom_size[geom_id].copy()
    return center - half, center + half


def reset_to_home(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    key_id = get_named_object_id(model, mujoco.mjtObj.mjOBJ_KEY, names.HOME_KEYFRAME)
    if key_id is None:
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        return False

    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    return True


class SimEnv:
    """MuJoCo environment: load, reset, step."""

    def __init__(self, scene_path: str | Path):
        self._loaded = load_model(scene_path)

    @property
    def model(self) -> mujoco.MjModel:
        return self._loaded.model

    @property
    def data(self) -> mujoco.MjData:
        return self._loaded.data

    @property
    def model_path(self) -> Path:
        return self._loaded.model_path

    def reset(self, keyframe: str | None = None) -> None:
        if keyframe is None:
            mujoco.mj_resetData(self.model, self.data)
        else:
            key_id = get_named_object_id(self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
            if key_id is None:
                raise ValueError(f"Unknown keyframe: {keyframe}")
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        mujoco.mj_forward(self.model, self.data)

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    def forward(self) -> None:
        mujoco.mj_forward(self.model, self.data)

    def get_body_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

    def get_site_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)

    def get_geom_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)

    def get_joint_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
