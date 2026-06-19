from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco


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


def step_model(model: mujoco.MjModel, data: mujoco.MjData, steps: int) -> None:
    for _ in range(steps):
        mujoco.mj_step(model, data)


def get_object_names(
    model: mujoco.MjModel, object_type: mujoco.mjtObj, count: int
) -> tuple[str, ...]:
    names: list[str] = []
    for index in range(count):
        name = mujoco.mj_id2name(model, object_type, index)
        names.append(name if name is not None else f"<{object_type.name}:{index}>")
    return tuple(names)


def get_named_object_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int | None:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id == -1:
        return None
    return int(object_id)


def get_home_qpos(model: mujoco.MjModel) -> tuple[float, ...] | None:
    key_id = get_named_object_id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id is None:
        return None
    return tuple(float(value) for value in model.key_qpos[key_id][: model.nq])


def reset_to_home(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    key_id = get_named_object_id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id is None:
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        return False

    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    return True


def apply_joint_overrides(
    model: mujoco.MjModel, data: mujoco.MjData, overrides: dict[str, float]
) -> None:
    for joint_name, joint_value in overrides.items():
        joint_id = get_named_object_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id is None:
            raise ValueError(f"Unknown joint name: {joint_name}")
        qpos_address = int(model.jnt_qposadr[joint_id])
        data.qpos[qpos_address] = joint_value
    mujoco.mj_forward(model, data)


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
