from __future__ import annotations

from typing import Protocol

import mujoco
import numpy as np


class Sensor(Protocol):
    def read(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray: ...


class FTSensor:
    """Read a tool0 force/torque sensor pair as a [torque; force] wrench."""

    def __init__(
        self,
        model: mujoco.MjModel,
        force_name: str = "ft_force",
        torque_name: str = "ft_torque",
    ):
        force_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, force_name)
        torque_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, torque_name)
        if force_sid < 0 or torque_sid < 0:
            raise ValueError(f"FT sensors not found: {force_name!r}, {torque_name!r}")
        self.force_adr = int(model.sensor_adr[force_sid])
        self.torque_adr = int(model.sensor_adr[torque_sid])

    def read(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        force = np.array(data.sensordata[self.force_adr : self.force_adr + 3], dtype=np.float64)
        torque = np.array(data.sensordata[self.torque_adr : self.torque_adr + 3], dtype=np.float64)
        return np.concatenate((torque, force))


class ContactSensor:
    """Read the net contact force between pad geoms and a target geom."""

    def __init__(self, pad_geom_ids: list[int], target_geom_id: int):
        self.pad_set = set(pad_geom_ids)
        self.target_geom_id = target_geom_id

    def read(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        total = np.zeros(3)
        for i in range(data.ncon):
            c = data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if (g1 in self.pad_set and g2 == self.target_geom_id) or (
                g1 == self.target_geom_id and g2 in self.pad_set
            ):
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, force)
                total += force[:3]
        return total
