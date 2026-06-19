from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ur5e_sim.core.env import SimEnv
from ur5e_sim.core.renderer import FrameRenderer
from ur5e_sim.core.robot import UR5eRobot
from ur5e_sim.core.sensors import Sensor


@dataclass
class RunConfig:
    dt: float = 0.002
    render_every: int = 1


class SimRunner:
    """Composable read-control-step-log-render loop.

    The controller and the termination predicate are the extension points: the
    controller maps the assembled state dict to a ctrl vector, and termination
    decides when the loop stops. This keeps the loop agnostic to the task (push
    phases, trajectory playback, etc.).
    """

    def __init__(
        self,
        env: SimEnv,
        robot: UR5eRobot,
        controller: Any,
        sensors: dict[str, Sensor],
        logger: Any,
        renderer: FrameRenderer | None = None,
        config: RunConfig | None = None,
    ):
        self.env = env
        self.robot = robot
        self.controller = controller
        self.sensors = sensors
        self.logger = logger
        self.renderer = renderer
        self.config = config or RunConfig()

    def _read_state(self, step: int) -> dict[str, Any]:
        state: dict[str, Any] = {
            "step": step,
            "time": float(self.env.data.time),
            "qpos": self.robot.joint_positions(),
            "qvel": self.robot.joint_velocities(),
        }
        for name, sensor in self.sensors.items():
            state[name] = sensor.read(self.env.model, self.env.data)
        return state

    def run(self, termination: Callable[[dict[str, Any]], bool], path: Path) -> Path:
        substeps = max(1, round(self.config.dt / self.env.model.opt.timestep))
        step = 0
        while True:
            self.env.forward()
            state = self._read_state(step)

            if termination(state):
                self.logger.record(**state)
                break

            ctrl = self.controller.compute_control(state)
            self.robot.set_ctrl(ctrl)
            self.env.step(substeps)

            self.logger.record(**state)

            if self.renderer is not None and step % self.config.render_every == 0:
                self.renderer.capture(self.env.data)

            step += 1

        return self.logger.save(path)
