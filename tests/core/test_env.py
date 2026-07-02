"""Core model-construction tests: builders, keyframes, and stepping.

The direct-XML task scenes are gone; these tests exercise the programmatic
builders (the canonical construction path) plus keyframe reset behavior.
"""

import mujoco
import numpy as np
import pytest

from ur5e_sim.core.env import SimEnv, reset_to_home
from ur5e_sim.core.model_builder import build_ur5e_model
from ur5e_sim.pushing.scene import build_push_model


class TestPushModel:
    def test_builds(self) -> None:
        m, d = build_push_model()
        assert m is not None
        assert d is not None

    def test_ready_keyframe_present(self) -> None:
        m, _ = build_push_model()
        assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "ready") >= 0

    def test_step_no_nan(self) -> None:
        m, d = build_push_model()  # already reset to 'ready' and forwarded
        for _ in range(100):
            mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        assert not np.any(np.isnan(d.qpos))
        assert not np.any(np.isnan(d.qvel))


class TestUR5eModel:
    def test_builds_and_resets_home(self) -> None:
        m, d = build_ur5e_model()
        # build_ur5e_model resets to 'home'; reset_to_home should find the keyframe.
        assert reset_to_home(m, d) is True

    def test_step_no_nan(self) -> None:
        m, d = build_ur5e_model()  # already reset to 'home' and forwarded
        for _ in range(100):
            mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        assert not np.any(np.isnan(d.qpos))
        assert not np.any(np.isnan(d.qvel))


class TestSimEnvErrors:
    def test_nonexistent_scene_raises(self) -> None:
        with pytest.raises(Exception):
            SimEnv("/nonexistent/path/scene.xml")
