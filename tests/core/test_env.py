"""Core SimEnv tests: loading, reset, step."""

from pathlib import Path

import numpy as np
import pytest

from ur5e_sim.core.env import SimEnv

REPO_ROOT = Path(__file__).resolve().parents[2]
PUSH_SCENE = str(REPO_ROOT / "scenes" / "tasks" / "push.xml")
IDENTIFICATION_SCENE = str(REPO_ROOT / "scenes" / "tasks" / "identification.xml")


class TestSimEnvLoadsPushScene:
    def test_loads_push_scene(self) -> None:
        env = SimEnv(PUSH_SCENE)
        assert env.model is not None
        assert env.data is not None

    def test_reset_to_ready_keyframe(self) -> None:
        env = SimEnv(PUSH_SCENE)
        env.reset(keyframe="ready")
        # After reset, qpos should be set (not all zeros for a keyframe with joint values)
        assert env.data.qpos is not None

    def test_step_no_nan(self) -> None:
        env = SimEnv(PUSH_SCENE)
        env.reset(keyframe="ready")
        env.step(100)
        env.forward()
        assert not np.any(np.isnan(env.data.qpos))
        assert not np.any(np.isnan(env.data.qvel))


class TestSimEnvLoadsIdentificationScene:
    def test_loads_identification_scene(self) -> None:
        env = SimEnv(IDENTIFICATION_SCENE)
        assert env.model is not None
        assert env.data is not None

    def test_reset_to_home_keyframe(self) -> None:
        env = SimEnv(IDENTIFICATION_SCENE)
        env.reset(keyframe="home")
        assert env.data.qpos is not None

    def test_step_no_nan(self) -> None:
        env = SimEnv(IDENTIFICATION_SCENE)
        env.reset(keyframe="home")
        env.step(100)
        env.forward()
        assert not np.any(np.isnan(env.data.qpos))
        assert not np.any(np.isnan(env.data.qvel))


class TestSimEnvErrors:
    def test_unknown_keyframe_raises(self) -> None:
        env = SimEnv(PUSH_SCENE)
        with pytest.raises(ValueError, match="Unknown keyframe"):
            env.reset(keyframe="nonexistent_keyframe")

    def test_nonexistent_scene_raises(self) -> None:
        with pytest.raises(Exception):
            SimEnv("/nonexistent/path/scene.xml")
