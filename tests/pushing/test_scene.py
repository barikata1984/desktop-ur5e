"""Quick verification of the push scene.

Loads the model, resets to the 'ready' keyframe, steps the simulation,
and checks key positions to verify scene integrity.
"""

import mujoco
import numpy as np

from ur5e_sim.pushing import paths


def test_scene_loads_and_is_stable():
    """The scene loads, resets to 'ready', and the arm/slider stay put for 1 s."""
    m = mujoco.MjModel.from_xml_path(paths.scene_path())
    d = mujoco.MjData(m)
    key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "ready")
    eef_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    slider_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "slider_center")

    mujoco.mj_resetDataKeyframe(m, d, key_id)
    mujoco.mj_forward(m, d)
    eef0 = d.site_xpos[eef_id].copy()
    slider0 = d.site_xpos[slider_id].copy()

    for _ in range(int(1.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)

    assert np.linalg.norm(d.site_xpos[eef_id] - eef0) < 0.01
    assert np.linalg.norm(d.site_xpos[slider_id] - slider0) < 0.01
