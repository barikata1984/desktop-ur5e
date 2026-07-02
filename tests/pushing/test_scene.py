"""Quick verification of the push model.

Builds the model via build_push_model (reset to the 'ready' keyframe),
steps the simulation, and checks key positions to verify scene integrity.
"""

import mujoco
import numpy as np

from ur5e_sim.pushing.scene import build_push_model


def test_scene_loads_and_is_stable():
    """The model builds, resets to 'ready', and the arm/slider stay put for 1 s."""
    m, d = build_push_model()  # already reset to 'ready' and forwarded
    eef_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    slider_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "slider_center")

    eef0 = d.site_xpos[eef_id].copy()
    slider0 = d.site_xpos[slider_id].copy()

    for _ in range(int(1.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)

    assert np.linalg.norm(d.site_xpos[eef_id] - eef0) < 0.01
    assert np.linalg.norm(d.site_xpos[slider_id] - slider0) < 0.01
