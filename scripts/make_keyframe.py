"""CLI: recompute the 'ready' keyframe via 6-DOF tool-down IK.

Prints the keyframe qpos/ctrl strings for the push model's 'ready' keyframe
(defined in ur5e_sim.pushing.scene.build_push_model).
"""

from __future__ import annotations

from ur5e_sim.pushing.keyframe import main

if __name__ == "__main__":
    main()
