"""CLI: recompute the 'ready' keyframe via 6-DOF tool-down IK.

Prints the keyframe qpos/ctrl strings to paste into scenes/tasks/push.xml.
"""

from __future__ import annotations

from ur5e_sim.pushing.keyframe import main

if __name__ == "__main__":
    main()
