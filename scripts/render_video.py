"""CLI: render the polished 2x2 grid video for a trial directory.

Usage::

    python scripts/render_video.py <trial_dir> [scene.xml]
"""

from __future__ import annotations

import sys
from pathlib import Path

from ur5e_sim.pushing import paths
from ur5e_sim.pushing.viz.grid_video import render_grid_video


def main() -> None:
    if len(sys.argv) < 2 or not (Path(sys.argv[1]) / "data.npz").exists():
        sys.exit("usage: python scripts/render_video.py <trial_dir> [scene.xml]")
    trial = Path(sys.argv[1])
    scene = sys.argv[2] if len(sys.argv) > 2 else paths.scene_path()
    render_grid_video(trial, scene)


if __name__ == "__main__":
    main()
