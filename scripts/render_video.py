"""CLI: render the polished 2x2 grid video for a trial directory.

Usage::

    python scripts/render_video.py <trial_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

from ur5e_sim.pushing.viz.grid_video import render_grid_video


def main() -> None:
    if len(sys.argv) < 2 or not (Path(sys.argv[1]) / "data.npz").exists():
        sys.exit("usage: python scripts/render_video.py <trial_dir>")
    trial = Path(sys.argv[1])
    render_grid_video(trial)


if __name__ == "__main__":
    main()
