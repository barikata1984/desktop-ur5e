"""CLI: run a closed-loop MPC push.

``tyro`` derives the CLI from the :class:`SimConfig` dataclass, so every nested
field is overridable, e.g.::

    python scripts/run_push.py --push.y-goal 0.85 --mpc.v-max 0.1
"""

from __future__ import annotations

import tyro

from ur5e_sim.pushing.config import SimConfig
from ur5e_sim.pushing.task import run
from ur5e_sim.pushing.viz.plots import plot_results


def main() -> None:
    cfg = tyro.cli(SimConfig)
    log, trial_dir = run(cfg)
    plot_results(log, trial_dir, cfg)


if __name__ == "__main__":
    main()
