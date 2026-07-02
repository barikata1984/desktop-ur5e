"""Grid search over excitation trajectory optimization parameters.

Sweeps EE velocity, joint velocity, and joint acceleration limits with
FT offset (16-column regressor) enabled. Workspace is read from the model's
workspace_region_geom (fixed across all conditions).

Parallelizes at the condition level: each condition runs 8 sequential restarts,
up to 14 conditions run simultaneously (one per physical core minus OS headroom).
"""

from __future__ import annotations

import itertools
import json
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from math import pi
from pathlib import Path

import numpy as np

from ur5e_sim.core import names
from ur5e_sim.core.env import get_workspace_bounds
from ur5e_sim.core.layout import DofLayout
from ur5e_sim.core.model_builder import build_ur5e_model
from ur5e_sim.identification.collision import CollisionConfig
from ur5e_sim.identification.constraints import JointLimits
from ur5e_sim.identification.io import (
    result_to_trajectory,
    save_optimization_result,
    save_trajectory_json,
)
from ur5e_sim.identification.optimizer import (
    EarlyStopConfig,
    ExcitationOptimizer,
    OptimizerConfig,
    WandbConfig,
)
from ur5e_sim.identification.workspace import EeVelocityConfig, WorkspaceConstraintConfig

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RESULTS_DIR = _REPO_ROOT / "results" / "grid_search"

MAX_CONDITION_WORKERS = 24


@dataclass(frozen=True)
class Condition:
    ws_x_half: float
    ws_y_half: float
    ee_vel: float
    dq: float
    ddq: float

    @property
    def label(self) -> str:
        def _pi_label(val: float) -> str:
            ratio = val / pi
            if abs(ratio - 0.5) < 1e-6:
                return "halfpi"
            if abs(ratio - 1.0) < 1e-6:
                return "1pi"
            if abs(ratio - 2.0) < 1e-6:
                return "2pi"
            if abs(ratio - 4.0) < 1e-6:
                return "4pi"
            return f"{val:.2f}"

        ws = f"wsx{int(self.ws_x_half * 100):02d}_y{int(self.ws_y_half * 100):02d}"
        ee = f"ee{int(self.ee_vel * 100):03d}"
        dq = f"dq{_pi_label(self.dq)}"
        ddq = f"ddq{_pi_label(self.ddq)}"
        return f"{ws}_{ee}_{dq}_{ddq}"


WS_CONFIGS = [
    (0.25, 0.35),
    (0.30, 0.30),
    (0.25, 0.30),
]
EE_VELS = [0.25, 0.20]
DQ_LIMITS = [pi / 2, pi]
DDQ_LIMITS = [2 * pi, 4 * pi]


def build_conditions() -> list[Condition]:
    return [
        Condition(ws_x, ws_y, ee, dq, ddq)
        for (ws_x, ws_y), ee, dq, ddq in itertools.product(
            WS_CONFIGS, EE_VELS, DQ_LIMITS, DDQ_LIMITS
        )
    ]


def _run_condition_worker(label: str, cond_dict: dict, output_dir: str) -> dict:
    """Top-level function for ProcessPoolExecutor (must be picklable)."""
    cond = Condition(**cond_dict)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "result.json"
    traj_path = out / "traj.json"

    model, data = build_ur5e_model(payload_xml="scenes/objects/payload_flat.xml")
    layout = DofLayout.from_model(model)
    q0 = layout.arm(data.qpos).copy()
    ws_lower, ws_upper = get_workspace_bounds(model, data)
    ws_center = (ws_lower + ws_upper) / 2
    z_half = (ws_upper[2] - ws_lower[2]) / 2
    box_lower = ws_center - np.array([cond.ws_x_half, cond.ws_y_half, z_half])
    box_upper = ws_center + np.array([cond.ws_x_half, cond.ws_y_half, z_half])

    opt_config = OptimizerConfig(
        num_joints=6,
        num_harmonics=5,
        base_freq=0.2,
        duration=5.0,
        fps=100.0,
        q0=q0,
        subsample_factor=5,
        n_monte_carlo=8,
        max_iter_per_start=150,
        objective_type="condition_number",
        seed=42,
        joint_limits=JointLimits(
            dq_max=np.full(6, cond.dq),
            ddq_max=np.full(6, cond.ddq),
        ),
        collision_config=CollisionConfig(),
        payload_workspace_config=WorkspaceConstraintConfig(
            box_lower=box_lower, box_upper=box_upper
        ),
        body_name=names.PAYLOAD_BODY,
        site_name=names.FT_SITE,
        ee_velocity_config=EeVelocityConfig(max_linear_velocity=cond.ee_vel),
        enable_velocity_constraint=False,
        enable_acceleration_constraint=True,
        use_fourier_bounds=True,
        with_ft_offset=True,
        ft_offset_column_scale=True,
        n_workers=1,
    )

    optimizer = ExcitationOptimizer(config=opt_config, model=model, data=data)
    result = optimizer.optimize(
        wandb_config=WandbConfig(enabled=False),
        early_stop_config=EarlyStopConfig(enabled=False),
    )

    save_optimization_result(result, result_path)
    trajectory = result_to_trajectory(result)
    save_trajectory_json(
        trajectory,
        traj_path,
        condition_number=result.condition_number,
        source=result_path.name,
    )

    return {
        "label": label,
        "condition_number": result.condition_number,
        "feasible": result.feasible,
        "wall_time": result.wall_time,
        "constraint_margins": result.constraint_margins,
        "trajectory_stats": result.trajectory_stats,
    }


def main() -> None:
    conditions = build_conditions()
    print(f"Grid search: {len(conditions)} conditions, {MAX_CONDITION_WORKERS} parallel workers")
    for i, c in enumerate(conditions):
        print(f"  [{i + 1:2d}] {c.label}")
    print(flush=True)

    trial_dir = _RESULTS_DIR / "trial_bias"
    trial_dir.mkdir(parents=True, exist_ok=True)

    futures_map: dict = {}
    summaries: dict[str, dict] = {}

    with ProcessPoolExecutor(max_workers=MAX_CONDITION_WORKERS) as pool:
        for cond in conditions:
            out_dir = trial_dir / cond.label
            cond_dict = {
                "ws_x_half": cond.ws_x_half,
                "ws_y_half": cond.ws_y_half,
                "ee_vel": cond.ee_vel,
                "dq": cond.dq,
                "ddq": cond.ddq,
            }
            fut = pool.submit(_run_condition_worker, cond.label, cond_dict, str(out_dir))
            futures_map[fut] = cond.label

        done_count = 0
        for fut in as_completed(futures_map):
            done_count += 1
            label = futures_map[fut]
            try:
                summary = fut.result()
                summaries[label] = summary
                cond_val = summary["condition_number"]
                feas = summary["feasible"]
                wt = summary["wall_time"]
                print(
                    f"  [{done_count}/{len(conditions)}] {label}: "
                    f"cond={cond_val:.3f}, feasible={feas}, time={wt:.0f}s",
                    flush=True,
                )
            except Exception as e:
                summaries[label] = {"label": label, "error": str(e)}
                print(f"  [{done_count}/{len(conditions)}] {label}: ERROR: {e}", flush=True)
                traceback.print_exc()

    ordered = [summaries.get(c.label, {"label": c.label, "error": "missing"}) for c in conditions]

    summary_path = trial_dir / "_summary.json"
    with open(summary_path, "w") as f:
        json.dump(ordered, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"Grid search complete. Summary: {summary_path}")
    print(f"{'=' * 60}")

    valid = [s for s in ordered if s.get("condition_number") and s["condition_number"] < 1e10]
    valid.sort(key=lambda s: s["condition_number"])
    print(f"\nValid results ({len(valid)}/{len(ordered)}):")
    print(f"{'Label':<45} {'cond':>8} {'feasible':>8}")
    print("-" * 65)
    for s in valid:
        print(f"{s['label']:<45} {s['condition_number']:8.3f} {str(s['feasible']):>8}")


if __name__ == "__main__":
    main()
