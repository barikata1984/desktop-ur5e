from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import mujoco
import numpy as np
from scipy.optimize import Bounds, minimize

from ur5e_sim.core import names
from ur5e_sim.identification.collision import CollisionConfig
from ur5e_sim.identification.constraints import (
    JointLimits,
    _TrajectoryCache,
    build_scipy_constraints,
    compute_fourier_bounds,
)
from ur5e_sim.identification.objective import (
    condition_number_objective,
    d_optimal_with_cond,
    evaluate_full_resolution,
)
from ur5e_sim.identification.workspace import EeVelocityConfig, WorkspaceConstraintConfig

log = logging.getLogger(__name__)


@dataclass
class OptimizerConfig:
    num_joints: int = 6
    num_harmonics: int = 5
    base_freq: float = 0.1
    duration: float = 10.0
    fps: float = 100.0
    q0: np.ndarray | None = None
    subsample_factor: int = 10
    n_monte_carlo: int = 20
    max_iter_per_start: int = 200
    objective_type: str = "d_optimal"  # "d_optimal" or "condition_number"
    optimizer_method: str = "SLSQP"
    ftol: float = 1e-6
    seed: int = 42
    joint_limits: JointLimits | None = None
    workspace_config: WorkspaceConstraintConfig | None = None
    payload_workspace_config: WorkspaceConstraintConfig | None = None
    collision_config: CollisionConfig | None = None
    ee_velocity_config: EeVelocityConfig | None = None
    enable_velocity_constraint: bool = True
    enable_acceleration_constraint: bool = True
    use_fourier_bounds: bool = False
    with_ft_offset: bool = False
    ft_offset_column_scale: bool = True
    n_workers: int = 1  # Number of parallel worker processes (1 = sequential)
    payload_xml: str | None = None  # Payload MJCF for worker model construction (mirrors CLI)

    def __post_init__(self) -> None:
        if self.joint_limits is None:
            self.joint_limits = JointLimits()


@dataclass
class EarlyStopConfig:
    """Early stopping configuration for multi-start optimization."""

    enabled: bool = False
    patience: int = 5
    min_improvement: float = 1e-3
    target_cond: float = 0.0  # Stop when condition number <= this (0 = disabled)


@dataclass
class WandbConfig:
    """Weights & Biases logging configuration."""

    enabled: bool = False
    project: str = "ur5e-excitation"
    run_name: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class OptimizationResult:
    x_opt: np.ndarray
    condition_number: float
    a_opt: np.ndarray
    b_opt: np.ndarray
    q0: np.ndarray
    config: OptimizerConfig
    n_evaluations: int
    wall_time: float
    n_restarts: int
    best_start_index: int
    # Diagnostics (populated after optimization)
    constraint_margins: dict[str, float] = field(default_factory=dict)
    feasible: bool = False
    trajectory_stats: dict[str, float] = field(default_factory=dict)
    restart_history: list[dict] = field(default_factory=list)


@dataclass
class _RestartResult:
    """Result from a single restart, returned by worker processes."""

    restart_index: int
    x: np.ndarray
    fun: float
    condition_number: float
    n_func_evals: int
    n_iters: int
    wall_time: float
    named_margins: dict[str, float]
    iter_logs: list[dict[str, float]]

    @property
    def constraint_margin(self) -> float:
        return min(self.named_margins.values())

    @property
    def feasible(self) -> bool:
        return self.constraint_margin >= 0


def _evaluate_margins(x: np.ndarray, constraints: list[dict]) -> dict[str, float]:
    """Evaluate every constraint once, keyed by its 'name'."""
    return {c.get("name", f"constraint_{i}"): float(c["fun"](x)) for i, c in enumerate(constraints)}


def _config_to_wandb_dict(cfg: OptimizerConfig) -> dict:
    """Convert OptimizerConfig to a flat dict suitable for wandb.config."""
    d: dict = {
        "num_joints": cfg.num_joints,
        "num_harmonics": cfg.num_harmonics,
        "base_freq": cfg.base_freq,
        "duration": cfg.duration,
        "fps": cfg.fps,
        "subsample_factor": cfg.subsample_factor,
        "n_monte_carlo": cfg.n_monte_carlo,
        "max_iter_per_start": cfg.max_iter_per_start,
        "objective_type": cfg.objective_type,
        "optimizer_method": cfg.optimizer_method,
        "ftol": cfg.ftol,
        "seed": cfg.seed,
        "n_decision_vars": 2 * cfg.num_joints * cfg.num_harmonics,
    }
    if cfg.q0 is not None:
        d["q0"] = cfg.q0.tolist()
    if cfg.joint_limits is not None:
        d["q_min"] = cfg.joint_limits.q_min.tolist()
        d["q_max"] = cfg.joint_limits.q_max.tolist()
        d["dq_max"] = cfg.joint_limits.dq_max.tolist()
        d["ddq_max"] = cfg.joint_limits.ddq_max.tolist()
    if cfg.workspace_config is not None:
        d["max_displacement"] = cfg.workspace_config.max_displacement
    d["collision_enabled"] = cfg.collision_config is not None
    d["payload_workspace_enabled"] = cfg.payload_workspace_config is not None
    d["use_fourier_bounds"] = cfg.use_fourier_bounds
    d["enable_velocity_constraint"] = cfg.enable_velocity_constraint
    d["enable_acceleration_constraint"] = cfg.enable_acceleration_constraint
    d["with_ft_offset"] = cfg.with_ft_offset
    d["ft_offset_column_scale"] = cfg.ft_offset_column_scale
    if cfg.ee_velocity_config is not None:
        d["ee_max_linear_velocity"] = cfg.ee_velocity_config.max_linear_velocity
    return d


def _build_cache_and_constraints_static(
    cfg: OptimizerConfig,
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> tuple[_TrajectoryCache, list[dict]]:
    """Build trajectory cache and scipy constraints from config (no instance needed)."""
    q0 = np.asarray(cfg.q0, dtype=np.float64)
    cache = _TrajectoryCache(
        num_joints=cfg.num_joints,
        num_harmonics=cfg.num_harmonics,
        base_freq=cfg.base_freq,
        duration=cfg.duration,
        fps=cfg.fps,
        q0=q0,
    )
    constraints = build_scipy_constraints(
        cache,
        cfg.joint_limits,
        workspace_config=cfg.workspace_config,
        collision_config=cfg.collision_config,
        model=model,
        data=data,
        payload_workspace_config=cfg.payload_workspace_config,
        payload_body_name=names.PAYLOAD_BODY,
        ee_velocity_config=cfg.ee_velocity_config,
        site_name=names.FT_SITE,
        enable_velocity_constraint=cfg.enable_velocity_constraint,
        enable_acceleration_constraint=cfg.enable_acceleration_constraint,
    )
    return cache, constraints


def _compute_fourier_bounds_static(cfg: OptimizerConfig) -> Bounds | None:
    """Compute scipy Bounds from analytical Fourier velocity/acceleration bounds."""
    if not cfg.use_fourier_bounds or cfg.joint_limits is None:
        return None
    limits = cfg.joint_limits
    # Pass dq_max/ddq_max only if the corresponding constraint is active
    dq = limits.dq_max if cfg.enable_velocity_constraint or cfg.use_fourier_bounds else None
    ddq = limits.ddq_max if cfg.enable_acceleration_constraint else None
    if dq is None and ddq is None:
        return None
    upper = compute_fourier_bounds(
        num_joints=cfg.num_joints,
        num_harmonics=cfg.num_harmonics,
        base_freq=cfg.base_freq,
        duration=cfg.duration,
        dq_max=dq,
        ddq_max=ddq,
    )
    return Bounds(lb=-upper, ub=upper)


def _make_objective(
    cfg: OptimizerConfig,
    cache: _TrajectoryCache,
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> tuple[Callable[[np.ndarray], float], list[float], list[float]]:
    """Build the scipy objective and its latest obj/cond telemetry boxes."""
    use_d_optimal = cfg.objective_type == "d_optimal"
    column_scale = cfg.ft_offset_column_scale and cfg.with_ft_offset
    latest_obj: list[float] = [float("inf")]
    latest_cond: list[float] = [float("inf")]

    def objective(x: np.ndarray) -> float:
        if use_d_optimal:
            obj_val, cond_val = d_optimal_with_cond(
                x,
                cache,
                model,
                data,
                names.PAYLOAD_BODY,
                cfg.subsample_factor,
                with_ft_offset=cfg.with_ft_offset,
                column_scale=column_scale,
                site_name=names.FT_SITE,
            )
        else:
            obj_val = condition_number_objective(
                x,
                cache,
                model,
                data,
                names.PAYLOAD_BODY,
                cfg.subsample_factor,
                with_ft_offset=cfg.with_ft_offset,
                column_scale=column_scale,
                site_name=names.FT_SITE,
            )
            cond_val = obj_val
        latest_obj[0] = obj_val
        latest_cond[0] = cond_val
        return obj_val

    return objective, latest_obj, latest_cond


def _final_condition_number(
    x: np.ndarray,
    result_fun: float,
    cfg: OptimizerConfig,
    cache: _TrajectoryCache,
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> float:
    """Re-evaluate the condition number for reporting after minimize."""
    if cfg.objective_type == "d_optimal":
        _, cond = d_optimal_with_cond(
            x,
            cache,
            model,
            data,
            names.PAYLOAD_BODY,
            cfg.subsample_factor,
            with_ft_offset=cfg.with_ft_offset,
            column_scale=cfg.ft_offset_column_scale and cfg.with_ft_offset,
            site_name=names.FT_SITE,
        )
        return cond
    return float(result_fun)


def _execute_restart(
    x0: np.ndarray,
    restart_index: int,
    cfg: OptimizerConfig,
    cache: _TrajectoryCache,
    constraints: list[dict],
    fourier_bounds: Bounds | None,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    iter_callback: Callable[[dict], None] | None = None,
) -> _RestartResult:
    """Run one restart (minimize + diagnostics), shared by sequential and parallel paths.

    ``iter_callback`` is invoked with each per-iteration log entry (sequential uses it for
    live wandb logging; the parallel worker passes None and replays logs after completion).
    """
    objective, latest_obj, latest_cond = _make_objective(cfg, cache, model, data)

    iter_logs: list[dict[str, float]] = []
    iter_count = [0]
    t0 = time.perf_counter()

    def _callback(xk: np.ndarray) -> None:
        iter_count[0] += 1
        entry = {
            "iter/condition_number": latest_cond[0],
            "iter/objective": latest_obj[0],
            "iter/restart_index": restart_index,
            "iter/iter_in_restart": iter_count[0],
            "iter/wall_time": time.perf_counter() - t0,
        }
        iter_logs.append(entry)
        if iter_callback is not None:
            iter_callback(entry)

    result = minimize(
        objective,
        x0,
        method=cfg.optimizer_method,
        bounds=fourier_bounds,
        constraints=constraints,
        options={"maxiter": cfg.max_iter_per_start, "ftol": cfg.ftol},
        callback=_callback,
    )
    wall_time = time.perf_counter() - t0

    cond = _final_condition_number(result.x, result.fun, cfg, cache, model, data)
    named_margins = _evaluate_margins(result.x, constraints)

    return _RestartResult(
        restart_index=restart_index,
        x=result.x.copy(),
        fun=float(result.fun),
        condition_number=cond,
        n_func_evals=result.nfev,
        n_iters=iter_count[0],
        wall_time=wall_time,
        named_margins=named_margins,
        iter_logs=iter_logs,
    )


def _run_single_restart(
    config: OptimizerConfig,
    x0: np.ndarray,
    restart_index: int,
) -> _RestartResult:
    """Run a single restart in a worker process.

    Builds its own MjModel/MjData via MjSpec.attach() to ensure process safety.
    Mirrors the CLI's model construction (payload_xml) so that body/site names
    referenced by ``config`` (e.g. ``payload_workspace_config``) resolve correctly.
    """
    from ur5e_sim.core.model_builder import build_ur5e_model

    model, data = build_ur5e_model(payload_xml=config.payload_xml)

    cache, constraints = _build_cache_and_constraints_static(config, model, data)
    fourier_bounds = _compute_fourier_bounds_static(config)

    return _execute_restart(
        x0, restart_index, config, cache, constraints, fourier_bounds, model, data
    )


class _RestartAggregator:
    """Shared bookkeeping for sequential and parallel restart loops."""

    def __init__(
        self,
        cfg: OptimizerConfig,
        early_stop: EarlyStopConfig,
        wandb_config: WandbConfig | None,
        wb_group: str,
        wb_enabled: bool,
    ) -> None:
        self.cfg = cfg
        self.early_stop = early_stop
        self.wandb_config = wandb_config
        self.wb_group = wb_group
        self.wb_enabled = wb_enabled
        self.use_d_optimal = cfg.objective_type == "d_optimal"

        self.best_x: np.ndarray | None = None
        self.best_cond = float("inf")
        self.best_idx = 0
        self.total_evals = 0
        self.actual_restarts = 0
        self.restart_summaries: list[dict] = []
        self.patience_counter = 0
        self._improved = False

    def process(self, rr: _RestartResult, wb_run: object | None = None) -> None:
        """Record one restart result: print, update best, append summary, log wandb."""
        cfg = self.cfg
        self.actual_restarts += 1
        self.total_evals += rr.n_func_evals

        if self.use_d_optimal:
            print(
                f"  start {rr.restart_index + 1}/{cfg.n_monte_carlo}: "
                f"cond={rr.condition_number:.4f}  D-opt={rr.fun:.4f}  "
                f"margin={rr.constraint_margin:.4f}  "
                f"feasible={rr.feasible}  ({rr.wall_time:.1f}s)",
                flush=True,
            )
        else:
            print(
                f"  start {rr.restart_index + 1}/{cfg.n_monte_carlo}: "
                f"cond={rr.condition_number:.4f}  margin={rr.constraint_margin:.4f}  "
                f"feasible={rr.feasible}  ({rr.wall_time:.1f}s)",
                flush=True,
            )

        improved = rr.condition_number < self.best_cond
        if improved:
            self.best_cond = rr.condition_number
            self.best_x = rr.x.copy()
            self.best_idx = rr.restart_index
        self._improved = improved

        self.restart_summaries.append(
            {
                "restart_index": rr.restart_index,
                "condition_number": rr.condition_number,
                "feasible": rr.feasible,
                "constraint_margin": rr.constraint_margin,
                "named_margins": rr.named_margins,
                "wall_time": rr.wall_time,
                "n_func_evals": rr.n_func_evals,
                "iter_logs": rr.iter_logs,
            }
        )

        if not self.wb_enabled:
            return

        # Sequential passes a live run (already logged per-iteration); parallel creates
        # the run here and batch-replays the collected iteration logs.
        if wb_run is None:
            wb_run = ExcitationOptimizer._init_wandb_restart_run(
                self.wandb_config, cfg, rr.restart_index, self.wb_group
            )
            if wb_run is None:
                return
            for step, iter_log in enumerate(rr.iter_logs, 1):
                wb_run.log(
                    {
                        "condition_number": iter_log["iter/condition_number"],
                        "objective": iter_log["iter/objective"],
                        "wall_time": iter_log["iter/wall_time"],
                    },
                    step=step,
                )

        restart_summary: dict = {
            "condition_number": rr.condition_number,
            "objective": rr.fun,
            "constraint_margin_min": rr.constraint_margin,
            "feasible": int(rr.feasible),
            "n_func_evals": rr.n_func_evals,
            "n_iters": rr.n_iters,
            "wall_time_s": rr.wall_time,
        }
        for name, mval in rr.named_margins.items():
            restart_summary[f"margin/{name}"] = mval
        wb_run.summary.update(restart_summary)
        wb_run.finish()

    def should_stop(self, rr: _RestartResult) -> str | None:
        """Return an early-stop reason (and print it) or None. Call after ``process``."""
        es = self.early_stop
        if not es.enabled:
            return None
        if es.target_cond > 0 and rr.feasible and rr.condition_number <= es.target_cond:
            print(
                f"  Early stop: target cond {es.target_cond} reached "
                f"(cond={rr.condition_number:.4f}, feasible=True)",
                flush=True,
            )
            return "target_cond"
        if self._improved and (self.best_cond < float("inf")):
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        if self.patience_counter >= es.patience:
            print(
                f"  Early stop: no improvement for {es.patience} restarts "
                f"(best={self.best_cond:.4f})",
                flush=True,
            )
            return "patience"
        return None


class ExcitationOptimizer:
    """Multi-start SLSQP optimizer for excitation trajectory design."""

    def __init__(
        self,
        config: OptimizerConfig,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> None:
        self.config = config
        self.model = model
        self.data = data

    def _get_x_size(self) -> int:
        return 2 * self.config.num_joints * self.config.num_harmonics

    def _generate_random_x0(
        self, rng: np.random.Generator, bounds: Bounds | None = None
    ) -> np.ndarray:
        nj = self.config.num_joints
        nh = self.config.num_harmonics
        x = np.zeros(2 * nj * nh, dtype=np.float64)
        for k in range(nh):
            scale = 0.3 / (k + 1)
            x[k * nj : (k + 1) * nj] = rng.uniform(-scale, scale, size=nj)
            x[nj * nh + k * nj : nj * nh + (k + 1) * nj] = rng.uniform(-scale, scale, size=nj)
        if bounds is not None:
            x = np.clip(x, bounds.lb, bounds.ub)
        return x

    def _build_cache_and_constraints(self) -> tuple[_TrajectoryCache, list[dict]]:
        """Build trajectory cache and all scipy constraints from config."""
        return _build_cache_and_constraints_static(self.config, self.model, self.data)

    def _compute_fourier_bounds(self) -> Bounds | None:
        """Compute scipy Bounds from analytical Fourier velocity bounds."""
        return _compute_fourier_bounds_static(self.config)

    def _compute_trajectory_stats(self, x: np.ndarray, cache: _TrajectoryCache) -> dict[str, float]:
        """Compute trajectory-level statistics for diagnostics."""
        sample = cache.get(x)
        return {
            "q_max": float(np.max(np.abs(sample.position))),
            "dq_max": float(np.max(np.abs(sample.velocity))),
            "ddq_max": float(np.max(np.abs(sample.acceleration))),
            "dq_per_joint_max": np.max(np.abs(sample.velocity), axis=0).tolist(),
            "ddq_per_joint_max": np.max(np.abs(sample.acceleration), axis=0).tolist(),
        }

    def optimize(
        self,
        wandb_config: WandbConfig | None = None,
        early_stop_config: EarlyStopConfig | None = None,
    ) -> OptimizationResult:
        """Run multi-start optimisation and return the best result.

        Args:
            wandb_config: If provided with enabled=True, log metrics to W&B.
            early_stop_config: If provided with enabled=True, stop early when
                the global best condition number stops improving.
        """
        cfg = self.config
        fourier_bounds = self._compute_fourier_bounds()

        # Pre-generate ALL x0 vectors (deterministic regardless of n_workers)
        rng = np.random.default_rng(cfg.seed)
        all_x0 = [
            self._generate_random_x0(rng, bounds=fourier_bounds) for _ in range(cfg.n_monte_carlo)
        ]

        if cfg.n_workers > 1:
            return self._optimize_parallel(all_x0, wandb_config, early_stop_config)
        return self._optimize_sequential(all_x0, wandb_config, early_stop_config)

    @staticmethod
    def _init_wandb_restart_run(
        wandb_config: WandbConfig,
        cfg: OptimizerConfig,
        restart_index: int,
        group: str,
    ) -> object | None:
        """Create a wandb run for a single restart."""
        try:
            import wandb

            return wandb.init(
                project=wandb_config.project,
                name=f"restart-{restart_index}",
                group=group,
                tags=wandb_config.tags or None,
                config={**_config_to_wandb_dict(cfg), "restart_index": restart_index},
                reinit="finish_previous",
            )
        except Exception:
            log.warning("wandb init failed for restart %d", restart_index, exc_info=True)
            return None

    @staticmethod
    def _log_wandb_summary(
        wandb_config: WandbConfig,
        cfg: OptimizerConfig,
        group: str,
        opt_result: OptimizationResult,
    ) -> None:
        """Create a summary wandb run with final metrics."""
        try:
            import wandb

            run = wandb.init(
                project=wandb_config.project,
                name="summary",
                group=group,
                tags=wandb_config.tags or None,
                config=_config_to_wandb_dict(cfg),
                reinit="finish_previous",
            )
        except Exception:
            log.warning("wandb init failed for summary run", exc_info=True)
            return

        summary: dict = {
            "final/condition_number": opt_result.condition_number,
            "final/best_restart_index": opt_result.best_start_index,
            "final/total_restarts": opt_result.n_restarts,
            "final/total_func_evals": opt_result.n_evaluations,
            "final/wall_time_s": opt_result.wall_time,
            "final/feasible": int(opt_result.feasible),
        }
        for name, margin in opt_result.constraint_margins.items():
            summary[f"final/margin/{name}"] = margin
        for key, val in opt_result.trajectory_stats.items():
            if isinstance(val, list):
                for j, v in enumerate(val):
                    summary[f"final/traj/{key}_j{j}"] = v
            else:
                summary[f"final/traj/{key}"] = val
        run.summary.update(summary)
        run.finish()

    def _optimize_sequential(
        self,
        all_x0: list[np.ndarray],
        wandb_config: WandbConfig | None = None,
        early_stop_config: EarlyStopConfig | None = None,
    ) -> OptimizationResult:
        """Sequential restart loop (original behavior)."""
        cfg = self.config
        cache, constraints = self._build_cache_and_constraints()
        fourier_bounds = self._compute_fourier_bounds()

        wb_enabled = wandb_config is not None and wandb_config.enabled
        wb_group = wandb_config.run_name or f"opt-{int(time.time())}" if wb_enabled else ""
        es = early_stop_config or EarlyStopConfig()
        agg = _RestartAggregator(cfg, es, wandb_config, wb_group, wb_enabled)

        t0 = time.perf_counter()

        for i in range(cfg.n_monte_carlo):
            # Per-restart wandb run: created before minimize so metrics log LIVE.
            wb_run = None
            iter_callback: Callable[[dict], None] | None = None
            if wb_enabled:
                wb_run = self._init_wandb_restart_run(wandb_config, cfg, i, wb_group)
                if wb_run is not None:

                    def iter_callback(entry: dict, _run: object = wb_run) -> None:
                        _run.log(
                            {
                                "condition_number": entry["iter/condition_number"],
                                "objective": entry["iter/objective"],
                                "wall_time": entry["iter/wall_time"],
                            },
                            step=entry["iter/iter_in_restart"],
                        )

            rr = _execute_restart(
                all_x0[i],
                i,
                cfg,
                cache,
                constraints,
                fourier_bounds,
                self.model,
                self.data,
                iter_callback=iter_callback,
            )
            agg.process(rr, wb_run=wb_run)
            if agg.should_stop(rr):
                break

        wall_time = time.perf_counter() - t0
        opt_result = self._build_final_result(
            best_x=agg.best_x,
            best_cond=agg.best_cond,
            best_idx=agg.best_idx,
            total_evals=agg.total_evals,
            actual_restarts=agg.actual_restarts,
            wall_time=wall_time,
            cache=cache,
            constraints=constraints,
        )
        opt_result.restart_history = agg.restart_summaries
        if wb_enabled:
            self._log_wandb_summary(wandb_config, cfg, wb_group, opt_result)
        return opt_result

    def _optimize_parallel(
        self,
        all_x0: list[np.ndarray],
        wandb_config: WandbConfig | None = None,
        early_stop_config: EarlyStopConfig | None = None,
    ) -> OptimizationResult:
        """Parallel restart loop using ProcessPoolExecutor."""
        from concurrent.futures import ProcessPoolExecutor, as_completed

        cfg = self.config
        wb_enabled = wandb_config is not None and wandb_config.enabled
        wb_group = wandb_config.run_name or f"opt-{int(time.time())}" if wb_enabled else ""
        es = early_stop_config or EarlyStopConfig()
        agg = _RestartAggregator(cfg, es, wandb_config, wb_group, wb_enabled)

        t0 = time.perf_counter()
        print(f"  Parallel mode: {cfg.n_workers} workers", flush=True)

        # Submit one future per restart for fine-grained early stopping
        futures = {}
        with ProcessPoolExecutor(max_workers=cfg.n_workers) as executor:
            for i in range(cfg.n_monte_carlo):
                future = executor.submit(_run_single_restart, cfg, all_x0[i], i)
                futures[future] = i

            for future in as_completed(futures):
                rr = future.result()
                agg.process(rr)
                if agg.should_stop(rr):
                    # Cancel remaining futures (best-effort); break avoids CancelledError.
                    for f in futures:
                        f.cancel()
                    break

        wall_time = time.perf_counter() - t0

        # Build final result using local cache/constraints for diagnostics
        cache, constraints = self._build_cache_and_constraints()
        opt_result = self._build_final_result(
            best_x=agg.best_x,
            best_cond=agg.best_cond,
            best_idx=agg.best_idx,
            total_evals=agg.total_evals,
            actual_restarts=agg.actual_restarts,
            wall_time=wall_time,
            cache=cache,
            constraints=constraints,
        )
        opt_result.restart_history = agg.restart_summaries
        if wb_enabled:
            self._log_wandb_summary(wandb_config, cfg, wb_group, opt_result)
        return opt_result

    def _build_final_result(
        self,
        *,
        best_x: np.ndarray | None,
        best_cond: float,
        best_idx: int,
        total_evals: int,
        actual_restarts: int,
        wall_time: float,
        cache: _TrajectoryCache,
        constraints: list[dict],
    ) -> OptimizationResult:
        """Build OptimizationResult from optimization state."""
        cfg = self.config
        q0 = np.asarray(cfg.q0, dtype=np.float64)

        n = cfg.num_joints * cfg.num_harmonics
        a_opt = best_x[:n].reshape(cfg.num_joints, cfg.num_harmonics)
        b_opt = best_x[n:].reshape(cfg.num_joints, cfg.num_harmonics)

        named_margins = _evaluate_margins(best_x, constraints)
        best_feasible = all(v >= 0 for v in named_margins.values())
        traj_stats = self._compute_trajectory_stats(best_x, cache)

        return OptimizationResult(
            x_opt=best_x,
            condition_number=best_cond,
            a_opt=a_opt,
            b_opt=b_opt,
            q0=q0,
            config=cfg,
            n_evaluations=total_evals,
            wall_time=wall_time,
            n_restarts=actual_restarts,
            best_start_index=best_idx,
            constraint_margins=named_margins,
            feasible=best_feasible,
            trajectory_stats=traj_stats,
        )

    def validate_trajectory(self, result: OptimizationResult) -> dict:
        """Full-resolution validation of an optimisation result."""
        cfg = self.config
        q0 = np.asarray(cfg.q0, dtype=np.float64)

        cond, _ = evaluate_full_resolution(
            result.x_opt,
            self.model,
            self.data,
            names.PAYLOAD_BODY,
            cfg.num_joints,
            cfg.num_harmonics,
            cfg.base_freq,
            cfg.duration,
            cfg.fps,
            q0,
            with_ft_offset=cfg.with_ft_offset,
            column_scale=cfg.ft_offset_column_scale and cfg.with_ft_offset,
            site_name=names.FT_SITE,
        )

        _, full_constraints = self._build_cache_and_constraints()

        margins: list[float] = []
        all_satisfied = True
        for c in full_constraints:
            val = c["fun"](result.x_opt)
            margins.append(val)
            if val < 0:
                all_satisfied = False

        return {
            "condition_number": cond,
            "all_constraints_satisfied": all_satisfied,
            "constraint_margins": margins,
        }
