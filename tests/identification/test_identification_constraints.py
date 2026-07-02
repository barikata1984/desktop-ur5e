import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from ur5e_sim.identification.collision import CollisionChecker, CollisionConfig  # noqa: E402
from ur5e_sim.identification.constraints import (  # noqa: E402
    JointLimits,
    _TrajectoryCache,
    build_scipy_constraints,
    build_trajectory_from_params,
    make_joint_acceleration_constraint,
    make_joint_position_constraint,
    make_joint_velocity_constraint,
)
from ur5e_sim.identification.workspace import (  # noqa: E402
    WorkspaceConstraintConfig,
    evaluate_workspace_displacement,
    make_workspace_constraint,
)

from .conftest import arm_to_full_qpos, load_identification_scene  # noqa: E402

NUM_JOINTS = 6
NUM_HARMONICS = 2
BASE_FREQ = 0.2
DURATION = 4.0
FPS = 50.0
Q0_ARM = np.array([np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0])


def _make_cache() -> _TrajectoryCache:
    return _TrajectoryCache(
        num_joints=NUM_JOINTS,
        num_harmonics=NUM_HARMONICS,
        base_freq=BASE_FREQ,
        duration=DURATION,
        fps=FPS,
        q0=Q0_ARM,
    )


def _make_x(scale: float, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = NUM_JOINTS * NUM_HARMONICS
    return rng.uniform(-scale, scale, size=2 * n)


def _load_scene():
    return load_identification_scene()


# --- Joint constraint tests ---


def test_small_amplitude_satisfies_all_joint_constraints() -> None:
    cache = _make_cache()
    limits = JointLimits()
    x = _make_x(scale=0.05)

    pos_c = make_joint_position_constraint(cache, limits)
    vel_c = make_joint_velocity_constraint(cache, limits)
    acc_c = make_joint_acceleration_constraint(cache, limits)

    assert pos_c(x) >= 0, "Position constraint violated for small amplitude"
    assert vel_c(x) >= 0, "Velocity constraint violated for small amplitude"
    assert acc_c(x) >= 0, "Acceleration constraint violated for small amplitude"


def test_large_amplitude_violates_joint_constraint() -> None:
    cache = _make_cache()
    limits = JointLimits()
    x = _make_x(scale=5.0)

    pos_c = make_joint_position_constraint(cache, limits)
    vel_c = make_joint_velocity_constraint(cache, limits)
    acc_c = make_joint_acceleration_constraint(cache, limits)

    any_violated = pos_c(x) < 0 or vel_c(x) < 0 or acc_c(x) < 0
    assert any_violated, "Expected at least one constraint violation for large amplitude"


# --- Workspace constraint tests ---
# These tests use _TrajectoryCache which produces 6-joint trajectories, but the
def test_workspace_displacement_with_known_trajectory() -> None:
    loaded = _load_scene()
    cache = _make_cache()
    x = _make_x(scale=0.05)
    sample = cache.get(x)

    distances = evaluate_workspace_displacement(loaded.model, loaded.data, sample.position)
    assert distances.shape[0] == sample.position.shape[0]
    assert distances[0] == pytest.approx(0.0, abs=1e-10)
    assert np.all(distances >= 0)


def test_workspace_constraint_small_motion() -> None:
    loaded = _load_scene()
    cache = _make_cache()
    ws_config = WorkspaceConstraintConfig(max_displacement=2.0)
    x = _make_x(scale=0.05)

    ws_fn = make_workspace_constraint(cache, ws_config, loaded.model, loaded.data)
    assert ws_fn(x) >= 0, "Small motion should satisfy generous workspace constraint"


# --- Collision checker tests ---


def test_collision_checker_home_config_is_safe() -> None:
    loaded = _load_scene()
    checker = CollisionChecker(loaded.model, loaded.data)
    q0_full = arm_to_full_qpos(Q0_ARM, loaded.model.nq)
    clearance = checker.check_single_config(q0_full)
    assert clearance > 0, f"Home configuration should be collision-free, got {clearance:.4f}"


def test_collision_checker_trajectory() -> None:
    loaded = _load_scene()
    checker = CollisionChecker(loaded.model, loaded.data)
    cache = _make_cache()
    x = _make_x(scale=0.05)
    sample = cache.get(x)
    clearance = checker.compute_min_clearance(sample.position)
    assert np.isfinite(clearance)


def test_collision_large_payload_detected_by_surface() -> None:
    """A huge payload box should collide with link capsules."""
    loaded = _load_scene()
    config = CollisionConfig(payload_half_extents=[0.5, 0.5, 0.5], payload_offset=[0, 0, 0.10])
    checker = CollisionChecker(loaded.model, loaded.data, config)
    q0_full = arm_to_full_qpos(Q0_ARM, loaded.model.nq)
    clearance = checker.check_single_config(q0_full)
    assert clearance < 0, "Large payload box surface should penetrate link capsules"


def test_collision_box_capsule_clearance_unit() -> None:
    """Unit test for box-capsule surface distance computation."""
    box_center = np.zeros(3)
    box_rot = np.eye(3)
    half = np.ones(3)

    c = CollisionChecker._box_capsule_clearance(
        box_center,
        box_rot,
        half,
        np.array([2.5, 0, 0]),
        np.array([3.5, 0, 0]),
        0.1,
    )
    np.testing.assert_allclose(c, 1.4, atol=1e-10)

    c = CollisionChecker._box_capsule_clearance(
        box_center,
        box_rot,
        half,
        np.array([0, 0, 0]),
        np.array([2, 0, 0]),
        0.1,
    )
    np.testing.assert_allclose(c, -0.1, atol=1e-10)


def test_collision_auto_extracts_payload_geometry() -> None:
    """CollisionChecker should auto-extract payload box geometry from model."""
    loaded = _load_scene()
    checker = CollisionChecker(loaded.model, loaded.data)
    np.testing.assert_allclose(checker._payload_half_extents, [0.05, 0.15, 0.05])
    np.testing.assert_allclose(checker._payload_offset, [0, 0, 0.05])


def test_collision_home_config_safe_with_actual_geometry() -> None:
    """Home configuration should be collision-free with model-extracted geometry."""
    loaded = _load_scene()
    checker = CollisionChecker(loaded.model, loaded.data)
    q0_full = arm_to_full_qpos(Q0_ARM, loaded.model.nq)
    clearance = checker.check_single_config(q0_full)
    assert clearance > 0, f"Home config should be safe, got clearance={clearance:.4f}"


def test_collision_payload_ground_clearance() -> None:
    """Payload ground clearance should use box vertices, not just body origin."""
    loaded = _load_scene()
    q0_full = arm_to_full_qpos(Q0_ARM, loaded.model.nq)

    config_safe = CollisionConfig(
        payload_half_extents=[0.01, 0.01, 0.01], payload_offset=[0, 0, 0.05]
    )
    checker_safe = CollisionChecker(loaded.model, loaded.data, config_safe)
    clearance_safe = checker_safe.check_single_config(q0_full)

    config_big = CollisionConfig(
        payload_half_extents=[0.01, 0.01, 0.5], payload_offset=[0, 0, 0.05]
    )
    checker_big = CollisionChecker(loaded.model, loaded.data, config_big)
    clearance_big = checker_big.check_single_config(q0_full)

    assert clearance_big < clearance_safe, "Larger box should have smaller ground clearance"


# --- build_scipy_constraints tests ---


def test_build_scipy_constraints_returns_expected_format() -> None:
    cache = _make_cache()
    limits = JointLimits()
    constraints = build_scipy_constraints(cache, limits)

    assert isinstance(constraints, list)
    assert len(constraints) >= 3
    for c in constraints:
        assert "type" in c
        assert c["type"] == "ineq"
        assert "fun" in c
        assert callable(c["fun"])


def test_build_scipy_constraints_with_workspace_and_collision() -> None:
    loaded = _load_scene()
    cache = _make_cache()
    limits = JointLimits()
    ws_config = WorkspaceConstraintConfig()
    col_config = CollisionConfig()

    constraints = build_scipy_constraints(
        cache,
        limits,
        workspace_config=ws_config,
        collision_config=col_config,
        model=loaded.model,
        data=loaded.data,
    )
    assert len(constraints) == 5  # 3 joint + workspace + collision


# --- Cache tests ---


def test_trajectory_cache_reuses_result() -> None:
    cache = _make_cache()
    x = _make_x(scale=0.1)

    sample1 = cache.get(x)
    sample2 = cache.get(x)
    assert sample1 is sample2, "Cache should return same object for same x"


def test_trajectory_cache_invalidates_on_different_x() -> None:
    cache = _make_cache()
    x1 = _make_x(scale=0.1, seed=1)
    x2 = _make_x(scale=0.1, seed=2)

    sample1 = cache.get(x1)
    sample2 = cache.get(x2)
    assert sample1 is not sample2


# --- build_trajectory_from_params round-trip ---


def test_build_trajectory_from_params_roundtrip() -> None:
    rng = np.random.default_rng(42)
    a = rng.uniform(-0.1, 0.1, size=(NUM_JOINTS, NUM_HARMONICS))
    b = rng.uniform(-0.1, 0.1, size=(NUM_JOINTS, NUM_HARMONICS))
    x = np.concatenate([a.ravel(), b.ravel()])

    sample = build_trajectory_from_params(
        x, NUM_JOINTS, NUM_HARMONICS, BASE_FREQ, DURATION, FPS, Q0_ARM
    )

    assert sample.position.shape == (int(DURATION * FPS) + 1, NUM_JOINTS)
    assert sample.velocity.shape == sample.position.shape
    assert sample.acceleration.shape == sample.position.shape

    np.testing.assert_allclose(sample.position[0], Q0_ARM, atol=1e-10)
    np.testing.assert_allclose(sample.position[-1], Q0_ARM, atol=1e-10)
