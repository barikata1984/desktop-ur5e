"""Regression tests for the Hogan-2016 Family-of-Modes pusher-slider MPC.

``PusherSliderMPC`` and ``simulate_analytical`` are pure numpy/scipy (no
MuJoCo stepping), so these tests exercise the analytical model directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from ur5e_sim.pushing.mpc import PusherSliderMPC, simulate_analytical


@pytest.fixture
def mpc() -> PusherSliderMPC:
    """Default-config MPC, contact centered on the '-y' face."""
    return PusherSliderMPC()


@pytest.fixture
def centered_contact(mpc: PusherSliderMPC) -> tuple[float, float]:
    """Aligned-frame contact point for a pusher centered on the default face."""
    px_body, py_body = 0.0, -mpc.b / 2
    return mpc._aligned_contact(px_body, py_body)


class TestSelectModeBoundaries:
    """`_select_mode` guards a recent numerical-slack bug fix at vn <= 0."""

    def test_tiny_negative_vn_is_sticking(
        self, mpc: PusherSliderMPC, centered_contact: tuple[float, float]
    ) -> None:
        px_a, py_a = centered_contact
        # SLSQP may return a slightly negative vn due to numerical slack; any
        # non-positive vn must fall back to sticking (mode 1), not divide by
        # a near-zero/wrong-signed vn.
        assert mpc._select_mode(-1e-8, 0.0, px_a, py_a) == 1
        assert mpc._select_mode(-1e-8, 0.05, px_a, py_a) == 1

    def test_zero_vn_is_sticking(
        self, mpc: PusherSliderMPC, centered_contact: tuple[float, float]
    ) -> None:
        px_a, py_a = centered_contact
        assert mpc._select_mode(0.0, 0.0, px_a, py_a) == 1
        assert mpc._select_mode(0.0, 0.05, px_a, py_a) == 1

    def test_ratio_well_above_gamma_t_is_slide_up(
        self, mpc: PusherSliderMPC, centered_contact: tuple[float, float]
    ) -> None:
        px_a, py_a = centered_contact
        gamma_t, _gamma_b = mpc._motion_cone(px_a, py_a)
        vn = 0.05
        vt = 2.0 * gamma_t * vn  # well outside the upper cone boundary
        assert mpc._select_mode(vn, vt, px_a, py_a) == 2

    def test_ratio_well_below_gamma_b_is_slide_down(
        self, mpc: PusherSliderMPC, centered_contact: tuple[float, float]
    ) -> None:
        px_a, py_a = centered_contact
        _gamma_t, gamma_b = mpc._motion_cone(px_a, py_a)
        vn = 0.05
        vt = 2.0 * gamma_b * vn  # well outside the lower cone boundary
        assert mpc._select_mode(vn, vt, px_a, py_a) == 3

    def test_ratio_strictly_inside_cone_is_sticking(
        self, mpc: PusherSliderMPC, centered_contact: tuple[float, float]
    ) -> None:
        px_a, py_a = centered_contact
        gamma_t, gamma_b = mpc._motion_cone(px_a, py_a)
        assert gamma_b < 0.0 < gamma_t  # sanity: a correct cone straddles zero
        assert mpc._select_mode(0.05, 0.0, px_a, py_a) == 1


class TestPredictionFreeResponseCharacterization:
    """Pins the CURRENT (A=0) prediction behavior.

    ``_build_prediction_matrices`` drops the state Jacobian (A = 0) and
    freezes B at the current state for the whole horizon (deviation #2 in
    the mpc.py module docstring), so the free response (u=0) is identical to
    x0 at every step, and the per-step motion cone used inside
    ``_solve_single_schedule`` (derived from the free response) therefore
    equals the x0 cone at every step too.

    EXPECTED TO CHANGE: once the nominal-trajectory linearisation is
    implemented (time-varying A_j(t), B_j(t)), the free response will evolve
    over the horizon and this test must be updated accordingly.
    """

    def test_free_response_is_constant_across_horizon(self, mpc: PusherSliderMPC) -> None:
        x0 = np.array([0.1, -0.2, 0.05, 0.0])
        px_a = -mpc.b / 2
        # M1 schedule: slide_up at step 0, stick afterwards -- a schedule that
        # does have nonzero B[3, :] (tangential-shift rate) in the sliding step,
        # so a non-constant free response would show up here if A were nonzero.
        schedule = [2] + [1] * (mpc.N - 1)

        _S, x_free = mpc._build_prediction_matrices(x0, px_a, schedule)
        x_free_steps = x_free.reshape(mpc.N, 4)

        np.testing.assert_allclose(x_free_steps, np.tile(x0, (mpc.N, 1)))

    def test_per_step_motion_cone_equals_x0_cone(self, mpc: PusherSliderMPC) -> None:
        x0 = np.array([0.0, 0.0, 0.0, 0.01])
        px_a = -mpc.b / 2
        schedule = [3] + [1] * (mpc.N - 1)  # M2: slide_down then stick

        _S, x_free = mpc._build_prediction_matrices(x0, px_a, schedule)
        py_a_predicted = x_free.reshape(mpc.N, 4)[:, 3]

        gamma_t_x0, gamma_b_x0 = mpc._motion_cone(px_a, x0[3])
        for py_a_k in py_a_predicted:
            gamma_t_k, gamma_b_k = mpc._motion_cone(px_a, py_a_k)
            assert gamma_t_k == pytest.approx(gamma_t_x0)
            assert gamma_b_k == pytest.approx(gamma_b_x0)


class TestComputeControlSanity:
    """`_solve_fom_qp` / `compute_control` respect the control bounds."""

    def test_control_output_is_finite_and_within_bounds(self, mpc: PusherSliderMPC) -> None:
        slider_pose = np.array([0.0, 0.0, 0.0])
        pusher_body = np.array([0.0, -mpc.b / 2])
        target_pose = np.array([0.0, 0.2, 0.0])  # target straight ahead in +y

        vn, vt = mpc.compute_control(slider_pose, pusher_body, target_pose)

        assert np.isfinite(vn)
        assert np.isfinite(vt)
        assert 0.0 <= vn <= mpc.v_max + 1e-9
        assert abs(vt) <= mpc.v_max + 1e-9

    def test_straight_push_control_pushes_forward_with_no_tangential_bias(
        self, mpc: PusherSliderMPC
    ) -> None:
        # Symmetric contact + symmetric target => no reason to slide tangentially.
        slider_pose = np.array([0.0, 0.0, 0.0])
        pusher_body = np.array([0.0, -mpc.b / 2])
        target_pose = np.array([0.0, 0.2, 0.0])

        vn, vt = mpc.compute_control(slider_pose, pusher_body, target_pose)

        assert vn > 0.0
        assert vt == pytest.approx(0.0, abs=1e-6)


class TestSimulateAnalyticalShortRollout:
    """`simulate_analytical` moves the slider toward a +y target."""

    def test_rollout_moves_toward_target_without_nan(self, mpc: PusherSliderMPC) -> None:
        x0_body = np.array([0.0, 0.0, 0.0, -mpc.b / 2])
        target = np.array([0.0, 0.2, 0.0])
        px_body = 0.0
        n_steps = 5

        states, controls = simulate_analytical(mpc, x0_body, target, px_body, n_steps, mpc.dt)

        assert states.shape == (n_steps + 1, 4)
        assert controls.shape == (n_steps, 2)
        assert np.all(np.isfinite(states))
        assert np.all(np.isfinite(controls))

        # Monotonic progress toward the target in y.
        assert np.all(np.diff(states[:, 1]) >= -1e-9)
        assert states[-1, 1] > states[0, 1]

        # A symmetric straight push should keep theta small (no tumbling).
        assert np.all(np.abs(states[:, 2]) < 0.1)
