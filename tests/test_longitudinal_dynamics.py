"""Mechanical checks with synthetic coefficients, not game identification."""

from dataclasses import replace
import math

import pytest

from aim120_model.longitudinal_dynamics import (
    BodyAeroCoefficients,
    LongitudinalProperties,
    LongitudinalState,
    StaticAeroPolynomial,
    evaluate_longitudinal,
    rk4_march_longitudinal,
)
from aim120_model.units import G0_MPS2


PROPERTIES = LongitudinalProperties(
    mass_kg=10.0,
    pitch_inertia_kg_m2=2.0,
    reference_area_m2=0.01,
    reference_length_m=0.1,
    density_kg_m3=0.8,
    thrust_n=300.0,
)


@pytest.mark.parametrize("alpha_deg", [-60.0, 0.0, 35.0, 90.0])
def test_force_projection_matches_cartesian_newton_law(alpha_deg: float) -> None:
    state = LongitudinalState(200.0, 0.2, math.radians(alpha_deg), 0.4)
    coeffs = BodyAeroCoefficients(0.3, 2.0, -0.15)
    result = evaluate_longitudinal(state, PROPERTIES, coeffs)

    # Independently sum body-axis forces in an inertial x/z-up frame, then
    # recover Cartesian acceleration from the returned V/gamma derivatives.
    theta = state.flight_path_angle_rad + state.alpha_rad
    qs = 0.5 * 0.8 * 200.0**2 * 0.01
    fx = (300.0 - qs * coeffs.c_a) * math.cos(theta) - qs * coeffs.c_n * math.sin(theta)
    fz = (300.0 - qs * coeffs.c_a) * math.sin(theta) + qs * coeffs.c_n * math.cos(theta)
    gamma = state.flight_path_angle_rad
    ax = result.speed_dot_mps2 * math.cos(gamma) - 200.0 * result.flight_path_angle_dot_rad_s * math.sin(gamma)
    az = result.speed_dot_mps2 * math.sin(gamma) + 200.0 * result.flight_path_angle_dot_rad_s * math.cos(gamma)
    assert ax == pytest.approx(fx / 10.0)
    assert az == pytest.approx(fz / 10.0 - G0_MPS2)
    assert result.alpha_dot_rad_s + result.flight_path_angle_dot_rad_s == pytest.approx(0.4)
    assert result.pitch_rate_dot_rad_s2 == pytest.approx(qs * 0.1 * -0.15 / 2.0)
    assert result.wind_normal_specific_force_g - result.trajectory_normal_acceleration_g == pytest.approx(math.cos(gamma))


def test_ballistic_motion_rotates_velocity_without_aerodynamic_load() -> None:
    state = LongitudinalState(200.0, 0.0, 0.0, 0.0)
    result = evaluate_longitudinal(
        state, replace(PROPERTIES, thrust_n=0.0), BodyAeroCoefficients(0.0, 0.0, 0.0)
    )
    assert result.speed_dot_mps2 == 0.0
    assert result.flight_path_angle_dot_rad_s == pytest.approx(-G0_MPS2 / 200.0)
    assert result.alpha_dot_rad_s == pytest.approx(G0_MPS2 / 200.0)
    assert result.pitch_rate_dot_rad_s2 == 0.0
    assert result.body_normal_specific_force_g == 0.0
    assert result.wind_normal_specific_force_g == 0.0
    assert result.trajectory_normal_acceleration_g == -1.0


def test_nonzero_trim_force_does_not_require_nonzero_total_moment() -> None:
    # A synthetic tail-down/nose-up example: coefficients are independent.
    cn = StaticAeroPolynomial(alpha_per_rad=3.0, delta_per_rad=-1.0)
    cm = StaticAeroPolynomial(alpha_per_rad=-0.5, delta_per_rad=1.0)
    alpha, delta = 0.2, 0.1
    coeffs = BodyAeroCoefficients(0.0, cn.coefficient(alpha, delta), cm.coefficient(alpha, delta))
    state = LongitudinalState(200.0, 0.0, alpha, 0.0)
    result = evaluate_longitudinal(state, PROPERTIES, coeffs)
    assert result.body_normal_force_n > 0.0
    assert result.pitch_rate_dot_rad_s2 == 0.0
    changed = evaluate_longitudinal(state, PROPERTIES, replace(coeffs, c_m=0.1))
    assert changed.wind_normal_specific_force_g == result.wind_normal_specific_force_g
    assert changed.pitch_rate_dot_rad_s2 > 0.0


@pytest.mark.parametrize("alpha", [-0.6, 0.0, 0.6])
def test_local_derivative_of_odd_nonlinear_aero_law(alpha: float) -> None:
    law = StaticAeroPolynomial(2.0, -0.5, 0.3, -0.1)
    delta, step = 0.07, 1e-6
    numerical_alpha = (law.coefficient(alpha + step, delta) - law.coefficient(alpha - step, delta)) / (2.0 * step)
    numerical_delta = (law.coefficient(alpha, delta + step) - law.coefficient(alpha, delta - step)) / (2.0 * step)
    assert law.alpha_derivative(alpha) == pytest.approx(numerical_alpha, rel=1e-6)
    assert law.delta_per_rad == pytest.approx(numerical_delta)
    assert law.coefficient(-alpha, -delta) == pytest.approx(-law.coefficient(alpha, delta))


def test_rk4_march_quadratic_drag_has_fourth_order_accuracy() -> None:
    # Exact solution to V_dot=-0.05*V**2; a frozen/measured V in the RHS
    # cannot produce the RK4 convergence rate of this free forward march.
    properties = LongitudinalProperties(1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0)
    coeffs = BodyAeroCoefficients(0.1, 0.0, 0.0)
    initial = LongitudinalState(10.0, 0.0, 0.0, 0.0, altitude_m=6300.0)

    def rhs(time_s, state):
        return evaluate_longitudinal(state, properties, coeffs)

    coarse = rk4_march_longitudinal(initial, rhs, end_time_s=2.0, dt_s=0.4)
    fine = rk4_march_longitudinal(initial, rhs, end_time_s=2.0, dt_s=0.2)
    exact_speed = 10.0 / (1.0 + 0.05 * 10.0 * 2.0)
    coarse_error = abs(coarse[-1][1].speed_mps - exact_speed)
    fine_error = abs(fine[-1][1].speed_mps - exact_speed)
    assert coarse_error / fine_error > 12.0
    assert fine_error < 1e-4
    assert fine[-1][1].downrange_m == pytest.approx(math.log(2.0) / 0.05, abs=1e-4)
    assert fine[-1][1].altitude_m == 6300.0


def test_rk4_march_free_flight_matches_cartesian_ballistic_solution() -> None:
    initial = LongitudinalState(150.0, 0.2, 0.1, 0.03, 12.0, 6300.0)
    properties = replace(PROPERTIES, density_kg_m3=0.0, thrust_n=0.0)

    def rhs(time_s, state):
        return evaluate_longitudinal(state, properties, BodyAeroCoefficients(0.0, 0.0, 0.0))

    trajectory = rk4_march_longitudinal(initial, rhs, end_time_s=2.0, dt_s=0.05)
    vx = 150.0 * math.cos(0.2)
    vz0 = 150.0 * math.sin(0.2)
    for time_s, state in trajectory:
        vz = vz0 - G0_MPS2 * time_s
        gamma = math.atan2(vz, vx)
        assert state.speed_mps == pytest.approx(math.hypot(vx, vz), abs=1e-6, rel=0.0)
        assert state.flight_path_angle_rad == pytest.approx(gamma, abs=1e-8)
        assert state.alpha_rad == pytest.approx(0.3 + 0.03 * time_s - gamma, abs=1e-8)
        assert state.downrange_m == pytest.approx(12.0 + vx * time_s, abs=1e-6, rel=0.0)
        assert state.altitude_m == pytest.approx(6300.0 + vz0 * time_s - 0.5 * G0_MPS2 * time_s**2, abs=1e-6, rel=0.0)


def test_rk4_march_preserves_state_across_burnout() -> None:
    initial = LongitudinalState(10.0, 0.0, 0.0, 0.0)
    properties = LongitudinalProperties(2.0, 1.0, 1.0, 1.0, 0.0, 20.0, 0.0)

    def rhs(time_s, state):
        current = replace(properties, thrust_n=20.0 if time_s < 0.7 else 0.0)
        return evaluate_longitudinal(state, current, BodyAeroCoefficients(0.0, 0.0, 0.0))

    trajectory = rk4_march_longitudinal(
        initial, rhs, end_time_s=1.3, dt_s=0.3, discontinuity_times_s=(0.7,)
    )
    at_burnout = dict(trajectory)[0.7]
    assert at_burnout.speed_mps == pytest.approx(17.0)
    assert at_burnout.downrange_m == pytest.approx(9.45)
    assert trajectory[-1][1].speed_mps == pytest.approx(17.0)
    assert trajectory[-1][1].downrange_m == pytest.approx(19.65)
