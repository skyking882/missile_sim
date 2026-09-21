"""Planar force/moment core for the next War Thunder rebuild identification.

Devaud et al. (2000), doi:10.1016/S0967-0661(00)00013-7, identifies
Reichert's nonlinear pitch model as its plant. Its full model section was
not accessible during this implementation: this module is not a reproduction
of that paper's complete equations or numerical benchmark.

Body-axis Newton--Euler equations were checked against Lee, Kim and Tahk,
https://arxiv.org/pdf/1209.0863, p. 3, eqs. (1)--(3). Here they are expressed
in a vertical, still-air plane with gravity and body-aligned thrust. The odd
polynomial basis also appears in the published longitudinal benchmark in
https://scholarsmine.mst.edu/cgi/viewcontent.cgi?article=4886&context=mec_aereng_facwork,
eqs. (10)--(11). No benchmark coefficients or prescribed Mach dependence are
imported into the game model.

Positive pitch/AoA and body-normal force point nose-up. C_A is a signed
body-axis coefficient, positive rearward; it is NOT wind-axis C_D. Positive
C_m gives nose-up moment about the current CG. All angles are radians.
Mass, inertia, density and thrust are instantaneous supplied properties;
burn history and aerodynamic functions are evaluated outside this core.
The rotational approximation is I*q_dot = moment, with no separate exhaust
angular-momentum, CG-motion, rate-damping or actuator model.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from .units import G0_MPS2


@dataclass(frozen=True)
class StaticAeroPolynomial:
    """C = k1*alpha + k2*alpha*abs(alpha) + k3*alpha**3 + kd*delta.

    Instantiate separately for C_N and C_m, at the Mach number of interest.
    k1 and kd are REQUIRED unknowns, not gains inferred from finsLatAccel.
    The nested linear candidate sets k2=k3=0; nonzero values require evidence.
    This is a local polynomial candidate, not a validated full-AoA envelope.
    Delta is the actual fin angle, not AoA or the controller command.
    """

    alpha_per_rad: float
    delta_per_rad: float
    alpha_abs_alpha_per_rad2: float = 0.0
    alpha_cubed_per_rad3: float = 0.0

    def coefficient(self, alpha_rad: float, delta_rad: float) -> float:
        return (
            self.alpha_per_rad * alpha_rad
            + self.alpha_abs_alpha_per_rad2 * alpha_rad * abs(alpha_rad)
            + self.alpha_cubed_per_rad3 * alpha_rad**3
            + self.delta_per_rad * delta_rad
        )

    def alpha_derivative(self, alpha_rad: float) -> float:
        """Local partial derivative dC/dalpha, holding delta and Mach fixed."""
        return (
            self.alpha_per_rad
            + 2.0 * self.alpha_abs_alpha_per_rad2 * abs(alpha_rad)
            + 3.0 * self.alpha_cubed_per_rad3 * alpha_rad**2
        )


@dataclass(frozen=True)
class BodyAeroCoefficients:
    """Instantaneous dimensionless C_A, C_N and C_m on a declared S and l.

    Coefficients must be evaluated at the current alpha, Mach and actual
    delta. No default missile values, force caps or force-to-moment lever
    arm are supplied. A wind-axis drag law must be transformed consistently
    before populating c_a; adding C_N*sin(alpha) to an existing total C_D
    would double-count that normal-force contribution.
    """

    c_a: float
    c_n: float
    c_m: float


@dataclass(frozen=True)
class LongitudinalState:
    speed_mps: float
    flight_path_angle_rad: float
    alpha_rad: float
    pitch_rate_rad_s: float
    downrange_m: float = 0.0
    altitude_m: float = 0.0


@dataclass(frozen=True)
class LongitudinalProperties:
    mass_kg: float
    pitch_inertia_kg_m2: float
    reference_area_m2: float
    reference_length_m: float
    density_kg_m3: float
    thrust_n: float
    gravity_mps2: float = G0_MPS2


@dataclass(frozen=True)
class LongitudinalEvaluation:
    speed_dot_mps2: float
    flight_path_angle_dot_rad_s: float
    alpha_dot_rad_s: float
    pitch_rate_dot_rad_s2: float
    downrange_dot_mps: float
    altitude_dot_mps: float
    axial_aero_force_n: float
    body_normal_force_n: float
    pitch_moment_nm: float
    wind_drag_force_n: float
    wind_lift_force_n: float
    body_normal_specific_force_g: float
    wind_normal_specific_force_g: float
    trajectory_normal_acceleration_g: float


def evaluate_longitudinal(
    state: LongitudinalState,
    properties: LongitudinalProperties,
    aero: BodyAeroCoefficients,
) -> LongitudinalEvaluation:
    """Evaluate the nonlinear ODE and signed CG loads, without PID or PN.

    With A=Q*S*C_A, N=Q*S*C_N and moment=Q*S*l*C_m:
      V_dot     = ((T-A)*cos(alpha)-N*sin(alpha))/m - g*sin(gamma)
      gamma_dot = (((T-A)*sin(alpha)+N*cos(alpha))/m-g*cos(gamma))/V
      alpha_dot = q-gamma_dot
      q_dot     = moment/I

    Thus theta=gamma+alpha obeys theta_dot=q. Normal force changes the
    flight path while moment changes body attitude; neither delta=alpha
    nor moment=-arm*(total normal force) is imposed. G outputs divide by
    standard g0, while the actual gravity term uses properties.gravity_mps2.
    """
    v = state.speed_mps
    m = properties.mass_kg
    inertia = properties.pitch_inertia_kg_m2
    sin_alpha = math.sin(state.alpha_rad)
    cos_alpha = math.cos(state.alpha_rad)
    sin_gamma = math.sin(state.flight_path_angle_rad)
    cos_gamma = math.cos(state.flight_path_angle_rad)

    qs = 0.5 * properties.density_kg_m3 * v * v * properties.reference_area_m2
    axial = qs * aero.c_a
    normal = qs * aero.c_n
    moment = qs * properties.reference_length_m * aero.c_m
    drag = axial * cos_alpha + normal * sin_alpha
    lift = normal * cos_alpha - axial * sin_alpha
    tangent_specific_force = (properties.thrust_n * cos_alpha - drag) / m
    normal_specific_force = (properties.thrust_n * sin_alpha + lift) / m
    normal_acceleration = normal_specific_force - properties.gravity_mps2 * cos_gamma
    gamma_dot = normal_acceleration / v

    return LongitudinalEvaluation(
        speed_dot_mps2=tangent_specific_force - properties.gravity_mps2 * sin_gamma,
        flight_path_angle_dot_rad_s=gamma_dot,
        alpha_dot_rad_s=state.pitch_rate_rad_s - gamma_dot,
        pitch_rate_dot_rad_s2=moment / inertia,
        downrange_dot_mps=v * cos_gamma,
        altitude_dot_mps=v * sin_gamma,
        axial_aero_force_n=axial,
        body_normal_force_n=normal,
        pitch_moment_nm=moment,
        wind_drag_force_n=drag,
        wind_lift_force_n=lift,
        body_normal_specific_force_g=normal / (m * G0_MPS2),
        wind_normal_specific_force_g=normal_specific_force / G0_MPS2,
        trajectory_normal_acceleration_g=normal_acceleration / G0_MPS2,
    )


def rk4_march_longitudinal(
    initial_state: LongitudinalState,
    rhs: Callable[[float, LongitudinalState], LongitudinalEvaluation],
    *,
    end_time_s: float,
    dt_s: float,
    discontinuity_times_s: tuple[float, ...] = (),
) -> list[tuple[float, LongitudinalState]]:
    """March one candidate model continuously from launch time t=0.

    ``rhs(t, state)`` must evaluate propulsion, atmosphere, aerodynamic
    coefficients and forces from that RK stage's time and simulated state.
    It must not read replay V/alpha/G, reset states at observation times, or
    mutate controller memory on each of the four calls. Continuous controller
    states require their own coupled integration; a discrete command may be
    held during a physical integration interval.

    Replay samples are compared with this trajectory AFTER the whole march.
    The initial state and candidate parameters remain fixed for each run.
    Position is integrated alongside V/gamma/alpha/q so density can use the
    simulated altitude. G is an output evaluated from the resulting state.

    Declare forcing discontinuities, such as motor burnout. Steps end exactly
    there; k4 uses the left-hand forcing, and the next k1 uses the right-hand
    forcing. State remains continuous across the boundary.
    """
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be positive and finite")
    if not math.isfinite(end_time_s) or end_time_s < 0.0:
        raise ValueError("end_time_s must be nonnegative and finite")

    def derivative(time_s: float, values: tuple[float, ...]) -> tuple[float, ...]:
        rate = rhs(time_s, LongitudinalState(*values))
        return (
            rate.speed_dot_mps2, rate.flight_path_angle_dot_rad_s,
            rate.alpha_dot_rad_s, rate.pitch_rate_dot_rad_s2,
            rate.downrange_dot_mps, rate.altitude_dot_mps,
        )

    def advance(values: tuple[float, ...], rate: tuple[float, ...], step: float) -> tuple[float, ...]:
        return tuple(value + step * change for value, change in zip(values, rate))

    values = (
        initial_state.speed_mps, initial_state.flight_path_angle_rad,
        initial_state.alpha_rad, initial_state.pitch_rate_rad_s,
        initial_state.downrange_m, initial_state.altitude_m,
    )
    samples = [(0.0, initial_state)]
    boundaries = {t for t in discontinuity_times_s if 0.0 < t <= end_time_s}
    segment_start = 0.0
    time_s = 0.0
    for segment_end in sorted(boundaries | {end_time_s}):
        count = math.ceil((segment_end - segment_start) / dt_s)
        for index in range(count):
            next_time = min(segment_start + (index + 1) * dt_s, segment_end)
            step = next_time - time_s
            k1 = derivative(time_s, values)
            k2 = derivative(time_s + step / 2.0, advance(values, k1, step / 2.0))
            k3 = derivative(time_s + step / 2.0, advance(values, k2, step / 2.0))
            k4_time = math.nextafter(next_time, time_s) if next_time in boundaries else next_time
            k4 = derivative(k4_time, advance(values, k3, step))
            rate = tuple((a + 2.0 * b + 2.0 * c + d) / 6.0 for a, b, c, d in zip(k1, k2, k3, k4))
            values = advance(values, rate, step)
            time_s = next_time
            samples.append((time_s, LongitudinalState(*values)))
        segment_start = segment_end
    return samples
