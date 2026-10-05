"""Independent planar fin-moment candidate for game replay comparison.

The vertical channel is algebraically trimmed at 1 g at fixed altitude.
Horizontal fin deflection comes from an acceleration-feedback PID; local
tail incidence produces a moment and integrated body rotation produces AoA.
The total lift law is an empirical aggregate. The equivalent tail force is
used for its moment, not added a second time to that total lift. The force
slope and the finsLatAccel control-force cap are separate inputs. Their
interpretation and PID input/output units remain candidate assumptions.
An optional independent angular-rate derivative supplies dissipative moment;
it is a model parameter, not a mapping of the datamine WdK field.
Restoring and control effectiveness may be supplied independently. Neither
their ratio nor the resulting trim gain is identified by the datamine.
Replay observations are not accepted by this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

from .atmosphere import StandardAtmosphere
from .drag_models import effective_cda0, effective_cda_alpha
from .events import _segment_sphere_entry_fraction
from .propulsion import PiecewisePropulsion
from .units import G0_MPS2 as G0


def planar_target_motion(scenario: dict[str, Any]):
    """Initial target position/velocity in the launch +x, horizontal +z plane.

    An explicit target heading defaults to absolute world degrees. Scenarios
    without one retain the historical approach along the initial LOS. The
    two named calculator conventions match TargetModel's course definitions.
    """
    azimuth = math.radians(scenario.get("target_azimuth_deg", 30.0))
    distance = scenario.get("initial_distance_m", 8000.0)
    speed = scenario["target_speed_kmh"] / 3.6
    position = (distance * math.cos(azimuth), distance * math.sin(azimuth))
    if "target_heading_deg" not in scenario and "target_course_reference" not in scenario:
        return position, (-speed * math.cos(azimuth), -speed * math.sin(azimuth))
    course = math.radians(scenario.get("target_heading_deg", 0.0))
    reference = scenario.get("target_course_reference", "absolute_world")
    if reference == "relative_to_los":
        course += azimuth + math.pi
    elif reference == "launch_axis":
        course += math.pi
    elif reference != "absolute_world":
        raise ValueError(f"unknown target_course_reference: {reference}")
    if not math.isfinite(course):
        raise ValueError("target heading must be finite")
    return position, (speed * math.cos(course), speed * math.sin(course))


@dataclass(frozen=True)
class AlphaServoParameters:
    lift_area_per_rad: float
    horizontal_fin_limit_rad: float
    vertical_fin_limit_rad: float
    stability_distance_m: float
    fin_force_area_per_rad: float
    fin_lateral_acceleration_limit_g: float
    length_m: float
    navigation_gain: float
    command_limit_g: float
    fuse_radius_m: float
    fuse_arming_time_s: float
    capture_time_constant_s: float
    acceleration_pid_kp: float
    acceleration_pid_ki: float
    acceleration_pid_kd: float
    acceleration_pid_integral_limit: float
    initial_alpha_deg: float = 11.0
    initial_body_rate_rad_s: float = 0.0
    guidance_delay_s: float = 0.5
    # Post-capture relaxation of the reachable AoA command toward PN.
    release_time_constant_s: float = 0.95
    release_angle_deg: float = 15.0
    # M_extra = -(qbar / V) * D_omega * omega. D_omega has units m^4.
    # Zero retains the previous candidate for controlled comparisons.
    rate_damping_derivative_m4: float = 0.0
    # None reproduces the historical equal-slope closure. An explicit value
    # separates static stability and local-inflow damping from control force.
    restoring_force_area_per_rad: float | None = None

    @property
    def restoring_area_per_rad(self) -> float:
        return (self.fin_force_area_per_rad if self.restoring_force_area_per_rad is None
                else self.restoring_force_area_per_rad)


@dataclass(frozen=True)
class FinMomentSample:
    local_alpha_rad: float
    control_force_raw_n: float
    control_force_n: float
    control_force_limit_n: float
    passive_force_n: float
    control_moment_nm: float
    passive_moment_nm: float
    equivalent_normal_force_n: float
    moment_nm: float
    inertia_kg_m2: float
    angular_acceleration_rad_s2: float
    rate_damping_moment_nm: float
    rate_damping_per_s: float


def fin_moment_sample(speed_mps: float, alpha_rad: float, body_rate_rad_s: float,
                      dynamic_pressure_pa: float, mass_kg: float,
                      fin_deflection_rad: float,
                      parameters: AlphaServoParameters) -> FinMomentSample:
    """Aft equivalent surface: M = -ell*F, with local rotational inflow.

    Positive delta produces positive body rotation. At delta=0, positive
    alpha produces a restoring moment. Local inflow supplies rate damping
    without a prescribed attitude-response time constant. Incidence and
    deflection effectiveness are independent when a restoring slope is given;
    omitting it retains the historical equal-slope closure.
    finsLatAccel limits the deflection-generated control force using current
    mass; passive incidence/rate moments are distinct from that actuator cap.
    The additional rate derivative is independent of the equivalent surface.
    It supplements, rather than replaces, local-inflow damping. With a chosen
    coefficient convention D_omega = -Sref*Lref^2*Cm_omega/2; no convention or
    numeric relationship to the game's WdK is assumed here.
    """
    arm = parameters.stability_distance_m
    local_alpha = math.atan2(speed_mps * math.sin(alpha_rad) + arm * body_rate_rad_s,
                            speed_mps * math.cos(alpha_rad))
    force_slope = dynamic_pressure_pa * parameters.fin_force_area_per_rad
    control_raw = -force_slope * fin_deflection_rad
    force_limit = mass_kg * G0 * parameters.fin_lateral_acceleration_limit_g
    control_force = max(-force_limit, min(force_limit, control_raw))
    passive_force = dynamic_pressure_pa * parameters.restoring_area_per_rad * local_alpha
    force = passive_force + control_force
    # Same provisional slender-body inertia used by the existing rebuild.
    inertia = mass_kg * parameters.length_m ** 2 / 12.0
    damping_coefficient = (dynamic_pressure_pa / max(speed_mps, 1e-6)
                           * parameters.rate_damping_derivative_m4)
    damping_moment = -damping_coefficient * body_rate_rad_s
    moment = -arm * force + damping_moment
    return FinMomentSample(local_alpha, control_raw, control_force, force_limit,
                           passive_force, -arm * control_force, -arm * passive_force,
                           force, moment, inertia, moment / inertia,
                           damping_moment, damping_coefficient / inertia)


@dataclass(frozen=True)
class AlphaServoResult:
    samples: list[dict[str, float]]
    event: str
    minimum_distance_m: float


def run_alpha_servo(
    config: dict[str, Any],
    parameters: AlphaServoParameters,
    scenario: dict[str, Any],
    *,
    max_time_s: float = 15.0,
    plant_dt_s: float = 0.002,
    control_dt_s: float = 0.02,
    lift_mach_table: tuple[tuple[float, float], ...] = (),
    moment_model: Callable[..., FinMomentSample] | None = None,
    guidance_mode: str = "capture_pn",
    guidance_request_model: Callable[[float, float], float] | None = None,
) -> AlphaServoResult:
    """March speed, flight heading, position, body heading and body rate.

    RK4 stages share one held fin deflection. Steps end at control, burn, fuse
    arming and lifetime boundaries. Interception uses swept relative motion.
    Force calculations use radians; reported AoA is converted to degrees.
    Optional lift multipliers are linear in Mach and held at table endpoints.
    """
    p = parameters
    if guidance_mode not in {"capture_pn", "pn_only"}:
        raise ValueError("guidance_mode must be capture_pn or pn_only")
    positive = (p.lift_area_per_rad, p.horizontal_fin_limit_rad,
                p.vertical_fin_limit_rad, p.fin_force_area_per_rad,
                p.restoring_area_per_rad, p.length_m,
                p.capture_time_constant_s, p.release_time_constant_s,
                plant_dt_s, control_dt_s, max_time_s)
    if any(not math.isfinite(value) or value <= 0.0 for value in positive):
        raise ValueError("fin-moment scales and time steps must be positive and finite")
    nonnegative = (p.stability_distance_m, p.fin_lateral_acceleration_limit_g,
                   p.rate_damping_derivative_m4,
                   p.acceleration_pid_kp,
                   p.acceleration_pid_ki, p.acceleration_pid_kd,
                   p.acceleration_pid_integral_limit)
    if any(not math.isfinite(value) or value < 0.0 for value in nonnegative):
        raise ValueError("fin arm, force cap, rate damping and PID coefficients must be nonnegative and finite")
    if not 0.0 < p.horizontal_fin_limit_rad <= math.pi / 2 or not 0.0 < p.vertical_fin_limit_rad <= math.pi / 2:
        raise ValueError("fin travel must lie in (0, pi/2]")
    if p.command_limit_g < 1.0 or p.fuse_radius_m < 0.0 or p.fuse_arming_time_s < 0.0:
        raise ValueError("invalid command limit or proximity fuse parameters")
    if any(not math.isfinite(mach) or not math.isfinite(gain) or gain <= 0.0
           for mach, gain in lift_mach_table):
        raise ValueError("lift Mach knots must be finite with positive multipliers")
    if any(b[0] <= a[0] for a, b in zip(lift_mach_table, lift_mach_table[1:])):
        raise ValueError("lift Mach knots must be strictly increasing")
    atmo = StandardAtmosphere().sample(scenario["launch_altitude_m"])
    gravity = float(config.get("atmosphere", {}).get("gravity_mps2", G0))
    prop = PiecewisePropulsion.from_config(config)
    initial_range = scenario.get("initial_distance_m", 8000.0)
    target0, target_v = planar_target_motion(scenario)
    # V, velocity heading, x, z, body heading, body angular rate.
    state = (scenario["launch_speed_kmh"] / 3.6, 0.0, 0.0, 0.0,
             math.radians(p.initial_alpha_deg), p.initial_body_rate_rad_s)
    end_time = min(max_time_s, float(config["performance"]["lifetime_s"]))
    command = 0.0
    fin_deflection = 0.0
    commanded_acceleration_g = 0.0
    pid_integral = previous_error = pid_proportional = pid_derivative = 0.0
    capturing = guidance_mode != "pn_only"
    release_t = 0.0
    tick = 0
    time_s = 0.0
    minimum = initial_range
    event = "lifetime"
    samples: list[dict[str, float]] = []

    def relative(t, s):
        return (target0[0] + t * target_v[0] - s[2], 0.0,
                target0[1] + t * target_v[1] - s[3])

    def sample_moment(speed, alpha_h, body_rate, qbar, mass, delta, total_alpha):
        if moment_model is None:
            return fin_moment_sample(speed, alpha_h, body_rate, qbar, mass, delta, p)
        return moment_model(speed, alpha_h, body_rate, qbar, mass, delta, p, total_alpha)

    def forces(t, s):
        speed = s[0]
        ah = (s[4] - s[1] + math.pi) % (2 * math.pi) - math.pi
        propulsion = prop.sample(t)
        qbar = 0.5 * atmo.density_kg_m3 * speed * speed
        multiplier = 1.0
        if lift_mach_table:
            mach = speed / atmo.speed_of_sound_mps
            multiplier = lift_mach_table[0][1] if mach <= lift_mach_table[0][0] else lift_mach_table[-1][1]
            for (m0, f0), (m1, f1) in zip(lift_mach_table, lift_mach_table[1:]):
                if m0 <= mach <= m1:
                    multiplier = f0 + (f1 - f0) * (mach - m0) / (m1 - m0)
                    break
        lift_slope = p.lift_area_per_rad * multiplier
        # Solve the vertical trim contribution while retaining T*sin(alpha).
        av = propulsion.mass_kg * gravity / (qbar * lift_slope + propulsion.thrust_n)
        for _ in range(8):
            alpha = math.hypot(ah, av)
            sinc = math.sin(alpha) / alpha if alpha else 1.0
            normal_gain = qbar * lift_slope + propulsion.thrust_n * sinc
            av = propulsion.mass_kg * gravity / normal_gain
        alpha = math.hypot(ah, av)
        drag = qbar * (effective_cda0(speed / atmo.speed_of_sound_mps, config)
                      + effective_cda_alpha(alpha, 0.0, speed / atmo.speed_of_sound_mps, config))
        a_normal = normal_gain * ah / propulsion.mass_kg
        return propulsion, qbar, av, alpha, normal_gain, drag, a_normal

    def record(t, s):
        ps, qbar, av, alpha, gain, drag, accel = forces(t, s)
        rel = relative(t, s)
        ah = (s[4] - s[1] + math.pi) % (2 * math.pi) - math.pi
        moment = sample_moment(s[0], ah, s[5], qbar, ps.mass_kg,
                               fin_deflection, alpha)
        row = dict(time_s=t, speed_kmh=s[0] * 3.6,
                   alpha_deg=math.degrees(alpha),
                   wind_normal_specific_force_g=math.hypot(accel / G0, gravity / G0),
                   horizontal_specific_force_g=accel / G0,
                   horizontal_alpha_rad=ah, vertical_alpha_rad=av,
                   commanded_horizontal_alpha_rad=command,
                   commanded_horizontal_acceleration_g=commanded_acceleration_g,
                   horizontal_fin_deflection_rad=fin_deflection,
                   vertical_fin_deflection_rad=(p.restoring_area_per_rad
                                                / p.fin_force_area_per_rad * av),
                   body_heading_rad=s[4], body_rate_rad_s=s[5],
                   tail_local_alpha_rad=moment.local_alpha_rad,
                   fin_control_force_raw_n=moment.control_force_raw_n,
                   fin_control_force_n=moment.control_force_n,
                   fin_control_force_limit_n=moment.control_force_limit_n,
                   fin_control_force_raw_g=moment.control_force_raw_n / (ps.mass_kg * G0),
                   fin_control_force_g=moment.control_force_n / (ps.mass_kg * G0),
                   fin_control_force_saturated=abs(moment.control_force_raw_n) > moment.control_force_limit_n,
                   fin_control_moment_nm=moment.control_moment_nm,
                   fin_passive_moment_nm=moment.passive_moment_nm,
                   equivalent_tail_normal_force_n=moment.equivalent_normal_force_n,
                   fin_moment_nm=moment.control_moment_nm + moment.passive_moment_nm,
                   rate_damping_moment_nm=moment.rate_damping_moment_nm,
                   rate_damping_per_s=moment.rate_damping_per_s,
                   total_aerodynamic_moment_nm=moment.moment_nm,
                   inertia_kg_m2=moment.inertia_kg_m2,
                   body_angular_acceleration_rad_s2=moment.angular_acceleration_rad_s2,
                   pid_proportional_output=pid_proportional,
                   pid_integral_output=pid_integral,
                   pid_derivative_output=pid_derivative,
                   heading_rad=s[1], x_m=s[2], z_m=s[3],
                   altitude_m=scenario["launch_altitude_m"],
                   target_x_m=target0[0] + t * target_v[0],
                   target_z_m=target0[1] + t * target_v[1],
                   distance_m=math.hypot(rel[0], rel[2]),
                   mass_kg=ps.mass_kg, thrust_n=ps.thrust_n, drag_n=drag)
        samples.append(row)

    while time_s < end_time - 1e-10:
        ps, qbar, av, alpha, gain, drag, accel = forces(time_s, state)
        # Static vertical trim uses the same independent moment derivatives.
        # It requires both adequate travel and adequate capped control force.
        vertical_fin = p.restoring_area_per_rad / p.fin_force_area_per_rad * av
        vertical_trim_force = qbar * p.restoring_area_per_rad * abs(av)
        if (abs(vertical_fin) > p.vertical_fin_limit_rad
                or vertical_trim_force > ps.mass_kg * G0 * p.fin_lateral_acceleration_limit_g):
            event = "vertical_trim_unavailable"
            break
        if time_s >= tick * control_dt_s - 1e-10:
            rx, _, rz = relative(time_s, state)
            speed, heading = state[:2]
            rvx = target_v[0] - speed * math.cos(heading)
            rvz = target_v[1] - speed * math.sin(heading)
            distance = math.hypot(rx, rz)
            closing = -(rx * rvx + rz * rvz) / max(distance, 1e-9)
            los_rate = (rx * rvz - rz * rvx) / max(distance * distance, 1e-9)
            a_cmd = 0.0
            if guidance_mode == "pn_only":
                if time_s >= p.guidance_delay_s - 1e-10:
                    a_cmd = p.navigation_gain * max(closing, 0.0) * los_rate
            elif time_s >= p.guidance_delay_s - 1e-10:
                transverse_target_speed = (rx * target_v[1] - rz * target_v[0]) / max(distance, 1e-9)
                lead = math.asin(max(-1.0, min(1.0, transverse_target_speed / speed)))
                error = (math.atan2(rz, rx) + lead - heading + math.pi) % (2 * math.pi) - math.pi
                capture_a = speed * math.sin(error) / p.capture_time_constant_s
                pn = p.navigation_gain * max(closing, 0.0) * los_rate
                if capturing and abs(error) < math.radians(p.release_angle_deg):
                    capturing = False
                    release_t = time_s
                    # Begin handover from delivered incidence, not an
                    # unreachable acceleration-derived angle request.
                    command = (state[4] - state[1] + math.pi) % (2 * math.pi) - math.pi
                if capturing:
                    a_cmd = capture_a
                else:
                    a_cmd = pn
            if guidance_request_model is not None:
                a_cmd = guidance_request_model(time_s, a_cmd / G0) * G0
            max_a = math.sqrt(max((G0 * p.command_limit_g) ** 2 - gravity ** 2, 0.0))
            a_cmd = max(-max_a, min(max_a, a_cmd))
            desired_alpha = ps.mass_kg * a_cmd / gain
            if guidance_mode == "pn_only":
                command = desired_alpha
            elif capturing:
                command = desired_alpha
            elif time_s > release_t + 1e-10:
                fraction = -math.expm1(-control_dt_s / p.release_time_constant_s)
                command += fraction * (desired_alpha - command)
            requested_acceleration = max(-max_a, min(max_a, gain * command / ps.mass_kg))
            commanded_acceleration_g = requested_acceleration / G0
            if time_s >= p.guidance_delay_s - 1e-10:
                # One PID: error in wind-normal g, output in fin-travel
                # fraction. These unit mappings remain candidate assumptions.
                error_g = commanded_acceleration_g - accel / G0
                pid_proportional = p.acceleration_pid_kp * error_g
                pid_derivative = p.acceleration_pid_kd * (error_g - previous_error) / control_dt_s
                proposed_integral = max(-p.acceleration_pid_integral_limit,
                                        min(p.acceleration_pid_integral_limit,
                                            pid_integral + p.acceleration_pid_ki * error_g * control_dt_s))
                output = pid_proportional + proposed_integral + pid_derivative
                if not (abs(output) > 1.0 and output * error_g > 0.0):
                    pid_integral = proposed_integral
                output = pid_proportional + pid_integral + pid_derivative
                fin_deflection = max(-1.0, min(1.0, output)) * p.horizontal_fin_limit_rad
                previous_error = error_g
            else:
                fin_deflection = 0.0
            record(time_s, state)
            tick += 1

        boundaries = [time_s + plant_dt_s, tick * control_dt_s, end_time]
        burn_boundary = prop.next_boundary_after(time_s)
        if burn_boundary is not None:
            boundaries.append(burn_boundary)
        if p.fuse_arming_time_s > time_s + 1e-10:
            boundaries.append(p.fuse_arming_time_s)
        next_time = min(boundaries)
        # Avoid landing one ulp before a burn/control boundary: otherwise the
        # scheduler advances while the propulsion sample is still left-sided.
        for boundary in boundaries[1:]:
            if abs(next_time - boundary) < 1e-10:
                next_time = boundary
        dt = next_time - time_s

        def rhs(t, s):
            ps, qbar, av, alpha, gain, drag, accel = forces(t, s)
            ah = (s[4] - s[1] + math.pi) % (2 * math.pi) - math.pi
            moment = sample_moment(s[0], ah, s[5], qbar, ps.mass_kg,
                                   fin_deflection, alpha)
            return ((ps.thrust_n * math.cos(alpha) - drag) / ps.mass_kg,
                    accel / s[0], s[0] * math.cos(s[1]), s[0] * math.sin(s[1]),
                    s[5], moment.angular_acceleration_rad_s2)

        def shifted(s, k, h):
            return tuple(a + h * b for a, b in zip(s, k))

        k1 = rhs(time_s, state)
        k2 = rhs(time_s + dt / 2, shifted(state, k1, dt / 2))
        k3 = rhs(time_s + dt / 2, shifted(state, k2, dt / 2))
        # Use the left-hand propulsion value at a discontinuity.
        k4 = rhs(math.nextafter(next_time, time_s), shifted(state, k3, dt))
        following = tuple(a + dt * (b + 2*c + 2*d + e) / 6
                          for a, b, c, d, e in zip(state, k1, k2, k3, k4))
        if not all(math.isfinite(value) for value in following) or following[0] <= 0.0:
            event = "numerical_failure"
            break
        r0, r1 = relative(time_s, state), relative(next_time, following)
        dr = tuple(b - a for a, b in zip(r0, r1))
        denominator = sum(value * value for value in dr)
        cpa_fraction = (-sum(a*b for a, b in zip(r0, dr)) / denominator
                        if denominator else 0.0)
        fraction = None
        if time_s >= p.fuse_arming_time_s - 1e-10:
            fraction = _segment_sphere_entry_fraction(r0, r1, p.fuse_radius_m)
        if fraction is not None:
            event = "fuse"
        elif 0.0 < cpa_fraction < 1.0:
            fraction = cpa_fraction
            event = "closest_approach"
        stop_fraction = 1.0 if fraction is None else fraction
        nearest = max(0.0, min(stop_fraction, cpa_fraction))
        minimum = min(minimum, math.sqrt(sum((a + nearest*b)**2 for a, b in zip(r0, dr))))
        state = tuple(a + stop_fraction*(b-a) for a, b in zip(state, following))
        time_s = next_time if fraction is None else time_s + dt * stop_fraction
        if fraction is not None:
            break
    if not samples or time_s > samples[-1]["time_s"] + 1e-10:
        record(time_s, state)
    return AlphaServoResult(samples, event, minimum)
