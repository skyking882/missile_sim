"""Three-dimensional proportional-navigation and minimal loft candidate."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .aerodynamics import body_axes_for_state, cg_wind_normal_basis
from .math3d import Vector, clamp_norm, cross, dot, norm, normalize, scale, sub
from .target import TargetState
from .tracking import TrackMode, TrackSolution
from .units import deg_to_rad, g_to_mps2, mps2_to_g


@dataclass(frozen=True)
class GuidanceOutput:
    enabled: bool
    range_m: float
    closing_speed_mps: float
    los_rate_vector_rad_s: Vector
    pn_acceleration_mps2: Vector
    loft_acceleration_mps2: Vector
    commanded_acceleration_mps2: Vector
    commanded_body_acceleration_g: tuple[float, float]
    controller_specific_force_command_g: tuple[float, float]
    wind_normal_specific_force_command_g: tuple[float, float]
    gravity_compensation_wind_normal_g: tuple[float, float]
    wind_normal_pitch_axis: Vector
    wind_normal_yaw_axis: Vector
    effective_gain: float
    time_to_go_s: float
    loft_active: bool
    within_lock_range: bool


def interpolate_table(x: float, table: list[list[float]]) -> float:
    if not table:
        return 1.0
    points = sorted((float(row[0]), float(row[1])) for row in table)
    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= x <= x1:
            fraction = (x - x0) / (x1 - x0)
            return y0 + fraction * (y1 - y0)
    return points[-1][1]


# Below these floors the 1/R and 1/R^2 PN terms are leftover numerics, not
# a guidance solution.  Do not substitute the floor into the divisor.
NEAR_RANGE_M = 1.0e-6
NEAR_CLOSING_MPS = 1.0e-6
NEAR_LOS_RATE = 1.0e-8


def _flush_tiny(value: float, floor: float) -> float:
    return 0.0 if abs(value) < floor else value


def pn_acceleration(
    relative_position: Vector,
    relative_velocity: Vector,
    missile_velocity: Vector,
    pn_gain: float,
    epsilon: float = NEAR_RANGE_M,
) -> tuple[Vector, float, Vector]:
    """Return candidate PN acceleration, closing speed, and LOS rate."""

    range_sq = dot(relative_position, relative_position)
    range_m = math.sqrt(range_sq)
    if range_m <= epsilon:
        return (0.0, 0.0, 0.0), 0.0, (0.0, 0.0, 0.0)
    missile_speed = norm(missile_velocity)
    v_hat = normalize(missile_velocity)
    los_rate = scale(cross(relative_position, relative_velocity), 1.0 / range_sq)
    los_rate = (
        _flush_tiny(los_rate[0], NEAR_LOS_RATE),
        _flush_tiny(los_rate[1], NEAR_LOS_RATE),
        _flush_tiny(los_rate[2], NEAR_LOS_RATE),
    )
    closing = -dot(relative_position, relative_velocity) / range_m
    if closing <= NEAR_CLOSING_MPS or missile_speed <= epsilon:
        return (0.0, 0.0, 0.0), _flush_tiny(closing, NEAR_CLOSING_MPS), los_rate
    acceleration = scale(cross(los_rate, v_hat), pn_gain * closing)
    return acceleration, closing, los_rate


def guidance_command(
    state: Any,
    target: TrackSolution | TargetState,
    time_s: float,
    config: dict[str, Any],
    enabled: bool,
) -> GuidanceOutput:
    """Build the existing PN/loft command from a copied track solution.

    ``TargetState`` remains accepted as a narrow compatibility shim for the
    older standalone geometry tests and legacy callers.  H2 always supplies a
    ``TrackSolution``; an invalid solution is an explicit zero-command state.
    """

    if isinstance(target, TargetState):
        track = TrackSolution(
            position=target.position,
            velocity=target.velocity,
            sample_time_s=time_s,
            solution_time_s=time_s,
            mode=TrackMode.IDEAL_TRUTH,
            valid=True,
            source="legacy_target_state",
        )
    elif isinstance(target, TrackSolution):
        track = target
    else:
        raise TypeError("guidance_command requires TrackSolution or TargetState")
    guidance_cfg = config["guidance"]
    relative_position = sub(track.position, state.position)
    relative_velocity = sub(track.velocity, state.velocity)
    range_m = norm(relative_position)

    if not track.valid:
        return GuidanceOutput(
            enabled=False,
            range_m=range_m,
            closing_speed_mps=0.0,
            los_rate_vector_rad_s=(0.0, 0.0, 0.0),
            pn_acceleration_mps2=(0.0, 0.0, 0.0),
            loft_acceleration_mps2=(0.0, 0.0, 0.0),
            commanded_acceleration_mps2=(0.0, 0.0, 0.0),
            commanded_body_acceleration_g=(0.0, 0.0),
            controller_specific_force_command_g=(0.0, 0.0),
            wind_normal_specific_force_command_g=(0.0, 0.0),
            gravity_compensation_wind_normal_g=(0.0, 0.0),
            wind_normal_pitch_axis=(0.0, 1.0, 0.0),
            wind_normal_yaw_axis=(0.0, 0.0, 1.0),
            effective_gain=0.0,
            time_to_go_s=0.0,
            loft_active=False,
            within_lock_range=False,
        )
    pn_raw, closing, los_rate = pn_acceleration(
        relative_position,
        relative_velocity,
        state.velocity,
        guidance_cfg["pn_gain"],
    )
    flight_gain = interpolate_table(time_s, guidance_cfg["flight_time_gain_table"])
    time_to_go = range_m / closing if closing > NEAR_CLOSING_MPS else 0.0
    hit_gain = interpolate_table(time_to_go, guidance_cfg["time_to_hit_gain_table"])
    effective_gain = flight_gain * hit_gain
    pn = scale(pn_raw, effective_gain)
    loft_active = False
    loft = (0.0, 0.0, 0.0)
    if enabled and guidance_cfg.get("lofting_enabled", False):
        exit_distance = float(guidance_cfg.get("loft_exit_distance_m", guidance_cfg["lock_range_m"]))
        exit_tgo = float(guidance_cfg.get("loft_exit_time_to_go_s", 0.0))
        if range_m > exit_distance and time_to_go > exit_tgo:
            loft_active = True
            desired_pitch = deg_to_rad(guidance_cfg["lofting_elevation_deg"])
            pitch_error = desired_pitch - float(state.pitch)
            loft_g = guidance_cfg["angle_to_acceleration_multiplier"] * pitch_error
            gravity_mps2 = float(config["atmosphere"]["gravity_mps2"])
            # Profile loft: cap the implied pitch rate at omega_max.  Frozen
            # H1/H2 configs omit loft_omega_max_deg_s and keep the uncapped
            # vertical-G command.
            omega_max_deg_s = guidance_cfg.get("loft_omega_max_deg_s")
            if omega_max_deg_s is not None:
                speed = max(norm(state.velocity), 50.0)
                omega_cmd = loft_g * gravity_mps2 / speed
                omega_max = deg_to_rad(float(omega_max_deg_s))
                if abs(omega_cmd) > omega_max > 0.0:
                    loft_g = math.copysign(omega_max * speed / gravity_mps2, loft_g)
            loft = (0.0, g_to_mps2(loft_g, gravity_mps2), 0.0)
    commanded = add_vectors(pn, loft)
    v_hat = normalize(state.velocity, fallback=body_axes_for_state(state).forward)
    commanded = sub(commanded, scale(v_hat, dot(commanded, v_hat)))
    max_accel = g_to_mps2(
        guidance_cfg["maximum_lateral_acceleration_g"],
        config["atmosphere"]["gravity_mps2"],
    )
    commanded = clamp_norm(commanded, max_accel)
    if not enabled:
        commanded = (0.0, 0.0, 0.0)
        pn = (0.0, 0.0, 0.0)
        loft = (0.0, 0.0, 0.0)
        loft_active = False
    axes = body_axes_for_state(state)
    gravity_mps2 = float(config["atmosphere"]["gravity_mps2"])
    gravity_vector = (0.0, -gravity_mps2, 0.0)
    body_pitch_g = mps2_to_g(dot(commanded, axes.up), gravity_mps2)
    body_yaw_g = mps2_to_g(dot(commanded, axes.right), gravity_mps2)
    # Required specific force keeps the gravity term, then is radially
    # clamped to reqAccelMax so a 38 g kinematic command plus 1 g hold
    # does not become a 39 g controller demand.
    raw_required = clamp_norm(sub(commanded, gravity_vector), max_accel)
    body_pitch_sf_g = mps2_to_g(dot(raw_required, axes.up), gravity_mps2)
    body_yaw_sf_g = mps2_to_g(dot(raw_required, axes.right), gravity_mps2)
    wind_basis = cg_wind_normal_basis(state, config)
    required_specific_force = clamp_norm(
        sub(
            raw_required,
            scale(wind_basis.forward, dot(raw_required, wind_basis.forward)),
        ),
        max_accel,
    )
    gravity_compensation = scale(gravity_vector, -1.0)
    gravity_compensation = sub(
        gravity_compensation,
        scale(wind_basis.forward, dot(gravity_compensation, wind_basis.forward)),
    )
    wind_command = (
        mps2_to_g(dot(required_specific_force, wind_basis.up), gravity_mps2),
        mps2_to_g(dot(required_specific_force, wind_basis.right), gravity_mps2),
    )
    gravity_command = (
        mps2_to_g(dot(gravity_compensation, wind_basis.up), gravity_mps2),
        mps2_to_g(dot(gravity_compensation, wind_basis.right), gravity_mps2),
    )
    plant_semantics = str(config["control"].get("plant_semantics", ""))
    candidate = plant_semantics in {
        "body_cm_tail_force_moment",
        "generalized_aero_moment",
    }
    if candidate:
        controller_command = wind_command
    elif plant_semantics == "fin_torque_body_aoa":
        controller_command = (body_pitch_sf_g, body_yaw_sf_g)
    else:
        # Frozen H1/H2 compatibility: kinematic body command, no gravity term.
        controller_command = (body_pitch_g, body_yaw_g)
    return GuidanceOutput(
        enabled=enabled,
        range_m=range_m,
        closing_speed_mps=closing,
        los_rate_vector_rad_s=los_rate,
        pn_acceleration_mps2=pn,
        loft_acceleration_mps2=loft,
        commanded_acceleration_mps2=commanded,
        commanded_body_acceleration_g=(body_pitch_g, body_yaw_g),
        controller_specific_force_command_g=controller_command,
        wind_normal_specific_force_command_g=wind_command,
        gravity_compensation_wind_normal_g=gravity_command,
        wind_normal_pitch_axis=wind_basis.up,
        wind_normal_yaw_axis=wind_basis.right,
        effective_gain=effective_gain,
        time_to_go_s=time_to_go,
        loft_active=loft_active,
        within_lock_range=range_m <= guidance_cfg["lock_range_m"],
    )


def add_vectors(a: Vector, b: Vector) -> Vector:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])
