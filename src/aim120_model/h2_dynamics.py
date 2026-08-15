"""H2 force bookkeeping and reduced-order translational/attitude dynamics.

H2 keeps the point-mass scope of H1, but makes the force directions and the
physical normal-load telemetry explicit.  The angular response remains a
reduced-order surrogate; it is not a claim about the game's hidden autopilot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

from .aerodynamics import H2AeroSample, StandardAtmosphere, compute_aerodynamics_h2
from .control import base_indicated_speed_schedule
from .dynamics import SimState, state_is_finite
from .math3d import Vector, add, dot, is_finite_vector, norm, scale
from .propulsion import PiecewisePropulsion, PropulsionSample


@dataclass(frozen=True)
class H2DynamicsDiagnostics:
    propulsion: PropulsionSample
    aero: H2AeroSample
    thrust_force_n: Vector
    drag_force_n: Vector
    natural_lift_force_n: Vector
    control_force_n: Vector
    gravity_force_n: Vector
    total_force_n: Vector
    non_gravity_acceleration_mps2: Vector
    specific_force_mps2: Vector
    acceleration_mps2: Vector
    axial_specific_force_g: float
    pitch_normal_acceleration_g: float
    yaw_normal_acceleration_g: float
    lateral_load_g: float
    trajectory_pitch_normal_acceleration_g: float
    trajectory_yaw_normal_acceleration_g: float
    trajectory_lateral_load_g: float
    total_specific_force_g: float
    drag_power_w: float
    lift_power_w: float
    pitch_angular_acceleration_rad_s2: float
    yaw_angular_acceleration_rad_s2: float
    pitch_body_aoa_force_g: float
    yaw_body_aoa_force_g: float
    pitch_fin_moment_equivalent_g: float
    yaw_fin_moment_equivalent_g: float
    pitch_tail_rate_incidence_rad: float
    yaw_tail_rate_incidence_rad: float
    pitch_tail_effective_incidence_rad: float
    yaw_tail_effective_incidence_rad: float
    pitch_natural_frequency_rad_s: float
    yaw_natural_frequency_rad_s: float
    pitch_tail_rate_damping_per_s: float
    yaw_tail_rate_damping_per_s: float
    pitch_residual_rate_damping_per_s: float
    yaw_residual_rate_damping_per_s: float


def _add_many(*vectors: Vector) -> Vector:
    result = (0.0, 0.0, 0.0)
    for vector in vectors:
        result = add(result, vector)
    return result


def _limit_unit_disk(pitch_value: float, yaw_value: float) -> tuple[float, float]:
    """Limit combined pitch/yaw authority without granting sqrt(2) more load."""

    magnitude = math.hypot(pitch_value, yaw_value)
    if magnitude <= 1.0 or magnitude <= 1e-12:
        return pitch_value, yaw_value
    scale_factor = 1.0 / magnitude
    return pitch_value * scale_factor, yaw_value * scale_factor


def forces_for_state_h2(
    state: SimState,
    time_s: float,
    config: dict[str, Any],
    propulsion: PiecewisePropulsion,
    powered: bool,
) -> H2DynamicsDiagnostics:
    atmosphere = StandardAtmosphere()
    prop = propulsion.sample(time_s, powered=powered)
    mass = max(float(state.mass), 1e-6)
    aero = compute_aerodynamics_h2(
        state.position[1],
        state.velocity,
        state.pitch,
        state.yaw,
        config,
        atmosphere,
    )
    gravity = float(config["atmosphere"]["gravity_mps2"])
    from .aerodynamics import body_axes

    axes = body_axes(state.pitch, state.yaw)
    thrust_force = scale(axes.forward, prop.thrust_n)
    plant_semantics = str(config["control"].get("plant_semantics", "direct_fin_g"))
    arm = float(config["aerodynamics"]["distance_cm_to_stabilizer_m"])
    pitch_body_aoa_force_g = (
        dot(aero.natural_lift_force_n, aero.flow_normal_pitch) / (gravity * mass)
    )
    yaw_body_aoa_force_g = (
        dot(aero.natural_lift_force_n, aero.flow_normal_yaw) / (gravity * mass)
    )
    pitch_fin_moment_equivalent_g = state.actual_pitch_acceleration_g
    yaw_fin_moment_equivalent_g = state.actual_yaw_acceleration_g
    pitch_tail_rate_incidence = 0.0
    yaw_tail_rate_incidence = 0.0
    pitch_tail_effective_incidence = 0.0
    yaw_tail_effective_incidence = 0.0
    scheduled_fins_g = 0.0
    pitch_fin_limit = 1.0
    yaw_fin_limit = 1.0
    speed_for_rate = max(aero.speed_mps, 1e-6)
    if plant_semantics == "fin_torque_body_aoa":
        fins_g = float(config["aerodynamics"]["fins_lateral_acceleration_g"])
        speed_schedule = base_indicated_speed_schedule(aero.dynamic_pressure_pa, config)
        scheduled_fins_g = fins_g * speed_schedule.fin_force_scale
        pitch_fin_limit = math.radians(
            max(float(config["aerodynamics"]["horizontal_fin_aoa_limit_deg"]), 1e-9)
        )
        yaw_fin_limit = math.radians(
            max(float(config["aerodynamics"]["vertical_fin_aoa_limit_deg"]), 1e-9)
        )

        # Tail force, and therefore tail moment, depends on local effective
        # incidence.  Body AoA is restoring; commanded fin angle is driving.
        pitch_tail_rate_incidence = state.pitch_rate * arm / speed_for_rate
        yaw_tail_rate_incidence = state.yaw_rate * arm / speed_for_rate
        pitch_tail_effective_incidence = (
            state.actual_pitch_fin_angle_rad
            - aero.pitch_alpha_rad
            - pitch_tail_rate_incidence
        )
        yaw_tail_effective_incidence = (
            state.actual_yaw_fin_angle_rad
            - aero.yaw_alpha_rad
            - yaw_tail_rate_incidence
        )
        pitch_moment_fraction, yaw_moment_fraction = _limit_unit_disk(
            pitch_tail_effective_incidence / pitch_fin_limit,
            yaw_tail_effective_incidence / yaw_fin_limit,
        )
        pitch_fin_moment_equivalent_g = scheduled_fins_g * pitch_moment_fraction
        yaw_fin_moment_equivalent_g = scheduled_fins_g * yaw_moment_fraction
    elif plant_semantics != "direct_fin_g":
        raise ValueError(f"unknown plant_semantics: {plant_semantics}")

    if plant_semantics == "fin_torque_body_aoa":
        # One resolved tail force supplies both its direct translation and its
        # moment.  Body CN-alpha lift is a separate airframe contribution; no
        # extra tail-force multiplier is introduced.
        control_force = _add_many(
            scale(aero.flow_normal_pitch, pitch_fin_moment_equivalent_g * gravity * mass),
            scale(aero.flow_normal_yaw, yaw_fin_moment_equivalent_g * gravity * mass),
        )
    else:
        control_force = _add_many(
            scale(
                aero.flow_normal_pitch,
                state.actual_pitch_acceleration_g * gravity * mass,
            ),
            scale(
                aero.flow_normal_yaw,
                state.actual_yaw_acceleration_g * gravity * mass,
            ),
        )
    gravity_force = (0.0, -mass * gravity, 0.0)
    total_force = _add_many(
        thrust_force,
        aero.drag_force_n,
        aero.natural_lift_force_n,
        control_force,
        gravity_force,
    )
    non_gravity = _add_many(
        thrust_force,
        aero.drag_force_n,
        aero.natural_lift_force_n,
        control_force,
    )
    specific_force = scale(non_gravity, 1.0 / mass)
    acceleration = scale(total_force, 1.0 / mass)
    # Acceleration-controller feedback belongs to the missile-mounted body
    # axes, matching guidance.commanded_body_acceleration_g exactly.
    pitch_g = dot(specific_force, axes.up) / gravity
    yaw_g = dot(specific_force, axes.right) / gravity
    trajectory_pitch_g = dot(specific_force, aero.flow_normal_pitch) / gravity
    trajectory_yaw_g = dot(specific_force, aero.flow_normal_yaw) / gravity
    axial_g = dot(specific_force, aero.air_velocity_hat) / gravity
    lateral_g = math.hypot(pitch_g, yaw_g)
    trajectory_lateral_g = math.hypot(trajectory_pitch_g, trajectory_yaw_g)
    total_g = norm(specific_force) / gravity

    length = max(float(config["geometry"]["length_m"]), 1e-6)
    inertia_per_mass = max(length * length / 12.0, 1e-6)
    pitch_natural_frequency = 0.0
    yaw_natural_frequency = 0.0
    pitch_tail_rate_damping = 0.0
    yaw_tail_rate_damping = 0.0
    pitch_residual_rate_damping = 0.0
    yaw_residual_rate_damping = 0.0
    if plant_semantics == "fin_torque_body_aoa":
        angular_response_scale = 1.0
        arm_magnitude = abs(arm)
        pitch_stiffness_s2 = max(
            scheduled_fins_g * gravity * arm_magnitude / (inertia_per_mass * pitch_fin_limit),
            0.0,
        )
        yaw_stiffness_s2 = max(
            scheduled_fins_g * gravity * arm_magnitude / (inertia_per_mass * yaw_fin_limit),
            0.0,
        )
        pitch_natural_frequency = math.sqrt(pitch_stiffness_s2)
        yaw_natural_frequency = math.sqrt(yaw_stiffness_s2)
        # The local-tail-flow term already contributes Cq = omega_n^2*r/V
        # in the linear region.  Supply only the unresolved remainder needed
        # for a critically damped closure; this adds no per-missile parameter.
        pitch_tail_rate_damping = pitch_stiffness_s2 * arm_magnitude / speed_for_rate
        yaw_tail_rate_damping = yaw_stiffness_s2 * arm_magnitude / speed_for_rate
        pitch_residual_rate_damping = max(
            0.0,
            2.0 * pitch_natural_frequency - pitch_tail_rate_damping,
        )
        yaw_residual_rate_damping = max(
            0.0,
            2.0 * yaw_natural_frequency - yaw_tail_rate_damping,
        )
        pitch_damping = pitch_residual_rate_damping
        yaw_damping = yaw_residual_rate_damping
    else:
        angular_response_scale = float(config["control"].get("angular_response_scale", 1.0))
        pitch_damping = float(config["control"]["angular_damping"])
        yaw_damping = pitch_damping
    pitch_moment_equivalent_g = pitch_fin_moment_equivalent_g
    yaw_moment_equivalent_g = yaw_fin_moment_equivalent_g
    if (
        plant_semantics == "direct_fin_g"
        and config["control"].get("fin_aoa_moment_enabled", False)
    ):
        fins_g = float(config["aerodynamics"]["fins_lateral_acceleration_g"])
        speed_schedule = base_indicated_speed_schedule(aero.dynamic_pressure_pa, config)
        fins_g *= speed_schedule.fin_force_scale
        pitch_fin_limit = math.radians(max(float(config["aerodynamics"]["horizontal_fin_aoa_limit_deg"]), 1e-9))
        yaw_fin_limit = math.radians(max(float(config["aerodynamics"]["vertical_fin_aoa_limit_deg"]), 1e-9))
        # The fin actuator supplies the commanded lateral load, while body AoA
        # supplies the stabilizing part of the tail moment.  This keeps CyK out
        # of the plant and makes fin angle, finsLatAccel and arm explicit.
        pitch_moment_equivalent_g -= fins_g * aero.pitch_alpha_rad / pitch_fin_limit
        yaw_moment_equivalent_g -= fins_g * aero.yaw_alpha_rad / yaw_fin_limit
    pitch_angular_accel = (
        pitch_moment_equivalent_g * gravity * arm / inertia_per_mass * angular_response_scale
    ) - pitch_damping * state.pitch_rate
    yaw_angular_accel = (
        yaw_moment_equivalent_g * gravity * arm / inertia_per_mass * angular_response_scale
    ) - yaw_damping * state.yaw_rate
    lift_force = _add_many(aero.natural_lift_force_n, control_force)
    return H2DynamicsDiagnostics(
        propulsion=prop,
        aero=aero,
        thrust_force_n=thrust_force,
        drag_force_n=aero.drag_force_n,
        natural_lift_force_n=aero.natural_lift_force_n,
        control_force_n=control_force,
        gravity_force_n=gravity_force,
        total_force_n=total_force,
        non_gravity_acceleration_mps2=specific_force,
        specific_force_mps2=specific_force,
        acceleration_mps2=acceleration,
        axial_specific_force_g=axial_g,
        pitch_normal_acceleration_g=pitch_g,
        yaw_normal_acceleration_g=yaw_g,
        lateral_load_g=lateral_g,
        trajectory_pitch_normal_acceleration_g=trajectory_pitch_g,
        trajectory_yaw_normal_acceleration_g=trajectory_yaw_g,
        trajectory_lateral_load_g=trajectory_lateral_g,
        total_specific_force_g=total_g,
        drag_power_w=dot(aero.drag_force_n, aero.air_velocity_mps),
        lift_power_w=dot(lift_force, aero.air_velocity_mps),
        pitch_angular_acceleration_rad_s2=pitch_angular_accel,
        yaw_angular_acceleration_rad_s2=yaw_angular_accel,
        pitch_body_aoa_force_g=pitch_body_aoa_force_g,
        yaw_body_aoa_force_g=yaw_body_aoa_force_g,
        pitch_fin_moment_equivalent_g=pitch_fin_moment_equivalent_g,
        yaw_fin_moment_equivalent_g=yaw_fin_moment_equivalent_g,
        pitch_tail_rate_incidence_rad=pitch_tail_rate_incidence,
        yaw_tail_rate_incidence_rad=yaw_tail_rate_incidence,
        pitch_tail_effective_incidence_rad=pitch_tail_effective_incidence,
        yaw_tail_effective_incidence_rad=yaw_tail_effective_incidence,
        pitch_natural_frequency_rad_s=pitch_natural_frequency,
        yaw_natural_frequency_rad_s=yaw_natural_frequency,
        pitch_tail_rate_damping_per_s=pitch_tail_rate_damping,
        yaw_tail_rate_damping_per_s=yaw_tail_rate_damping,
        pitch_residual_rate_damping_per_s=pitch_residual_rate_damping,
        yaw_residual_rate_damping_per_s=yaw_residual_rate_damping,
    )


def _dynamic_vector(state: SimState) -> tuple[float, ...]:
    return (
        state.position[0], state.position[1], state.position[2],
        state.velocity[0], state.velocity[1], state.velocity[2],
        state.pitch, state.yaw, state.pitch_rate, state.yaw_rate, state.mass,
    )


def _state_from_dynamic(state: SimState, values: tuple[float, ...]) -> SimState:
    return replace(
        state,
        position=(values[0], values[1], values[2]),
        velocity=(values[3], values[4], values[5]),
        pitch=values[6],
        yaw=values[7],
        pitch_rate=values[8],
        yaw_rate=values[9],
        mass=values[10],
    )


def _derivative_vector(
    state: SimState,
    time_s: float,
    config: dict[str, Any],
    propulsion: PiecewisePropulsion,
    powered: bool,
) -> tuple[float, ...]:
    diagnostics = forces_for_state_h2(state, time_s, config, propulsion, powered)
    return (
        state.velocity[0], state.velocity[1], state.velocity[2],
        diagnostics.acceleration_mps2[0], diagnostics.acceleration_mps2[1], diagnostics.acceleration_mps2[2],
        state.pitch_rate, state.yaw_rate,
        diagnostics.pitch_angular_acceleration_rad_s2,
        diagnostics.yaw_angular_acceleration_rad_s2,
        diagnostics.propulsion.mass_flow_kg_s,
    )


def _combine(values: tuple[float, ...], derivative: tuple[float, ...], factor: float) -> tuple[float, ...]:
    return tuple(value + factor * delta for value, delta in zip(values, derivative))


def rk4_step_h2(
    state: SimState,
    time_s: float,
    dt_s: float,
    config: dict[str, Any],
    propulsion: PiecewisePropulsion,
    powered: bool,
) -> SimState:
    values = _dynamic_vector(state)
    k1 = _derivative_vector(state, time_s, config, propulsion, powered)
    s2 = _state_from_dynamic(state, _combine(values, k1, dt_s / 2.0))
    k2 = _derivative_vector(s2, time_s + dt_s / 2.0, config, propulsion, powered)
    s3 = _state_from_dynamic(state, _combine(values, k2, dt_s / 2.0))
    k3 = _derivative_vector(s3, time_s + dt_s / 2.0, config, propulsion, powered)
    s4 = _state_from_dynamic(state, _combine(values, k3, dt_s))
    k4 = _derivative_vector(s4, time_s + dt_s, config, propulsion, powered)
    next_values = tuple(
        value + dt_s * (d1 + 2.0 * d2 + 2.0 * d3 + d4) / 6.0
        for value, d1, d2, d3, d4 in zip(values, k1, k2, k3, k4)
    )
    return _state_from_dynamic(state, next_values)


__all__ = [
    "H2DynamicsDiagnostics",
    "forces_for_state_h2",
    "rk4_step_h2",
    "state_is_finite",
]
