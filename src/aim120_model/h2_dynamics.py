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
from .dynamics import SimState, clamp_rates, state_is_finite
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
    total_specific_force_g: float
    drag_power_w: float
    lift_power_w: float
    pitch_angular_acceleration_rad_s2: float
    yaw_angular_acceleration_rad_s2: float


def _add_many(*vectors: Vector) -> Vector:
    result = (0.0, 0.0, 0.0)
    for vector in vectors:
        result = add(result, vector)
    return result


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
    pitch_g = dot(specific_force, aero.flow_normal_pitch) / gravity
    yaw_g = dot(specific_force, aero.flow_normal_yaw) / gravity
    axial_g = dot(specific_force, aero.air_velocity_hat) / gravity
    lateral_g = math.hypot(pitch_g, yaw_g)
    total_g = norm(specific_force) / gravity

    length = max(float(config["geometry"]["length_m"]), 1e-6)
    inertia_per_mass = max(length * length / 12.0, 1e-6)
    arm = float(config["aerodynamics"]["distance_cm_to_stabilizer_m"])
    angular_response_scale = float(config["control"].get("angular_response_scale", 1.0))
    damping = float(config["control"]["angular_damping"])
    pitch_angular_accel = (
        state.actual_pitch_acceleration_g * gravity * arm / inertia_per_mass * angular_response_scale
    ) - damping * state.pitch_rate
    yaw_angular_accel = (
        state.actual_yaw_acceleration_g * gravity * arm / inertia_per_mass * angular_response_scale
    ) - damping * state.yaw_rate
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
        total_specific_force_g=total_g,
        drag_power_w=dot(aero.drag_force_n, aero.air_velocity_mps),
        lift_power_w=dot(lift_force, aero.air_velocity_mps),
        pitch_angular_acceleration_rad_s2=pitch_angular_accel,
        yaw_angular_acceleration_rad_s2=yaw_angular_accel,
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
    "clamp_rates",
    "state_is_finite",
]
