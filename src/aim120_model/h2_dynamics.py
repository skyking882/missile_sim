"""H2 force bookkeeping and reduced-order translational/attitude dynamics.

H2 keeps the point-mass scope of H1, but makes the force directions and the
physical normal-load telemetry explicit.  The angular response remains a
reduced-order surrogate; it is not a claim about the game's hidden autopilot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

from .aerodynamics import (
    BodyAxes,
    H2AeroSample,
    StandardAtmosphere,
    body_axes_for_state,
    cg_wind_normal_basis,
    caliber_circular_area,
    compute_aerodynamics_h2,
    normalize_quaternion,
    pitch_yaw_from_quaternion,
    quaternion_derivative,
    quaternion_from_pitch_yaw,
)
from .control import base_indicated_speed_schedule
from .dynamics import SimState, state_is_finite
from .math3d import Vector, add, cross, dot, is_finite_vector, norm, normalize, scale, sub
from .propulsion import PiecewisePropulsion, PropulsionSample


@dataclass(frozen=True)
class H2DynamicsDiagnostics:
    propulsion: PropulsionSample
    aero: H2AeroSample
    thrust_force_n: Vector
    drag_force_n: Vector
    natural_lift_force_n: Vector
    fixed_lifting_surface_force_n: Vector
    control_force_n: Vector
    gravity_force_n: Vector
    total_force_n: Vector
    non_gravity_acceleration_mps2: Vector
    specific_force_mps2: Vector
    acceleration_mps2: Vector
    axial_specific_force_g: float
    pitch_normal_acceleration_g: float
    yaw_normal_acceleration_g: float
    wind_normal_pitch_acceleration_g: float
    wind_normal_yaw_acceleration_g: float
    lateral_load_g: float
    trajectory_pitch_normal_acceleration_g: float
    trajectory_yaw_normal_acceleration_g: float
    trajectory_lateral_load_g: float
    total_specific_force_g: float
    drag_power_w: float
    lift_power_w: float
    body_tail_force_power_at_cg_w: float
    pitch_angular_acceleration_rad_s2: float
    yaw_angular_acceleration_rad_s2: float
    body_reference_area_m2: float
    body_reference_length_m: float
    body_cp_cg_arm_over_diameter: float
    body_cn_alpha_per_rad: float
    body_cn_q: float
    body_cm_alpha_per_rad: float
    body_cm_q: float
    generalized_cm_alpha_dot_per_rad: float
    generalized_pitch_alpha_dot_hat: float
    generalized_yaw_alpha_dot_hat: float
    generalized_cm_alpha_dot_runtime_enabled: bool
    pitch_body_normal_force_n: float
    yaw_body_normal_force_n: float
    pitch_body_static_moment_nm: float
    yaw_body_static_moment_nm: float
    pitch_body_rate_moment_nm: float
    yaw_body_rate_moment_nm: float
    pitch_body_total_moment_nm: float
    yaw_body_total_moment_nm: float
    fixed_lifting_surface_multiplier: float
    body_normal_force_area_slope_m2_per_rad: float
    fixed_lifting_surface_area_slope_m2_per_rad: float
    fixed_lifting_surface_station_x_m: float
    pitch_fixed_lifting_surface_alpha_rad: float
    yaw_fixed_lifting_surface_alpha_rad: float
    pitch_fixed_lifting_surface_force_n: float
    yaw_fixed_lifting_surface_force_n: float
    pitch_fixed_lifting_surface_moment_nm: float
    yaw_fixed_lifting_surface_moment_nm: float
    pitch_total_body_wing_tail_normal_force_n: float
    yaw_total_body_wing_tail_normal_force_n: float
    tail_station_x_m: float
    pitch_tail_force_n: float
    yaw_tail_force_n: float
    pitch_tail_moment_nm: float
    yaw_tail_moment_nm: float
    pitch_tail_authority_fraction: float
    yaw_tail_authority_fraction: float
    tail_alpha_force_multiplier: float
    tail_delta_force_multiplier: float
    pitch_tail_alpha_force_slope_n_per_rad: float
    yaw_tail_alpha_force_slope_n_per_rad: float
    pitch_tail_delta_force_slope_n_per_rad: float
    yaw_tail_delta_force_slope_n_per_rad: float
    pitch_tail_alpha_force_n: float
    yaw_tail_alpha_force_n: float
    pitch_tail_delta_force_n: float
    yaw_tail_delta_force_n: float
    pitch_tail_net_force_pre_cap_n: float
    yaw_tail_net_force_pre_cap_n: float
    tail_force_cap_n: float
    tail_force_cap_scale: float
    tail_force_cap_active: bool
    pitch_tail_alpha_moment_nm: float
    yaw_tail_alpha_moment_nm: float
    pitch_tail_delta_moment_nm: float
    yaw_tail_delta_moment_nm: float
    pitch_residual_damping_moment_nm: float
    yaw_residual_damping_moment_nm: float
    pitch_total_moment_nm: float
    yaw_total_moment_nm: float
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


@dataclass(frozen=True)
class SplitTailForce:
    """Algebraic split of the empirical tail authority on two normal axes."""

    pitch_alpha_slope_n_per_rad: float
    yaw_alpha_slope_n_per_rad: float
    pitch_delta_slope_n_per_rad: float
    yaw_delta_slope_n_per_rad: float
    pitch_alpha_force_n: float
    yaw_alpha_force_n: float
    pitch_delta_force_n: float
    yaw_delta_force_n: float
    pitch_pre_cap_force_n: float
    yaw_pre_cap_force_n: float
    pitch_force_n: float
    yaw_force_n: float
    force_cap_n: float
    cap_scale: float
    cap_active: bool


def split_empirical_tail_force(
    *,
    mass_kg: float,
    gravity_mps2: float,
    dynamic_pressure_pa: float,
    q_base_pa: float,
    acceleration_authority_g: float,
    pitch_alpha_rad: float,
    yaw_alpha_rad: float,
    pitch_delta_rad: float,
    yaw_delta_rad: float,
    pitch_authority_reference_rad: float,
    yaw_authority_reference_rad: float,
    tail_alpha_force_multiplier: float = 1.0,
    tail_delta_force_multiplier: float = 1.0,
) -> SplitTailForce:
    """Return q*Aeq*(k_alpha*alpha-k_delta*delta), radially capped.

    ``Aeq`` is an empirical equivalent authority area.  It is not a measured
    fixed-tail area or a War Thunder aerodynamic derivative.
    """

    base_force_per_reference = (
        mass_kg * gravity_mps2 * acceleration_authority_g
        * dynamic_pressure_pa / q_base_pa
    )
    pitch_base_slope = base_force_per_reference / pitch_authority_reference_rad
    yaw_base_slope = base_force_per_reference / yaw_authority_reference_rad
    pitch_alpha_slope = pitch_base_slope * tail_alpha_force_multiplier
    yaw_alpha_slope = yaw_base_slope * tail_alpha_force_multiplier
    pitch_delta_slope = -pitch_base_slope * tail_delta_force_multiplier
    yaw_delta_slope = -yaw_base_slope * tail_delta_force_multiplier
    pitch_alpha_force = pitch_alpha_slope * pitch_alpha_rad
    yaw_alpha_force = yaw_alpha_slope * yaw_alpha_rad
    pitch_delta_force = pitch_delta_slope * pitch_delta_rad
    yaw_delta_force = yaw_delta_slope * yaw_delta_rad
    pitch_pre_cap = pitch_alpha_force + pitch_delta_force
    yaw_pre_cap = yaw_alpha_force + yaw_delta_force
    force_cap = max(base_force_per_reference, 0.0)
    magnitude = math.hypot(pitch_pre_cap, yaw_pre_cap)
    cap_scale = 1.0 if magnitude <= force_cap or magnitude <= 1e-12 else force_cap / magnitude
    return SplitTailForce(
        pitch_alpha_slope_n_per_rad=pitch_alpha_slope,
        yaw_alpha_slope_n_per_rad=yaw_alpha_slope,
        pitch_delta_slope_n_per_rad=pitch_delta_slope,
        yaw_delta_slope_n_per_rad=yaw_delta_slope,
        pitch_alpha_force_n=pitch_alpha_force,
        yaw_alpha_force_n=yaw_alpha_force,
        pitch_delta_force_n=pitch_delta_force,
        yaw_delta_force_n=yaw_delta_force,
        pitch_pre_cap_force_n=pitch_pre_cap,
        yaw_pre_cap_force_n=yaw_pre_cap,
        pitch_force_n=pitch_pre_cap * cap_scale,
        yaw_force_n=yaw_pre_cap * cap_scale,
        force_cap_n=force_cap,
        cap_scale=cap_scale,
        cap_active=cap_scale < 1.0,
    )


def combined_empirical_tail_force_reference(**kwargs: float) -> SplitTailForce:
    """Frozen combined-chi reference used only for k=1 equivalence tests."""

    mass = float(kwargs["mass_kg"])
    gravity = float(kwargs["gravity_mps2"])
    q_inf = float(kwargs["dynamic_pressure_pa"])
    q_base = float(kwargs["q_base_pa"])
    authority_g = float(kwargs["acceleration_authority_g"])
    pitch_reference = float(kwargs["pitch_authority_reference_rad"])
    yaw_reference = float(kwargs["yaw_authority_reference_rad"])
    force_cap = mass * gravity * authority_g * q_inf / q_base
    pitch_slope = force_cap / pitch_reference
    yaw_slope = force_cap / yaw_reference
    pitch_alpha_force = pitch_slope * float(kwargs["pitch_alpha_rad"])
    yaw_alpha_force = yaw_slope * float(kwargs["yaw_alpha_rad"])
    pitch_delta_force = -pitch_slope * float(kwargs["pitch_delta_rad"])
    yaw_delta_force = -yaw_slope * float(kwargs["yaw_delta_rad"])
    pitch_pre_cap = force_cap * (
        float(kwargs["pitch_alpha_rad"]) - float(kwargs["pitch_delta_rad"])
    ) / pitch_reference
    yaw_pre_cap = force_cap * (
        float(kwargs["yaw_alpha_rad"]) - float(kwargs["yaw_delta_rad"])
    ) / yaw_reference
    magnitude = math.hypot(pitch_pre_cap, yaw_pre_cap)
    cap_scale = 1.0 if magnitude <= force_cap or magnitude <= 1e-12 else force_cap / magnitude
    return SplitTailForce(
        pitch_alpha_slope_n_per_rad=pitch_slope,
        yaw_alpha_slope_n_per_rad=yaw_slope,
        pitch_delta_slope_n_per_rad=-pitch_slope,
        yaw_delta_slope_n_per_rad=-yaw_slope,
        pitch_alpha_force_n=pitch_alpha_force,
        yaw_alpha_force_n=yaw_alpha_force,
        pitch_delta_force_n=pitch_delta_force,
        yaw_delta_force_n=yaw_delta_force,
        pitch_pre_cap_force_n=pitch_pre_cap,
        yaw_pre_cap_force_n=yaw_pre_cap,
        pitch_force_n=pitch_pre_cap * cap_scale,
        yaw_force_n=yaw_pre_cap * cap_scale,
        force_cap_n=force_cap,
        cap_scale=cap_scale,
        cap_active=cap_scale < 1.0,
    )


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


def _allocate_empirical_authority_disk(
    pitch_demand: float,
    yaw_demand: float,
) -> tuple[float, float]:
    """Radially allocate empirical fin authority; this is not a stall model."""

    return _limit_unit_disk(pitch_demand, yaw_demand)


def _body_angular_velocity_inertial(state: SimState, axes: BodyAxes) -> Vector:
    return add(scale(axes.right, state.pitch_rate), scale(axes.up, -state.yaw_rate))


def _station_velocity_inertial(state: SimState, axes: BodyAxes, station_x_m: float) -> Vector:
    arm = scale(axes.forward, station_x_m)
    return add(state.velocity, cross(_body_angular_velocity_inertial(state, axes), arm))


def _flow_at_station(
    state: SimState,
    axes: BodyAxes,
    station_x_m: float,
    config: dict[str, Any],
) -> tuple[float, float, float, Vector, Vector]:
    wind = tuple(float(value) for value in config["atmosphere"].get("wind_mps", (0.0, 0.0, 0.0)))
    local_air_velocity = sub(_station_velocity_inertial(state, axes, station_x_m), wind)
    speed = norm(local_air_velocity)
    local_hat = normalize(local_air_velocity, fallback=axes.forward)
    forward = dot(local_hat, axes.forward)
    pitch_alpha = math.atan2(-dot(local_hat, axes.up), forward)
    yaw_alpha = math.atan2(-dot(local_hat, axes.right), forward)
    pitch_normal = normalize(
        sub(axes.up, scale(local_hat, dot(axes.up, local_hat))),
        fallback=axes.up,
    )
    yaw_normal = normalize(
        sub(axes.right, scale(local_hat, dot(axes.right, local_hat))),
        fallback=axes.right,
    )
    return speed, pitch_alpha, yaw_alpha, pitch_normal, yaw_normal


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
    plant_semantics = str(config["control"].get("plant_semantics", "direct_fin_g"))
    legacy_fin_torque_plant = plant_semantics == "fin_torque_body_aoa"
    body_cm_tail_plant = plant_semantics == "body_cm_tail_force_moment"
    generalized_aero_plant = plant_semantics == "generalized_aero_moment"
    if plant_semantics not in {
        "direct_fin_g",
        "fin_torque_body_aoa",
        "body_cm_tail_force_moment",
        "generalized_aero_moment",
    }:
        raise ValueError(f"unknown plant_semantics: {plant_semantics}")
    axes = body_axes_for_state(state)
    normal_force_velocity = None
    if body_cm_tail_plant:
        body_candidate = config["aerodynamics"]["body_cp_force_candidate"]
        body_station_x_m = (
            float(body_candidate["cp_cg_arm_over_diameter"])
            * float(config["geometry"]["caliber_m"])
        )
        normal_force_velocity = _station_velocity_inertial(state, axes, body_station_x_m)
    aero = compute_aerodynamics_h2(
        state.position[1],
        state.velocity,
        state.pitch,
        state.yaw,
        config,
        atmosphere,
        axes_override=axes,
        normal_force_velocity_mps=normal_force_velocity,
    )
    # The generalized candidate deliberately resolves normal force and pitch/
    # yaw moment as separate nondimensional outputs.  It has no force station:
    # the shared inputs are CG flow alpha, actual fin angle, body rate, and the
    # local alpha-rate proxy.  This block is inactive for legacy and v1 body-Cm
    # candidates, preserving their existing equations point-for-point.
    generalized_body_force_vector = (0.0, 0.0, 0.0)
    generalized_delta_force_vector = (0.0, 0.0, 0.0)
    generalized_pitch_body_static_moment = 0.0
    generalized_yaw_body_static_moment = 0.0
    generalized_pitch_body_rate_moment = 0.0
    generalized_yaw_body_rate_moment = 0.0
    generalized_pitch_delta_moment = 0.0
    generalized_yaw_delta_moment = 0.0
    generalized_pitch_alpha_force = 0.0
    generalized_yaw_alpha_force = 0.0
    generalized_pitch_delta_force = 0.0
    generalized_yaw_delta_force = 0.0
    generalized_pitch_alpha_force_slope = 0.0
    generalized_yaw_alpha_force_slope = 0.0
    generalized_pitch_delta_force_slope = 0.0
    generalized_yaw_delta_force_slope = 0.0
    generalized_pitch_alpha_dot_hat = 0.0
    generalized_yaw_alpha_dot_hat = 0.0
    generalized_cm_alpha_dot_per_rad = 0.0
    generalized_cm_alpha_dot_runtime_enabled = False
    generalized_pitch_cm_rate = 0.0
    generalized_yaw_cm_rate = 0.0
    generalized_reference_area_m2 = 0.0
    generalized_reference_length_m = 0.0
    if generalized_aero_plant:
        generalized = config["aerodynamics"].get("generalized_aero_moment_candidate")
        if not isinstance(generalized, dict):
            raise ValueError(
                "generalized_aero_moment requires generalized_aero_moment_candidate"
            )
        generalized_reference_area_m2 = caliber_circular_area(config)
        generalized_reference_length_m = float(config["geometry"]["caliber_m"])
        speed_for_generalized = max(float(aero.speed_mps), 1e-6)
        q_dyn = float(aero.normal_force_dynamic_pressure_pa)
        q_hat_pitch = state.pitch_rate * generalized_reference_length_m / (2.0 * speed_for_generalized)
        q_hat_yaw = state.yaw_rate * generalized_reference_length_m / (2.0 * speed_for_generalized)
        # The current reduced state has body rate but no independently
        # integrated alpha-dot measurement.  Do not alias alpha-dot to q:
        # that would make Cm_q and Cm_alpha-dot structurally collinear.
        generalized_pitch_alpha_dot_hat = 0.0
        generalized_yaw_alpha_dot_hat = 0.0
        pitch_alpha = float(aero.pitch_alpha_rad)
        yaw_alpha = float(aero.yaw_alpha_rad)
        pitch_delta = float(state.actual_pitch_fin_angle_rad)
        yaw_delta = float(state.actual_yaw_fin_angle_rad)
        cn_alpha = float(generalized["cn_alpha_per_rad"])
        cn_delta = float(generalized["cn_delta_per_rad"])
        cm_alpha = float(generalized["cm_alpha_per_rad"])
        cm_delta = float(generalized["cm_delta_per_rad"])
        cm_q = float(generalized["cm_q"])
        generalized_pitch_alpha_force_slope = q_dyn * generalized_reference_area_m2 * cn_alpha
        generalized_yaw_alpha_force_slope = generalized_pitch_alpha_force_slope
        generalized_pitch_delta_force_slope = q_dyn * generalized_reference_area_m2 * cn_delta
        generalized_yaw_delta_force_slope = generalized_pitch_delta_force_slope
        generalized_pitch_alpha_force = generalized_pitch_alpha_force_slope * pitch_alpha
        generalized_yaw_alpha_force = generalized_yaw_alpha_force_slope * yaw_alpha
        generalized_pitch_delta_force = generalized_pitch_delta_force_slope * pitch_delta
        generalized_yaw_delta_force = generalized_yaw_delta_force_slope * yaw_delta
        generalized_body_force_vector = _add_many(
            scale(aero.flow_normal_pitch, generalized_pitch_alpha_force),
            scale(aero.flow_normal_yaw, generalized_yaw_alpha_force),
        )
        generalized_delta_force_vector = _add_many(
            scale(aero.flow_normal_pitch, generalized_pitch_delta_force),
            scale(aero.flow_normal_yaw, generalized_yaw_delta_force),
        )
        generalized_pitch_body_static_moment = (
            q_dyn * generalized_reference_area_m2 * generalized_reference_length_m
            * cm_alpha * pitch_alpha
        )
        generalized_yaw_body_static_moment = (
            q_dyn * generalized_reference_area_m2 * generalized_reference_length_m
            * cm_alpha * yaw_alpha
        )
        generalized_pitch_cm_rate = cm_q * q_hat_pitch
        generalized_yaw_cm_rate = cm_q * q_hat_yaw
        generalized_pitch_body_rate_moment = (
            q_dyn * generalized_reference_area_m2 * generalized_reference_length_m
            * generalized_pitch_cm_rate
        )
        generalized_yaw_body_rate_moment = (
            q_dyn * generalized_reference_area_m2 * generalized_reference_length_m
            * generalized_yaw_cm_rate
        )
        generalized_pitch_delta_moment = (
            q_dyn * generalized_reference_area_m2 * generalized_reference_length_m
            * cm_delta * pitch_delta
        )
        generalized_yaw_delta_moment = (
            q_dyn * generalized_reference_area_m2 * generalized_reference_length_m
            * cm_delta * yaw_delta
        )
    body_normal_force_vector = (
        generalized_body_force_vector
        if generalized_aero_plant
        else aero.natural_lift_force_n
    )
    gravity = float(config["atmosphere"]["gravity_mps2"])
    thrust_force = scale(axes.forward, prop.thrust_n)
    arm = float(config["aerodynamics"]["distance_cm_to_stabilizer_m"])
    pitch_body_aoa_force_g = (
        dot(body_normal_force_vector, aero.flow_normal_pitch) / (gravity * mass)
    )
    yaw_body_aoa_force_g = (
        dot(body_normal_force_vector, aero.flow_normal_yaw) / (gravity * mass)
    )
    fixed_lifting_surface_multiplier = 0.0
    body_normal_force_area_slope = 0.0
    fixed_lifting_surface_area_slope = 0.0
    fixed_lifting_surface_station_x = 0.0
    pitch_fixed_lifting_surface_alpha = 0.0
    yaw_fixed_lifting_surface_alpha = 0.0
    pitch_fixed_lifting_surface_force = 0.0
    yaw_fixed_lifting_surface_force = 0.0
    pitch_fixed_lifting_surface_moment = 0.0
    yaw_fixed_lifting_surface_moment = 0.0
    fixed_lifting_surface_force_vector = (0.0, 0.0, 0.0)
    cg_flow_normal_pitch = aero.flow_normal_pitch
    cg_flow_normal_yaw = aero.flow_normal_yaw
    if body_cm_tail_plant:
        fixed_lift = config["aerodynamics"].get("fixed_lifting_surface_candidate")
    else:
        fixed_lift = None
    if isinstance(fixed_lift, dict):
        fixed_lifting_surface_multiplier = float(
            fixed_lift["fixed_lifting_surface_multiplier"]
        )
        body_normal_force_area_slope = float(
            fixed_lift["body_normal_force_area_slope_m2_per_rad"]
        )
        fixed_lifting_surface_area_slope = (
            fixed_lifting_surface_multiplier * body_normal_force_area_slope
        )
        fixed_lifting_surface_station_x = float(fixed_lift["station_x_m"])
        if abs(fixed_lifting_surface_station_x) <= 1e-15:
            # Preserve the prior near-CG diagnostic exactly: x_W=0 used the
            # CG station flow and the common CG dynamic pressure.  Only a
            # non-zero station introduces local-q/alpha changes.
            (
                fixed_lifting_surface_speed,
                pitch_fixed_lifting_surface_alpha,
                yaw_fixed_lifting_surface_alpha,
                fixed_flow_normal_pitch,
                fixed_flow_normal_yaw,
            ) = _flow_at_station(state, axes, 0.0, config)
            fixed_lifting_surface_dynamic_pressure = aero.dynamic_pressure_pa
        else:
            (
                fixed_lifting_surface_speed,
                pitch_fixed_lifting_surface_alpha,
                yaw_fixed_lifting_surface_alpha,
                fixed_flow_normal_pitch,
                fixed_flow_normal_yaw,
            ) = _flow_at_station(
                state, axes, fixed_lifting_surface_station_x, config
            )
            fixed_lifting_surface_dynamic_pressure = (
                0.5
                * aero.atmosphere.density_kg_m3
                * fixed_lifting_surface_speed
                * fixed_lifting_surface_speed
            )
        pitch_fixed_lifting_surface_force = (
            fixed_lifting_surface_dynamic_pressure
            * fixed_lifting_surface_area_slope
            * pitch_fixed_lifting_surface_alpha
        )
        yaw_fixed_lifting_surface_force = (
            fixed_lifting_surface_dynamic_pressure
            * fixed_lifting_surface_area_slope
            * yaw_fixed_lifting_surface_alpha
        )
        fixed_lifting_surface_force_vector = _add_many(
            scale(fixed_flow_normal_pitch, pitch_fixed_lifting_surface_force),
            scale(fixed_flow_normal_yaw, yaw_fixed_lifting_surface_force),
        )
        fixed_lifting_surface_arm_vector = scale(
            axes.forward, fixed_lifting_surface_station_x
        )
        fixed_lifting_surface_moment_vector = cross(
            fixed_lifting_surface_arm_vector, fixed_lifting_surface_force_vector
        )
        pitch_fixed_lifting_surface_moment = dot(
            fixed_lifting_surface_moment_vector, axes.right
        )
        yaw_fixed_lifting_surface_moment = -dot(
            fixed_lifting_surface_moment_vector, axes.up
        )
    pitch_fin_moment_equivalent_g = state.actual_pitch_acceleration_g
    yaw_fin_moment_equivalent_g = state.actual_yaw_acceleration_g
    pitch_fin_translation_equivalent_g = state.actual_pitch_acceleration_g
    yaw_fin_translation_equivalent_g = state.actual_yaw_acceleration_g
    pitch_tail_rate_incidence = 0.0
    yaw_tail_rate_incidence = 0.0
    pitch_tail_effective_incidence = 0.0
    yaw_tail_effective_incidence = 0.0
    scheduled_fins_g = 0.0
    pitch_fin_limit = 1.0
    yaw_fin_limit = 1.0
    tail_station_x_m = arm
    pitch_tail_force_n = 0.0
    yaw_tail_force_n = 0.0
    pitch_tail_moment_nm = 0.0
    yaw_tail_moment_nm = 0.0
    pitch_tail_force_vector = (0.0, 0.0, 0.0)
    yaw_tail_force_vector = (0.0, 0.0, 0.0)
    pitch_moment_fraction = 0.0
    yaw_moment_fraction = 0.0
    tail_alpha_force_multiplier = 0.0
    tail_delta_force_multiplier = 0.0
    pitch_tail_alpha_force_slope = 0.0
    yaw_tail_alpha_force_slope = 0.0
    pitch_tail_delta_force_slope = 0.0
    yaw_tail_delta_force_slope = 0.0
    pitch_tail_alpha_force = 0.0
    yaw_tail_alpha_force = 0.0
    pitch_tail_delta_force = 0.0
    yaw_tail_delta_force = 0.0
    pitch_tail_net_force_pre_cap = 0.0
    yaw_tail_net_force_pre_cap = 0.0
    tail_force_cap = 0.0
    tail_force_cap_scale = 1.0
    tail_force_cap_active = False
    pitch_tail_alpha_moment = 0.0
    yaw_tail_alpha_moment = 0.0
    pitch_tail_delta_moment = 0.0
    yaw_tail_delta_moment = 0.0
    speed_for_rate = max(aero.speed_mps, 1e-6)
    tail_flow_normal_pitch = aero.flow_normal_pitch
    tail_flow_normal_yaw = aero.flow_normal_yaw
    if legacy_fin_torque_plant or body_cm_tail_plant:
        fins_g = float(config["aerodynamics"]["fins_lateral_acceleration_g"])
        speed_schedule = base_indicated_speed_schedule(aero.dynamic_pressure_pa, config)
        scheduled_fins_g = fins_g * speed_schedule.fin_force_scale
        if body_cm_tail_plant:
            pitch_fin_limit = float(config["control"]["fin_actuator_travel"]["pitch_limit_rad"])
            yaw_fin_limit = float(config["control"]["fin_actuator_travel"]["yaw_limit_rad"])
        else:
            pitch_fin_limit = math.radians(
                max(float(config["aerodynamics"]["horizontal_fin_aoa_limit_deg"]), 1e-9)
            )
            yaw_fin_limit = math.radians(
                max(float(config["aerodynamics"]["vertical_fin_aoa_limit_deg"]), 1e-9)
            )

        if body_cm_tail_plant:
            tail_station_x_m = float(config["aerodynamics"]["tail_station_x_m"])
            (
                _tail_speed,
                tail_pitch_alpha,
                tail_yaw_alpha,
                tail_flow_normal_pitch,
                tail_flow_normal_yaw,
            ) = _flow_at_station(
                state,
                axes,
                tail_station_x_m,
                config,
            )
            (
                _cg_speed,
                cg_pitch_alpha,
                cg_yaw_alpha,
                _cg_pitch_normal,
                _cg_yaw_normal,
            ) = _flow_at_station(
                state,
                axes,
                0.0,
                config,
            )
            pitch_tail_rate_incidence = tail_pitch_alpha - cg_pitch_alpha
            yaw_tail_rate_incidence = tail_yaw_alpha - cg_yaw_alpha
            pitch_tail_effective_incidence = state.actual_pitch_fin_angle_rad - tail_pitch_alpha
            yaw_tail_effective_incidence = state.actual_yaw_fin_angle_rad - tail_yaw_alpha
            empirical_authority = config["aerodynamics"]["empirical_fin_authority"]
            pitch_authority_reference = float(empirical_authority["pitch_incidence_reference_rad"])
            yaw_authority_reference = float(empirical_authority["yaw_incidence_reference_rad"])
        else:
            # Legacy v12 stores an unsigned arm and is intentionally unchanged.
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
            pitch_authority_reference = pitch_fin_limit
            yaw_authority_reference = yaw_fin_limit
        if body_cm_tail_plant:
            split_config = config["aerodynamics"]["split_tail_candidate"]
            tail_alpha_force_multiplier = float(
                split_config["tail_alpha_force_multiplier"]
            )
            tail_delta_force_multiplier = float(
                split_config["tail_delta_force_multiplier"]
            )
            split_tail = split_empirical_tail_force(
                mass_kg=mass,
                gravity_mps2=gravity,
                dynamic_pressure_pa=aero.dynamic_pressure_pa,
                q_base_pa=float(split_config["q_base_pa"]),
                acceleration_authority_g=fins_g,
                pitch_alpha_rad=tail_pitch_alpha,
                yaw_alpha_rad=tail_yaw_alpha,
                pitch_delta_rad=state.actual_pitch_fin_angle_rad,
                yaw_delta_rad=state.actual_yaw_fin_angle_rad,
                pitch_authority_reference_rad=pitch_authority_reference,
                yaw_authority_reference_rad=yaw_authority_reference,
                tail_alpha_force_multiplier=tail_alpha_force_multiplier,
                tail_delta_force_multiplier=tail_delta_force_multiplier,
            )
            pitch_tail_alpha_force_slope = split_tail.pitch_alpha_slope_n_per_rad
            yaw_tail_alpha_force_slope = split_tail.yaw_alpha_slope_n_per_rad
            pitch_tail_delta_force_slope = split_tail.pitch_delta_slope_n_per_rad
            yaw_tail_delta_force_slope = split_tail.yaw_delta_slope_n_per_rad
            pitch_tail_alpha_force = split_tail.pitch_alpha_force_n
            yaw_tail_alpha_force = split_tail.yaw_alpha_force_n
            pitch_tail_delta_force = split_tail.pitch_delta_force_n
            yaw_tail_delta_force = split_tail.yaw_delta_force_n
            pitch_tail_net_force_pre_cap = split_tail.pitch_pre_cap_force_n
            yaw_tail_net_force_pre_cap = split_tail.yaw_pre_cap_force_n
            tail_force_cap = split_tail.force_cap_n
            tail_force_cap_scale = split_tail.cap_scale
            tail_force_cap_active = split_tail.cap_active
            pitch_tail_force_n = split_tail.pitch_force_n
            yaw_tail_force_n = split_tail.yaw_force_n
            # Compatibility authority fraction retains the old chi=(delta-alpha)
            # sign, while the split force fields expose the physical force sign.
            if tail_force_cap > 0.0:
                pitch_moment_fraction = -pitch_tail_force_n / tail_force_cap
                yaw_moment_fraction = -yaw_tail_force_n / tail_force_cap
            # Positive command must turn the nose positive.  At an aft station
            # that requires a negative physical tail force; the same force then
            # supplies M_tail=N_tail*x_t with no same-sign G shortcut.
            pitch_tail_force_vector = scale(tail_flow_normal_pitch, pitch_tail_force_n)
            yaw_tail_force_vector = scale(tail_flow_normal_yaw, yaw_tail_force_n)
            tail_arm_vector = scale(axes.forward, tail_station_x_m)
            pitch_tail_alpha_moment = dot(
                cross(tail_arm_vector, scale(tail_flow_normal_pitch, pitch_tail_alpha_force)),
                axes.right,
            )
            yaw_tail_alpha_moment = -dot(
                cross(tail_arm_vector, scale(tail_flow_normal_yaw, yaw_tail_alpha_force)),
                axes.up,
            )
            pitch_tail_delta_moment = dot(
                cross(tail_arm_vector, scale(tail_flow_normal_pitch, pitch_tail_delta_force)),
                axes.right,
            )
            yaw_tail_delta_moment = -dot(
                cross(tail_arm_vector, scale(tail_flow_normal_yaw, yaw_tail_delta_force)),
                axes.up,
            )
            pitch_tail_moment_vector = cross(tail_arm_vector, pitch_tail_force_vector)
            yaw_tail_moment_vector = cross(tail_arm_vector, yaw_tail_force_vector)
            pitch_tail_moment_nm = dot(pitch_tail_moment_vector, axes.right)
            yaw_tail_moment_nm = -dot(yaw_tail_moment_vector, axes.up)
            moment_scale = gravity * mass * abs(tail_station_x_m)
            pitch_fin_moment_equivalent_g = pitch_tail_moment_nm / moment_scale
            yaw_fin_moment_equivalent_g = yaw_tail_moment_nm / moment_scale
        else:
            pitch_moment_fraction, yaw_moment_fraction = _limit_unit_disk(
                pitch_tail_effective_incidence / pitch_authority_reference,
                yaw_tail_effective_incidence / yaw_authority_reference,
            )
            pitch_delta_fraction, yaw_delta_fraction = _limit_unit_disk(
                state.actual_pitch_fin_angle_rad / pitch_authority_reference,
                state.actual_yaw_fin_angle_rad / yaw_authority_reference,
            )
            # Translation uses commanded fin angle so trim G stays
            # finsLatAccel*(δ/finsAoa).  Arm is not a path-G multiplier.
            # Moment uses local tail incidence so the airframe still
            # weathercocks; bandwidth stays ∝ sqrt(arm).
            pitch_fin_moment_equivalent_g = scheduled_fins_g * pitch_moment_fraction
            yaw_fin_moment_equivalent_g = scheduled_fins_g * yaw_moment_fraction
            pitch_fin_translation_equivalent_g = scheduled_fins_g * pitch_delta_fraction
            yaw_fin_translation_equivalent_g = scheduled_fins_g * yaw_delta_fraction
            pitch_tail_force_n = pitch_fin_translation_equivalent_g * gravity * mass
            yaw_tail_force_n = yaw_fin_translation_equivalent_g * gravity * mass
            pitch_tail_moment_nm = pitch_fin_moment_equivalent_g * gravity * mass * arm
            yaw_tail_moment_nm = yaw_fin_moment_equivalent_g * gravity * mass * arm

    if legacy_fin_torque_plant:
        control_force = _add_many(
            scale(aero.flow_normal_pitch, pitch_fin_translation_equivalent_g * gravity * mass),
            scale(aero.flow_normal_yaw, yaw_fin_translation_equivalent_g * gravity * mass),
        )
    elif body_cm_tail_plant:
        control_force = _add_many(pitch_tail_force_vector, yaw_tail_force_vector)
    elif generalized_aero_plant:
        control_force = generalized_delta_force_vector
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
        body_normal_force_vector,
        fixed_lifting_surface_force_vector,
        control_force,
        gravity_force,
    )
    non_gravity = _add_many(
        thrust_force,
        aero.drag_force_n,
        body_normal_force_vector,
        fixed_lifting_surface_force_vector,
        control_force,
    )
    specific_force = scale(non_gravity, 1.0 / mass)
    acceleration = scale(total_force, 1.0 / mass)
    # Acceleration-controller feedback belongs to the missile-mounted body
    # axes, matching guidance.commanded_body_acceleration_g exactly.
    pitch_g = dot(specific_force, axes.up) / gravity
    yaw_g = dot(specific_force, axes.right) / gravity
    wind_basis = cg_wind_normal_basis(state, config)
    wind_pitch_g = dot(specific_force, wind_basis.up) / gravity
    wind_yaw_g = dot(specific_force, wind_basis.right) / gravity
    trajectory_pitch_g = dot(specific_force, aero.flow_normal_pitch) / gravity
    trajectory_yaw_g = dot(specific_force, aero.flow_normal_yaw) / gravity
    axial_g = dot(specific_force, aero.air_velocity_hat) / gravity
    lateral_g = math.hypot(pitch_g, yaw_g)
    trajectory_lateral_g = math.hypot(trajectory_pitch_g, trajectory_yaw_g)
    total_g = norm(specific_force) / gravity

    length = max(float(config["geometry"]["length_m"]), 1e-6)
    inertia_per_mass = max(length * length / 12.0, 1e-6)
    inertia_kg_m2 = mass * inertia_per_mass
    body_reference_area_m2 = 0.0
    body_reference_length_m = 0.0
    body_cp_cg_arm_over_diameter = 0.0
    body_cn_alpha_per_rad = 0.0
    body_cn_q = 0.0
    body_cm_alpha_per_rad = 0.0
    body_cm_q = 0.0
    pitch_body_static_moment_nm = 0.0
    yaw_body_static_moment_nm = 0.0
    pitch_body_rate_moment_nm = 0.0
    yaw_body_rate_moment_nm = 0.0
    pitch_body_total_moment_nm = 0.0
    yaw_body_total_moment_nm = 0.0
    pitch_residual_damping_moment_nm = 0.0
    yaw_residual_damping_moment_nm = 0.0
    pitch_natural_frequency = 0.0
    yaw_natural_frequency = 0.0
    pitch_tail_rate_damping = 0.0
    yaw_tail_rate_damping = 0.0
    pitch_residual_rate_damping = 0.0
    yaw_residual_rate_damping = 0.0
    if generalized_aero_plant:
        body_reference_area_m2 = generalized_reference_area_m2
        body_reference_length_m = generalized_reference_length_m
        generalized = config["aerodynamics"]["generalized_aero_moment_candidate"]
        body_cn_alpha_per_rad = float(generalized["cn_alpha_per_rad"])
        body_cn_q = 0.0
        body_cm_alpha_per_rad = float(generalized["cm_alpha_per_rad"])
        body_cm_q = float(generalized["cm_q"])
        generalized_cm_alpha_dot_per_rad = 0.0
        generalized_cm_alpha_dot_runtime_enabled = False
        pitch_body_static_moment_nm = generalized_pitch_body_static_moment
        yaw_body_static_moment_nm = generalized_yaw_body_static_moment
        pitch_body_rate_moment_nm = generalized_pitch_body_rate_moment
        yaw_body_rate_moment_nm = generalized_yaw_body_rate_moment
        pitch_body_total_moment_nm = (
            pitch_body_static_moment_nm + pitch_body_rate_moment_nm
        )
        yaw_body_total_moment_nm = (
            yaw_body_static_moment_nm + yaw_body_rate_moment_nm
        )
        pitch_tail_force_n = generalized_pitch_delta_force
        yaw_tail_force_n = generalized_yaw_delta_force
        pitch_tail_moment_nm = generalized_pitch_delta_moment
        yaw_tail_moment_nm = generalized_yaw_delta_moment
        pitch_tail_delta_force_n = generalized_pitch_delta_force
        yaw_tail_delta_force_n = generalized_yaw_delta_force
        pitch_tail_delta_force_slope = generalized_pitch_delta_force_slope
        yaw_tail_delta_force_slope = generalized_yaw_delta_force_slope
        # Static CN terms are exported through the body-normal-force fields;
        # the tail fields below remain reserved for the v1 split-tail model.
        pitch_tail_net_force_pre_cap = generalized_pitch_delta_force
        yaw_tail_net_force_pre_cap = generalized_yaw_delta_force
        pitch_tail_delta_moment = generalized_pitch_delta_moment
        yaw_tail_delta_moment = generalized_yaw_delta_moment
        pitch_fin_moment_equivalent_g = (
            pitch_tail_moment_nm / max(gravity * mass * body_reference_length_m, 1e-12)
        )
        yaw_fin_moment_equivalent_g = (
            yaw_tail_moment_nm / max(gravity * mass * body_reference_length_m, 1e-12)
        )
        # This is the resolved generalized Cm_q rate term.  Cm_alpha_dot is
        # frozen at zero because no independent alpha-dot state is available;
        # this is not a synthetic closure or station-derived damping coefficient.
        speed_for_rate = max(float(aero.speed_mps), 1e-6)
        rate_moment_scale = (
            float(aero.normal_force_dynamic_pressure_pa)
            * generalized_reference_area_m2
            * generalized_reference_length_m
            * generalized_reference_length_m
            / (2.0 * speed_for_rate)
        )
        pitch_tail_rate_damping = max(
            0.0,
            -rate_moment_scale
            * float(generalized["cm_q"])
            / max(inertia_kg_m2, 1e-12),
        )
        yaw_tail_rate_damping = pitch_tail_rate_damping
    elif legacy_fin_torque_plant:
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
    elif body_cm_tail_plant:
        body_candidate = config["aerodynamics"]["body_cp_force_candidate"]
        body_reference_area_m2 = caliber_circular_area(config)
        body_reference_length_m = float(config["geometry"]["caliber_m"])
        body_cp_cg_arm_over_diameter = float(
            body_candidate["cp_cg_arm_over_diameter"]
        )
        body_cn_alpha_per_rad = float(body_candidate["cn_alpha_body_per_rad"])
        body_cn_q = float(body_candidate["derived_cn_q"])
        body_cm_alpha_per_rad = float(body_candidate["derived_cm_alpha_per_rad"])
        body_cm_q = float(body_candidate["derived_cm_q"])
        body_arm_inertial = scale(axes.forward, body_station_x_m)
        body_moment_vector = cross(body_arm_inertial, aero.natural_lift_force_n)
        pitch_body_total_moment_nm = dot(body_moment_vector, axes.right)
        yaw_body_total_moment_nm = -dot(body_moment_vector, axes.up)
        static_body_aero = compute_aerodynamics_h2(
            state.position[1],
            state.velocity,
            state.pitch,
            state.yaw,
            config,
            atmosphere,
            axes_override=axes,
            normal_force_velocity_mps=state.velocity,
        )
        static_body_moment_vector = cross(
            body_arm_inertial,
            static_body_aero.natural_lift_force_n,
        )
        pitch_body_static_moment_nm = dot(static_body_moment_vector, axes.right)
        yaw_body_static_moment_nm = -dot(static_body_moment_vector, axes.up)
        pitch_body_rate_moment_nm = (
            pitch_body_total_moment_nm - pitch_body_static_moment_nm
        )
        yaw_body_rate_moment_nm = yaw_body_total_moment_nm - yaw_body_static_moment_nm
        # Body and tail rate damping now arise only from local station velocity.
        # No synthetic critical residual or independent Cm term is executed.
        pitch_tail_rate_damping = (
            scheduled_fins_g
            * gravity
            * tail_station_x_m ** 2
            / (inertia_per_mass * pitch_authority_reference * speed_for_rate)
        )
        yaw_tail_rate_damping = (
            scheduled_fins_g
            * gravity
            * tail_station_x_m ** 2
            / (inertia_per_mass * yaw_authority_reference * speed_for_rate)
        )
        pitch_damping = 0.0
        yaw_damping = 0.0
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
    if body_cm_tail_plant:
        pitch_total_moment_nm = (
            pitch_body_total_moment_nm
            + pitch_fixed_lifting_surface_moment
            + pitch_tail_moment_nm
        )
        yaw_total_moment_nm = (
            yaw_body_total_moment_nm
            + yaw_fixed_lifting_surface_moment
            + yaw_tail_moment_nm
        )
        pitch_angular_accel = pitch_total_moment_nm / inertia_kg_m2
        yaw_angular_accel = yaw_total_moment_nm / inertia_kg_m2
    elif generalized_aero_plant:
        pitch_total_moment_nm = (
            pitch_body_total_moment_nm
            + pitch_fixed_lifting_surface_moment
            + pitch_tail_moment_nm
        )
        yaw_total_moment_nm = (
            yaw_body_total_moment_nm
            + yaw_fixed_lifting_surface_moment
            + yaw_tail_moment_nm
        )
        pitch_angular_accel = pitch_total_moment_nm / inertia_kg_m2
        yaw_angular_accel = yaw_total_moment_nm / inertia_kg_m2
    else:
        pitch_angular_accel = (
            pitch_moment_equivalent_g * gravity * arm / inertia_per_mass * angular_response_scale
        ) - pitch_damping * state.pitch_rate
        yaw_angular_accel = (
            yaw_moment_equivalent_g * gravity * arm / inertia_per_mass * angular_response_scale
        ) - yaw_damping * state.yaw_rate
        if not legacy_fin_torque_plant:
            pitch_tail_force_n = state.actual_pitch_acceleration_g * gravity * mass
            yaw_tail_force_n = state.actual_yaw_acceleration_g * gravity * mass
            pitch_tail_moment_nm = pitch_moment_equivalent_g * gravity * mass * arm
            yaw_tail_moment_nm = yaw_moment_equivalent_g * gravity * mass * arm
        pitch_residual_damping_moment_nm = -pitch_damping * state.pitch_rate * inertia_kg_m2
        yaw_residual_damping_moment_nm = -yaw_damping * state.yaw_rate * inertia_kg_m2
        pitch_total_moment_nm = pitch_tail_moment_nm + pitch_residual_damping_moment_nm
        yaw_total_moment_nm = yaw_tail_moment_nm + yaw_residual_damping_moment_nm
    lift_force = _add_many(
        body_normal_force_vector,
        fixed_lifting_surface_force_vector,
        control_force,
    )
    total_body_wing_tail_normal_force = lift_force
    return H2DynamicsDiagnostics(
        propulsion=prop,
        aero=aero,
        thrust_force_n=thrust_force,
        drag_force_n=aero.drag_force_n,
        natural_lift_force_n=body_normal_force_vector,
        fixed_lifting_surface_force_n=fixed_lifting_surface_force_vector,
        control_force_n=control_force,
        gravity_force_n=gravity_force,
        total_force_n=total_force,
        non_gravity_acceleration_mps2=specific_force,
        specific_force_mps2=specific_force,
        acceleration_mps2=acceleration,
        axial_specific_force_g=axial_g,
        pitch_normal_acceleration_g=pitch_g,
        yaw_normal_acceleration_g=yaw_g,
        wind_normal_pitch_acceleration_g=wind_pitch_g,
        wind_normal_yaw_acceleration_g=wind_yaw_g,
        lateral_load_g=lateral_g,
        trajectory_pitch_normal_acceleration_g=trajectory_pitch_g,
        trajectory_yaw_normal_acceleration_g=trajectory_yaw_g,
        trajectory_lateral_load_g=trajectory_lateral_g,
        total_specific_force_g=total_g,
        drag_power_w=dot(aero.drag_force_n, aero.air_velocity_mps),
        lift_power_w=dot(lift_force, aero.air_velocity_mps),
        body_tail_force_power_at_cg_w=dot(lift_force, aero.air_velocity_mps),
        pitch_angular_acceleration_rad_s2=pitch_angular_accel,
        yaw_angular_acceleration_rad_s2=yaw_angular_accel,
        body_reference_area_m2=body_reference_area_m2,
        body_reference_length_m=body_reference_length_m,
        body_cp_cg_arm_over_diameter=body_cp_cg_arm_over_diameter,
        body_cn_alpha_per_rad=body_cn_alpha_per_rad,
        body_cn_q=body_cn_q,
        body_cm_alpha_per_rad=body_cm_alpha_per_rad,
        body_cm_q=body_cm_q,
        generalized_cm_alpha_dot_per_rad=generalized_cm_alpha_dot_per_rad,
        generalized_pitch_alpha_dot_hat=generalized_pitch_alpha_dot_hat,
        generalized_yaw_alpha_dot_hat=generalized_yaw_alpha_dot_hat,
        generalized_cm_alpha_dot_runtime_enabled=generalized_cm_alpha_dot_runtime_enabled,
        pitch_body_normal_force_n=dot(body_normal_force_vector, aero.flow_normal_pitch),
        yaw_body_normal_force_n=dot(body_normal_force_vector, aero.flow_normal_yaw),
        pitch_body_static_moment_nm=pitch_body_static_moment_nm,
        yaw_body_static_moment_nm=yaw_body_static_moment_nm,
        pitch_body_rate_moment_nm=pitch_body_rate_moment_nm,
        yaw_body_rate_moment_nm=yaw_body_rate_moment_nm,
        pitch_body_total_moment_nm=pitch_body_total_moment_nm,
        yaw_body_total_moment_nm=yaw_body_total_moment_nm,
        fixed_lifting_surface_multiplier=fixed_lifting_surface_multiplier,
        body_normal_force_area_slope_m2_per_rad=body_normal_force_area_slope,
        fixed_lifting_surface_area_slope_m2_per_rad=fixed_lifting_surface_area_slope,
        fixed_lifting_surface_station_x_m=fixed_lifting_surface_station_x,
        pitch_fixed_lifting_surface_alpha_rad=pitch_fixed_lifting_surface_alpha,
        yaw_fixed_lifting_surface_alpha_rad=yaw_fixed_lifting_surface_alpha,
        pitch_fixed_lifting_surface_force_n=pitch_fixed_lifting_surface_force,
        yaw_fixed_lifting_surface_force_n=yaw_fixed_lifting_surface_force,
        pitch_fixed_lifting_surface_moment_nm=pitch_fixed_lifting_surface_moment,
        yaw_fixed_lifting_surface_moment_nm=yaw_fixed_lifting_surface_moment,
        pitch_total_body_wing_tail_normal_force_n=dot(
            total_body_wing_tail_normal_force, cg_flow_normal_pitch
        ),
        yaw_total_body_wing_tail_normal_force_n=dot(
            total_body_wing_tail_normal_force, cg_flow_normal_yaw
        ),
        tail_station_x_m=tail_station_x_m,
        pitch_tail_force_n=pitch_tail_force_n,
        yaw_tail_force_n=yaw_tail_force_n,
        pitch_tail_moment_nm=pitch_tail_moment_nm,
        yaw_tail_moment_nm=yaw_tail_moment_nm,
        pitch_tail_authority_fraction=pitch_moment_fraction,
        yaw_tail_authority_fraction=yaw_moment_fraction,
        tail_alpha_force_multiplier=tail_alpha_force_multiplier,
        tail_delta_force_multiplier=tail_delta_force_multiplier,
        pitch_tail_alpha_force_slope_n_per_rad=pitch_tail_alpha_force_slope,
        yaw_tail_alpha_force_slope_n_per_rad=yaw_tail_alpha_force_slope,
        pitch_tail_delta_force_slope_n_per_rad=pitch_tail_delta_force_slope,
        yaw_tail_delta_force_slope_n_per_rad=yaw_tail_delta_force_slope,
        pitch_tail_alpha_force_n=pitch_tail_alpha_force,
        yaw_tail_alpha_force_n=yaw_tail_alpha_force,
        pitch_tail_delta_force_n=pitch_tail_delta_force,
        yaw_tail_delta_force_n=yaw_tail_delta_force,
        pitch_tail_net_force_pre_cap_n=pitch_tail_net_force_pre_cap,
        yaw_tail_net_force_pre_cap_n=yaw_tail_net_force_pre_cap,
        tail_force_cap_n=tail_force_cap,
        tail_force_cap_scale=tail_force_cap_scale,
        tail_force_cap_active=tail_force_cap_active,
        pitch_tail_alpha_moment_nm=pitch_tail_alpha_moment,
        yaw_tail_alpha_moment_nm=yaw_tail_alpha_moment,
        pitch_tail_delta_moment_nm=pitch_tail_delta_moment,
        yaw_tail_delta_moment_nm=yaw_tail_delta_moment,
        pitch_residual_damping_moment_nm=pitch_residual_damping_moment_nm,
        yaw_residual_damping_moment_nm=yaw_residual_damping_moment_nm,
        pitch_total_moment_nm=pitch_total_moment_nm,
        yaw_total_moment_nm=yaw_total_moment_nm,
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


def _uses_quaternion_candidate(config: dict[str, Any]) -> bool:
    return config["control"].get("plant_semantics") in {
        "body_cm_tail_force_moment",
        "generalized_aero_moment",
        "fin_torque_body_aoa",
    }


def _dynamic_vector(state: SimState, config: dict[str, Any]) -> tuple[float, ...]:
    if _uses_quaternion_candidate(config):
        quaternion = state.orientation_quaternion or quaternion_from_pitch_yaw(
            state.pitch, state.yaw
        )
        return (
            state.position[0], state.position[1], state.position[2],
            state.velocity[0], state.velocity[1], state.velocity[2],
            quaternion[0], quaternion[1], quaternion[2], quaternion[3],
            state.pitch_rate, state.yaw_rate, state.mass,
        )
    return (
        state.position[0], state.position[1], state.position[2],
        state.velocity[0], state.velocity[1], state.velocity[2],
        state.pitch, state.yaw, state.pitch_rate, state.yaw_rate, state.mass,
    )


def _state_from_dynamic(
    state: SimState,
    values: tuple[float, ...],
    config: dict[str, Any],
) -> SimState:
    if _uses_quaternion_candidate(config):
        quaternion = normalize_quaternion((values[6], values[7], values[8], values[9]))
        pitch, yaw = pitch_yaw_from_quaternion(quaternion)
        return replace(
            state,
            position=(values[0], values[1], values[2]),
            velocity=(values[3], values[4], values[5]),
            pitch=pitch,
            yaw=yaw,
            pitch_rate=values[10],
            yaw_rate=values[11],
            mass=values[12],
            orientation_quaternion=quaternion,
        )
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
    if _uses_quaternion_candidate(config):
        quaternion = state.orientation_quaternion or quaternion_from_pitch_yaw(
            state.pitch, state.yaw
        )
        quaternion_rate = quaternion_derivative(
            quaternion,
            state.pitch_rate,
            state.yaw_rate,
        )
        return (
            state.velocity[0], state.velocity[1], state.velocity[2],
            diagnostics.acceleration_mps2[0], diagnostics.acceleration_mps2[1], diagnostics.acceleration_mps2[2],
            quaternion_rate[0], quaternion_rate[1], quaternion_rate[2], quaternion_rate[3],
            diagnostics.pitch_angular_acceleration_rad_s2,
            diagnostics.yaw_angular_acceleration_rad_s2,
            diagnostics.propulsion.mass_flow_kg_s,
        )
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
    values = _dynamic_vector(state, config)
    k1 = _derivative_vector(state, time_s, config, propulsion, powered)
    s2 = _state_from_dynamic(state, _combine(values, k1, dt_s / 2.0), config)
    k2 = _derivative_vector(s2, time_s + dt_s / 2.0, config, propulsion, powered)
    s3 = _state_from_dynamic(state, _combine(values, k2, dt_s / 2.0), config)
    k3 = _derivative_vector(s3, time_s + dt_s / 2.0, config, propulsion, powered)
    s4 = _state_from_dynamic(state, _combine(values, k3, dt_s), config)
    k4 = _derivative_vector(s4, time_s + dt_s, config, propulsion, powered)
    next_values = tuple(
        value + dt_s * (d1 + 2.0 * d2 + 2.0 * d3 + d4) / 6.0
        for value, d1, d2, d3, d4 in zip(values, k1, k2, k3, k4)
    )
    return _state_from_dynamic(state, next_values, config)


__all__ = [
    "H2DynamicsDiagnostics",
    "forces_for_state_h2",
    "rk4_step_h2",
    "state_is_finite",
]
