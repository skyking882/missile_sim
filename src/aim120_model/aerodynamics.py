"""Candidate reference-area, drag, and natural-lift calculations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .atmosphere import AtmosphereSample, StandardAtmosphere
from .drag_models import drag_force_from_cda, effective_cda0, effective_cda_alpha, mach_multiplier
from .math3d import Vector, add, clamp, dot, norm, normalize, scale, sub


@dataclass(frozen=True)
class BodyAxes:
    forward: Vector
    up: Vector
    right: Vector


@dataclass(frozen=True)
class AeroSample:
    atmosphere: AtmosphereSample
    speed_mps: float
    mach: float
    dynamic_pressure_pa: float
    reference_area_m2: float
    angle_of_attack_rad: float
    pitch_alpha_rad: float
    yaw_alpha_rad: float
    drag_coefficient: float
    lift_coefficient_pitch: float
    lift_coefficient_yaw: float
    drag_force_n: Vector
    natural_lift_force_n: Vector


@dataclass(frozen=True)
class H2AeroSample:
    atmosphere: AtmosphereSample
    air_velocity_mps: Vector
    air_velocity_hat: Vector
    speed_mps: float
    mach: float
    dynamic_pressure_pa: float
    reference_area_m2: float
    angle_of_attack_rad: float
    pitch_alpha_rad: float
    yaw_alpha_rad: float
    cda0_m2: float
    cda_alpha_m2: float
    total_cda_m2: float
    drag_coefficient: float
    lift_coefficient_pitch: float
    lift_coefficient_yaw: float
    flow_normal_pitch: Vector
    flow_normal_yaw: Vector
    drag_force_n: Vector
    natural_lift_force_n: Vector


def body_axes(pitch_rad: float, yaw_rad: float) -> BodyAxes:
    cp = math.cos(pitch_rad)
    sp = math.sin(pitch_rad)
    cy = math.cos(yaw_rad)
    sy = math.sin(yaw_rad)
    forward = (cp * cy, sp, cp * sy)
    up = (-sp * cy, cp, -sp * sy)
    right = (-sy, 0.0, cy)
    return BodyAxes(forward, up, right)


def reference_area(config: dict[str, Any]) -> float:
    geometry = config["geometry"]
    base = math.pi * geometry["caliber_m"] ** 2 / 4.0
    mode = geometry.get("reference_area_mode", "caliber_times_wing_multiplier")
    if mode == "caliber_area":
        return base
    if mode == "caliber_times_wing_multiplier":
        return base * geometry["wing_area_multiplier"]
    if mode == "wing_multiplier_only":
        return geometry["wing_area_multiplier"]
    raise ValueError(f"unknown reference_area_mode: {mode}")


def _mach_multiplier(mach: float, settings: dict[str, Any]) -> float:
    width = max(float(settings.get("width", 0.45)), 1e-6)
    center = float(settings.get("center", 1.0))
    base = float(settings.get("base", 1.0))
    peak = float(settings.get("transonic_peak", 0.0))
    return base + peak * math.exp(-((mach - center) / width) ** 2)


def compute_aerodynamics_h2(
    altitude_m: float,
    velocity_mps: Vector,
    pitch_rad: float,
    yaw_rad: float,
    config: dict[str, Any],
    atmosphere: StandardAtmosphere | None = None,
) -> H2AeroSample:
    """H2 flow-normal geometry with explicit effective CdA bookkeeping."""

    atmosphere_model = atmosphere or StandardAtmosphere()
    atm = atmosphere_model.sample(altitude_m)
    axes = body_axes(pitch_rad, yaw_rad)
    wind = tuple(config["atmosphere"].get("wind_mps", (0.0, 0.0, 0.0)))
    air_velocity = sub(velocity_mps, (float(wind[0]), float(wind[1]), float(wind[2])))
    speed = norm(air_velocity)
    v_hat = normalize(air_velocity, fallback=axes.forward)
    mach = speed / atm.speed_of_sound_mps if atm.speed_of_sound_mps > 0.0 else 0.0
    q = 0.5 * atm.density_kg_m3 * speed * speed
    forward_component = dot(v_hat, axes.forward)
    pitch_alpha = math.atan2(-dot(v_hat, axes.up), forward_component)
    yaw_alpha = math.atan2(-dot(v_hat, axes.right), forward_component)
    total_alpha = math.sqrt(pitch_alpha * pitch_alpha + yaw_alpha * yaw_alpha)
    cda0 = effective_cda0(mach, config)
    cda_alpha = effective_cda_alpha(pitch_alpha, yaw_alpha, mach, config)
    total_cda = cda0 + cda_alpha
    drag_force = drag_force_from_cda(q, total_cda, v_hat)
    flow_normal_pitch = normalize(
        sub(axes.up, scale(v_hat, dot(axes.up, v_hat))),
        fallback=axes.up,
    )
    flow_normal_yaw = normalize(
        sub(axes.right, scale(v_hat, dot(axes.right, v_hat))),
        fallback=axes.right,
    )
    aero_cfg = config["aerodynamics"]
    lift_multiplier = mach_multiplier(mach, aero_cfg["mach_lift"])
    max_cy = float(aero_cfg["max_cy_at_aoa"])
    cy_pitch = clamp(float(aero_cfg["cy_k"]) * lift_multiplier * pitch_alpha, -max_cy, max_cy)
    cy_yaw = clamp(float(aero_cfg["cy_k"]) * lift_multiplier * yaw_alpha, -max_cy, max_cy)
    if aero_cfg.get("natural_lift_enabled", True):
        lift_area = area_for_h2_lift(config)
        lift_pitch = q * lift_area * cy_pitch * float(aero_cfg.get("natural_lift_fraction", 1.0))
        lift_yaw = q * lift_area * cy_yaw * float(aero_cfg.get("natural_lift_fraction", 1.0))
        natural_lift = add(
            scale(flow_normal_pitch, lift_pitch),
            scale(flow_normal_yaw, lift_yaw),
        )
    else:
        natural_lift = (0.0, 0.0, 0.0)
    reference = max(area_for_h2_lift(config), 1e-12)
    return H2AeroSample(
        atmosphere=atm,
        air_velocity_mps=air_velocity,
        air_velocity_hat=v_hat,
        speed_mps=speed,
        mach=mach,
        dynamic_pressure_pa=q,
        reference_area_m2=reference,
        angle_of_attack_rad=total_alpha,
        pitch_alpha_rad=pitch_alpha,
        yaw_alpha_rad=yaw_alpha,
        cda0_m2=cda0,
        cda_alpha_m2=cda_alpha,
        total_cda_m2=total_cda,
        drag_coefficient=total_cda / reference,
        lift_coefficient_pitch=cy_pitch,
        lift_coefficient_yaw=cy_yaw,
        flow_normal_pitch=flow_normal_pitch,
        flow_normal_yaw=flow_normal_yaw,
        drag_force_n=drag_force,
        natural_lift_force_n=natural_lift,
    )


def area_for_h2_lift(config: dict[str, Any]) -> float:
    """Use the same explicit basis as the candidate lift-area hypothesis."""

    from .drag_models import area_basis

    return area_basis(config) * float(config.get("drag_model", {}).get("lift_area_scale", 1.0))


def compute_aerodynamics(
    altitude_m: float,
    velocity_mps: Vector,
    pitch_rad: float,
    yaw_rad: float,
    config: dict[str, Any],
    atmosphere: StandardAtmosphere | None = None,
) -> AeroSample:
    atmosphere_model = atmosphere or StandardAtmosphere()
    atm = atmosphere_model.sample(altitude_m)
    axes = body_axes(pitch_rad, yaw_rad)
    speed = norm(velocity_mps)
    v_hat = normalize(velocity_mps, fallback=axes.forward)
    mach = speed / atm.speed_of_sound_mps if atm.speed_of_sound_mps > 0.0 else 0.0
    q = 0.5 * atm.density_kg_m3 * speed * speed
    area = reference_area(config)
    forward_component = clamp(dot(v_hat, axes.forward), -1.0, 1.0)
    alpha = math.acos(forward_component) if speed > 1e-9 else 0.0
    pitch_alpha = math.atan2(dot(v_hat, axes.up), max(abs(forward_component), 1e-9))
    yaw_alpha = math.atan2(dot(v_hat, axes.right), max(abs(forward_component), 1e-9))
    aero_cfg = config["aerodynamics"]
    cx = aero_cfg["cx_k"] * _mach_multiplier(mach, aero_cfg["mach_drag"])
    aoa_interpretation = aero_cfg.get("aoa_drag_interpretation", "radians_squared")
    if aoa_interpretation == "radians_squared":
        aoa_term = alpha * alpha
    elif aoa_interpretation == "sin_squared":
        aoa_term = math.sin(alpha) ** 2
    elif aoa_interpretation == "normalized_90deg_squared":
        aoa_term = (alpha / (math.pi / 2.0)) ** 2
    else:
        raise ValueError(f"unknown aoa_drag_interpretation: {aoa_interpretation}")
    cx += aero_cfg["cx_vs_aoa"] * aoa_term
    drag_n = q * area * cx
    drag_force = scale(v_hat, -drag_n)

    lift_multiplier = _mach_multiplier(mach, aero_cfg["mach_lift"])
    max_cy = float(aero_cfg["max_cy_at_aoa"])
    cy_pitch = clamp(aero_cfg["cy_k"] * lift_multiplier * pitch_alpha, -max_cy, max_cy)
    cy_yaw = clamp(aero_cfg["cy_k"] * lift_multiplier * yaw_alpha, -max_cy, max_cy)
    if aero_cfg.get("natural_lift_enabled", True):
        natural_fraction = float(aero_cfg.get("natural_lift_fraction", 1.0))
        lift_pitch = q * area * cy_pitch * natural_fraction
        lift_yaw = q * area * cy_yaw * natural_fraction
        natural_lift = add_vectors(scale(axes.up, lift_pitch), scale(axes.right, lift_yaw))
    else:
        natural_lift = (0.0, 0.0, 0.0)
    return AeroSample(
        atmosphere=atm,
        speed_mps=speed,
        mach=mach,
        dynamic_pressure_pa=q,
        reference_area_m2=area,
        angle_of_attack_rad=alpha,
        pitch_alpha_rad=pitch_alpha,
        yaw_alpha_rad=yaw_alpha,
        drag_coefficient=cx,
        lift_coefficient_pitch=cy_pitch,
        lift_coefficient_yaw=cy_yaw,
        drag_force_n=drag_force,
        natural_lift_force_n=natural_lift,
    )


def add_vectors(a: Vector, b: Vector) -> Vector:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])
