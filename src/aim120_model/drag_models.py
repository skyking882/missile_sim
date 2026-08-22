"""Effective zero-AoA drag-area models for H2.

The first H2 quantity is CdA0(M), not a uniquely identified reference area or
Cx table.  This preserves the area/Cx identifiability limitation described in
plan2 while still allowing a controlled one-dimensional scale calibration.
"""

from __future__ import annotations

import math
from typing import Any

from .math3d import Vector, dot, scale


def area_basis(config: dict[str, Any], mode: str | None = None) -> float:
    geometry = config["geometry"]
    base = math.pi * geometry["caliber_m"] ** 2 / 4.0
    selected = mode or config.get("drag_model", {}).get(
        "area_basis_mode", geometry.get("reference_area_mode", "caliber_times_wing_multiplier")
    )
    if selected == "caliber_area":
        return base
    if selected == "caliber_times_wing_multiplier":
        return base * geometry["wing_area_multiplier"]
    if selected == "wing_multiplier_only":
        return geometry["wing_area_multiplier"]
    raise ValueError(f"unknown effective CdA area basis: {selected}")


def mach_multiplier(mach: float, settings: dict[str, Any]) -> float:
    width = max(float(settings.get("width", 0.45)), 1e-9)
    center = float(settings.get("center", 1.0))
    base = float(settings.get("base", 1.0))
    peak = float(settings.get("transonic_peak", 0.0))
    return base + peak * math.exp(-((mach - center) / width) ** 2)


def effective_cda0(mach: float, config: dict[str, Any]) -> float:
    """Return positive effective zero-AoA drag area in square metres."""

    drag_cfg = config["drag_model"]
    mode = drag_cfg.get("shape_mode", "scaled_h1_shape")
    if mode == "scaled_h1_shape":
        shape = mach_multiplier(mach, config["aerodynamics"]["mach_drag"])
        cda = area_basis(config) * float(config["aerodynamics"]["cx_k"]) * shape
    elif mode == "scaled_h1_shape_supersonic_decay":
        shape = mach_multiplier(mach, config["aerodynamics"]["mach_drag"])
        start = float(drag_cfg.get("decay_start_mach", 1.2))
        exponent = float(drag_cfg.get("decay_exponent", 0.5))
        floor = float(drag_cfg.get("decay_floor", 0.6))
        decay = 1.0 if mach <= start else max((start / max(mach, 1e-9)) ** exponent, floor)
        cda = area_basis(config) * float(config["aerodynamics"]["cx_k"]) * shape * decay
    elif mode == "smooth_step_gaussian":
        center = float(drag_cfg.get("center", 1.0))
        width = max(float(drag_cfg.get("width", 0.45)), 1e-9)
        step = 0.5 * (1.0 + math.tanh((mach - center) / width))
        c_sub = float(drag_cfg.get("cda_sub_m2", 0.01))
        c_sup = float(drag_cfg.get("cda_sup_m2", 0.02))
        transonic = float(drag_cfg.get("transonic_peak_m2", 0.0)) * math.exp(
            -((mach - center) / max(float(drag_cfg.get("transonic_width", width)), 1e-9)) ** 2
        )
        cda = c_sub + (c_sup - c_sub) * step + transonic
    else:
        raise ValueError(f"unknown H2 drag shape_mode: {mode}")
    value = float(drag_cfg.get("drag_scale", 1.0)) * cda
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"effective CdA0 must be finite and positive, got {value}")
    return value


def effective_cda_alpha(alpha_pitch_rad: float, alpha_yaw_rad: float, mach: float, config: dict[str, Any]) -> float:
    """Low-dimensional even AoA drag area; angles are radians internally."""

    drag_cfg = config["drag_model"]
    cap = max(float(drag_cfg.get("alpha_drag_cap_rad", math.pi / 2.0)), 0.0)
    alpha_sq = alpha_pitch_rad * alpha_pitch_rad + alpha_yaw_rad * alpha_yaw_rad
    if cap > 0.0:
        alpha_sq = min(alpha_sq, cap * cap)
    shape = mach_multiplier(mach, config["aerodynamics"].get("mach_drag", {}))
    return (
        area_basis(config)
        * float(config["aerodynamics"]["cx_vs_aoa"])
        * float(drag_cfg.get("alpha_drag_scale", 1.0))
        * shape
        * alpha_sq
    )


def drag_force_from_cda(dynamic_pressure_pa: float, cda_m2: float, v_hat: Vector) -> Vector:
    drag_n = max(0.0, float(dynamic_pressure_pa) * max(0.0, float(cda_m2)))
    return scale(v_hat, -drag_n)


def drag_power_w(force_n: Vector, air_velocity_mps: Vector) -> float:
    return dot(force_n, air_velocity_mps)

