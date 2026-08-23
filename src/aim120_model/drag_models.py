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


# 1943-law Cx(M) * 1.10, with the M=3.5 and M=4.0 knots extrapolated.
# Interior evaluation is linear interpolation in Mach; outside the table
# the endpoint Cx is held so CdA0 cannot go negative at low Mach.
CX_1943_X1_10_TABLE: tuple[tuple[float, float], ...] = (
    (0.8, 0.173),
    (0.9, 0.209),
    (1.0, 0.330),
    (1.1, 0.402),
    (1.2, 0.413),
    (1.4, 0.396),
    (1.6, 0.380),
    (1.8, 0.358),
    (1.93, 0.352),
    (2.01, 0.342),
    (2.29, 0.314),
    (2.66, 0.288),
    (3.10, 0.270),
    (3.5, 0.250),
    (4.0, 0.231),
)

INTERPOLATED_CX_1943_X1_10 = "interpolated_cx_1943_x1_10"


def mach_multiplier(mach: float, settings: dict[str, Any]) -> float:
    width = max(float(settings.get("width", 0.45)), 1e-9)
    center = float(settings.get("center", 1.0))
    base = float(settings.get("base", 1.0))
    peak = float(settings.get("transonic_peak", 0.0))
    return base + peak * math.exp(-((mach - center) / width) ** 2)


def _cx_mach_table(drag_cfg: dict[str, Any]) -> tuple[tuple[float, float], ...]:
    raw = drag_cfg.get("cx_vs_mach")
    if raw is None:
        return CX_1943_X1_10_TABLE
    table = tuple((float(mach), float(cx)) for mach, cx in raw)
    if len(table) < 2:
        raise ValueError("drag_model.cx_vs_mach must contain at least two knots")
    previous = table[0][0]
    for mach, cx in table[1:]:
        if not math.isfinite(mach) or mach <= previous:
            raise ValueError("drag_model.cx_vs_mach Mach knots must be strictly increasing")
        if not math.isfinite(cx) or cx <= 0.0:
            raise ValueError("drag_model.cx_vs_mach Cx values must be finite and positive")
        previous = mach
    return table


def interpolate_cx_vs_mach(mach: float, table: tuple[tuple[float, float], ...] | None = None) -> float:
    """Linear interpolation in Mach; hold the endpoint Cx outside the table."""

    knots = table if table is not None else CX_1943_X1_10_TABLE
    if not math.isfinite(mach) or mach <= knots[0][0]:
        return knots[0][1]
    if mach >= knots[-1][0]:
        return knots[-1][1]
    for index in range(1, len(knots)):
        mach_hi, cx_hi = knots[index]
        if mach <= mach_hi:
            mach_lo, cx_lo = knots[index - 1]
            fraction = (mach - mach_lo) / (mach_hi - mach_lo)
            return cx_lo + fraction * (cx_hi - cx_lo)
    return knots[-1][1]


def effective_cda0(mach: float, config: dict[str, Any]) -> float:
    """Return positive effective zero-AoA drag area in square metres."""

    drag_cfg = config["drag_model"]
    mode = drag_cfg.get("shape_mode", "scaled_h1_shape")
    if mode == "scaled_h1_shape":
        shape = mach_multiplier(mach, config["aerodynamics"]["mach_drag"])
        cda = area_basis(config) * float(config["aerodynamics"]["cx_k"]) * shape
    elif mode == INTERPOLATED_CX_1943_X1_10:
        # Datamine CxK remains the per-missile scale on the shared interpolated
        # 1943*1.10 Cx(M) table.  Frozen H1/H2 configs keep scaled_h1_shape.
        cda = (
            area_basis(config)
            * float(config["aerodynamics"]["cx_k"])
            * interpolate_cx_vs_mach(mach, _cx_mach_table(drag_cfg))
        )
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
    """Even AoA drag area; angles are radians internally.

    Profile H2 spec: CdA_α = S_d · CxAoA · min(α, cap)² with S_d = πd²/4
    (no wingAreaMult) and no Mach shape.  Frozen H1/H2 configs omit those
    keys and keep the historical wing-area × Gaussian-shape term.
    """

    drag_cfg = config["drag_model"]
    cap = max(float(drag_cfg.get("alpha_drag_cap_rad", math.pi / 2.0)), 0.0)
    alpha_sq = alpha_pitch_rad * alpha_pitch_rad + alpha_yaw_rad * alpha_yaw_rad
    if cap > 0.0:
        alpha_sq = min(alpha_sq, cap * cap)
    alpha_area_mode = drag_cfg.get("alpha_drag_area_basis_mode")
    if alpha_area_mode is None and drag_cfg.get("shape_mode") == INTERPOLATED_CX_1943_X1_10:
        alpha_area_mode = "caliber_area"
    area = area_basis(config, str(alpha_area_mode)) if alpha_area_mode else area_basis(config)
    apply_mach_shape = drag_cfg.get("alpha_drag_mach_shape")
    if apply_mach_shape is None:
        apply_mach_shape = drag_cfg.get("shape_mode") != INTERPOLATED_CX_1943_X1_10
    shape = (
        mach_multiplier(mach, config["aerodynamics"].get("mach_drag", {}))
        if apply_mach_shape
        else 1.0
    )
    return (
        area
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

