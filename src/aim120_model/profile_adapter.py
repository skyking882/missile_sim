"""Adapt unit-explicit missile profiles to the shared H2 candidate runtime.

The adapter is deliberately conservative: profile values are copied when
present, missing physical coefficients become neutral (usually zero), and only
runtime/discretization values use the documented shared defaults.  This opens
experimentation without claiming validation or borrowing another missile's
frozen configuration.
"""

from __future__ import annotations

import copy
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

from .drag_models import CX_1943_X1_10_TABLE, INTERPOLATED_CX_1943_X1_10


SUPPORTED_MODEL_TYPES = {
    "dynamics": {"h2_reduced_order"},
    "propulsion": {"staged_solid_rocket"},
    "aerodynamics": {"conventional_fin"},
    "control": {"aerodynamic_fin"},
    "guidance": {"pn", "pn_loft"},
}


LEGACY_CRITICAL_DAMPED_PLANT = "critical_damped_v12"
BODY_CM_TAIL_FORCE_PLANT = "body_cm_tail_force_moment_v1"
GENERALIZED_AERO_MOMENT_PLANT = "generalized_aero_moment_v1"
SUPPORTED_PLANT_MODELS = {
    LEGACY_CRITICAL_DAMPED_PLANT,
    BODY_CM_TAIL_FORCE_PLANT,
    GENERALIZED_AERO_MOMENT_PLANT,
}


def _fins_aoa_limit_rad(raw: float) -> float:
    """Datamine finsAoaHor/Ver is stored as radians."""

    return float(raw)


def _fins_aoa_limit_deg(raw: float) -> float:
    return math.degrees(float(raw))


# Shared H2 solver/mapping layer.  These are runtime-model semantics rather
# than missile-datamine facts.  Missile-specific values are overlaid from the
# selected unit-explicit profile below.  The values originate from the frozen
# H2/v1 layer that has regression and reference-comparison coverage.
UNIVERSAL_H2_LAYER: dict[str, Any] = {
    "aerodynamics": {
        # gameparams.blkx:shellBallisticsParams.props.CxAoA
        "global_cx_vs_aoa": 9.0,
        "missing_max_cy_at_aoa": 1.0,
        "max_cy_interpretation": "coefficient_cap",
        "thrust_vector_angle_deg": 0.0,
        "natural_lift_fraction": 1.0,
        "mach_drag": {"base": 1.0, "transonic_peak": 0.28, "center": 1.0, "width": 0.45},
        "mach_lift": {"base": 1.0, "transonic_peak": 0.10, "center": 1.0, "width": 0.60},
    },
    "drag_model": {
        # AIM-120A H2 fitted 0.2995 is frozen in configs/aim120a_h2.yaml only.
        # Profile missiles use datamine CxK * interpolated 1943*1.10 Cx(M).
        "effective_drag_scale": 1.0,
        "shape_mode": INTERPOLATED_CX_1943_X1_10,
        "alpha_drag_scale": 1.0,
        "alpha_drag_cap_rad": 1.2,
    },
    "performance": {"maximum_speed_is_hard_clamp": False},
    "guidance": {
        "guidance_timeout_s": 0.6,
        "guidance_timeout_semantics": "unresolved_do_not_disable_after_timeout",
        "maximum_angular_rate_deg_s": 60.0,
        "loft_exit_time_to_go_s": 18.0,
        "target_elevation_deg": -3.5,
        "omega_max_deg_s": 0.75,
        "angle_to_acceleration_multiplier": 20.0,
        "flight_time_gain_table": [[0.0, 1.0]],
        "time_to_hit_gain_table": [[10.0, 1.0], [25.0, 0.8], [50.0, 0.5]],
    },
    "control": {
        "limit_angle_of_attack_enabled": False,
        # The datamine exposes fin deflection, not a body-angle limit.  This is
        # therefore an explicit shared H2 controller guard, not a profile fact.
        "maximum_body_angle_of_attack_deg": 30.0,
        "actuator_time_constant_s": 0.08,
        "derivative_filter_time_constant_s": 0.03,
    },
    "atmosphere": {"wind_mps": [0.0, 0.0, 0.0], "gravity_mps2": 9.80665},
    "numerics": {
        "dt_s": 0.02,
        "event_epsilon": 1.0e-9,
        "max_steps": 10000,
        "integrator": "rk4_fixed_step_with_explicit_stage_boundaries",
    },
}


def unsupported_model_types(profile: dict[str, Any]) -> list[str]:
    family = profile.get("model_family", {})
    return [
        f"{axis}={family.get(axis)}"
        for axis, allowed in SUPPORTED_MODEL_TYPES.items()
        if family.get(axis) not in allowed
    ]


@lru_cache(maxsize=8)
def load_runtime_defaults(path: str) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data.get("enable_supported_profiles"):
        raise ValueError("profile H2 runtime permission is disabled")
    return data


def _number(value: Any, fallback: float, assumptions: list[str], label: str) -> float:
    if value is None:
        assumptions.append(f"{label} missing -> universal H2 layer {fallback:.9g}")
        return float(fallback)
    return float(value)


def _positive_finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be positive and finite")
    return number


def _nonnegative_finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be non-negative and finite")
    return number


def _finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _bounded_coefficient(value: Any, label: str) -> float:
    """Validate an explicitly supplied, signed candidate coefficient.

    The bound is a numerical safety boundary for this unsupported reduced
    model, not a claim about a physical missile coefficient.
    """

    number = _finite(value, label)
    if abs(number) > 1000.0:
        raise ValueError(f"{label} must be within +/-1000 for the candidate model")
    return number


def _mach_settings(node: dict[str, Any], fallback: dict[str, Any], assumptions: list[str], label: str) -> dict[str, float]:
    if node.get("model") == "scalar" and isinstance(node.get("parameters"), (int, float)):
        return {"base": float(node["parameters"]), "transonic_peak": 0.0, "center": 1.0, "width": float(fallback["width"])}
    if node.get("model") != "not_declared":
        assumptions.append(f"{label} {node.get('model')} not mapped -> universal H2 runtime shape")
    else:
        assumptions.append(f"{label} not declared -> universal H2 runtime shape")
    return {key: float(value) for key, value in fallback.items()}


def _gain_table(
    value: Any,
    fallback: list[list[float]],
    assumptions: list[str],
    label: str,
) -> list[list[float]]:
    if value is None:
        assumptions.append(f"{label} missing -> universal H2 runtime table")
        return copy.deepcopy(fallback)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    table: list[list[float]] = []
    previous_time = -math.inf
    for index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError(f"{label}[{index}] must be [time_s, gain]")
        time_s = _nonnegative_finite(row[0], f"{label}[{index}][0]")
        gain = _nonnegative_finite(row[1], f"{label}[{index}][1]")
        if time_s <= previous_time:
            raise ValueError(f"{label} times must be strictly increasing")
        table.append([time_s, gain])
        previous_time = time_s
    return table


def build_h2_candidate_config(profile: dict[str, Any], defaults: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    unsupported = unsupported_model_types(profile)
    if unsupported:
        raise ValueError("unsupported profile model types: " + ", ".join(unsupported))
    assumptions: list[str] = []
    layer_aero = UNIVERSAL_H2_LAYER["aerodynamics"]
    layer_drag = UNIVERSAL_H2_LAYER["drag_model"]
    layer_guidance = UNIVERSAL_H2_LAYER["guidance"]
    layer_control = UNIVERSAL_H2_LAYER["control"]
    geometry = profile["geometry"]
    aero = profile["aerodynamics"]
    performance = profile["performance"]
    guidance = profile["guidance"]
    control = profile["control"]
    pid = control["pid"]
    plant_model = str(defaults.get("plant_model", LEGACY_CRITICAL_DAMPED_PLANT))
    if plant_model not in SUPPORTED_PLANT_MODELS:
        raise ValueError(
            f"unsupported profile H2 plant_model {plant_model!r}; "
            f"expected one of {sorted(SUPPORTED_PLANT_MODELS)}"
        )

    raw_pid = {
        "p": _number(pid.get("p"), 0.0, assumptions, "control.pid.p"),
        "i": _number(pid.get("i"), 0.0, assumptions, "control.pid.i"),
        "d": _number(pid.get("d"), 0.0, assumptions, "control.pid.d"),
    }
    mapped_pid = raw_pid
    SHARED_BASE_INDICATED_SPEED_KMH = 1800.0
    base_indicated_speed = control.get("base_indicated_speed_kmh")
    use_fin_force_q_scaling = (
        plant_model == LEGACY_CRITICAL_DAMPED_PLANT
        and bool(defaults.get("fin_force_q_scaling", False))
    )
    if base_indicated_speed is None:
        if use_fin_force_q_scaling:
            base_indicated_speed = float(
                defaults.get(
                    "shared_base_indicated_speed_kmh",
                    SHARED_BASE_INDICATED_SPEED_KMH,
                )
            )
            assumptions.append("control.base_indicated_speed_kmh missing -> shared 1800")
        else:
            assumptions.append(
                "control.base_indicated_speed_kmh missing -> speed scheduling unavailable; baseline mode only"
            )
    else:
        base_indicated_speed = float(base_indicated_speed)
        if base_indicated_speed <= 0.0:
            raise ValueError("control.base_indicated_speed_kmh must be positive when declared")
    assumptions.append(
        "control PID uses the selected profile's raw p/i/d values; no AIM-120A floor or gain substitution"
    )
    assumptions.append(
        "attitude response uses length-derived transverse inertia; the retired shared "
        "angular_response_scale is not applied"
    )
    assumptions.append(
        "body pitch/yaw rates are integrated without a shared hard clamp; seeker angular-rate limits remain "
        "observation constraints and are not mapped into the airframe plant"
    )
    if plant_model == LEGACY_CRITICAL_DAMPED_PLANT:
        assumptions.append(
            "body CN-alpha is not added to path G; packed lift "
            "F_N = m g A η_q disk(α/α_max) with A=finsLatAccel"
        )
        assumptions.append(
            "candidate path G: a_lat/g = finsLatAccel*(q/q_base)*(alpha/finsAoa) on the unit disk; "
            "fin deflection still sets tail moment/bandwidth via distFromCmToStab. "
            "finsAoaHor/Ver is treated as radians. "
            "reqAccelMax radially caps the gravity-compensated specific-force command. "
            "loadFactorMax radially caps F_N only; drag and thrust are not scaled. "
            "baseIndSpeed q/q_base scales fin force when fin_force_q_scaling is on; "
            "missing profile baseIndSpeed uses shared 1800 km/h and does not fall back to none"
        )
        assumptions.append(
            "candidate rotation: I=m L^2/12, I ω̇ = K(δ-α)-Cω with K=N'Δ, "
            "C=2√(K I), ζ=1; distFromCmToStab is the static margin Δ; "
            "no ω·Δ/V in the weathervane spring"
        )
        assumptions.append(
            "CdA_α = (πd²/4)·CxAoA·min(α, 1.2)² with no wingAreaMult and no Mach shape; "
            "CdA0 = S_w·CxK·C_et(M); optional CdA_δ default 0"
        )
        if bool(defaults.get("acceleration_outer_rate_inner", False)):
            assumptions.append(
                "candidate controller: ω_cmd = (f_meas+ĝ) g/V + (f_c-f_meas) g/(V τ_p); "
                "δ_cmd = sat((ω_cmd-ω)/ω_ref)·α_max on the paired disk. "
                "τ_p and ω_ref default 0.35; path-close integral defaults to 0. "
                "raw accelControl P/I/D is not added to q_cmd because those "
                "datamine gains are not (rad/s)/g and are not applied to q_cmd; "
                "pitch_pid_output telemetry is zeroed"
            )
    elif plant_model == BODY_CM_TAIL_FORCE_PLANT:
        assumptions.append(
            "candidate body model: circular Sref, CN_alpha_body=2 rad^-1, lref=caliber, "
            "and lambda=(body CP ahead of CG arm)/diameter=3"
        )
        assumptions.append(
            "candidate body force is evaluated once at r_CP=[lambda*D,0,0] using V_CP=V_CG+omega cross r_CP; "
            "M_body=r_CP cross F_body, so derived CNq/Cm_alpha/Cm_q are diagnostics rather than independent terms"
        )
        assumptions.append(
            "candidate empirical fin authority: finsLatAccel is not a fixed-area tail-force coefficient; "
            "distFromCmToStab is an unverified empirical arm provisionally interpreted as metres, "
            "x_t=-abs(distance), M_tail=N_tail*x_t"
        )
        assumptions.append(
            "candidate rotation: I_perp=m*length^2/12, M=M_body+M_tail, synthetic residual=0"
        )
        assumptions.append(
            "candidate fixed-lift station x_W is a signed CG-relative metre value; "
            "it is an unsupported geometry hypothesis and is not mapped from wingAreaMult"
        )
        assumptions.append(
            "candidate controller: raw accelControl PID closes the outer body-normal-acceleration loop and its "
            "output is interpreted as a body-rate command in rad/s; a shared body-rate inner-loop time constant "
            "uses local tail moment effectiveness to request fin angle"
        )
    else:
        assumptions.append(
            "generalized candidate uses explicit dimensionless CN/Cm derivatives; "
            "these are unsupported local coefficients and are not copied from a game file"
        )
        assumptions.append(
            "CN and Cm share CG alpha, fin delta, body-rate and alpha-rate inputs, "
            "but force and moment have independent derivative parameters"
        )
        assumptions.append(
            "generalized candidate has no force station, x_W/k_W, distFromCmToStab, "
            "or finsLatAccel force-to-moment conversion"
        )

    profile_cx_aoa = aero["cx_vs_aoa"].get("coefficient_per_rad2")
    if profile_cx_aoa is None:
        cx_aoa = float(layer_aero["global_cx_vs_aoa"])
        assumptions.append(
            "aerodynamics.cx_vs_aoa missing in missile BLK -> gameparams shellBallisticsParams.props.CxAoA 9"
        )
    else:
        cx_aoa = float(profile_cx_aoa)
    max_cy = _number(aero["max_cy_at_aoa"].get("value"), layer_aero["missing_max_cy_at_aoa"], assumptions, "aerodynamics.max_cy_at_aoa")
    fins_g = _number(aero.get("fins_lateral_acceleration_g"), 0.0, assumptions, "aerodynamics.fins_lateral_acceleration_g")
    actuator_tau = _number(control.get("actuator_time_constant_s"), layer_control["actuator_time_constant_s"], assumptions, "control.actuator_time_constant_s")
    maximum_rate = control.get("max_pitch_yaw_rate_deg_s")
    if maximum_rate is None:
        maximum_rate = guidance.get("maximum_angular_rate_deg_s")
    maximum_rate = _number(
        maximum_rate,
        layer_guidance["maximum_angular_rate_deg_s"],
        assumptions,
        "guidance.maximum_angular_rate_deg_s",
    )
    lock_range = guidance.get("lock_range_m")
    if lock_range is None:
        lock_range = float(performance["maximum_distance_m"])
        assumptions.append(f"guidance.lock_range_m missing -> {lock_range:g}")
    loft_exit_distance = guidance.get("loft_exit_distance_m")
    if loft_exit_distance is None:
        loft_exit_distance = float(lock_range)
        assumptions.append(f"guidance.loft_exit_distance_m missing -> lock range {loft_exit_distance:g}")
    loft_exit_tgo = guidance.get("loft_exit_time_to_go_s")
    if loft_exit_tgo is None:
        loft_exit_tgo = float(layer_guidance["loft_exit_time_to_go_s"])
        assumptions.append(f"guidance.loft_exit_time_to_go_s missing -> {loft_exit_tgo:g}")
    loft_omega_raw = guidance.get("loft_omega_max_deg_s")
    if loft_omega_raw is None:
        loft_omega_raw = defaults.get("loft_omega_max_deg_s")
    loft_omega_max_deg_s = (
        None
        if loft_omega_raw is None
        else _positive_finite(loft_omega_raw, "loft_omega_max_deg_s")
    )
    if loft_omega_max_deg_s is not None and guidance.get("loft_omega_max_deg_s") is None:
        assumptions.append(
            f"guidance.loft_omega_max_deg_s missing -> shared default {loft_omega_max_deg_s:g}"
        )
    flight_time_gain_table = _gain_table(
        guidance.get("flight_time_gain_table"),
        layer_guidance["flight_time_gain_table"],
        assumptions,
        "guidance.flight_time_gain_table",
    )

    stages = [
        {
            "name": str(stage["name"]),
            "duration_s": float(stage["duration_s"]),
            "thrust_n": float(stage["thrust_n"]),
            "mass_lost_kg": float(stage["mass_lost_kg"]),
            "isp_s": float(stage["isp_s"]),
        }
        for stage in profile["propulsion"]["stages"]
    ]
    legacy_runtime_name = str(defaults["runtime_name"])
    natural_lift_enabled = True
    body_force_candidate: dict[str, Any] | None = None
    generalized_force_moment_candidate: dict[str, Any] | None = None
    legacy_rate_inner = False
    rate_loop_time_constant_s = 0.1
    fin_arm_as_length_fraction = False
    fin_translation_share = 1.0
    stall_cap_enabled = False
    if plant_model == LEGACY_CRITICAL_DAMPED_PLANT:
        runtime_name = legacy_runtime_name
        body_lift = defaults.get("legacy_body_lift") or {}
        natural_lift_enabled = bool(body_lift.get("enabled", True))
        cn_alpha_per_rad = _positive_finite(
            body_lift.get("cn_alpha_per_rad", 2.0),
            "legacy_body_lift.cn_alpha_per_rad",
        )
        fin_translation_share = _positive_finite(
            body_lift.get("fin_translation_share", 1.0),
            "legacy_body_lift.fin_translation_share",
        )
        stall_cap_enabled = bool(body_lift.get("stall_cap_enabled", True))
        normal_force_model = "body_cn_linear"
        release_version = "profile-adapter-v27-h2-spec"
        force_geometry_version = "h2_spec_packed_lift_cm_np_v13"
        plant_semantics = "fin_torque_body_aoa"
        fin_arm_as_length_fraction = False
        legacy_rate_inner = bool(defaults.get("acceleration_outer_rate_inner", False))
        if legacy_rate_inner:
            control_model_version = "spec_g_outer_rate_inner_v15"
            rate_loop_time_constant_s = _positive_finite(
                defaults.get("body_rate_inner_loop_time_constant_s", 0.1),
                "body_rate_inner_loop_time_constant_s",
            )
        else:
            control_model_version = (
                "raw_pid_fin_angle_tail_force_moment_body_cn_derived_critical_damping_v12"
            )
    elif plant_model == BODY_CM_TAIL_FORCE_PLANT:
        candidate = defaults.get("body_cm_candidate")
        if not isinstance(candidate, dict):
            raise ValueError("body_cm_candidate settings are required for body_cm_tail_force_moment_v1")
        cn_alpha_per_rad = _positive_finite(
            candidate.get("cn_alpha_body_per_rad", 2.0),
            "body_cm_candidate.cn_alpha_body_per_rad",
        )
        cp_cg_arm_over_diameter = _positive_finite(
            candidate.get("cp_cg_arm_over_diameter", 3.0),
            "body_cm_candidate.cp_cg_arm_over_diameter",
        )
        tail_distance_m = _positive_finite(
            aero["fin_moment_arm_m"], "aerodynamics.fin_moment_arm_m"
        )
        cm_alpha_per_rad = cn_alpha_per_rad * cp_cg_arm_over_diameter
        cn_q = -2.0 * cn_alpha_per_rad * cp_cg_arm_over_diameter
        cm_q = -2.0 * cn_alpha_per_rad * cp_cg_arm_over_diameter ** 2
        rate_loop_time_constant_s = _positive_finite(
            candidate.get("body_rate_inner_loop_time_constant_s", 0.1),
            "body_cm_candidate.body_rate_inner_loop_time_constant_s",
        )
        tail_alpha_multiplier = _positive_finite(
            candidate.get("tail_alpha_force_multiplier", 1.0),
            "body_cm_candidate.tail_alpha_force_multiplier",
        )
        tail_delta_multiplier = _positive_finite(
            candidate.get("tail_delta_force_multiplier", 1.0),
            "body_cm_candidate.tail_delta_force_multiplier",
        )
        fixed_lifting_surface_multiplier = _nonnegative_finite(
            candidate.get("fixed_lifting_surface_multiplier", 0.0),
            "body_cm_candidate.fixed_lifting_surface_multiplier",
        )
        fixed_lifting_surface_station_x_m = _finite(
            candidate.get("fixed_lifting_surface_station_x_m", 0.0),
            "body_cm_candidate.fixed_lifting_surface_station_x_m",
        )
        runtime_name = "profile_h2_body_cm_split_tail_fixed_lift_v3_candidate"
        normal_force_model = "body_circular_cn2"
        release_version = "profile-adapter-v12-body-cm-split-tail-fixed-lift-candidate"
        force_geometry_version = "body_cp_split_tail_near_cg_fixed_lift_v4_candidate"
        control_model_version = "raw_pid_accel_outer_rate_inner_three_source_fin_v17_candidate"
        plant_semantics = "body_cm_tail_force_moment"
        body_force_candidate = {
            "reference_area_mode": "caliber_circular_area",
            "cn_alpha_body_per_rad": cn_alpha_per_rad,
            "cp_cg_arm_over_diameter": cp_cg_arm_over_diameter,
            "force_evaluation": "single_force_at_cp_local_velocity",
            "moment_evaluation": "r_cp_cross_f_body",
            "derived_cn_q": cn_q,
            "derived_cm_alpha_per_rad": cm_alpha_per_rad,
            "derived_cm_q": cm_q,
            "inertia_closure": "I_perp=m*length^2/12_provisional",
        }
    else:
        candidate = defaults.get("generalized_aero_moment_candidate")
        if not isinstance(candidate, dict):
            raise ValueError(
                "generalized_aero_moment_candidate settings are required for "
                "generalized_aero_moment_v1"
            )
        generalized_cn_alpha = _bounded_coefficient(
            candidate.get("cn_alpha_per_rad", 2.0),
            "generalized_aero_moment_candidate.cn_alpha_per_rad",
        )
        generalized_cn_delta = _bounded_coefficient(
            candidate.get("cn_delta_per_rad", -40.0),
            "generalized_aero_moment_candidate.cn_delta_per_rad",
        )
        generalized_cm_alpha = _bounded_coefficient(
            candidate.get("cm_alpha_per_rad", -6.0),
            "generalized_aero_moment_candidate.cm_alpha_per_rad",
        )
        generalized_cm_delta = _bounded_coefficient(
            candidate.get("cm_delta_per_rad", 18.0),
            "generalized_aero_moment_candidate.cm_delta_per_rad",
        )
        generalized_cm_q = _bounded_coefficient(
            candidate.get("cm_q", -36.0),
            "generalized_aero_moment_candidate.cm_q",
        )
        requested_generalized_cm_alpha_dot = _bounded_coefficient(
            candidate.get("cm_alpha_dot", 0.0),
            "generalized_aero_moment_candidate.cm_alpha_dot",
        )
        generalized_cm_alpha_dot = 0.0
        if abs(requested_generalized_cm_alpha_dot) > 0.0:
            assumptions.append(
                "generalized_aero_moment_candidate.cm_alpha_dot is retained as a requested input only; "
                "runtime freezes Cm_alpha_dot=0 because no independent alpha_dot state is available, "
                "so current rate excitation identifies Cm_q only"
            )
        rate_loop_time_constant_s = _positive_finite(
            candidate.get("body_rate_inner_loop_time_constant_s", 0.1),
            "generalized_aero_moment_candidate.body_rate_inner_loop_time_constant_s",
        )
        runtime_name = "profile_h2_generalized_aero_moment_v1_candidate"
        normal_force_model = "generalized_coefficients"
        natural_lift_enabled = False
        cn_alpha_per_rad = generalized_cn_alpha
        release_version = "profile-adapter-v13-generalized-aero-moment-candidate"
        force_geometry_version = "generalized_cn_cm_shared_state_v1_candidate"
        control_model_version = "raw_pid_accel_outer_rate_inner_generalized_v1_candidate"
        plant_semantics = "generalized_aero_moment"
        generalized_force_moment_candidate = {
            "reference_area_mode": "caliber_circular_area",
            "reference_length_mode": "caliber_m",
            "cn_alpha_per_rad": generalized_cn_alpha,
            "cn_delta_per_rad": generalized_cn_delta,
            "cm_alpha_per_rad": generalized_cm_alpha,
            "cm_delta_per_rad": generalized_cm_delta,
            "cm_q": generalized_cm_q,
            "cm_alpha_dot": generalized_cm_alpha_dot,
            "cm_alpha_dot_requested": requested_generalized_cm_alpha_dot,
            "cm_alpha_dot_runtime_status": (
                "frozen_zero_unsupported_no_independent_alpha_dot_state"
            ),
            "cn_q_per_rad": 0.0,
            "cn_alpha_dot_per_rad": 0.0,
            "delta_definition": "actual_fin_angle_rad",
            "alpha_definition": "CG_wind_normal_alpha_rad",
            "alpha_dot_definition": (
                "unavailable_in_current_state; runtime Cm_alpha_dot is frozen at zero"
            ),
            "q_hat_definition": "body_rate*reference_length/(2*speed)",
            "equations": {
                "CN": "CN_alpha*alpha + CN_delta*delta",
                "Cm": "Cm_alpha*alpha + Cm_delta*delta + Cm_q*q_hat",
                "normal_force": "q_dyn*S*CN",
                "pitch_moment": "q_dyn*S*d*Cm",
            },
            "parameter_boundary": (
                "CN_alpha, CN_delta, Cm_alpha, Cm_delta, and Cm_q are signed unsupported candidate values "
                "bounded to +/-1000; Cm_alpha_dot is explicitly frozen at runtime zero because alpha_dot "
                "is not independently represented. None is inferred from x_W/k_W, distFromCmToStab, "
                "finsLatAccel, raw PID, or drag"
            ),
            "inertia_closure": "I_perp=m*length^2/12_provisional",
        }
    wing_multiplier = float(geometry["wing_area_multiplier"])
    profile_lift_scale = float(aero["lift_area_scale"])
    lift_area_scale = profile_lift_scale / wing_multiplier
    assumptions.append(
        f"aerodynamics.lift_area_scale normalized by geometry.wing_area_multiplier -> {lift_area_scale:.9g}"
    )
    effective_drag_scale = float(aero["drag_scale"]) * float(layer_drag["effective_drag_scale"])
    shape_mode = str(defaults.get("drag_shape_mode", layer_drag["shape_mode"]))
    assumptions.append(
        "profile CdA0 = CxK * S_w * interpolated 1943-law*1.10 Cx(M); "
        "the AIM-120A fitted 0.2995 scale is not applied "
        f"(effective_drag_scale={effective_drag_scale:.9g}, shape_mode={shape_mode})"
    )
    load_factor_max_raw = performance.get("load_factor_max_g")
    if load_factor_max_raw is None:
        load_factor_max_g = float(guidance["maximum_lateral_acceleration_g"])
        assumptions.append(
            "performance.load_factor_max_g missing -> guidance.maximum_lateral_acceleration_g (reqAccelMax)"
        )
    else:
        load_factor_max_g = _positive_finite(
            load_factor_max_raw, "performance.load_factor_max_g"
        )
    config = {
        "schema_version": 3,
        "release_version": release_version,
        "model_label": f"{profile['missile_id']}_{runtime_name}",
        "runtime_adapter": runtime_name,
        "aero_model_version": "effective_cda_v1",
        "force_geometry_version": force_geometry_version,
        "control_model_version": control_model_version,
        "reference": {
            "source": "Unit-explicit missile profile adapted to shared H2 candidate runtime",
            "solver_reproduction_claimed": False,
            "runtime_boundary": (
                defaults["boundary"]
                if plant_model == LEGACY_CRITICAL_DAMPED_PLANT
                else (
                    "Unsupported local reduced-order generalized CN/Cm candidate; "
                    "not War Thunder native physics or a validated solver reproduction."
                    if plant_model == GENERALIZED_AERO_MOMENT_PLANT
                    else "Unsupported local reduced-order body-Cm/tail-force candidate; not War Thunder native physics or a validated solver reproduction."
                )
            ),
        },
        "geometry": {
            "initial_mass_kg": float(geometry["initial_mass_kg"]),
            "caliber_m": float(geometry["caliber_m"]),
            "length_m": float(geometry["length_m"]),
            "wing_area_multiplier": float(geometry["wing_area_multiplier"]),
            "reference_area_mode": str(geometry["reference_area_mode"]),
        },
        "aerodynamics": {
            "cx_k": float(aero["cx_k"]),
            "cx_vs_aoa": cx_aoa,
            "normal_force_model": normal_force_model,
            "cn_alpha_per_rad": cn_alpha_per_rad,
            "normal_force_cap_enabled": (
                plant_model == LEGACY_CRITICAL_DAMPED_PLANT and stall_cap_enabled
            ),
            "fin_translation_share": fin_translation_share,
            "cx_vs_fin_delta": float(defaults.get("cx_vs_fin_delta", 0.0)),
            "cy_k": cn_alpha_per_rad,
            "max_cy_at_aoa": max_cy,
            "max_cy_interpretation": str(layer_aero["max_cy_interpretation"]),
            "fins_lateral_acceleration_g": fins_g,
            "path_g_from_alpha": plant_model == LEGACY_CRITICAL_DAMPED_PLANT,
            "path_g_scales_with_arm_times_length": False,
            "distance_cm_to_stabilizer_m": float(aero["fin_moment_arm_m"]),
            **(
                {"fin_arm_as_length_fraction": True}
                if fin_arm_as_length_fraction
                else {}
            ),
            "horizontal_fin_aoa_limit_deg": _fins_aoa_limit_deg(aero["fin_aoa_limit_rad"]["horizontal"]),
            "vertical_fin_aoa_limit_deg": _fins_aoa_limit_deg(aero["fin_aoa_limit_rad"]["vertical"]),
            "thrust_vector_angle_deg": float(layer_aero["thrust_vector_angle_deg"]),
            "natural_lift_enabled": natural_lift_enabled,
            "natural_lift_fraction": float(layer_aero["natural_lift_fraction"]),
            "mach_drag": _mach_settings(aero["mach_drag_correction"], layer_aero["mach_drag"], assumptions, "aerodynamics.mach_drag_correction"),
            "mach_lift": _mach_settings(aero["mach_lift_correction"], layer_aero["mach_lift"], assumptions, "aerodynamics.mach_lift_correction"),
        },
        "drag_model": {
            "shape_mode": shape_mode,
            "drag_scale": effective_drag_scale,
            "area_basis_mode": str(geometry["reference_area_mode"]),
            "alpha_drag_scale": float(layer_drag["alpha_drag_scale"]),
            "alpha_drag_cap_rad": float(layer_drag["alpha_drag_cap_rad"]),
            "alpha_drag_area_basis_mode": (
                "caliber_area"
                if shape_mode == INTERPOLATED_CX_1943_X1_10
                else str(geometry["reference_area_mode"])
            ),
            "alpha_drag_mach_shape": shape_mode != INTERPOLATED_CX_1943_X1_10,
            "lift_area_scale": lift_area_scale,
            **(
                {
                    "cx_vs_mach": [list(knot) for knot in CX_1943_X1_10_TABLE],
                    "cx_vs_mach_source": "1943_law_x1_10_linear_interpolation",
                }
                if shape_mode == INTERPOLATED_CX_1943_X1_10
                else {}
            ),
        },
        "propulsion": {"stages": stages, "unpowered_variant": "zero_force_zero_mass_loss"},
        "performance": {
            "maximum_speed_mps": float(performance["maximum_speed_mps"]),
            "maximum_distance_m": float(performance["maximum_distance_m"]),
            "lifetime_s": float(performance["lifetime_s"]),
            "load_factor_max_g": load_factor_max_g,
            "maximum_speed_is_hard_clamp": bool(UNIVERSAL_H2_LAYER["performance"]["maximum_speed_is_hard_clamp"]),
        },
        "guidance": {
            "type": str(guidance["seeker_type"]),
            "lock_range_m": float(lock_range),
            "maximum_lateral_acceleration_g": float(guidance["maximum_lateral_acceleration_g"]),
            "maximum_angular_rate_deg_s": maximum_rate,
            "pn_gain": float(guidance["pn_gain"]),
            "guidance_timeout_s": float(layer_guidance["guidance_timeout_s"]),
            "guidance_timeout_semantics": str(layer_guidance["guidance_timeout_semantics"]),
            "proximity_fuse_enabled": True,
            "proximity_radius_m": float(guidance["proximity_radius_m"]),
            "lofting_enabled": bool(guidance["lofting_enabled"]),
            "lofting_elevation_deg": float(guidance["lofting_elevation_deg"]),
            "loft_exit_distance_m": float(loft_exit_distance),
            "loft_exit_time_to_go_s": float(loft_exit_tgo),
            **(
                {"loft_omega_max_deg_s": loft_omega_max_deg_s}
                if loft_omega_max_deg_s is not None
                else {}
            ),
            "target_elevation_deg": float(layer_guidance["target_elevation_deg"]),
            "omega_max_deg_s": float(layer_guidance["omega_max_deg_s"]),
            "angle_to_acceleration_multiplier": float(layer_guidance["angle_to_acceleration_multiplier"]),
            "flight_time_gain_table": flight_time_gain_table,
            "time_to_hit_gain_table": copy.deepcopy(layer_guidance["time_to_hit_gain_table"]),
        },
        "control": {
            "limit_angle_of_attack_enabled": bool(layer_control["limit_angle_of_attack_enabled"]),
            "maximum_body_angle_of_attack_deg": float(layer_control["maximum_body_angle_of_attack_deg"]),
            "feedback_measurement": (
                "body_specific_force_g"
                if plant_model == LEGACY_CRITICAL_DAMPED_PLANT
                else "cg_wind_normal_specific_force_g"
            ),
            "controller_command_basis": (
                "legacy_body_up_right_required_specific_force"
                if plant_model == LEGACY_CRITICAL_DAMPED_PLANT
                else "cg_velocity_wind_normal_required_specific_force"
            ),
            "plant_semantics": plant_semantics,
            "integral_limit_semantics": "term",
            "fin_aoa_moment_enabled": True,
            # The body-Cm / generalized candidates use the raw PID as an
            # acceleration outer loop whose output commands body rate.
            # The current fin-torque runtime can opt into the same cascade.
            "pid_output_semantics": (
                "fin_angle_rad"
                if plant_model == LEGACY_CRITICAL_DAMPED_PLANT and not legacy_rate_inner
                else "body_rate_command_rad_s"
            ),
            "pid_error_scale": 1.0,
            "base_indicated_speed_kmh": base_indicated_speed,
            "base_indicated_speed_mode": (
                "none"
                if plant_model == GENERALIZED_AERO_MOMENT_PLANT
                else (
                    "fin_authority_q"
                    if base_indicated_speed is not None
                    and (
                        plant_model == BODY_CM_TAIL_FORCE_PLANT
                        or bool(defaults.get("fin_force_q_scaling", False))
                    )
                    else "none"
                )
            ),
            **(
                {
                    "base_indicated_speed_ratio_max": _positive_finite(
                        defaults.get("fin_force_q_ratio_max", 4.0),
                        "fin_force_q_ratio_max",
                    )
                }
                if plant_model == LEGACY_CRITICAL_DAMPED_PLANT
                and defaults.get("fin_force_q_ratio_max") is not None
                else {}
            ),
            "pid": {
                "switch_time_s": 3.4028234663852886e38,
                "p": mapped_pid["p"],
                "i": mapped_pid["i"],
                "d": mapped_pid["d"],
                "integral_limit": _number(pid.get("integral_limit"), 1.0, assumptions, "control.pid.integral_limit"),
            },
            # Compatibility field for frozen normalized-command configs.
            "fin_command_limit": float(control["fin_command_limit"]),
            "actuator_time_constant_s": actuator_tau,
            "derivative_filter_time_constant_s": float(layer_control["derivative_filter_time_constant_s"]),
        },
        "atmosphere": copy.deepcopy(UNIVERSAL_H2_LAYER["atmosphere"]),
        "numerics": copy.deepcopy(UNIVERSAL_H2_LAYER["numerics"]),
    }
    if body_force_candidate is not None:
        pitch_travel_rad = _fins_aoa_limit_rad(aero["fin_aoa_limit_rad"]["horizontal"])
        yaw_travel_rad = _fins_aoa_limit_rad(aero["fin_aoa_limit_rad"]["vertical"])
        config["aerodynamics"]["body_cp_force_candidate"] = body_force_candidate
        config["attitude_candidate"] = {
            "primary_orientation": "unit_quaternion_body_to_inertial_wxyz",
            "body_basis": "forward_up_right",
            "body_angular_velocity_components": "[0,-yaw_rate,+pitch_rate]",
            "roll_dynamics": "not_included",
            "pitch_yaw": "derived_telemetry_only",
        }
        config["aerodynamics"]["tail_station_x_m"] = -tail_distance_m
        config["aerodynamics"]["tail_station_semantics"] = {
            "source_field": "distFromCmToStab",
            "interpretation": "unverified_empirical_arm_provisionally_metres_aft",
        }
        config["control"]["fin_actuator_travel"] = {
            "pitch_limit_rad": pitch_travel_rad,
            "yaw_limit_rad": yaw_travel_rad,
            "source_field": "finsAoaHor/finsAoaVer",
        }
        config["aerodynamics"]["empirical_fin_authority"] = {
            "acceleration_authority_g": fins_g,
            "pitch_incidence_reference_rad": pitch_travel_rad,
            "yaw_incidence_reference_rad": yaw_travel_rad,
            "radial_allocation": "unit_disk_empirical_authority_allocation_not_stall_model",
            "source_field": "finsLatAccel",
        }
        sea_level_density = 1.225000018
        if base_indicated_speed is None:
            raise ValueError(
                "split-tail candidate requires a positive profile base_indicated_speed_kmh"
            )
        base_speed_mps = float(base_indicated_speed) / 3.6
        q_base = 0.5 * sea_level_density * base_speed_mps ** 2
        config["aerodynamics"]["split_tail_candidate"] = {
            "model": "algebraic_split_empirical_tail_v1_candidate",
            "tail_alpha_force_multiplier": tail_alpha_multiplier,
            "tail_delta_force_multiplier": tail_delta_multiplier,
            "q_base_pa": q_base,
            "tail_force_cap_mode": "radial_current_mass_q_over_qbase",
            "tail_gain_mass_mode": "current_mass",
            "tail_alpha_moment_arm_m": -tail_distance_m,
            "tail_delta_moment_arm_m": -tail_distance_m,
            "fin_mechanical_limit_pitch_rad": pitch_travel_rad,
            "fin_mechanical_limit_yaw_rad": yaw_travel_rad,
            "fin_authority_angle_reference_pitch_rad": pitch_travel_rad,
            "fin_authority_angle_reference_yaw_rad": yaw_travel_rad,
            "boundary": "unsupported fixed-tail/fixed-airframe restoring derivative multiplier; not a real missile coefficient",
        }
        body_area_slope_m2_per_rad = (
            math.pi * float(geometry["caliber_m"]) ** 2 / 4.0 * cn_alpha_per_rad
        )
        config["aerodynamics"]["fixed_lifting_surface_candidate"] = {
            "model": "linear_station_wind_normal_force_v2_candidate",
            "fixed_lifting_surface_multiplier": fixed_lifting_surface_multiplier,
            "body_normal_force_area_slope_m2_per_rad": body_area_slope_m2_per_rad,
            "fixed_lifting_surface_area_slope_m2_per_rad": (
                fixed_lifting_surface_multiplier * body_area_slope_m2_per_rad
            ),
            "station_x_m": fixed_lifting_surface_station_x_m,
            "moment_model": "r_cg_cross_f_equals_zero",
            "boundary": "unsupported near-CG lifting-surface multiplier and signed x_W station; not derived from wingAreaMult or a real missile coefficient",
        }
        config["control"]["candidate_rate_inner_loop"] = {
            "time_constant_s": rate_loop_time_constant_s,
            "outer_pid_output_semantics": "body_rate_command_rad_s",
            "inner_loop_semantics": "first_order_rate_target_with_local_angular_acceleration_and_tail_effectiveness_inversion",
            "source": "shared_unsupported_candidate_assumption",
        }
    if plant_model == LEGACY_CRITICAL_DAMPED_PLANT and legacy_rate_inner:
        path_tau_raw = control.get("path_rate_time_constant_s")
        if path_tau_raw is None:
            path_tau_raw = defaults.get("path_rate_time_constant_s", 0.35)
            assumptions.append(
                "control.path_rate_time_constant_s missing -> shared default "
                f"{float(path_tau_raw):g}"
            )
        omega_ref_raw = control.get("rate_error_for_full_fin_rad_s")
        if omega_ref_raw is None:
            omega_ref_raw = defaults.get("rate_error_for_full_fin_rad_s", 0.35)
            assumptions.append(
                "control.rate_error_for_full_fin_rad_s missing -> shared default "
                f"{float(omega_ref_raw):g}"
            )
        close_ki_raw = control.get("path_close_integral_gain_per_s")
        if close_ki_raw is None:
            close_ki_raw = defaults.get("path_close_integral_gain_per_s", 0.0)
            assumptions.append(
                "control.path_close_integral_gain_per_s missing -> shared default "
                f"{float(close_ki_raw):g}"
            )
        close_i_limit_raw = control.get("path_close_integral_limit_g_s")
        if close_i_limit_raw is None:
            close_i_limit_raw = defaults.get("path_close_integral_limit_g_s", 20.0)
            assumptions.append(
                "control.path_close_integral_limit_g_s missing -> shared default 20"
            )
        config["control"]["candidate_rate_inner_loop"] = {
            "time_constant_s": rate_loop_time_constant_s,
            "path_rate_time_constant_s": _positive_finite(
                path_tau_raw,
                "path_rate_time_constant_s",
            ),
            "rate_error_for_full_fin_rad_s": _positive_finite(
                omega_ref_raw,
                "rate_error_for_full_fin_rad_s",
            ),
            "path_close_integral_gain_per_s": _nonnegative_finite(
                close_ki_raw,
                "path_close_integral_gain_per_s",
            ),
            "path_close_integral_limit_g_s": _positive_finite(
                close_i_limit_raw,
                "path_close_integral_limit_g_s",
            ),
            "outer_pid_output_semantics": "body_rate_command_rad_s",
            "inner_loop_semantics": (
                "proportional_rate_error_to_profile_fin_fraction_shared_omega_ref"
            ),
            "source": "shared_unsupported_candidate_assumption",
        }
    if generalized_force_moment_candidate is not None:
        pitch_travel_rad = _fins_aoa_limit_rad(aero["fin_aoa_limit_rad"]["horizontal"])
        yaw_travel_rad = _fins_aoa_limit_rad(aero["fin_aoa_limit_rad"]["vertical"])
        config["aerodynamics"]["generalized_aero_moment_candidate"] = (
            generalized_force_moment_candidate
        )
        config["attitude_candidate"] = {
            "primary_orientation": "unit_quaternion_body_to_inertial_wxyz",
            "body_basis": "forward_up_right",
            "body_angular_velocity_components": "[0,-yaw_rate,+pitch_rate]",
            "roll_dynamics": "not_included",
            "pitch_yaw": "derived_telemetry_only",
        }
        config["control"]["fin_actuator_travel"] = {
            "pitch_limit_rad": pitch_travel_rad,
            "yaw_limit_rad": yaw_travel_rad,
            "source_field": "finsAoaHor/finsAoaVer",
        }
        config["control"]["candidate_rate_inner_loop"] = {
            "time_constant_s": rate_loop_time_constant_s,
            "outer_pid_output_semantics": "body_rate_command_rad_s",
            "inner_loop_semantics": (
                "first_order_rate_target_with_independent_Cm_delta_effectiveness"
            ),
            "source": "shared_unsupported_candidate_assumption",
        }
    sensor_model = guidance.get("sensor_model")
    if isinstance(sensor_model, dict):
        # Keep the raw mapped candidate beside the shared guidance values.
        config["guidance"]["sensor_model"] = copy.deepcopy(sensor_model)
    else:
        # Every runnable profile can now opt into sensor_track.  This fallback
        # uses only profile-level geometry already present in the contract; it
        # is not a guessed radar/IR seeker model and does not copy AIM-120A
        # Doppler, RCS, or noise parameters to another missile.
        config["guidance"]["sensor_model"] = {
            "provider": "profile_kinematic_v1",
            "seeker_type": str(guidance["seeker_type"]),
            "lock_range_m": float(lock_range),
            "maximum_angular_rate_deg_s": float(maximum_rate),
            "parameter_sources": {
                "provider": {
                    "source": "assumed",
                    "path": None,
                    "note": "未声明 seeker 参数时使用 profile 几何 fallback；不代表真实 seeker 方程。",
                },
                "seeker_type": {
                    "source": "profile",
                    "path": "guidance.seeker_type",
                    "note": "保留 profile 的 seeker 类型标签。",
                },
                "lock_range_m": {
                    "source": "profile_or_runtime_fallback",
                    "path": "guidance.lock_range_m",
                    "note": "使用已有锁定距离；缺失时沿用现有 runtime 最大距离回退。",
                },
                "maximum_angular_rate_deg_s": {
                    "source": "profile_or_runtime_fallback",
                    "path": "guidance.maximum_angular_rate_deg_s",
                    "note": "使用已有最大角速率；不增加独立角度门。",
                },
            },
        }
    return config, assumptions


__all__ = [
    "BODY_CM_TAIL_FORCE_PLANT",
    "GENERALIZED_AERO_MOMENT_PLANT",
    "LEGACY_CRITICAL_DAMPED_PLANT",
    "SUPPORTED_MODEL_TYPES",
    "SUPPORTED_PLANT_MODELS",
    "UNIVERSAL_H2_LAYER",
    "build_h2_candidate_config",
    "load_runtime_defaults",
    "unsupported_model_types",
]
