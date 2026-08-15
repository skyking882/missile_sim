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
from functools import lru_cache
from pathlib import Path
from typing import Any


SUPPORTED_MODEL_TYPES = {
    "dynamics": {"h2_reduced_order"},
    "propulsion": {"staged_solid_rocket"},
    "aerodynamics": {"conventional_fin"},
    "control": {"aerodynamic_fin"},
    "guidance": {"pn", "pn_loft"},
}


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
        "effective_drag_scale": 0.29949663,
        "shape_mode": "scaled_h1_shape",
        "alpha_drag_scale": 1.0,
        "alpha_drag_cap_rad": 1.2,
    },
    "performance": {"maximum_speed_is_hard_clamp": False},
    "guidance": {
        "guidance_timeout_s": 0.6,
        "guidance_timeout_semantics": "unresolved_do_not_disable_after_timeout",
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
        "angular_response_scale": 0.04,
        "angular_damping": 1.0,
        "maximum_angular_rate_deg_s": 60.0,
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


def _mach_settings(node: dict[str, Any], fallback: dict[str, Any], assumptions: list[str], label: str) -> dict[str, float]:
    if node.get("model") == "scalar" and isinstance(node.get("parameters"), (int, float)):
        return {"base": float(node["parameters"]), "transonic_peak": 0.0, "center": 1.0, "width": float(fallback["width"])}
    if node.get("model") != "not_declared":
        assumptions.append(f"{label} {node.get('model')} not mapped -> universal H2 runtime shape")
    else:
        assumptions.append(f"{label} not declared -> universal H2 runtime shape")
    return {key: float(value) for key, value in fallback.items()}


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

    raw_pid = {
        "p": _number(pid.get("p"), 0.0, assumptions, "control.pid.p"),
        "i": _number(pid.get("i"), 0.0, assumptions, "control.pid.i"),
        "d": _number(pid.get("d"), 0.0, assumptions, "control.pid.d"),
    }
    mapped_pid = raw_pid
    angular_response_scale = float(layer_control["angular_response_scale"])
    assumptions.append(
        "control PID uses the selected profile's raw p/i/d values; no AIM-120A floor or gain substitution"
    )
    assumptions.append(
        "control.angular_response_scale is shared without inverse arm/length normalization; "
        "profile arm and length remain active in the attitude equation"
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
    maximum_rate = _number(maximum_rate, layer_control["maximum_angular_rate_deg_s"], assumptions, "control.max_pitch_yaw_rate_deg_s")
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
    runtime_name = str(defaults["runtime_name"])
    natural_lift_enabled = False
    assumptions.append("natural lift disabled: CyK is absent from the supported missile profiles and is not imputed")
    wing_multiplier = float(geometry["wing_area_multiplier"])
    profile_lift_scale = float(aero["lift_area_scale"])
    lift_area_scale = profile_lift_scale / wing_multiplier
    assumptions.append(
        f"aerodynamics.lift_area_scale normalized by geometry.wing_area_multiplier -> {lift_area_scale:.9g}"
    )
    effective_drag_scale = float(aero["drag_scale"]) * float(layer_drag["effective_drag_scale"])
    assumptions.append(
        f"aerodynamics.drag_scale multiplied by universal effective CdA scale -> {effective_drag_scale:.9g}"
    )
    config = {
        "schema_version": 3,
        "release_version": "profile-adapter-v2-true-difference",
        "model_label": f"{profile['missile_id']}_{runtime_name}",
        "aero_model_version": "effective_cda_v1",
        "force_geometry_version": "flow_normal_v1",
        "control_model_version": "raw_pid_body_g_fin_aoa_moment_v4",
        "reference": {
            "source": "Unit-explicit missile profile adapted to shared H2 candidate runtime",
            "solver_reproduction_claimed": False,
            "runtime_boundary": defaults["boundary"],
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
            "max_cy_at_aoa": max_cy,
            "max_cy_interpretation": str(layer_aero["max_cy_interpretation"]),
            "fins_lateral_acceleration_g": fins_g,
            "distance_cm_to_stabilizer_m": float(aero["fin_moment_arm_m"]),
            "horizontal_fin_aoa_limit_deg": float(aero["fin_aoa_limit_rad"]["horizontal"]) * 180.0 / 3.141592653589793,
            "vertical_fin_aoa_limit_deg": float(aero["fin_aoa_limit_rad"]["vertical"]) * 180.0 / 3.141592653589793,
            "thrust_vector_angle_deg": float(layer_aero["thrust_vector_angle_deg"]),
            "natural_lift_enabled": natural_lift_enabled,
            "natural_lift_fraction": float(layer_aero["natural_lift_fraction"]),
            "mach_drag": _mach_settings(aero["mach_drag_correction"], layer_aero["mach_drag"], assumptions, "aerodynamics.mach_drag_correction"),
            "mach_lift": _mach_settings(aero["mach_lift_correction"], layer_aero["mach_lift"], assumptions, "aerodynamics.mach_lift_correction"),
        },
        "drag_model": {
            "shape_mode": str(layer_drag["shape_mode"]),
            "drag_scale": effective_drag_scale,
            "area_basis_mode": str(geometry["reference_area_mode"]),
            "alpha_drag_scale": float(layer_drag["alpha_drag_scale"]),
            "alpha_drag_cap_rad": float(layer_drag["alpha_drag_cap_rad"]),
            "lift_area_scale": lift_area_scale,
        },
        "propulsion": {"stages": stages, "unpowered_variant": "zero_force_zero_mass_loss"},
        "performance": {
            "maximum_speed_mps": float(performance["maximum_speed_mps"]),
            "maximum_distance_m": float(performance["maximum_distance_m"]),
            "lifetime_s": float(performance["lifetime_s"]),
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
            "target_elevation_deg": float(layer_guidance["target_elevation_deg"]),
            "omega_max_deg_s": float(layer_guidance["omega_max_deg_s"]),
            "angle_to_acceleration_multiplier": float(layer_guidance["angle_to_acceleration_multiplier"]),
            "flight_time_gain_table": copy.deepcopy(layer_guidance["flight_time_gain_table"]),
            "time_to_hit_gain_table": copy.deepcopy(layer_guidance["time_to_hit_gain_table"]),
        },
        "control": {
            "limit_angle_of_attack_enabled": bool(layer_control["limit_angle_of_attack_enabled"]),
            "maximum_body_angle_of_attack_deg": float(layer_control["maximum_body_angle_of_attack_deg"]),
            "feedback_measurement": "body_specific_force_g",
            "integral_limit_semantics": "term",
            "fin_aoa_moment_enabled": True,
            "pid": {
                "switch_time_s": 3.4028234663852886e38,
                "p": mapped_pid["p"],
                "i": mapped_pid["i"],
                "d": mapped_pid["d"],
                "integral_limit": _number(pid.get("integral_limit"), 1.0, assumptions, "control.pid.integral_limit"),
            },
            # Unit command already means the profile's full finsLatAccel
            # authority.  Scaling this again by finsAoa would double-count the
            # profile-specific fin limit and permit acceleration above it.
            "fin_command_limit": float(control["fin_command_limit"]),
            "actuator_time_constant_s": actuator_tau,
            "derivative_filter_time_constant_s": float(layer_control["derivative_filter_time_constant_s"]),
            "angular_response_scale": angular_response_scale,
            "angular_damping": float(layer_control["angular_damping"]),
            "max_pitch_yaw_rate_deg_s": maximum_rate,
        },
        "atmosphere": copy.deepcopy(UNIVERSAL_H2_LAYER["atmosphere"]),
        "numerics": copy.deepcopy(UNIVERSAL_H2_LAYER["numerics"]),
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


__all__ = ["SUPPORTED_MODEL_TYPES", "UNIVERSAL_H2_LAYER", "build_h2_candidate_config", "load_runtime_defaults", "unsupported_model_types"]
