"""Frozen, full-forward R-77-1 replay-fit experiment.

This explicitly scoped planar candidate is separate from the general H2 GUI
model.  Its time-varying request, plant, missile inputs, and scenario are pinned
to the accepted 2.57.1.107 fit.  It performs a fresh free-flight integration;
it does not replay stored trajectory samples.
"""

from __future__ import annotations

from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any

from .alpha_servo import AlphaServoParameters, AlphaServoResult, run_alpha_servo
from .alpha_servo import FinMomentSample, planar_target_motion
from .atmosphere import StandardAtmosphere
from .drag_models import effective_cda0, effective_cda_alpha
from .units import G0_MPS2 as G0


_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "r77_1_fixed_experiment_2.57.1.107.json"


@lru_cache(maxsize=1)
def _definition() -> dict[str, Any]:
    data = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    if data.get("mode_id") != "r77_1_fixed_257_request_fit_v1":
        raise RuntimeError("invalid fixed R-77-1 experiment configuration")
    return data


def public_definition() -> dict[str, Any]:
    """Small UI/API descriptor; no paths or internal solver config are exposed."""
    data = _definition()
    return {
        "mode_id": data["mode_id"],
        "label": data["label"],
        "parameter_source": {
            "version": data["source_version"],
            "commit": data["source_commit"],
        },
        "scenario": dict(data["scenario"]),
        "fit_artifacts": {
            "fit_json_sha256": data["source_artifacts"]["fit_json_sha256"],
            "accepted_forward_tsv_sha256": data["source_artifacts"]["accepted_forward_tsv_sha256"],
        },
    }


_SCENARIO_DEFAULTS: dict[str, Any] = {
    "launch_pitch_deg": 0.0,
    "launch_heading_deg": 0.0,
    "target_altitude_m": 6300.0,
    "target_vertical_heading_deg": 0.0,
    "target_constant_turn_g": 0.0,
    "max_simulation_time_s": None,
    "observation_mode": "ideal_truth",
    "loft_enabled": False,
    "simulation_dt_s": None,
    "datalink_enabled": True,
    "datalink_disconnect_time_s": None,
    "inertial_drift_direction": [0.0, 0.0, 1.0],
}


def _equal(value: Any, expected: Any) -> bool:
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        return value == expected
    if isinstance(expected, (int, float)):
        if isinstance(value, bool):
            return False
        try:
            number = float(value)
        except (TypeError, ValueError):
            return False
        return math.isfinite(number) and math.isclose(number, float(expected), rel_tol=0.0, abs_tol=1e-9)
    if isinstance(expected, list):
        return isinstance(value, (list, tuple)) and len(value) == len(expected) and all(
            _equal(actual, wanted) for actual, wanted in zip(value, expected)
        )
    return value == expected


def validate_experiment_scenario(scenario: Any) -> dict[str, Any]:
    if not isinstance(scenario, dict):
        raise ValueError("R-77-1 固定实验必须提供完整的 JSON 场景对象。")
    expected = dict(_definition()["scenario"])
    expected.update(_SCENARIO_DEFAULTS)
    allowed = set(expected)
    unknown = sorted(set(scenario) - allowed)
    if unknown:
        raise ValueError("R-77-1 固定实验不接受额外场景字段：" + ", ".join(unknown))
    for key, wanted in expected.items():
        if key in scenario and not _equal(scenario[key], wanted):
            raise ValueError(f"R-77-1 固定实验场景已锁定；{key} 必须为 {wanted!r}。")
        if key not in scenario and key in _definition()["scenario"]:
            raise ValueError(f"R-77-1 固定实验缺少固定场景字段：{key}。")
    return expected


class _R77FixedMomentModel:
    def __init__(self, config: dict[str, Any], parameters: AlphaServoParameters):
        self.config = config
        self.parameters = parameters
        self.atmo = StandardAtmosphere().sample(6300.0)
        # Preserve the accepted candidate's independently selected damping
        # derivative and the local-incidence contribution already in its plant.
        self.damping_m4 = (
            parameters.rate_damping_derivative_m4
            + parameters.restoring_area_per_rad * parameters.stability_distance_m ** 2
        )

    def sample(
        self,
        speed: float,
        alpha_h: float,
        body_rate: float,
        pressure: float,
        mass: float,
        control: float,
        parameters: AlphaServoParameters,
        total_alpha: float,
    ) -> FinMomentSample:
        p = parameters
        lift = pressure * p.lift_area_per_rad * alpha_h
        mach = speed / self.atmo.speed_of_sound_mps
        drag = pressure * (
            effective_cda0(mach, self.config)
            + effective_cda_alpha(total_alpha, 0.0, mach, self.config)
        )
        cosine, sine = math.cos(alpha_h), math.sin(alpha_h)
        normal = lift * cosine + drag * sine
        axial = drag * cosine - lift * sine
        residual = max(
            abs(lift - (normal * cosine - axial * sine)),
            abs(drag - (axial * cosine + normal * sine)),
        )

        # The accepted blue candidate uses its fitted q*A*delta control scale.
        equivalent_control = pressure * p.fin_force_area_per_rad * control
        limit = mass * G0 * p.fin_lateral_acceleration_limit_g
        limited = max(-limit, min(limit, equivalent_control))
        control_moment = p.stability_distance_m * limited
        damping_coefficient = pressure / max(speed, 1e-6) * self.damping_m4
        damping_moment = -damping_coefficient * body_rate
        aerodynamic_center_moment = control_moment + damping_moment
        passive_moment = -p.stability_distance_m * normal
        moment = passive_moment + aerodynamic_center_moment
        inertia = mass * p.length_m ** 2 / 12.0
        return _R77FixedMomentSample(
            local_alpha_rad=alpha_h,
            control_force_raw_n=-equivalent_control,
            control_force_n=-limited,
            control_force_limit_n=limit,
            passive_force_n=normal,
            control_moment_nm=control_moment,
            passive_moment_nm=passive_moment,
            equivalent_normal_force_n=normal,
            moment_nm=moment,
            inertia_kg_m2=inertia,
            angular_acceleration_rad_s2=moment / inertia,
            rate_damping_moment_nm=damping_moment,
            rate_damping_per_s=damping_coefficient / inertia,
            body_axial_force_n=axial,
            aerodynamic_center_moment_nm=aerodynamic_center_moment,
            normalized_control=control / p.horizontal_fin_limit_rad,
            force_rotation_residual_n=residual,
            moment_transfer_residual_nm=abs(moment - (-p.stability_distance_m * normal + aerodynamic_center_moment)),
        )


from dataclasses import dataclass


@dataclass(frozen=True)
class _R77FixedMomentSample(FinMomentSample):
    body_axial_force_n: float
    aerodynamic_center_moment_nm: float
    normalized_control: float
    force_rotation_residual_n: float
    moment_transfer_residual_nm: float


def _interpolate(time_s: float, knots: list[list[float]] | list[float], values: list[float] | None = None) -> float:
    if values is None:
        pairs = knots
        times = [float(row[0]) for row in pairs]  # type: ignore[index]
        vals = [float(row[1]) for row in pairs]  # type: ignore[index]
    else:
        times = [float(value) for value in knots]  # type: ignore[arg-type]
        vals = [float(value) for value in values]
    if time_s <= times[0]:
        return vals[0]
    if time_s >= times[-1]:
        return vals[-1]
    for index in range(1, len(times)):
        if time_s <= times[index]:
            fraction = (time_s - times[index - 1]) / (times[index] - times[index - 1])
            return vals[index - 1] + fraction * (vals[index] - vals[index - 1])
    return vals[-1]


def _scheduled_request(time_s: float, live_pn_g: float) -> float:
    definition = _definition()
    schedule = definition["guidance_schedule"]
    knots = schedule["knots_s"]
    fitted = schedule["pre_gain_request_knots_g"]
    last_time = float(knots[-1])
    pn_only_at = float(schedule["pn_only_after_s"])
    if time_s <= last_time:
        pre_gain = _interpolate(time_s, knots, fitted)
    elif time_s >= pn_only_at:
        pre_gain = live_pn_g
    else:
        fraction = (time_s - last_time) / (pn_only_at - last_time)
        weight = fraction * fraction * (3.0 - 2.0 * fraction)
        pre_gain = (1.0 - weight) * float(fitted[-1]) + weight * live_pn_g
    gain = _interpolate(time_s, schedule["raw_time_gain_table"])
    return pre_gain * gain


def _convert_sample(row: dict[str, Any], scenario: dict[str, Any], sound_speed: float) -> dict[str, Any]:
    x = float(row["x_m"])
    z = float(row["z_m"])
    altitude = float(scenario["launch_altitude_m"])
    target_x = float(row["target_x_m"])
    target_z = float(row["target_z_m"])
    speed = float(row["speed_kmh"]) / 3.6
    heading = float(row["heading_rad"])
    target0, target_v = planar_target_motion(scenario)
    del target0
    target_velocity = [float(target_v[0]), 0.0, float(target_v[1])]
    command_g = float(row["commanded_horizontal_acceleration_g"])
    actual_g = float(row["wind_normal_specific_force_g"])
    result = {
        "time_s": float(row["time_s"]),
        "position_m": [x, altitude, z],
        "velocity_mps": [speed * math.cos(heading), 0.0, speed * math.sin(heading)],
        "target_position_m": [target_x, altitude, target_z],
        "target_velocity_mps": target_velocity,
        "distance_to_target_m": float(row["distance_m"]),
        "commanded_acceleration_g": [0.0, 0.0, command_g],
        "actual_overload_g": actual_g,
        "trajectory_lateral_load_g": actual_g,
        "trajectory_pitch_normal_acceleration_g": command_g,
        "trajectory_yaw_normal_acceleration_g": 0.0,
        "mach": speed / sound_speed,
        "thrust_n": float(row["thrust_n"]),
        "drag_n": float(row["drag_n"]),
        "mass_kg": float(row["mass_kg"]),
        "loft_active": False,
    }
    for key in (
        "speed_kmh", "alpha_deg", "wind_normal_specific_force_g",
        "horizontal_specific_force_g", "horizontal_alpha_rad", "vertical_alpha_rad",
        "commanded_horizontal_alpha_rad", "commanded_horizontal_acceleration_g",
        "horizontal_fin_deflection_rad", "vertical_fin_deflection_rad", "body_heading_rad",
        "body_rate_rad_s", "tail_local_alpha_rad", "fin_control_force_raw_n",
        "fin_control_force_n", "fin_control_force_limit_n", "fin_control_force_saturated",
        "fin_control_moment_nm", "fin_passive_moment_nm", "equivalent_tail_normal_force_n",
        "fin_moment_nm", "rate_damping_moment_nm", "rate_damping_per_s",
        "total_aerodynamic_moment_nm", "inertia_kg_m2", "body_angular_acceleration_rad_s2",
        "pid_proportional_output", "pid_integral_output", "pid_derivative_output",
        "pid_raw_output", "capture_active", "raw_pn_request_g", "capture_heading_error_deg",
        "body_normal_force_n", "body_axial_force_n", "aerodynamic_center_moment_nm",
        "normalized_control", "force_rotation_residual_n", "moment_transfer_residual_nm",
    ):
        if key in row:
            result[key] = row[key]
    return result


def simulate_fixed_r77_1(
    missile_profile: dict[str, Any],
    scenario: Any,
    *,
    catalog_source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the frozen R-77-1 candidate and adapt telemetry to the GUI schema."""
    if missile_profile.get("missile_id", missile_profile.get("id")) != "su_r_77_1":
        raise ValueError("此实验模式只支持 R-77-1（su_r_77_1）。")
    fixed_scenario = validate_experiment_scenario(scenario)
    definition = _definition()
    config = definition["model_config"]
    parameters = AlphaServoParameters(**definition["parameters"])
    plant = _R77FixedMomentModel(config, parameters)
    integration = definition["integration"]
    atmo = StandardAtmosphere().sample(fixed_scenario["launch_altitude_m"])
    target0, target_v = planar_target_motion(fixed_scenario)
    result: AlphaServoResult = run_alpha_servo(
        config,
        parameters,
        fixed_scenario,
        max_time_s=float(config["performance"]["lifetime_s"]),
        plant_dt_s=float(integration["plant_dt_s"]),
        control_dt_s=float(integration["control_dt_s"]),
        moment_model=plant.sample,
        guidance_mode="pn_only",
        guidance_request_model=_scheduled_request,
    )
    samples = [_convert_sample(row, fixed_scenario, atmo.speed_of_sound_mps) for row in result.samples]
    terminal = samples[-1]
    speeds = [math.sqrt(sum(float(value) ** 2 for value in sample["velocity_mps"])) for sample in samples]
    event = {"fuse": "proximity_fuse", "closest_approach": "closest_approach"}.get(result.event, result.event)
    burn_time = sum(float(stage["duration_s"]) for stage in config["propulsion"]["stages"])
    summary = {
        "termination_event": event,
        "termination_detail": event,
        "flight_time_s": terminal["time_s"],
        "terminal_distance_m": terminal["distance_to_target_m"],
        "terminal_speed_kmh": speeds[-1] * 3.6,
        "terminal_altitude_m": terminal["position_m"][1],
        "maximum_speed_kmh": max(speeds) * 3.6,
        "maximum_altitude_m": max(sample["position_m"][1] for sample in samples),
        "maximum_commanded_g": max(abs(float(sample["commanded_horizontal_acceleration_g"])) for sample in samples),
        "maximum_actual_g": max(float(sample["actual_overload_g"]) for sample in samples),
        "maximum_trajectory_normal_g": max(float(sample["trajectory_lateral_load_g"]) for sample in samples),
        "minimum_distance_m": result.minimum_distance_m,
        "loft_enabled": False,
        "scenario_loft_enabled": False,
        "profile_lofting_enabled": False,
        "burnout_time_s": burn_time,
        "observation_mode": "not_applicable_planar_truth_target",
        "observation_provider": "not_applicable",
        "track_mode": "not_applicable",
        "seeker_state": "not_applicable",
        "seeker_display_state": "not_applicable",
        "maximum_track_error_m": 0.0,
        "first_radar_track_time_s": None,
        "first_radar_lost_time_s": None,
        "first_lock_loss_time_s": None,
        "first_reacquire_time_s": None,
        "last_radar_reject_reason": None,
        "last_observation_reject_reason": None,
    }
    markers = [{"time_s": burn_time, "label": "发动机燃尽", "kind": "burnout"}]
    markers.append({"time_s": terminal["time_s"], "label": "终止事件", "kind": "termination"})
    descriptor = public_definition()
    result_catalog_source = catalog_source or {}
    return {
        "schema_version": 1,
        "model_mode": descriptor["mode_id"],
        "missile": {
            "id": "su_r_77_1",
            "name": missile_profile.get("display_name", missile_profile.get("name", "R-77-1")),
            "status": "Experimental",
            "parameter_source": descriptor["parameter_source"],
            "catalog_parameter_source": {
                "version": result_catalog_source.get("version"),
                "commit": result_catalog_source.get("commit"),
            },
        },
        "scenario": fixed_scenario,
        "summary": summary,
        "markers": markers,
        "samples": samples,
        "model": {
            "label": descriptor["label"],
            "mode_id": descriptor["mode_id"],
            "parameter_source": descriptor["parameter_source"],
            "method": "free_forward_planar_fin_moment_candidate",
            "plant": "single_body_normal_force_at_distFromCmToStab_with_fitted_fin_control_couple_and_rate_damping",
            "guidance": "PN4_with_fitted_pre_gain_request_knots_and_cubic_return_to_live_PN",
            "integrator": "RK4 fixed step, full forward state integration",
            "plant_time_step_s": integration["plant_dt_s"],
            "control_time_step_s": integration["control_dt_s"],
            "source_artifacts": descriptor["fit_artifacts"],
            "scenario_geometry": "fixed horizontal plane at 6300 m; target course absolute_world",
        },
    }
