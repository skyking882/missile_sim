"""Stable, side-effect-free entry point for local GUI and API consumers.

The browser never contains trajectory equations.  It submits a scenario here and
this module adapts it to the frozen H2 simulator contract.
"""

from __future__ import annotations

import copy
import math
from typing import Any

from .h2_simulator import H2Simulator
from .math3d import norm


SUPPORTED_ENGINE_TYPES = {"piecewise_constant_thrust"}
SUPPORTED_CONTROL_TYPES = {"pn_pid_first_order"}


class SimulationInputError(ValueError):
    """A user-correctable scenario or profile error."""


class UnsupportedPhysicsError(ValueError):
    """The selected missile asks for a model this runtime does not implement."""


_FIELDS: dict[str, tuple[float, float, str]] = {
    "launch_speed_kmh": (0.0, 5000.0, "发射速度"),
    "launch_altitude_m": (0.0, 30000.0, "发射高度"),
    "launch_pitch_deg": (-90.0, 90.0, "发射俯仰角"),
    "launch_heading_deg": (-180.0, 180.0, "发射航向角"),
    "target_speed_kmh": (0.0, 5000.0, "目标速度"),
    "target_altitude_m": (0.0, 30000.0, "目标高度"),
    "initial_distance_m": (1.0, 200000.0, "初始距离"),
    "target_azimuth_deg": (-180.0, 180.0, "目标方位角"),
    "target_heading_deg": (-360.0, 360.0, "目标航向角"),
    "target_vertical_heading_deg": (-90.0, 90.0, "目标垂直航向"),
    "target_constant_turn_g": (-15.0, 15.0, "目标恒定转弯G"),
}


def validate_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize the public scenario schema."""

    if not isinstance(scenario, dict):
        raise SimulationInputError("场景必须是 JSON 对象。")
    normalized: dict[str, Any] = {}
    for key, (minimum, maximum, label) in _FIELDS.items():
        if key not in scenario or scenario[key] in (None, ""):
            raise SimulationInputError(f"缺少必填项：{label}。")
        value = scenario[key]
        if isinstance(value, bool):
            raise SimulationInputError(f"{label}必须是数字。")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise SimulationInputError(f"{label}必须是数字。") from exc
        if not math.isfinite(number):
            raise SimulationInputError(f"{label}必须是有限数字。")
        if number < minimum or number > maximum:
            raise SimulationInputError(
                f"{label}超出范围：应为 {minimum:g} 到 {maximum:g}。"
            )
        normalized[key] = number

    optional = scenario.get("max_simulation_time_s")
    if optional in (None, ""):
        normalized["max_simulation_time_s"] = None
    else:
        try:
            maximum_time = float(optional)
        except (TypeError, ValueError) as exc:
            raise SimulationInputError("最大模拟时间必须是数字。") from exc
        if not math.isfinite(maximum_time) or not 0.1 <= maximum_time <= 300.0:
            raise SimulationInputError("最大模拟时间应为 0.1 到 300 秒。")
        normalized["max_simulation_time_s"] = maximum_time

    observation_mode = scenario.get("observation_mode", "ideal_truth")
    if observation_mode not in {"ideal_truth", "sensor_track"}:
        raise SimulationInputError("制导观测必须是 ideal_truth 或 sensor_track。")
    normalized["observation_mode"] = str(observation_mode)

    simulation_dt = scenario.get("simulation_dt_s")
    if simulation_dt in (None, ""):
        normalized["simulation_dt_s"] = None
    else:
        try:
            simulation_dt = float(simulation_dt)
        except (TypeError, ValueError) as exc:
            raise SimulationInputError("仿真时间步必须是数字。") from exc
        if not math.isfinite(simulation_dt) or not 1.0e-4 <= simulation_dt <= 1.0:
            raise SimulationInputError("仿真时间步应为 0.0001 到 1 秒。")
        normalized["simulation_dt_s"] = simulation_dt

    datalink_enabled = scenario.get("datalink_enabled", True)
    if not isinstance(datalink_enabled, bool):
        raise SimulationInputError("datalink_enabled 必须是布尔值。")
    normalized["datalink_enabled"] = datalink_enabled

    datalink_disconnect = scenario.get("datalink_disconnect_time_s")
    if datalink_disconnect in (None, ""):
        normalized["datalink_disconnect_time_s"] = None
    else:
        try:
            datalink_disconnect = float(datalink_disconnect)
        except (TypeError, ValueError) as exc:
            raise SimulationInputError("Datalink 断开时间必须是数字。") from exc
        if not math.isfinite(datalink_disconnect) or datalink_disconnect < 0.0 or datalink_disconnect > 300.0:
            raise SimulationInputError("Datalink 断开时间应为 0 到 300 秒。")
        normalized["datalink_disconnect_time_s"] = datalink_disconnect

    drift_direction = scenario.get("inertial_drift_direction", [0.0, 0.0, 1.0])
    if drift_direction is None:
        drift_direction = [0.0, 0.0, 1.0]
    if isinstance(drift_direction, (str, bytes)):
        raise SimulationInputError("惯性漂移方向必须是三个数字。")
    try:
        drift_values = [float(value) for value in drift_direction]
    except (TypeError, ValueError) as exc:
        raise SimulationInputError("惯性漂移方向必须是三个数字。") from exc
    if len(drift_values) != 3 or not all(math.isfinite(value) for value in drift_values):
        raise SimulationInputError("惯性漂移方向必须是三个有限数字。")
    drift_norm = math.sqrt(sum(value * value for value in drift_values))
    if drift_norm <= 1.0e-12:
        raise SimulationInputError("惯性漂移方向不能是零向量。")
    normalized["inertial_drift_direction"] = [value / drift_norm for value in drift_values]
    return normalized


def _validate_profile(profile: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(profile, dict):
        raise SimulationInputError("导弹配置不是有效对象。")
    if "model_family" in profile:
        unsupported = profile.get("_runtime_unsupported", [])
        if unsupported:
            raise UnsupportedPhysicsError("该导弹请求了不支持的模型类型：" + "、".join(unsupported))
        model_config = profile.get("_model_config")
        if not isinstance(model_config, dict):
            raise UnsupportedPhysicsError("该导弹没有可用的冻结或 profile candidate runtime 参数。")
        return model_config
    physics = profile.get("physics")
    if not isinstance(physics, dict):
        raise UnsupportedPhysicsError("该导弹没有可用的物理模型。")
    engine_type = physics.get("engine_type")
    control_type = physics.get("control_type")
    if engine_type not in SUPPORTED_ENGINE_TYPES:
        raise UnsupportedPhysicsError(f"不支持的发动机类型：{engine_type or '未声明'}。")
    if control_type not in SUPPORTED_CONTROL_TYPES:
        raise UnsupportedPhysicsError(f"不支持的控制类型：{control_type or '未声明'}。")
    model_config = profile.get("_model_config")
    if not isinstance(model_config, dict):
        raise UnsupportedPhysicsError("该导弹没有已加载的冻结模型参数。")
    return model_config


def _case_from_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": "gui_scenario",
        "model_variant": "full",
        "observation_mode": scenario["observation_mode"],
        "simulation_dt_s": scenario["simulation_dt_s"],
        "datalink_enabled": scenario["datalink_enabled"],
        "datalink_disconnect_time_s": scenario["datalink_disconnect_time_s"],
        "inertial_drift_direction": scenario["inertial_drift_direction"],
        "initial_conditions": {
            "start_speed_kmh": scenario["launch_speed_kmh"],
            "launch_altitude_m": scenario["launch_altitude_m"],
            "launch_angle_deg": scenario["launch_pitch_deg"],
            "launch_yaw_deg": scenario["launch_heading_deg"],
            "target_speed_kmh": scenario["target_speed_kmh"],
            "target_altitude_m": scenario["target_altitude_m"],
            "initial_target_distance_m": scenario["initial_distance_m"],
            "target_azimuth_deg": scenario["target_azimuth_deg"],
            "target_course_deg": scenario["target_heading_deg"],
            "target_course_reference": "statshark_relative_to_los",
            "target_constant_g_turn": scenario["target_constant_turn_g"],
            "target_vertical_course_deg": scenario["target_vertical_heading_deg"],
        },
    }


def _command_g(sample: dict[str, Any]) -> float:
    values = sample.get("commanded_acceleration_g", [0.0, 0.0])
    return math.hypot(float(values[0]), float(values[1]))


def _event_name(event_type: str) -> str:
    return {
        "impact": "hit",
        "fuse": "proximity_fuse",
        "ground": "ground",
        "lifetime": "lifetime",
        "max_distance": "max_range",
        "numerical_failure": "numerical_failure",
    }.get(event_type, event_type)


def _summarize(result: dict[str, Any], config: dict[str, Any], limited_by_scenario: bool) -> dict[str, Any]:
    samples = result["samples"]
    terminal = samples[-1]
    speeds = [norm(tuple(sample["velocity_mps"])) for sample in samples]
    stages = config["propulsion"]["stages"]
    burnout_time = sum(float(stage["duration_s"]) for stage in stages)
    event = _event_name(str(result["event_type"]))
    track_errors = [float(sample.get("track_position_error_m", 0.0)) for sample in samples]
    radar_track_times = [
        float(sample["time_s"])
        for sample in samples
        if sample.get("track_mode") == "radar_track"
    ]
    radar_lost_times = [
        float(sample["time_s"])
        for sample in samples
        if sample.get("seeker_state") == "lost"
    ]
    lock_loss_times = [
        float(sample["time_s"])
        for sample in samples
        if sample.get("track_mode") == "ins_search"
    ]
    reacquire_times: list[float] = []
    had_lock_loss = False
    for sample in samples:
        if sample.get("track_mode") == "ins_search":
            had_lock_loss = True
        elif had_lock_loss and sample.get("track_mode") == "radar_track":
            reacquire_times.append(float(sample["time_s"]))
    reject_reasons = [
        str(sample.get("radar_reject_reason"))
        for sample in samples
        if sample.get("radar_reject_reason")
    ]
    return {
        "termination_event": event,
        "termination_detail": (
            "reached_scenario_time_limit"
            if event == "lifetime" and limited_by_scenario
            else event
        ),
        "flight_time_s": float(terminal["time_s"]),
        "terminal_distance_m": float(terminal["distance_to_target_m"]),
        "terminal_speed_kmh": speeds[-1] * 3.6,
        "terminal_altitude_m": float(terminal["position_m"][1]),
        "maximum_speed_kmh": max(speeds) * 3.6,
        "maximum_altitude_m": max(float(sample["position_m"][1]) for sample in samples),
        "maximum_commanded_g": max(_command_g(sample) for sample in samples),
        "maximum_actual_g": max(float(sample.get("actual_overload_g", 0.0)) for sample in samples),
        "maximum_trajectory_normal_g": max(float(sample.get("trajectory_lateral_load_g", 0.0)) for sample in samples),
        "minimum_distance_m": min(float(sample["distance_to_target_m"]) for sample in samples),
        "loft_enabled": bool(config["guidance"].get("lofting_enabled", False)),
        "burnout_time_s": burnout_time,
        "observation_mode": result.get("observation_mode", "ideal_truth"),
        "observation_provider": result.get("observation_provider", "ideal_truth"),
        "track_mode": terminal.get("track_mode"),
        "seeker_state": terminal.get("seeker_state"),
        "seeker_display_state": terminal.get("seeker_display_state"),
        "maximum_track_error_m": max(track_errors) if track_errors else 0.0,
        "first_radar_track_time_s": min(radar_track_times) if radar_track_times else None,
        "first_radar_lost_time_s": min(radar_lost_times) if radar_lost_times else None,
        "first_lock_loss_time_s": min(lock_loss_times) if lock_loss_times else None,
        "first_reacquire_time_s": min(reacquire_times) if reacquire_times else None,
        "last_radar_reject_reason": reject_reasons[-1] if reject_reasons else None,
        "last_observation_reject_reason": next(
            (
                str(sample.get("observation_reject_reason"))
                for sample in reversed(samples)
                if sample.get("observation_reject_reason")
            ),
            None,
        ),
    }


def _stage_markers(config: dict[str, Any]) -> list[dict[str, Any]]:
    markers: list[dict[str, Any]] = []
    elapsed = 0.0
    stages = config["propulsion"]["stages"]
    for index, stage in enumerate(stages):
        elapsed += float(stage["duration_s"])
        label = "发动机燃尽" if index == len(stages) - 1 else f"{stage.get('name', index + 1)} 结束"
        markers.append({"time_s": elapsed, "label": label, "kind": "burnout" if index == len(stages) - 1 else "stage"})
    return markers


def simulate(missile_profile: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    """Run one missile and return a JSON-serializable, read-only result.

    This function performs no filesystem writes and is the only GUI calculation
    entry point.
    """

    base_config = _validate_profile(missile_profile)
    normalized = validate_scenario(scenario)
    config = copy.deepcopy(base_config)
    physical_lifetime = float(config["performance"]["lifetime_s"])
    requested_limit = normalized["max_simulation_time_s"]
    limited_by_scenario = requested_limit is not None and float(requested_limit) < physical_lifetime
    if requested_limit is not None:
        config["performance"]["lifetime_s"] = min(physical_lifetime, float(requested_limit))
    result = H2Simulator(config).run(_case_from_scenario(normalized))
    if result["event_type"] == "numerical_failure":
        raise ArithmeticError("模拟器检测到非有限数值或积分未能正常终止。")
    return {
        "schema_version": 1,
        "missile": {
            "id": missile_profile.get("missile_id", missile_profile.get("id")),
            "name": missile_profile.get("display_name", missile_profile.get("name")),
            "status": missile_profile.get("model_status", missile_profile.get("status")),
        },
        "scenario": normalized,
        "summary": _summarize(result, config, limited_by_scenario),
        "markers": _stage_markers(config) + [{
            "time_s": float(result["terminal_time_s"]),
            "label": "终止事件",
            "kind": "termination",
        }],
        "samples": result["samples"],
        "model": {
            "label": result["model_label"],
            "aerodynamics": result["aero_model_version"],
            "geometry": result["force_geometry_version"],
            "control": result["control_model_version"],
            "integrator": result["integrator"],
            "time_step_s": result["time_step_s"],
            "observation_mode": result["observation_mode"],
            "requested_observation_mode": result.get("requested_observation_mode", result["observation_mode"]),
            "guidance_update_hz": result["guidance_update_hz"],
            "observation_provider": result.get("observation_provider", "ideal_truth"),
            "lock_state_machine": "TRK->INS+SRC->TRK" if result["observation_mode"] == "sensor_track" else "not_applicable",
            "radar_model": (
                "deterministic_gate_candidate_v1"
                if result.get("observation_provider") == "radar_datalink_ins_v1"
                else "not_applicable"
                if result["observation_mode"] == "sensor_track"
                else "ideal_truth"
            ),
            "datalink_update": (
                "every_guidance_tick"
                if result.get("observation_provider") == "radar_datalink_ins_v1"
                else "not_applicable"
            ),
            "datalink_update_count": result.get("datalink_update_count", 0),
            "random_measurement_noise": False,
            "multipath_enabled": False,
            "sarh_model_enabled": False,
            "runtime_adapter": missile_profile.get("_runtime_adapter", "frozen_config"),
            "runtime_boundary": missile_profile.get("_runtime_boundary", missile_profile.get("_model_config", {}).get("reference", {}).get("source")),
            "runtime_assumptions": list(missile_profile.get("_runtime_assumptions", [])),
        },
    }


__all__ = [
    "SimulationInputError",
    "UnsupportedPhysicsError",
    "simulate",
    "validate_scenario",
]
