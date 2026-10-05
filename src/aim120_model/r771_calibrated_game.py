"""Opt-in GUI adapter for the experimental War Thunder R-77-1 calibration.

The production adapter imports only the standard-library core. Replay files
and fitting scripts are not runtime inputs. The target is external and missile
state is freely integrated from scenario creation conditions.
"""
from __future__ import annotations

import copy
import math
from dataclasses import asdict

from .aerodynamics import body_axes_from_quaternion, pitch_yaw_from_quaternion, quaternion_from_pitch_yaw
from .dynamics import SimState
from .math3d import norm, sub
from .single_surface_game import RateAngleLoftGame, SurfaceCoefficients
from .target import TargetModel, TargetState


MODE_ID = "r771_calibrated_game_259_v1"
SOURCE_VERSION = "2.59.0.27"
SOURCE_COMMIT = "4ac255b829a27346b9300fccdc138362c2535fbe"
LABEL = "R-77-1 标定模型 · 实验"


def public_definition():
    return {"mode_id": MODE_ID, "label": LABEL, "status": "Experimental",
        "missile_id": "su_r_77_1", "parameter_source": {
            "version": SOURCE_VERSION, "commit": SOURCE_COMMIT},
        "observation_modes": ["ideal_truth"], "fixed_scenario": False,
        "description": "使用当前 R-77-1 参数，可自定义场景并切换 Loft；仅支持理想观测，尚未覆盖全部游戏实测。"}


class ScenarioTarget:
    """Pure absolute-time target motion with the existing course conventions.

    Constant-G horizontal turns integrate the constant-speed target definition
    analytically, so plant substeps and fuse queries never advance target memory.
    """
    def __init__(self, initial_conditions, gravity):
        self.model = TargetModel(initial_conditions, gravity)
        self.initial_state = self.model.initial_state

    def state_at(self, time_s):
        initial = self.initial_state
        vx, vy, vz = initial.velocity
        speed = math.hypot(vx, vz)
        if abs(self.model.turn_g) <= 1e-12 or speed <= 1e-9:
            return TargetState(tuple(x+time_s*v for x, v in zip(initial.position, initial.velocity)),
                initial.velocity)
        omega = self.model.turn_g*self.model.gravity_mps2/speed
        angle = omega*time_s
        c, s = math.cos(angle), math.sin(angle)
        position = (initial.position[0]+(vx*s+vz*(c-1.))/omega,
                    initial.position[1]+vy*time_s,
                    initial.position[2]+(vx*(1.-c)+vz*s)/omega)
        return TargetState(position, (c*vx-s*vz, vy, s*vx+c*vz))


def creation_state(config, case):
    initial = case["initial_conditions"]
    pitch, yaw = map(math.radians, (initial["launch_angle_deg"], initial["launch_yaw_deg"]))
    quaternion = quaternion_from_pitch_yaw(pitch, yaw)
    forward = body_axes_from_quaternion(quaternion).forward
    return SimState((0., initial["launch_altitude_m"], 0.),
        tuple(x*initial["start_speed_kmh"]/3.6 for x in forward), pitch, yaw, 0., 0.,
        config["geometry"]["initial_mass_kg"], orientation_quaternion=quaternion)


def simulate_calibrated_r771(profile, scenario):
    # Local import keeps the additive dispatcher free of a module cycle.
    from .public_api import (_case_from_scenario, _stage_markers, _summarize,
        _validate_profile, validate_scenario)

    if profile.get("missile_id", profile.get("id")) != "su_r_77_1":
        raise ValueError("R-77-1 标定模型只支持 R-77-1。")
    provenance = profile.get("provenance", {})
    if provenance.get("source_version") != SOURCE_VERSION or provenance.get("source_commit") != SOURCE_COMMIT:
        raise ValueError(f"R-77-1 标定模型只支持当前已标定参数 {SOURCE_VERSION} / {SOURCE_COMMIT[:12]}。")
    normalized = validate_scenario(scenario)
    if normalized["observation_mode"] != "ideal_truth":
        raise ValueError("R-77-1 标定模型仅支持 ideal_truth（理想观测），尚未接入传感器航迹。")
    if normalized["simulation_dt_s"] is not None:
        raise ValueError("R-77-1 标定模型固定使用 48 Hz 控制节拍，不能指定 simulation_dt_s。")
    if not normalized["datalink_enabled"] or normalized["datalink_disconnect_time_s"] is not None:
        raise ValueError("R-77-1 标定模型使用理想观测，不支持数据链断开场景。")
    config = copy.deepcopy(_validate_profile(profile))
    profile_loft = bool(config["guidance"].get("lofting_enabled", False))
    config["guidance"]["lofting_enabled"] = bool(normalized["loft_enabled"] and profile_loft)
    if config["guidance"]["lofting_enabled"]:
        for key in ("lofting_elevation_deg", "target_elevation_deg",
                    "angle_to_acceleration_multiplier", "loft_omega_max_deg_s"):
            value = config["guidance"].get(key)
            if value is None or not math.isfinite(float(value)):
                raise ValueError(f"当前 R-77-1 参数缺少有效 Loft 字段：{key}。")
    physical_lifetime = float(config["performance"]["lifetime_s"])
    requested = normalized["max_simulation_time_s"]
    end_time = min(physical_lifetime, requested) if requested is not None else physical_lifetime
    limited = requested is not None and requested < physical_lifetime
    case = _case_from_scenario(normalized)
    target = ScenarioTarget(case["initial_conditions"], config["atmosphere"]["gravity_mps2"])
    model = RateAngleLoftGame(config, force_basis="wind_linear", plant_substeps=4, loft_cap_unit="rad_s")
    raw = model.run(creation_state(config, case), target.state_at, end_time_s=end_time)
    if raw["summary"]["termination_event"] == "numerical_failure":
        raise ArithmeticError("标定模型检测到非有限数值。")
    samples = []
    for sample in raw["samples"]:
        t = sample["time_s"]
        target_state = target.state_at(t)
        request = model.request_records[t]
        atmosphere = model.atmosphere.sample(sample["position_m"][1])
        speed = norm(sub(sample["velocity_mps"], model.wind))
        _, yaw = pitch_yaw_from_quaternion(sample["orientation_quaternion"])
        feedback = (sample["feedback_pitch_g"], sample["feedback_yaw_g"])
        samples.append({**sample, "target_position_m": target_state.position,
            "target_velocity_mps": target_state.velocity,
            "distance_to_target_m": request["range_m"],
            "commanded_acceleration_g": request["request_g"],
            "actual_overload_g": math.hypot(*feedback),
            "yaw_rad": yaw, "alpha_deg": math.degrees(sample["angle_of_attack_rad"]),
            "pitch_alpha_deg": math.degrees(sample["pitch_alpha_rad"]),
            "speed_kmh": norm(sample["velocity_mps"])*3.6,
            "mach": speed/atmosphere.speed_of_sound_mps, "drag_n": request["wind_drag_n"],
            "loft_active": request["loft_active"], "loft_bias_g": request["loft_g"],
            "loft_weight": request.get("loft_weight", 0.),
            "loft_range_fraction": request.get("loft_range_fraction", 0.),
            "loft_relative_body_angle_rad": request.get("loft_relative_body_angle_rad", 0.),
            "los_rate_rad_s": request["los_rate_rad_s"],
            "observation_mode": "ideal_truth", "observation_provider": "ideal_truth",
            "track_mode": "ideal_truth", "seeker_state": "not_applicable",
            "seeker_display_state": "理想观测", "track_position_error_m": 0.})
    adapted = {"samples": samples, "event_type": raw["summary"]["termination_event"],
        "observation_mode": "ideal_truth", "observation_provider": "ideal_truth"}
    summary = _summarize(adapted, config, limited, normalized["loft_enabled"], profile_loft)
    summary["minimum_distance_m"] = raw["summary"]["minimum_range_m"]
    summary["maximum_alpha_deg"] = raw["summary"]["maximum_alpha_deg"]
    summary["controller_updates"] = raw["summary"]["controller_updates"]
    return {"schema_version": 1, "model_mode": MODE_ID,
        "missile": {"id": "su_r_77_1", "name": profile.get("display_name", profile.get("name")),
            "status": "Experimental", "parameter_source": {
                "version": SOURCE_VERSION, "commit": SOURCE_COMMIT}},
        "scenario": normalized, "summary": summary,
        "markers": _stage_markers(config)+[{"time_s": summary["flight_time_s"],
            "label": "终止事件", "kind": "termination"}],
        "samples": samples, "model": {**raw["model"], "label": LABEL,
            "status": "Experimental", "goal_complete": False,
            "aerodynamics": "calibrated_single_equivalent_surface",
            "geometry": "one_resultant_and_reference_couple_at_distFromCmToStab",
            "control": "raw_profile_PID_fixed_48Hz", "integrator": "RK4",
            "time_step_s": 1/48, "plant_step_s": 1/192,
            "observation_mode": "ideal_truth", "requested_observation_mode": "ideal_truth",
            "observation_provider": "ideal_truth", "radar_model": "not_applicable",
            "guidance_update_hz": 48, "raw_PID_preserved": True,
            "surface_coefficients": asdict(SurfaceCoefficients()),
            "raw_loft_fields": {key: config["guidance"].get(key) for key in
                ("lofting_elevation_deg", "target_elevation_deg",
                 "angle_to_acceleration_multiplier", "loft_omega_max_deg_s")},
            "parameter_source": {"version": SOURCE_VERSION, "commit": SOURCE_COMMIT},
            "runtime_assumptions": [
                "D: raw Loft endpoints are empirical amplitudes at the G request interface.",
                "D: raw omega field supplies a deg/s scale for the target LOS-rate transition.",
                "D: raw angle multiplier supplies transition steepness and body-relative angle feedback.",
                "D: one-way unloading memory and range-squared fading are candidate game mechanisms.",
                "Force and reference-couple coefficients are replay calibrations, not identified engine equations."],
            "limitations": ["Only current-version R-77-1 is supported by this opt-in calibration.",
                "Large off-axis, envelope and arbitrary maneuver cases are not validated.",
                "Native replay timing and tied observation states remain unresolved.",
                "Sensor-track and data-link loss behavior are not implemented by this mode."]}}
