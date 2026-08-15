"""Minimal PID and first-order fin/actuator response."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .math3d import clamp


BASE_INDICATED_SPEED_MODES = {
    "none",
    "fin_authority_q",
    "matched_q",
    "pid_output_q",
}


@dataclass(frozen=True)
class BaseIndicatedSpeedSchedule:
    """Candidate speed schedule derived from dynamic pressure.

    ``dynamic_pressure_ratio`` is q/q_ref, where q_ref uses sea-level density
    and the profile's raw baseIndSpeed.  The three scale fields keep candidate
    placement explicit instead of silently rewriting the raw PID values.
    """

    mode: str
    base_indicated_speed_kmh: float | None
    indicated_speed_kmh: float
    dynamic_pressure_ratio: float
    pid_output_scale: float
    requested_fin_scale: float
    fin_force_scale: float


def base_indicated_speed_schedule(
    dynamic_pressure_pa: float,
    config: dict[str, Any],
    sea_level_density_kg_m3: float = 1.225000018,
) -> BaseIndicatedSpeedSchedule:
    """Return B0/B1/B2/B3 schedule factors without changing raw PID gains."""

    control = config["control"]
    mode = str(control.get("base_indicated_speed_mode", "none"))
    if mode not in BASE_INDICATED_SPEED_MODES:
        raise ValueError(f"unknown base_indicated_speed_mode: {mode}")
    q = max(float(dynamic_pressure_pa), 0.0)
    density = max(float(sea_level_density_kg_m3), 1e-12)
    indicated_speed_kmh = math.sqrt(2.0 * q / density) * 3.6
    raw_base = control.get("base_indicated_speed_kmh")
    base = None if raw_base is None else float(raw_base)
    if mode != "none" and (base is None or not math.isfinite(base) or base <= 0.0):
        raise ValueError("active baseIndSpeed candidate requires a positive per-profile base_indicated_speed_kmh")
    if base is None or not math.isfinite(base) or base <= 0.0:
        ratio = 1.0
    else:
        ratio = (indicated_speed_kmh / base) ** 2
    if mode == "fin_authority_q":
        pid_scale, fin_command_scale, fin_force_scale = 1.0, 1.0, ratio
    elif mode == "matched_q":
        pid_scale, fin_command_scale, fin_force_scale = 1.0, 1.0 / max(ratio, 1e-12), ratio
    elif mode == "pid_output_q":
        pid_scale, fin_command_scale, fin_force_scale = ratio, 1.0, 1.0
    else:
        pid_scale, fin_command_scale, fin_force_scale = 1.0, 1.0, 1.0
    return BaseIndicatedSpeedSchedule(
        mode=mode,
        base_indicated_speed_kmh=base,
        indicated_speed_kmh=indicated_speed_kmh,
        dynamic_pressure_ratio=ratio,
        pid_output_scale=pid_scale,
        requested_fin_scale=fin_command_scale,
        fin_force_scale=fin_force_scale,
    )


def update_control_feedback(
    state: Any,
    command_body_acceleration_g: tuple[float, float],
    config: dict[str, Any],
    dt_s: float,
    enabled: bool,
    authority_scale: float = 1.0,
    feedback_measurement: str | None = None,
    speed_schedule: BaseIndicatedSpeedSchedule | None = None,
) -> dict[str, float]:
    """Return feedback-state updates; the force and moment are applied in dynamics."""

    if not enabled:
        return {
            "pitch_pid_integral": 0.0,
            "yaw_pid_integral": 0.0,
            "previous_pitch_error": 0.0,
            "previous_yaw_error": 0.0,
            "pitch_error_derivative": 0.0,
            "yaw_error_derivative": 0.0,
            "actual_pitch_acceleration_g": 0.0,
            "actual_yaw_acceleration_g": 0.0,
            "actual_pitch_fin_angle_rad": 0.0,
            "actual_yaw_fin_angle_rad": 0.0,
            "pitch_fin_command": 0.0,
            "yaw_fin_command": 0.0,
            "pitch_pid_output": 0.0,
            "yaw_pid_output": 0.0,
            "pitch_requested_fin_command": 0.0,
            "yaw_requested_fin_command": 0.0,
        }

    control_cfg = config["control"]
    pid = control_cfg["pid"]
    dt = max(float(dt_s), 1e-9)
    filter_tau = max(float(control_cfg.get("derivative_filter_time_constant_s", 0.03)), 1e-9)
    alpha = min(1.0, dt / (filter_tau + dt))
    authority = clamp(float(authority_scale), 0.0, 1.0)
    schedule = speed_schedule or BaseIndicatedSpeedSchedule(
        mode="none",
        base_indicated_speed_kmh=None,
        indicated_speed_kmh=0.0,
        dynamic_pressure_ratio=1.0,
        pid_output_scale=1.0,
        requested_fin_scale=1.0,
        fin_force_scale=1.0,
    )
    measurement_mode = feedback_measurement or control_cfg.get("feedback_measurement", "actuator_state")
    updates: dict[str, float] = {}
    commands = (float(command_body_acceleration_g[0]), float(command_body_acceleration_g[1]))
    for axis, command, integral_name, previous_name, derivative_name, actual_name, fin_angle_name, fin_name, fin_limit_key in (
        (
            "pitch",
            commands[0],
            "pitch_pid_integral",
            "previous_pitch_error",
            "pitch_error_derivative",
            "actual_pitch_acceleration_g",
            "actual_pitch_fin_angle_rad",
            "pitch_fin_command",
            "horizontal_fin_aoa_limit_deg",
        ),
        (
            "yaw",
            commands[1],
            "yaw_pid_integral",
            "previous_yaw_error",
            "yaw_error_derivative",
            "actual_yaw_acceleration_g",
            "actual_yaw_fin_angle_rad",
            "yaw_fin_command",
            "vertical_fin_aoa_limit_deg",
        ),
    ):
        actuator_state = float(getattr(state, actual_name))
        if measurement_mode in {"physical_normal_g", "body_specific_force_g"}:
            # H2 separates the physical normal-load measurement used by the
            # controller from the actuator's own response state.  The true-
            # difference runtime stores body-axis specific force here, in the
            # same axes as the guidance command.
            measured = float(getattr(state, f"measured_{axis}_normal_g", actuator_state))
        else:
            # Preserve the H1 actuator-state feedback path exactly.
            measured = actuator_state
        error = command - measured
        stored_integral = float(getattr(state, integral_name))
        integral_limit = float(pid["integral_limit"])
        integral_semantics = str(control_cfg.get("integral_limit_semantics", "state"))
        if integral_semantics == "term":
            # accelControlIntgLim bounds the accumulated I contribution.  The
            # state therefore advances by Ki*error*dt and enters the PID sum
            # directly; the raw per-profile gain and limit remain unchanged.
            integral_delta = float(pid["i"]) * error * dt
        elif integral_semantics == "state":
            # Compatibility path for the frozen H1/H2 artifacts.
            integral_delta = error * dt
        else:
            raise ValueError(f"unknown integral_limit_semantics: {integral_semantics}")
        integral = clamp(stored_integral + integral_delta, -integral_limit, integral_limit)
        raw_derivative = (error - float(getattr(state, previous_name))) / dt
        derivative = float(getattr(state, derivative_name)) + alpha * (raw_derivative - float(getattr(state, derivative_name)))
        integral_output = integral if integral_semantics == "term" else float(pid["i"]) * integral
        output = float(pid["p"]) * error + integral_output + float(pid["d"]) * derivative
        scheduled_output = output * schedule.pid_output_scale
        maximum_fin_angle = math.radians(max(float(config["aerodynamics"][fin_limit_key]), 1e-9))
        output_semantics = str(control_cfg.get("pid_output_semantics", "normalized_fin_command"))
        if output_semantics == "fin_angle_rad":
            # Raw accelControl P/I/D output requests a physical fin angle.
            # finsAoa is the only actuator-angle clamp; it must not also act as
            # a gain on an invented [-1, 1] PID output.
            requested_fin_angle = clamp(
                scheduled_output * schedule.requested_fin_scale,
                -maximum_fin_angle,
                maximum_fin_angle,
            )
            requested_fin = requested_fin_angle / maximum_fin_angle
            desired_fin_angle = requested_fin_angle * authority
        elif output_semantics == "normalized_fin_command":
            # Frozen H1/H2 compatibility path.
            requested_fin = clamp(
                scheduled_output * schedule.requested_fin_scale,
                -control_cfg["fin_command_limit"],
                control_cfg["fin_command_limit"],
            )
            desired_fin_angle = requested_fin * authority * maximum_fin_angle
        else:
            raise ValueError(f"unknown pid_output_semantics: {output_semantics}")
        actuator_alpha = min(1.0, dt / (max(control_cfg["actuator_time_constant_s"], 1e-9) + dt))
        previous_fin_angle = float(getattr(state, fin_angle_name))
        actual_fin_angle = previous_fin_angle + actuator_alpha * (desired_fin_angle - previous_fin_angle)
        fin = clamp(actual_fin_angle / maximum_fin_angle, -1.0, 1.0)
        actual = (
            fin
            * float(config["aerodynamics"]["fins_lateral_acceleration_g"])
            * schedule.fin_force_scale
        )
        updates[integral_name] = integral
        updates[previous_name] = error
        updates[derivative_name] = derivative
        updates[actual_name] = actual
        updates[fin_angle_name] = actual_fin_angle
        updates[fin_name] = fin
        updates[f"{axis}_pid_output"] = output
        updates[f"{axis}_requested_fin_command"] = requested_fin
    return updates


__all__ = [
    "BASE_INDICATED_SPEED_MODES",
    "BaseIndicatedSpeedSchedule",
    "base_indicated_speed_schedule",
    "update_control_feedback",
]
