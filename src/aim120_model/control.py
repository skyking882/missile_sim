"""Minimal PID and first-order fin/actuator response."""

from __future__ import annotations

from typing import Any

from .math3d import clamp


def update_control_feedback(
    state: Any,
    command_body_acceleration_g: tuple[float, float],
    config: dict[str, Any],
    dt_s: float,
    enabled: bool,
    authority_scale: float = 1.0,
    feedback_measurement: str | None = None,
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
    measurement_mode = feedback_measurement or control_cfg.get("feedback_measurement", "actuator_state")
    updates: dict[str, float] = {}
    commands = (float(command_body_acceleration_g[0]), float(command_body_acceleration_g[1]))
    for axis, command, integral_name, previous_name, derivative_name, actual_name, fin_name in (
        (
            "pitch",
            commands[0],
            "pitch_pid_integral",
            "previous_pitch_error",
            "pitch_error_derivative",
            "actual_pitch_acceleration_g",
            "pitch_fin_command",
        ),
        (
            "yaw",
            commands[1],
            "yaw_pid_integral",
            "previous_yaw_error",
            "yaw_error_derivative",
            "actual_yaw_acceleration_g",
            "yaw_fin_command",
        ),
    ):
        actuator_state = float(getattr(state, actual_name))
        if measurement_mode == "physical_normal_g":
            # H2 separates the physical normal-load measurement used by the
            # controller from the actuator's own response state.  Otherwise
            # natural lift/drag would be fed back as if it were fin output,
            # and a zero-authority ablation would still apply control force.
            measured = float(getattr(state, f"measured_{axis}_normal_g", actuator_state))
        else:
            # Preserve the H1 actuator-state feedback path exactly.
            measured = actuator_state
        error = command - measured
        integral_limit = float(pid["integral_limit"])
        high_demand = control_cfg.get("high_demand_integral", {})
        if high_demand.get("enabled", False):
            maximum_command_g = max(
                float(config["guidance"]["maximum_lateral_acceleration_g"]),
                1e-9,
            )
            demand_fraction = abs(command) / maximum_command_g
            if demand_fraction >= float(high_demand.get("command_fraction", 1.0)):
                # accelControlIntgLim is treated as a limit on the I-term in
                # the high-demand branch.  Dividing by I converts that limit
                # to the stored integral-state units used by this local PID.
                integral_gain = abs(float(pid["i"]))
                if integral_gain > 1e-12:
                    integral_limit = max(
                        integral_limit,
                        float(high_demand.get("term_limit", 1.0)) / integral_gain,
                    )
        integral = clamp(float(getattr(state, integral_name)) + error * dt, -integral_limit, integral_limit)
        raw_derivative = (error - float(getattr(state, previous_name))) / dt
        derivative = float(getattr(state, derivative_name)) + alpha * (raw_derivative - float(getattr(state, derivative_name)))
        output = pid["p"] * error + pid["i"] * integral + pid["d"] * derivative
        requested_fin = clamp(output, -control_cfg["fin_command_limit"], control_cfg["fin_command_limit"])
        fin = requested_fin * authority
        desired_actual = fin * config["aerodynamics"]["fins_lateral_acceleration_g"]
        actuator_alpha = min(1.0, dt / (max(control_cfg["actuator_time_constant_s"], 1e-9) + dt))
        actual = actuator_state + actuator_alpha * (desired_actual - actuator_state)
        updates[integral_name] = integral
        updates[previous_name] = error
        updates[derivative_name] = derivative
        updates[actual_name] = actual
        updates[fin_name] = fin
        updates[f"{axis}_pid_output"] = output
        updates[f"{axis}_requested_fin_command"] = requested_fin
    return updates
