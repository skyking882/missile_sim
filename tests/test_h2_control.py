import copy
import math
from pathlib import Path

from aim120_model.config import load_model_config
from aim120_model.control import update_control_feedback
from aim120_model.dynamics import SimState, clamp_body_angle_of_attack


CONFIG = load_model_config(Path(__file__).parents[1] / "configs" / "aim120a_h2.yaml")


def test_physical_normal_feedback_and_zero_fin_authority_are_separate():
    state = SimState(
        (0.0, 3000.0, 0.0),
        (300.0, 0.0, 0.0),
        0.0, 0.0, 0.0, 0.0, 147.87,
        measured_pitch_normal_g=5.0,
        measured_yaw_normal_g=-2.0,
    )
    updates = update_control_feedback(
        state,
        (0.0, 0.0),
        CONFIG,
        0.02,
        enabled=True,
        authority_scale=0.0,
        feedback_measurement="physical_normal_g",
    )
    assert updates["pitch_pid_output"] < 0.0
    assert updates["yaw_pid_output"] > 0.0
    assert updates["pitch_requested_fin_command"] < 0.0
    assert updates["yaw_requested_fin_command"] > 0.0
    assert updates["pitch_fin_command"] == 0.0
    assert updates["yaw_fin_command"] == 0.0


def test_disabled_controller_clears_actuator_and_pid_telemetry():
    state = SimState((0.0, 3000.0, 0.0), (300.0, 0.0, 0.0), 0.0, 0.0, 0.0, 0.0, 147.87)
    updates = update_control_feedback(state, (10.0, -10.0), CONFIG, 0.02, enabled=False)
    assert updates["pitch_pid_output"] == 0.0
    assert updates["yaw_pid_output"] == 0.0
    assert updates["pitch_fin_command"] == 0.0
    assert updates["yaw_fin_command"] == 0.0


def test_high_demand_integral_unwinds_continuously_after_load_crosses_command():
    state = SimState(
        (0.0, 3000.0, 0.0),
        (300.0, 0.0, 0.0),
        0.0, 0.0, 0.0, 0.0, 147.87,
        yaw_pid_integral=10.0,
        measured_yaw_normal_g=25.0,
    )
    baseline_config = copy.deepcopy(CONFIG)
    baseline_config["control"]["high_demand_integral"] = {
        "enabled": True,
        "command_fraction": 0.4,
        "term_limit": 1.0,
        "unwind_multiplier": 1.0,
    }
    unwind_config = copy.deepcopy(CONFIG)
    unwind_config["control"]["high_demand_integral"] = {
        "enabled": True,
        "command_fraction": 0.4,
        "term_limit": 1.0,
        "unwind_multiplier": 4.0,
    }

    baseline = update_control_feedback(
        state,
        (0.0, 20.0),
        baseline_config,
        0.02,
        enabled=True,
        feedback_measurement="physical_normal_g",
    )
    unwound = update_control_feedback(
        state,
        (0.0, 20.0),
        unwind_config,
        0.02,
        enabled=True,
        feedback_measurement="physical_normal_g",
    )

    assert 0.0 < unwound["yaw_pid_integral"] < baseline["yaw_pid_integral"] < state.yaw_pid_integral


def test_body_aoa_guard_projects_attitude_onto_total_angle_limit():
    config = copy.deepcopy(CONFIG)
    config["control"].update({
        "limit_angle_of_attack_enabled": True,
        "maximum_body_angle_of_attack_deg": 30.0,
    })
    state = SimState(
        (0.0, 3000.0, 0.0),
        (300.0, 0.0, 0.0),
        0.6, 0.4, 1.0, 1.0, 147.87,
    )
    limited = clamp_body_angle_of_attack(state, config)
    assert abs((limited.pitch * limited.pitch + limited.yaw * limited.yaw) ** 0.5 - math.radians(30.0)) < 1e-12
    assert limited.pitch_rate == 0.0
    assert limited.yaw_rate == 0.0
