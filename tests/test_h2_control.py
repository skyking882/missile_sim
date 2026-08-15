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


def test_true_difference_integral_limit_clamps_the_i_term_not_the_raw_error_sum():
    state = SimState(
        (0.0, 3000.0, 0.0),
        (300.0, 0.0, 0.0),
        0.0, 0.0, 0.0, 0.0, 147.87,
        yaw_pid_integral=0.9,
        measured_yaw_normal_g=0.0,
    )
    config = copy.deepcopy(CONFIG)
    config["control"]["integral_limit_semantics"] = "term"
    config["control"]["pid"]["p"] = 0.0
    config["control"]["pid"]["d"] = 0.0
    updates = update_control_feedback(
        state,
        (0.0, 20.0),
        config,
        1.0,
        enabled=True,
        feedback_measurement="physical_normal_g",
    )

    assert updates["yaw_pid_integral"] == config["control"]["pid"]["integral_limit"]
    assert updates["yaw_pid_output"] == config["control"]["pid"]["integral_limit"]


def test_actuator_state_is_fin_angle_and_full_angle_maps_to_fins_lat_accel():
    state = SimState((0.0, 3000.0, 0.0), (300.0, 0.0, 0.0), 0.0, 0.0, 0.0, 0.0, 147.87)
    config = copy.deepcopy(CONFIG)
    config["control"]["pid"].update({"p": 1.0, "i": 0.0, "d": 0.0})
    config["control"]["integral_limit_semantics"] = "term"
    dt = config["control"]["actuator_time_constant_s"]
    updates = update_control_feedback(state, (100.0, 0.0), config, dt, enabled=True)
    maximum_angle = math.radians(config["aerodynamics"]["horizontal_fin_aoa_limit_deg"])

    assert abs(updates["actual_pitch_fin_angle_rad"] - 0.5 * maximum_angle) < 1e-12
    assert abs(updates["pitch_fin_command"] - 0.5) < 1e-12
    assert abs(
        updates["actual_pitch_acceleration_g"]
        - 0.5 * config["aerodynamics"]["fins_lateral_acceleration_g"]
    ) < 1e-12


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
