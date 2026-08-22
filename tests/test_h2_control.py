import copy
import math
from pathlib import Path

from aim120_model.aerodynamics import quaternion_from_pitch_yaw
from aim120_model.config import load_model_config
from aim120_model.control import base_indicated_speed_schedule, update_control_feedback
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


def test_base_indicated_speed_candidates_place_q_ratio_without_rewriting_pid():
    config = copy.deepcopy(CONFIG)
    config["control"]["base_indicated_speed_kmh"] = 1800.0
    rho0 = 1.225000018
    q_ref = 0.5 * rho0 * (1800.0 / 3.6) ** 2

    expected = {
        "none": (1.0, 1.0, 1.0),
        "fin_authority_q": (1.0, 1.0, 4.0),
        "matched_q": (1.0, 0.25, 4.0),
        "pid_output_q": (4.0, 1.0, 1.0),
    }
    raw_pid = copy.deepcopy(config["control"]["pid"])
    for mode, scales in expected.items():
        config["control"]["base_indicated_speed_mode"] = mode
        schedule = base_indicated_speed_schedule(4.0 * q_ref, config, rho0)
        assert abs(schedule.indicated_speed_kmh - 3600.0) < 1e-9
        assert abs(schedule.dynamic_pressure_ratio - 4.0) < 1e-12
        observed_scales = (
            schedule.pid_output_scale,
            schedule.requested_fin_scale,
            schedule.fin_force_scale,
        )
        assert all(abs(observed - wanted) < 1e-12 for observed, wanted in zip(observed_scales, scales))
        assert config["control"]["pid"] == raw_pid


def test_fin_authority_candidate_scales_force_while_matched_candidate_cancels_linear_response():
    state = SimState((0.0, 3000.0, 0.0), (300.0, 0.0, 0.0), 0.0, 0.0, 0.0, 0.0, 147.87)
    config = copy.deepcopy(CONFIG)
    config["control"]["pid"].update({"p": 0.01, "i": 0.0, "d": 0.0})
    config["control"]["integral_limit_semantics"] = "term"
    config["control"]["base_indicated_speed_kmh"] = 1800.0
    rho0 = 1.225000018
    q_ref = 0.5 * rho0 * (1800.0 / 3.6) ** 2
    dt = config["control"]["actuator_time_constant_s"]

    config["control"]["base_indicated_speed_mode"] = "fin_authority_q"
    b1 = update_control_feedback(
        state,
        (100.0, 0.0),
        config,
        dt,
        enabled=True,
        speed_schedule=base_indicated_speed_schedule(4.0 * q_ref, config, rho0),
    )
    config["control"]["base_indicated_speed_mode"] = "matched_q"
    b2 = update_control_feedback(
        state,
        (100.0, 0.0),
        config,
        dt,
        enabled=True,
        speed_schedule=base_indicated_speed_schedule(4.0 * q_ref, config, rho0),
    )

    fins_g = config["aerodynamics"]["fins_lateral_acceleration_g"]
    assert abs(b1["actual_pitch_acceleration_g"] - 2.0 * fins_g) < 1e-12
    assert abs(b2["actual_pitch_acceleration_g"] - 0.5 * fins_g) < 1e-12


def test_active_base_indicated_speed_candidate_rejects_missing_profile_value():
    config = copy.deepcopy(CONFIG)
    config["control"]["base_indicated_speed_mode"] = "fin_authority_q"
    config["control"]["base_indicated_speed_kmh"] = None
    try:
        base_indicated_speed_schedule(1000.0, config)
    except ValueError as exc:
        assert "positive per-profile" in str(exc)
    else:
        raise AssertionError("missing baseIndSpeed must not use an invented universal fallback")


def test_true_difference_pid_output_is_radians_clamped_only_by_profile_fin_aoa():
    state = SimState((0.0, 3000.0, 0.0), (300.0, 0.0, 0.0), 0.0, 0.0, 0.0, 0.0, 147.87)
    config = copy.deepcopy(CONFIG)
    config["control"]["pid_output_semantics"] = "fin_angle_rad"
    config["control"]["integral_limit_semantics"] = "term"
    config["control"]["pid"].update({"p": 0.01, "i": 0.0, "d": 0.0})
    config["aerodynamics"]["horizontal_fin_aoa_limit_deg"] = 20.0
    dt = config["control"]["actuator_time_constant_s"]

    updates = update_control_feedback(state, (100.0, 0.0), config, dt, enabled=True)
    maximum_angle = math.radians(20.0)

    assert updates["pitch_pid_output"] == 1.0
    assert updates["pitch_requested_fin_command"] == 1.0
    assert abs(updates["actual_pitch_fin_angle_rad"] - 0.5 * maximum_angle) < 1e-12
    assert abs(
        updates["actual_pitch_acceleration_g"]
        - 0.5 * config["aerodynamics"]["fins_lateral_acceleration_g"]
    ) < 1e-12


def test_true_difference_sub_limit_pid_angle_is_not_multiplied_by_fins_aoa():
    state = SimState((0.0, 3000.0, 0.0), (300.0, 0.0, 0.0), 0.0, 0.0, 0.0, 0.0, 147.87)
    config = copy.deepcopy(CONFIG)
    config["control"]["pid_output_semantics"] = "fin_angle_rad"
    config["control"]["integral_limit_semantics"] = "term"
    config["control"]["pid"].update({"p": 0.01, "i": 0.0, "d": 0.0})
    config["aerodynamics"]["horizontal_fin_aoa_limit_deg"] = 20.0
    dt = config["control"]["actuator_time_constant_s"]

    updates = update_control_feedback(state, (10.0, 0.0), config, dt, enabled=True)

    assert updates["pitch_pid_output"] == 0.1
    assert abs(updates["actual_pitch_fin_angle_rad"] - 0.05) < 1e-12
    assert abs(updates["pitch_requested_fin_command"] - 0.1 / math.radians(20.0)) < 1e-12


def test_pid_error_units_can_explicitly_use_mps2_without_changing_raw_gains():
    state = SimState(
        (0.0, 3000.0, 0.0),
        (300.0, 0.0, 0.0),
        0.0, 0.0, 0.0, 0.0, 147.87,
    )
    config = copy.deepcopy(CONFIG)
    config["control"]["pid_output_semantics"] = "fin_angle_rad"
    config["control"]["integral_limit_semantics"] = "term"
    config["control"]["pid"].update({"p": 0.01, "i": 0.0, "d": 0.0})
    config["control"]["pid_error_units"] = "mps2"

    updates = update_control_feedback(state, (1.0, 0.0), config, 0.02, enabled=True)

    assert abs(
        updates["pitch_pid_output"]
        - 0.01 * config["atmosphere"]["gravity_mps2"]
    ) < 1e-12


def test_pid_error_scale_supports_a_shared_diagnostic_between_g_and_mps2():
    state = SimState(
        (0.0, 3000.0, 0.0),
        (300.0, 0.0, 0.0),
        0.0, 0.0, 0.0, 0.0, 147.87,
    )
    config = copy.deepcopy(CONFIG)
    config["control"]["pid_output_semantics"] = "fin_angle_rad"
    config["control"]["integral_limit_semantics"] = "term"
    config["control"]["pid"].update({"p": 0.01, "i": 0.0, "d": 0.0})
    config["control"]["pid_error_units"] = "g"
    config["control"]["pid_error_scale"] = 3.0

    updates = update_control_feedback(state, (1.0, 0.0), config, 0.02, enabled=True)

    assert abs(updates["pitch_pid_output"] - 0.03) < 1e-12


def test_fin_torque_plant_actuator_does_not_generate_g_directly():
    state = SimState((0.0, 3000.0, 0.0), (300.0, 0.0, 0.0), 0.0, 0.0, 0.0, 0.0, 147.87)
    config = copy.deepcopy(CONFIG)
    config["control"]["plant_semantics"] = "fin_torque_body_aoa"
    config["control"]["pid_output_semantics"] = "fin_angle_rad"
    config["control"]["pid"].update({"p": 1.0, "i": 0.0, "d": 0.0})
    dt = config["control"]["actuator_time_constant_s"]

    updates = update_control_feedback(state, (100.0, 0.0), config, dt, enabled=True)

    assert updates["actual_pitch_fin_angle_rad"] > 0.0
    assert updates["actual_pitch_acceleration_g"] == 0.0
    assert updates["actual_yaw_acceleration_g"] == 0.0


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
    assert limited.orientation_quaternion is None

    quat_state = SimState(
        (0.0, 3000.0, 0.0),
        (300.0, 0.0, 0.0),
        0.6, 0.4, 1.0, 1.0, 147.87,
        orientation_quaternion=quaternion_from_pitch_yaw(0.6, 0.4),
    )
    limited_quat = clamp_body_angle_of_attack(quat_state, config)
    assert limited_quat.orientation_quaternion is not None
    expected = quaternion_from_pitch_yaw(limited_quat.pitch, limited_quat.yaw)
    assert max(abs(a - b) for a, b in zip(limited_quat.orientation_quaternion, expected)) < 1e-12
