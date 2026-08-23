import copy
from pathlib import Path

from aim120_model.config import load_model_config
from aim120_model.dynamics import SimState
from aim120_model.guidance import guidance_command, pn_acceleration
from aim120_model.math3d import norm
from aim120_model.target import TargetState


CONFIG = load_model_config(Path(__file__).parents[1] / "configs" / "aim120a_statshark.yaml")


def make_state():
    return SimState((0.0, 3000.0, 0.0), (300.0, 0.0, 0.0), 0.0, 0.0, 0.0, 0.0, 147.87)


def test_pn_turns_toward_positive_and_negative_lateral_targets():
    state = make_state()
    positive = TargetState((20000.0, 3000.0, 1000.0), (0.0, 0.0, 0.0))
    negative = TargetState((20000.0, 3000.0, -1000.0), (0.0, 0.0, 0.0))
    positive_output = guidance_command(state, positive, 0.0, CONFIG, enabled=True)
    negative_output = guidance_command(state, negative, 0.0, CONFIG, enabled=True)
    assert positive_output.commanded_body_acceleration_g[1] > 0.0
    assert negative_output.commanded_body_acceleration_g[1] < 0.0
    assert abs(positive_output.commanded_body_acceleration_g[1] + negative_output.commanded_body_acceleration_g[1]) < 1e-9


def test_forward_target_has_no_false_lateral_command():
    state = make_state()
    target = TargetState((20000.0, 3000.0, 0.0), (0.0, 0.0, 0.0))
    output = guidance_command(state, target, 0.0, CONFIG, enabled=True)
    assert abs(output.commanded_body_acceleration_g[1]) < 1e-12


def test_pn_does_not_inflate_one_over_r_on_numerical_leftovers():
    acceleration, closing, los = pn_acceleration(
        (1.0e-8, 0.0, 0.0),
        (-300.0, 0.0, 1.0e-8),
        (300.0, 0.0, 0.0),
        4.0,
    )
    assert acceleration == (0.0, 0.0, 0.0)
    assert closing == 0.0
    assert los == (0.0, 0.0, 0.0)
    output = guidance_command(
        make_state(),
        TargetState((1.0e-8, 3000.0, 0.0), (0.0, 0.0, 0.0)),
        0.0,
        CONFIG,
        True,
    )
    assert output.time_to_go_s == 0.0
    assert output.commanded_body_acceleration_g == (0.0, 0.0)


def test_pn_acceleration_is_transverse_and_bounded_by_guidance_layer():
    acceleration, closing, _los = pn_acceleration((20000.0, 0.0, 1000.0), (-300.0, 0.0, 0.0), (300.0, 0.0, 0.0), 4.0)
    assert closing > 0.0
    assert abs(acceleration[0]) < 1e-12
    output = guidance_command(make_state(), TargetState((20000.0, 3000.0, 1000.0), (0.0, 0.0, 0.0)), 0.0, CONFIG, True)
    assert norm(output.commanded_acceleration_mps2) <= CONFIG["guidance"]["maximum_lateral_acceleration_g"] * CONFIG["atmosphere"]["gravity_mps2"] + 1e-9


def test_loft_omega_max_caps_pitch_rate_command():
    config = copy.deepcopy(CONFIG)
    config["guidance"]["lofting_enabled"] = True
    config["guidance"]["lofting_elevation_deg"] = 20.0
    config["guidance"]["loft_exit_distance_m"] = 1000.0
    config["guidance"]["loft_exit_time_to_go_s"] = 1.0
    config["guidance"]["angle_to_acceleration_multiplier"] = 20.0
    config["control"] = {"plant_semantics": "fin_torque_body_aoa"}
    state = make_state()
    target = TargetState((30000.0, 3000.0, 0.0), (-300.0, 0.0, 0.0))
    uncapped = guidance_command(state, target, 0.0, config, True)
    config["guidance"]["loft_omega_max_deg_s"] = 3.0
    capped = guidance_command(state, target, 0.0, config, True)
    assert uncapped.loft_active is True
    assert capped.loft_active is True
    assert abs(capped.loft_acceleration_mps2[1]) < abs(uncapped.loft_acceleration_mps2[1])


def test_frozen_h2_controller_command_stays_kinematic_without_gravity_term():
    config = copy.deepcopy(CONFIG)
    config["guidance"]["lofting_enabled"] = False
    output = guidance_command(
        make_state(),
        TargetState((20000.0, 3000.0, 0.0), (0.0, 0.0, 0.0)),
        0.0,
        config,
        True,
    )
    assert abs(output.commanded_body_acceleration_g[0]) < 1e-9
    assert abs(output.controller_specific_force_command_g[0]) < 1e-9

