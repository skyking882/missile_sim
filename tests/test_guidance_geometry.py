import copy
import math
from pathlib import Path

from aim120_model.config import load_model_config
from aim120_model.dynamics import SimState
from aim120_model.guidance import estimate_time_to_go_s, guidance_command, pn_acceleration
from aim120_model.math3d import dot, norm, normalize, sub
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


def test_beam_t_go_uses_relative_speed_floor_on_profile_mode():
    relative_position = (0.0, 0.0, 8000.0)
    relative_velocity = (-300.0, 0.0, -300.0)
    range_m = 8000.0
    closing = 300.0
    frozen = {"guidance": {"time_to_go_mode": "closing_speed"}}
    profile = {
        "guidance": {
            "time_to_go_mode": "closing_or_relative_speed",
            "time_to_go_relative_speed_weight": 1.0,
        }
    }
    frozen_tgo = estimate_time_to_go_s(range_m, closing, relative_velocity, frozen)
    profile_tgo = estimate_time_to_go_s(range_m, closing, relative_velocity, profile)
    relative_speed = (300.0 ** 2 + 300.0 ** 2) ** 0.5
    assert abs(frozen_tgo - range_m / closing) < 1e-12
    assert abs(profile_tgo - range_m / relative_speed) < 1e-12
    assert profile_tgo < frozen_tgo
    head_on_rel = (-600.0, 0.0, 0.0)
    head_on_tgo = estimate_time_to_go_s(12000.0, 600.0, head_on_rel, profile)
    assert abs(head_on_tgo - 20.0) < 1e-12


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


def _midcourse_config(**overrides):
    config = copy.deepcopy(CONFIG)
    settings = {
        "enabled": True,
        "turn_time_constant_s": 0.5,
        "lock_delay_s": 0.8,
        "blend_time_s": 0.5,
        "speed_floor_mps": 200.0,
    }
    settings.update(overrides)
    config["guidance"]["midcourse"] = settings
    return config


def test_midcourse_disabled_matches_pre_feature_behavior_bit_for_bit():
    # config["guidance"] entirely lacking a "midcourse" key is exactly the
    # pre-feature runtime (every frozen H1/H2 config).  enabled=False with
    # deliberately aggressive knobs (tiny tau, huge lock/blend, near-zero
    # speed floor) must land on the same command bit-for-bit, proving the
    # gate -- not the numbers -- is what keeps the term off.
    state = make_state()
    target = TargetState((20000.0, 3000.0, 6000.0), (-250.0, 0.0, 150.0))
    config_absent = copy.deepcopy(CONFIG)
    config_absent["guidance"].pop("midcourse", None)
    config_disabled = _midcourse_config(
        enabled=False,
        turn_time_constant_s=0.05,
        lock_delay_s=5.0,
        blend_time_s=5.0,
        speed_floor_mps=1.0,
    )
    absent = guidance_command(state, target, 0.5, config_absent, enabled=True)
    disabled = guidance_command(state, target, 0.5, config_disabled, enabled=True)
    assert disabled.commanded_acceleration_mps2 == absent.commanded_acceleration_mps2
    assert disabled.commanded_body_acceleration_g == absent.commanded_body_acceleration_g
    assert disabled.controller_specific_force_command_g == absent.controller_specific_force_command_g
    assert disabled.wind_normal_specific_force_command_g == absent.wind_normal_specific_force_command_g
    assert disabled.gravity_compensation_wind_normal_g == absent.gravity_compensation_wind_normal_g
    assert disabled.pn_acceleration_mps2 == absent.pn_acceleration_mps2
    assert disabled.loft_acceleration_mps2 == absent.loft_acceleration_mps2
    assert disabled.effective_gain == absent.effective_gain
    assert disabled.time_to_go_s == absent.time_to_go_s
    assert disabled.loft_active == absent.loft_active
    assert disabled.within_lock_range == absent.within_lock_range
    assert disabled.closing_speed_mps == absent.closing_speed_mps
    assert disabled.los_rate_vector_rad_s == absent.los_rate_vector_rad_s
    assert disabled.range_m == absent.range_m


def test_midcourse_lead_turn_dominates_at_launch_and_fades_after_blend_window():
    config = _midcourse_config()
    config["guidance"]["lofting_enabled"] = False
    state = make_state()
    # Stationary target 90 deg off the nose (+z); missile heads +x, so PN's
    # closing speed is exactly zero at t=0 and cannot contribute -- any
    # commanded acceleration here is the lead-turn term alone.
    target = TargetState((0.0, 3000.0, 20000.0), (0.0, 0.0, 0.0))
    at_launch = guidance_command(state, target, 0.0, config, enabled=True)
    assert at_launch.heading_error_rad > math.radians(80.0)
    assert at_launch.midcourse_weight == 1.0
    to_pip = normalize(sub(target.position, state.position))
    commanded_dir = normalize(at_launch.commanded_acceleration_mps2)
    assert dot(commanded_dir, to_pip) > 0.9
    max_accel = config["guidance"]["maximum_lateral_acceleration_g"] * config["atmosphere"]["gravity_mps2"]
    assert norm(at_launch.commanded_acceleration_mps2) > 0.9 * max_accel

    after_blend = guidance_command(state, target, 2.0, config, enabled=True)
    assert after_blend.midcourse_weight == 0.0

