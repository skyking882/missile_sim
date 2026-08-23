from __future__ import annotations

import copy
import json
import math
from dataclasses import replace
from pathlib import Path

from aim120_model.aerodynamics import quaternion_from_pitch_yaw
from aim120_model.control import base_indicated_speed_schedule, update_control_feedback
from aim120_model.dynamics import SimState
from aim120_model.guidance import guidance_command
from aim120_model.h2_dynamics import _uses_quaternion_candidate, forces_for_state_h2
from aim120_model.h2_simulator import H2Simulator
from aim120_model.target import TargetState
from aim120_model.profile_adapter import (
    LEGACY_CRITICAL_DAMPED_PLANT,
    build_h2_candidate_config,
)
from aim120_model.propulsion import PiecewisePropulsion
from aim120_model.public_api import simulate
from missile_gui.library import scan_library


ROOT = Path(__file__).resolve().parents[1]


def _defaults(**overrides: object) -> dict:
    defaults = json.loads(
        (ROOT / "config" / "profile_h2_runtime_defaults.json").read_text(encoding="utf-8")
    )
    defaults["plant_model"] = LEGACY_CRITICAL_DAMPED_PLANT
    defaults.update(overrides)
    return defaults


def _config(**overrides: object) -> dict:
    profile = json.loads((ROOT / "missiles" / "us_aim_120a.json").read_text(encoding="utf-8"))
    config, _assumptions = build_h2_candidate_config(profile, _defaults(**overrides))
    return config


def _profile_config(missile_id: str, **overrides: object) -> dict:
    profile = json.loads((ROOT / "missiles" / f"{missile_id}.json").read_text(encoding="utf-8"))
    config, _assumptions = build_h2_candidate_config(profile, _defaults(**overrides))
    return config


def test_fin_torque_adapter_enables_body_cn_alpha() -> None:
    config = _config()
    assert config["force_geometry_version"] == "h2_spec_packed_lift_cm_np_v13"
    assert config["runtime_adapter"] == "profile_h2_fin_torque_aoa_v12"
    assert config["release_version"] == "profile-adapter-v27-h2-spec"
    assert config["drag_model"]["shape_mode"] == "interpolated_cx_1943_x1_10"
    assert config["drag_model"]["alpha_drag_area_basis_mode"] == "caliber_area"
    assert config["drag_model"]["alpha_drag_mach_shape"] is False
    assert config["performance"]["load_factor_max_g"] == 35.0
    assert config["aerodynamics"]["path_g_from_alpha"] is True
    assert config["aerodynamics"]["path_g_scales_with_arm_times_length"] is False
    assert config["control"]["candidate_rate_inner_loop"]["path_rate_time_constant_s"] == 0.35
    assert config["control"]["candidate_rate_inner_loop"]["path_close_integral_gain_per_s"] == 0.0
    assert config["control"]["candidate_rate_inner_loop"]["rate_error_for_full_fin_rad_s"] == 0.35
    assert not config["aerodynamics"].get("fin_arm_as_length_fraction")
    assert "path_g_scales_with_wing_area_multiplier" not in config["aerodynamics"]
    assert config["control"]["base_indicated_speed_mode"] == "fin_authority_q"
    assert config["control"]["base_indicated_speed_ratio_max"] == 4.0
    assert config["aerodynamics"]["cx_vs_fin_delta"] == 0.0
    assert abs(config["aerodynamics"]["horizontal_fin_aoa_limit_deg"] - math.degrees(0.268941)) < 1e-4
    assert _uses_quaternion_candidate(config) is True


def test_missing_base_ind_speed_uses_shared_1800_when_q_scaling() -> None:
    profile = json.loads((ROOT / "missiles" / "us_aim_120a.json").read_text(encoding="utf-8"))
    profile["control"]["base_indicated_speed_kmh"] = None
    config, assumptions = build_h2_candidate_config(profile, _defaults())
    assert config["control"]["base_indicated_speed_kmh"] == 1800.0
    assert config["control"]["base_indicated_speed_mode"] == "fin_authority_q"
    assert any("missing -> shared 1800" in item for item in assumptions)

    disabled, disabled_assumptions = build_h2_candidate_config(
        profile, _defaults(fin_force_q_scaling=False)
    )
    assert disabled["control"]["base_indicated_speed_kmh"] is None
    assert disabled["control"]["base_indicated_speed_mode"] == "none"
    assert any("baseline mode only" in item for item in disabled_assumptions)


def test_legacy_fin_torque_initial_state_carries_quaternion() -> None:
    config = _config()
    state = H2Simulator(config).initial_state(
        {
            "initial_conditions": {
                "launch_angle_deg": 0.0,
                "launch_yaw_deg": 10.0,
                "start_speed_kmh": 1200.0,
                "launch_altitude_m": 3000.0,
            }
        }
    )
    assert state.orientation_quaternion is not None
    expected = quaternion_from_pitch_yaw(state.pitch, state.yaw)
    assert max(abs(a - b) for a, b in zip(state.orientation_quaternion, expected)) < 1e-12


def test_body_cn_alpha_is_capped_at_max_cy() -> None:
    config = _profile_config("cn_pl12")
    state = SimState(
        (0.0, 3000.0, 0.0), (400.0, 0.0, 0.0), 0.8, 0.0, 0.0, 0.0, 198.0,
    )
    diagnostics = forces_for_state_h2(
        state, 0.0, config, PiecewisePropulsion.from_config(config), powered=False
    )
    max_cy = float(config["aerodynamics"]["max_cy_at_aoa"])
    uncapped = config["aerodynamics"]["cn_alpha_per_rad"] * diagnostics.aero.pitch_alpha_rad
    assert uncapped > max_cy
    assert abs(diagnostics.aero.lift_coefficient_pitch - max_cy) < 1e-9


def test_fin_delta_drag_defaults_off_and_adds_axial_drag_when_enabled() -> None:
    config = _profile_config("cn_pl12")
    limit = math.radians(config["aerodynamics"]["horizontal_fin_aoa_limit_deg"])
    state = SimState(
        (0.0, 3000.0, 0.0), (400.0, 0.0, 0.0), 0.0, 0.0, 0.0, 0.0, 198.0,
        actual_pitch_fin_angle_rad=limit,
    )
    propulsion = PiecewisePropulsion.from_config(config)
    off = forces_for_state_h2(state, 0.0, config, propulsion, powered=False)
    assert max(abs(value) for value in off.fin_drag_force_n) == 0.0
    enabled = copy.deepcopy(config)
    enabled["aerodynamics"]["cx_vs_fin_delta"] = 2.0
    on = forces_for_state_h2(state, 0.0, enabled, propulsion, powered=False)
    assert on.fin_drag_force_n[0] < 0.0
    assert abs(on.drag_force_n[0] - off.drag_force_n[0]) < 1e-9
    assert on.axial_specific_force_g < off.axial_specific_force_g
    straight = SimState(
        (0.0, 3000.0, 0.0), (400.0, 0.0, 0.0), 0.0, 0.0, 0.0, 0.0, 198.0,
    )
    straight_on = forces_for_state_h2(straight, 0.0, enabled, propulsion, powered=False)
    straight_off = forces_for_state_h2(straight, 0.0, config, propulsion, powered=False)
    assert max(abs(a - b) for a, b in zip(straight_on.total_force_n, straight_off.total_force_n)) < 1e-9


def test_zero_alpha_has_no_path_g_even_at_full_fin() -> None:
    config = _profile_config("cn_pl12")
    limit = math.radians(config["aerodynamics"]["horizontal_fin_aoa_limit_deg"])
    state = SimState(
        (0.0, 3000.0, 0.0), (400.0, 0.0, 0.0), 0.0, 0.0, 0.0, 0.0, 198.0,
        actual_pitch_fin_angle_rad=limit,
    )
    diagnostics = forces_for_state_h2(
        state, 0.0, config, PiecewisePropulsion.from_config(config), powered=False
    )
    assert abs(diagnostics.aero.pitch_alpha_rad) < 1e-9
    assert abs(diagnostics.trajectory_pitch_normal_acceleration_g) < 0.2
    assert diagnostics.pitch_angular_acceleration_rad_s2 != 0.0


def test_legacy_body_lift_knob_can_disable_natural_lift() -> None:
    config = _config(
        legacy_body_lift={
            "enabled": False,
            "cn_alpha_per_rad": 2.0,
            "fin_translation_share": 1.0,
        }
    )
    assert config["aerodynamics"]["natural_lift_enabled"] is False


def test_path_g_follows_alpha_over_fins_aoa() -> None:
    config = _profile_config("cn_pl12")
    limit = math.radians(config["aerodynamics"]["horizontal_fin_aoa_limit_deg"])
    alpha = 0.5 * limit
    state = SimState(
        (0.0, 3000.0, 0.0), (400.0, 0.0, 0.0), alpha, 0.0, 0.0, 0.0, 198.0,
    )
    diagnostics = forces_for_state_h2(
        state, 0.0, config, PiecewisePropulsion.from_config(config), powered=False
    )
    scale = base_indicated_speed_schedule(diagnostics.aero.dynamic_pressure_pa, config).fin_force_scale
    fins_g = float(config["aerodynamics"]["fins_lateral_acceleration_g"])
    expected = fins_g * scale * (diagnostics.aero.pitch_alpha_rad / limit)
    assert diagnostics.aero.pitch_alpha_rad > 0.0
    assert abs(diagnostics.pitch_body_aoa_force_g) < 0.05
    assert abs(diagnostics.trajectory_pitch_normal_acceleration_g - expected) < 0.15


def test_path_g_scales_with_dynamic_pressure_ratio() -> None:
    config = _profile_config("cn_pl12")
    assert config["control"]["base_indicated_speed_mode"] == "fin_authority_q"
    limit = math.radians(config["aerodynamics"]["horizontal_fin_aoa_limit_deg"])
    fins_g = float(config["aerodynamics"]["fins_lateral_acceleration_g"])
    fraction = 0.5
    g0 = fins_g * fraction
    v_base = float(config["control"]["base_indicated_speed_kmh"]) / 3.6
    propulsion = PiecewisePropulsion.from_config(config)

    def _at_alpha(speed_mps: float, altitude_m: float) -> object:
        state = SimState(
            (0.0, altitude_m, 0.0),
            (speed_mps, 0.0, 0.0),
            fraction * limit, 0.0, 0.0, 0.0, 198.0,
        )
        return forces_for_state_h2(state, 0.0, config, propulsion, powered=False)

    at_base = _at_alpha(v_base, 0.0)
    half_speed = _at_alpha(0.5 * v_base, 0.0)
    twice_speed = _at_alpha(2.0 * v_base, 0.0)
    over_cap = _at_alpha(3.0 * v_base, 0.0)
    high = _at_alpha(v_base, 12000.0)
    assert abs(at_base.trajectory_pitch_normal_acceleration_g - g0) < 0.2
    assert abs(at_base.aero.dynamic_pressure_pa / half_speed.aero.dynamic_pressure_pa - 4.0) < 1e-6
    assert abs(
        at_base.trajectory_pitch_normal_acceleration_g
        / half_speed.trajectory_pitch_normal_acceleration_g
        - 4.0
    ) < 0.1
    load_cap = float(config["performance"]["load_factor_max_g"])
    assert twice_speed.trajectory_lateral_load_g <= load_cap + 1e-6
    assert over_cap.trajectory_lateral_load_g <= load_cap + 1e-6
    assert twice_speed.trajectory_pitch_normal_acceleration_g <= load_cap + 0.3
    assert over_cap.trajectory_pitch_normal_acceleration_g <= load_cap + 0.3
    assert 4.0 * g0 > load_cap
    assert high.trajectory_pitch_normal_acceleration_g < at_base.trajectory_pitch_normal_acceleration_g * 0.5


def test_plant_path_g_is_not_clamped_to_req_accel_max() -> None:
    config = _profile_config("cn_pl12")
    limit = math.radians(config["aerodynamics"]["horizontal_fin_aoa_limit_deg"])
    state = SimState(
        (0.0, 0.0, 0.0), (400.0, 0.0, 0.0), 0.5 * limit, 0.0, 0.0, 0.0, 198.0,
    )
    diagnostics = forces_for_state_h2(
        state, 0.0, config, PiecewisePropulsion.from_config(config), powered=False
    )
    req_accel_max = float(config["guidance"]["maximum_lateral_acceleration_g"])
    fins_g = float(config["aerodynamics"]["fins_lateral_acceleration_g"])
    scale = base_indicated_speed_schedule(
        diagnostics.aero.dynamic_pressure_pa, config
    ).fin_force_scale
    expected = fins_g * scale * (diagnostics.aero.pitch_alpha_rad / limit)
    assert abs(diagnostics.trajectory_pitch_normal_acceleration_g - expected) < 0.2
    assert abs(diagnostics.trajectory_pitch_normal_acceleration_g - req_accel_max) > 1.0
    assert diagnostics.trajectory_lateral_load_g <= float(config["performance"]["load_factor_max_g"]) + 1e-6


def test_load_factor_max_caps_packed_lift_not_drag_or_moment() -> None:
    config = _profile_config("cn_pl12")
    cap = float(config["performance"]["load_factor_max_g"])
    assert cap == 38.0
    gravity = float(config["atmosphere"]["gravity_mps2"])
    mass = 198.0
    limit = math.radians(config["aerodynamics"]["horizontal_fin_aoa_limit_deg"])
    state = SimState(
        (0.0, 0.0, 0.0),
        (1000.0, 0.0, 0.0),
        0.2,
        0.0,
        0.0,
        0.0,
        mass,
        actual_pitch_fin_angle_rad=limit,
    )
    diagnostics = forces_for_state_h2(
        state, 0.0, config, PiecewisePropulsion.from_config(config), powered=False
    )
    packed_lift_g = math.sqrt(sum(value * value for value in diagnostics.control_force_n)) / (
        mass * gravity
    )
    assert packed_lift_g <= cap + 1e-9
    assert diagnostics.trajectory_lateral_load_g <= cap + 1e-6
    q = diagnostics.aero.dynamic_pressure_pa
    expected_drag = -q * diagnostics.aero.total_cda_m2
    assert abs(diagnostics.drag_force_n[0] - expected_drag) < 1.0
    assert diagnostics.pitch_angular_acceleration_rad_s2 != 0.0


def test_path_g_at_equal_alpha_fraction_follows_fins_lat_accel_not_arm() -> None:
    pl12 = _profile_config("cn_pl12")
    aam4 = _profile_config("jp_aam4")
    derby = _profile_config("il_derby")
    fraction = 0.5

    def _diag(config: dict, mass: float) -> object:
        limit = math.radians(config["aerodynamics"]["horizontal_fin_aoa_limit_deg"])
        state = SimState(
            (0.0, 3000.0, 0.0), (400.0, 0.0, 0.0), fraction * limit, 0.0, 0.0, 0.0, mass,
        )
        return forces_for_state_h2(
            state, 0.0, config, PiecewisePropulsion.from_config(config), powered=False
        )

    pl12_diag = _diag(pl12, 198.0)
    aam4_diag = _diag(aam4, 222.0)
    derby_diag = _diag(derby, 118.0)
    pl12_scale = base_indicated_speed_schedule(pl12_diag.aero.dynamic_pressure_pa, pl12).fin_force_scale
    aam4_scale = base_indicated_speed_schedule(aam4_diag.aero.dynamic_pressure_pa, aam4).fin_force_scale
    derby_scale = base_indicated_speed_schedule(derby_diag.aero.dynamic_pressure_pa, derby).fin_force_scale
    assert abs(pl12_diag.trajectory_pitch_normal_acceleration_g - 41.4036 * fraction * pl12_scale) < 0.3
    assert abs(aam4_diag.trajectory_pitch_normal_acceleration_g - 32.1114 * fraction * aam4_scale) < 0.3
    assert abs(derby_diag.trajectory_pitch_normal_acceleration_g - 46.7469 * fraction * derby_scale) < 0.3
    assert derby_diag.trajectory_pitch_normal_acceleration_g > pl12_diag.trajectory_pitch_normal_acceleration_g
    assert pl12_diag.trajectory_pitch_normal_acceleration_g > aam4_diag.trajectory_pitch_normal_acceleration_g
    assert abs(aam4_diag.pitch_angular_acceleration_rad_s2) > abs(pl12_diag.pitch_angular_acceleration_rad_s2)


def test_runtime_flag_selects_rate_inner_without_changing_raw_pid() -> None:
    enabled = _config(acceleration_outer_rate_inner=True)
    disabled = _config(acceleration_outer_rate_inner=False)
    assert enabled["control"]["pid"] == disabled["control"]["pid"]
    assert enabled["control"]["plant_semantics"] == "fin_torque_body_aoa"
    assert enabled["control"]["pid_output_semantics"] == "body_rate_command_rad_s"
    assert disabled["control"]["pid_output_semantics"] == "fin_angle_rad"
    assert "candidate_rate_inner_loop" not in disabled["control"]
    assert enabled["control_model_version"] == "spec_g_outer_rate_inner_v15"


def test_legacy_rate_inner_discards_pid_state() -> None:
    config = _config()
    state = SimState(
        (0.0, 3000.0, 0.0),
        (400.0, 0.0, 0.0),
        0.0, 0.0, 0.0, 0.0, 147.87,
        measured_pitch_normal_g=0.0,
        pitch_pid_integral=0.4,
        previous_pitch_error=1.0,
    )
    diagnostics = forces_for_state_h2(
        state, 0.0, config, PiecewisePropulsion.from_config(config), powered=False
    )
    updates = update_control_feedback(
        state, (10.0, 0.0), config, 0.02, enabled=True, plant_diagnostics=diagnostics
    )
    assert updates["pitch_pid_output"] == 0.0
    assert updates["pitch_pid_integral"] == 0.0
    assert updates["previous_pitch_error"] == 0.0
    assert updates["commanded_pitch_rate_rad_s"] != 0.0
    profiles, errors = scan_library(ROOT / "missiles", ROOT)
    assert errors == []
    indexed = {profile["missile_id"]: profile for profile in profiles}
    result = simulate(indexed["us_aim_120a"], _off_axis_scenario(10.0, time_s=1.0))
    assert result["samples"][0]["pid_output_applied"] is False


def test_profile_can_override_rate_loop_time_constants() -> None:
    profile = json.loads((ROOT / "missiles" / "us_aim_120a.json").read_text(encoding="utf-8"))
    profile["control"]["path_rate_time_constant_s"] = 0.5
    profile["control"]["rate_error_for_full_fin_rad_s"] = 0.2
    profile["control"]["path_close_integral_gain_per_s"] = 2.0
    profile["control"]["path_close_integral_limit_g_s"] = 8.0
    config, assumptions = build_h2_candidate_config(profile, _defaults())
    loop = config["control"]["candidate_rate_inner_loop"]
    assert loop["path_rate_time_constant_s"] == 0.5
    assert loop["rate_error_for_full_fin_rad_s"] == 0.2
    assert loop["path_close_integral_gain_per_s"] == 2.0
    assert loop["path_close_integral_limit_g_s"] == 8.0
    assert all("path_rate_time_constant_s missing" not in item for item in assumptions)
    assert all("rate_error_for_full_fin_rad_s missing" not in item for item in assumptions)
    assert all("path_close_integral_gain_per_s missing" not in item for item in assumptions)
    missing, missing_assumptions = build_h2_candidate_config(
        json.loads((ROOT / "missiles" / "us_aim_120a.json").read_text(encoding="utf-8")),
        _defaults(),
    )
    assert missing["control"]["candidate_rate_inner_loop"]["path_rate_time_constant_s"] == 0.35
    assert missing["control"]["candidate_rate_inner_loop"]["path_close_integral_gain_per_s"] == 0.0
    assert any("path_rate_time_constant_s missing" in item for item in missing_assumptions)
    assert any("path_close_integral_gain_per_s missing" in item for item in missing_assumptions)


def _pitch_close_step(command_g: float, ki: float, stored_integral: float = 0.0) -> dict[str, float]:
    config = _config()
    config["control"]["candidate_rate_inner_loop"]["path_close_integral_gain_per_s"] = ki
    state = SimState(
        (0.0, 3000.0, 0.0),
        (400.0, 0.0, 0.0),
        0.0, 0.0, 0.0, 0.0, 147.87,
        measured_pitch_normal_g=0.0,
        pitch_path_close_integral_g_s=stored_integral,
    )
    diagnostics = forces_for_state_h2(
        state, 0.0, config, PiecewisePropulsion.from_config(config), powered=False
    )
    return update_control_feedback(
        state, (command_g, 0.0), config, 0.02, enabled=True, plant_diagnostics=diagnostics
    )


def test_path_close_integral_adds_to_rate_after_first_step() -> None:
    gravity = 9.80665
    path_tau = _config()["control"]["candidate_rate_inner_loop"]["path_rate_time_constant_s"]
    hold_rate = (0.0 - 1.0) * gravity / 400.0
    close_p = 2.0 * gravity / 400.0 / path_tau
    first = _pitch_close_step(2.0, ki=2.0)
    assert abs(first["commanded_pitch_rate_rad_s"] - (hold_rate + close_p)) < 1e-12
    assert abs(first["pitch_path_close_integral_g_s"] - 2.0 * 0.02) < 1e-12
    assert abs(first["pitch_requested_fin_command"]) < 0.99
    second = _pitch_close_step(2.0, ki=2.0, stored_integral=first["pitch_path_close_integral_g_s"])
    close_i = 2.0 * first["pitch_path_close_integral_g_s"] * gravity / 400.0
    assert abs(second["commanded_pitch_rate_rad_s"] - (hold_rate + close_p + close_i)) < 1e-12
    assert second["pitch_pid_integral"] == 0.0


def test_path_close_integral_freezes_when_fins_saturated() -> None:
    first = _pitch_close_step(20.0, ki=2.0)
    assert abs(first["pitch_requested_fin_command"]) >= 0.99
    assert first["pitch_path_close_integral_g_s"] == 0.0


def test_fin_torque_rate_inner_uses_g_error_to_command_body_rate() -> None:
    config = _config()
    config["control"]["pid"].update({"p": 0.0, "i": 0.0, "d": 0.0})
    state = SimState(
        (0.0, 3000.0, 0.0),
        (400.0, 0.0, 0.0),
        0.0,
        0.0,
        0.0,
        0.0,
        147.87,
        measured_pitch_normal_g=0.0,
    )
    diagnostics = forces_for_state_h2(
        state,
        0.0,
        config,
        PiecewisePropulsion.from_config(config),
        powered=False,
    )
    updates = update_control_feedback(
        state,
        (10.0, 0.0),
        config,
        0.02,
        enabled=True,
        plant_diagnostics=diagnostics,
    )
    gravity = config["atmosphere"]["gravity_mps2"]
    hold_rate = (0.0 - 1.0) * gravity / 400.0
    path_tau = config["control"]["candidate_rate_inner_loop"]["path_rate_time_constant_s"]
    close_rate = 10.0 * gravity / 400.0 / path_tau
    expected_rate = hold_rate + close_rate
    assert abs(updates["commanded_pitch_rate_rad_s"] - expected_rate) < 1e-12
    assert updates["actual_pitch_fin_angle_rad"] > 0.0
    assert updates["actual_pitch_acceleration_g"] == 0.0


def test_level_flight_specific_force_command_is_one_g() -> None:
    config = _config()
    config["guidance"]["lofting_enabled"] = False
    state = SimState((0.0, 3000.0, 0.0), (400.0, 0.0, 0.0), 0.0, 0.0, 0.0, 0.0, 147.87)
    output = guidance_command(
        state,
        TargetState((20000.0, 3000.0, 0.0), (0.0, 0.0, 0.0)),
        0.0,
        config,
        enabled=True,
    )
    assert abs(output.commanded_body_acceleration_g[0]) < 1e-9
    assert abs(output.controller_specific_force_command_g[0] - 1.0) < 1e-9
    assert abs(output.controller_specific_force_command_g[1]) < 1e-9


def test_gravity_compensated_command_stays_within_req_accel_max() -> None:
    config = _config()
    config["guidance"]["lofting_enabled"] = False
    config["guidance"]["pn_gain"] = 40.0
    config["guidance"]["flight_time_gain_table"] = [[0.0, 1.0]]
    config["guidance"]["time_to_hit_gain_table"] = [[0.0, 1.0]]
    cap = float(config["guidance"]["maximum_lateral_acceleration_g"])
    state = SimState((0.0, 3000.0, 0.0), (400.0, 0.0, 0.0), 0.0, 0.0, 0.0, 0.0, 147.87)
    output = guidance_command(
        state,
        TargetState((200.0, 4000.0, 0.0), (0.0, 0.0, 0.0)),
        1.0,
        config,
        enabled=True,
    )
    body_sf = math.hypot(*output.controller_specific_force_command_g)
    wind_sf = math.hypot(*output.wind_normal_specific_force_command_g)
    kinematic = math.hypot(*output.commanded_body_acceleration_g)
    assert abs(kinematic - cap) < 1e-9
    assert abs(output.commanded_body_acceleration_g[0] - cap) < 1e-6
    assert body_sf <= cap + 1e-9
    assert wind_sf <= cap + 1e-9
    assert abs(body_sf - cap) < 1e-9


def test_level_glide_specific_force_hold_bounds_altitude_drop() -> None:
    profiles, errors = scan_library(ROOT / "missiles", ROOT)
    assert errors == []
    indexed = {profile["missile_id"]: profile for profile in profiles}
    result = simulate(
        indexed["cn_pl12"],
        {
            "launch_speed_kmh": 1200.0,
            "launch_altitude_m": 3000.0,
            "launch_pitch_deg": 0.0,
            "launch_heading_deg": 0.0,
            "target_speed_kmh": 900.0,
            "target_altitude_m": 3000.0,
            "initial_distance_m": 40000.0,
            "target_azimuth_deg": 0.0,
            "target_heading_deg": 0.0,
            "target_vertical_heading_deg": 0.0,
            "target_constant_turn_g": 0.0,
            "max_simulation_time_s": 8.0,
            "loft_enabled": False,
        },
    )
    start_alt = float(result["samples"][0]["position_m"][1])
    end_alt = float(result["samples"][-1]["position_m"][1])
    first = result["samples"][0]
    assert abs(float(first["controller_specific_force_command_g"][0]) - 1.0) < 0.15
    assert start_alt - end_alt < 80.0


def test_opposing_body_rate_reduces_fin_demand() -> None:
    config = _config()
    base = SimState(
        (0.0, 3000.0, 0.0),
        (400.0, 0.0, 0.0),
        0.0,
        0.0,
        0.0,
        0.0,
        147.87,
        measured_pitch_normal_g=0.0,
    )
    opposing = replace(base, pitch_rate=2.0)

    def _run(state: SimState) -> dict[str, float]:
        diagnostics = forces_for_state_h2(
            state,
            0.0,
            config,
            PiecewisePropulsion.from_config(config),
            powered=False,
        )
        return update_control_feedback(
            state,
            (5.0, 0.0),
            config,
            0.02,
            enabled=True,
            plant_diagnostics=diagnostics,
        )

    positive = _run(base)
    reduced = _run(opposing)
    assert positive["actual_pitch_fin_angle_rad"] > reduced["actual_pitch_fin_angle_rad"]


def _off_axis_scenario(azimuth_deg: float, time_s: float = 25.0) -> dict:
    return {
        "launch_speed_kmh": 1200.0,
        "launch_altitude_m": 6500.0,
        "launch_pitch_deg": 0.0,
        "launch_heading_deg": 0.0,
        "target_speed_kmh": 1200.0,
        "target_altitude_m": 6500.0,
        "initial_distance_m": 8000.0,
        "target_azimuth_deg": azimuth_deg,
        "target_heading_deg": 0.0,
        "target_vertical_heading_deg": 0.0,
        "target_constant_turn_g": 0.0,
        "max_simulation_time_s": time_s,
        "loft_enabled": False,
    }


def _heading_deg(sample: dict) -> float:
    velocity = sample["velocity_mps"]
    return math.degrees(math.atan2(float(velocity[2]), float(velocity[0])))


def _heading_at(samples: list[dict], time_s: float) -> float:
    origin = _heading_deg(samples[0])
    for sample in samples:
        if float(sample["time_s"]) >= time_s:
            return abs(_heading_deg(sample) - origin)
    return abs(_heading_deg(samples[-1]) - origin)


def test_rate_inner_lets_r77_out_turn_pl12_at_90_deg_off_axis() -> None:
    profiles, errors = scan_library(ROOT / "missiles", ROOT)
    assert errors == []
    indexed = {profile["missile_id"]: profile for profile in profiles}
    scenario = _off_axis_scenario(90.0, time_s=20.0)
    pl12 = simulate(indexed["cn_pl12"], scenario)
    r77 = simulate(indexed["su_r_77"], scenario)
    pl12_g = max(float(sample["trajectory_lateral_load_g"]) for sample in pl12["samples"])
    r77_g = max(float(sample["trajectory_lateral_load_g"]) for sample in r77["samples"])
    # Path G follows finsLatAccel, not arm.  R-77 still out-loads PL-12 and
    # the longer R-77 arm shows up as a faster heading catch at 1 s.
    assert r77_g > pl12_g
    assert _heading_at(r77["samples"], 1.0) > _heading_at(pl12["samples"], 1.0)


def test_statshark_8km_40deg_straight_x_both_fuse_with_q_scale() -> None:
    profiles, errors = scan_library(ROOT / "missiles", ROOT)
    assert errors == []
    indexed = {profile["missile_id"]: profile for profile in profiles}
    scenario = {
        "launch_speed_kmh": 1200.0,
        "launch_altitude_m": 7500.0,
        "launch_pitch_deg": 0.0,
        "launch_heading_deg": 0.0,
        "target_speed_kmh": 900.0,
        "target_altitude_m": 7500.0,
        "initial_distance_m": 8000.0,
        "target_azimuth_deg": 40.0,
        "target_heading_deg": -40.0,
        "target_course_reference": "statshark_relative_to_los",
        "target_vertical_heading_deg": 0.0,
        "target_constant_turn_g": 0.0,
        "max_simulation_time_s": 40.0,
        "loft_enabled": True,
    }
    pl12 = simulate(indexed["cn_pl12"], scenario)["summary"]
    r77 = simulate(indexed["su_r_77"], scenario)["summary"]
    # Straight 40 deg still fuses after restoring q/q_base on fin force.
    assert r77["termination_event"] == "proximity_fuse"
    assert pl12["termination_event"] == "proximity_fuse"


def test_off_axis_envelope_follows_packed_lift_and_r77_reaches_90_deg() -> None:
    profiles, errors = scan_library(ROOT / "missiles", ROOT)
    assert errors == []
    indexed = {profile["missile_id"]: profile for profile in profiles}
    scenario_80 = _off_axis_scenario(80.0)
    pl12_80 = simulate(indexed["cn_pl12"], scenario_80)
    aam4_80 = simulate(indexed["jp_aam4"], scenario_80)
    derby_80 = simulate(indexed["il_derby"], scenario_80)
    r77_80 = simulate(indexed["su_r_77"], scenario_80)
    # Packed lift follows finsLatAccel, so Derby out-turns AAM-4 on heading.
    # Spec τ_p=0.35 without a path-close integral is slower than the previous
    # calibrated D-loop; 80/90 deg 8 km now only fuses for R-77.
    assert r77_80["summary"]["termination_event"] == "proximity_fuse"
    assert aam4_80["summary"]["termination_event"] != "proximity_fuse"
    assert derby_80["summary"]["termination_event"] != "proximity_fuse"
    assert pl12_80["summary"]["termination_event"] != "proximity_fuse"
    assert _heading_at(derby_80["samples"], 1.0) > _heading_at(aam4_80["samples"], 1.0)
    assert _heading_at(r77_80["samples"], 1.0) > _heading_at(pl12_80["samples"], 1.0)

    scenario_90 = _off_axis_scenario(90.0)
    pl12_90 = simulate(indexed["cn_pl12"], scenario_90)
    aam4_90 = simulate(indexed["jp_aam4"], scenario_90)
    derby_90 = simulate(indexed["il_derby"], scenario_90)
    r77_90 = simulate(indexed["su_r_77"], scenario_90)
    assert r77_90["summary"]["termination_event"] == "proximity_fuse"
    assert aam4_90["summary"]["termination_event"] != "proximity_fuse"
    assert derby_90["summary"]["termination_event"] != "proximity_fuse"
    assert pl12_90["summary"]["termination_event"] != "proximity_fuse"


def test_pl12_spec_identities_at_q_base() -> None:
    config = _profile_config("cn_pl12")
    gravity = float(config["atmosphere"]["gravity_mps2"])
    mass = float(config["geometry"]["initial_mass_kg"])
    length = float(config["geometry"]["length_m"])
    arm = float(config["aerodynamics"]["distance_cm_to_stabilizer_m"])
    fins_g = float(config["aerodynamics"]["fins_lateral_acceleration_g"])
    alpha_max = math.radians(config["aerodynamics"]["horizontal_fin_aoa_limit_deg"])
    from aim120_model.drag_models import area_basis, effective_cda0

    wing_area = area_basis(config)
    caliber_area = area_basis(config, "caliber_area")
    base_kmh = float(config["control"]["base_indicated_speed_kmh"])
    q_base = 0.5 * 1.225000018 * (base_kmh / 3.6) ** 2
    cn_alpha = fins_g * mass * gravity / (q_base * wing_area * alpha_max)
    assert abs(cn_alpha * q_base * wing_area * alpha_max / (mass * gravity) - fins_g) < 1e-9
    assert abs(cn_alpha - 30.9) < 0.05
    assert abs(effective_cda0(3.10, config) - 0.0196) < 5e-4
    assert abs(effective_cda0(1.93, config) - 0.0255) < 5e-4
    assert abs(caliber_area * 1.4 - wing_area) < 1e-12

    v_base = base_kmh / 3.6
    state = SimState((0.0, 0.0, 0.0), (v_base, 0.0, 0.0), 0.0, 0.0, 0.0, 0.0, mass)
    diagnostics = forces_for_state_h2(
        state, 0.0, config, PiecewisePropulsion.from_config(config), powered=False
    )
    expected_wn = math.sqrt(
        fins_g * gravity * arm / ((length * length / 12.0) * alpha_max)
    )
    assert abs(diagnostics.pitch_natural_frequency_rad_s - expected_wn) < 0.15
    assert abs(expected_wn - 10.0) < 0.3
    assert abs(diagnostics.pitch_residual_rate_damping_per_s - 2.0 * expected_wn) < 0.3
    assert diagnostics.pitch_tail_rate_damping_per_s == 0.0


def test_weathervane_spring_uses_delta_minus_alpha_not_rate_incidence() -> None:
    config = _profile_config("cn_pl12")
    limit = math.radians(config["aerodynamics"]["horizontal_fin_aoa_limit_deg"])
    still = SimState(
        (0.0, 0.0, 0.0),
        (500.0, 0.0, 0.0),
        0.0,
        0.0,
        0.0,
        0.0,
        198.0,
        actual_pitch_fin_angle_rad=0.5 * limit,
    )
    rotating = replace(still, pitch_rate=20.0)
    propulsion = PiecewisePropulsion.from_config(config)
    still_diag = forces_for_state_h2(still, 0.0, config, propulsion, powered=False)
    rotating_diag = forces_for_state_h2(rotating, 0.0, config, propulsion, powered=False)
    assert abs(
        still_diag.pitch_fin_moment_equivalent_g - rotating_diag.pitch_fin_moment_equivalent_g
    ) < 1e-9
    assert rotating_diag.pitch_tail_rate_incidence_rad > 0.0
    assert abs(rotating_diag.pitch_tail_effective_incidence_rad - 0.5 * limit) < 1e-9
    assert rotating_diag.pitch_residual_rate_damping_per_s == 2.0 * rotating_diag.pitch_natural_frequency_rad_s
    spring = rotating_diag.pitch_fin_moment_equivalent_g * config["atmosphere"]["gravity_mps2"] * float(
        config["aerodynamics"]["distance_cm_to_stabilizer_m"]
    ) / (float(config["geometry"]["length_m"]) ** 2 / 12.0)
    damping = rotating_diag.pitch_residual_rate_damping_per_s * rotating.pitch_rate
    assert abs(rotating_diag.pitch_angular_acceleration_rad_s2 - (spring - damping)) < 1e-6
