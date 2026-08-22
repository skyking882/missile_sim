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
    assert config["aerodynamics"]["natural_lift_enabled"] is True
    assert config["aerodynamics"]["cn_alpha_per_rad"] == 2.0
    assert config["aerodynamics"]["cy_k"] == 2.0
    assert config["force_geometry_version"] == "fin_delta_g_loadfactormax_cap_v11_quat"
    assert config["runtime_adapter"] == "profile_h2_fin_torque_aoa_v11"
    assert config["release_version"] == "profile-adapter-v23-shared-baseind-1800"
    assert config["performance"]["load_factor_max_g"] == 35.0
    assert config["aerodynamics"]["normal_force_model"] == "body_cn_linear"
    assert config["aerodynamics"]["fin_translation_share"] == 1.0
    assert config["aerodynamics"]["path_g_scales_with_arm_times_length"] is True
    assert not config["aerodynamics"].get("fin_arm_as_length_fraction")
    assert "path_g_scales_with_wing_area_multiplier" not in config["aerodynamics"]
    assert config["control"]["base_indicated_speed_mode"] == "fin_authority_q"
    assert config["control"]["base_indicated_speed_ratio_max"] == 4.0
    assert config["aerodynamics"]["normal_force_cap_enabled"] is True
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


def test_fin_translation_share_scales_path_g_not_moment() -> None:
    full = _profile_config("cn_pl12")
    half = copy.deepcopy(full)
    half["aerodynamics"]["fin_translation_share"] = 0.5
    limit = math.radians(full["aerodynamics"]["horizontal_fin_aoa_limit_deg"])
    state = SimState(
        (0.0, 3000.0, 0.0), (400.0, 0.0, 0.0), 0.0, 0.0, 0.0, 0.0, 198.0,
        actual_pitch_fin_angle_rad=limit,
    )
    full_diag = forces_for_state_h2(
        state, 0.0, full, PiecewisePropulsion.from_config(full), powered=False
    )
    half_diag = forces_for_state_h2(
        state, 0.0, half, PiecewisePropulsion.from_config(half), powered=False
    )
    assert abs(
        half_diag.trajectory_pitch_normal_acceleration_g
        - 0.5 * full_diag.trajectory_pitch_normal_acceleration_g
    ) < 0.05
    assert abs(
        half_diag.pitch_angular_acceleration_rad_s2
        - full_diag.pitch_angular_acceleration_rad_s2
    ) < 1e-6


def test_legacy_body_lift_knob_can_disable_natural_lift() -> None:
    config = _config(
        legacy_body_lift={
            "enabled": False,
            "cn_alpha_per_rad": 2.0,
            "fin_translation_share": 1.0,
        }
    )
    assert config["aerodynamics"]["natural_lift_enabled"] is False


def test_body_cn_alpha_adds_path_g_at_aoa() -> None:
    config = _profile_config("cn_pl12")
    state = SimState(
        (0.0, 3000.0, 0.0), (400.0, 0.0, 0.0), 0.1, 0.0, 0.0, 0.0, 198.0,
    )
    diagnostics = forces_for_state_h2(
        state, 0.0, config, PiecewisePropulsion.from_config(config), powered=False
    )
    assert diagnostics.aero.pitch_alpha_rad > 0.0
    assert diagnostics.pitch_body_aoa_force_g > 0.0
    assert diagnostics.trajectory_pitch_normal_acceleration_g > 0.0


def test_path_g_scales_with_dynamic_pressure_ratio() -> None:
    config = _profile_config("cn_pl12")
    assert config["control"]["base_indicated_speed_mode"] == "fin_authority_q"
    limit = math.radians(config["aerodynamics"]["horizontal_fin_aoa_limit_deg"])
    fins_g = float(config["aerodynamics"]["fins_lateral_acceleration_g"])
    g0 = fins_g * float(config["aerodynamics"]["distance_cm_to_stabilizer_m"]) * float(
        config["geometry"]["length_m"]
    )
    v_base = float(config["control"]["base_indicated_speed_kmh"]) / 3.6
    propulsion = PiecewisePropulsion.from_config(config)

    def _full_fin(speed_mps: float, altitude_m: float) -> object:
        state = SimState(
            (0.0, altitude_m, 0.0),
            (speed_mps, 0.0, 0.0),
            0.0, 0.0, 0.0, 0.0, 198.0,
            actual_pitch_fin_angle_rad=limit,
        )
        return forces_for_state_h2(state, 0.0, config, propulsion, powered=False)

    at_base = _full_fin(v_base, 0.0)
    half_speed = _full_fin(0.5 * v_base, 0.0)
    twice_speed = _full_fin(2.0 * v_base, 0.0)
    over_cap = _full_fin(3.0 * v_base, 0.0)
    high = _full_fin(v_base, 12000.0)
    assert abs(at_base.trajectory_pitch_normal_acceleration_g - g0) < 0.05
    assert abs(at_base.aero.dynamic_pressure_pa / half_speed.aero.dynamic_pressure_pa - 4.0) < 1e-6
    assert abs(
        at_base.trajectory_pitch_normal_acceleration_g
        / half_speed.trajectory_pitch_normal_acceleration_g
        - 4.0
    ) < 0.05
    load_cap = float(config["performance"]["load_factor_max_g"])
    assert abs(twice_speed.trajectory_pitch_normal_acceleration_g - load_cap) < 0.2
    assert abs(over_cap.trajectory_pitch_normal_acceleration_g - load_cap) < 0.2
    assert twice_speed.lateral_load_g <= load_cap + 1e-6
    assert over_cap.lateral_load_g <= load_cap + 1e-6
    assert 4.0 * g0 > load_cap
    assert high.trajectory_pitch_normal_acceleration_g < at_base.trajectory_pitch_normal_acceleration_g * 0.5
    assert abs(
        at_base.trajectory_pitch_normal_acceleration_g / high.trajectory_pitch_normal_acceleration_g
        - at_base.aero.dynamic_pressure_pa / high.aero.dynamic_pressure_pa
    ) < 0.05


def test_plant_path_g_is_not_clamped_to_req_accel_max() -> None:
    config = _profile_config("cn_pl12")
    limit = math.radians(config["aerodynamics"]["horizontal_fin_aoa_limit_deg"])
    state = SimState(
        (0.0, 3000.0, 0.0), (400.0, 0.0, 0.0), 0.0, 0.0, 0.0, 0.0, 198.0,
        actual_pitch_fin_angle_rad=limit,
    )
    diagnostics = forces_for_state_h2(
        state, 0.0, config, PiecewisePropulsion.from_config(config), powered=False
    )
    req_accel_max = float(config["guidance"]["maximum_lateral_acceleration_g"])
    fins_g = float(config["aerodynamics"]["fins_lateral_acceleration_g"])
    scale = base_indicated_speed_schedule(
        diagnostics.aero.dynamic_pressure_pa, config
    ).fin_force_scale
    expected = (
        fins_g
        * scale
        * float(config["aerodynamics"]["distance_cm_to_stabilizer_m"])
        * float(config["geometry"]["length_m"])
    )
    assert abs(diagnostics.trajectory_pitch_normal_acceleration_g - expected) < 0.05
    assert abs(diagnostics.trajectory_pitch_normal_acceleration_g - req_accel_max) > 1.0
    assert diagnostics.lateral_load_g < float(config["performance"]["load_factor_max_g"])


def test_load_factor_max_caps_total_lateral_specific_force_not_moment() -> None:
    config = _profile_config("cn_pl12")
    cap = float(config["performance"]["load_factor_max_g"])
    assert cap == 38.0
    limit = math.radians(config["aerodynamics"]["horizontal_fin_aoa_limit_deg"])
    state = SimState(
        (0.0, 0.0, 0.0),
        (1000.0, 0.0, 0.0),
        0.2,
        0.0,
        0.0,
        0.0,
        198.0,
        actual_pitch_fin_angle_rad=limit,
    )
    diagnostics = forces_for_state_h2(
        state, 0.0, config, PiecewisePropulsion.from_config(config), powered=False
    )
    assert diagnostics.lateral_load_g <= cap + 1e-6
    assert diagnostics.trajectory_lateral_load_g <= cap + 0.05
    assert diagnostics.pitch_angular_acceleration_rad_s2 != 0.0


def _arm_length_path_g(config: dict, fins_g: float, fraction: float, q_scale: float) -> float:
    uncapped = (
        fins_g
        * fraction
        * q_scale
        * float(config["aerodynamics"]["distance_cm_to_stabilizer_m"])
        * float(config["geometry"]["length_m"])
    )
    cap = float(config["performance"]["load_factor_max_g"])
    return min(uncapped, cap)


def test_path_g_scales_with_arm_times_length_so_aam4_outpulls_derby() -> None:
    pl12 = _profile_config("cn_pl12")
    aam4 = _profile_config("jp_aam4")
    derby = _profile_config("il_derby")
    phoenix = _profile_config("us_aim_54a")
    fraction = 0.5

    def _diag(config: dict, mass: float) -> object:
        limit = math.radians(config["aerodynamics"]["horizontal_fin_aoa_limit_deg"])
        state = SimState(
            (0.0, 3000.0, 0.0), (400.0, 0.0, 0.0), 0.0, 0.0, 0.0, 0.0, mass,
            actual_pitch_fin_angle_rad=fraction * limit,
        )
        return forces_for_state_h2(
            state, 0.0, config, PiecewisePropulsion.from_config(config), powered=False
        )

    pl12_diag = _diag(pl12, 198.0)
    aam4_diag = _diag(aam4, 222.0)
    derby_diag = _diag(derby, 118.0)
    phoenix_diag = _diag(phoenix, 446.562)
    pl12_scale = base_indicated_speed_schedule(pl12_diag.aero.dynamic_pressure_pa, pl12).fin_force_scale
    aam4_scale = base_indicated_speed_schedule(aam4_diag.aero.dynamic_pressure_pa, aam4).fin_force_scale
    derby_scale = base_indicated_speed_schedule(derby_diag.aero.dynamic_pressure_pa, derby).fin_force_scale
    phoenix_scale = base_indicated_speed_schedule(phoenix_diag.aero.dynamic_pressure_pa, phoenix).fin_force_scale
    assert abs(pl12_diag.trajectory_pitch_normal_acceleration_g - _arm_length_path_g(pl12, 41.4036, fraction, pl12_scale)) < 0.05
    assert abs(aam4_diag.trajectory_pitch_normal_acceleration_g - _arm_length_path_g(aam4, 32.1114, fraction, aam4_scale)) < 0.05
    assert abs(derby_diag.trajectory_pitch_normal_acceleration_g - _arm_length_path_g(derby, 46.7469, fraction, derby_scale)) < 0.05
    assert abs(phoenix_diag.trajectory_pitch_normal_acceleration_g - _arm_length_path_g(phoenix, 22.0, fraction, phoenix_scale)) < 0.05
    assert phoenix["control"]["base_indicated_speed_mode"] == "fin_authority_q"
    assert aam4_diag.trajectory_pitch_normal_acceleration_g > pl12_diag.trajectory_pitch_normal_acceleration_g
    assert pl12_diag.trajectory_pitch_normal_acceleration_g > derby_diag.trajectory_pitch_normal_acceleration_g
    assert aam4_diag.pitch_angular_acceleration_rad_s2 > pl12_diag.pitch_angular_acceleration_rad_s2
    assert phoenix_diag.pitch_angular_acceleration_rad_s2 < pl12_diag.pitch_angular_acceleration_rad_s2 * 0.25


def test_runtime_flag_selects_rate_inner_without_changing_raw_pid() -> None:
    enabled = _config(acceleration_outer_rate_inner=True)
    disabled = _config(acceleration_outer_rate_inner=False)
    assert enabled["control"]["pid"] == disabled["control"]["pid"]
    assert enabled["control"]["plant_semantics"] == "fin_torque_body_aoa"
    assert enabled["control"]["pid_output_semantics"] == "body_rate_command_rad_s"
    assert disabled["control"]["pid_output_semantics"] == "fin_angle_rad"
    assert "candidate_rate_inner_loop" not in disabled["control"]
    assert enabled["control_model_version"] == "raw_pid_accel_outer_rate_inner_fin_torque_v13"


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
    config, assumptions = build_h2_candidate_config(profile, _defaults())
    loop = config["control"]["candidate_rate_inner_loop"]
    assert loop["path_rate_time_constant_s"] == 0.5
    assert loop["rate_error_for_full_fin_rad_s"] == 0.2
    assert all("path_rate_time_constant_s missing" not in item for item in assumptions)
    assert all("rate_error_for_full_fin_rad_s missing" not in item for item in assumptions)
    missing, missing_assumptions = build_h2_candidate_config(
        json.loads((ROOT / "missiles" / "us_aim_120a.json").read_text(encoding="utf-8")),
        _defaults(),
    )
    assert missing["control"]["candidate_rate_inner_loop"]["path_rate_time_constant_s"] == 0.35
    assert any("path_rate_time_constant_s missing" in item for item in missing_assumptions)


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
    close_rate = 10.0 * gravity / 400.0 / 0.35
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


def test_r77_and_aam4_fuse_80_deg_8km_while_derby_and_pl12_miss() -> None:
    profiles, errors = scan_library(ROOT / "missiles", ROOT)
    assert errors == []
    indexed = {profile["missile_id"]: profile for profile in profiles}
    scenario = _off_axis_scenario(80.0)
    pl12 = simulate(indexed["cn_pl12"], scenario)
    aam4 = simulate(indexed["jp_aam4"], scenario)
    derby = simulate(indexed["il_derby"], scenario)
    r77 = simulate(indexed["su_r_77"], scenario)
    # arm*length path G: AAM-4 out-pulls Derby; R-77 still the strongest.
    assert r77["summary"]["termination_event"] == "proximity_fuse"
    assert aam4["summary"]["termination_event"] == "proximity_fuse"
    assert derby["summary"]["termination_event"] != "proximity_fuse"
    assert pl12["summary"]["termination_event"] != "proximity_fuse"
    assert aam4["summary"]["minimum_distance_m"] < derby["summary"]["minimum_distance_m"]
    assert _heading_at(aam4["samples"], 1.0) > _heading_at(derby["samples"], 1.0)
    assert _heading_at(aam4["samples"], 1.0) > _heading_at(pl12["samples"], 1.0)
