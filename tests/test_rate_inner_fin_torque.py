from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

from aim120_model.control import update_control_feedback
from aim120_model.dynamics import SimState
from aim120_model.h2_dynamics import forces_for_state_h2
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
    assert config["force_geometry_version"] == "fin_delta_g_no_arm_scale_v7"
    assert config["runtime_adapter"] == "profile_h2_fin_torque_aoa_v11"
    assert config["release_version"] == "profile-adapter-v15-path-g-finslataccel-arm-moment-only"
    assert not config["aerodynamics"].get("fin_arm_as_length_fraction")
    assert "path_g_scales_with_wing_area_multiplier" not in config["aerodynamics"]
    assert config["control"]["base_indicated_speed_mode"] == "none"
    assert abs(config["aerodynamics"]["horizontal_fin_aoa_limit_deg"] - math.degrees(0.268941)) < 1e-4


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


def test_path_g_does_not_scale_with_dynamic_pressure() -> None:
    config = _profile_config("cn_pl12")
    assert config["control"]["base_indicated_speed_mode"] == "none"
    limit = math.radians(config["aerodynamics"]["horizontal_fin_aoa_limit_deg"])
    slow = SimState(
        (0.0, 7500.0, 0.0), (200.0, 0.0, 0.0), 0.0, 0.0, 0.0, 0.0, 198.0,
        actual_pitch_fin_angle_rad=limit,
    )
    fast = SimState(
        (0.0, 7500.0, 0.0), (700.0, 0.0, 0.0), 0.0, 0.0, 0.0, 0.0, 198.0,
        actual_pitch_fin_angle_rad=limit,
    )
    propulsion = PiecewisePropulsion.from_config(config)
    slow_g = forces_for_state_h2(slow, 0.0, config, propulsion, powered=False)
    fast_g = forces_for_state_h2(fast, 0.0, config, propulsion, powered=False)
    expected = 41.4036
    assert abs(slow_g.trajectory_pitch_normal_acceleration_g - expected) < 0.05
    assert abs(fast_g.trajectory_pitch_normal_acceleration_g - expected) < 0.05


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
    assert abs(diagnostics.trajectory_pitch_normal_acceleration_g - fins_g) < 0.05
    assert fins_g > req_accel_max
    assert diagnostics.trajectory_pitch_normal_acceleration_g > req_accel_max


def test_path_g_follows_fins_lat_accel_while_arm_only_scales_moment() -> None:
    pl12 = _profile_config("cn_pl12")
    aam4 = _profile_config("jp_aam4")
    phoenix = _profile_config("us_aim_54a")
    pl12_limit = math.radians(pl12["aerodynamics"]["horizontal_fin_aoa_limit_deg"])
    aam4_limit = math.radians(aam4["aerodynamics"]["horizontal_fin_aoa_limit_deg"])
    phoenix_limit = math.radians(phoenix["aerodynamics"]["horizontal_fin_aoa_limit_deg"])
    fraction = 0.5
    pl12_state = SimState(
        (0.0, 3000.0, 0.0), (400.0, 0.0, 0.0), 0.0, 0.0, 0.0, 0.0, 198.0,
        actual_pitch_fin_angle_rad=fraction * pl12_limit,
    )
    aam4_state = SimState(
        (0.0, 3000.0, 0.0), (400.0, 0.0, 0.0), 0.0, 0.0, 0.0, 0.0, 222.0,
        actual_pitch_fin_angle_rad=fraction * aam4_limit,
    )
    phoenix_state = SimState(
        (0.0, 3000.0, 0.0), (400.0, 0.0, 0.0), 0.0, 0.0, 0.0, 0.0, 446.562,
        actual_pitch_fin_angle_rad=fraction * phoenix_limit,
    )
    pl12_diag = forces_for_state_h2(
        pl12_state, 0.0, pl12, PiecewisePropulsion.from_config(pl12), powered=False
    )
    aam4_diag = forces_for_state_h2(
        aam4_state, 0.0, aam4, PiecewisePropulsion.from_config(aam4), powered=False
    )
    phoenix_diag = forces_for_state_h2(
        phoenix_state, 0.0, phoenix, PiecewisePropulsion.from_config(phoenix), powered=False
    )
    assert abs(pl12_diag.trajectory_pitch_normal_acceleration_g - 41.4036 * fraction) < 0.05
    assert abs(aam4_diag.trajectory_pitch_normal_acceleration_g - 32.1114 * fraction) < 0.05
    assert abs(phoenix_diag.trajectory_pitch_normal_acceleration_g - 22.0 * fraction) < 0.05
    # Same fin fraction: PL-12 out-loads AAM-4 on path G, but the longer AAM-4
    # arm still produces more angular acceleration (pointing bandwidth).
    assert pl12_diag.trajectory_pitch_normal_acceleration_g > aam4_diag.trajectory_pitch_normal_acceleration_g
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
    expected_rate = 10.0 * config["atmosphere"]["gravity_mps2"] / 400.0 / 0.35
    assert abs(updates["commanded_pitch_rate_rad_s"] - expected_rate) < 1e-12
    assert updates["actual_pitch_fin_angle_rad"] > 0.0
    assert updates["actual_pitch_acceleration_g"] == 0.0


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
    assert _heading_at(r77["samples"], 1.0) > _heading_at(pl12["samples"], 1.0) + 1.0


def test_statshark_8km_40deg_straight_x_both_fuse_without_q_scale() -> None:
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
    # Straight 40 deg no longer starves terminal G once q/q_base is off the
    # plant, so both fuse.  The ranking split moves to the 90 deg case.
    assert r77["termination_event"] == "proximity_fuse"
    assert pl12["termination_event"] == "proximity_fuse"


def test_r77_and_pl12_both_fuse_90_deg_8km_with_arm_out_of_path_g() -> None:
    profiles, errors = scan_library(ROOT / "missiles", ROOT)
    assert errors == []
    indexed = {profile["missile_id"]: profile for profile in profiles}
    scenario = _off_axis_scenario(90.0)
    pl12 = simulate(indexed["cn_pl12"], scenario)
    aam4 = simulate(indexed["jp_aam4"], scenario)
    r77 = simulate(indexed["su_r_77"], scenario)
    # Steady-state G is now finsLatAccel, so this 90 deg geometry no longer
    # starves PL-12.  Arm ranking remains in pointing rate.
    assert r77["summary"]["termination_event"] == "proximity_fuse"
    assert pl12["summary"]["termination_event"] == "proximity_fuse"
    assert aam4["summary"]["termination_event"] == "proximity_fuse"
    assert r77["summary"]["flight_time_s"] < pl12["summary"]["flight_time_s"]
    assert _heading_at(aam4["samples"], 1.0) > _heading_at(pl12["samples"], 1.0) + 1.5
