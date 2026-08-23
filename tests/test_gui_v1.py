from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

from aim120_model import H2Simulator, load_cases, load_model_config
from aim120_model.public_api import (
    SimulationInputError,
    UnsupportedPhysicsError,
    simulate,
    validate_scenario,
)
from missile_gui.library import public_profile, scan_library


ROOT = Path(__file__).resolve().parents[1]


def _scenario() -> dict[str, float | None]:
    return {
        "launch_speed_kmh": 1200.0,
        "launch_altitude_m": 6500.0,
        "launch_pitch_deg": 0.0,
        "launch_heading_deg": 0.0,
        "target_speed_kmh": 1200.0,
        "target_altitude_m": 6500.0,
        "initial_distance_m": 12000.0,
        "target_azimuth_deg": 10.0,
        "target_heading_deg": 0.0,
        "target_vertical_heading_deg": 0.0,
        "target_constant_turn_g": 0.0,
        "max_simulation_time_s": 0.2,
    }


def _off_axis_38_scenario() -> dict[str, float | None]:
    scenario = _scenario()
    scenario.update({
        "initial_distance_m": 15000.0,
        "target_azimuth_deg": 38.0,
        "max_simulation_time_s": None,
    })
    return scenario


def _r77_high_demand_scenario() -> dict[str, float | None]:
    scenario = _scenario()
    scenario.update({
        "loft_enabled": True,
        "target_speed_kmh": 900.0,
        "initial_distance_m": 8000.0,
        "target_azimuth_deg": 45.0,
        "target_heading_deg": -10.0,
        "max_simulation_time_s": None,
    })
    return scenario


def _heading_minus20_scenario() -> dict[str, float | None]:
    scenario = _r77_high_demand_scenario()
    scenario["loft_enabled"] = True
    scenario["target_heading_deg"] = -20.0
    return scenario


def test_gui_library_statuses_do_not_overclaim_validation() -> None:
    profiles, errors = scan_library(ROOT / "missiles", ROOT)
    assert errors == []
    indexed = {profile["id"]: profile for profile in profiles}
    assert indexed["aim-120a"]["status"] == "Experimental"
    assert indexed["pl-12"]["status"] == "Experimental"
    assert indexed["r-77"]["status"] == "Experimental"
    assert public_profile(indexed["pl-12"])["runnable"] is True
    assert public_profile(indexed["r-77"])["runnable"] is True
    assert not any(profile["status"] == "Validated" for profile in profiles)


def test_public_profile_never_exposes_loaded_config_or_source_path() -> None:
    profiles, _ = scan_library(ROOT / "missiles", ROOT)
    exposed = public_profile(profiles[0])
    assert "_model_config" not in exposed
    assert "_source_file" not in exposed
    assert exposed["runnable"] is True


def test_invalid_missile_json_is_reported_without_hiding_valid_entries() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        directory = Path(temp_dir)
        (directory / "bad.json").write_text('{"id":', encoding="utf-8")
        (directory / "good.json").write_text(json.dumps({
            "id": "placeholder",
            "name": "Placeholder",
            "country": "Test",
            "series": "Test",
            "status": "Unsupported physics",
            "physics": {"engine_type": "none", "control_type": "none"},
        }), encoding="utf-8")
        profiles, errors = scan_library(directory, ROOT)
        assert len(profiles) == 1
        assert len(errors) == 1
        assert "不是合法 JSON" in errors[0]


def test_scenario_validation_reports_missing_and_range_errors() -> None:
    missing = _scenario()
    del missing["target_speed_kmh"]
    try:
        validate_scenario(missing)
    except SimulationInputError as exc:
        assert "目标速度" in str(exc)
    else:
        raise AssertionError("missing field should fail")
    invalid = _scenario()
    invalid["launch_altitude_m"] = 40000.0
    try:
        validate_scenario(invalid)
    except SimulationInputError as exc:
        assert "超出范围" in str(exc)
    else:
        raise AssertionError("out-of-range field should fail")


def test_public_loft_switch_defaults_off_and_blocks_profile_loft() -> None:
    profiles, _ = scan_library(ROOT / "missiles", ROOT)
    aim120a = next(profile for profile in profiles if profile["id"] == "aim-120a")
    scenario = _scenario()
    scenario.pop("loft_enabled", None)

    normalized = validate_scenario(scenario)
    assert normalized["loft_enabled"] is False
    result = simulate(aim120a, scenario)

    assert result["summary"]["scenario_loft_enabled"] is False
    assert result["summary"]["profile_lofting_enabled"] is True
    assert result["summary"]["loft_enabled"] is False
    assert all(sample["loft_active"] is False for sample in result["samples"])


def test_public_loft_switch_explicit_on_runs_existing_gate() -> None:
    profiles, _ = scan_library(ROOT / "missiles", ROOT)
    aim120a = next(profile for profile in profiles if profile["id"] == "aim-120a")
    scenario = _scenario()
    scenario.update({
        "loft_enabled": True,
        "initial_distance_m": 60000.0,
        "max_simulation_time_s": 1.0,
    })

    result = simulate(aim120a, scenario)

    assert result["summary"]["scenario_loft_enabled"] is True
    assert result["summary"]["loft_enabled"] is True
    assert any(sample["loft_active"] for sample in result["samples"])


def test_unified_simulate_returns_summary_markers_and_target_trajectory_without_writes() -> None:
    profiles, _ = scan_library(ROOT / "missiles", ROOT)
    aim120a = next(profile for profile in profiles if profile["id"] == "aim-120a")
    before = {path.relative_to(ROOT) for path in (ROOT / "outputs").rglob("*") if path.is_file()}
    result = simulate(aim120a, _scenario())
    after = {path.relative_to(ROOT) for path in (ROOT / "outputs").rglob("*") if path.is_file()}
    assert before == after
    assert result["summary"]["termination_event"] == "lifetime"
    assert result["summary"]["termination_detail"] == "reached_scenario_time_limit"
    assert result["summary"]["burnout_time_s"] == 7.0
    assert result["markers"][-1]["kind"] == "termination"
    assert "target_position_m" in result["samples"][0]
    assert "target_velocity_mps" in result["samples"][0]


def test_unified_simulate_rejects_unsupported_engine_and_control_types() -> None:
    profile = {
        "id": "unsupported",
        "name": "Unsupported",
        "status": "Unsupported physics",
        "physics": {"engine_type": "unknown", "control_type": "unknown"},
    }
    try:
        simulate(profile, _scenario())
    except UnsupportedPhysicsError as exc:
        assert "发动机类型" in str(exc)
    else:
        raise AssertionError("unsupported physics should fail")


def test_gui_contains_interactive_colored_3d_scene() -> None:
    html = (ROOT / "src" / "missile_gui" / "static" / "index.html").read_text(encoding="utf-8")
    javascript = (ROOT / "src" / "missile_gui" / "static" / "app.js").read_text(encoding="utf-8")
    assert 'id="chart-3d"' in html
    assert 'id="observation-mode-select"' in html
    assert 'id="loft-enabled-toggle"' in html
    assert "默认关闭" in html
    assert "sensor_track" in html
    assert 'id="reset-3d"' in html
    assert "发动机切段" in html
    assert "燃尽" in html
    assert "终止事件" in html
    assert "class Trajectory3D" in javascript
    assert 'on("wheel"' in javascript
    assert 'mode:event.shiftKey||event.button===2?"pan":"rotate"' in javascript
    assert "showHover(event)" in javascript
    assert "COLORS_3D.stage" in javascript
    assert "off_axis_70" in html
    assert "off_axis_70" in javascript
    assert "COLORS_3D.burnout" in javascript
    assert "COLORS_3D.termination" in javascript
    assert "last_radar_reject_reason" in javascript
    assert "场景 Loft 开关" in javascript


def test_all_supported_profiles_are_runnable_through_universal_adapter() -> None:
    profiles, errors = scan_library(ROOT / "missiles", ROOT)
    assert errors == []
    public = [public_profile(profile) for profile in profiles]
    assert sum(item["runnable"] for item in public) == 116
    assert sum(not item["runnable"] for item in public) == 4
    pl12 = next(profile for profile in profiles if profile.get("missile_id") == "cn_pl12")
    result = simulate(pl12, _scenario())
    assert result["model"]["runtime_adapter"] == "profile_h2_fin_torque_aoa_v12"
    assert result["model"]["runtime_assumptions"]
    assert result["missile"]["status"] == "experimental"


def test_fin_torque_layer_preserves_aim120a_aim120b_kinematic_equivalence() -> None:
    profiles, errors = scan_library(ROOT / "missiles", ROOT)
    assert errors == []
    indexed = {profile.get("missile_id"): profile for profile in profiles}
    aim120a = indexed["us_aim_120a"]
    aim120b = indexed["us_aim_120b"]

    config_a = copy.deepcopy(aim120a["_model_config"])
    config_b = copy.deepcopy(aim120b["_model_config"])
    assert config_a.pop("model_label") == "us_aim_120a_profile_h2_fin_torque_aoa_v12"
    assert config_b.pop("model_label") == "us_aim_120b_profile_h2_fin_torque_aoa_v12"
    sensor_model_a = config_a["guidance"].pop("sensor_model")
    sensor_model_b = config_b["guidance"].pop("sensor_model")
    assert sensor_model_a["active_radar"] is True
    assert sensor_model_b.get("active_radar") is True or sensor_model_b.get("provider") == "profile_kinematic_v1"
    assert config_a == config_b

    cases = load_cases(ROOT / "configs" / "aim120a_v1_cases.json")
    simulator_a = H2Simulator(aim120a["_model_config"])
    simulator_b = H2Simulator(aim120b["_model_config"])
    for case in cases:
        result_a = simulator_a.run(case)
        result_b = simulator_b.run(case)
        assert result_a["event_type"] == "fuse"
        assert result_b["event_type"] == result_a["event_type"]
        assert result_b["terminal_time_s"] == result_a["terminal_time_s"]
        samples_a = [{key: value for key, value in sample.items() if key != "model_label"} for sample in result_a["samples"]]
        samples_b = [{key: value for key, value in sample.items() if key != "model_label"} for sample in result_b["samples"]]
        assert samples_b == samples_a


def test_universal_control_mapping_keeps_representative_profiles_controllable() -> None:
    profiles, errors = scan_library(ROOT / "missiles", ROOT)
    assert errors == []
    indexed = {profile.get("missile_id"): profile for profile in profiles}

    for missile_id in ("il_derby", "su_r_27er", "su_r_77", "us_aim7f_sparrow"):
        result = simulate(indexed[missile_id], _off_axis_38_scenario())
        assert result["summary"]["minimum_distance_m"] < 100.0


def test_fin_torque_mapping_preserves_raw_pid_and_profile_geometry() -> None:
    profiles, errors = scan_library(ROOT / "missiles", ROOT)
    assert errors == []
    indexed = {profile.get("missile_id"): profile for profile in profiles}
    for missile_id in ("us_aim_120a", "jp_aam4", "us_aim7f_sparrow", "su_r_27er", "su_r_77"):
        profile = indexed[missile_id]
        config = profile["_model_config"]
        for term in ("p", "i", "d", "integral_limit"):
            assert config["control"]["pid"][term] == profile["control"]["pid"][term]
        assert config["aerodynamics"]["distance_cm_to_stabilizer_m"] == profile["aerodynamics"]["fin_moment_arm_m"]
        assert config["control"]["plant_semantics"] == "fin_torque_body_aoa"
        assert "angular_response_scale" not in config["control"]
        assert "angular_damping" not in config["control"]
        assert "max_pitch_yaw_rate_deg_s" not in config["control"]
        assert config["guidance"]["maximum_angular_rate_deg_s"] == profile["guidance"]["maximum_angular_rate_deg_s"]


def test_r77_high_demand_case_tracks_without_permanent_integral_error() -> None:
    profiles, errors = scan_library(ROOT / "missiles", ROOT)
    assert errors == []
    r77 = next(profile for profile in profiles if profile.get("missile_id") == "su_r_77")
    result = simulate(r77, _r77_high_demand_scenario())

    # Rate-inner candidate still pulls on this lofted 45 deg case instead of
    # freezing on a wound-up I term.  Peak path G is no longer q-boosted.
    assert result["summary"]["minimum_distance_m"] < 40.0
    assert result["summary"]["maximum_trajectory_normal_g"] >= 12.0


def test_heading_minus20_case_keeps_aim120a_intercept_under_rate_inner() -> None:
    profiles, errors = scan_library(ROOT / "missiles", ROOT)
    assert errors == []
    indexed = {profile.get("missile_id"): profile for profile in profiles}
    results = {
        missile_id: simulate(indexed[missile_id], _heading_minus20_scenario())
        for missile_id in ("us_aim_120a", "cn_pl12", "su_r_77", "jp_aam4")
    }

    for missile_id in ("us_aim_120a", "cn_pl12", "su_r_77"):
        config = indexed[missile_id]["_model_config"]["control"]
        assert config["pid_output_semantics"] == "fin_angle_rad"
        assert "candidate_rate_inner_loop" not in config
    assert results["us_aim_120a"]["summary"]["minimum_distance_m"] < 200.0


def test_sensor_track_step_budget_reaches_requested_time_after_stage_splits() -> None:
    profiles, errors = scan_library(ROOT / "missiles", ROOT)
    assert errors == []
    r_darter = next(profile for profile in profiles if profile.get("missile_id") == "r_darter")
    scenario = _heading_minus20_scenario()
    scenario["observation_mode"] = "sensor_track"
    scenario["max_simulation_time_s"] = 7.0
    result = simulate(r_darter, scenario)

    assert result["summary"]["termination_event"] == "lifetime"
    assert result["summary"]["termination_detail"] == "reached_scenario_time_limit"
    assert result["summary"]["flight_time_s"] == 7.0


def _command_g(sample: dict) -> float:
    values = sample["commanded_acceleration_g"]
    return (float(values[0]) ** 2 + float(values[1]) ** 2) ** 0.5


def test_fuse_terminal_sample_does_not_rebuild_one_over_r_command() -> None:
    profiles, _ = scan_library(ROOT / "missiles", ROOT)
    aim120a = next(profile for profile in profiles if profile["missile_id"] == "us_aim_120a")
    result = simulate(aim120a, _off_axis_38_scenario())
    assert result["summary"]["termination_event"] == "proximity_fuse"
    last = result["samples"][-1]
    previous = result["samples"][-2]
    assert last["distance_to_target_m"] <= 12.000001
    assert abs(_command_g(last) - _command_g(previous)) < 1.0
    assert last["time_to_go_s"] < 1.0e6
