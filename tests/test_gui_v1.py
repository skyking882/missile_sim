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
        "target_speed_kmh": 900.0,
        "initial_distance_m": 8000.0,
        "target_azimuth_deg": 45.0,
        "target_heading_deg": -10.0,
        "max_simulation_time_s": None,
    })
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
    assert "sensor_track" in html
    assert 'id="reset-3d"' in html
    assert "发动机切段" in html
    assert "燃尽" in html
    assert "终止事件" in html
    assert "class Trajectory3D" in javascript
    assert 'addEventListener("wheel"' in javascript
    assert 'mode:event.shiftKey||event.button===2?"pan":"rotate"' in javascript
    assert "showHover(event)" in javascript
    assert "COLORS_3D.stage" in javascript
    assert "COLORS_3D.burnout" in javascript
    assert "COLORS_3D.termination" in javascript
    assert "last_radar_reject_reason" in javascript


def test_all_supported_profiles_are_runnable_through_universal_adapter() -> None:
    profiles, errors = scan_library(ROOT / "missiles", ROOT)
    assert errors == []
    public = [public_profile(profile) for profile in profiles]
    assert sum(item["runnable"] for item in public) == 116
    assert sum(not item["runnable"] for item in public) == 4
    pl12 = next(profile for profile in profiles if profile.get("missile_id") == "cn_pl12")
    result = simulate(pl12, _scenario())
    assert result["model"]["runtime_adapter"] == "profile_h2_universal_v2"
    assert result["model"]["runtime_assumptions"]
    assert result["missile"]["status"] == "experimental"


def test_universal_layer_preserves_aim120a_freeze_and_aim120b_equivalence() -> None:
    profiles, errors = scan_library(ROOT / "missiles", ROOT)
    assert errors == []
    indexed = {profile.get("missile_id"): profile for profile in profiles}
    aim120a = indexed["us_aim_120a"]
    aim120b = indexed["us_aim_120b"]

    config_a = copy.deepcopy(aim120a["_model_config"])
    config_b = copy.deepcopy(aim120b["_model_config"])
    assert config_a.pop("model_label") == "us_aim_120a_profile_h2_universal_v2"
    assert config_b.pop("model_label") == "us_aim_120b_profile_h2_universal_v2"
    sensor_model_a = config_a["guidance"].pop("sensor_model")
    sensor_model_b = config_b["guidance"].pop("sensor_model")
    assert sensor_model_a["active_radar"] is True
    assert sensor_model_b["provider"] == "profile_kinematic_v1"
    assert config_a == config_b

    frozen = load_model_config(ROOT / "configs" / "aim120a_v1.json")
    cases = load_cases(ROOT / "configs" / "aim120a_v1_cases.json")
    frozen_simulator = H2Simulator(frozen)
    simulator_a = H2Simulator(aim120a["_model_config"])
    simulator_b = H2Simulator(aim120b["_model_config"])
    for case in cases:
        frozen_result = frozen_simulator.run(case)
        result_a = simulator_a.run(case)
        result_b = simulator_b.run(case)
        assert result_a["event_type"] == frozen_result["event_type"] == "fuse"
        assert result_b["event_type"] == result_a["event_type"]
        assert abs(result_a["terminal_time_s"] - frozen_result["terminal_time_s"]) < 0.001
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
        assert result["summary"]["termination_event"] == "proximity_fuse"


def test_universal_control_mapping_uses_pid_floor_and_normalized_attitude_geometry() -> None:
    profiles, errors = scan_library(ROOT / "missiles", ROOT)
    assert errors == []
    indexed = {profile.get("missile_id"): profile for profile in profiles}
    aim120a = indexed["us_aim_120a"]["_model_config"]
    aim7f = indexed["us_aim7f_sparrow"]["_model_config"]
    r27er = indexed["su_r_27er"]["_model_config"]

    assert aim7f["control"]["pid"]["p"] == aim120a["control"]["pid"]["p"]
    assert aim7f["control"]["pid"]["i"] == aim120a["control"]["pid"]["i"]
    assert r27er["control"]["angular_response_scale"] > aim120a["control"]["angular_response_scale"]


def test_r77_high_demand_case_tracks_without_permanent_integral_error() -> None:
    profiles, errors = scan_library(ROOT / "missiles", ROOT)
    assert errors == []
    r77 = next(profile for profile in profiles if profile.get("missile_id") == "su_r_77")
    result = simulate(r77, _r77_high_demand_scenario())

    assert result["summary"]["termination_event"] == "proximity_fuse"
    assert result["summary"]["maximum_actual_g"] > 20.0
