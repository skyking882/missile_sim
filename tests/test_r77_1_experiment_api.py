import csv
import hashlib
import json
from pathlib import Path

import pytest

from aim120_model.profile_catalog import load_profile_catalog
from aim120_model.public_api import SimulationInputError, simulate
from missile_gui.library import scan_library


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_MODE = "r77_1_fixed_257_request_fit_v1"
EXPERIMENT_CONFIG_PATH = ROOT / "config/r77_1_fixed_experiment_2.57.1.107.json"


def _experiment_config():
    return json.loads(EXPERIMENT_CONFIG_PATH.read_text(encoding="utf-8"))


def _fixed_scenario():
    return _experiment_config()["scenario"]


def _profiles():
    _, active = load_profile_catalog(ROOT)
    profiles, errors = scan_library(
        active["profiles_dir"], ROOT,
        expected_source_version=active["source_version"],
        expected_source_commit=active["source_commit"],
    )
    assert not errors
    return profiles


def test_fixed_r77_api_matches_accepted_free_forward_trajectory():
    fit_path = ROOT / "outputs/r77_1_fixed_pid_request.json"
    reference_path = ROOT / "outputs/r77_1_fixed_pid_request_trajectory.tsv"
    missing = [path.relative_to(ROOT).as_posix() for path in (fit_path, reference_path) if not path.is_file()]
    if missing:
        pytest.skip("research trajectory comparison requires omitted files: " + ", ".join(missing))
    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    reference = list(csv.DictReader(reference_path.open(encoding="utf-8"), delimiter="\t"))
    profile = next(item for item in _profiles() if item["missile_id"] == "su_r_77_1")

    result = simulate(profile, _fixed_scenario(), model_mode=EXPERIMENT_MODE)

    assert result["model_mode"] == EXPERIMENT_MODE
    assert result["summary"]["termination_event"] == "proximity_fuse"
    assert result["summary"]["flight_time_s"] == pytest.approx(fit["terminal_time_s"], abs=1e-12)
    assert result["summary"]["minimum_distance_m"] == pytest.approx(fit["minimum_distance_m"], abs=1e-12)
    assert len(result["samples"]) == len(reference)
    for actual, expected in zip(result["samples"], reference):
        assert actual["time_s"] == pytest.approx(float(expected["time_s"]), abs=1e-12)
        for field in ("alpha_deg", "wind_normal_specific_force_g", "speed_kmh"):
            assert actual[field] == pytest.approx(float(expected[field]), abs=1e-9)
    assert result["missile"]["parameter_source"] == {
        "version": "2.57.1.107",
        "commit": "3f20645862c538b46c94e3f60ee30da9dfacd81a",
    }
    assert result["missile"]["catalog_parameter_source"]["version"] == "2.59.0.27"


def test_fixed_r77_mode_rejects_other_missiles_and_scenarios():
    profiles = _profiles()
    profile = next(item for item in profiles if item["missile_id"] == "su_r_77_1")
    other = next(item for item in profiles if item["missile_id"] == "us_aim_120a")
    scenario = _fixed_scenario()

    with pytest.raises(SimulationInputError, match="只支持 R-77-1"):
        simulate(other, scenario, model_mode=EXPERIMENT_MODE)
    changed = dict(scenario, initial_distance_m=9000)
    with pytest.raises(SimulationInputError, match="场景已锁定"):
        simulate(profile, changed, model_mode=EXPERIMENT_MODE)
    with pytest.raises(SimulationInputError, match="未知模型模式"):
        simulate(profile, scenario, model_mode="all_missiles_fitted")


def test_general_mode_matches_pre_experiment_result_hash():
    profiles = _profiles()
    profile = next(item for item in profiles if item["missile_id"] == "us_aim_120a")
    scenario = {
        "launch_speed_kmh": 1200,
        "launch_altitude_m": 6500,
        "launch_pitch_deg": 0,
        "launch_heading_deg": 0,
        "target_speed_kmh": 1200,
        "target_altitude_m": 6500,
        "initial_distance_m": 15000,
        "target_azimuth_deg": 0,
        "target_heading_deg": 0,
        "target_vertical_heading_deg": 0,
        "target_constant_turn_g": 0,
        "target_course_reference": "relative_to_los",
        "observation_mode": "ideal_truth",
        "loft_enabled": False,
        "max_simulation_time_s": 0.5,
    }
    result = simulate(profile, scenario)
    digest = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert digest == "cc7f8a4b841fab4c11cd88598224042bfd1d5388bb68cc8c0f5e8b2adfbbcca1"
