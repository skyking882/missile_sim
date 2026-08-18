from __future__ import annotations

import json
from pathlib import Path

from aim120_model.public_api import (
    SimulationInputError,
    _case_from_scenario,
    validate_scenario,
)
from aim120_model.target import TargetModel


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = (
    ROOT
    / "data"
    / "reference_external"
    / "sensorwhale_pl12_off_axis_38_course0_20260818.json"
)


def _scenario() -> dict[str, object]:
    return {
        "launch_speed_kmh": 1200.0,
        "launch_altitude_m": 6500.0,
        "launch_pitch_deg": 0.0,
        "launch_heading_deg": 0.0,
        "target_speed_kmh": 1200.0,
        "target_altitude_m": 6500.0,
        "initial_distance_m": 15000.0,
        "target_azimuth_deg": 38.0,
        "target_heading_deg": 0.0,
        "target_vertical_heading_deg": 0.0,
        "target_constant_turn_g": 0.0,
        "max_simulation_time_s": None,
    }


def test_public_scenario_default_course_reference_is_unchanged() -> None:
    normalized = validate_scenario(_scenario())
    assert normalized["target_course_reference"] == "statshark_relative_to_los"


def test_public_scenario_accepts_sensorwhale_course_reference() -> None:
    scenario = _scenario()
    scenario["target_course_reference"] = "sensorwhale_launch_axis"
    normalized = validate_scenario(scenario)
    case = _case_from_scenario(normalized)
    assert (
        case["initial_conditions"]["target_course_reference"]
        == "sensorwhale_launch_axis"
    )


def test_public_scenario_rejects_unknown_course_reference() -> None:
    scenario = _scenario()
    scenario["target_course_reference"] = "unknown"
    try:
        validate_scenario(scenario)
    except SimulationInputError as exc:
        assert "目标航向参考系" in str(exc)
    else:
        raise AssertionError("unknown target course reference was accepted")


def test_frozen_sensorwhale_case_matches_minus_azimuth_local_kinematics() -> None:
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    sensor_case = _case_from_scenario(validate_scenario(reference["local_scenario"]))
    sensor_model = TargetModel(sensor_case["initial_conditions"], 9.80665)

    local_scenario = dict(reference["local_scenario"])
    local_scenario.update(reference["equivalent_existing_local_input"])
    local_case = _case_from_scenario(validate_scenario(local_scenario))
    local_model = TargetModel(local_case["initial_conditions"], 9.80665)

    assert sensor_model.initial_state == local_model.initial_state
    assert sensor_model.initial_state.position == (
        11820.161304100828,
        6500.0,
        9234.922129884875,
    )
    assert abs(sensor_model.initial_state.velocity[0] + 333.3333333333333) < 1e-12
    assert abs(sensor_model.initial_state.velocity[1]) < 1e-12
    assert abs(sensor_model.initial_state.velocity[2]) < 1e-12
