from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

from aim120_model.control import scheduled_pid_values, update_control_feedback
from aim120_model.dynamics import SimState
from aim120_model.profile_adapter import build_h2_candidate_config, load_runtime_defaults
from missile_gui.library import scan_library


ROOT = Path(__file__).resolve().parents[1]


def test_pid_schedule_nodes_interpolation_and_endpoint_clamping():
    schedule = [
        {"time_s": 2.5, "p": 0.000542, "i": 0.02081, "d": 0.000073, "integral_limit": 1.0},
        {"time_s": 3.5, "p": 0.000542, "i": 0.02081, "d": 0.000028, "integral_limit": 1.0},
    ]
    pid = {"p": schedule[0]["p"], "i": schedule[0]["i"], "d": schedule[0]["d"], "integral_limit": 1.0, "schedule": schedule}
    assert scheduled_pid_values(pid, 2.5) == {key: schedule[0][key] for key in ("p", "i", "d", "integral_limit")}
    assert scheduled_pid_values(pid, 3.0) == {
        "p": 0.000542,
        "i": 0.02081,
        "d": 0.0000505,
        "integral_limit": 1.0,
    }
    assert scheduled_pid_values(pid, 3.5) == {key: schedule[1][key] for key in ("p", "i", "d", "integral_limit")}
    assert scheduled_pid_values(pid, 0.0) == scheduled_pid_values(pid, 2.5)
    assert scheduled_pid_values(pid, 9.0) == scheduled_pid_values(pid, 3.5)


def test_pid_schedule_keeps_integrator_and_previous_error_across_time_knots():
    config = json.loads((ROOT / "configs/aim120a_v1.json").read_text())
    config = copy.deepcopy(config)
    config["control"]["pid"] = {
        "switch_time_s": 1e30,
        "p": 0.0,
        "i": 0.2,
        "d": 0.0,
        "integral_limit": 2.0,
        "schedule": [
            {"time_s": 1.0, "p": 0.0, "i": 0.2, "d": 0.0, "integral_limit": 2.0},
            {"time_s": 2.0, "p": 0.0, "i": 0.4, "d": 0.0, "integral_limit": 2.0},
        ],
    }
    config["control"]["integral_limit_semantics"] = "term"
    state = SimState(
        (0.0, 3000.0, 0.0), (300.0, 0.0, 0.0), 0.0, 0.0, 0.0, 0.0, 147.87,
        pitch_pid_integral=0.1,
        previous_pitch_error=3.0,
        measured_pitch_normal_g=0.0,
    )
    first = update_control_feedback(state, (2.0, 0.0), config, 0.1, enabled=True, time_s=1.0)
    state = replace(state, **first)
    second = update_control_feedback(state, (2.0, 0.0), config, 0.1, enabled=True, time_s=1.5)
    # Ki(t)=0.3 at the midpoint: 0.1 initial + 0.2*2*0.1 + 0.3*2*0.1.
    assert abs(first["pitch_pid_integral"] - 0.14) < 1e-12
    assert abs(second["pitch_pid_integral"] - 0.20) < 1e-12
    assert first["previous_pitch_error"] == 2.0
    assert second["previous_pitch_error"] == 2.0


def test_active_aim120a_and_rb99_scheduled_pid_profiles_run():
    defaults = load_runtime_defaults(str(ROOT / "config/profile_h2_runtime_defaults.json"))
    profiles, errors = scan_library(ROOT / "data/profile_sets/2.59.0.27/missiles", ROOT)
    assert errors == []
    by_id = {profile["missile_id"]: profile for profile in profiles}
    scenario = {
        "loft_enabled": True,
        "observation_mode": "ideal_truth",
        "target_course_reference": "relative_to_los",
        "launch_speed_kmh": 1200.0,
        "launch_altitude_m": 6200.0,
        "launch_pitch_deg": 0.0,
        "launch_heading_deg": 0.0,
        "target_speed_kmh": 1550.0,
        "target_altitude_m": 6200.0,
        "initial_distance_m": 8000.0,
        "target_azimuth_deg": 30.0,
        "target_heading_deg": 0.0,
        "target_vertical_heading_deg": 0.0,
        "target_constant_turn_g": 0.0,
        "max_simulation_time_s": 8.0,
    }
    from aim120_model.public_api import simulate

    for missile_id in ("us_aim_120a", "swd_rb99"):
        profile = by_id[missile_id]
        rebuilt, assumptions = build_h2_candidate_config(profile, defaults)
        profile["_model_config"] = rebuilt
        profile["_runtime_assumptions"] = assumptions
        result = simulate(profile, scenario)
        assert result["summary"]["termination_event"] in {"proximity_fuse", "fuse", "lifetime", "scenario_limit"}
        assert result["missile"]["parameter_source"] == {
            "version": "2.59.0.27",
            "commit": "4ac255b829a27346b9300fccdc138362c2535fbe",
        }
        assert result["samples"]
        assert all(sample["time_s"] == sample["time_s"] for sample in result["samples"])
        assert any("elapsed seconds since launch" in note for note in result["model"]["runtime_assumptions"])
