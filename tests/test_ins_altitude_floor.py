"""Public INS floor contract and a deterministic descending-track flight."""
from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

import pytest

from aim120_model.public_api import SimulationInputError, _case_from_scenario, create_surface_missile, simulate, validate_scenario
from aim120_model.surface_runtime import SurfaceRuntime, simulate_surface
from aim120_model.target import TargetState
from missile_gui.library import scan_library

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def profiles():
    loaded, errors = scan_library(ROOT / "missiles", ROOT)
    assert not errors
    return {p["missile_id"]: p for p in loaded}


def scene(**changes):
    values = dict(launch_speed_kmh=1200, launch_altitude_m=8600, launch_pitch_deg=0, launch_heading_deg=0,
                  target_speed_kmh=1000, target_altitude_m=2500, initial_distance_m=40000, target_azimuth_deg=0,
                  target_heading_deg=0, target_vertical_heading_deg=-20, target_constant_turn_g=0,
                  max_simulation_time_s=1, observation_mode="sensor_track", loft_enabled=False,
                  datalink_disconnect_time_s=0.02)
    values.update(changes)
    return values


def call(entry, profile, scenario, **options):
    if entry == "simulate":
        return simulate(profile, scenario, **options)
    if entry == "simulate_surface":
        return simulate_surface(profile, scenario, **options)
    normalized = validate_scenario(scenario)
    defaults = json.loads((ROOT / "config/effective_surface_defaults.json").read_text())
    reference = SurfaceRuntime(profile, normalized, _case_from_scenario(normalized), defaults)
    return create_surface_missile(profile, launch_position_m=reference.creation["position_xyz_m"],
                                  launch_velocity_mps=reference.creation["velocity_xyz_mps"],
                                  launch_pitch_deg=normalized["launch_pitch_deg"],
                                  launch_heading_deg=normalized["launch_heading_deg"], target=reference.target,
                                  observation_mode=normalized["observation_mode"], loft=normalized["loft_enabled"],
                                  end_time_s=normalized["max_simulation_time_s"],
                                  datalink_disconnect_time_s=normalized["datalink_disconnect_time_s"], **options)


def finish(runtime):
    while not runtime.done:
        runtime.step()
    return runtime.result()


@pytest.mark.parametrize("entry", ["simulate", "simulate_surface", "create_surface_missile"])
@pytest.mark.parametrize("name,mode", [("us_aim_120a", "sensor_track"), ("su_r_27t", "sensor_track"),
                                     ("us_aim_120a", "ideal_truth")])
def test_public_floor_none_is_bitwise_identical_to_omitted(profiles, entry, name, mode):
    scenario = scene(observation_mode=mode)
    omitted = call(entry, profiles[name], scenario)
    explicit = call(entry, profiles[name], scenario, min_altitude_m=None)
    if entry == "create_surface_missile":
        omitted, explicit = finish(omitted), finish(explicit)
    assert pickle.dumps(omitted) == pickle.dumps(explicit)


@pytest.mark.parametrize("entry", ["simulate", "simulate_surface", "create_surface_missile"])
@pytest.mark.parametrize("name", ["us_aim_120a", "su_r_27t"])
def test_public_floor_reaches_sensor_and_kinematic_ins(profiles, entry, name):
    scenario = scene(initial_distance_m=40000 if name == "us_aim_120a" else 10000,
                     launch_altitude_m=2500, target_altitude_m=1000,
                     target_vertical_heading_deg=0, max_simulation_time_s=0.1)
    result = call(entry, profiles[name], scenario, min_altitude_m=5000.0)
    if entry == "create_surface_missile":
        assert result.provider.inertial.min_altitude_m == 5000.0
        result = finish(result)
    else:
        assert result["model"]["min_altitude_m"] == 5000.0
    # Direct observations may stay below the floor. The sensor path disconnects
    # and coasts above it; the optical kinematic path still observes directly.
    if name == "us_aim_120a":
        coast = [r for r in result["samples"] if r["track_mode"] in ("inertial", "ins_search")]
        assert coast
        assert all(r["observed_target_position_m"][1] >= 5000.0 for r in coast)


@pytest.mark.parametrize("entry", ["simulate", "simulate_surface", "create_surface_missile"])
@pytest.mark.parametrize("value", [math.nan, True, "50"])
def test_public_floor_rejects_invalid_values(profiles, entry, value):
    with pytest.raises(ValueError, match="min_altitude_m"):
        call(entry, profiles["us_aim_120a"], scene(), min_altitude_m=value)


@pytest.mark.parametrize("entry", ["simulate", "simulate_surface", "create_surface_missile"])
def test_public_floor_rejects_ideal_truth_mode(profiles, entry):
    with pytest.raises(ValueError, match="min_altitude_m"):
        call(entry, profiles["us_aim_120a"], scene(observation_mode="ideal_truth"), min_altitude_m=50.0)


def test_public_floor_rejects_legacy_h2(profiles):
    with pytest.raises(SimulationInputError, match="min_altitude_m"):
        simulate(profiles["us_aim_120a"], scene(), model_mode="legacy_h2", min_altitude_m=50.0)


def test_descending_track_disconnect_then_level_target_keeps_floor_and_missile_airborne(profiles):
    class LevelTarget:
        def state_at(self, t):
            return TargetState((40000.0 + 250.0 * t, 2500.0 - 100.0 * min(t, 1.0), 0.0),
                               (250.0, -100.0 if t < 1.0 else 0.0, 0.0))

    launch = dict(launch_position_m=(0.0, 8600.0, 0.0), launch_velocity_mps=(1200.0 / 3.6, 0.0, 0.0),
                  launch_pitch_deg=0.0, launch_heading_deg=0.0, target=LevelTarget(), loft=False,
                  datalink_disconnect_time_s=0.02)
    full = finish(create_surface_missile(profiles["us_aim_120a"], **launch))
    bounded = finish(create_surface_missile(profiles["us_aim_120a"], **launch, min_altitude_m=50.0))
    assert full["samples"][0]["observed_target_position_m"][1] == 2500.0
    assert full["samples"][0]["observed_target_velocity_mps"][1] == -100.0
    assert min(r["observed_target_position_m"][1] for r in full["samples"]) < 0.0
    assert min(r["observed_target_position_m"][1] for r in bounded["samples"]) == 50.0
    clipped = [r for r in bounded["samples"] if r["observed_target_position_m"][1] == 50.0]
    assert clipped and all(r["observed_target_velocity_mps"][1] == 0.0 for r in clipped)
    assert bounded["event_type"] == "lifetime"
    assert min(r["position_m"][1] for r in bounded["samples"]) > 0.0
    assert all(r["target_position_m"][1] >= 2400.0 for r in bounded["samples"])
