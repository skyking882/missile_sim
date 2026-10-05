"""Opt-in surface multipath on the radar seeker's measured position."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from aim120_model.profile_catalog import load_profile_catalog
from aim120_model.public_api import SimulationInputError, simulate
from aim120_model.radar_seeker import RadarSeekerObserver
from missile_gui.library import scan_library

ROOT = Path(__file__).resolve().parents[1]


def profile(name="cn_pl12"):
    _, catalog = load_profile_catalog(ROOT)
    profiles, errors = scan_library(catalog["profiles_dir"], ROOT)
    assert not errors
    return {p["missile_id"]: p for p in profiles}[name]


def scenario(**changes):
    s = dict(launch_speed_kmh=1200, launch_altitude_m=8000, launch_pitch_deg=0, launch_heading_deg=0,
             target_speed_kmh=900, target_altitude_m=30, initial_distance_m=25000, target_azimuth_deg=0,
             target_heading_deg=0, target_course_reference="relative_to_los", target_vertical_heading_deg=0,
             target_constant_turn_g=0, max_simulation_time_s=40, observation_mode="sensor_track", loft_enabled=True)
    s.update(changes)
    return s


def test_measurement_moves_toward_the_surface_image():
    radar = RadarSeekerObserver(profile()["guidance"]["sensor_model"])
    radar.enable_multipath(0.5)
    # [0, 1, 60, 0]: strength 0.5 at 30 m, so the point moves a quarter of the way to the image.
    assert radar._multipath((1.0, 30.0, 2.0), (3.0, -2.0, 4.0)) == ((1.0, 15.0, 2.0), (3.0, -1.0, 4.0))
    assert radar._multipath((1.0, 80.0, 2.0), (3.0, 0.0, 4.0))[0] == (1.0, 80.0, 2.0)
    radar.enable_multipath(1.0)
    assert radar._multipath((0.0, 10.0, 0.0), (0.0, 0.0, 0.0))[0][1] == pytest.approx(-20.0 / 3.0)


def test_zero_gain_and_targets_above_the_table_match_default():
    p = profile()
    assert simulate(p, scenario(), multipath_gain=0.0)["samples"] == simulate(p, scenario())["samples"]
    high = scenario(target_altitude_m=300)
    assert simulate(p, high, multipath_gain=1.0)["samples"] == simulate(p, high)["samples"]


def test_low_target_miss_goes_into_the_ground():
    p = profile()
    assert simulate(p, scenario())["summary"]["termination_event"] == "proximity_fuse"
    result = simulate(p, scenario(), multipath_gain=0.5)
    assert result["summary"]["termination_event"] == "ground"
    assert result["model"]["multipath_gain"] == 0.5
    assert result["model"]["multipath_effect"] == [[0.0, 1.0], [60.0, 0.0]]


def test_multipath_validation():
    p = profile()
    for gain in (1.5, -0.1, True, "0.5"):
        with pytest.raises(SimulationInputError):
            simulate(p, scenario(), multipath_gain=gain)
    with pytest.raises(SimulationInputError):
        simulate(p, scenario(), multipath_gain=0.5, model_mode="legacy_h2")
    without = copy.deepcopy(p)
    without["guidance"]["sensor_model"]["radar_seeker"]["multipath_effect"] = None
    with pytest.raises(SimulationInputError):
        simulate(without, scenario(), multipath_gain=0.5)
