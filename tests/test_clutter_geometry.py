from __future__ import annotations

from pathlib import Path

from aim120_model.evasive_target import EvasionPlan, evasive_target_factory
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
    s = dict(launch_speed_kmh=1100, launch_altitude_m=8000, launch_pitch_deg=0, launch_heading_deg=0,
             target_speed_kmh=1000, target_altitude_m=8000, initial_distance_m=10000, target_azimuth_deg=0,
             target_heading_deg=0, target_course_reference="relative_to_los", target_vertical_heading_deg=0,
             target_constant_turn_g=0, max_simulation_time_s=30, observation_mode="sensor_track", loft_enabled=True)
    s.update(changes)
    return s


def seeker():
    radar = RadarSeekerObserver(profile()["guidance"]["sensor_model"])
    radar.clutter_model = "geometric_mainlobe"
    return radar


def clutter(radar, missile_alt, target_alt, horizontal):
    relative = (horizontal, target_alt - missile_alt, 0.0)
    range_m = (horizontal ** 2 + relative[1] ** 2) ** 0.5
    return radar._on_clutter(target_alt < missile_alt, (0.0, missile_alt, 0.0), range_m, relative)


def test_geometric_clutter_follows_beam_ground_intersection():
    radar = seeker()
    assert not clutter(radar, 8000.0, 7999.8, 10000.0)  # Co-altitude high: sky behind the target.
    assert clutter(radar, 2000.0, 1999.8, 10000.0)      # Co-altitude low: beam edge hits ground at ~7.7 km.
    assert clutter(radar, 8000.0, 2000.0, 8000.0)       # Steep look-down: ground right behind the target.
    assert not clutter(radar, 2000.0, 6000.0, 8000.0)   # Look-up: beam edge never reaches the ground.


def test_explicit_look_down_equals_default():
    p, s = profile(), scenario()
    assert simulate(p, s, clutter_model="look_down")["samples"] == simulate(p, s)["samples"]


def test_high_co_altitude_beam_never_reports_clutter_notch():
    p, s = profile(), scenario()
    beam = evasive_target_factory(EvasionPlan("beam", 0.0, max_turn_g=9.0))
    default = simulate(p, s, target_factory=beam)
    geometric = simulate(p, s, target_factory=beam, clutter_model="geometric_mainlobe")
    assert any(x["radar_reject_reason"] == "ground_clutter_notch" for x in default["samples"])
    assert not any(x["radar_reject_reason"] == "ground_clutter_notch" for x in geometric["samples"])
    assert geometric["model"]["clutter_model"] == "geometric_mainlobe"


def test_clutter_model_validation():
    for kwargs in (dict(clutter_model="terrain"), dict(clutter_model="geometric_mainlobe", model_mode="legacy_h2")):
        try:
            simulate(profile(), scenario(), **kwargs)
        except SimulationInputError:
            continue
        raise AssertionError(kwargs)


def test_look_down_angle_needs_a_real_depression():
    radar = RadarSeekerObserver(profile()["guidance"]["sensor_model"])
    radar.clutter_model = "look_down_angle"
    assert not clutter(radar, 8000.0, 7999.8, 10000.0)   # Co-altitude: no notch.
    assert not clutter(radar, 8000.0, 7800.0, 10000.0)   # 1.1 deg below: under the 2 deg default.
    assert clutter(radar, 8000.0, 7500.0, 10000.0)       # 2.9 deg below: look-down.
    assert not clutter(radar, 2000.0, 1999.8, 10000.0)   # Low altitude alone is not enough.
    p, s = profile(), scenario()
    r = simulate(p, s, clutter_model="look_down_angle", clutter_min_depression_deg=4.0)
    assert r["model"]["clutter_min_depression_deg"] == 4.0
    try:
        simulate(p, s, clutter_model="geometric_mainlobe", clutter_min_depression_deg=4.0)
    except SimulationInputError:
        return
    raise AssertionError("depression threshold accepted for another clutter model")
