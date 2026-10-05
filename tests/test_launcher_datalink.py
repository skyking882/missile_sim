from __future__ import annotations

from pathlib import Path

from aim120_model.evasive_target import EvasionPlan, evasive_target_factory
from aim120_model.profile_catalog import load_profile_catalog
from aim120_model.public_api import SimulationInputError, simulate
from missile_gui.library import scan_library

ROOT = Path(__file__).resolve().parents[1]
OPTIONS = dict(launcher_radar_gimbal_deg=60.0, require_seeker_lock=True, clutter_model="look_down_angle")


def profile(name="cn_pl12"):
    _, catalog = load_profile_catalog(ROOT)
    profiles, errors = scan_library(catalog["profiles_dir"], ROOT)
    assert not errors
    return {p["missile_id"]: p for p in profiles}[name]


def scenario(**changes):
    s = dict(launch_speed_kmh=1200, launch_altitude_m=8000, launch_pitch_deg=0, launch_heading_deg=0,
             target_speed_kmh=1000, target_altitude_m=8000, initial_distance_m=10000, target_azimuth_deg=0,
             target_heading_deg=0, target_course_reference="relative_to_los", target_vertical_heading_deg=0,
             target_constant_turn_g=0, max_simulation_time_s=40, observation_mode="sensor_track", loft_enabled=True)
    s.update(changes)
    return s


def test_options_off_leave_the_run_unchanged():
    p, s = profile(), scenario(target_altitude_m=5000, target_heading_deg=90)
    plain = simulate(p, s, clutter_model="look_down_angle")
    again = simulate(p, s, clutter_model="look_down_angle", launcher_radar_gimbal_deg=None, require_seeker_lock=False)
    assert plain["samples"] == again["samples"]
    assert "datalink_lost_time_s" not in plain["summary"]


def test_look_down_beam_notches_the_launcher_and_the_unlocked_missile_cannot_fuse():
    # Target 3 km below, beaming: the seeker never locks. With a truth datalink the
    # missile still hits; once the launcher's radar notches the target too, it misses.
    p, s = profile(), scenario(target_altitude_m=5000, target_heading_deg=90)
    beam = evasive_target_factory(EvasionPlan("beam", 0.0, max_turn_g=9.0))
    truth_link = simulate(p, s, target_factory=beam, clutter_model="look_down_angle")
    assert truth_link["summary"]["termination_event"] == "proximity_fuse"
    gated = simulate(p, s, target_factory=beam, early_miss_s=2.0, **OPTIONS)
    assert gated["summary"]["datalink_lost_reason"] == "launcher_clutter_notch"
    assert gated["summary"]["seeker_locked"] is False
    assert gated["summary"]["termination_event"] != "proximity_fuse"


def test_target_outside_the_radar_cone_drops_the_datalink_at_once():
    p, s = profile(), scenario(target_azimuth_deg=70, initial_distance_m=15000)
    r = simulate(p, s, **OPTIONS)
    assert r["summary"]["datalink_lost_reason"] == "launcher_gimbal"
    assert r["summary"]["datalink_lost_time_s"] == 0.0


def test_co_altitude_head_on_keeps_the_datalink_and_hits():
    p, s = profile(), scenario()
    r = simulate(p, s, **OPTIONS)
    assert r["summary"]["seeker_locked"] is True
    assert r["summary"]["datalink_lost_reason"] == ""
    assert r["summary"]["termination_event"] == "proximity_fuse"


def test_validation():
    for bad in (0.0, 200.0, True, "60"):
        try:
            simulate(profile(), scenario(), launcher_radar_gimbal_deg=bad)
        except SimulationInputError:
            continue
        raise AssertionError(bad)
    try:
        simulate(profile(), scenario(), model_mode="legacy_h2", require_seeker_lock=True)
    except SimulationInputError:
        return
    raise AssertionError("legacy mode accepted require_seeker_lock")
