from __future__ import annotations

import math
from pathlib import Path

from aim120_model.chaff import ChaffProgram, ChaffSpec, ChaffingTarget, chaffing_factory
from aim120_model.evasive_target import EvasionPlan, evasive_target_factory
from aim120_model.profile_catalog import load_profile_catalog
from aim120_model.public_api import simulate
from aim120_model.target import TargetState
from missile_gui.library import scan_library

ROOT = Path(__file__).resolve().parents[1]


def profile(name="us_aim_120c_5"):
    _, catalog = load_profile_catalog(ROOT)
    profiles, errors = scan_library(catalog["profiles_dir"], ROOT)
    assert not errors
    return {p["missile_id"]: p for p in profiles}[name]


def scenario(**changes):
    s = dict(launch_speed_kmh=1100, launch_altitude_m=8000, launch_pitch_deg=0, launch_heading_deg=0,
             target_speed_kmh=1000, target_altitude_m=8000, initial_distance_m=20000, target_azimuth_deg=0,
             target_heading_deg=0, target_course_reference="relative_to_los", target_vertical_heading_deg=0,
             target_constant_turn_g=0, max_simulation_time_s=60, observation_mode="sensor_track", loft_enabled=True)
    s.update(changes)
    return s


class Straight:
    def state_at(self, t):
        return TargetState((1000.0 + 250.0 * t, 8000.0, 0.0), (250.0, 0.0, 0.0))


def test_envelope_follows_assumed_profile_and_lifetime():
    spec = ChaffSpec(rcs_ratio=2.0)
    assert spec.envelope(0.0) == 0.0
    assert abs(spec.envelope(0.075) - 0.5) < 1e-12
    assert spec.envelope(5.0) == 1.0
    assert abs(spec.envelope(13.5) - 0.5) < 1e-12
    assert spec.envelope(15.0) == 0.0


def test_program_release_times_and_validation():
    assert ChaffProgram(2.0, interval_s=0.5, per_salvo=2, total=5).release_times() == [2.0, 2.0, 2.5, 2.5, 3.0]
    for bad in (dict(rcs_ratio=0.0), dict(rcs_ratio=1.0, stop_time_constant_s=0.0),
                dict(rcs_ratio=1.0, profile_s=(1.0, 0.5, 2.0, 3.0))):
        try:
            ChaffSpec(**bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_bundles_stop_in_air_and_expire():
    target = ChaffingTarget(Straight(), ChaffSpec(rcs_ratio=3.0), ChaffProgram(1.0, total=1))
    assert target.decoys_at(0.5) == []
    assert target.decoys_at(1.02) == []  # Released, but before the RCS rise starts.
    [bundle] = target.decoys_at(3.0)
    assert math.hypot(*bundle.velocity) < 1e-3
    assert abs(bundle.rcs_m2 - 3.0) < 1e-12
    assert abs(bundle.position[0] - (1250.0 + 250.0 * 0.1)) < 1.0
    assert target.decoys_at(16.5) == []


def test_empty_decoy_path_reproduces_single_target_seeker():
    p, s = profile(), scenario()
    evade = evasive_target_factory(EvasionPlan("beam", 8.0))
    plain = simulate(p, s, target_factory=evade)
    empty = simulate(p, s, target_factory=chaffing_factory(evade, ChaffSpec(rcs_ratio=5.0), ChaffProgram(0.0, total=0)))
    assert empty["samples"] == plain["samples"]
    assert empty["summary"]["decoy_track_time_s"] == 0.0


def test_resolvable_chaff_does_not_capture_a_head_on_target():
    # Bundles stop in air; a head-on target keeps ~280 m/s ground radial speed,
    # far outside the 20 m/s Doppler cell, so even very strong chaff is ignored.
    p, s = profile(), scenario()
    plain = simulate(p, s)
    chaffed = simulate(p, s, target_factory=chaffing_factory(None, ChaffSpec(rcs_ratio=50.0), ChaffProgram(2.0, 0.5)))
    assert chaffed["summary"]["decoy_track_time_s"] == 0.0
    assert chaffed["summary"]["termination_event"] == plain["summary"]["termination_event"]


def test_strong_chaff_captures_a_beaming_target_and_weak_chaff_does_not():
    # Look-up geometry: with the missile above, the clutter notch removes both
    # the beaming aircraft and the near-stationary bundles, so chaff cannot act.
    p, s = profile(), scenario(loft_enabled=False, target_altitude_m=9500)
    evade = evasive_target_factory(EvasionPlan("beam", 0.0, max_turn_g=9.0))
    strong = simulate(p, s, target_factory=chaffing_factory(evade, ChaffSpec(rcs_ratio=10.0), ChaffProgram(4.0, 0.5)))
    weak = simulate(p, s, target_factory=chaffing_factory(evade, ChaffSpec(rcs_ratio=0.01), ChaffProgram(4.0, 0.5)))
    assert strong["summary"]["decoy_track_time_s"] > 0.0
    assert weak["summary"]["decoy_track_time_s"] == 0.0


def test_clear_sky_beam_lets_any_chaff_capture_but_not_without_chaff():
    # Look-up geometry: missile at 8 km, beaming target at 9.5 km, no clutter behind it.
    p, s = profile(), scenario(loft_enabled=False, target_altitude_m=9500)
    evade = evasive_target_factory(EvasionPlan("beam", 0.0, max_turn_g=9.0))
    weak = chaffing_factory(evade, ChaffSpec(rcs_ratio=0.01), ChaffProgram(4.0, 0.5))
    pd = simulate(p, s, target_factory=weak, clutter_model="look_down_angle")
    cw = simulate(p, s, target_factory=weak, clutter_model="look_down_angle", cw_on_clear_beam=True)
    assert pd["summary"]["decoy_track_time_s"] == 0.0
    assert cw["summary"]["cw_mode_time_s"] > 0.0
    assert cw["summary"]["decoy_track_time_s"] > 0.0
    empty = chaffing_factory(evade, ChaffSpec(rcs_ratio=0.01), ChaffProgram(0.0, total=0))
    plain = simulate(p, s, target_factory=empty, clutter_model="look_down_angle")
    flagged = simulate(p, s, target_factory=empty, clutter_model="look_down_angle", cw_on_clear_beam=True)
    assert flagged["samples"] == plain["samples"]  # No chaff: the target is still tracked as before.


def test_look_down_beam_stays_in_doppler_mode_until_the_end_game():
    # Target 3 km below at 20 km: about 9 deg look-down, ground clutter behind it.
    # Only in the last second or two, when the missile has come down to the
    # target's altitude, does the line of sight flatten below the threshold.
    p, s = profile(), scenario(loft_enabled=False, target_altitude_m=5000)
    evade = evasive_target_factory(EvasionPlan("beam", 0.0, max_turn_g=9.0))
    r = simulate(p, s, target_factory=chaffing_factory(evade, ChaffSpec(rcs_ratio=0.01), ChaffProgram(4.0, 0.5)),
                 clutter_model="look_down_angle", cw_on_clear_beam=True)
    summary = r["summary"]
    assert 0.0 < summary["cw_mode_time_s"] < 2.0
    first = summary["first_decoy_track_time_s"]
    assert first is None or first > summary["flight_time_s"] - 2.0
