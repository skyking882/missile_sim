from __future__ import annotations

import math
from pathlib import Path

from aim120_model.evasive_target import EvasionPlan, EvasiveTarget, evasive_target_factory
from aim120_model.public_api import SimulationInputError, simulate
from aim120_model.profile_catalog import load_profile_catalog
from aim120_model.target import TargetState
from missile_gui.library import scan_library

ROOT = Path(__file__).resolve().parents[1]
G = 9.80665


class StraightBase:
    def __init__(self, position=(20000.0, 8000.0, 0.0), velocity=(-280.0, 0.0, 0.0)):
        self.position, self.velocity = position, velocity

    def state_at(self, t):
        return TargetState(tuple(p + v * t for p, v in zip(self.position, self.velocity)), self.velocity)


def heading(v):
    return math.atan2(v[2], v[0])


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def scenario(**changes):
    s = dict(launch_speed_kmh=1100, launch_altitude_m=8000, launch_pitch_deg=0, launch_heading_deg=0,
             target_speed_kmh=1000, target_altitude_m=8000, initial_distance_m=20000, target_azimuth_deg=0,
             target_heading_deg=0, target_course_reference="relative_to_los", target_vertical_heading_deg=0,
             target_constant_turn_g=0, max_simulation_time_s=40, observation_mode="sensor_track", loft_enabled=True)
    s.update(changes)
    return s


def profile(name="us_aim_120c_5"):
    _, catalog = load_profile_catalog(ROOT)
    profiles, errors = scan_library(catalog["profiles_dir"], ROOT)
    assert not errors
    return {p["missile_id"]: p for p in profiles}[name]


def test_matches_base_until_start():
    base = StraightBase()
    target = EvasiveTarget(base, EvasionPlan("beam", 5.0), G)
    target.observe_missile(0.0, (0.0, 8000.0, 0.0), (300.0, 0.0, 0.0))
    for t in (0.0, 2.5, 5.0):
        assert target.state_at(t) == base.state_at(t)


def test_beam_converges_perpendicular_to_los_with_turn_limit():
    target = EvasiveTarget(StraightBase(), EvasionPlan("beam", 0.0, max_turn_g=6.0), G)
    missile = (0.0, 8000.0, 0.0)
    previous = target.state_at(0.0)
    limit = 6.0 * G / 280.0 * (1 / 48) + 1e-9
    for k in range(1, 48 * 20 + 1):
        target.observe_missile((k - 1) / 48, missile, (0.0, 0.0, 0.0))
        state = target.state_at(k / 48)
        assert abs(wrap(heading(state.velocity) - heading(previous.velocity))) <= limit
        assert abs(math.hypot(state.velocity[0], state.velocity[2]) - 280.0) < 1e-6
        previous = state
    los = (state.position[0] - missile[0], state.position[2] - missile[2])
    radial = (los[0] * state.velocity[0] + los[1] * state.velocity[2]) / math.hypot(*los)
    assert abs(radial) < 1.0


def test_drag_turns_away_from_missile():
    target = EvasiveTarget(StraightBase(), EvasionPlan("drag", 0.0), G)
    missile = (0.0, 8000.0, 0.0)
    for k in range(1, 48 * 20 + 1):
        target.observe_missile((k - 1) / 48, missile, (0.0, 0.0, 0.0))
        state = target.state_at(k / 48)
    los = (state.position[0] - missile[0], state.position[2] - missile[2])
    radial = (los[0] * state.velocity[0] + los[1] * state.velocity[2]) / math.hypot(*los)
    assert radial > 279.0


def test_earlier_queries_match_forward_queries():
    # Terminal-event interpolation queries a time before the latest one.
    ahead, direct = (EvasiveTarget(StraightBase(), EvasionPlan("beam", 1.0), G) for _ in range(2))
    for target in (ahead, direct):
        target.observe_missile(0.0, (0.0, 8000.0, 0.0), (0.0, 0.0, 0.0))
    late = ahead.state_at(3.0)
    for t in (2.0, 2.01, 1.5):
        a, b = ahead.state_at(t), direct.state_at(t)
        assert all(abs(x - y) < 1e-9 for x, y in zip(a.position + a.velocity, b.position + b.velocity))
    assert ahead.state_at(3.0) == late


def test_plan_validation():
    for bad in (dict(kind="loop", start_s=0.0), dict(kind="beam", start_s=-1.0),
                dict(kind="beam", start_s=0.0, max_turn_g=0.0)):
        try:
            EvasionPlan(**bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_late_start_reproduces_default_runtime_exactly():
    p, s = profile(), scenario()
    default = simulate(p, s)
    wrapped = simulate(p, s, target_factory=evasive_target_factory(EvasionPlan("beam", 1e6)))
    assert wrapped["summary"] == default["summary"]
    assert wrapped["samples"] == default["samples"]
    assert "target_model" not in default["model"]
    assert wrapped["model"]["target_model"]["kind"] == "beam"


def test_legacy_runtime_rejects_target_factory():
    try:
        simulate(profile(), scenario(), model_mode="legacy_h2",
                 target_factory=evasive_target_factory(EvasionPlan("beam", 0.0)))
    except SimulationInputError:
        return
    raise AssertionError("legacy runtime accepted an external target")


def test_early_miss_leaves_hits_unchanged():
    p, s = profile(), scenario()
    assert simulate(p, s, early_miss_s=2.0)["summary"] == simulate(p, s)["summary"]


def test_early_miss_after_a_pass_keeps_the_closest_approach():
    # A late beam lets the missile pass just outside the fuse; the cut comes
    # after the pass, so the near-miss distance is unchanged.
    p, s = profile(), scenario(max_simulation_time_s=120)
    factory = evasive_target_factory(EvasionPlan("beam", 9.0, max_turn_g=9.0))
    full = simulate(p, s, target_factory=factory)["summary"]
    cut = simulate(p, s, target_factory=factory, early_miss_s=2.0)["summary"]
    assert full["termination_event"] != "proximity_fuse" and full["minimum_distance_m"] < 200
    assert cut["termination_event"] == "early_miss"
    assert cut["flight_time_s"] < full["flight_time_s"] - 30
    assert abs(cut["minimum_distance_m"] - full["minimum_distance_m"]) < 1.0


def test_early_miss_on_a_slow_chase_keeps_the_classification_only():
    # The arrival-time rule stops a missile still closing too slowly, so the
    # reported closest approach is the distance at the cut (an overestimate).
    p, s = profile(), scenario(max_simulation_time_s=120, initial_distance_m=40000)
    factory = evasive_target_factory(EvasionPlan("drag", 0.0, max_turn_g=9.0))
    full = simulate(p, s, target_factory=factory)["summary"]
    cut = simulate(p, s, target_factory=factory, early_miss_s=2.0)["summary"]
    assert full["termination_event"] != "proximity_fuse"
    assert cut["termination_event"] == "early_miss"
    assert cut["minimum_distance_m"] >= full["minimum_distance_m"]


def test_early_miss_validation():
    for bad in (0.0, -1.0, float("nan"), True):
        try:
            simulate(profile(), scenario(), early_miss_s=bad)
        except SimulationInputError:
            continue
        raise AssertionError(bad)
    try:
        simulate(profile(), scenario(), model_mode="legacy_h2", early_miss_s=2.0)
    except SimulationInputError:
        return
    raise AssertionError("legacy runtime accepted early_miss_s")
