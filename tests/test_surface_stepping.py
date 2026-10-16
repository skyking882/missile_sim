"""Shared-clock interface and exact legacy compatibility, without new physics."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from aim120_model.chaff import ChaffProgram, ChaffSpec, chaffing_factory
from aim120_model.effective_surface_core import axes
from aim120_model.public_api import SimulationInputError, _case_from_scenario, create_surface_missile, simulate, validate_scenario
from aim120_model.surface_runtime import SurfaceRuntime
from aim120_model.target import TargetState
from missile_gui.library import scan_library

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def profiles():
    loaded, errors = scan_library(ROOT / "missiles", ROOT)
    assert not errors
    return {p["missile_id"]: p for p in loaded}


def scenario(**changes):
    values = dict(launch_speed_kmh=1200, launch_altitude_m=8000, launch_pitch_deg=0, launch_heading_deg=0,
                  target_speed_kmh=1000, target_altitude_m=8000, initial_distance_m=10000, target_azimuth_deg=0,
                  target_heading_deg=0, target_course_reference="relative_to_los", target_vertical_heading_deg=0,
                  target_constant_turn_g=0, max_simulation_time_s=0.107, observation_mode="sensor_track", loft_enabled=True)
    values.update(changes)
    return values


def runtime(profile, scene, target_factory=None):
    normalized = validate_scenario(scene)
    defaults = json.loads((ROOT / "config/effective_surface_defaults.json").read_text())
    return SurfaceRuntime(profile, normalized, _case_from_scenario(normalized), defaults, target_factory)


def finish(model):
    while not model.done:
        previous = model.time_s
        model.step()
        assert previous < model.time_s <= model._end_time
    return model.result()


def explicit(profile, reference, **changes):
    options = dict(launch_position_m=reference.creation["position_xyz_m"],
                   launch_velocity_mps=reference.creation["velocity_xyz_mps"],
                   launch_pitch_deg=reference.scenario["launch_pitch_deg"],
                   launch_heading_deg=reference.scenario["launch_heading_deg"], target=reference.target,
                   observation_mode=reference.scenario["observation_mode"], loft=reference.scenario["loft_enabled"],
                   end_time_s=reference.scenario["max_simulation_time_s"],
                   datalink_enabled=reference.scenario["datalink_enabled"],
                   datalink_disconnect_time_s=reference.scenario["datalink_disconnect_time_s"])
    options.update(changes)
    return create_surface_missile(profile, **options)


@pytest.mark.parametrize("name,scene,early,event", [
    ("cn_pl12", scenario(), None, "lifetime"),
    ("su_r_77_1", scenario(observation_mode="ideal_truth", launch_pitch_deg=27, launch_heading_deg=-35), None, "lifetime"),
    ("swd_rb99", scenario(initial_distance_m=1000, max_simulation_time_s=5), None, "fuse"),
    ("su_r_77_1", scenario(launch_altitude_m=3, launch_pitch_deg=-30), None, "ground"),
    ("us_aim_120c_5", scenario(target_heading_deg=180, target_speed_kmh=2000,
                              initial_distance_m=50000, max_simulation_time_s=20), 2, "early_miss"),
])
def test_simulate_equals_manual_including_terminal_sample(profiles, name, scene, early, event):
    expected = runtime(profiles[name], scene).simulate(scene["max_simulation_time_s"], early)
    manual = runtime(profiles[name], scene)
    assert manual.begin(scene["max_simulation_time_s"], early) is manual
    actual = finish(manual)
    assert actual == expected
    assert manual.event == actual["event_type"] == event
    assert actual["samples"][-1]["time_s"] == manual.time_s
    assert actual["samples"][-1]["position_m"] == manual.state[:3]
    assert sum(row["time_s"] == manual.time_s for row in actual["samples"]) == 1
    assert manual.dt == 1 / 48
    if event == "lifetime":
        assert manual.time_s == scene["max_simulation_time_s"]
        assert manual.time_s - actual["samples"][-2]["time_s"] < manual.dt


@pytest.mark.parametrize("observation,loft,pitch,heading", [
    ("ideal_truth", False, 0, 0), ("sensor_track", True, 27, -35),
    ("sensor_track", False, -12, 80),
])
def test_explicit_aligned_launch_matches_old_exactly(profiles, observation, loft, pitch, heading):
    scene = scenario(observation_mode=observation, loft_enabled=loft,
                     launch_pitch_deg=pitch, launch_heading_deg=heading, max_simulation_time_s=1)
    legacy = runtime(profiles["cn_pl12"], scene)
    factory = explicit(profiles["cn_pl12"], legacy)
    assert factory.state[3:6] == legacy.creation["velocity_xyz_mps"]
    assert finish(factory) == legacy.simulate(1)


def test_explicit_misalignment_and_world_translation(profiles):
    scene = scenario(observation_mode="ideal_truth", loft_enabled=False)
    reference = runtime(profiles["cn_pl12"], scene)
    aligned = explicit(profiles["cn_pl12"], reference)
    misaligned = explicit(profiles["cn_pl12"], runtime(profiles["cn_pl12"], scene),
                          launch_position_m=(1500., 8100., -2000.), launch_velocity_mps=(300., -40., 75.))
    assert misaligned.state[:3] == (1500., 8100., -2000.)
    assert misaligned.state[3:6] == (300., -40., 75.)
    assert misaligned.state[6:10] == aligned.state[6:10]
    assert axes(misaligned.state[6:10])[0] == (1., 0., 0.)
    assert misaligned._rows[0]["alpha_deg"] > 0
    misaligned.step()
    assert misaligned.state[:3] != misaligned.creation["position_xyz_m"]

    offset = (1500., 0., -2000.)
    class Translated:
        def state_at(self, t):
            state = reference.target.state_at(t)
            return TargetState(tuple(a + b for a, b in zip(state.position, offset)), state.velocity)
    translated = explicit(profiles["cn_pl12"], runtime(profiles["cn_pl12"], scene),
                          launch_position_m=offset[:1] + (8000.,) + offset[2:], target=Translated())
    translated.step()
    aligned.step()
    assert math.dist(translated.state[:3], tuple(a + b for a, b in zip(aligned.state[:3], offset))) < 1e-9
    assert math.dist(translated.state[3:6], aligned.state[3:6]) < 1e-9


class HistoryTarget:
    """Aircraft advances in world time; missiles query retained linear segments."""
    def __init__(self, launch_world_s, initial):
        self.launch_world_s = launch_world_s
        self.history = [(launch_world_s, initial)]
        self.queries = []
        self.observed = []

    def advance_aircraft(self, world_s):
        old_time, old = self.history[-1]
        assert world_s > old_time
        position = tuple(p + v * (world_s - old_time) for p, v in zip(old.position, old.velocity))
        self.history.append((world_s, TargetState(position, old.velocity)))

    def state_at(self, time_s):
        self.queries.append(time_s)
        world_s = self.launch_world_s + time_s
        assert self.history[0][0] <= world_s <= self.history[-1][0]
        for (left, old), (right, new) in zip(self.history, self.history[1:]):
            if left <= world_s <= right:
                fraction = (world_s - left) / (right - left)
                return TargetState(tuple(a + fraction * (b - a) for a, b in zip(old.position, new.position)),
                                   tuple(a + fraction * (b - a) for a, b in zip(old.velocity, new.velocity)))
        return self.history[0][1]

    def observe_missile(self, time_s, position, velocity):
        self.observed.append((time_s, position, velocity))


def test_common_clock_history_includes_within_tick_terminal_time(profiles):
    # Ground crossing needs a terminal-time query inside the already-saved aircraft segment.
    target = HistoryTarget(25., TargetState((10000., 8000., 0.), (-250., 0., 0.)))
    model = create_surface_missile(profiles["su_r_77_1"], launch_position_m=(100., 3., -50.),
                                   launch_velocity_mps=(300., -150., 0.), launch_pitch_deg=-30,
                                   launch_heading_deg=0, target=target, end_time_s=1,
                                   observation_mode="ideal_truth", loft=False)
    while not model.done:
        next_relative = min(model.time_s + model.dt, 1)
        target.advance_aircraft(target.launch_world_s + next_relative)
        model.step()
    assert model.event == "ground"
    saved_times = [world - target.launch_world_s for world, _ in target.history]
    assert 0 < model.time_s < model.dt
    assert model.time_s not in saved_times
    assert target.queries[-1] == model.time_s
    assert [x[0] for x in target.observed] == [x["time_s"] for x in model.result()["samples"]]


def test_interleaved_instances_match_isolated_and_leave_profiles_unchanged(profiles):
    before = copy.deepcopy(profiles)
    scene = scenario(max_simulation_time_s=2, target_constant_turn_g=5, target_azimuth_deg=30)
    names = ["us_aim_120c_5", "su_r_77_1"]
    expected = [runtime(profiles[name], scene).simulate(2) for name in names]
    models = [explicit(profiles[name], runtime(profiles[name], scene)) for name in names]
    while not all(m.done for m in models):
        for model in models:
            if not model.done: model.step()
    assert [m.result() for m in models] == expected
    assert profiles == before
    assert models[0].integral is not models[1].integral
    assert models[0].provider.inertial is not models[1].provider.inertial


def test_launcher_support_first_loss_permanent_and_public_api(profiles):
    profile = profiles["cn_pl12"]
    scene = scenario(initial_distance_m=50000, max_simulation_time_s=0.2, loft_enabled=False)
    calls = []
    def support(t, truth):
        calls.append((t, truth))
        return "launcher_turn" if t >= 2 / 48 else ""
    model = explicit(profile, runtime(profile, scene), launcher_support=support)
    finish(model)
    assert model.provider.datalink_lost_time_s == 2 / 48
    assert model.provider.datalink_lost_reason == "launcher_turn"
    assert not model.provider.datalink_connected
    assert [t for t, truth in calls] == [0, 1 / 48, 2 / 48]
    assert all(isinstance(truth, TargetState) for t, truth in calls)
    # Returning support later cannot reconnect: loss stops further callback queries.
    public = simulate(profile, scene, launcher_support=lambda t, truth: "launcher_turn" if t == 2 / 48 else "")
    assert public["samples"] == model.result()["samples"]
    assert public["summary"]["datalink_lost_time_s"] == 2 / 48
    assert public["summary"]["datalink_lost_reason"] == "launcher_turn"
    assert simulate(profile, scene)["samples"] == simulate(profile, scene, launcher_support=lambda t, truth: "")["samples"]


@pytest.mark.parametrize("options", [
    {"launcher_support": lambda t, truth: "", "launcher_radar_gimbal_deg": 60},
    {"launcher_support": 1},
    {"launcher_support": lambda t, truth: "", "model_mode": "legacy_h2"},
])
def test_public_support_rejects_incompatible_options(profiles, options):
    with pytest.raises(SimulationInputError):
        simulate(profiles["cn_pl12"], scenario(), **options)


def test_factory_sensor_options_match_public_and_decoys(profiles):
    profile = profiles["cn_pl12"]
    scene = scenario(target_altitude_m=5000, target_heading_deg=90, max_simulation_time_s=0.2)
    options = dict(clutter_model="look_down_angle", clutter_min_depression_deg=2,
                   cw_on_clear_beam=True, launcher_radar_gimbal_deg=60, require_seeker_lock=True,
                   multipath_gain=0.25)
    wrapping = chaffing_factory(None, ChaffSpec(rcs_ratio=2), ChaffProgram(0, interval_s=0.05, total=3))
    reference = runtime(profile, scene, wrapping)
    model = explicit(profile, reference, **options)
    assert finish(model)["samples"] == simulate(profile, scene, target_factory=wrapping, **options)["samples"]


def test_early_miss_is_opt_in(profiles):
    scene = scenario(target_heading_deg=180, target_speed_kmh=2000, initial_distance_m=50000, max_simulation_time_s=20)
    full = explicit(profiles["us_aim_120c_5"], runtime(profiles["us_aim_120c_5"], scene))
    early = explicit(profiles["us_aim_120c_5"], runtime(profiles["us_aim_120c_5"], scene), early_miss_s=2)
    finish(full)
    finish(early)
    assert early.event == "early_miss" and early.time_s < full.time_s
    assert full.event != "early_miss"


def test_lifecycle_and_endpoint_idempotence(profiles):
    model = runtime(profiles["cn_pl12"], scenario())
    assert model.time_s == 0 and not model.done and model.event is None
    with pytest.raises(RuntimeError, match="begin"): model.step()
    with pytest.raises(RuntimeError, match="termination"): model.result()
    model.begin(0.107)
    with pytest.raises(RuntimeError, match="already begun"): model.begin(0.107)
    with pytest.raises(RuntimeError, match="termination"): model.result()
    finish(model)
    before = copy.deepcopy(model.result())
    updates = model.provider.update_count
    state = model.state
    assert model.step() == state
    assert model.step() == state
    assert model.result() == before
    assert model.provider.update_count == updates
    initial = runtime(profiles["cn_pl12"], scenario(initial_distance_m=1))
    initial.begin(0.107)
    assert initial.done and initial.event == "fuse" and initial.time_s == 0
    assert len(initial.result()["samples"]) == 1


@pytest.mark.parametrize("change", [
    {"launch_position_m": (0, math.nan, 0)}, {"launch_velocity_mps": (1, 2)},
    {"target": object()}, {"end_time_s": math.inf}, {"end_time_s": -1},
    {"early_miss_s": 0}, {"launcher_support": lambda t, truth: "", "launcher_radar_gimbal_deg": 60},
    {"launcher_support": lambda t, truth: "", "observation_mode": "ideal_truth"},
])
def test_factory_validates_interface_contract(profiles, change):
    with pytest.raises(ValueError):
        explicit(profiles["cn_pl12"], runtime(profiles["cn_pl12"], scenario()), **change)


def test_callback_invalid_return_surfaces(profiles):
    with pytest.raises(TypeError, match="return a string"):
        explicit(profiles["cn_pl12"], runtime(profiles["cn_pl12"], scenario(initial_distance_m=50000)),
                 launcher_support=lambda t, truth: None)


def test_max_distance_terminal_matches_manual(profiles):
    profile = copy.deepcopy(profiles["su_r_77_1"])
    profile["performance"]["maximum_distance_m"] = 5
    scene = scenario()
    expected = runtime(profile, scene).simulate(0.107)
    manual = explicit(profile, runtime(profile, scene))
    assert finish(manual) == expected
    assert manual.event == "max_distance" and 0 < manual.time_s < manual.dt


def test_zero_duration_and_factory_lifetime_cap(profiles):
    scene = scenario()
    empty = explicit(profiles["cn_pl12"], runtime(profiles["cn_pl12"], scene), end_time_s=0)
    assert empty.done and empty.event == "lifetime" and empty.time_s == 0
    assert len(empty.result()["samples"]) == 1
    capped = explicit(profiles["cn_pl12"], runtime(profiles["cn_pl12"], scene), end_time_s=10000)
    assert capped._end_time == profiles["cn_pl12"]["performance"]["lifetime_s"]
