"""Frozen replay-anchor acceptance suite for the v12 / IOG control loop.

Every numeric target below is digitized from 2026-08 War Thunder game
replays (StatShark G/alpha telemetry read against the matching launch
geometry), collected across three review rounds while fixing the v12
(`pid_output_semantics="fin_angle_rad"`) integrator-windup bug and the IOG
midcourse-fin-routing candidate built on top of it:

  A1  35 km head-on + loft, cn_pl12                  (2026-08b regression)
  A2  30 deg fast launch, cn_pl12                     (2026-08c regression)
  A3  30 deg slow launch, cn_pl12                     (2026-08a baseline)
  A4  30 deg slow launch, su_r_77                     (2026-08a baseline)
  A5  original windup bug case, cn_pl12               (2026-08a bug repro)
  A6  statshark 8 km 40 deg double-fuse, cn_pl12+su_r_77 (pre-existing;
      see test_rate_inner_fin_torque.py for the underlying test)

This file exists so every future change to the v12/IOG control path keeps
the whole anchor set green in one run.  Do not loosen a tolerance to make a
change pass -- fix the change, or report the conflict instead.

A2 is the one exception: extensive 2026-08c investigation (multiple
tracking-mode formula placements and clamp values, all converging to the
same result) found that even round 1's exact, unclamped tracking formula
does not reach the A2 targets.  At the reported peak the fin is at ~80% of
travel (not saturated) and the integral already sits in the 0.30 rad range
the fast-missile unload arc is supposed to need, yet achieved G still falls
far short of the replay.  Corrected diagnosis (2026-08d): NOT an authority
ceiling -- the packed-lift law reproduces the replay's own operating point
(alpha=13.8 deg, eta=1.27 -> 24.2 g vs replay 23.5 g, 3%).  The shortfall is
the post-handover guidance command profile (model PN commands decay faster
than the game's effective command), convolved with target-course
reconstruction uncertainty: the replay's target was climbing/maneuvering
while this anchor scenario assumes straight level flight, so the G(t)
timeline is not a clean plant anchor.  The per-frame G-alpha-eta relation
from this same dataset remains valid (it is the basis of the lift law).
at this speed/altitude rather than a control-loop integrator problem, which
is out of this file's reach (h2_dynamics.py lift/drag and
packed_lift_slope_scale are off-limits).  A2 is marked ``xfail(strict=True)``
so it fails loudly -- prompting removal of the marker -- if a future change
ever makes it pass.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from aim120_model.public_api import simulate
from missile_gui.library import scan_library

ROOT = Path(__file__).resolve().parents[1]


def _indexed_profiles() -> dict:
    profiles, errors = scan_library(ROOT / "missiles", ROOT)
    assert errors == []
    return {profile["missile_id"]: profile for profile in profiles}


def _alpha_deg(sample: dict) -> float:
    return math.degrees(float(sample["angle_of_attack_rad"]))


def _actual_g(sample: dict) -> float:
    return float(sample.get("actual_overload_g", 0.0))


def _at_time(samples: list, time_s: float) -> dict:
    return min(samples, key=lambda sample: abs(float(sample["time_s"]) - time_s))


def test_a1_35km_head_on_loft_anchor() -> None:
    """cn_pl12, 35 km head-on with loft: TOF / terminal speed / apex / miss."""
    indexed = _indexed_profiles()
    scenario = dict(
        loft_enabled=True,
        observation_mode="ideal_truth",
        target_course_reference="statshark_relative_to_los",
        launch_speed_kmh=1188,
        launch_altitude_m=7000,
        launch_pitch_deg=0,
        launch_heading_deg=0,
        target_speed_kmh=1188,
        target_altitude_m=7000,
        initial_distance_m=35000,
        target_azimuth_deg=0,
        target_heading_deg=0,
        target_vertical_heading_deg=0,
        target_constant_turn_g=0,
        max_simulation_time_s=90.0,
    )
    result = simulate(indexed["cn_pl12"], scenario)
    samples = result["samples"]
    summary = result["summary"]
    apex_m = max(float(sample["position_m"][1]) for sample in samples)
    terminal_speed_mps = math.hypot(*samples[-1]["velocity_mps"])
    assert abs(summary["flight_time_s"] - 33.26) <= 0.4
    assert abs(terminal_speed_mps - 487.0) <= 10.0
    assert abs(apex_m - 8845.0) <= 90.0
    assert abs(summary["minimum_distance_m"] - 10.0) <= 0.5


@pytest.mark.xfail(
    strict=True,
    reason=(
        "2026-08c: unresolved gap, investigated not force-fit -- see module "
        "docstring. Fin reaches only ~80% of travel (not saturated) with the "
        "integral already in the targeted 0.30 rad range, yet achieved G "
        "still falls far short of the replay; corrected diagnosis: post-handover "
        "guidance command profile + target-course reconstruction uncertainty "
        "(replay target was maneuvering; the plant law itself is verified at the "
        "replay's own operating point: alpha=13.8deg, eta=1.27 -> 24.2g vs 23.5g). "
        "Not an authority ceiling and not a control-loop integrator problem."
    ),
)
def test_a2_30deg_fast_pl12_anchor() -> None:
    """cn_pl12, 30 deg fast launch with loft: G/alpha at 4 anchor times."""
    indexed = _indexed_profiles()
    scenario = dict(
        loft_enabled=True,
        observation_mode="ideal_truth",
        target_course_reference="statshark_relative_to_los",
        launch_speed_kmh=1508,
        launch_altitude_m=6783,
        launch_pitch_deg=0,
        launch_heading_deg=0,
        target_speed_kmh=1549,
        target_altitude_m=6783,
        initial_distance_m=8000,
        target_azimuth_deg=30,
        target_heading_deg=0,
        target_vertical_heading_deg=0,
        target_constant_turn_g=0,
        max_simulation_time_s=40.0,
    )
    result = simulate(indexed["cn_pl12"], scenario)
    samples = result["samples"]
    game_g = {2.2: 23.5, 3.2: 15.5, 4.2: 7.1, 5.1: 1.1}
    game_alpha = {2.2: 13.8, 3.2: 8.8, 4.2: 3.8, 5.1: 0.8}
    for time_s, game_value in game_g.items():
        sample = _at_time(samples, time_s)
        assert abs(_actual_g(sample) - game_value) <= 5.0
    for time_s, game_value in game_alpha.items():
        sample = _at_time(samples, time_s)
        assert abs(_alpha_deg(sample) - game_value) <= 3.5


def test_a3_30deg_slow_pl12_anchor() -> None:
    """cn_pl12, 30 deg slow launch with loft: alpha at 5 anchor times."""
    indexed = _indexed_profiles()
    scenario = dict(
        loft_enabled=True,
        launch_speed_kmh=1200,
        launch_altitude_m=6200,
        launch_pitch_deg=0,
        launch_heading_deg=0,
        target_speed_kmh=1550,
        target_altitude_m=6200,
        initial_distance_m=8000,
        target_azimuth_deg=30,
        target_heading_deg=0,
        target_vertical_heading_deg=0,
        target_constant_turn_g=0,
        max_simulation_time_s=15.0,
    )
    result = simulate(indexed["cn_pl12"], scenario)
    samples = result["samples"]
    game_alpha = {0.7: 11.5, 1.2: 14.1, 1.4: 15.2, 2.1: 10.4, 3.2: 5.1}
    for time_s, game_value in game_alpha.items():
        sample = _at_time(samples, time_s)
        assert abs(_alpha_deg(sample) - game_value) <= 3.0


def test_a4_r77_slow_alpha_peak_anchor() -> None:
    """su_r_77, same slow-launch geometry as A3: alpha peak in [0.8, 2.2]s."""
    indexed = _indexed_profiles()
    scenario = dict(
        loft_enabled=True,
        launch_speed_kmh=1500,
        launch_altitude_m=6200,
        launch_pitch_deg=0,
        launch_heading_deg=0,
        target_speed_kmh=1550,
        target_altitude_m=6200,
        initial_distance_m=8000,
        target_azimuth_deg=30,
        target_heading_deg=0,
        target_vertical_heading_deg=0,
        target_constant_turn_g=0,
        max_simulation_time_s=15.0,
    )
    result = simulate(indexed["su_r_77"], scenario)
    samples = result["samples"]
    window = [_alpha_deg(sample) for sample in samples if 0.8 <= float(sample["time_s"]) <= 2.2]
    assert max(window) >= 18.0


def test_a5_user_bug_case_anchor() -> None:
    """cn_pl12, the original mid-course windup bug repro (loft off)."""
    indexed = _indexed_profiles()
    scenario = dict(
        loft_enabled=False,
        observation_mode="ideal_truth",
        target_course_reference="statshark_relative_to_los",
        launch_speed_kmh=1100,
        launch_altitude_m=6500,
        launch_pitch_deg=0,
        launch_heading_deg=0,
        target_speed_kmh=1200,
        target_altitude_m=6500,
        initial_distance_m=8000,
        target_azimuth_deg=30,
        target_heading_deg=-30,
        target_vertical_heading_deg=0,
        target_constant_turn_g=0,
        max_simulation_time_s=40.0,
    )
    result = simulate(indexed["cn_pl12"], scenario)
    samples = result["samples"]
    summary = result["summary"]
    peak_g = -1.0
    peak_time_s = 0.0
    for sample in samples:
        g = _actual_g(sample)
        if g > peak_g:
            peak_g = g
            peak_time_s = float(sample["time_s"])
    assert peak_time_s <= 2.7
    max_pitch_i = max(abs(float(sample["pitch_pid_integral"])) for sample in samples)
    max_yaw_i = max(abs(float(sample["yaw_pid_integral"])) for sample in samples)
    assert max(max_pitch_i, max_yaw_i) <= 0.32
    antiphase_count = 0
    for sample in samples:
        time_s = float(sample["time_s"])
        if time_s <= 4.0:
            continue
        cmd_pitch = float(sample["commanded_acceleration_g"][0])
        act_pitch = float(sample.get("body_axis_pitch_specific_force_g", 0.0))
        if abs(cmd_pitch) > 1.0 and abs(act_pitch) > 1.0 and cmd_pitch * act_pitch < 0.0:
            antiphase_count += 1
    assert antiphase_count == 0
    assert abs(summary["minimum_distance_m"] - 10.0) <= 0.5


def test_a6_statshark_40deg_double_fuse_anchor() -> None:
    """cn_pl12 + su_r_77, 8 km 40 deg straight-x: both must still fuse."""
    indexed = _indexed_profiles()
    scenario = dict(
        launch_speed_kmh=1200.0,
        launch_altitude_m=7500.0,
        launch_pitch_deg=0.0,
        launch_heading_deg=0.0,
        target_speed_kmh=900.0,
        target_altitude_m=7500.0,
        initial_distance_m=8000.0,
        target_azimuth_deg=40.0,
        target_heading_deg=-40.0,
        target_course_reference="statshark_relative_to_los",
        target_vertical_heading_deg=0.0,
        target_constant_turn_g=0.0,
        max_simulation_time_s=40.0,
        loft_enabled=True,
    )
    pl12 = simulate(indexed["cn_pl12"], scenario)["summary"]
    r77 = simulate(indexed["su_r_77"], scenario)["summary"]
    assert pl12["termination_event"] == "proximity_fuse"
    assert pl12["minimum_distance_m"] < 15.0
    assert r77["termination_event"] == "proximity_fuse"
    assert r77["minimum_distance_m"] < 15.0
