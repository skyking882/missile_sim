#!/usr/bin/env python3
"""CN20 closed-loop 2x2 exam: fixed-CN lift law x momentum-tilt induced drag.

Background (see docs/CN20_CLOSED_LOOP.md for the full write-up): the
instantaneous-mass CN_alpha fit for the R-77-1 gives ~19.9 /rad (caliber
reference area), constant across M1.35-3.56, which is the signature of a
fixed-coefficient standard-aero force law rather than the shipped
acceleration-semantics packed-lift law (`packed_lift_slope_scale` /
`packed_lift_force_eta_law`).  Separately, `docs/ALPHA_PLAYBACK_R77_1.md`'s
open-loop MODE 2 diagnostic twice found that a momentum-tilt induced-drag
term (`D_i = L_aero*tan(alpha)`) tracks the replay's measured speed far
better than the shipped `q*S_d*cx_vs_aoa*alpha^2` term.

This script is a CLOSED-LOOP exam (guidance + control + rotation all active,
unlike the open-loop alpha-playback harness) of the 2x2 combination of these
two candidates against the R-77-1 level-shot replay
(`data/replays/r77_1_level_20260824.tsv`, 46 frames).  It is an EXAM
HARNESS, not a fitting script: CN_alpha is fixed at 19.9 and never adjusted
based on the results printed here.

The two experimental switches are config-gated additions to
`aim120_model.h2_dynamics.forces_for_state_h2` (force channel only; the
moment/rotation channel is untouched):

  - `aerodynamics.packed_lift_fixed_cn = {"cn_alpha_per_rad": 19.9}`
    replaces the force-channel slope with a fixed-coefficient standard-aero
    law: aero lateral G per radian of alpha = CN_alpha*q*S_d/(m*g).
  - `aerodynamics.induced_drag_mode = "momentum_tilt"` zeroes the shipped
    cx_vs_aoa alpha^2 drag term and instead adds an along-velocity
    retarding force equal to the packed channel's aero lateral force times
    tan(alpha).

Both keys are read from `missiles/su_r_77_1.json`'s `aerodynamics` block by
`aim120_model.profile_adapter.build_h2_candidate_config` (mirroring the
existing `packed_lift_force_eta_law` per-missile-override pattern) -- see
that module for the pass-through.  Absent key -> shipped behavior,
bit-identical.

For the four variants below, the profile JSON and runtime-defaults JSON are
loaded the standard way (see scripts/run_m0_scenarios.py and
scripts/alpha_playback.py) and then deep-copied IN MEMORY per variant before
injecting the experimental aerodynamics keys; `missiles/su_r_77_1.json` and
`config/profile_h2_runtime_defaults.json` on disk are never written to.

This script performs no filesystem writes; it only prints to stdout.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from aim120_model.profile_adapter import (  # noqa: E402
    build_h2_candidate_config,
    load_runtime_defaults,
    unsupported_model_types,
)
from aim120_model.public_api import simulate  # noqa: E402

MISSILE_PATH = PROJECT_DIR / "missiles" / "su_r_77_1.json"
DEFAULTS_PATH = PROJECT_DIR / "config" / "profile_h2_runtime_defaults.json"
REPLAY_PATH = PROJECT_DIR / "data" / "replays" / "r77_1_level_20260824.tsv"

FIXED_CN_ALPHA_PER_RAD = 19.9

# Copied verbatim from scripts/run_m0_scenarios.py's A7_LEVEL_STRAIGHT_TARGET
# (same scenario tests/test_replay_anchors.py's A7 anchor and
# run_a7_r77_1() use for su_r_77_1) -- the mission-specified level-shot exam
# geometry.  Nothing here is tuned.
SCENARIO: dict[str, Any] = dict(
    loft_enabled=True,
    observation_mode="ideal_truth",
    target_course_reference="statshark_relative_to_los",
    launch_speed_kmh=1200, launch_altitude_m=6300,
    launch_pitch_deg=0, launch_heading_deg=0,
    target_speed_kmh=1200, target_altitude_m=6300,
    initial_distance_m=8000, target_azimuth_deg=30,
    target_heading_deg=0, target_vertical_heading_deg=0,
    target_constant_turn_g=0, max_simulation_time_s=40.0,
)

DELOAD_WINDOW_S = (2.3, 4.9)
ENDGAME_WINDOW_S = (5.5, 7.1)
ALPHA_RELEASE_THRESHOLD_DEG = 17.0

# The replay is a sparse ~0.15-0.25s-spaced 46-frame table; "first tabulated
# row below 17deg after the peak" on that coarse grid lands at t=2.7s (game
# alpha: ...,2.5->18.4, 2.7->16.3), not 2.1s.  The ~2.1s reference below is
# instead the launch-capture alpha-plateau end already established in
# src/aim120_model/profile_adapter.py's midcourse_lead_turn.blend_time_s
# assumption note and docs/M0_RESIDUALS.md's R-77-1 table ("Alpha plateau
# window (t=0.9-2.1s)"): the platform (~22-23.7deg) visibly holds until
# about 2.1s in the replay before a sustained decline begins.  It is a fixed
# reference value taken from that prior analysis, not re-derived here by
# applying the 17deg rule to the coarse replay grid.  The 17deg rule IS
# applied, consistently, to each variant's own dense (dt=0.02s) model curve
# below, which is the fair way to compare a continuous simulated curve
# against this reference.
GAME_ALPHA_RELEASE_S = 2.1


def load_base_profile() -> dict[str, Any]:
    profile = json.loads(MISSILE_PATH.read_text(encoding="utf-8"))
    unsupported = unsupported_model_types(profile)
    if unsupported:
        raise ValueError(f"su_r_77_1: unsupported model types {unsupported}")
    return profile


def build_variant_profile(
    base_profile: dict[str, Any],
    base_defaults: dict[str, Any],
    *,
    fixed_cn: bool,
    momentum_tilt: bool,
) -> dict[str, Any]:
    """Deep-copy profile/defaults in memory and inject the Step-1 keys.

    Mirrors the essential wiring scripts/run_m0_scenarios.py's
    load_m0_profile() uses (build_h2_candidate_config -> attach
    _model_config/_runtime_assumptions/_runtime_unsupported), except the
    injected keys live on the *profile*'s aerodynamics block (same place
    packed_lift_force_eta_law already lives), matching the per-missile
    override pattern build_h2_candidate_config already implements.
    """

    profile = copy.deepcopy(base_profile)
    defaults = copy.deepcopy(base_defaults)
    if fixed_cn:
        profile["aerodynamics"]["packed_lift_fixed_cn"] = {
            "cn_alpha_per_rad": FIXED_CN_ALPHA_PER_RAD
        }
    if momentum_tilt:
        profile["aerodynamics"]["induced_drag_mode"] = "momentum_tilt"
    config, assumptions = build_h2_candidate_config(profile, defaults)
    profile["_model_config"] = config
    profile["_runtime_assumptions"] = assumptions
    profile["_runtime_unsupported"] = []
    return profile


VARIANTS: list[tuple[str, bool, bool]] = [
    ("V1_shipped", False, False),
    ("V2_shipped_momentum_tilt", False, True),
    ("V3_fixed_cn19p9", True, False),
    ("V4_fixed_cn19p9_momentum_tilt", True, True),
]


# --------------------------------------------------------------------------
# Replay loading and shared sample accessors
# --------------------------------------------------------------------------

def load_replay_rows() -> list[dict[str, str]]:
    with REPLAY_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
    if len(rows) != 46:
        raise ValueError(f"expected 46 replay frames, found {len(rows)}")
    return rows


def find_row_by_time(rows: list[dict[str, str]], target_t: float) -> dict[str, str]:
    best = min(rows, key=lambda r: abs(float(r["missile_flight_time_s"]) - target_t))
    if abs(float(best["missile_flight_time_s"]) - target_t) > 1e-6:
        raise ValueError(f"no exact replay frame at t={target_t}")
    return best


def _alpha_deg(sample: dict) -> float:
    return math.degrees(float(sample["angle_of_attack_rad"]))


def _g(sample: dict) -> float:
    # trajectory-normal (flow-normal) basis, including the T*sin(alpha)
    # thrust-lateral component -- matches how the game displays G (see
    # docs/UPDATE_V1.0.2.md's "G-basis correction" note and
    # tests/test_replay_anchors.py's _actual_g()).
    return float(sample.get("trajectory_lateral_load_g", 0.0))


def _speed_kmh(sample: dict) -> float:
    return math.hypot(*sample["velocity_mps"]) * 3.6


def _at_time(samples: list, time_s: float) -> dict:
    return min(samples, key=lambda sample: abs(float(sample["time_s"]) - time_s))


def alpha_release_time_s(samples: list) -> float | None:
    """First sample time, scanning forward from the global alpha peak, where
    alpha drops below ALPHA_RELEASE_THRESHOLD_DEG.  None if it never does."""

    peak_index = max(range(len(samples)), key=lambda i: _alpha_deg(samples[i]))
    for sample in samples[peak_index:]:
        if _alpha_deg(sample) < ALPHA_RELEASE_THRESHOLD_DEG:
            return float(sample["time_s"])
    return None


# --------------------------------------------------------------------------
# Per-variant run + summary
# --------------------------------------------------------------------------

def run_variant(name: str, profile: dict[str, Any]) -> dict[str, Any]:
    result = simulate(profile, SCENARIO)
    samples = result["samples"]
    summary = result["summary"]

    peak_index = max(range(len(samples)), key=lambda i: _alpha_deg(samples[i]))
    peak_alpha_deg = _alpha_deg(samples[peak_index])
    peak_alpha_time_s = float(samples[peak_index]["time_s"])
    release_t = alpha_release_time_s(samples)

    late_arc_g = [_g(s) for s in samples if float(s["time_s"]) > 5.5]
    late_arc_peak_g = max(late_arc_g) if late_arc_g else float("nan")
    late_arc_peak_g_time_s = (
        float(next(s for s in samples if _g(s) == late_arc_peak_g and float(s["time_s"]) > 5.5)["time_s"])
        if late_arc_g
        else float("nan")
    )

    g_at = {t: _g(_at_time(samples, t)) for t in (2.1, 3.4, 6.9)}
    speed_at = {t: _speed_kmh(_at_time(samples, t)) for t in (2.3, 4.3, 6.9)}

    return {
        "name": name,
        "samples": samples,
        "summary": summary,
        "peak_alpha_deg": peak_alpha_deg,
        "peak_alpha_time_s": peak_alpha_time_s,
        "alpha_release_time_s": release_t,
        "late_arc_peak_g": late_arc_peak_g,
        "late_arc_peak_g_time_s": late_arc_peak_g_time_s,
        "g_at": g_at,
        "speed_at": speed_at,
        "termination_event": summary["termination_event"],
        "minimum_distance_m": summary["minimum_distance_m"],
        "flight_time_s": summary["flight_time_s"],
    }


def mean_abs_delta_g(
    samples: list, rows: list[dict[str, str]], window: tuple[float, float]
) -> tuple[float, int]:
    lo, hi = window
    deltas = []
    for row in rows:
        t = float(row["missile_flight_time_s"])
        if lo <= t <= hi:
            game_g = float(row["overload_g"])
            model_g = _g(_at_time(samples, t))
            deltas.append(abs(model_g - game_g))
    if not deltas:
        return float("nan"), 0
    return sum(deltas) / len(deltas), len(deltas)


# --------------------------------------------------------------------------
# Printing
# --------------------------------------------------------------------------

def print_summary_table(rows: list[dict[str, str]], variant_results: list[dict[str, Any]]) -> None:
    game_peak_alpha = max(float(r["angle_of_attack_deg"]) for r in rows)
    game_late_arc_g = max(
        float(r["overload_g"]) for r in rows if float(r["missile_flight_time_s"]) > 5.5
    )
    game_g_at = {t: float(find_row_by_time(rows, t)["overload_g"]) for t in (2.1, 3.4, 6.9)}
    game_speed_at = {t: float(find_row_by_time(rows, t)["speed_kmh"]) for t in (2.3, 4.3, 6.9)}

    print("game truth (data/replays/r77_1_level_20260824.tsv):")
    print(f"  alpha-release time (17deg threshold reference)   ~= {GAME_ALPHA_RELEASE_S:.1f} s")
    print(f"  peak alpha                                        = {game_peak_alpha:.1f} deg")
    print(f"  late-arc peak G (t>5.5s)                           = {game_late_arc_g:.1f}")
    print(f"  G at t=2.1/3.4/6.9 s                                = "
          f"{game_g_at[2.1]:.1f} / {game_g_at[3.4]:.1f} / {game_g_at[6.9]:.1f}")
    print(f"  speed(km/h) at t=2.3/4.3/6.9 s                      = "
          f"{game_speed_at[2.3]:.0f} / {game_speed_at[4.3]:.0f} / {game_speed_at[6.9]:.0f}")
    print("  termination                                        = proximity fuse, ~7.2-7.4 s")
    print()

    header = (
        f"{'variant':30} {'rel_t_s':>8} {'pk_a_deg':>9} {'la_pk_G':>8} "
        f"{'G@2.1':>7} {'G@3.4':>7} {'G@6.9':>7} "
        f"{'v@2.3':>7} {'v@4.3':>7} {'v@6.9':>7} "
        f"{'term':>16} {'min_d_m':>8} {'t_flight':>9}"
    )
    print(header)
    for r in variant_results:
        rel_t = "none" if r["alpha_release_time_s"] is None else f"{r['alpha_release_time_s']:.2f}"
        print(
            f"{r['name']:30} {rel_t:>8} {r['peak_alpha_deg']:9.2f} {r['late_arc_peak_g']:8.2f} "
            f"{r['g_at'][2.1]:7.2f} {r['g_at'][3.4]:7.2f} {r['g_at'][6.9]:7.2f} "
            f"{r['speed_at'][2.3]:7.0f} {r['speed_at'][4.3]:7.0f} {r['speed_at'][6.9]:7.0f} "
            f"{r['termination_event']:>16} {r['minimum_distance_m']:8.2f} {r['flight_time_s']:9.3f}"
        )


def print_window_error_table(
    rows: list[dict[str, str]], variant_results: list[dict[str, Any]]
) -> dict[str, tuple[float, float]]:
    print()
    print(
        f"{'variant':30} {'deload[2.3,4.9] mean|dG|':>26} (n) "
        f"{'endgame[5.5,7.1] mean|dG|':>27} (n)"
    )
    per_variant_windows: dict[str, tuple[float, float]] = {}
    for r in variant_results:
        deload_mean, deload_n = mean_abs_delta_g(r["samples"], rows, DELOAD_WINDOW_S)
        endgame_mean, endgame_n = mean_abs_delta_g(r["samples"], rows, ENDGAME_WINDOW_S)
        per_variant_windows[r["name"]] = (deload_mean, endgame_mean)
        print(
            f"{r['name']:30} {deload_mean:26.3f} ({deload_n:2d}) "
            f"{endgame_mean:27.3f} ({endgame_n:2d})"
        )
    return per_variant_windows


def print_best_variant_table(
    rows: list[dict[str, str]],
    variant_results: list[dict[str, Any]],
    per_variant_windows: dict[str, tuple[float, float]],
) -> str:
    # Objective, pre-declared selection rule: lowest combined (unweighted
    # sum) mean|dG| across the de-load and endgame windows.  Not chosen by
    # eye after inspecting the per-frame numbers below.
    best_name = min(
        per_variant_windows, key=lambda name: sum(per_variant_windows[name])
    )
    best = next(r for r in variant_results if r["name"] == best_name)
    combined = sum(per_variant_windows[best_name])
    print()
    print(
        f"best variant by combined mean|dG| (de-load + endgame) = {best_name} "
        f"(combined={combined:.3f})"
    )
    print()
    print(
        f"{'id':5} {'t':>6} {'G_game':>7} {'G_mod':>7} {'dG':>6} "
        f"{'a_game':>7} {'a_mod':>7} {'v_game':>8} {'v_mod':>8} {'dv':>7}"
    )
    for row in rows:
        t = float(row["missile_flight_time_s"])
        sample = _at_time(best["samples"], t)
        game_g = float(row["overload_g"])
        model_g = _g(sample)
        game_alpha = float(row["angle_of_attack_deg"])
        model_alpha = _alpha_deg(sample)
        game_v = float(row["speed_kmh"])
        model_v = _speed_kmh(sample)
        print(
            f"{row['sample_id']:5} {t:6.3f} {game_g:7.2f} {model_g:7.2f} {model_g - game_g:6.2f} "
            f"{game_alpha:7.1f} {model_alpha:7.1f} {game_v:8.1f} {model_v:8.1f} {model_v - game_v:7.1f}"
        )
    return best_name


def main() -> int:
    base_profile = load_base_profile()
    base_defaults = load_runtime_defaults(str(DEFAULTS_PATH))
    rows = load_replay_rows()

    print("=" * 90)
    print("CN20 CLOSED-LOOP EXAM -- R-77-1 level shot, 2x2 fixed_cn x momentum_tilt")
    print(f"CN_alpha fixed at {FIXED_CN_ALPHA_PER_RAD} (caliber reference area); not tuned.")
    print("=" * 90)

    variant_results = []
    for name, fixed_cn, momentum_tilt in VARIANTS:
        profile = build_variant_profile(
            base_profile, base_defaults, fixed_cn=fixed_cn, momentum_tilt=momentum_tilt
        )
        variant_results.append(run_variant(name, profile))

    print()
    print("-" * 90)
    print("(i) per-variant summary")
    print("-" * 90)
    print_summary_table(rows, variant_results)

    print()
    print("-" * 90)
    print("(ii) mean |delta G| over de-load [2.3,4.9] s and endgame [5.5,7.1] s windows")
    print("-" * 90)
    per_variant_windows = print_window_error_table(rows, variant_results)

    print()
    print("-" * 90)
    print("(iii) per-frame table, best variant")
    print("-" * 90)
    best_name = print_best_variant_table(rows, variant_results, per_variant_windows)

    print()
    print("=" * 90)
    print(f"End of exam output. Best variant by combined window error: {best_name}")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
