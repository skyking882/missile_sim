#!/usr/bin/env python3
"""Blind-run the standard scenario suite with the M0 strict-provenance variant.

M0 (config/profile_m0_strict.json) purges every REPLAY-FITTED runtime constant
identified in docs/M0_PROVENANCE_AUDIT.md from the shared H2 candidate runtime:
packed_lift_slope_scale -> 1.0, the midcourse lead-turn/launch-capture-handover
mechanism disabled entirely, the per-missile packed_lift_force_eta_law override
stripped, and the drag Mach curve rebuilt as the pure 1943-law shape at scale
x1.0 (no fitted global scale, no telemetry-anchored knots).

This script does not modify any shipped module.  It replicates -- minimally --
the same wiring missile_gui.library.scan_library() uses to attach a runtime
config to a missile profile dict (see that module's ``_load_one``), because
scan_library() hardcodes the path to config/profile_h2_runtime_defaults.json
and therefore cannot be pointed at the M0 variant.  The two extra M0-only
substitutions (dropping packed_lift_force_eta_law, replacing the Cx(M) table)
have no defaults-level knob in profile_adapter.py at all, so they are applied
here, after build_h2_candidate_config() returns, directly on the returned
config dict -- see config/profile_m0_strict.json's "m0_overrides" block for the
exact values and the reasoning.

Every scenario dict below is copied verbatim from tests/test_replay_anchors.py
(A1/A3/A5/A6/A7 for cn_pl12) or is that same A7 scenario with the missile swapped
to su_r_77_1 (the "R-77-1 level shot" case).  Nothing is tuned; this script only
runs the existing public_api.simulate() and reports what comes out.
"""

from __future__ import annotations

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

MISSILES_DIR = PROJECT_DIR / "missiles"
M0_DEFAULTS_PATH = PROJECT_DIR / "config" / "profile_m0_strict.json"


def load_m0_profile(missile_id: str) -> dict[str, Any]:
    """Load one missiles/*.json profile and attach an M0-adapted runtime config.

    Mirrors missile_gui.library._load_one()'s essential wiring (unsupported
    check -> build_h2_candidate_config -> attach _model_config), minus the
    multi-file scanning/error-collection/GUI-label machinery that module also
    does, and pointed at the M0 defaults file instead of the shipped one.
    """

    path = MISSILES_DIR / f"{missile_id}.json"
    profile = json.loads(path.read_text(encoding="utf-8"))
    unsupported = unsupported_model_types(profile)
    if unsupported:
        raise ValueError(f"{missile_id}: unsupported model types {unsupported}")
    defaults = load_runtime_defaults(str(M0_DEFAULTS_PATH))
    config, assumptions = build_h2_candidate_config(profile, defaults)

    overrides = defaults.get("m0_overrides", {})
    for key in overrides.get("strip_profile_aerodynamics_keys", []):
        config["aerodynamics"].pop(key, None)
    if "drag_model_cx_vs_mach" in overrides:
        config["drag_model"]["cx_vs_mach"] = [
            list(knot) for knot in overrides["drag_model_cx_vs_mach"]
        ]
        config["drag_model"]["cx_vs_mach_source"] = overrides.get(
            "drag_model_cx_vs_mach_source", "m0_override"
        )

    profile["_model_config"] = config
    profile["_runtime_assumptions"] = assumptions
    profile["_runtime_unsupported"] = []
    return profile


# --- Scenario dicts, copied verbatim from tests/test_replay_anchors.py -----

A1_35KM_LOFT = dict(
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

A3_30DEG_SLOW = dict(
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

A5_WINDUP_BUG = dict(
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

A6_STATSHARK_DOUBLE_FUSE = dict(
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

A7_LEVEL_STRAIGHT_TARGET = dict(
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


def _alpha_deg(sample: dict) -> float:
    return math.degrees(float(sample["angle_of_attack_rad"]))


def _actual_g(sample: dict) -> float:
    # Trajectory-normal (flow-normal) basis: matches the game's displayed G,
    # which includes the T*sin(alpha) thrust-lateral component. See
    # docs/M0_PROVENANCE_AUDIT.md, "trajectory-normal display basis".
    return float(sample.get("trajectory_lateral_load_g", 0.0))


def _speed_kmh(sample: dict) -> float:
    return math.hypot(*sample["velocity_mps"]) * 3.6


def _at_time(samples: list, time_s: float) -> dict:
    return min(samples, key=lambda sample: abs(float(sample["time_s"]) - time_s))


def _window(samples: list, lo_s: float, hi_s: float) -> list:
    return [s for s in samples if lo_s <= float(s["time_s"]) <= hi_s]


def run_a1(profiles: dict[str, Any]) -> dict[str, Any]:
    result = simulate(profiles["cn_pl12"], A1_35KM_LOFT)
    samples = result["samples"]
    summary = result["summary"]
    apex_m = max(float(s["position_m"][1]) for s in samples)
    terminal_speed_mps = math.hypot(*samples[-1]["velocity_mps"])
    mach_times = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    return {
        "scenario": "A1 35km head-on loft, cn_pl12",
        "flight_time_s": summary["flight_time_s"],
        "terminal_speed_mps": terminal_speed_mps,
        "apex_m": apex_m,
        "minimum_distance_m": summary["minimum_distance_m"],
        "termination_event": summary["termination_event"],
        "mach_by_time": {t: float(_at_time(samples, t)["mach"]) for t in mach_times},
    }


def run_a3(profiles: dict[str, Any]) -> dict[str, Any]:
    result = simulate(profiles["cn_pl12"], A3_30DEG_SLOW)
    samples = result["samples"]
    summary = result["summary"]
    times = [0.7, 1.2, 1.4, 2.1, 3.2]
    return {
        "scenario": "A3 30deg slow launch, cn_pl12",
        "alpha_deg_by_time": {t: _alpha_deg(_at_time(samples, t)) for t in times},
        "termination_event": summary["termination_event"],
        "flight_time_s": summary["flight_time_s"],
        "minimum_distance_m": summary["minimum_distance_m"],
    }


def run_a7_pl12(profiles: dict[str, Any]) -> dict[str, Any]:
    result = simulate(profiles["cn_pl12"], A7_LEVEL_STRAIGHT_TARGET)
    samples = result["samples"]
    summary = result["summary"]
    times = [2.9, 3.4, 3.9, 4.4]
    return {
        "scenario": "A7 level shot, cn_pl12",
        "g_by_time": {t: _actual_g(_at_time(samples, t)) for t in times},
        "alpha_deg_by_time": {t: _alpha_deg(_at_time(samples, t)) for t in times},
        "termination_event": summary["termination_event"],
        "flight_time_s": summary["flight_time_s"],
        "minimum_distance_m": summary["minimum_distance_m"],
    }


def run_a7_r77_1(profiles: dict[str, Any]) -> dict[str, Any]:
    result = simulate(profiles["su_r_77_1"], A7_LEVEL_STRAIGHT_TARGET)
    samples = result["samples"]
    summary = result["summary"]
    plateau_window = _window(samples, 0.9, 2.1)
    plateau_alpha = [_alpha_deg(s) for s in plateau_window]
    g_times = [1.1, 2.1, 3.4, 5.6, 6.9]
    speed_times = [2.3, 6.9]
    return {
        "scenario": "R-77-1 level shot (A7 scenario, su_r_77_1)",
        "alpha_plateau_window_deg": {
            "min": min(plateau_alpha) if plateau_alpha else None,
            "max": max(plateau_alpha) if plateau_alpha else None,
        },
        "g_by_time": {t: _actual_g(_at_time(samples, t)) for t in g_times},
        "alpha_deg_by_time": {t: _alpha_deg(_at_time(samples, t)) for t in g_times},
        "speed_kmh_by_time": {t: _speed_kmh(_at_time(samples, t)) for t in speed_times},
        "termination_event": summary["termination_event"],
        "flight_time_s": summary["flight_time_s"],
        "minimum_distance_m": summary["minimum_distance_m"],
    }


def run_a5(profiles: dict[str, Any]) -> dict[str, Any]:
    result = simulate(profiles["cn_pl12"], A5_WINDUP_BUG)
    summary = result["summary"]
    samples = result["samples"]
    peak_g = -1.0
    peak_time_s = 0.0
    for sample in samples:
        g = _actual_g(sample)
        if g > peak_g:
            peak_g = g
            peak_time_s = float(sample["time_s"])
    max_pitch_i = max(abs(float(s["pitch_pid_integral"])) for s in samples)
    max_yaw_i = max(abs(float(s["yaw_pid_integral"])) for s in samples)
    return {
        "scenario": "A5 windup-bug case, cn_pl12",
        "termination_event": summary["termination_event"],
        "minimum_distance_m": summary["minimum_distance_m"],
        "flight_time_s": summary["flight_time_s"],
        "peak_g": peak_g,
        "peak_g_time_s": peak_time_s,
        "max_abs_pid_integral_rad": max(max_pitch_i, max_yaw_i),
    }


def run_a6(profiles: dict[str, Any]) -> dict[str, Any]:
    pl12 = simulate(profiles["cn_pl12"], A6_STATSHARK_DOUBLE_FUSE)["summary"]
    r77 = simulate(profiles["su_r_77"], A6_STATSHARK_DOUBLE_FUSE)["summary"]
    return {
        "scenario": "A6 statshark 40deg 8km double-fuse, cn_pl12 + su_r_77",
        "cn_pl12": {
            "termination_event": pl12["termination_event"],
            "minimum_distance_m": pl12["minimum_distance_m"],
            "flight_time_s": pl12["flight_time_s"],
        },
        "su_r_77": {
            "termination_event": r77["termination_event"],
            "minimum_distance_m": r77["minimum_distance_m"],
            "flight_time_s": r77["flight_time_s"],
        },
    }


def main() -> int:
    profiles = {
        missile_id: load_m0_profile(missile_id)
        for missile_id in ("cn_pl12", "su_r_77", "su_r_77_1")
    }
    for missile_id, profile in profiles.items():
        cfg = profile["_model_config"]
        assert cfg["aerodynamics"].get("packed_lift_force_eta_law") is None, missile_id
        assert abs(cfg["aerodynamics"]["packed_lift_slope_scale"] - 1.0) < 1e-12, missile_id
        assert cfg["guidance"]["midcourse"]["enabled"] is False, missile_id
        assert len(cfg["drag_model"]["cx_vs_mach"]) == 10, missile_id

    report = {
        "m0_defaults_path": str(M0_DEFAULTS_PATH),
        "results": {
            "A1": run_a1(profiles),
            "A3": run_a3(profiles),
            "A7_pl12": run_a7_pl12(profiles),
            "A7_r77_1": run_a7_r77_1(profiles),
            "A5": run_a5(profiles),
            "A6": run_a6(profiles),
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if len(sys.argv) > 1 and sys.argv[1] == "--json-out":
        out_path = Path(sys.argv[2])
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nwritten: {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
