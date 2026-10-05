#!/usr/bin/env python3
"""Sweep evasion start time against launch range and report escape windows.

For each initial range and evasion kind, runs the general runtime with a
closed-loop ``EvasiveTarget`` starting at each grid time. An escape is any
termination other than the proximity fuse. Windows are contiguous runs of
escaping start times at the grid resolution; they inherit every limitation of
the kinematic target (no energy loss, roll-in, altitude change or chaff).

Writes a CSV under outputs/escape_window/ and prints the windows.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aim120_model.evasive_target import KINDS, EvasionPlan, evasive_target_factory  # noqa: E402
from aim120_model.profile_catalog import load_profile_catalog  # noqa: E402
from aim120_model.public_api import simulate  # noqa: E402
from missile_gui.library import scan_library  # noqa: E402

_PROFILE = None


def _load(missile_id):
    _, catalog = load_profile_catalog(ROOT)
    profiles, errors = scan_library(catalog["profiles_dir"], ROOT)
    found = {p["missile_id"]: p for p in profiles}
    if missile_id not in found:
        raise SystemExit(f"unknown missile {missile_id}; library errors: {errors[:3]}")
    return found[missile_id]


def _init(missile_id):
    global _PROFILE
    _PROFILE = _load(missile_id)


def _scenario(args, range_m):
    return dict(launch_speed_kmh=args.launch_speed_kmh, launch_altitude_m=args.launch_altitude_m,
                launch_pitch_deg=0, launch_heading_deg=0, target_speed_kmh=args.target_speed_kmh,
                target_altitude_m=args.target_altitude_m, initial_distance_m=range_m,
                target_azimuth_deg=0, target_heading_deg=args.target_course_deg,
                target_course_reference="relative_to_los", target_vertical_heading_deg=0,
                target_constant_turn_g=0, max_simulation_time_s=args.max_time_s,
                observation_mode=args.observation_mode, loft_enabled=args.loft)


def _run(job):
    scenario, kind, start, turn_g = job
    factory = None if kind == "none" else evasive_target_factory(EvasionPlan(kind, start, turn_g))
    summary = simulate(_PROFILE, scenario, target_factory=factory)["summary"]
    return dict(range_m=scenario["initial_distance_m"], kind=kind, start_s=start,
                event=summary["termination_event"], flight_time_s=round(summary["flight_time_s"], 3),
                min_distance_m=round(summary["minimum_distance_m"], 1),
                escaped=summary["termination_event"] != "proximity_fuse",
                radar_reject=summary.get("last_radar_reject_reason"))


def _windows(rows):
    """Contiguous escaping start times -> [(first, last)]."""
    out, current = [], None
    for row in sorted(rows, key=lambda r: r["start_s"]):
        if row["escaped"]:
            current = [row["start_s"], row["start_s"]] if current is None else [current[0], row["start_s"]]
        elif current is not None:
            out.append(tuple(current))
            current = None
    if current is not None:
        out.append(tuple(current))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--missile", default="us_aim_120c_5")
    parser.add_argument("--ranges-km", default="10,15,20,25,30,40")
    parser.add_argument("--kinds", default=",".join(KINDS))
    parser.add_argument("--start-max-s", type=float, default=40.0)
    parser.add_argument("--start-step-s", type=float, default=1.0)
    parser.add_argument("--turn-g", type=float, default=9.0)
    parser.add_argument("--launch-altitude-m", type=float, default=8000.0)
    parser.add_argument("--launch-speed-kmh", type=float, default=1100.0)
    parser.add_argument("--target-altitude-m", type=float, default=8000.0)
    parser.add_argument("--target-speed-kmh", type=float, default=1000.0)
    parser.add_argument("--target-course-deg", type=float, default=0.0,
                        help="pre-evasion course relative to LOS; 0 = head-on")
    parser.add_argument("--observation-mode", default="sensor_track", choices=("sensor_track", "ideal_truth"))
    parser.add_argument("--loft", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-time-s", type=float, default=120.0)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    ranges = [float(x) * 1000 for x in args.ranges_km.split(",")]
    kinds = [k for k in args.kinds.split(",") if k]
    for kind in kinds:
        if kind not in KINDS:
            raise SystemExit(f"unknown kind {kind}; choose from {KINDS}")
    steps = int(round(args.start_max_s / args.start_step_s))
    starts = [round(i * args.start_step_s, 6) for i in range(steps + 1)]

    began = time.perf_counter()
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args.missile,)) as pool:
        baselines = {r["range_m"]: r for r in pool.map(
            _run, [(_scenario(args, rng), "none", 0.0, args.turn_g) for rng in ranges])}
        jobs = []
        for rng in ranges:
            base = baselines[rng]
            for kind in kinds:
                for start in starts:
                    # A start after the non-evading intercept cannot change the outcome.
                    if base["escaped"] or start < base["flight_time_s"]:
                        jobs.append((_scenario(args, rng), kind, start, args.turn_g))
        rows = list(pool.map(_run, jobs, chunksize=4))
    elapsed = time.perf_counter() - began

    out = args.out or ROOT / "outputs" / "escape_window" / (
        f"{args.missile}_{args.observation_mode}_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["range_m", "kind", "start_s", "event", "flight_time_s", "min_distance_m", "escaped", "radar_reject"]
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(list(baselines.values()) + rows)

    print(f"{args.missile} {args.observation_mode} | launch {args.launch_altitude_m:.0f} m "
          f"{args.launch_speed_kmh:.0f} km/h | target {args.target_altitude_m:.0f} m "
          f"{args.target_speed_kmh:.0f} km/h course {args.target_course_deg:g} | turn {args.turn_g:g} g | "
          f"grid {args.start_step_s:g} s")
    print(f"{len(rows) + len(baselines)} runs in {elapsed:.1f} s on {args.workers} workers -> {out}")
    for rng in ranges:
        base = baselines[rng]
        note = "no evasion: miss" if base["escaped"] else f"no evasion: hit at {base['flight_time_s']:.1f} s"
        print(f"\n{rng / 1000:g} km ({note})")
        for kind in kinds:
            spans = _windows([r for r in rows if r["range_m"] == rng and r["kind"] == kind])
            text = ", ".join(f"{a:g}-{b:g} s" if a != b else f"{a:g} s" for a, b in spans) or "none"
            print(f"  {kind:5s} escape starts: {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
