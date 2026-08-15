#!/usr/bin/env python3
"""Generate the small deterministic 3/9-style target trajectory used by Plan 8."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


G0 = 9.80665


def _point(
    time_s: float,
    duration_s: float,
    initial_distance_m: float,
    altitude_m: float,
    speed_mps: float,
    turn_start_s: float,
    turn_g: float,
    beam_sign: float,
) -> tuple[float, float, float, float, float, float]:
    if time_s <= turn_start_s:
        return initial_distance_m - speed_mps * time_s, altitude_m, 0.0, -speed_mps, 0.0, 0.0

    turn_time = max(0.0, time_s - turn_start_s)
    start_x = initial_distance_m - speed_mps * turn_start_s
    turn_rate = abs(turn_g) * G0 / max(speed_mps, 1.0e-9)
    if turn_rate <= 1.0e-12:
        return start_x - speed_mps * turn_time, altitude_m, 0.0, -speed_mps, 0.0, 0.0
    angle = min(math.pi / 2.0, turn_rate * turn_time)
    turn_duration = (math.pi / 2.0) / turn_rate
    if turn_time <= turn_duration:
        x = start_x - speed_mps * math.sin(angle) / turn_rate
        z = beam_sign * speed_mps * (1.0 - math.cos(angle)) / turn_rate
        vx = -speed_mps * math.cos(angle)
        vz = beam_sign * speed_mps * math.sin(angle)
        return x, altitude_m, z, vx, 0.0, vz

    end_x = start_x - speed_mps / turn_rate
    end_z = beam_sign * speed_mps / turn_rate
    cruise_time = turn_time - turn_duration
    return end_x, altitude_m, end_z + beam_sign * speed_mps * cruise_time, 0.0, 0.0, beam_sign * speed_mps


def generate(
    duration_s: float,
    sample_dt_s: float,
    initial_distance_m: float,
    altitude_m: float,
    speed_mps: float,
    turn_start_s: float,
    turn_g: float,
    beam_side: str,
) -> list[dict[str, float]]:
    if duration_s <= 0.0 or sample_dt_s <= 0.0 or initial_distance_m <= 0.0 or speed_mps <= 0.0:
        raise ValueError("duration, sample dt, initial distance, and speed must be positive")
    if turn_start_s < 0.0 or turn_start_s > duration_s:
        raise ValueError("turn-start-s must be within the trajectory duration")
    if turn_g < 0.0:
        raise ValueError("turn-g must be non-negative")
    if beam_side not in {"left", "right"}:
        raise ValueError("beam-side must be left or right")
    beam_sign = 1.0 if beam_side == "left" else -1.0
    count = int(math.floor(duration_s / sample_dt_s + 1.0e-9))
    times = [min(duration_s, index * sample_dt_s) for index in range(count + 1)]
    if not times or times[-1] < duration_s - 1.0e-9:
        times.append(duration_s)
    rows: list[dict[str, float]] = []
    for time_s in times:
        x, y, z, vx, vy, vz = _point(
            time_s,
            duration_s,
            initial_distance_m,
            altitude_m,
            speed_mps,
            turn_start_s,
            turn_g,
            beam_sign,
        )
        rows.append({
            "time_s": time_s,
            "x_m": x,
            "y_m": y,
            "z_m": z,
            "vx_mps": vx,
            "vy_mps": vy,
            "vz_mps": vz,
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a deterministic 3/9-style notch trajectory CSV")
    parser.add_argument("--duration-s", type=float, default=39.0)
    parser.add_argument("--sample-dt-s", type=float, default=0.02)
    parser.add_argument("--initial-distance-m", type=float, default=20000.0)
    parser.add_argument("--altitude-m", type=float, default=6500.0)
    parser.add_argument("--speed-mps", type=float, default=300.0)
    parser.add_argument("--turn-start-s", type=float, default=4.0)
    parser.add_argument("--turn-g", type=float, default=9.0)
    parser.add_argument("--beam-side", choices=("left", "right"), default="left")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scenarios/trajectories/aim120_notch_39.csv"),
    )
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing trajectory: {output}")
    rows = generate(
        args.duration_s,
        args.sample_dt_s,
        args.initial_distance_m,
        args.altitude_m,
        args.speed_mps,
        args.turn_start_s,
        args.turn_g,
        args.beam_side,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("time_s", "x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps"))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} trajectory points to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
