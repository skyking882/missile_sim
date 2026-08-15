#!/usr/bin/env python3
"""Sweep one shared PID error scale without mutating per-missile PID gains."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aim120_model.h2_simulator import H2Simulator  # noqa: E402
from aim120_model.public_api import _case_from_scenario, validate_scenario  # noqa: E402
from missile_gui.library import scan_library  # noqa: E402


MISSILES = ("cn_pl12", "jp_aam4", "su_r_77")
SCENARIOS = {
    "S0": {"distance": 8000.0, "target_speed": 900.0, "azimuth": 45.0, "heading": -10.0},
    "S1": {"distance": 8000.0, "target_speed": 900.0, "azimuth": 45.0, "heading": -20.0},
    "S2": {"distance": 4500.0, "target_speed": 1200.0, "azimuth": 60.0, "heading": -45.0},
}


def _scenario(name: str) -> dict[str, Any]:
    item = SCENARIOS[name]
    return validate_scenario({
        "launch_speed_kmh": 1200.0,
        "launch_altitude_m": 6500.0,
        "launch_pitch_deg": 0.0,
        "launch_heading_deg": 0.0,
        "target_speed_kmh": item["target_speed"],
        "target_altitude_m": 6500.0,
        "initial_distance_m": item["distance"],
        "target_azimuth_deg": item["azimuth"],
        "target_heading_deg": item["heading"],
        "target_vertical_heading_deg": 0.0,
        "target_constant_turn_g": 0.0,
        "observation_mode": "ideal_truth",
    })


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    samples = result["samples"]
    saturated = [
        sample for sample in samples
        if max(
            abs(float(sample["pitch_requested_fin_command"])),
            abs(float(sample["yaw_requested_fin_command"])),
        ) >= 0.999
    ]
    return {
        "event": result["event_type"],
        "minimum_distance_m": min(float(sample["distance_to_target_m"]) for sample in samples),
        "maximum_actual_g": max(float(sample["actual_overload_g"]) for sample in samples),
        "requested_fin_saturation_fraction": len(saturated) / len(samples),
        "first_requested_fin_saturation_s": (
            float(saturated[0]["time_s"]) if saturated else None
        ),
        "maximum_actual_fin_deg": max(
            math.degrees(math.hypot(
                float(sample["actual_pitch_fin_angle_rad"]),
                float(sample["actual_yaw_fin_angle_rad"]),
            ))
            for sample in samples
        ),
        "maximum_body_rate_deg_s": max(
            math.degrees(math.hypot(
                float(sample["pitch_rate_rad_s"]),
                float(sample["yaw_rate_rad_s"]),
            ))
            for sample in samples
        ),
        "maximum_aoa_deg": max(
            math.degrees(float(sample["angle_of_attack_rad"])) for sample in samples
        ),
    }


def run(scenario_names: list[str], scales: list[float]) -> list[dict[str, Any]]:
    profiles, errors = scan_library(ROOT / "missiles", ROOT)
    if errors:
        raise ValueError("missile library errors: " + " | ".join(errors[:3]))
    by_id = {profile["missile_id"]: profile for profile in profiles}
    source_pids = {missile_id: copy.deepcopy(by_id[missile_id]["control"]["pid"]) for missile_id in MISSILES}
    rows: list[dict[str, Any]] = []
    for scenario_name in scenario_names:
        case = _case_from_scenario(_scenario(scenario_name))
        for scale in scales:
            for missile_id in MISSILES:
                profile = by_id[missile_id]
                config = copy.deepcopy(profile["_model_config"])
                config["control"]["pid_error_units"] = "g"
                config["control"]["pid_error_scale"] = scale
                result = H2Simulator(config).run(case)
                if profile["control"]["pid"] != source_pids[missile_id]:
                    raise AssertionError(f"raw PID mutated for {missile_id}")
                rows.append({
                    "scenario": scenario_name,
                    "pid_error_scale": scale,
                    "missile_id": missile_id,
                    **_summary(result),
                })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", action="append", choices=SCENARIOS, required=True)
    parser.add_argument("--scale", action="append", type=float, required=True)
    args = parser.parse_args(argv)
    rows = run(args.scenario, args.scale)
    print(json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
