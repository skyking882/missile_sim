#!/usr/bin/env python3
"""Run B0/B1/B2/B3 baseIndSpeed candidates on one shared engagement."""

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


MISSILES = ("us_aim_120a", "cn_pl12", "su_r_77", "jp_aam4", "r_darter", "il_derby")
MODES = ("none", "fin_authority_q", "matched_q", "pid_output_q")


def _first_sign_reversal(samples: list[dict[str, Any]], key: str, component: int | None = None) -> float | None:
    first_sign = 0
    for sample in samples:
        value: Any = sample.get(key, 0.0)
        if component is not None:
            value = value[component]
        number = float(value)
        sign = 1 if number > 1e-9 else -1 if number < -1e-9 else 0
        if sign == 0:
            continue
        if first_sign == 0:
            first_sign = sign
        elif sign != first_sign:
            return float(sample["time_s"])
    return None


def _summary(result: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    samples = result["samples"]
    closest = min(samples, key=lambda sample: float(sample["distance_to_target_m"]))
    saturated = [
        sample
        for sample in samples
        if max(
            abs(float(sample.get("pitch_requested_fin_command", 0.0))),
            abs(float(sample.get("yaw_requested_fin_command", 0.0))),
        ) >= 0.999
    ]
    return {
        "event": result["event_type"],
        "hit": result["event_type"] in {"impact", "fuse"},
        "flight_time_s": float(result["terminal_time_s"]),
        "minimum_distance_m": float(closest["distance_to_target_m"]),
        "maximum_body_g": max(float(sample["lateral_load_g"]) for sample in samples),
        "maximum_trajectory_g": max(float(sample["trajectory_lateral_load_g"]) for sample in samples),
        "maximum_aoa_deg": max(math.degrees(float(sample["angle_of_attack_rad"])) for sample in samples),
        "maximum_indicated_speed_kmh": max(float(sample["indicated_speed_kmh"]) for sample in samples),
        "maximum_q_ratio": max(float(sample["base_indicated_speed_q_ratio"]) for sample in samples),
        "closest_time_s": float(closest["time_s"]),
        "closest_body_g": float(closest["lateral_load_g"]),
        "closest_indicated_speed_kmh": float(closest["indicated_speed_kmh"]),
        "closest_q_ratio": float(closest["base_indicated_speed_q_ratio"]),
        "fin_saturation_fraction": len(saturated) / len(samples),
        "command_yaw_first_reversal_s": _first_sign_reversal(samples, "commanded_acceleration_g", 1),
        "pid_yaw_first_reversal_s": _first_sign_reversal(samples, "yaw_pid_output"),
        "fin_angle_yaw_first_reversal_s": _first_sign_reversal(samples, "actual_yaw_fin_angle_rad"),
        "base_indicated_speed_kmh": config["control"]["base_indicated_speed_kmh"],
        "raw_pid": copy.deepcopy(config["control"]["pid"]),
    }


def run() -> dict[str, Any]:
    profiles, errors = scan_library(ROOT / "missiles", ROOT)
    if errors:
        raise ValueError("missile library errors: " + " | ".join(errors[:3]))
    by_id = {profile["missile_id"]: profile for profile in profiles}
    scenario = validate_scenario({
        "launch_speed_kmh": 1200.0,
        "launch_altitude_m": 6500.0,
        "launch_pitch_deg": 0.0,
        "launch_heading_deg": 0.0,
        "target_speed_kmh": 900.0,
        "target_altitude_m": 6500.0,
        "initial_distance_m": 8000.0,
        "target_azimuth_deg": 45.0,
        "target_heading_deg": -20.0,
        "target_vertical_heading_deg": 0.0,
        "target_constant_turn_g": 0.0,
        "observation_mode": "ideal_truth",
    })
    case = _case_from_scenario(scenario)
    rows: list[dict[str, Any]] = []
    for missile_id in MISSILES:
        profile = by_id[missile_id]
        source_pid = copy.deepcopy(profile["control"]["pid"])
        base_speed = profile["control"].get("base_indicated_speed_kmh")
        for mode in MODES:
            config = copy.deepcopy(profile["_model_config"])
            config["control"]["base_indicated_speed_mode"] = mode
            result = H2Simulator(config).run(case)
            if profile["control"]["pid"] != source_pid:
                raise AssertionError(f"raw PID mutated for {missile_id}")
            rows.append({
                "missile_id": missile_id,
                "display_name": profile["display_name"],
                "candidate": {
                    "none": "B0_baseline",
                    "fin_authority_q": "B1_fin_authority_times_q_ratio",
                    "matched_q": "B2_fin_authority_times_q_ratio_fin_command_div_q_ratio",
                    "pid_output_q": "B3_pid_output_times_q_ratio",
                }[mode],
                "mode": mode,
                "source_base_indicated_speed_kmh": base_speed,
                "summary": _summary(result, config),
            })
    return {
        "schema_version": 1,
        "status": "diagnostic_candidates_not_solver_identification",
        "scenario": scenario,
        "candidate_definitions": {
            "B0_baseline": "no baseIndSpeed scaling",
            "B1_fin_authority_times_q_ratio": "fin force and AoA restoring moment multiplied by q/q_ref",
            "B2_fin_authority_times_q_ratio_fin_command_div_q_ratio": "B1 plus requested fin command divided by q/q_ref",
            "B3_pid_output_times_q_ratio": "raw PID sum multiplied by q/q_ref before fin-command clamp",
            "q_ratio": "q / (0.5 * rho_sea_level * (baseIndSpeed/3.6)^2)",
        },
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing result: {output}")
    payload = run()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {len(payload['rows'])} candidate rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
