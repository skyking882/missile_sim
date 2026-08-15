#!/usr/bin/env python3
"""Validate frozen H2 PN/loft behaviour without fitting any guidance parameter."""

from __future__ import annotations

import copy
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from aim120_model.config import find_case, load_cases, load_model_config
from aim120_model.guidance import guidance_command
from aim120_model.h2_simulator import H2Simulator
from aim120_model.metrics import terminal_summary
from aim120_model.target import TargetModel


def peak_altitude(result: dict[str, object]) -> dict[str, float]:
    samples = result["samples"]
    sample = max(samples, key=lambda item: float(item["position_m"][1]))
    return {
        "peak_altitude_m": float(sample["position_m"][1]),
        "peak_time_s": float(sample["time_s"]),
    }


def guidance_direction_check(config: dict[str, object], case: dict[str, object]) -> dict[str, object]:
    no_loft = copy.deepcopy(config)
    no_loft["guidance"]["lofting_enabled"] = False
    simulator = H2Simulator(no_loft)
    initial = case["initial_conditions"]
    gravity = float(no_loft["atmosphere"]["gravity_mps2"])
    positive_case = copy.deepcopy(case)
    positive_case["initial_conditions"]["target_azimuth_deg"] = 20.0
    negative_case = copy.deepcopy(case)
    negative_case["initial_conditions"]["target_azimuth_deg"] = -20.0
    positive_state = simulator.initial_state(positive_case)
    negative_state = simulator.initial_state(negative_case)
    positive_target = TargetModel(positive_case["initial_conditions"], gravity).initial_state
    negative_target = TargetModel(negative_case["initial_conditions"], gravity).initial_state
    positive = guidance_command(positive_state, positive_target, 0.0, no_loft, enabled=True)
    negative = guidance_command(negative_state, negative_target, 0.0, no_loft, enabled=True)
    # A left/right mirror preserves the vertical/pitch command and reverses
    # only the yaw command.
    pitch_difference = positive.commanded_body_acceleration_g[0] - negative.commanded_body_acceleration_g[0]
    yaw_sum = positive.commanded_body_acceleration_g[1] + negative.commanded_body_acceleration_g[1]
    return {
        "lofting_enabled": False,
        "positive_target_command_g": list(positive.commanded_body_acceleration_g),
        "negative_target_command_g": list(negative.commanded_body_acceleration_g),
        "pitch_mirror_residual_g": abs(pitch_difference),
        "yaw_mirror_residual_g": abs(yaw_sum),
        "positive_yaw_sign": math.copysign(1.0, positive.commanded_body_acceleration_g[1]),
        "negative_yaw_sign": math.copysign(1.0, negative.commanded_body_acceleration_g[1]),
        "pass": (
            positive.commanded_body_acceleration_g[1] > 0.0
            and negative.commanded_body_acceleration_g[1] < 0.0
            and abs(pitch_difference) < 1e-9
            and abs(yaw_sum) < 1e-9
        ),
    }


def mirror_trajectory_residual(positive: dict[str, object], negative: dict[str, object]) -> dict[str, float | bool]:
    samples_a = positive["samples"]
    samples_b = negative["samples"]
    count = min(len(samples_a), len(samples_b))
    max_position_residual = 0.0
    max_velocity_residual = 0.0
    max_attitude_residual = 0.0
    max_time_residual = 0.0
    for a, b in zip(samples_a[:count], samples_b[:count]):
        max_time_residual = max(max_time_residual, abs(float(a["time_s"]) - float(b["time_s"])))
        position_a = a["position_m"]
        position_b = b["position_m"]
        velocity_a = a["velocity_mps"]
        velocity_b = b["velocity_mps"]
        position_error = max(
            abs(float(position_a[0]) - float(position_b[0])),
            abs(float(position_a[1]) - float(position_b[1])),
            abs(float(position_a[2]) + float(position_b[2])),
        )
        velocity_error = max(
            abs(float(velocity_a[0]) - float(velocity_b[0])),
            abs(float(velocity_a[1]) - float(velocity_b[1])),
            abs(float(velocity_a[2]) + float(velocity_b[2])),
        )
        attitude_error = max(
            abs(float(a["pitch_rad"]) - float(b["pitch_rad"])),
            abs(float(a["yaw_rad"]) + float(b["yaw_rad"])),
        )
        max_position_residual = max(max_position_residual, position_error)
        max_velocity_residual = max(max_velocity_residual, velocity_error)
        max_attitude_residual = max(max_attitude_residual, attitude_error)
    terminal_a = terminal_summary(positive)
    terminal_b = terminal_summary(negative)
    event_match = terminal_a["event_type"] == terminal_b["event_type"]
    terminal_time_residual = abs(terminal_a["terminal_time_s"] - terminal_b["terminal_time_s"])
    pass_value = (
        len(samples_a) == len(samples_b)
        and event_match
        and terminal_time_residual < 1e-8
        and max_position_residual < 1e-6
        and max_velocity_residual < 1e-9
        and max_attitude_residual < 1e-10
    )
    return {
        "sample_count_positive": len(samples_a),
        "sample_count_negative": len(samples_b),
        "max_position_mirror_residual_m": max_position_residual,
        "max_velocity_mirror_residual_mps": max_velocity_residual,
        "max_attitude_mirror_residual_rad": max_attitude_residual,
        "max_time_mirror_residual_s": max_time_residual,
        "terminal_time_residual_s": terminal_time_residual,
        "event_match": event_match,
        "pass": pass_value,
    }


def run_summary(result: dict[str, object]) -> dict[str, object]:
    summary = terminal_summary(result)
    summary.update(peak_altitude(result))
    summary["loft_active_sample_count"] = sum(1 for sample in result["samples"] if sample["loft_active"])
    return summary


def main() -> int:
    config_path = PROJECT_DIR / "configs" / "aim120a_h2.yaml"
    cases_path = PROJECT_DIR / "configs" / "cases.yaml"
    config = load_model_config(config_path)
    cases = load_cases(cases_path)
    baseline = find_case(cases, "baseline_full")
    off_axis = find_case(cases, "off_axis_20deg_local_only")
    simulator = H2Simulator(config)

    baseline_with_loft = simulator.run(baseline)
    no_loft_config = copy.deepcopy(config)
    no_loft_config["guidance"]["lofting_enabled"] = False
    baseline_without_loft = H2Simulator(no_loft_config).run(baseline)

    positive_case = copy.deepcopy(off_axis)
    positive_case["name"] = "off_axis_20deg_positive_validation"
    negative_case = copy.deepcopy(off_axis)
    negative_case["name"] = "off_axis_20deg_negative_validation"
    negative_case["initial_conditions"]["target_azimuth_deg"] = -20.0
    positive_result = simulator.run(positive_case)
    negative_result = simulator.run(negative_case)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_label": config["model_label"],
        "aero_model_version": config["aero_model_version"],
        "force_geometry_version": config["force_geometry_version"],
        "control_model_version": config["control_model_version"],
        "parameters_frozen": {
            "pn_gain": config["guidance"]["pn_gain"],
            "maximum_lateral_acceleration_g": config["guidance"]["maximum_lateral_acceleration_g"],
            "maximum_angular_rate_deg_s": config["guidance"]["maximum_angular_rate_deg_s"],
            "lofting_elevation_deg": config["guidance"]["lofting_elevation_deg"],
            "angle_to_acceleration_multiplier": config["guidance"]["angle_to_acceleration_multiplier"],
            "time_to_hit_gain_table": config["guidance"]["time_to_hit_gain_table"],
        },
        "pn_without_loft_direction_check": guidance_direction_check(config, off_axis),
        "loft_switch_check": {
            "with_loft": run_summary(baseline_with_loft),
            "without_loft": run_summary(baseline_without_loft),
            "loft_changes_summary": (
                run_summary(baseline_with_loft)["peak_altitude_m"]
                != run_summary(baseline_without_loft)["peak_altitude_m"]
            ),
        },
        "azimuth_mirror_check": {
            "positive": run_summary(positive_result),
            "negative": run_summary(negative_result),
            "residual": mirror_trajectory_residual(positive_result, negative_result),
        },
        "interpretation": {
            "guidance_parameter_fitting_performed": False,
            "statshark_new_calculation_performed_this_run": False,
            "phase_f_status": "validation_only_not_parameter_fit",
            "h2_4_power_only_terminal_event_still_fails": True,
        },
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = PROJECT_DIR / "outputs" / "h2" / f"pn_loft_validation_h2_{stamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("PN no-loft:", report["pn_without_loft_direction_check"])
    print("with loft:", report["loft_switch_check"]["with_loft"])
    print("without loft:", report["loft_switch_check"]["without_loft"])
    print("azimuth mirror:", report["azimuth_mirror_check"]["residual"])
    print(f"written: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
