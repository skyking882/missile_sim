#!/usr/bin/env python3
"""Identify one effective H2 CdA scale from the existing power-only anchor."""

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
from aim120_model.h2_simulator import H2Simulator
from aim120_model.units import mps_to_kmh


TARGET_TIME_S = 7.0
TARGET_SPEED_KMH = 3630.0


def _speed_at(result: dict[str, object], time_s: float) -> float:
    samples = result["samples"]
    assert isinstance(samples, list)
    if not samples or float(samples[-1]["time_s"]) < time_s - 1e-9:
        raise ValueError("simulation terminated before the requested calibration time")
    for left, right in zip(samples, samples[1:]):
        t0 = float(left["time_s"])
        t1 = float(right["time_s"])
        if t0 <= time_s <= t1 + 1e-12:
            fraction = 0.0 if abs(t1 - t0) <= 1e-15 else (time_s - t0) / (t1 - t0)
            velocity = tuple(
                float(left["velocity_mps"][i])
                + fraction * (float(right["velocity_mps"][i]) - float(left["velocity_mps"][i]))
                for i in range(3)
            )
            return mps_to_kmh(math.sqrt(sum(component * component for component in velocity)))
    sample = samples[-1]
    velocity = tuple(float(value) for value in sample["velocity_mps"])
    return mps_to_kmh(math.sqrt(sum(component * component for component in velocity)))


def evaluate(config: dict[str, object], case: dict[str, object], drag_scale: float) -> dict[str, float | str]:
    candidate = copy.deepcopy(config)
    candidate["drag_model"]["drag_scale"] = float(drag_scale)
    result = H2Simulator(candidate).run(case)
    speed = _speed_at(result, TARGET_TIME_S)
    return {
        "drag_scale": float(drag_scale),
        "speed_at_7s_kmh": float(speed),
        "speed_error_kmh": float(speed - TARGET_SPEED_KMH),
        "speed_error_percent": float(100.0 * (speed - TARGET_SPEED_KMH) / TARGET_SPEED_KMH),
        "event_type": str(result["event_type"]),
        "terminal_time_s": float(result["terminal_time_s"]),
    }


def main() -> int:
    config_path = PROJECT_DIR / "configs" / "aim120a_h2.yaml"
    cases_path = PROJECT_DIR / "configs" / "cases.yaml"
    output_path = PROJECT_DIR / "outputs" / "h2" / "drag_identification_report.json"
    config = load_model_config(config_path)
    case = find_case(load_cases(cases_path), "power_only")
    initial_scale = float(config["drag_model"].get("drag_scale", 1.0))
    low = 0.01
    high = 1.0
    low_eval = evaluate(config, case, low)
    high_eval = evaluate(config, case, high)
    if float(low_eval["speed_error_kmh"]) < 0.0 or float(high_eval["speed_error_kmh"]) > 0.0:
        raise RuntimeError(
            "calibration bracket does not straddle the target: "
            f"low={low_eval['speed_at_7s_kmh']:.3f}, high={high_eval['speed_at_7s_kmh']:.3f}"
        )
    history: list[dict[str, float | str]] = [low_eval, high_eval]
    for _ in range(36):
        midpoint = 0.5 * (low + high)
        current = evaluate(config, case, midpoint)
        history.append(current)
        if float(current["speed_error_kmh"]) > 0.0:
            low = midpoint
        else:
            high = midpoint
    fitted_scale = 0.5 * (low + high)
    fitted_eval = evaluate(config, case, fitted_scale)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_label": config["model_label"],
        "aero_model_version": config["aero_model_version"],
        "calibration": {
            "quantity": "effective_zero_aoa_drag_area_scale",
            "case_name": case["name"],
            "reference_time_s": TARGET_TIME_S,
            "reference_speed_kmh": TARGET_SPEED_KMH,
            "reference_origin": "existing_plan2_power_only_7s_anchor",
            "initial_config_drag_scale": initial_scale,
            "fitted_drag_scale": fitted_scale,
            "initial_config_evaluation": evaluate(config, case, initial_scale),
            "fitted_evaluation": fitted_eval,
            "acceptance_limit_percent": 5.0,
            "acceptance_target_percent": 3.0,
            "status": "pass" if abs(float(fitted_eval["speed_error_percent"])) < 5.0 else "fail",
        },
        "identifiability_boundary": (
            "This fit identifies one effective CdA scale under the selected H2 shape and "
            "area basis. It does not uniquely identify physical reference area, Cx, or a "
            "full Mach/AoA drag table."
        ),
        "compensation_boundary": "No PN, loft, PID, or fin authority was enabled for the anchor case.",
        "statshark_new_calculation_performed_this_run": False,
        "bracket": {"low": low_eval, "high": high_eval},
        "iterations": history,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"fitted drag_scale={fitted_scale:.9f} "
        f"speed7={float(fitted_eval['speed_at_7s_kmh']):.3f}km/h "
        f"error={float(fitted_eval['speed_error_percent']):+.5f}%"
    )
    print(f"written: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
