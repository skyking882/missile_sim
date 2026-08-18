#!/usr/bin/env python3
"""Compare the local PL-12 runtime with a frozen SensorWhale public reference.

This runner aligns target kinematics. It does not claim that the protected
SensorWhale flight engine has been reproduced.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aim120_model.public_api import simulate  # noqa: E402
from missile_gui.library import scan_library  # noqa: E402


REFERENCE_PATH = (
    ROOT
    / "data"
    / "reference_external"
    / "sensorwhale_pl12_off_axis_38_course0_20260818.json"
)


def main() -> int:
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    profiles, errors = scan_library(ROOT / "missiles", ROOT)
    if errors:
        raise ValueError("missile library errors: " + " | ".join(errors[:3]))
    profile = next(item for item in profiles if item.get("missile_id") == "cn_pl12")
    result = simulate(profile, reference["local_scenario"])
    samples = result["samples"]
    first = samples[0]
    local_summary = result["summary"]
    report: dict[str, Any] = {
        "reference_boundary": reference["boundary"],
        "course_mapping": {
            "formula": "local_los_course_deg = sensorwhale_course_deg - target_azimuth_deg",
            "sensorwhale_course_deg": reference["sensorwhale_input"]["targetCourseDeg"],
            "target_azimuth_deg": reference["sensorwhale_input"]["targetAzimuthDeg"],
            "equivalent_local_los_course_deg": reference["equivalent_existing_local_input"]["target_heading_deg"],
        },
        "initial_target_kinematics": {
            "position_m": first["target_position_m"],
            "velocity_mps": first["target_velocity_mps"],
            "horizontal_speed_mps": math.hypot(
                float(first["target_velocity_mps"][0]),
                float(first["target_velocity_mps"][2]),
            ),
        },
        "sensorwhale_public_summary": reference["public_result_summary"],
        "local_summary": local_summary,
        "local_peak_aoa_deg": max(
            math.degrees(float(sample["angle_of_attack_rad"])) for sample in samples
        ),
        "model_boundary": result["model"]["runtime_boundary"],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
