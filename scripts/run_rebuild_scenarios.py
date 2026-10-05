#!/usr/bin/env python3
"""Run the transparent rebuild v1 against one common cross-missile smoke case.

The script only assembles an in-memory profile and prints the public summary;
it does not change profiles or write simulation artifacts.  The case is a
PL-12-like 8 km/30 degree smoke geometry for both missiles, not an R-77 replay
or a replay-fit comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from aim120_model.profile_adapter import (  # noqa: E402
    build_h2_candidate_config,
    load_runtime_defaults,
    unsupported_model_types,
)
from aim120_model.public_api import simulate  # noqa: E402


SCENARIO = {
    "loft_enabled": True,
    "observation_mode": "ideal_truth",
    "target_course_reference": "statshark_relative_to_los",
    "launch_speed_kmh": 1200.0,
    "launch_altitude_m": 6200.0,
    "launch_pitch_deg": 0.0,
    "launch_heading_deg": 0.0,
    "target_speed_kmh": 1550.0,
    "target_altitude_m": 6200.0,
    "initial_distance_m": 8000.0,
    "target_azimuth_deg": 30.0,
    "target_heading_deg": 0.0,
    "target_vertical_heading_deg": 0.0,
    "target_constant_turn_g": 0.0,
    "max_simulation_time_s": 15.0,
}


def load_rebuild_profile(missile_id: str) -> dict[str, Any]:
    profile_path = PROJECT_DIR / "missiles" / f"{missile_id}.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    unsupported = unsupported_model_types(profile)
    if unsupported:
        raise ValueError(f"{missile_id}: unsupported model types {unsupported}")
    defaults = load_runtime_defaults(
        str(PROJECT_DIR / "config" / "profile_h2_rebuild_defaults.json")
    )
    config, assumptions = build_h2_candidate_config(profile, defaults)
    profile["_model_config"] = config
    profile["_runtime_assumptions"] = assumptions
    profile["_runtime_unsupported"] = []
    return profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "missiles",
        nargs="*",
        default=["su_r_77_1", "cn_pl12"],
        help="profile ids (default: su_r_77_1 cn_pl12)",
    )
    args = parser.parse_args()
    for missile_id in args.missiles:
        result = simulate(load_rebuild_profile(missile_id), SCENARIO)
        summary = result["summary"]
        print(
            f"{missile_id} (common_smoke): model={result['model']['label']} "
            f"event={summary['termination_event']} t={summary['flight_time_s']:.3f}s "
            f"range={summary['terminal_distance_m']:.1f}m "
            f"speed={summary['terminal_speed_kmh']:.1f}km/h"
        )


if __name__ == "__main__":
    main()
