#!/usr/bin/env python3
"""Run the two frozen v1.0.0 trajectory examples without writing files."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aim120_model import (  # noqa: E402
    EffectiveControllerEnvelope,
    H2Simulator,
    __version__,
    load_cases,
    load_model_config,
)


def summarize(result: dict) -> dict:
    terminal = result["samples"][-1]
    velocity = terminal["velocity_mps"]
    peak_command_g = max(
        math.hypot(*sample["commanded_acceleration_g"])
        for sample in result["samples"]
    )
    return {
        "case": result["case_name"],
        "event": result["event_type"],
        "time_s": round(result["terminal_time_s"], 6),
        "miss_distance_m": round(terminal["distance_to_target_m"], 6),
        "terminal_speed_kmh": round(math.sqrt(sum(v * v for v in velocity)) * 3.6, 3),
        "peak_command_g": round(peak_command_g, 3),
        "loft_used": any(sample["loft_active"] for sample in result["samples"]),
    }


def main() -> int:
    config = load_model_config(ROOT / "configs" / "aim120a_v1.json")
    cases = load_cases(ROOT / "configs" / "aim120a_v1_cases.json")
    simulator = H2Simulator(config)
    controller_config = config["effective_controller"]
    controller = EffectiveControllerEnvelope(
        gain=controller_config["gain"],
        authority_fraction=controller_config["authority_fraction"],
    )
    output = {
        "release_version": __version__,
        "trajectory_results": [summarize(simulator.run(case)) for case in cases],
        "effective_controller_example": {
            "a_cmd_yaw_g": 10.0,
            "current_g_magnitude": round(
                controller.predict_current_g_magnitude(
                    10.0,
                    config["aerodynamics"]["fins_lateral_acceleration_g"],
                ),
                6,
            ),
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
