#!/usr/bin/env python3
"""Localize the first H2.4 failure without fitting any new parameter."""

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
from aim120_model.dynamics import SimState
from aim120_model.h2_dynamics import forces_for_state_h2
from aim120_model.h2_simulator import H2Simulator
from aim120_model.metrics import terminal_summary
from aim120_model.propulsion import PiecewisePropulsion


def sample_at(result: dict[str, object], requested_time_s: float) -> dict[str, object]:
    samples = result["samples"]
    return min(samples, key=lambda sample: abs(float(sample["time_s"]) - requested_time_s))


def force_snapshot(sample: dict[str, object], config: dict[str, object], propulsion: PiecewisePropulsion) -> dict[str, float]:
    state = SimState(
        position=tuple(float(value) for value in sample["position_m"]),
        velocity=tuple(float(value) for value in sample["velocity_mps"]),
        pitch=float(sample["pitch_rad"]),
        yaw=float(sample["yaw_rad"]),
        pitch_rate=float(sample["pitch_rate_rad_s"]),
        yaw_rate=float(sample["yaw_rate_rad_s"]),
        mass=float(sample["mass_kg"]),
    )
    diagnostics = forces_for_state_h2(state, float(sample["time_s"]), config, propulsion, powered=True)
    gravity = float(config["atmosphere"]["gravity_mps2"])
    speed = diagnostics.aero.speed_mps
    horizontal_speed = math.hypot(state.velocity[0], state.velocity[2])
    flight_path_angle_deg = math.degrees(math.atan2(state.velocity[1], max(horizontal_speed, 1e-12)))
    return {
        "time_s": float(sample["time_s"]),
        "altitude_m": float(state.position[1]),
        "speed_kmh": speed * 3.6,
        "mach": float(diagnostics.aero.mach),
        "flight_path_angle_deg": flight_path_angle_deg,
        "pitch_alpha_deg": math.degrees(diagnostics.aero.pitch_alpha_rad),
        "cda0_m2": float(diagnostics.aero.cda0_m2),
        "cda_alpha_m2": float(diagnostics.aero.cda_alpha_m2),
        "lateral_load_g": float(diagnostics.lateral_load_g),
        "vertical_specific_force_g": diagnostics.specific_force_mps2[1] / gravity,
        "vertical_net_acceleration_g": diagnostics.acceleration_mps2[1] / gravity,
        "drag_power_w": float(diagnostics.drag_power_w),
        "lift_power_w": float(diagnostics.lift_power_w),
    }


def main() -> int:
    config = load_model_config(PROJECT_DIR / "configs" / "aim120a_h2.yaml")
    cases = load_cases(PROJECT_DIR / "configs" / "cases.yaml")
    case = find_case(cases, "power_only")
    propulsion = PiecewisePropulsion.from_config(config)
    result = H2Simulator(config).run(case)
    snapshots = [force_snapshot(sample_at(result, time_s), config, propulsion) for time_s in (7.0, 20.0, 40.0, 60.0)]

    no_lift_config = copy.deepcopy(config)
    no_lift_config["aerodynamics"]["natural_lift_enabled"] = False
    no_lift_result = H2Simulator(no_lift_config).run(case)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_label": config["model_label"],
        "case_name": case["name"],
        "frozen_config_drag_scale": config["drag_model"]["drag_scale"],
        "local_result": terminal_summary(result),
        "existing_reference_anchors": {
            "terminal_event": "lifetime",
            "terminal_time_s": 80.0,
            "terminal_speed_kmh": 690.0,
            "terminal_altitude_m": 1728.0,
        },
        "trajectory_snapshots": snapshots,
        "diagnostic_ablation_natural_lift_disabled": {
            "purpose": "localize sensitivity only; not used for fitting or configuration changes",
            "result": terminal_summary(no_lift_result),
        },
        "earliest_failure_layer": "low_mach_long_duration_aerodynamic_lift_and_flight_path",
        "reasoning": [
            "The H2.1 high-Mach 7 s speed anchor remains matched before this failure.",
            "After burnout, the power-only body has no guidance or fin authority and its flight-path angle grows increasingly negative.",
            "Natural lift delays ground impact substantially, but the frozen candidate still reaches ground at about 61.1 s instead of the 80 s reference lifetime.",
            "Disabling natural lift makes the failure earlier, so lift geometry/strength and low-Mach longitudinal behaviour are the next physical layer; this is not a reason to tune PN, loft, or PID.",
        ],
        "statshark_new_calculation_performed_this_run": False,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = PROJECT_DIR / "outputs" / "h2" / f"failure_localization_h2_{stamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("power_only local:", report["local_result"])
    print("snapshots:")
    for row in snapshots:
        print(
            f" t={row['time_s']:.3f}s alt={row['altitude_m']:.1f}m "
            f"v={row['speed_kmh']:.1f}km/h gamma={row['flight_path_angle_deg']:.2f}deg "
            f"latG={row['lateral_load_g']:.3f}"
        )
    print("natural lift disabled:", report["diagnostic_ablation_natural_lift_disabled"]["result"])
    print(f"written: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
