#!/usr/bin/env python3
"""Run one Plan 8 tabulated-target scenario and write a new JSON artifact."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aim120_model.h2_simulator import H2Simulator  # noqa: E402
from aim120_model.math3d import norm  # noqa: E402
from aim120_model.trajectory import TabulatedTargetModel, TabulatedTrajectory  # noqa: E402
from missile_gui.library import scan_library  # noqa: E402


def _summary(result: dict, config: dict) -> dict:
    samples = result["samples"]
    terminal = samples[-1]
    speeds = [norm(tuple(sample["velocity_mps"])) for sample in samples]
    command_g = [math.hypot(*sample.get("commanded_acceleration_g", (0.0, 0.0))) for sample in samples]
    track_errors = [float(sample.get("track_position_error_m", 0.0)) for sample in samples]
    radar_tracks = [sample["time_s"] for sample in samples if sample.get("track_mode") == "radar_track"]
    radar_lost = [sample["time_s"] for sample in samples if sample.get("seeker_state") == "lost"]
    lock_loss = [sample["time_s"] for sample in samples if sample.get("track_mode") == "ins_search"]
    reacquire = []
    had_lock_loss = False
    for sample in samples:
        if sample.get("track_mode") == "ins_search":
            had_lock_loss = True
        elif had_lock_loss and sample.get("track_mode") == "radar_track":
            reacquire.append(sample["time_s"])
    reject_reasons = [str(sample["radar_reject_reason"]) for sample in samples if sample.get("radar_reject_reason")]
    event_name = {
        "fuse": "proximity_fuse",
        "impact": "hit",
        "max_distance": "max_range",
    }.get(result["event_type"], result["event_type"])
    return {
        "termination_event": event_name,
        "flight_time_s": terminal["time_s"],
        "terminal_distance_m": terminal["distance_to_target_m"],
        "terminal_speed_kmh": speeds[-1] * 3.6,
        "terminal_altitude_m": terminal["position_m"][1],
        "minimum_distance_m": min(sample["distance_to_target_m"] for sample in samples),
        "maximum_speed_kmh": max(speeds) * 3.6,
        "maximum_commanded_g": max(command_g),
        "maximum_actual_g": max(float(sample.get("actual_overload_g", 0.0)) for sample in samples),
        "maximum_track_error_m": max(track_errors),
        "track_mode": terminal.get("track_mode"),
        "seeker_state": terminal.get("seeker_state"),
        "seeker_display_state": terminal.get("seeker_display_state"),
        "observation_provider": result.get("observation_provider"),
        "first_radar_track_time_s": min(radar_tracks) if radar_tracks else None,
        "first_radar_lost_time_s": min(radar_lost) if radar_lost else None,
        "first_lock_loss_time_s": min(lock_loss) if lock_loss else None,
        "first_reacquire_time_s": min(reacquire) if reacquire else None,
        "last_radar_reject_reason": reject_reasons[-1] if reject_reasons else None,
        "observation_mode": result["observation_mode"],
        "burnout_time_s": sum(float(stage["duration_s"]) for stage in config["propulsion"]["stages"]),
    }


def run(args: argparse.Namespace) -> dict:
    profiles, errors = scan_library(ROOT / "missiles", ROOT)
    if errors:
        raise ValueError("missile library errors: " + " | ".join(errors[:3]))
    profile = next((item for item in profiles if item.get("missile_id") == args.missile), None)
    if profile is None:
        raise ValueError(f"unknown missile profile: {args.missile}")
    config = profile.get("_model_config")
    if not isinstance(config, dict):
        raise ValueError(f"missile profile is not runnable: {args.missile}")
    trajectory_path = (ROOT / args.trajectory).resolve() if not args.trajectory.is_absolute() else args.trajectory.resolve()
    trajectory = TabulatedTrajectory.from_csv(trajectory_path)
    target_model = TabulatedTargetModel(trajectory)
    initial_target = target_model.initial_state
    initial_distance = norm(initial_target.position)
    horizontal_azimuth_deg = math.degrees(math.atan2(initial_target.position[2], initial_target.position[0]))
    target_speed_kmh = norm(initial_target.velocity) * 3.6
    scenario = {
        "missile_id": args.missile,
        "trajectory": str(trajectory_path.relative_to(ROOT)) if trajectory_path.is_relative_to(ROOT) else str(trajectory_path),
        "observation_mode": args.observation_mode,
        "datalink_disconnect_time_s": args.datalink_disconnect_time_s,
        "simulation_dt_s": args.simulation_dt_s,
    }
    case = {
        "name": "radar_guidance_tabulated_target",
        "model_variant": "full",
        "observation_mode": args.observation_mode,
        "simulation_dt_s": args.simulation_dt_s,
        "datalink_enabled": True,
        "datalink_disconnect_time_s": args.datalink_disconnect_time_s,
        "inertial_drift_direction": (0.0, 0.0, 1.0),
        "_target_model": target_model,
        "initial_conditions": {
            "start_speed_kmh": target_speed_kmh,
            "launch_altitude_m": initial_target.position[1] + 500.0,
            "launch_angle_deg": 0.0,
            "launch_yaw_deg": 0.0,
            "target_speed_kmh": target_speed_kmh,
            "target_altitude_m": initial_target.position[1],
            "initial_target_distance_m": initial_distance,
            "target_azimuth_deg": horizontal_azimuth_deg,
            "target_course_deg": 0.0,
            "target_course_reference": "absolute_world",
            "target_constant_g_turn": 0.0,
            "target_vertical_course_deg": 0.0,
        },
    }
    runtime_config = copy.deepcopy(config)
    runtime_config["performance"]["lifetime_s"] = min(float(runtime_config["performance"]["lifetime_s"]), trajectory.end_time_s)
    result = H2Simulator(runtime_config).run(case)
    return {
        "schema_version": 1,
        "missile": {
            "id": profile.get("missile_id"),
            "name": profile.get("display_name"),
            "status": profile.get("model_status"),
        },
        "scenario": scenario,
        "summary": _summary(result, runtime_config),
        "samples": result["samples"],
        "model": {
            "label": result["model_label"],
            "aerodynamics": result["aero_model_version"],
            "geometry": result["force_geometry_version"],
            "control": result["control_model_version"],
            "integrator": result["integrator"],
            "time_step_s": result["time_step_s"],
            "guidance_update_hz": result["guidance_update_hz"],
            "observation_mode": result["observation_mode"],
            "observation_provider": result.get("observation_provider", "ideal_truth"),
            "lock_state_machine": "TRK->INS+SRC->TRK" if result["observation_mode"] == "sensor_track" else "not_applicable",
            "radar_model": (
                "deterministic_gate_candidate_v1"
                if result.get("observation_provider") == "radar_datalink_ins_v1"
                else "not_applicable"
                if result["observation_mode"] == "sensor_track"
                else "ideal_truth"
            ),
            "datalink_update": (
                "every_guidance_tick"
                if result.get("observation_provider") == "radar_datalink_ins_v1"
                else "not_applicable"
            ),
            "random_measurement_noise": False,
            "multipath_enabled": False,
            "sarh_model_enabled": False,
            "runtime_adapter": profile.get("_runtime_adapter"),
            "runtime_boundary": profile.get("_runtime_boundary"),
            "runtime_assumptions": profile.get("_runtime_assumptions", []),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a tabulated radar-guidance scenario")
    parser.add_argument("--missile", default="us_aim_120a")
    parser.add_argument("--trajectory", type=Path, default=Path("scenarios/trajectories/aim120_notch_39.csv"))
    parser.add_argument("--observation-mode", choices=("ideal_truth", "sensor_track"), default="ideal_truth")
    parser.add_argument("--datalink-disconnect-time-s", type=float, default=None)
    parser.add_argument("--simulation-dt-s", type=float, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing result: {output}")
    payload = run(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {len(payload['samples'])} samples to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
