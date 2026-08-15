#!/usr/bin/env python3
"""Run the H6 synthetic gates before any StatShark Calculate action."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aim120_model.fin_dynamics import FinDynamicsParams
from aim120_model.fin_dynamics_replay import fit_full_trajectory, replay_attitude
from aim120_model.fin_energy import augment_force_direction, compare_force_directions, fit_fin_drag_candidates
from aim120_model.fin_force_inverse import normalize_backend_result
from aim120_model.h6_utils import rms, utc_now_iso, wrap_angle, write_json


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "h6_fin_dynamics"
MASS_KG = 147.87
ARM_M = 0.175
DT_S = 0.02
Q_REF = 55000.0
TRUTH = FinDynamicsParams(0.060, 0.50, 0.80, q_ref_pa=Q_REF)


def _integrate_path(times: Sequence[float], chi_values: Sequence[float], speed_mps: float) -> List[Tuple[float, float, float]]:
    positions = [(0.0, 0.0, 0.0)]
    for index in range(1, len(times)):
        dt = float(times[index]) - float(times[index - 1])
        chi0 = float(chi_values[index - 1])
        chi1 = float(chi_values[index])
        vx = speed_mps * 0.5 * (math.cos(chi0) + math.cos(chi1))
        vz = speed_mps * 0.5 * (math.sin(chi0) + math.sin(chi1))
        x0, y0, z0 = positions[-1]
        positions.append((x0 + dt * vx, y0, z0 + dt * vz))
    return positions


def build_synthetic_trajectory(force_scale: float, sign: float, q_pa: float, case_id: str) -> List[Dict[str, Any]]:
    times = [index * DT_S for index in range(401)]
    speed = 300.0
    accelerations = [
        sign * 35.0 * force_scale * (math.sin(0.8 * time) + 0.45 * math.sin(1.7 * time + 0.25))
        for time in times
    ]
    chi_values = [0.0]
    for left, right in zip(accelerations, accelerations[1:]):
        chi_values.append(chi_values[-1] + DT_S * 0.5 * (left + right) / speed)
    positions = _integrate_path(times, chi_values, speed)
    forcing_rows = [
        {
            "case_id": case_id,
            "model_id": case_id,
            "source_kind": "synthetic_test",
            "time_s": time,
            "x_m": positions[index][0],
            "y_m": positions[index][1],
            "z_m": positions[index][2],
            "speed_mps": speed,
            "mass_kg": MASS_KG,
            "pitch_rad": 0.0,
            "flight_path_yaw_rad": chi_values[index],
            "dynamic_pressure_pa": q_pa,
            "fin_normal_accel_mps2": accelerations[index],
            "fin_force_n": MASS_KG * accelerations[index],
            "body_normal_accel_mps2": 0.0,
            "normal_accel_yaw_mps2": accelerations[index],
            "thrust_n": 0.0,
            "drag_n": 0.0,
        }
        for index, time in enumerate(times)
    ]
    replayed = replay_attitude(forcing_rows, TRUTH, distance_cm_to_stabilizer_m=ARM_M, max_step_s=DT_S)
    for source, attitude in zip(forcing_rows, replayed):
        source["yaw_rad"] = float(attitude["predicted_psi_rad"])
        source["yaw_rate_rad_s"] = float(attitude["predicted_yaw_rate_rad_s"])
        source["beta_yaw_rad"] = wrap_angle(source["yaw_rad"] - source["flight_path_yaw_rad"])
    return forcing_rows


def backend_response_from_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "times": [row["time_s"] for row in rows],
        "missileX": [row["x_m"] for row in rows],
        "missileY": [row["y_m"] for row in rows],
        "missileZ": [row["z_m"] for row in rows],
        "missileSpeedMs": [row["speed_mps"] for row in rows],
        "angle": [math.degrees(row["pitch_rad"]) for row in rows],
        "yaw": [math.degrees(row["yaw_rad"]) for row in rows],
        "currentMass": [row["mass_kg"] for row in rows],
        "currentThrust": [0.0 for _ in rows],
        "drag": [0.0 for _ in rows],
        "Cd": [0.0 for _ in rows],
        "currentG": [row["normal_accel_yaw_mps2"] / 9.80665 for row in rows],
        "aCmdYaw": [row["normal_accel_yaw_mps2"] / 9.80665 for row in rows],
    }


def _relative_error(actual: float, expected: float) -> float:
    return abs(float(actual) - float(expected)) / max(abs(float(expected)), 1.0e-12)


def build_report() -> Dict[str, Any]:
    trajectories = [
        build_synthetic_trajectory(0.5, 1.0, 45000.0, "SYN_F050_QLOW"),
        build_synthetic_trajectory(1.0, 1.0, 55000.0, "SYN_F100_QREF"),
        build_synthetic_trajectory(1.5, -1.0, 90000.0, "SYN_F150_QHIGH_NEG"),
    ]
    train_fit = fit_full_trajectory(trajectories[:2], distance_cm_to_stabilizer_m=ARM_M, max_step_s=DT_S)
    fitted = train_fit["parameters"]
    holdout_replay = replay_attitude(trajectories[2], fitted, distance_cm_to_stabilizer_m=ARM_M, max_step_s=DT_S)
    holdout_angle_rmse = rms(row["psi_residual_rad"] for row in holdout_replay)
    normalized = normalize_backend_result(
        backend_response_from_rows(trajectories[1]),
        model_id="H6_SYN_F100",
        case_id="SYN_BACKEND",
        angle_unit="deg",
        body={"mass_kg": MASS_KG, "cy_k": 0.0},
    )
    normalized_rows = normalized["rows"]
    direction_rows = augment_force_direction(trajectories[1][::20], "flow_normal")
    direction_rows_body = augment_force_direction(trajectories[1][::20], "body_normal")
    direction_comparison = compare_force_directions(trajectories[1][::20])
    # Build an exactly declared quadratic residual for the model selector.
    drag_rows = []
    for row in trajectories[0][::20] + trajectories[1][::20] + trajectories[2][::20]:
        item = dict(row)
        item["extra_drag_residual_n"] = 0.37 * float(row["fin_force_n"]) ** 2 / float(row["dynamic_pressure_pa"])
        drag_rows.append(item)
    drag_fit = fit_fin_drag_candidates(drag_rows)
    flow_projection_max = max(abs(float(row["fin_axial_projection_n"])) for row in direction_rows)
    body_projection_max = max(abs(float(row["fin_axial_projection_n"])) for row in direction_rows_body)
    checks = {
        "known_B_f_recovered": _relative_error(fitted.b_f_ref, TRUTH.b_f_ref) <= 0.08,
        "known_K_beta_recovered": _relative_error(fitted.k_beta_ref, TRUTH.k_beta_ref) <= 0.08,
        "known_C_r_recovered": _relative_error(fitted.c_r_ref, TRUTH.c_r_ref) <= 0.08,
        "design_full_rank": bool(train_fit["identifiability"]["design_full_rank"]),
        "design_correlation_gate": bool(train_fit["identifiability"]["correlation_gate"]),
        "backend_schema_normalized": normalized["status"] == "normalized" and len(normalized_rows) == len(trajectories[1]),
        "flow_normal_has_zero_axial_work": flow_projection_max <= 1.0e-8,
        "body_normal_has_nonzero_beta_projection": body_projection_max > 1.0e-3,
        "D0_not_falsely_positive_on_zero_residual": fit_fin_drag_candidates([
            dict(row, extra_drag_residual_n=0.0) for row in drag_rows
        ])["candidate_fits"]["none"]["rmse_n"] == 0.0,
        "D1_recovers_declared_quadratic_signal": (
            drag_fit["candidate_fits"]["fin_load_squared"]["status"] == "fit_complete"
            and _relative_error(drag_fit["candidate_fits"]["fin_load_squared"]["coefficients"][0], 0.37) <= 0.05
        ),
        "whole_trajectory_holdout_replay": holdout_angle_rmse is not None and math.degrees(holdout_angle_rmse) <= 0.5,
        "yaw_sign_preserved": trajectories[2][100]["fin_normal_accel_mps2"] < 0.0 and trajectories[2][100]["flight_path_yaw_rad"] < 0.0,
    }
    return {
        "schema_version": 1,
        "generated_at_utc": utc_now_iso(),
        "model_label": "local_candidate_H6_fin_plant_v1",
        "source_kind": "synthetic_test",
        "new_statshark_calculate_performed_this_run": False,
        "truth_parameters": TRUTH.as_dict(),
        "fit_parameters_from_train": fitted.as_dict(),
        "train_fit": {
            "trajectory_count": len(trajectories[:2]),
            "angle_rmse_rad": train_fit["angle_rmse_rad"],
            "yaw_rate_rmse_rad_s": train_fit["yaw_rate_rmse_rad_s"],
            "identifiability": train_fit["identifiability"],
        },
        "holdout": {
            "case_id": trajectories[2][0]["case_id"],
            "angle_rmse_deg": math.degrees(holdout_angle_rmse) if holdout_angle_rmse is not None else None,
            "sample_count": len(holdout_replay),
        },
        "backend_ingest_smoke": {
            "status": normalized["status"],
            "row_count": len(normalized_rows),
            "validation": normalized["validation"],
        },
        "direction_comparison": direction_comparison,
        "energy_model_diagnostics": {
            "flow_projection_max_n": flow_projection_max,
            "body_projection_max_n": body_projection_max,
        },
        "drag_model_diagnostics": drag_fit,
        "acceptance_checks": checks,
        "status": "synthetic_identifiability_pass" if all(checks.values()) else "synthetic_identifiability_failed",
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report()
    output = OUTPUT_DIR / "synthetic_identifiability_report.json"
    write_json(output, report)
    print(json.dumps({"status": report["status"], "output": str(output.resolve())}, ensure_ascii=False))
    return 0 if report["status"] == "synthetic_identifiability_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
