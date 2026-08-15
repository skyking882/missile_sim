#!/usr/bin/env python3
"""Run synthetic replay validation and gate real trajectory holdout validation."""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from aim120_model.atmosphere import StandardAtmosphere
from aim120_model.axial_replay import replay_trajectory, trajectory_replay_metrics
from aim120_model.config import load_model_config
from aim120_model.glide_drag_envelope import LogCdaEnvelope
from aim120_model.inverse_cda import estimate_inverse_cda, fit_log_cda_knots


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _rhs(speed: float, altitude_m: float, gamma_rad: float, mass_kg: float, envelope: LogCdaEnvelope, atmosphere: StandardAtmosphere, gravity_mps2: float) -> float:
    atm = atmosphere.sample(altitude_m)
    mach = speed / atm.speed_of_sound_mps
    q = 0.5 * atm.density_kg_m3 * speed * speed
    return -q * envelope.cda_m2(mach) / mass_kg - gravity_mps2 * math.sin(gamma_rad)


def _synthetic_trajectory(envelope: LogCdaEnvelope, noise_fraction: float) -> list[dict[str, Any]]:
    atmosphere = StandardAtmosphere()
    gravity = 9.80665
    altitude = 3000.0
    gamma = 0.0
    mass = 100.0
    # Use the replay's nominal 0.02 s integration interval so the no-noise
    # case tests model/units/signs rather than comparing two different RK4
    # truncation errors.
    times = [index * 0.02 for index in range(2001)]
    speed = 720.0
    rows: list[dict[str, Any]] = []
    for index, time_s in enumerate(times):
        noise = noise_fraction * math.sin(0.73 * index + 0.2)
        observed_speed = speed * (1.0 + noise)
        rows.append({
            "trajectory_id": "synthetic_G2",
            "case_id": "synthetic_G2",
            "source_kind": "synthetic_test",
            "time_s": time_s,
            "speed_mps": observed_speed,
            "altitude_m": altitude,
            "flight_path_angle_rad": gamma,
            "mass_kg": mass,
            "powered": False,
            "thrust_n": 0.0,
        })
        if index == len(times) - 1:
            break
        dt = times[index + 1] - time_s
        k1 = _rhs(speed, altitude, gamma, mass, envelope, atmosphere, gravity)
        k2 = _rhs(speed + 0.5 * dt * k1, altitude, gamma, mass, envelope, atmosphere, gravity)
        k3 = _rhs(speed + 0.5 * dt * k2, altitude, gamma, mass, envelope, atmosphere, gravity)
        k4 = _rhs(speed + dt * k3, altitude, gamma, mass, envelope, atmosphere, gravity)
        speed += dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return rows


def main() -> int:
    config = load_model_config(PROJECT_DIR / "configs" / "aim120a_h4_glide_drag_envelope.yaml")
    output_dir = PROJECT_DIR / "outputs" / "h4_glide_drag"
    source_manifest = json.loads((output_dir / "source_manifest.json").read_text(encoding="utf-8")) if (output_dir / "source_manifest.json").exists() else {}
    statshark_flag = bool(source_manifest.get("statshark_new_calculation_performed_this_run", False))
    envelope = LogCdaEnvelope.from_cda_knots(
        config["synthetic_validation"]["known_knots"],
        [0.010 + 0.0015 * math.exp(-((float(mach) - 1.05) / 0.18) ** 2) for mach in config["synthetic_validation"]["known_knots"]],
    )
    synthetic_cases: list[dict[str, Any]] = []
    for noise_fraction in config["synthetic_validation"]["noise_levels_fraction"]:
        rows = _synthetic_trajectory(envelope, float(noise_fraction))
        replayed = replay_trajectory(rows, envelope, max_step_s=0.02)
        synthetic_cases.append({
            "noise_fraction": float(noise_fraction),
            "source_kind": "synthetic_test",
            "trajectory_id": "synthetic_G2",
            "metrics": trajectory_replay_metrics(replayed),
            "first_sample": replayed[0],
            "last_sample": replayed[-1],
        })
    filtered = json.loads((output_dir / "filtered_samples.json").read_text(encoding="utf-8"))
    reference_rows = [
        row for row in filtered.get("rows", [])
        if row.get("accepted") and row.get("source_kind") == "statshark_reference"
    ]
    fit_path = output_dir / "cda_knots_fit.json"
    real_report = {
        "schema_version": 4,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_label": config["model_label"],
        "statshark_new_calculation_performed_this_run": statshark_flag,
        "status": "blocked_missing_reference_fit" if not reference_rows or not fit_path.exists() else "pending_partial_fit_holdout",
        "method": "whole_trajectory_holdout_only; no point_random_split",
        "reference_trajectory_count": len({str(row.get("trajectory_id")) for row in reference_rows}),
        "reason": "A real holdout replay requires an independently fitted H4-G0 envelope from reference trajectories.",
    }
    if reference_rows and fit_path.exists():
        fit_payload = json.loads(fit_path.read_text(encoding="utf-8"))
        holdout_candidates = config["trajectory_split"].get("predeclared_validation_ids", [])
        available_ids = sorted({str(row.get("trajectory_id")) for row in reference_rows})
        holdout_id = str(holdout_candidates[0]) if holdout_candidates else ("G5" if "G5" in available_ids else available_ids[-1])
        train_rows = [row for row in reference_rows if str(row.get("trajectory_id")) != holdout_id]
        holdout_rows = [row for row in reference_rows if str(row.get("trajectory_id")) == holdout_id]
        try:
            train_inverse = estimate_inverse_cda(train_rows, gravity_mps2=float(config["atmosphere"]["gravity_mps2"]))
            train_fit = fit_log_cda_knots(train_inverse, fit_payload["mach_knots"], minimum_samples_per_node=3)
            train_envelope = LogCdaEnvelope.from_cda_knots(train_fit["mach_knots"], train_fit["cda_knots_m2"])
            replayed = replay_trajectory(holdout_rows, train_envelope)
            metrics = trajectory_replay_metrics(replayed)
            relative_tolerance = 0.05
            real_report.update({
                "status": "partial_fit_holdout_pass" if metrics["speed_relative_rmse"] <= relative_tolerance else "partial_fit_holdout_fail",
                "holdout_trajectory_id": holdout_id,
                "training_trajectory_ids": sorted({str(row.get("trajectory_id")) for row in train_rows}),
                "training_fit_mach_knots": train_fit["mach_knots"],
                "metrics": metrics,
                "acceptance": {
                    "speed_relative_rmse_tolerance": relative_tolerance,
                    "speed_relative_rmse_pass": metrics["speed_relative_rmse"] <= relative_tolerance,
                },
                "boundary": "Holdout uses observed altitude and flight-path angle as exogenous inputs; it validates the axial replay only, not full 6-DoF guidance or backend equivalence.",
            })
        except (TypeError, ValueError) as exc:
            real_report.update({"status": "partial_fit_holdout_blocked", "holdout_trajectory_id": holdout_id, "error": str(exc)})
    synthetic_report = {
        "schema_version": 4,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_label": config["model_label"],
        "source_kind": "synthetic_test",
        "statshark_new_calculation_performed_this_run": statshark_flag,
        "known_envelope": {
            "mach_knots": list(envelope.mach_knots),
            "cda_knots_m2": [math.exp(value) for value in envelope.log_cda_knots],
            "interpolation": envelope.interpolation,
        },
        "cases": synthetic_cases,
        "acceptance": {
            "zero_noise_rmse_target_mps": 1.0e-6,
            "zero_noise_pass": synthetic_cases[0]["metrics"]["speed_rmse_mps"] <= 1.0e-6,
            "noise_response_monotonic": synthetic_cases[0]["metrics"]["speed_rmse_mps"] <= synthetic_cases[1]["metrics"]["speed_rmse_mps"] <= synthetic_cases[2]["metrics"]["speed_rmse_mps"],
        },
        "note": "Synthetic validation is a pipeline test and is not reference evidence or a cda_knots_fit artifact.",
    }
    replay_report = {
        "schema_version": 4,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_label": config["model_label"],
        "statshark_new_calculation_performed_this_run": statshark_flag,
        "real_reference_status": real_report["status"],
        "synthetic_status": "pass" if synthetic_report["acceptance"]["zero_noise_pass"] else "fail",
        "synthetic_cases": [
            {
                "noise_fraction": case["noise_fraction"],
                "metrics": case["metrics"],
            }
            for case in synthetic_cases
        ],
        "replay_boundary": "Observed altitude and flight-path angle are exogenous; this validates the axial glide model only, not full 6-DoF dynamics.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "trajectory_holdout_report.json").write_text(json.dumps(_json_safe(real_report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "trajectory_replay_report.json").write_text(json.dumps(_json_safe(replay_report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "synthetic_validation_report.json").write_text(json.dumps(_json_safe(synthetic_report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"synthetic_zero_noise_pass={synthetic_report['acceptance']['zero_noise_pass']} real_holdout_status={real_report['status']}")
    print(f"written: {output_dir / 'synthetic_validation_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
