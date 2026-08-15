#!/usr/bin/env python3
"""Audit H4 reference coverage before any full-envelope fit is allowed."""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from aim120_model.config import load_model_config
from aim120_model.h4_coverage import coverage_report, gravity_cancellation_audit


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


def _reference_integrity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("trajectory_id", "unknown"))].append(row)
    result: dict[str, Any] = {}
    for trajectory_id, values in grouped.items():
        masses = [float(row["mass_kg"]) for row in values if row.get("mass_kg") is not None and math.isfinite(float(row["mass_kg"]))]
        thrust = [abs(float(row.get("thrust_n", 0.0))) for row in values if row.get("thrust_n") is not None]
        result[trajectory_id] = {
            "sample_count": len(values),
            "mass_min_kg": min(masses) if masses else None,
            "mass_max_kg": max(masses) if masses else None,
            "mass_constant_within_1e-6": bool(masses) and max(masses) - min(masses) <= 1.0e-6,
            "max_abs_thrust_n": max(thrust) if thrust else None,
            "zero_thrust": bool(thrust) and max(thrust) <= 1.0e-6,
        }
    return result


def _sensitivity(config: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    filters = config["filters"]
    combinations = []
    for lateral in filters["lateral_load_thresholds_g"]:
        for alpha in filters["alpha_thresholds_deg"]:
            for q_min in filters["q_thresholds_pa"]:
                combinations.append({
                    "lateral_load_threshold_g": float(lateral),
                    "alpha_threshold_deg": float(alpha),
                    "q_min_pa": float(q_min),
                    "accepted_rows": len(rows),
                    "status": "pending_reference_data" if not rows else "requires_reingest_with_threshold",
                })
    return {
        "status": "blocked_missing_reference_data" if not rows else "pending_threshold_reingest",
        "combinations": combinations,
        "gravity_cancellation_thresholds": filters["gravity_cancellation_thresholds"],
        "note": "Threshold sensitivity cannot establish H4 stability until independent reference trajectories exist.",
    }


def main() -> int:
    output_dir = PROJECT_DIR / "outputs" / "h4_glide_drag"
    config = load_model_config(PROJECT_DIR / "configs" / "aim120a_h4_glide_drag_envelope.yaml")
    payload = json.loads((output_dir / "filtered_samples.json").read_text(encoding="utf-8"))
    source_manifest = json.loads((output_dir / "source_manifest.json").read_text(encoding="utf-8")) if (output_dir / "source_manifest.json").exists() else {}
    statshark_flag = bool(source_manifest.get("statshark_new_calculation_performed_this_run", False))
    all_rows = [dict(row) for row in payload.get("rows", [])]
    rows = [
        row for row in all_rows
        if row.get("accepted") and row.get("source_kind") == "statshark_reference"
    ]
    generated_at = datetime.now(timezone.utc).isoformat()
    coverage = coverage_report(
        rows,
        target_range=(float(config["target"]["mach_min"]), float(config["target"]["mach_max"])),
        minimum_overlap_width_mach=float(config["target"]["minimum_overlap_width_mach"]),
    )
    coverage.update({
        "schema_version": 4,
        "generated_at_utc": generated_at,
        "model_label": config["model_label"],
        "statshark_new_calculation_performed_this_run": statshark_flag,
        "h3_local_range_not_counted_as_h4_reference": {"min": 1.0144726856870405, "max": 3.0225389575417285},
        "reference_integrity": _reference_integrity(rows),
    })
    overlap = {
        "schema_version": 4,
        "generated_at_utc": generated_at,
        "model_label": config["model_label"],
        "status": "blocked_missing_reference_data" if not rows else "pending_inverse_cda",
        "trajectory_count": len({str(row.get("trajectory_id")) for row in rows}),
        "overlap_consistency": [],
        "required_before_fit": ["G1/G2 overlap", "G2/G3 overlap", "G4 altitude overlap", "G5 repeatability"],
    }
    cancellation = gravity_cancellation_audit(rows, float(config["atmosphere"]["gravity_mps2"]))
    cancellation.update({
        "schema_version": 4,
        "generated_at_utc": generated_at,
        "model_label": config["model_label"],
        "status": "blocked_missing_inverse_reference_samples" if not rows else "pending_inverse_cda",
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "coverage_report.json").write_text(json.dumps(_json_safe(coverage), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "overlap_consistency.json").write_text(json.dumps(_json_safe(overlap), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "inverse_cda_diagnostics.json").write_text(json.dumps(_json_safe(cancellation), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "sensitivity_report.json").write_text(json.dumps(_json_safe(_sensitivity(config, rows)), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"reference_rows={len(rows)} trajectories={coverage['trajectory_count']} status={coverage['support_status']}")
    print(f"written: {output_dir / 'coverage_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
