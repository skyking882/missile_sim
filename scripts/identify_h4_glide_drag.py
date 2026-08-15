#!/usr/bin/env python3
"""Gate H4-G0 fitting on independent reference coverage."""

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

from aim120_model.config import load_model_config
from aim120_model.inverse_cda import estimate_inverse_cda, fit_log_cda_knots, summarize_inverse_cda


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def main() -> int:
    config = load_model_config(PROJECT_DIR / "configs" / "aim120a_h4_glide_drag_envelope.yaml")
    output_dir = PROJECT_DIR / "outputs" / "h4_glide_drag"
    filtered = json.loads((output_dir / "filtered_samples.json").read_text(encoding="utf-8"))
    source_manifest = json.loads((output_dir / "source_manifest.json").read_text(encoding="utf-8")) if (output_dir / "source_manifest.json").exists() else {}
    statshark_flag = bool(source_manifest.get("statshark_new_calculation_performed_this_run", False))
    rows = [
        row for row in filtered.get("rows", [])
        if row.get("accepted") and row.get("source_kind") == "statshark_reference"
    ]
    coverage = json.loads((output_dir / "coverage_report.json").read_text(encoding="utf-8")) if (output_dir / "coverage_report.json").exists() else {}
    now = datetime.now(timezone.utc).isoformat()
    if not rows:
        report = {
            "schema_version": 4,
            "generated_at_utc": now,
            "model_label": config["model_label"],
            "status": "blocked_missing_statshark_reference_time_series",
            "fit_level": "H4-G0",
            "statshark_new_calculation_performed_this_run": statshark_flag,
            "cda_knots_fit_generated": False,
            "reason": "Plan 4 forbids manufacturing cda_knots_fit.json from H3 local or synthetic data.",
            "coverage_snapshot": coverage,
            "required_inputs": ["G1", "G2", "G3", "G4", "G5 independent glide trajectories"],
            "missing_data": [
                "independent statshark_reference time series",
                "M0.2-1.0 direct support",
                "M3.0-4.5 direct support",
                "G1/G2/G3 overlap",
                "G4 different-height overlap",
                "G5 repeatability or complete trajectory holdout",
            ],
        }
        (output_dir / "model_selection_report.json").write_text(json.dumps(_json_safe(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("H4-G0 fit blocked: no independent statshark_reference samples; cda_knots_fit.json not created")
        return 0

    inverse_rows = estimate_inverse_cda(rows, gravity_mps2=float(config["atmosphere"]["gravity_mps2"]))
    inverse_summary = summarize_inverse_cda(inverse_rows)
    gravity_ratios = [
        abs(float(row["gravity_axial_mps2"])) / max(abs(float(row["axial_drag_accel_mps2"])), 1.0e-9)
        for row in inverse_rows
        if row.get("inverse_cda_valid") and row.get("gravity_axial_mps2") is not None and row.get("axial_drag_accel_mps2") is not None
    ]
    inverse_payload = {
        "schema_version": 4,
        "generated_at_utc": now,
        "model_label": config["model_label"],
        "method": "local least-squares speed slope over up to five adjacent visible samples; m*dV/dt = -q*CdA - m*g*sin(gamma)",
        "statshark_new_calculation_performed_this_run": statshark_flag,
        "rows": inverse_rows,
        "by_trajectory": inverse_summary,
        "gravity_cancellation": {
            "sample_count": len(gravity_ratios),
            "mean_ratio": sum(gravity_ratios) / len(gravity_ratios) if gravity_ratios else None,
            "max_ratio": max(gravity_ratios) if gravity_ratios else None,
            "fraction_above_1": sum(1 for value in gravity_ratios if value > 1.0) / len(gravity_ratios) if gravity_ratios else None,
            "fraction_above_3": sum(1 for value in gravity_ratios if value > 3.0) / len(gravity_ratios) if gravity_ratios else None,
        },
        "status": "inverse_diagnostics_available" if any(row.get("inverse_cda_valid") for row in inverse_rows) else "blocked_no_positive_inverse_cda",
        "interpretation_boundary": "Inverse CdA is an effective diagnostic of the visible trajectory and does not reveal the StatShark backend solver.",
    }
    (output_dir / "inverse_cda_diagnostics.json").write_text(json.dumps(_json_safe(inverse_payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    candidate_knots = [float(value) for value in config["target"]["mach_knots"]]
    fit_payload = None
    fit_error = None
    try:
        fit = fit_log_cda_knots(inverse_rows, candidate_knots, minimum_samples_per_node=3)
        fit_payload = {
            "schema_version": 4,
            "generated_at_utc": now,
            "model_label": config["model_label"],
            "fit_level": "H4-G0-partial",
            "status": "partial_direct_support",
            "statshark_new_calculation_performed_this_run": statshark_flag,
            "interpolation": config["model"]["interpolation"],
            "positivity_parameterization": config["model"]["positivity_parameterization"],
            "outside_support_policy": config["model"]["outside_support_policy"],
            "mach_knots": fit["mach_knots"],
            "cda_knots_m2": fit["cda_knots_m2"],
            "direct_inverse_mach_range": {
                "min": min(float(row["mach"]) for row in inverse_rows if row.get("inverse_cda_valid")),
                "max": max(float(row["mach"]) for row in inverse_rows if row.get("inverse_cda_valid")),
            },
            "fit_support_mach_range": {"min": min(fit["mach_knots"]), "max": max(fit["mach_knots"])},
            "valid_inverse_sample_count": fit["valid_inverse_sample_count"],
            "valid_inverse_trajectory_count": fit["valid_inverse_trajectory_count"],
            "node_records": fit["node_records"],
            "coverage_gate_snapshot": coverage,
            "fit_boundary": "This is a partial H4-G0 effective CdA fit. Mach gaps below the direct inverse range and any endpoint outside the knot range are labeled extrapolation and are not confirmed envelope support.",
        }
        (output_dir / "cda_knots_fit.json").write_text(json.dumps(_json_safe(fit_payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (TypeError, ValueError) as exc:
        fit_error = str(exc)

    report = {
        "schema_version": 4,
        "generated_at_utc": now,
        "model_label": config["model_label"],
        "status": "partial_fit_generated_not_full_target_envelope" if fit_payload else "fit_blocked_insufficient_inverse_nodes",
        "fit_level": "H4-G0-partial" if fit_payload else "H4-G0",
        "statshark_new_calculation_performed_this_run": statshark_flag,
        "cda_knots_fit_generated": bool(fit_payload),
        "reference_sample_count": len(rows),
        "inverse_diagnostics_status": inverse_payload["status"],
        "inverse_summary": inverse_summary,
        "fit_error": fit_error,
        "coverage_snapshot": coverage,
        "fit_boundary": "The generated model is usable only within its fitted knot range for this audit; it must not be presented as a validated M0.2-4.5 envelope.",
    }
    (output_dir / "model_selection_report.json").write_text(json.dumps(_json_safe(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if fit_payload:
        print(
            "H4-G0 partial fit generated: "
            f"knots={len(fit_payload['mach_knots'])} "
            f"mach={fit_payload['fit_support_mach_range']['min']:.2f}-{fit_payload['fit_support_mach_range']['max']:.2f}"
        )
    else:
        print(f"H4-G0 fit blocked: {fit_error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
