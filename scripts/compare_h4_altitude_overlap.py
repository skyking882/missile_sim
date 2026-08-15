#!/usr/bin/env python3
"""Gate the different-height H4 comparison on G2/G4 reference data."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def main() -> int:
    output_dir = PROJECT_DIR / "outputs" / "h4_glide_drag"
    filtered = json.loads((output_dir / "filtered_samples.json").read_text(encoding="utf-8"))
    source_manifest = json.loads((output_dir / "source_manifest.json").read_text(encoding="utf-8")) if (output_dir / "source_manifest.json").exists() else {}
    statshark_flag = bool(source_manifest.get("statshark_new_calculation_performed_this_run", False))
    rows = [
        row for row in filtered.get("rows", [])
        if row.get("accepted") and row.get("source_kind") == "statshark_reference"
    ]
    trajectory_ids = sorted({str(row.get("trajectory_id")) for row in rows})
    inverse_payload = json.loads((output_dir / "inverse_cda_diagnostics.json").read_text(encoding="utf-8")) if (output_dir / "inverse_cda_diagnostics.json").exists() else {}
    inverse_rows = [
        row for row in inverse_payload.get("rows", [])
        if row.get("inverse_cda_valid") and str(row.get("trajectory_id")) in {"G2", "G4"}
    ]
    g2 = [row for row in inverse_rows if str(row.get("trajectory_id")) == "G2"]
    g4 = [row for row in inverse_rows if str(row.get("trajectory_id")) == "G4"]
    pairs = []
    for left in g2:
        if not g4:
            continue
        right = min(g4, key=lambda row: abs(float(row["mach"]) - float(left["mach"])))
        mach_delta = abs(float(left["mach"]) - float(right["mach"]))
        if mach_delta <= 0.02:
            cda_left = float(left["inverse_cda_m2"])
            cda_right = float(right["inverse_cda_m2"])
            pairs.append({
                "g2_time_s": left.get("time_s"),
                "g4_time_s": right.get("time_s"),
                "g2_mach": left.get("mach"),
                "g4_mach": right.get("mach"),
                "mach_delta": mach_delta,
                "altitude_delta_m": float(right["altitude_m"]) - float(left["altitude_m"]),
                "cda_delta_m2": cda_right - cda_left,
                "cda_ratio_g4_over_g2": cda_right / cda_left if cda_left > 0.0 else None,
            })
    g2_raw = {round(float(row["time_s"]), 3): row for row in rows if str(row.get("trajectory_id")) == "G2"}
    g5_raw = {round(float(row["time_s"]), 3): row for row in rows if str(row.get("trajectory_id")) == "G5"}
    repeat_rows = []
    for time_s in sorted(set(g2_raw) & set(g5_raw)):
        left, right = g2_raw[time_s], g5_raw[time_s]
        repeat_rows.append({
            "time_s": time_s,
            "speed_delta_mps": float(right["speed_mps"]) - float(left["speed_mps"]),
            "altitude_delta_m": float(right["altitude_m"]) - float(left["altitude_m"]),
            "target_distance_delta_m": float(right.get("target_distance_m", 0.0)) - float(left.get("target_distance_m", 0.0)),
            "mach_delta": float(right["mach"]) - float(left["mach"]),
        })
    result = {
        "schema_version": 4,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_label": "local_candidate_H4_glide_drag_envelope",
        "statshark_new_calculation_performed_this_run": statshark_flag,
        "status": "height_pairing_available" if pairs else ("blocked_missing_G4_reference" if len(trajectory_ids) < 2 else "pending_inverse_cda_and_height_pairing"),
        "required_cases": ["G2", "G4"],
        "reference_trajectory_ids": trajectory_ids,
        "comparison": pairs,
        "pairing_summary": {
            "pair_count": len(pairs),
            "mean_abs_cda_delta_m2": sum(abs(float(row["cda_delta_m2"])) for row in pairs) / len(pairs) if pairs else None,
            "mean_cda_ratio_g4_over_g2": sum(float(row["cda_ratio_g4_over_g2"]) for row in pairs) / len(pairs) if pairs else None,
        },
        "interpretation_boundary": "Do not add an environment parameter until same-Mach different-height glide residuals are independently observed and replay-validated.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "altitude_overlap_report.json").write_text(json.dumps(_json_safe(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    repeatability = {
        "schema_version": 4,
        "generated_at_utc": result["generated_at_utc"],
        "model_label": result["model_label"],
        "statshark_new_calculation_performed_this_run": statshark_flag,
        "reference_pair": ["G2", "G5"],
        "matched_sample_count": len(repeat_rows),
        "max_abs_differences": {
            field: max((abs(float(row[field])) for row in repeat_rows), default=None)
            for field in ("speed_delta_mps", "altitude_delta_m", "target_distance_delta_m", "mach_delta")
        },
        "rows": repeat_rows,
        "status": "repeatable_at_visible_precision" if repeat_rows and all(abs(float(row[field])) <= 1.0e-9 for row in repeat_rows for field in ("speed_delta_mps", "altitude_delta_m", "target_distance_delta_m", "mach_delta")) else "repeatability_needs_review",
        "note": "G5 is the exact-repeat input case for G2; equality is assessed on the extracted visible labels, not hidden solver state.",
    }
    (output_dir / "repeatability_report.json").write_text(json.dumps(_json_safe(repeatability), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"status={result['status']} reference_trajectories={len(trajectory_ids)}")
    print(f"written: {output_dir / 'altitude_overlap_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
