#!/usr/bin/env python3
"""Re-ingest the acquired raw cases under the predeclared H4 filter grid."""

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
sys.path.insert(0, str(SCRIPT_DIR))

from aim120_model.config import load_model_config
from aim120_model.h4_coverage import coverage_report
from aim120_model.sample_filters import LowGFilterSettings
from ingest_h4_glide_reference import _load_input


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
    manifest = json.loads((output_dir / "source_manifest.json").read_text(encoding="utf-8"))
    input_paths = [Path(item["path"]) for item in manifest.get("input_artifacts", [])]
    base = manifest.get("filter_settings", {})
    combinations = []
    for lateral in config["filters"]["lateral_load_thresholds_g"]:
        for alpha in config["filters"]["alpha_thresholds_deg"]:
            for q_min in config["filters"]["q_thresholds_pa"]:
                settings = LowGFilterSettings(
                    lateral_load_threshold_g=float(lateral),
                    alpha_threshold_deg=float(alpha),
                    flight_path_threshold_deg=float(base.get("flight_path_threshold_deg", 8.0)),
                    q_min_pa=float(q_min),
                )
                rows = []
                for path in input_paths:
                    _artifact, input_rows = _load_input(path, "statshark_reference", settings)
                    rows.extend(row for row in input_rows if row.get("accepted"))
                coverage = coverage_report(
                    rows,
                    target_range=(float(config["target"]["mach_min"]), float(config["target"]["mach_max"])),
                    minimum_overlap_width_mach=float(config["target"]["minimum_overlap_width_mach"]),
                )
                combinations.append({
                    "filter_settings": settings.to_dict(),
                    "accepted_rows": len(rows),
                    "trajectory_count": coverage["trajectory_count"],
                    "actual_direct_support_range": coverage["actual_direct_support_range"],
                    "missing_target_ranges": coverage["missing_target_ranges"],
                    "adjacent_overlap": coverage["adjacent_overlap"],
                    "status": "audited",
                })
    payload = {
        "schema_version": 4,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_label": config["model_label"],
        "statshark_new_calculation_performed_this_run": bool(manifest.get("statshark_new_calculation_performed_this_run", False)),
        "status": "audited",
        "method": "re-ingest of the same acquired raw labels under predeclared alpha, lateral-load, and dynamic-pressure thresholds; no new StatShark calculation",
        "combinations": combinations,
    }
    (output_dir / "sensitivity_report.json").write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"sensitivity_combinations={len(combinations)} status={payload['status']}")
    print(f"written: {output_dir / 'sensitivity_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
