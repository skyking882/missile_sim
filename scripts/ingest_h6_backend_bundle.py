#!/usr/bin/env python3
"""Ingest H6 raw StatShark bundles without silently dropping failures."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aim120_model.fin_force_inverse import normalize_capture_bundle  # noqa: E402
from aim120_model.h6_utils import sha256_file, utc_now_iso, write_json  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "h6_fin_dynamics"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "statshark_h6_fin_dynamics"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=RAW_DIR / "formal_capture_bundle.json")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        report = {
            "schema_version": 1,
            "generated_at_utc": utc_now_iso(),
            "source_kind": "statshark_backend_timeseries",
            "input_path": str(input_path.resolve()),
            "input_exists": False,
            "status": "blocked_no_backend_bundle",
            "reason": "F1-F6 raw request/response capture has not been supplied yet.",
            "calculate_actions_must_not_be_inferred_from_this_report": True,
        }
        write_json(output_dir / "schema_report.json", report)
        print(json.dumps(report, ensure_ascii=False))
        return 0
    try:
        bundle = json.loads(input_path.read_text(encoding="utf-8"))
        normalized = normalize_capture_bundle(bundle)
    except (OSError, ValueError, TypeError) as exc:
        report = {
            "schema_version": 1,
            "generated_at_utc": utc_now_iso(),
            "input_path": str(input_path.resolve()),
            "input_sha256": sha256_file(input_path),
            "status": "invalid_bundle",
            "error": repr(exc),
        }
        write_json(output_dir / "schema_report.json", report)
        print(json.dumps(report, ensure_ascii=False))
        return 1
    normalized_path = output_dir / "h6_normalized_samples.json"
    write_json(normalized_path, normalized)
    records = normalized["normalized_records"]
    ui_only_capture = bundle.get("raw_backend_response_available") is False or bundle.get("source_kind") == "statshark_ui_rendered_evidence"
    report: Dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": utc_now_iso(),
        "source_kind": bundle.get("source_kind", "statshark_backend_timeseries"),
        "input_path": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path),
        "record_count": len(records),
        "normalized_record_count": sum(1 for record in records if record.get("status") == "normalized"),
        "non_success_records_preserved": sum(1 for record in records if record.get("status") != "normalized"),
        "row_count": len(normalized["normalized_rows"]),
        "record_statuses": [
            {
                "case_id": record.get("case_id"),
                "model_id": record.get("model_id"),
                "status": record.get("status"),
                "capture_status": record.get("capture_status"),
                "issues": record.get("validation", {}).get("issues", []),
            }
            for record in records
        ],
        "required_backend_arrays": ["times", "missileX", "missileY", "missileZ", "missileSpeedMs", "angle", "yaw"],
        "raw_response_available": not ui_only_capture,
        "raw_response_retained": not ui_only_capture,
        "ui_rendered_evidence_retained": ui_only_capture,
        "capture_boundary": (
            "UI request fields, rendered plot facts, and frontend diagnostics are retained; "
            "response bodies/trajectory arrays were not exposed by the available browser capability."
            if ui_only_capture else None
        ),
        "status": (
            "blocked_schema_ui_only"
            if ui_only_capture else ("pass" if normalized["normalized_rows"] else "blocked_no_valid_motion_records")
        ),
    }
    write_json(output_dir / "schema_report.json", report)
    print(json.dumps({"status": report["status"], "normalized": str(normalized_path.resolve()), "rows": report["row_count"]}, ensure_ascii=False))
    return 0 if report["status"] in ("pass", "blocked_schema_ui_only") else 1


if __name__ == "__main__":
    raise SystemExit(main())
