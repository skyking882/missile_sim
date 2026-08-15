#!/usr/bin/env python3
"""Produce H6 axis-map and force-scale diagnostics from normalized rows."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aim120_model.h6_utils import finite, utc_now_iso, write_json  # noqa: E402


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "normalized_rows" in payload:
        return [dict(row) for row in payload["normalized_rows"]]
    return [dict(row) for record in payload.get("normalized_records", []) for row in record.get("rows", [])]


def _summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[str(row.get("model_id", "unknown"))].append(row)
    output: Dict[str, Any] = {}
    for model_id, group in sorted(by_model.items()):
        accelerations = [float(row["fin_normal_accel_mps2"]) for row in group if finite(row.get("fin_normal_accel_mps2"))]
        lateral = [float(row["normal_accel_yaw_mps2"]) for row in group if finite(row.get("normal_accel_yaw_mps2"))]
        output[model_id] = {
            "sample_count": len(group),
            "fin_accel_peak_mps2": max((abs(value) for value in accelerations), default=None),
            "fin_accel_mean_abs_mps2": sum(abs(value) for value in accelerations) / len(accelerations) if accelerations else None,
            "yaw_normal_peak_mps2": max((abs(value) for value in lateral), default=None),
            "yaw_response_sign": "positive" if sum(lateral) > 0.0 else "negative" if sum(lateral) < 0.0 else "zero_or_mixed",
            "case_ids": sorted({str(row.get("case_id")) for row in group}),
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "outputs" / "h6_fin_dynamics" / "h6_normalized_samples.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "h6_fin_dynamics" / "force_scale_report.json")
    args = parser.parse_args()
    if not args.input.exists():
        report = {"status": "blocked_no_normalized_samples", "generated_at_utc": utc_now_iso()}
        write_json(args.output, report)
        print(json.dumps(report, ensure_ascii=False))
        return 0
    rows = _load_rows(args.input)
    report: Dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": utc_now_iso(),
        "source_kind": "statshark_backend_timeseries",
        "row_count": len(rows),
        "model_summaries": _summary(rows),
        "force_scale_candidates": ["constant", "q_over_q_ref", "q_over_q_plus_q0", "narrow_mach_piecewise"],
        "status": "pass_diagnostic" if rows else "blocked_no_rows",
        "formal_parameter_status": "null_until_excitation_and_coverage_gates_pass",
    }
    write_json(args.output, report)
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
