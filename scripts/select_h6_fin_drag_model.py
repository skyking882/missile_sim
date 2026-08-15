#!/usr/bin/env python3
"""Select H6-D0/D1/D2 diagnostics, preserving the projection-first gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aim120_model.fin_energy import augment_force_direction, compare_force_directions, energy_balance, fit_fin_drag_candidates  # noqa: E402
from aim120_model.h6_utils import utc_now_iso, write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "outputs" / "h6_fin_dynamics" / "h6_normalized_samples.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "h6_fin_dynamics" / "fin_drag_model_comparison.json")
    args = parser.parse_args()
    if not args.input.exists():
        report = {
            "schema_version": 1,
            "generated_at_utc": utc_now_iso(),
            "status": "blocked_no_normalized_samples",
            "selected_model": None,
        }
        write_json(args.output, report)
        print(json.dumps(report, ensure_ascii=False))
        return 0
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    rows = [dict(row) for row in payload.get("normalized_rows", [])]
    if not rows:
        report = {"schema_version": 1, "generated_at_utc": utc_now_iso(), "status": "blocked_no_rows", "selected_model": None}
        write_json(args.output, report)
        print(json.dumps(report, ensure_ascii=False))
        return 0
    # The raw row table is not overwritten; each direction comparison is a
    # derived object and the extra-drag selector remains gated by holdout data.
    flow_energy = energy_balance(augment_force_direction(rows, "flow_normal"))
    body_energy = energy_balance(augment_force_direction(rows, "body_normal"))
    report = {
        "schema_version": 1,
        "generated_at_utc": utc_now_iso(),
        "source_kind": "statshark_backend_timeseries",
        "force_direction_comparison": compare_force_directions(rows),
        "flow_normal_energy": {"sample_count": len(flow_energy), "computed_residuals": sum(row.get("energy_balance_status") == "computed" for row in flow_energy)},
        "body_normal_energy": {"sample_count": len(body_energy), "computed_residuals": sum(row.get("energy_balance_status") == "computed" for row in body_energy)},
        "nested_drag_candidates": fit_fin_drag_candidates(rows),
        "selected_model": "H6-D0_projection_only_pending_training_and_holdout",
        "independent_fin_drag_status": "null_until_same_sign_two_trajectory_and_holdout_gate",
        "status": "diagnostic_complete_projection_gate_pending",
    }
    write_json(args.output, report)
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
