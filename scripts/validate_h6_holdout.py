#!/usr/bin/env python3
"""Validate the frozen H6 effective plant on whole trajectories."""

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

from aim120_model.fin_dynamics import FinDynamicsParams  # noqa: E402
from aim120_model.fin_dynamics_replay import replay_attitude  # noqa: E402
from aim120_model.h6_utils import rms, utc_now_iso, write_json  # noqa: E402


def _trajectories(payload: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in payload.get("normalized_rows", []):
        groups[str(row.get("case_id", "unknown"))].append(dict(row))
    return [sorted(group, key=lambda row: float(row["time_s"])) for group in groups.values() if len(group) >= 2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "outputs" / "h6_fin_dynamics" / "h6_normalized_samples.json")
    parser.add_argument("--fit", type=Path, default=PROJECT_ROOT / "outputs" / "h6_fin_dynamics" / "moment_damping_fit.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "h6_fin_dynamics" / "holdout_report.json")
    args = parser.parse_args()
    if not args.input.exists() or not args.fit.exists():
        report = {"schema_version": 1, "generated_at_utc": utc_now_iso(), "status": "blocked_missing_input_or_fit", "holdout_pass": None}
        write_json(args.output, report)
        print(json.dumps(report, ensure_ascii=False))
        return 0
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    fit_payload = json.loads(args.fit.read_text(encoding="utf-8"))
    parameters = fit_payload.get("parameters")
    if not parameters:
        report = {"schema_version": 1, "generated_at_utc": utc_now_iso(), "status": "blocked_fit_parameters_null", "holdout_pass": None}
        write_json(args.output, report)
        print(json.dumps(report, ensure_ascii=False))
        return 0
    params = FinDynamicsParams(
        float(parameters["B_f_ref"]),
        float(parameters["K_beta_ref"]),
        float(parameters["C_r_ref"]),
        float(parameters.get("q_ref_pa", 1.0)),
        float(parameters.get("B_q_exponent", 0.0)),
        float(parameters.get("K_beta_q_exponent", 0.0)),
        float(parameters.get("C_r_q_exponent", 0.0)),
    )
    summaries: List[Dict[str, Any]] = []
    for trajectory in _trajectories(payload):
        replayed = replay_attitude(trajectory, params)
        angle_rmse = rms(row["psi_residual_rad"] for row in replayed)
        rate_rmse = rms(row["yaw_rate_residual_rad_s"] for row in replayed)
        summaries.append({
            "case_id": trajectory[0].get("case_id"),
            "sample_count": len(replayed),
            "angle_rmse_deg": math.degrees(angle_rmse) if angle_rmse is not None else None,
            "yaw_rate_rmse_rad_s": rate_rmse,
            "terminal_angle_error_deg": math.degrees(replayed[-1]["psi_residual_rad"]),
        })
    # H6 F6 itself is identified by its case/model label later; for generic
    # normalized data this report remains an explicit whole-trajectory check.
    holdout_pass = bool(summaries) and all(
        item["angle_rmse_deg"] is not None and item["angle_rmse_deg"] <= 0.5
        for item in summaries
    )
    report = {
        "schema_version": 1,
        "generated_at_utc": utc_now_iso(),
        "source_kind": "statshark_backend_timeseries",
        "frozen_parameters": params.as_dict(),
        "trajectory_summaries": summaries,
        "holdout_pass": holdout_pass,
        "status": "pass" if holdout_pass else "blocked_or_failed_threshold",
        "f6_must_not_be_used_for_refit": True,
    }
    write_json(args.output, report)
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
