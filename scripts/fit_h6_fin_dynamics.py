#!/usr/bin/env python3
"""Fit the effective H6 angular plant only after normalized backend data exist."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aim120_model.fin_dynamics_replay import fit_full_trajectory  # noqa: E402
from aim120_model.h6_utils import utc_now_iso, write_json  # noqa: E402


def _load_trajectories(path: Path) -> List[List[Dict[str, Any]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("normalized_rows", [])
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("case_id", row.get("model_id", "unknown")))].append(dict(row))
    return [sorted(group, key=lambda row: float(row["time_s"])) for group in groups.values() if len(group) >= 2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "outputs" / "h6_fin_dynamics" / "h6_normalized_samples.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "h6_fin_dynamics" / "moment_damping_fit.json")
    parser.add_argument("--arm-m", type=float, default=0.175)
    args = parser.parse_args()
    if not args.input.exists():
        report = {
            "schema_version": 1,
            "generated_at_utc": utc_now_iso(),
            "fit_method": "full_trajectory_replay_with_derivative_seed",
            "status": "blocked_no_normalized_samples",
            "parameters": None,
        }
        write_json(args.output, report)
        print(json.dumps(report, ensure_ascii=False))
        return 0
    trajectories = _load_trajectories(args.input)
    if not trajectories:
        report = {
            "schema_version": 1,
            "generated_at_utc": utc_now_iso(),
            "fit_method": "full_trajectory_replay_with_derivative_seed",
            "status": "blocked_no_complete_trajectories",
            "parameters": None,
        }
        write_json(args.output, report)
        print(json.dumps(report, ensure_ascii=False))
        return 0
    try:
        fit = fit_full_trajectory(trajectories, distance_cm_to_stabilizer_m=args.arm_m)
        params = fit["parameters"]
        report = {
            "schema_version": 1,
            "generated_at_utc": utc_now_iso(),
            "source_kind": "statshark_backend_timeseries",
            "trajectory_count": len(trajectories),
            "fit_method": fit["fit_method"],
            "parameters": params.as_dict(),
            "seed": fit["seed"].as_dict(),
            "seed_report": fit["seed_report"],
            "angle_rmse_rad": fit["angle_rmse_rad"],
            "yaw_rate_rmse_rad_s": fit["yaw_rate_rmse_rad_s"],
            "identifiability": fit["identifiability"],
            "parameter_status": "effective_only_pending_holdout",
            "status": "fit_complete",
        }
    except (ValueError, TypeError, KeyError) as exc:
        report = {
            "schema_version": 1,
            "generated_at_utc": utc_now_iso(),
            "status": "fit_error",
            "error": repr(exc),
            "parameters": None,
        }
    write_json(args.output, report)
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve())}, ensure_ascii=False))
    return 0 if report["status"] == "fit_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
