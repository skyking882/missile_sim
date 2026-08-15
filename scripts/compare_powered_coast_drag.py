#!/usr/bin/env python3
"""Compare inverse CdA between powered and coast samples in shared bins."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from aim120_model.config import load_model_config


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _bin(value: float, width: float) -> float:
    return math.floor(float(value) / width) * width


def _group_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean_cda_m2": None, "std_cda_m2": None}
    return {
        "count": len(values),
        "mean_cda_m2": mean(values),
        "std_cda_m2": pstdev(values) if len(values) > 1 else 0.0,
    }


def build_comparison(rows: list[Mapping[str, Any]], mach_bin_width: float = 0.2, altitude_bin_width: float = 500.0) -> dict[str, Any]:
    bins: dict[tuple[float, float], dict[str, list[float]]] = defaultdict(lambda: {"powered": [], "coast": []})
    for row in rows:
        if not row.get("accepted"):
            continue
        cda = row.get("observed_cda_m2")
        mach = row.get("mach")
        altitude = row.get("altitude_m")
        if cda is None or mach is None or altitude is None:
            continue
        if not all(math.isfinite(float(value)) for value in (cda, mach, altitude)):
            continue
        key = (_bin(float(mach), mach_bin_width), _bin(float(altitude), altitude_bin_width))
        state = "powered" if bool(row.get("powered", False)) else "coast"
        bins[key][state].append(float(cda))

    comparisons: list[dict[str, Any]] = []
    for (mach_start, altitude_start), group in sorted(bins.items()):
        powered = _group_stats(group["powered"])
        coast = _group_stats(group["coast"])
        if powered["count"] and coast["count"]:
            powered_mean = float(powered["mean_cda_m2"])
            coast_mean = float(coast["mean_cda_m2"])
            comparisons.append({
                "mach_bin_start": mach_start,
                "mach_bin_end": mach_start + mach_bin_width,
                "altitude_bin_start_m": altitude_start,
                "altitude_bin_end_m": altitude_start + altitude_bin_width,
                "powered": powered,
                "coast": coast,
                "powered_minus_coast_cda_m2": powered_mean - coast_mean,
            })

    differences = [float(item["powered_minus_coast_cda_m2"]) for item in comparisons]
    absolute = [abs(value) for value in differences]
    return {
        "mach_bin_width": mach_bin_width,
        "altitude_bin_width_m": altitude_bin_width,
        "overlap_bin_count": len(comparisons),
        "overlap_bins": comparisons,
        "aggregate": {
            "mean_powered_minus_coast_cda_m2": mean(differences) if differences else None,
            "mean_abs_powered_minus_coast_cda_m2": mean(absolute) if absolute else None,
            "max_abs_powered_minus_coast_cda_m2": max(absolute) if absolute else None,
        },
        "interpretation": {
            "constant_burn_offset_testable": len(comparisons) >= 3,
            "mach_dependent_burn_offset_testable": len(comparisons) >= 5,
            "height_pressure_effect_testable": len({item["altitude_bin_start_m"] for item in comparisons}) >= 2,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=PROJECT_DIR / "outputs" / "h3_low_g_drag" / "inverse_cda_samples.json")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "outputs" / "h3_low_g_drag" / "powered_coast_comparison.json")
    args = parser.parse_args()

    config = load_model_config(PROJECT_DIR / "configs" / "aim120a_h3_low_g_drag.yaml")
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    rows = payload.get("samples", [])
    comparison = build_comparison(rows)
    result = {
        "schema_version": 3,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_label": config["model_label"],
        "source_kind": payload.get("source_kind", "unknown"),
        "statshark_new_calculation_performed_this_run": False,
        "input_path": str(args.input.resolve()),
        "input_source_is_statshark_reference": False,
        "comparison": comparison,
        "decision_boundary": (
            "Powered/coast overlap is a local H2 pipeline diagnostic only. It cannot establish "
            "a StatShark burn correction without independent reference time series."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_safe(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"overlap_bins={comparison['overlap_bin_count']} "
        f"constant_testable={comparison['interpretation']['constant_burn_offset_testable']}"
    )
    print(f"written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
