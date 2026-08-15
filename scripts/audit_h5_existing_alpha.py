#!/usr/bin/env python3
"""Audit the frozen H4 visible trajectories for H5 alpha excitation."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "statshark_h4"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "h5_body_alpha2"


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _alpha_rad(row: Mapping[str, Any]) -> float:
    if _finite(row.get("alpha_rad")):
        return float(row["alpha_rad"])
    if _finite(row.get("alpha_deg")):
        return math.radians(float(row["alpha_deg"]))
    raise ValueError("H4 row has no alpha")


def _summary(values: Sequence[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "median": median(values) if values else None,
        "max": max(values) if values else None,
    }


def _pearson(x_values: Sequence[float], y_values: Sequence[float]) -> float | None:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return None
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    denominator = math.sqrt(
        sum((x - x_mean) ** 2 for x in x_values)
        * sum((y - y_mean) ** 2 for y in y_values)
    )
    return numerator / denominator if denominator > 0.0 else None


def _load_case(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = dict(payload.get("metadata", {}))
    case_id = str(metadata.get("case_id", path.stem.split("_")[0]))
    rows: list[dict[str, Any]] = []
    for raw in payload.get("result", {}).get("samples", []):
        row = dict(raw)
        row["case_id"] = case_id
        row["trajectory_id"] = case_id
        row["source_kind"] = "statshark_reference_visible_readout"
        row["alpha_rad"] = _alpha_rad(row)
        row["alpha_deg"] = math.degrees(row["alpha_rad"])
        row["mass_kg"] = metadata.get("static_mass_kg")
        row["powered"] = False
        row["thrust_n"] = 0.0
        rows.append(row)
    return metadata, rows


def _time_deltas(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    times = sorted(float(row["time_s"]) for row in rows if _finite(row.get("time_s")))
    return [right - left for left, right in zip(times, times[1:]) if right > left]


def _case_report(metadata: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    primary = [row for row in rows if _finite(row.get("mach")) and 1.45 <= float(row["mach"]) <= 1.55]
    center = [row for row in rows if _finite(row.get("mach")) and 1.47 <= float(row["mach"]) <= 1.53]
    alpha_primary = [abs(math.degrees(_alpha_rad(row))) for row in primary]
    alpha_center = [abs(math.degrees(_alpha_rad(row))) for row in center]
    gamma_primary = [abs(float(row["flight_path_angle_deg"])) for row in primary if _finite(row.get("flight_path_angle_deg"))]
    alpha_bands = {
        "P0": sum(1 for value in alpha_primary if value <= 0.4),
        "P1": sum(1 for value in alpha_primary if 1.0 <= value <= 1.8),
        "P2": sum(1 for value in alpha_primary if 2.3 <= value <= 3.2),
    }
    return {
        "case_id": metadata.get("case_id"),
        "source_kind": "statshark_reference_visible_readout",
        "missile_variant": metadata.get("missile_variant"),
        "raw_sample_count": len(rows),
        "mach_range_all": _summary([float(row["mach"]) for row in rows if _finite(row.get("mach"))]),
        "alpha_deg_range_all": _summary([abs(math.degrees(_alpha_rad(row))) for row in rows]),
        "primary_window": {
            "mach_range": [1.45, 1.55],
            "sample_count": len(primary),
            "alpha_abs_deg": _summary(alpha_primary),
            "flight_path_abs_deg": _summary(gamma_primary),
            "alpha_bands": alpha_bands,
            "center_support_count": len(center),
            "center_alpha_abs_deg": _summary(alpha_center),
            "at_least_five_samples": len(primary) >= 5,
            "has_three_symmetric_side_points": len(center) >= 3,
        },
        "sampling": {
            "dt_s": _summary(_time_deltas(rows)),
            "display_precision": metadata.get("display_precision"),
            "sampling_method": metadata.get("sampling_method"),
        },
        "custom_model_boundary": metadata.get("custom_model_boundary"),
        "user_visible_inputs": metadata.get("user_visible_inputs"),
        "field_snapshot_keys_present": sorted(set(metadata) | set(rows[0].keys()) if rows else set(metadata)),
        "cx_aoa_values_observed": sorted({row.get("cx_aoa") for row in rows if _finite(row.get("cx_aoa"))}),
        "alpha_mach_pearson_primary": _pearson(
            [float(row["mach"]) for row in primary],
            [abs(math.degrees(_alpha_rad(row))) for row in primary],
        ),
    }


def build_review() -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    metadata_by_case: dict[str, dict[str, Any]] = {}
    for path in sorted(RAW_DIR.glob("G[1-7]_*.json")):
        metadata, rows = _load_case(path)
        metadata_by_case[str(metadata.get("case_id"))] = metadata
        cases.append(_case_report(metadata, rows))
        all_rows.extend(rows)

    primary = [row for row in all_rows if _finite(row.get("mach")) and 1.45 <= float(row["mach"]) <= 1.55]
    center = [row for row in all_rows if _finite(row.get("mach")) and 1.47 <= float(row["mach"]) <= 1.53]
    actual_abs_alpha = sorted({round(abs(math.degrees(_alpha_rad(row))), 6) for row in primary})
    band_medians = {}
    for band, lower, upper in (("P0", 0.0, 0.4), ("P1", 1.0, 1.8), ("P2", 2.3, 3.2)):
        values = [abs(math.degrees(_alpha_rad(row))) for row in primary if lower <= abs(math.degrees(_alpha_rad(row))) <= upper]
        band_medians[band] = median(values) if values else None

    variant_values = sorted({str(metadata.get("missile_variant")) for metadata in metadata_by_case.values()})
    boundary_values = sorted({json.dumps(metadata.get("custom_model_boundary"), sort_keys=True) for metadata in metadata_by_case.values()})
    cx_values = sorted({row.get("cx_aoa") for row in primary if _finite(row.get("cx_aoa"))})
    separation = {
        "P1_minus_P0_deg": (
            band_medians["P1"] - band_medians["P0"]
            if band_medians["P1"] is not None and band_medians["P0"] is not None else None
        ),
        "P2_minus_P1_deg": (
            band_medians["P2"] - band_medians["P1"]
            if band_medians["P2"] is not None and band_medians["P1"] is not None else None
        ),
        "P2_median_deg": band_medians["P2"],
    }
    alpha_excitation_pass = (
        separation["P1_minus_P0_deg"] is not None
        and separation["P2_minus_P1_deg"] is not None
        and separation["P1_minus_P0_deg"] >= 0.8
        and separation["P2_minus_P1_deg"] >= 0.8
        and separation["P2_median_deg"] >= 2.3
    )
    center_cases = sorted({str(row["case_id"]) for row in center})
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_label": "local_candidate_H5_body_alpha2_M1p5",
        "new_statshark_calculation_performed_this_run": False,
        "source_kind": "statshark_reference_visible_readout",
        "input_directory": str(RAW_DIR.resolve()),
        "case_reports": cases,
        "aggregate_primary_window": {
            "mach_window": [1.45, 1.55],
            "center_window": [1.47, 1.53],
            "raw_sample_count": len(all_rows),
            "primary_sample_count": len(primary),
            "center_sample_count": len(center),
            "center_support_cases": center_cases,
            "center_support_case_count": len(center_cases),
            "actual_abs_alpha_values_deg": actual_abs_alpha,
            "actual_alpha_band_medians_deg": band_medians,
            "alpha_separation": separation,
            "alpha_excitation_gate": "pass_planning_only" if alpha_excitation_pass else "insufficient_alpha_excitation",
            "alpha_levels_are_independent_interventions": False,
        },
        "configuration_audit": {
            "unique_missile_variants": variant_values,
            "unique_custom_boundaries": [json.loads(value) for value in boundary_values],
            "cx_aoa_values_observed": cx_values,
            "required_cx_aoa_levels": [0, 9, 18],
            "all_required_cx_levels_present": cx_values == [0, 9, 18],
            "full_model_snapshot_present": False,
            "other_parameter_differences_excluded": False,
            "status": "planning_data_only_no_cx_aoa_ablation",
        },
        "identifiability_audit": {
            "existing_data_has_actual_alpha_span_near_m1p5": bool(actual_abs_alpha),
            "existing_data_has_low_mid_high_labels": all(value is not None for value in band_medians.values()),
            "existing_data_can_identify_cx_aoa_slope": False,
            "existing_data_can_separate_cda0_and_alpha2": False,
            "reason": "H4 has one zero-active-fin configuration and alpha is coupled to time/Mach history; C0/C9/C18 interventions are absent.",
            "observed_alpha_mach_coupling_is_diagnostic_only": True,
        },
        "formal_h5_gates": {
            "G_H5_0_configuration_isolated": False,
            "G_H5_1_m1p5_coverage": all(case["primary_window"]["at_least_five_samples"] for case in cases),
            "G_H5_2_alpha_excitation": False,
            "G_H5_3_alpha2_identified": False,
            "G_H5_4_cx_aoa_scaling": False,
            "G_H5_5_model_form": False,
            "G_H5_6_trajectory_replay": False,
            "G_H5_7_h4_join": False,
        },
        "required_next_evidence": {
            "formal_case_matrix": [
                "H5-P0-C0", "H5-P0-C9", "H5-P0-C18",
                "H5-P1-C0", "H5-P1-C9", "H5-P1-C18",
                "H5-P2-C0", "H5-P2-C9", "H5-P2-C18",
            ],
            "new_statshark_calculation_authorization_required": True,
            "minimum_reconnaissance_batch": ["H5-P0-C9", "H5-P2-C0", "H5-P2-C9", "H5-P2-C18"],
        },
        "status": "phase1_existing_alpha_review_complete_formal_h5_blocked_without_cx_ablation",
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "phase1_existing_alpha_review.json"
    output_path.write_text(json.dumps(build_review(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "status": "written"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
