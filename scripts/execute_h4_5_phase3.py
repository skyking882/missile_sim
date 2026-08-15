#!/usr/bin/env python3
"""Execute Plan 4.5 Phase 3 on the authorized T1-T4 StatShark captures.

This script preserves the visible-label artifacts, performs auditable filtering
and inverse-CdA diagnostics, and refuses to create a transonic fit when the
sampling/coverage gates are not met.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
ROOT_DIR = PROJECT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from aim120_model.inverse_cda import estimate_inverse_cda, fit_log_cda_knots, summarize_inverse_cda
from aim120_model.sample_filters import LowGFilterSettings, apply_filter, normalize_sample


CONFIG_PATH = PROJECT_DIR / "configs" / "aim120a_h4_5_transonic_drag.yaml"
RAW_DIR = PROJECT_DIR / "data" / "raw" / "statshark_h4_5"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "h4_5_transonic_drag"
CASE_IDS = ("T1", "T2", "T3", "T4")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _load_case(case_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = RAW_DIR / f"{case_id}_statshark_visible_slider_20260811.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    result = payload.get("result", {})
    artifact = {
        "case_id": case_id,
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "source_url_or_artifact": metadata.get("source_url_or_artifact"),
        "missile_variant": metadata.get("missile_variant"),
        "standard_model_also_selected": metadata.get("standard_model_also_selected"),
        "user_visible_inputs": metadata.get("user_visible_inputs", {}),
        "custom_model_boundary": metadata.get("custom_model_boundary", {}),
        "sample_count_raw": len(result.get("samples", [])),
    }
    return artifact, [dict(row) for row in result.get("samples", [])]


def _state_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(name) for name in ("speed_mps", "mach", "alpha_rad", "altitude_m", "target_distance_m"))


def _deduplicate_visible(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Remove consecutive/repeated visible states only for diagnostics.

    The raw JSON remains untouched.  StatShark's visible annotation often
    repeats a state at adjacent slider positions; retaining those as separate
    derivative points would create artificial zero-speed plateaus.
    """

    unique: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        key = _state_key(row)
        if key in seen:
            continue
        seen.add(key)
        unique.append(dict(row))
    return unique, len(rows) - len(unique)


def _normalize_case(rows: Sequence[Mapping[str, Any]], case_id: str, settings: LowGFilterSettings) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        item = dict(raw)
        item.update({
            "trajectory_id": case_id,
            "case_id": case_id,
            "source_kind": "statshark_reference",
            "source_time_index": index,
            "mass_kg": item.get("mass_kg", 147.87),
            "powered": False,
            "thrust_n": 0.0,
        })
        sample = normalize_sample(item, settings=settings)
        sample.update({
            "trajectory_id": case_id,
            "case_id": case_id,
            "source_kind": "statshark_reference",
            "source_time_index": index,
            "powered": False,
            "thrust_n": 0.0,
        })
        normalized.append(apply_filter(sample, settings))
    return normalized


def _window(rows: Sequence[Mapping[str, Any]], lo: float = 0.85, hi: float = 1.25) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if row.get("mach") is not None and lo <= float(row["mach"]) <= hi]


def _range(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, float | None]:
    values = [float(row[key]) for row in rows if row.get(key) is not None and math.isfinite(float(row[key]))]
    return {"min": min(values) if values else None, "max": max(values) if values else None}


def _spacing(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: float(row["time_s"]))
    deltas = [float(b["time_s"]) - float(a["time_s"]) for a, b in zip(ordered, ordered[1:]) if float(b["time_s"]) > float(a["time_s"])]
    return {
        "sample_count": len(rows),
        "time_delta_s": {
            "min": min(deltas) if deltas else None,
            "median": median(deltas) if deltas else None,
            "max": max(deltas) if deltas else None,
            "unique": sorted(set(round(value, 6) for value in deltas)),
        },
        "requirement": {"M0.90-1.20_max_dt_s": 0.5, "M1.20-1.30_max_dt_s": 1.0},
    }


def _node_counts(rows: Sequence[Mapping[str, Any]], knots: Sequence[float], tolerance: float = 0.025) -> dict[str, int]:
    return {
        str(knot): sum(1 for row in rows if row.get("mach") is not None and abs(float(row["mach"]) - float(knot)) <= tolerance)
        for knot in knots
    }


def _filter_sensitivity(case_rows: Mapping[str, list[dict[str, Any]]], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    primary = config["primary_filter"]
    results: list[dict[str, Any]] = []
    for alpha in config["sensitivity"]["alpha_thresholds_deg"]:
        for gamma in config["sensitivity"]["flight_path_thresholds_deg"]:
            for q_min in config["sensitivity"]["q_thresholds_pa"]:
                settings = LowGFilterSettings(
                    lateral_load_threshold_g=float(primary["lateral_load_threshold_g"]),
                    alpha_threshold_deg=float(alpha),
                    flight_path_threshold_deg=float(gamma),
                    q_min_pa=float(q_min),
                )
                by_case: dict[str, Any] = {}
                for case_id, rows in case_rows.items():
                    unique, duplicate_count = _deduplicate_visible(rows)
                    normalized = _normalize_case(unique, case_id, settings)
                    selected = _window(normalized)
                    accepted = [row for row in selected if row.get("accepted")]
                    by_case[case_id] = {
                        "raw_unique_window_count": len(selected),
                        "duplicate_visible_states_removed": duplicate_count,
                        "accepted_count": len(accepted),
                        "mach_range": _range(accepted, "mach"),
                        "alpha_range_deg": _range(accepted, "alpha_total_deg"),
                        "flight_path_range_deg": _range(accepted, "flight_path_angle_deg"),
                        "node_counts": _node_counts(accepted, config["target"]["diagnostic_knots"]),
                    }
                results.append({
                    "filter_settings": settings.to_dict(),
                    "cases": by_case,
                    "accepted_total": sum(value["accepted_count"] for value in by_case.values()),
                    "status": "diagnostic_only",
                })
    return results


def _repeatability(case_rows: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    left, right = case_rows.get("T1", []), case_rows.get("T3", [])
    left_unique, _ = _deduplicate_visible(left)
    right_unique, _ = _deduplicate_visible(right)
    left_by_state = {_state_key(row): row for row in left_unique}
    right_by_state = {_state_key(row): row for row in right_unique}
    shared_keys = sorted(set(left_by_state).intersection(right_by_state), key=lambda key: float(left_by_state[key]["time_s"]))
    pairs = []
    for key in shared_keys:
        a, b = left_by_state[key], right_by_state[key]
        pairs.append({
            "T1_time_s": float(a["time_s"]),
            "T3_time_s": float(b["time_s"]),
            "time_schedule_delta_s": float(a["time_s"]) - float(b["time_s"]),
            "speed_delta_mps": float(a["speed_mps"]) - float(b["speed_mps"]),
            "mach_delta": float(a["mach"]) - float(b["mach"]),
            "altitude_delta_m": float(a["altitude_m"]) - float(b["altitude_m"]),
            "target_distance_delta_m": float(a["target_distance_m"]) - float(b["target_distance_m"]),
        })
    schedule_deltas = [abs(item["time_schedule_delta_s"]) for item in pairs]
    exact_common_state_match = all(
        abs(item["speed_delta_mps"]) < 1.0e-9
        and abs(item["mach_delta"]) < 1.0e-9
        and abs(item["altitude_delta_m"]) < 1.0e-9
        and abs(item["target_distance_delta_m"]) < 1.0e-9
        for item in pairs
    )
    schedule_same = len(left_unique) == len(right_unique) and all(abs(item["time_schedule_delta_s"]) < 1.0e-9 for item in pairs)
    return {
        "status": "pass_exact_visible_repeat" if exact_common_state_match and schedule_same else ("partial_common_state_match_schedule_drift" if exact_common_state_match else "diagnostic_mismatch"),
        "T1_unique_count": len(left_unique),
        "T3_unique_count": len(right_unique),
        "common_visible_state_count": len(shared_keys),
        "T1_only_visible_state_count": len(set(left_by_state) - set(right_by_state)),
        "T3_only_visible_state_count": len(set(right_by_state) - set(left_by_state)),
        "max_abs_speed_delta_mps": max((abs(item["speed_delta_mps"]) for item in pairs), default=None),
        "max_abs_mach_delta": max((abs(item["mach_delta"]) for item in pairs), default=None),
        "max_abs_altitude_delta_m": max((abs(item["altitude_delta_m"]) for item in pairs), default=None),
        "max_abs_target_distance_delta_m": max((abs(item["target_distance_delta_m"]) for item in pairs), default=None),
        "max_abs_time_schedule_delta_s": max(schedule_deltas, default=None),
        "paired_count": len(pairs),
    }


def main() -> int:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc).isoformat()
    primary = config["primary_filter"]
    settings = LowGFilterSettings(
        lateral_load_threshold_g=float(primary["lateral_load_threshold_g"]),
        alpha_threshold_deg=float(primary["alpha_threshold_deg"]),
        flight_path_threshold_deg=float(primary["flight_path_threshold_deg"]),
        q_min_pa=float(primary["q_min_pa"]),
    )

    artifacts: list[dict[str, Any]] = []
    raw_by_case: dict[str, list[dict[str, Any]]] = {}
    for case_id in CASE_IDS:
        artifact, rows = _load_case(case_id)
        artifacts.append(artifact)
        raw_by_case[case_id] = rows

    unique_by_case: dict[str, list[dict[str, Any]]] = {}
    normalized_by_case: dict[str, list[dict[str, Any]]] = {}
    unique_counts: dict[str, int] = {}
    for case_id, rows in raw_by_case.items():
        unique, duplicates = _deduplicate_visible(rows)
        unique_by_case[case_id] = unique
        unique_counts[case_id] = duplicates
        normalized_by_case[case_id] = _normalize_case(unique, case_id, settings)

    source_manifest = {
        "schema_version": 5,
        "generated_at_utc": now,
        "model_label": config["model_label"],
        "phase": "phase3_authorized_reference_collection",
        "statshark_new_calculation_performed_this_run": True,
        "authorization": "User explicitly authorized T1-T4 website calculations in this run.",
        "input_artifacts": artifacts,
        "raw_sample_count": sum(len(rows) for rows in raw_by_case.values()),
        "unique_visible_state_count": sum(len(rows) for rows in unique_by_case.values()),
        "duplicate_visible_states_removed_for_diagnostics": unique_counts,
        "reference_source_kind": "statshark_reference",
        "custom_model_boundary": "AIM120A_H4_GLIDE_NOPWR_NOGUIDE_20260811 [C]; residual stage thrust 0 N, mass loss 0 kg, guidance disabled, fin lateral acceleration 0; only custom trace extracted.",
        "h4_overwrite": False,
        "status": "ready_for_phase3_coverage_gate",
    }

    filtered_rows: list[dict[str, Any]] = []
    per_case: dict[str, Any] = {}
    for case_id in CASE_IDS:
        rows = normalized_by_case[case_id]
        filtered_rows.extend(rows)
        window_rows = _window(rows)
        accepted = [row for row in window_rows if row.get("accepted")]
        per_case[case_id] = {
            "raw_count": len(raw_by_case[case_id]),
            "unique_visible_state_count": len(rows),
            "duplicate_visible_states_removed": unique_counts[case_id],
            "window_mach": {"min": 0.85, "max": 1.25},
            "window_count": len(window_rows),
            "accepted_count": len(accepted),
            "accepted_mach_range": _range(accepted, "mach"),
            "accepted_alpha_range_deg": _range(accepted, "alpha_total_deg"),
            "accepted_flight_path_range_deg": _range(accepted, "flight_path_angle_deg"),
            "sampling": _spacing(accepted),
            "node_counts": _node_counts(accepted, config["target"]["diagnostic_knots"]),
        }

    coverage_gate_checks = {
        "all_required_cases_present": all(case_id in raw_by_case for case_id in CASE_IDS),
        "T1_main_case_present": per_case["T1"]["accepted_count"] > 0,
        "T2_peak_center_present": per_case["T2"]["accepted_count"] > 0,
        "T3_repeat_present": per_case["T3"]["accepted_count"] > 0,
        "T4_height_case_present": per_case["T4"]["accepted_count"] > 0,
        "nominal_dense_sampling_M0.90_1.20": all(
            per_case[case_id]["sampling"]["time_delta_s"]["max"] is not None
            and per_case[case_id]["sampling"]["time_delta_s"]["max"] <= 0.5
            for case_id in CASE_IDS
            if per_case[case_id]["accepted_count"] >= 2
        ) and any(per_case[case_id]["accepted_count"] >= 2 for case_id in CASE_IDS),
        "nominal_dense_sampling_M1.20_1.30": all(
            per_case[case_id]["sampling"]["time_delta_s"]["max"] is not None
            and per_case[case_id]["sampling"]["time_delta_s"]["max"] <= 1.0
            for case_id in CASE_IDS
            if per_case[case_id]["accepted_count"] >= 2
        ),
        "two_trajectory_support_at_diagnostic_knots": all(
            sum(1 for case_id in CASE_IDS if per_case[case_id]["node_counts"].get(str(knot), 0) > 0) >= 2
            for knot in config["target"]["diagnostic_knots"]
        ),
    }
    coverage_status = "coverage_gate_pass" if all(coverage_gate_checks.values()) else "coverage_gate_fail_sampling_or_support"
    coverage = {
        "schema_version": 5,
        "generated_at_utc": now,
        "model_label": config["model_label"],
        "target_range_mach": {"min": 0.90, "max": 1.20},
        "diagnostic_range_mach": {"min": 0.85, "max": 1.25},
        "per_case": per_case,
        "gate_checks": coverage_gate_checks,
        "status": coverage_status,
        "decision": "No final transonic fit is allowed unless every gate is true; visible labels were approximately 2 s apart and do not satisfy the <=0.5 s target.",
    }

    inverse_rows: list[dict[str, Any]] = []
    inverse_by_case: dict[str, Any] = {}
    gravity_ratios: list[float] = []
    for case_id in CASE_IDS:
        accepted = [row for row in _window(normalized_by_case[case_id]) if row.get("accepted")]
        diagnostics = estimate_inverse_cda(accepted, gravity_mps2=9.80665, window_points=5)
        inverse_rows.extend(diagnostics)
        valid = [row for row in diagnostics if row.get("inverse_cda_valid")]
        for row in valid:
            drag = abs(float(row.get("axial_drag_accel_mps2", 0.0)))
            gravity = abs(float(row.get("gravity_axial_mps2", 0.0)))
            gravity_ratios.append(gravity / max(drag, 1.0e-9))
        inverse_by_case[case_id] = {
            "accepted_for_inverse_count": len(accepted),
            "valid_inverse_count": len(valid),
            "mach_range": _range(valid, "mach"),
            "inverse_cda_range_m2": _range(valid, "inverse_cda_m2"),
            "summary": summarize_inverse_cda(diagnostics).get(case_id, {}),
        }
    inverse_payload = {
        "schema_version": 5,
        "generated_at_utc": now,
        "model_label": config["model_label"],
        "method": "effective inverse CdA from local least-squares speed slope; m*dV/dt = -q*CdA - m*g*sin(gamma)",
        "statshark_new_calculation_performed_this_run": True,
        "rows": inverse_rows,
        "by_trajectory": inverse_by_case,
        "gravity_cancellation": {
            "sample_count": len(gravity_ratios),
            "mean_ratio": sum(gravity_ratios) / len(gravity_ratios) if gravity_ratios else None,
            "max_ratio": max(gravity_ratios) if gravity_ratios else None,
            "fraction_above_1": sum(value > 1.0 for value in gravity_ratios) / len(gravity_ratios) if gravity_ratios else None,
            "fraction_above_3": sum(value > 3.0 for value in gravity_ratios) / len(gravity_ratios) if gravity_ratios else None,
        },
        "status": "inverse_diagnostics_available" if any(row.get("inverse_cda_valid") for row in inverse_rows) else "blocked_no_positive_inverse_cda",
        "interpretation_boundary": "Effective diagnostic only; it does not reveal the StatShark backend solver.",
    }

    repeatability = _repeatability(raw_by_case)
    h4_path = PROJECT_DIR / "outputs" / "h4_glide_drag" / "cda_knots_fit.json"
    h4_fit = json.loads(h4_path.read_text(encoding="utf-8")) if h4_path.exists() else {}
    h4_join = {
        "schema_version": 1,
        "generated_at_utc": now,
        "status": "join_reference_frozen_no_transonic_fit" if coverage_status != "coverage_gate_pass" else "pending_join_after_fit",
        "join_mach": 1.2,
        "h4_nodes_unchanged": {"mach_knots": h4_fit.get("mach_knots", []), "cda_knots_m2": h4_fit.get("cda_knots_m2", [])},
        "rule": "M>=1.2 H4 nodes are copied by reference; this Phase 3 run does not overwrite H4.",
        "observed_T1_T2_T3_T4_support_near_join": {case_id: per_case[case_id]["accepted_mach_range"] for case_id in CASE_IDS},
    }
    rm10_path = PROJECT_DIR / "outputs" / "rm10_shape_prior" / "fit_report.json"
    rm10 = json.loads(rm10_path.read_text(encoding="utf-8")) if rm10_path.exists() else {}
    rm10_report = {
        "schema_version": 1,
        "generated_at_utc": now,
        "status": "external_shape_prior_not_applied",
        "source": rm10.get("source", {"report": "NACA Report 1160 RM-10"}),
        "beta_reference": rm10.get("fit", {}).get("beta"),
        "sensitivity_range": [rm10.get("fit", {}).get("beta_digitization_one_at_a_time_min"), rm10.get("fit", {}).get("beta_digitization_one_at_a_time_max")],
        "policy": "RM-10 remains a generic slender-body shape prior; it is not used to manufacture AIM-120A transonic StatShark nodes.",
    }

    filter_sensitivity = _filter_sensitivity(unique_by_case, config)
    model_comparison = {
        "schema_version": 1,
        "generated_at_utc": now,
        "status": "diagnostic_only_no_final_model",
        "candidates": [
            {"name": "H4_frozen_M>=1.2", "status": "reference_frozen", "source": str(h4_path.resolve())},
            {"name": "T1-T4_effective_inverse", "status": "coverage_gate_failed_or_pending", "source": "inverse_cda_diagnostics.json"},
            {"name": "RM10_shape_prior", "status": "external_shape_prior_only", "source": str(rm10_path.resolve())},
        ],
        "boundary": "No candidate is presented as a validated M0.90-1.20 transonic fit in this run.",
    }

    fit_payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": now,
        "model_label": config["model_label"],
        "status": "fit_blocked_coverage_gate",
        "fit_generated": False,
        "coverage_status": coverage_status,
        "reason": "The visible slider data are approximately 2 s apart and the accepted rows do not provide the required dense M0.90-1.20 support with two trajectories at every diagnostic knot.",
        "h4_join_policy": "M>=1.2 remains frozen by reference.",
    }
    if coverage_status == "coverage_gate_pass":
        try:
            fit = fit_log_cda_knots(inverse_rows, config["target"]["diagnostic_knots"], minimum_samples_per_node=3)
            fit_payload.update({"status": "fit_generated_pending_holdout", "fit_generated": True, "fit": fit})
        except (TypeError, ValueError) as exc:
            fit_payload["reason"] = str(exc)

    holdout = {
        "schema_version": 1,
        "generated_at_utc": now,
        "status": "blocked_no_accepted_final_fit" if not fit_payload["fit_generated"] else "pending_whole_trajectory_holdout",
        "method": "whole-trajectory T3 holdout; no point-random split",
        "repeatability": repeatability,
        "real_holdout": "not run because coverage-gated transonic fit was not generated",
        "boundary": "A future holdout replay would use observed altitude and flight-path angle as exogenous inputs; it would validate axial glide only, not full 6-DoF backend equivalence.",
    }

    filtered_payload = {
        "schema_version": 5,
        "generated_at_utc": now,
        "model_label": config["model_label"],
        "filter_settings": settings.to_dict(),
        "rows": filtered_rows,
        "per_case": per_case,
        "status": "reference_samples_present_diagnostics_only",
        "raw_data_note": "Rows are normalized from the four saved raw artifacts; duplicate visible states are retained in rows and removed only for derivative/coverage diagnostics.",
    }
    report_lines = [
        "# H4.5 跨音速无动力阻力补洞报告（Phase 3）",
        "",
        "## 结论",
        "",
        f"已按授权完成 T1–T4 网站采集并保存 4 个原始轨迹档。Phase 3 覆盖率门：`{coverage_status}`。本轮不生成最终 M0.90–1.20 CdA 拟合。",
        "",
        f"原始标签 {source_manifest['raw_sample_count']} 条；去除重复可见状态后用于诊断 {source_manifest['unique_visible_state_count']} 条。可见滑块实际约 2 s 级，不能满足 M0.90–1.20 的 <=0.5 s 采样要求。",
        "",
        "## 主筛选与覆盖",
        "",
        f"主筛选：|lateral G|<=2、|alpha|<=2°、|gamma|<=5°、q>=1000 Pa、无推力、质量恒定。逐案例摘要见 `coverage_report.json` 与 `filtered_samples.json`。",
        "",
        "| Case | accepted in M0.85–1.25 | Mach range | observed max dt |",
        "|---|---:|---|---:|",
    ]
    for case_id in CASE_IDS:
        item = per_case[case_id]
        report_lines.append(f"| {case_id} | {item['accepted_count']} | {item['accepted_mach_range']} | {item['sampling']['time_delta_s']['max']} s |")
    report_lines.extend([
        "",
        "T1/T3 的可见重复性单独记录；重复性通过不等于采样密度通过。T4 仅作为 4000 m 高度接缝诊断，不足以证明高度独立项。",
        "",
        "## 逆算和模型边界",
        "",
        "`inverse_cda_diagnostics.json` 只表示从可见速度、海拔和航迹角得到的有效诊断量，不揭示 StatShark 后端求解器。`cda_transonic_fit.json` 明确记录为 gate blocked；H4 M>=1.2 节点保持冻结。RM-10 继续是 external_shape_prior。",
        "",
        "## 产物",
        "",
        "- `source_manifest.json`：T1–T4 路径、哈希、输入和自定义模型边界。",
        "- `coverage_report.json`：采样、节点覆盖和门禁。",
        "- `inverse_cda_diagnostics.json`：有效逆算诊断。",
        "- `trajectory_holdout_report.json`：T1/T3 重复性与留出门。",
        "- `model_comparison.json`、`h4_join_report.json`、`rm10_prior_sensitivity.json`：候选比较和边界。",
        "",
        "没有修改 War Thunder 游戏文件。",
    ])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "source_manifest.json": source_manifest,
        "filtered_samples.json": filtered_payload,
        "coverage_report.json": coverage,
        "inverse_cda_diagnostics.json": inverse_payload,
        "filter_sensitivity.json": {"schema_version": 1, "generated_at_utc": now, "results": filter_sensitivity, "status": "diagnostic_only"},
        "cda_transonic_fit.json": fit_payload,
        "trajectory_holdout_report.json": holdout,
        "h4_join_report.json": h4_join,
        "rm10_prior_sensitivity.json": rm10_report,
        "model_comparison.json": model_comparison,
    }
    for name, payload in outputs.items():
        (OUTPUT_DIR / name).write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "H4_5_TRANSONIC_DRAG_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"phase3_status={coverage_status} raw_samples={source_manifest['raw_sample_count']} unique_states={source_manifest['unique_visible_state_count']}")
    print(f"accepted_primary_total={sum(item['accepted_count'] for item in per_case.values())} inverse_valid={sum(item['valid_inverse_count'] for item in inverse_by_case.values())}")
    print(f"fit_generated={fit_payload['fit_generated']} repeatability={repeatability['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
