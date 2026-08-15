#!/usr/bin/env python3
"""Execute Plan 4.5 Phase 0–1 without running new StatShark calculations."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
ROOT_DIR = PROJECT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from aim120_model.atmosphere import StandardAtmosphere
from aim120_model.inverse_cda import estimate_inverse_cda
from aim120_model.sample_filters import LowGFilterSettings, apply_filter, normalize_sample


CONFIG_PATH = PROJECT_DIR / "configs" / "aim120a_h4_5_transonic_drag.yaml"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "h4_5_transonic_drag"
RAW_DIR = PROJECT_DIR / "data" / "raw" / "statshark_h4"


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
    return {
        "case_id": case_id,
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "metadata": metadata,
        "result": result,
    }, [dict(row) for row in result.get("samples", [])]


def _range(values: Sequence[float]) -> dict[str, float | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return {"min": min(finite) if finite else None, "max": max(finite) if finite else None}


def _sample_spacing(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    times = sorted(float(row["time_s"]) for row in rows if row.get("time_s") is not None)
    deltas = [right - left for left, right in zip(times, times[1:]) if right > left]
    speeds = [float(row["speed_mps"]) for row in rows if row.get("speed_mps") is not None]
    speed_steps = [abs(right - left) for left, right in zip(speeds, speeds[1:]) if right != left]
    return {
        "sample_count": len(rows),
        "time_delta_s": {
            "min": min(deltas) if deltas else None,
            "median": median(deltas) if deltas else None,
            "max": max(deltas) if deltas else None,
            "unique": sorted(set(round(value, 6) for value in deltas)),
        },
        "speed_step_mps": {
            "min_nonzero": min(speed_steps) if speed_steps else None,
            "median_nonzero": median(speed_steps) if speed_steps else None,
            "max_nonzero": max(speed_steps) if speed_steps else None,
        },
        "display_precision_from_source": "speed 1 km/h; Mach 0.01; alpha 0.1 deg; altitude and target distance 1 m",
    }


def _node_counts(rows: Sequence[Mapping[str, Any]], knots: Sequence[float], tolerance: float = 0.025) -> dict[str, int]:
    return {
        str(knot): sum(1 for row in rows if row.get("mach") is not None and abs(float(row["mach"]) - float(knot)) <= tolerance)
        for knot in knots
    }


def _connected_runs(rows: Sequence[Mapping[str, Any]], max_gap_s: float = 2.5) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: float(row["time_s"]))
    if not ordered:
        return []
    runs: list[list[Mapping[str, Any]]] = [[ordered[0]]]
    for row in ordered[1:]:
        if float(row["time_s"]) - float(runs[-1][-1]["time_s"]) <= max_gap_s:
            runs[-1].append(row)
        else:
            runs.append([row])
    return [
        {
            "sample_count": len(run),
            "time_min_s": float(run[0]["time_s"]),
            "time_max_s": float(run[-1]["time_s"]),
            "mach_min": min(float(row["mach"]) for row in run),
            "mach_max": max(float(row["mach"]) for row in run),
        }
        for run in runs
    ]


def _filter_case(rows: Sequence[Mapping[str, Any]], settings: LowGFilterSettings) -> tuple[list[dict[str, Any]], dict[str, int]]:
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        item = dict(raw)
        item["trajectory_id"] = str(raw.get("case_id", "unknown"))
        item["case_id"] = str(raw.get("case_id", "unknown"))
        item["source_kind"] = "statshark_reference"
        item["source_time_index"] = index
        if "mass_kg" not in item:
            item["mass_kg"] = 147.87
        sample = normalize_sample(item, settings=settings)
        sample["trajectory_id"] = item["trajectory_id"]
        sample["case_id"] = item["case_id"]
        sample["source_kind"] = "statshark_reference"
        sample["source_time_index"] = index
        filtered = apply_filter(sample, settings)
        filtered["powered"] = False
        filtered["thrust_n"] = 0.0
        normalized.append(filtered)
    counts: dict[str, int] = defaultdict(int)
    for row in normalized:
        for reason in row.get("rejection_reasons", []):
            counts[str(reason)] += 1
    return normalized, dict(counts)


def _filter_summary(case_rows: Mapping[str, list[dict[str, Any]]], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    target_knots = config["target"]["diagnostic_knots"]
    sensitivity = config["sensitivity"]
    results: list[dict[str, Any]] = []
    for alpha in sensitivity["alpha_thresholds_deg"]:
        for gamma in sensitivity["flight_path_thresholds_deg"]:
            for q_min in sensitivity["q_thresholds_pa"]:
                settings = LowGFilterSettings(
                    lateral_load_threshold_g=float(config["primary_filter"]["lateral_load_threshold_g"]),
                    alpha_threshold_deg=float(alpha),
                    flight_path_threshold_deg=float(gamma),
                    q_min_pa=float(q_min),
                )
                case_summaries: dict[str, Any] = {}
                for case_id, rows in case_rows.items():
                    normalized, reject_counts = _filter_case(rows, settings)
                    accepted = [row for row in normalized if row.get("accepted")]
                    case_summaries[case_id] = {
                        "accepted_count": len(accepted),
                        "mach_range": _range([float(row["mach"]) for row in accepted]),
                        "alpha_range_deg": _range([float(row["alpha_total_deg"]) for row in accepted]),
                        "flight_path_range_deg": _range([float(row["flight_path_angle_deg"]) for row in accepted]),
                        "node_counts": _node_counts(accepted, target_knots),
                        "connected_runs": _connected_runs(accepted),
                        "rejection_reason_counts": reject_counts,
                    }
                results.append({
                    "filter_settings": settings.to_dict(),
                    "cases": case_summaries,
                    "accepted_total": sum(item["accepted_count"] for item in case_summaries.values()),
                    "status": "diagnostic_only",
                })
    return results


def main() -> int:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    case_payloads: dict[str, dict[str, Any]] = {}
    case_rows: dict[str, list[dict[str, Any]]] = {}
    for case_id in config["existing_cases"]:
        payload, rows = _load_case(case_id)
        case_payloads[case_id] = payload
        case_rows[case_id] = rows
    target_min = float(config["target"]["mach_min"])
    target_max = float(config["target"]["mach_max"])
    window_rows = {
        case_id: [
            dict(row)
            for row in rows
            if row.get("mach") is not None and target_min - 0.05 <= float(row["mach"]) <= target_max + 0.05
        ]
        for case_id, rows in case_rows.items()
    }

    h4_fit_path = PROJECT_DIR / "outputs" / "h4_glide_drag" / "cda_knots_fit.json"
    frozen_files = [
        ROOT_DIR / "plan4.md",
        PROJECT_DIR / ".md" / "H4_GLIDE_DRAG_ENVELOPE_REPORT.md",
        h4_fit_path,
        RAW_DIR / "G3_statshark_visible_slider_20260811.json",
        RAW_DIR / "G6_statshark_visible_slider_20260811.json",
        PROJECT_DIR / "data" / "reference_external" / "rm10" / "rm10_figure13_composite_digitization.csv",
        PROJECT_DIR / "data" / "reference_external" / "rm10" / "NACA_TR_1160_RM10.pdf",
        PROJECT_DIR / "outputs" / "rm10_shape_prior" / "fit_report.json",
        PROJECT_DIR / "outputs" / "rm10_shape_prior" / "rm10_h4_hybrid_nodes.csv",
    ]
    freeze_records = []
    for path in frozen_files:
        freeze_records.append({
            "path": str(path.resolve()),
            "exists": path.exists(),
            "sha256": _sha256(path) if path.exists() else None,
            "size_bytes": path.stat().st_size if path.exists() else None,
        })
    h4_fit = json.loads(h4_fit_path.read_text(encoding="utf-8")) if h4_fit_path.exists() else {}
    phase0 = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_label": config["model_label"],
        "new_statshark_calculation_performed_this_run": False,
        "h4_freeze_rule": config["frozen_h4"],
        "frozen_files": freeze_records,
        "h4_nodes_snapshot": {
            "mach_knots": h4_fit.get("mach_knots", []),
            "cda_knots_m2": h4_fit.get("cda_knots_m2", []),
            "source_status": h4_fit.get("status"),
        },
        "independent_output_dir": str(OUTPUT_DIR.resolve()),
        "status": "phase0_frozen_no_h4_overwrite",
    }

    extracted: dict[str, Any] = {}
    spacing: dict[str, Any] = {}
    for case_id, selected in window_rows.items():
        extracted[case_id] = {
            "source_file": case_payloads[case_id]["path"],
            "source_sha256": case_payloads[case_id]["sha256"],
            "source_kind": "statshark_reference",
            "selection_rule": {"mach_min": target_min - 0.05, "mach_max": target_max + 0.05},
            "samples": selected,
        }
        spacing[case_id] = _sample_spacing(selected)

    sensitivity = _filter_summary(window_rows, config)
    primary = next(
        item for item in sensitivity
        if item["filter_settings"]["alpha_threshold_deg"] == 2.0
        and item["filter_settings"]["flight_path_threshold_deg"] == 5.0
        and item["filter_settings"]["q_min_pa"] == 1000.0
    )
    inverse_diagnostics: dict[str, Any] = {}
    for case_id, rows in window_rows.items():
        normalized, _reject_counts = _filter_case(rows, LowGFilterSettings(
            lateral_load_threshold_g=2.0,
            alpha_threshold_deg=2.0,
            flight_path_threshold_deg=5.0,
            q_min_pa=1000.0,
        ))
        accepted = [row for row in normalized if row.get("accepted")]
        inverse_rows = estimate_inverse_cda(accepted, gravity_mps2=9.80665, window_points=5)
        valid = [row for row in inverse_rows if row.get("inverse_cda_valid")]
        negative = [row for row in inverse_rows if row.get("inverse_cda_reason") == "non_positive_inverse_cda"]
        inverse_diagnostics[case_id] = {
            "accepted_for_inverse_count": len(accepted),
            "valid_inverse_count": len(valid),
            "negative_inverse_count": len(negative),
            "mach_range": _range([float(row["mach"]) for row in valid]),
            "inverse_cda_range_m2": _range([float(row["inverse_cda_m2"]) for row in valid]),
            "status": "diagnostic_only_no_fit",
        }

    phase1 = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_label": config["model_label"],
        "source_kind": "statshark_reference",
        "new_statshark_calculation_performed_this_run": False,
        "target_window_mach": {"min": target_min, "max": target_max},
        "extracted_samples": extracted,
        "sampling_resolution": spacing,
        "primary_filter_result": primary,
        "filter_sensitivity": sensitivity,
        "inverse_diagnostics": inverse_diagnostics,
        "pre_fit_decision": {
            "low_dimensional_prefit_allowed": False,
            "reason": "Existing G3/G6 labels are nominally about 2 s apart and the strict small-alpha window does not provide dense, repeated M0.90-1.20 support; use only coverage/diagnostics until T1-T4 are authorized and acquired.",
            "required_next_data": [
                "independent T1/T2/T3 trajectories",
                "nominal <=0.5 s sampling in M0.90-1.20",
                "abs(alpha)<=2 deg and abs(gamma)<=5 deg under the primary filter",
                "at least two trajectories at each key Mach knot",
            ],
        },
        "status": "phase1_existing_data_review_complete_no_final_fit",
    }

    t_cases = {
        "T1": {"launch_altitude_m": 2000, "initial_mach_target": [1.28, 1.32], "launch_angle_deg": 2, "target_mach": [0.9, 1.3], "role": "main transonic connection training"},
        "T2": {"launch_altitude_m": 2000, "initial_mach_target": [1.13, 1.17], "launch_angle_deg": [1, 2], "target_mach": [0.85, 1.15], "role": "peak-centre training"},
        "T3": {"inputs": "exact field-by-field repeat of T1", "role": "repeatability and complete holdout"},
        "T4": {"launch_altitude_m": 4000, "initial_mach_target": [1.28, 1.32], "launch_angle_deg": 2, "target_mach": [0.95, 1.3], "role": "different-height seam validation"},
    }
    phase2_review = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_label": config["model_label"],
        "new_statshark_calculation_performed_this_run": False,
        "frozen_custom_variant": "AIM120A_H4_GLIDE_NOPWR_NOGUIDE_20260811 [C]",
        "configuration_hash_status": "not_available_as_a_full_saved_UI_config_hash_in_existing_artifacts; must be confirmed before T1-T4",
        "sampling_requirement": {"mach_0_90_to_1_20_max_dt_s": 0.5, "mach_1_20_to_1_30_max_dt_s": 1.0},
        "cases": t_cases,
        "authorization_gate": "STOP before new StatShark Calculate; user must separately authorize T1-T4 website calculations",
        "status": "phase2_ready_for_user_case_review",
    }

    report_lines = [
        "# H4.5 跨音速无动力阻力补洞报告",
        "",
        "## 1. 结论",
        "",
        "本轮执行了 Plan 4.5 Phase 0–1，没有运行新的 StatShark Calculate。现有 G3/G6 只能作为二级诊断；在严格小迎角和采样密度要求下，不允许生成最终跨音速 CdA 曲线或低维预拟合。",
        "",
        f"复审窗口为 M{target_min:.2f}–M{target_max:.2f}；详细原始样本与筛选矩阵见 `outputs/h4_5_transonic_drag/phase1_existing_data_review.json`。",
        "",
        "## 2. Phase 0 冻结",
        "",
        f"- 新 StatShark 计算：`False`。\n- H4 `M>=1.2` 节点保持引用冻结；SHA-256 与节点快照见 `phase0_freeze_manifest.json`。\n- 独立输出目录：`{OUTPUT_DIR.resolve()}`。",
        "",
        "## 3. G3/G6 复审",
        "",
    ]
    for case_id in ("G3", "G6"):
        rows = extracted[case_id]["samples"]
        report_lines.append(f"- {case_id}: {len(rows)} 条 M0.85–1.25 窗口标签；时间分辨率摘要 `{spacing[case_id]['time_delta_s']}`；仅作 `statshark_reference` secondary diagnostic。")
    report_lines.extend([
        "",
        "## 4. 筛选敏感性和可辨识性门",
        "",
        f"主筛选结果：{json.dumps(primary, ensure_ascii=False)}。",
        "",
        "已运行 alpha 2/2.5/3°、航迹角 3/5/8°、q 500/1000/2000 Pa 的诊断矩阵；放宽筛选只用于观察覆盖变化，不进入最终拟合。",
        "",
        "现有可见标签名义上约 2 s 间隔，无法满足 M0.90–1.20 目标的 <=0.5 s 密采样要求；因此低维预拟合判定为 `False`。",
        "",
        "## 5. T1–T4 执行前审查",
        "",
        "已列出 T1/T2/T3/T4 输入和预计 Mach 覆盖，但现有 artifact 没有完整可复核的 clone 配置哈希。根据 Plan 4.5 权限边界，本轮在新网站计算授权门停止。",
        "",
        "下一步需要单独授权 T1–T4 的网站 Calculate；授权后才进入 Phase 3，保存新原始数据并重新运行覆盖/逆算/前向留出。",
        "",
        "## 6. 边界",
        "",
        "RM-10 继续保持 external_shape_prior，不升级为 AIM-120A 的 StatShark 识别曲线；H4 M1.2 以上节点未被本阶段重拟合；没有修改 War Thunder 游戏文件。",
    ])

    (OUTPUT_DIR / "phase0_freeze_manifest.json").write_text(json.dumps(_json_safe(phase0), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "phase1_existing_data_review.json").write_text(json.dumps(_json_safe(phase1), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "phase2_case_review.json").write_text(json.dumps(_json_safe(phase2_review), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "H4_5_TRANSONIC_DRAG_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"phase0={phase0['status']} phase1={phase1['status']} phase2={phase2_review['status']}")
    print(f"written: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
