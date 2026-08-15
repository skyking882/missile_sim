#!/usr/bin/env python3
"""Ingest the bounded H5 StatShark capture without manufacturing a fit.

The browser exposes quantized, user-visible Plotly labels rather than the
solver state.  This script preserves those labels, parses only the displayed
quantities, derives a small flight-path-angle diagnostic from altitude/time,
and refuses the production H5 fit unless the declared 3 x 3 coverage gates
are actually met.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RAW_DIR = PROJECT_DIR / "data" / "raw" / "statshark_h5_body_alpha2"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "h5_body_alpha2"
BUNDLE_PATH = RAW_DIR / "formal_capture_bundle.json"
H4_PATH = PROJECT_DIR / "outputs" / "h4_glide_drag" / "cda_knots_fit.json"

sys.path.insert(0, str(PROJECT_DIR / "src"))

from aim120_model.atmosphere import StandardAtmosphere
from aim120_model.body_alpha2_drag import H4ShapePrior
from aim120_model.inverse_cda import estimate_inverse_cda, summarize_inverse_cda


PRIMARY_MACH = (1.45, 1.55)
CENTER_MACH = (1.47, 1.53)
TARGET_DT = (0.25, 0.50)
CX_LEVELS = (0.0, 9.0, 18.0)
HISTORIES = ("P0", "P1", "P2")
CASE_ORDER = tuple(
    "H5-{}-C{}".format(history, cx)
    for history in HISTORIES
    for cx in (0, 9, 18)
)
ALPHA_BANDS = {
    "P0": (None, 0.4),
    "P1": (1.0, 1.8),
    "P2": (2.3, 3.2),
}

VISIBLE_LABEL_RE = re.compile(
    r"\|\s*(?P<speed>-?[0-9]+(?:\.[0-9]+)?)\s*km/h\s*"
    r"\|\s*M(?P<mach>-?[0-9]+(?:\.[0-9]+)?)\s*"
    r"\|\s*(?P<alpha>-?[0-9]+(?:\.[0-9]+)?)\s*°\s*"
    r"\|\s*(?P<load>-?[0-9]+(?:\.[0-9]+)?)\s*G\s*"
    r"\|\s*(?P<altitude>-?[0-9]+(?:\.[0-9]+)?)\s*m\s*"
    r"\|\s*(?P<target_distance>-?[0-9]+(?:\.[0-9]+)?)\s*m"
)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _visible_label(raw_label: Any) -> str:
    text = str(raw_label or "")
    # Plotly appends its inline SVG stylesheet to the annotation text in the
    # DOM serialization.  The visible annotation ends before this suffix.
    return text.split("   .cls-0", 1)[0].rstrip()


def _parse_visible_sample(
    raw_sample: Mapping[str, Any],
    case: Mapping[str, Any],
    atmosphere: StandardAtmosphere,
) -> dict[str, Any]:
    raw_label = str(raw_sample.get("raw_label", ""))
    visible_label = _visible_label(raw_label)
    match = VISIBLE_LABEL_RE.search(visible_label)
    base = {
        "case_id": str(case.get("case_id", "unknown")),
        "history": str(case.get("history", "unknown")),
        "model_name": str(case.get("model_name", "unknown")),
        "cx_aoa": float(case.get("cx_aoa", float("nan"))),
        "time_s": float(raw_sample.get("time_s", float("nan"))),
        "raw_label": raw_label,
        "visible_annotation": visible_label,
        "source_kind": "statshark_reference_visible_readout",
        "mass_kg": 147.87,
        "thrust_n": 0.0,
        "powered": False,
        "lateral_load_g": 0.0,
        "parse_status": "unparsed",
    }
    if match is None:
        base["parse_error"] = "visible_annotation_pattern_not_found"
        return base

    speed_kmh = float(match.group("speed"))
    altitude_m = float(match.group("altitude"))
    speed_mps = speed_kmh / 3.6
    atmosphere_sample = atmosphere.sample(altitude_m)
    base.update(
        {
            "speed_kmh_display": speed_kmh,
            "speed_mps": speed_mps,
            "mach_display": float(match.group("mach")),
            "mach": float(match.group("mach")),
            "alpha_deg_display": float(match.group("alpha")),
            "alpha_deg": abs(float(match.group("alpha"))),
            "alpha_rad": math.radians(abs(float(match.group("alpha")))),
            "lateral_load_g": abs(float(match.group("load"))),
            "altitude_m": altitude_m,
            "target_distance_m": float(match.group("target_distance")),
            "dynamic_pressure_pa": 0.5 * atmosphere_sample.density_kg_m3 * speed_mps * speed_mps,
            "speed_of_sound_mps": atmosphere_sample.speed_of_sound_mps,
            "parse_status": "parsed",
        }
    )
    return base


def _finite_slope(rows: Sequence[Mapping[str, Any]], index: int, field: str) -> float | None:
    if len(rows) < 2:
        return None
    if index <= 0:
        left, right = rows[0], rows[1]
    elif index >= len(rows) - 1:
        left, right = rows[-2], rows[-1]
    else:
        left, right = rows[index - 1], rows[index + 1]
    t_left = float(left["time_s"])
    t_right = float(right["time_s"])
    if t_right <= t_left:
        return None
    return (float(right[field]) - float(left[field])) / (t_right - t_left)


def _derive_flight_path_angles(rows: Sequence[Mapping[str, Any]]) -> None:
    ordered = sorted(rows, key=lambda item: float(item.get("time_s", float("nan"))))
    for index, row in enumerate(ordered):
        altitude_slope = _finite_slope(ordered, index, "altitude_m")
        speed = float(row.get("speed_mps", float("nan")))
        if altitude_slope is None or not _finite(speed) or speed <= 0.0:
            row["flight_path_angle_rad"] = float("nan")
            row["flight_path_angle_deg"] = float("nan")
            row["altitude_rate_mps"] = None
            continue
        ratio = max(-1.0, min(1.0, altitude_slope / speed))
        gamma = math.asin(ratio)
        row["altitude_rate_mps"] = altitude_slope
        row["flight_path_angle_rad"] = gamma
        row["flight_path_angle_deg"] = math.degrees(gamma)


def _parse_case(case: Mapping[str, Any], atmosphere: StandardAtmosphere) -> list[dict[str, Any]]:
    rows = [
        _parse_visible_sample(sample, case, atmosphere)
        for sample in case.get("samples", [])
    ]
    parsed = [row for row in rows if row.get("parse_status") == "parsed"]
    _derive_flight_path_angles(parsed)
    return rows


def _stats(values: Iterable[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if _finite(value)]
    return {
        "count": len(finite),
        "min": min(finite) if finite else None,
        "median": median(finite) if finite else None,
        "max": max(finite) if finite else None,
    }


def _within(value: float, bounds: tuple[float | None, float | None]) -> bool:
    lower, upper = bounds
    if lower is not None and value < lower:
        return False
    if upper is not None and value > upper:
        return False
    return True


def _coverage_for_case(case: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    parsed = [row for row in rows if row.get("parse_status") == "parsed"]
    primary = [
        row for row in parsed
        if PRIMARY_MACH[0] <= float(row["mach"]) <= PRIMARY_MACH[1]
    ]
    center = [
        row for row in parsed
        if CENTER_MACH[0] <= float(row["mach"]) <= CENTER_MACH[1]
    ]
    alpha_values = [abs(float(row["alpha_deg"])) for row in primary]
    times = sorted(float(row["time_s"]) for row in primary)
    intervals = [right - left for left, right in zip(times, times[1:])]
    history = str(case.get("history", "unknown"))
    nominal_band = ALPHA_BANDS.get(history, (None, None))
    nominal = [value for value in alpha_values if _within(value, nominal_band)]
    bands = {
        name: sum(1 for value in alpha_values if _within(value, bounds))
        for name, bounds in ALPHA_BANDS.items()
    }
    sampling_in_band = bool(intervals) and all(
        TARGET_DT[0] - 1.0e-9 <= value <= TARGET_DT[1] + 1.0e-9
        for value in intervals
    )
    captured = str(case.get("status", "")) == "captured"
    return {
        "case_id": str(case.get("case_id", "unknown")),
        "history": history,
        "model_name": str(case.get("model_name", "unknown")),
        "cx_aoa": float(case.get("cx_aoa", float("nan"))),
        "calculate_action_index": case.get("calculate_action_index"),
        "capture_status": str(case.get("status", "unknown")),
        "failure_text_preserved_in_raw_bundle": bool(case.get("failure")),
        "raw_sample_count": len(case.get("samples", [])),
        "parsed_sample_count": len(parsed),
        "primary_window": {
            "mach_range": list(PRIMARY_MACH),
            "sample_count": len(primary),
            "center_sample_count": len(center),
            "alpha_deg": _stats(alpha_values),
            "alpha_band_counts": bands,
            "nominal_history_band_deg": list(nominal_band),
            "nominal_history_sample_count": len(nominal),
            "time_s_range": _stats([float(row["time_s"]) for row in primary]),
            "sampling_intervals_s": _stats(intervals),
            "sampling_interval_gate": sampling_in_band,
            "at_least_five_samples": len(primary) >= 5,
            "nominal_history_gate": len(nominal) >= 5,
        },
        "center_window": {
            "mach_range": list(CENTER_MACH),
            "sample_count": len(center),
            "alpha_deg": _stats(abs(float(row["alpha_deg"])) for row in center),
        },
        "case_capture_gate": captured and len(parsed) >= 5,
        "formal_case_gate": captured and len(primary) >= 5 and len(nominal) >= 5 and sampling_in_band,
        "input": case.get("input", {}),
    }


def _snapshot_entries(snapshot: Mapping[str, Any], tab_hint: str) -> list[Mapping[str, Any]]:
    for name, entries in snapshot.get("tabs", {}).items():
        if tab_hint.lower() in str(name).lower() or tab_hint in str(name):
            if isinstance(entries, list):
                return entries
    return []


def _entry_value(entries: Sequence[Mapping[str, Any]], index: int) -> Any:
    for entry in entries:
        if int(entry.get("index", -1)) == index:
            return entry.get("value")
    return None


def _float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _snapshot_audit() -> dict[str, Any]:
    expected = {
        "AIM120A_H5_BODY_A2_C0": 0.0,
        "AIM120A_H5_BODY_A2_C9": 9.0,
        "AIM120A_H5_BODY_A2_C18": 18.0,
    }
    model_reports: list[dict[str, Any]] = []
    basic_common: dict[str, list[Any]] = {}
    aero_vectors: dict[str, list[float | None]] = {}
    for model_name, expected_cx in expected.items():
        path = RAW_DIR / ("model_snapshot_" + model_name + ".json")
        report: dict[str, Any] = {
            "model_name": model_name,
            "path": str(path.resolve()),
            "exists": path.is_file(),
            "sha256": _sha256(path) if path.is_file() else None,
        }
        if not path.is_file():
            model_reports.append(report)
            continue
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        tabs = snapshot.get("tabs", {})
        basic = _snapshot_entries(snapshot, "basic")
        aero = _snapshot_entries(snapshot, "aero")
        engine = _snapshot_entries(snapshot, "engine")
        if not aero:
            aero = _snapshot_entries(snapshot, "空气动力学")
        if not engine:
            engine = _snapshot_entries(snapshot, "发动机")
        if not basic:
            basic = _snapshot_entries(snapshot, "基本属性")
        basic_values = [
            _entry_value(basic, index) for index in range(2, 8)
        ]
        # C0's aerodynamic tab begins at index 0; the later reopened tabs
        # repeat the basic fields at indices 0..7 and begin aero at index 8.
        aero_start = 8 if len(aero) >= 20 else 0
        aero_vector = [
            _float_or_none(_entry_value(aero, aero_start + index))
            for index in range(12)
        ]
        report.update(
            {
                "captured_at_utc": snapshot.get("captured_at_utc"),
                "tab_count": len(tabs),
                "required_tabs_present": len(tabs) >= 6,
                "basic_visible_values": basic_values,
                "basic_fields_match_expected": (
                    len(basic_values) == 6
                    and basic_values[0] == model_name
                    and basic_values[1] == "local"
                    and _float_or_none(basic_values[2]) == 147.87
                    and _float_or_none(basic_values[3]) == 0.1778
                    and _float_or_none(basic_values[4]) == 3.66
                    and _float_or_none(basic_values[5]) == 1.275
                ),
                "aerodynamic_vector": aero_vector,
                "expected_aerodynamic_vector": [
                    1.425, 2.2, expected_cx, 1.0, 0.0, 0.17500001,
                    0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                ],
                "aerodynamic_vector_matches_expected": all(
                    actual is not None and abs(actual - expected_value) <= 1.0e-7
                    for actual, expected_value in zip(
                        aero_vector,
                        [1.425, 2.2, expected_cx, 1.0, 0.0, 0.17500001, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                    )
                ),
                "engine_stage_zero_thrust_fields_present": (
                    _float_or_none(_entry_value(engine, 20)) == 0.0
                    and _float_or_none(_entry_value(engine, 22)) == 0.0
                    and _float_or_none(_entry_value(engine, 23)) == 0.0
                    and _float_or_none(_entry_value(engine, 32)) == 0.0
                    and _float_or_none(_entry_value(engine, 34)) == 0.0
                    and _float_or_none(_entry_value(engine, 35)) == 0.0
                ),
            }
        )
        basic_common[model_name] = basic_values[1:]
        aero_vectors[model_name] = aero_vector
        model_reports.append(report)
    complete = all(
        report.get("exists")
        and report.get("required_tabs_present")
        and report.get("basic_fields_match_expected")
        and report.get("aerodynamic_vector_matches_expected")
        and report.get("engine_stage_zero_thrust_fields_present")
        for report in model_reports
    ) and len(model_reports) == 3
    common_equal = len({json.dumps(value, sort_keys=True) for value in basic_common.values()}) == 1
    cx_vector_isolation = (
        len(aero_vectors) == 3
        and all(
            all(
                left == right
                for index, (left, right) in enumerate(zip(aero_vectors["AIM120A_H5_BODY_A2_C0"], vector))
                if index != 2
            )
            for vector in aero_vectors.values()
        )
        and [aero_vectors[name][2] for name in expected if name in aero_vectors] == list(expected.values())
    )
    return {
        "model_reports": model_reports,
        "all_three_complete_visible_snapshots": complete,
        "common_basic_fields_equal": common_equal,
        "only_CxAoA_differs_in_aerodynamic_vector": cx_vector_isolation,
        "configuration_snapshot_gate": complete and common_equal and cx_vector_isolation,
    }


def _normalized_action_ledger(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    formal_mapping = {
        2: "H5-P2-C0",
        3: "H5-P2-C9",
        4: "H5-P2-C18",
        5: "H5-P2-C18",
        11: "H5-P0-C9",
        12: "H5-P0-C18",
        13: "H5-P1-C0",
        14: "H5-P1-C9",
        15: "H5-P1-C18",
    }
    output: list[dict[str, Any]] = []
    for raw in bundle.get("action_ledger", []):
        item = dict(raw)
        index = int(item.get("action_index", -1))
        if index == 6 or index == 8:
            item["formal_case_id"] = None
        elif index == 9:
            item["formal_case_id"] = "H5-P0-C0"
        elif index in formal_mapping:
            item["formal_case_id"] = formal_mapping[index]
        output.append(item)
    return output


def _case_rows_for_inverse(all_case_rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case_rows in all_case_rows.values():
        rows.extend(
            dict(row)
            for row in case_rows
            if row.get("parse_status") == "parsed" and _finite(row.get("flight_path_angle_rad"))
        )
    return rows


def _add_h4_diagnostics(rows: Sequence[Mapping[str, Any]], prior: H4ShapePrior) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        mach = float(item.get("mach", float("nan")))
        if _finite(mach) and prior.mach_knots[0] <= mach <= prior.mach_knots[-1]:
            prior_cda = prior.cda_m2(mach)
            item["h4_prior_cda_m2"] = prior_cda
            item["h4_prior_support_label"] = "direct_support_or_interpolation"
            if item.get("inverse_cda_valid") and _finite(item.get("inverse_cda_m2")):
                item["residual_cda_vs_h4_m2"] = float(item["inverse_cda_m2"]) - prior_cda
        else:
            item["h4_prior_cda_m2"] = None
            item["h4_prior_support_label"] = "outside_frozen_support"
            item["residual_cda_vs_h4_m2"] = None
        output.append(item)
    return output


def _pairwise_c0_c9_diagnostic(
    case_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for history in HISTORIES:
        c0 = {
            round(float(row["time_s"]), 6): row
            for row in case_rows.get("H5-{}-C0".format(history), [])
            if row.get("parse_status") == "parsed"
        }
        c9 = {
            round(float(row["time_s"]), 6): row
            for row in case_rows.get("H5-{}-C9".format(history), [])
            if row.get("parse_status") == "parsed"
        }
        for time_s in sorted(set(c0).intersection(c9)):
            left, right = c0[time_s], c9[time_s]
            if PRIMARY_MACH[0] <= float(left["mach"]) <= PRIMARY_MACH[1] or PRIMARY_MACH[0] <= float(right["mach"]) <= PRIMARY_MACH[1]:
                records.append(
                    {
                        "history": history,
                        "time_s": time_s,
                        "mach_c0": left["mach"],
                        "mach_c9": right["mach"],
                        "alpha_c0_deg": left["alpha_deg"],
                        "alpha_c9_deg": right["alpha_deg"],
                        "speed_delta_c9_minus_c0_mps": float(right["speed_mps"]) - float(left["speed_mps"]),
                        "altitude_delta_c9_minus_c0_m": float(right["altitude_m"]) - float(left["altitude_m"]),
                    }
                )
    return records


def _build_report(
    bundle: Mapping[str, Any],
    coverage: Sequence[Mapping[str, Any]],
    acceptance: Mapping[str, Any],
    inverse_summary: Mapping[str, Any],
) -> str:
    lines = [
        "# H5 零主动舵面整弹迎角平方阻力：正式采集结果",
        "",
        "> 结论：本轮完成了本地修复、模型快照、15 次授权内的正式采集和诊断；Plan 5 主拟合被覆盖率门阻塞，未输出 `CdA0_1p5` 或 `K_alpha2_nominal_1p5` 的正式估计。",
        "",
        "## 1. 停止状态",
        "",
        "- 最终状态：`{}`。".format(acceptance["final_status"]),
        "- 15/15 次 `Calculate` 已使用；失败和空结果仍计入并保留，当前授权下不再追加计算。",
        "- 三个模型的可见字段快照均已保存；游戏文件、H1–H4.5 产物未覆盖。",
        "- 逆算 CdA 仅是由页面可见速度、海拔和迎角读数形成的诊断，不等于 StatShark 后端公式复现。",
        "",
        "## 2. 正式矩阵覆盖",
        "",
        "| 案例 | 状态 | 原始/解析点 | M1.45–1.55 | 名义迎角点 | 页面迎角范围 | 主门 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for item in coverage:
        window = item["primary_window"]
        alpha = window["alpha_deg"]
        lines.append(
            "| {case} | {status} | {raw}/{parsed} | {primary} | {nominal} | {amin}–{amax}° | {gate} |".format(
                case=item["case_id"],
                status=item["capture_status"],
                raw=item["raw_sample_count"],
                parsed=item["parsed_sample_count"],
                primary=window["sample_count"],
                nominal=window["nominal_history_sample_count"],
                amin="not available" if alpha["min"] is None else "{:.1f}".format(alpha["min"]),
                amax="not available" if alpha["max"] is None else "{:.1f}".format(alpha["max"]),
                gate="pass" if item["formal_case_gate"] else "fail",
            )
        )
    lines.extend(
        [
            "",
            "P0 的已捕获 C0/C9 历史在主窗口虽有密采样，但只有约 0.0–0.3° 的少数点满足 P0；P1 的 C0/C9 捕获落在约 1.0–1.1°；P2 的 C0/C9 捕获实际约 1.2°，不满足预声明的 2.3–3.2°。C18 三个正式案例均没有可用滑块读数。",
            "",
            "## 3. 验收门",
            "",
            "| 门 | 结果 |",
            "|---|---|",
        ]
    )
    for name, result in acceptance["gates"].items():
        lines.append("| `{}` | {} |".format(name, "PASS" if result else "FAIL"))
    lines.extend(
        [
            "",
            "## 4. 诊断产物",
            "",
            "- `h5_normalized_samples.json`：保留原始 UI label，并附加显示值解析和由海拔时间差得到的 flight-path-angle 诊断。",
            "- `h5_coverage_report.json`：逐案例窗口、迎角档位、采样间隔、矩阵和停止条件。",
            "- `h5_inverse_cda_diagnostics.json`：可见状态反算 CdA 与冻结 H4 形状的差值诊断。有效样本摘要键数：`{}`。".format(len(inverse_summary)),
            "- `h5_fit_diagnostics.json`：主参数保持 null；没有把不完整矩阵当作正式拟合。",
            "- `h5_formal_source_manifest.json`：授权、次数、快照、原始 bundle 和 SHA-256 provenance。",
            "",
            "## 5. 模型边界",
            "",
            "本轮分析对象仍是零主动舵面、完整外形整弹的有效 `CdA`。它不是拆除尾翼/弹翼后的裸弹体；没有拟合 fin 阻力、`sin²(alpha)`、交叉项或全 Mach 函数，也没有把旧 H4 的 M1.5 有效节点当作新的零迎角截距。",
        ]
    )
    return "\n".join(lines) + "\n"


def build_outputs() -> dict[str, Any]:
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    atmosphere = StandardAtmosphere()
    cases_by_id: dict[str, Mapping[str, Any]] = {
        str(case.get("case_id")): case for case in bundle.get("formal_cases", [])
    }
    all_case_rows: dict[str, list[dict[str, Any]]] = {
        case_id: _parse_case(case, atmosphere)
        for case_id, case in cases_by_id.items()
    }
    coverage = [
        _coverage_for_case(cases_by_id[case_id], all_case_rows.get(case_id, []))
        for case_id in CASE_ORDER
        if case_id in cases_by_id
    ]
    snapshot_audit = _snapshot_audit()
    prior_manifest = json.loads(
        (OUTPUT_DIR / "phase0_freeze_manifest.json").read_text(encoding="utf-8")
    )
    h4_prior = H4ShapePrior.from_json(H4_PATH)
    inverse_input = _case_rows_for_inverse(all_case_rows)
    inverse_rows = estimate_inverse_cda(inverse_input, window_points=5)
    inverse_rows = _add_h4_diagnostics(inverse_rows, h4_prior)
    inverse_summary = summarize_inverse_cda(inverse_rows)
    formal_case_count = len(CASE_ORDER)
    captured_count = sum(1 for item in coverage if item["case_capture_gate"])
    formal_gate_count = sum(1 for item in coverage if item["formal_case_gate"])
    c18_cases = [item for item in coverage if item["cx_aoa"] == 18.0]
    p0_cases = [item for item in coverage if item["history"] == "P0"]
    p1_cases = [item for item in coverage if item["history"] == "P1"]
    p2_cases = [item for item in coverage if item["history"] == "P2"]
    all_captured = captured_count == formal_case_count
    p0_alpha_gate = bool(p0_cases) and all(
        item["primary_window"]["nominal_history_gate"] for item in p0_cases
    )
    p1_alpha_gate = bool(p1_cases) and all(
        item["primary_window"]["nominal_history_gate"] for item in p1_cases
    )
    p2_alpha_gate = bool(p2_cases) and all(
        item["primary_window"]["nominal_history_gate"] for item in p2_cases
    )
    c18_gate = bool(c18_cases) and all(item["case_capture_gate"] for item in c18_cases)
    matrix_gate = all_captured and formal_gate_count == formal_case_count and c18_gate
    final_status = "h5_formal_fit_blocked_missing_C18_and_alpha_coverage"
    gates = {
        "local_weighted_rmse_fix": True,
        "partial_snapshot_return_fix": True,
        "synthetic_design_matrix_gate": True,
        "configuration_snapshot_gate": bool(snapshot_audit["configuration_snapshot_gate"]),
        "three_by_three_capture_gate": matrix_gate,
        "P0_nominal_alpha_gate": p0_alpha_gate,
        "P1_nominal_alpha_gate": p1_alpha_gate,
        "P2_nominal_alpha_gate": p2_alpha_gate,
        "C18_field_scaling_gate": c18_gate,
        "P0_P1_P2_overlap_gate": matrix_gate and p0_alpha_gate and p1_alpha_gate and p2_alpha_gate,
        "main_full_trajectory_fit_gate": False,
        "calculate_budget_not_exceeded": int(bundle["action_budget"]["calculate_actions_used"]) <= int(bundle["action_budget"]["authorized_max_calculate_actions"]),
        "game_files_unmodified": bool(prior_manifest.get("game_files_modified") is False),
        "prior_H4_H4_5_unchanged": bool(prior_manifest.get("h4_h4_5_outputs_overwritten") is False and prior_manifest.get("prior_h4_5_freeze_comparison", {}).get("all_unchanged")),
    }
    normalized_rows = [
        row
        for case_id in CASE_ORDER
        for row in all_case_rows.get(case_id, [])
    ]
    normalized_payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_bundle": str(BUNDLE_PATH.resolve()),
        "source_bundle_sha256": _sha256(BUNDLE_PATH),
        "source_kind": "statshark_reference_visible_readout",
        "raw_labels_preserved": True,
        "display_values_are_quantized": True,
        "alpha_unit_for_analysis": "rad",
        "rows": normalized_rows,
        "failed_or_empty_formal_cases": [
            {
                "case_id": str(case.get("case_id")),
                "status": str(case.get("status")),
                "failure": case.get("failure"),
                "raw_case_record_preserved": True,
            }
            for case in cases_by_id.values()
            if str(case.get("status")) != "captured"
        ],
    }
    coverage_payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_bundle": str(BUNDLE_PATH.resolve()),
        "action_budget": bundle.get("action_budget", {}),
        "action_ledger_normalized": _normalized_action_ledger(bundle),
        "declared_windows": {
            "primary_mach": list(PRIMARY_MACH),
            "center_mach": list(CENTER_MACH),
            "target_sampling_interval_s": list(TARGET_DT),
            "alpha_bands_deg": {key: list(value) for key, value in ALPHA_BANDS.items()},
        },
        "cases": coverage,
        "matrix": {
            "formal_case_order": list(CASE_ORDER),
            "formal_case_count": formal_case_count,
            "captured_case_count": captured_count,
            "formal_case_gate_count": formal_gate_count,
            "captured_CxAoA_levels": sorted({item["cx_aoa"] for item in coverage if item["case_capture_gate"]}),
            "captured_histories": sorted({item["history"] for item in coverage if item["case_capture_gate"]}),
            "full_3x3_capture": matrix_gate,
        },
        "model_snapshot_audit": snapshot_audit,
        "stopping_conditions": [
            "three_angle_matrix_not_formed",
            "C18_has_no_usable_formal_readout",
            "P0_and_P2_nominal_alpha_gates_not_met",
            "calculate_action_budget_exhausted",
        ],
        "status": final_status,
    }
    inverse_payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_kind": "statshark_reference_visible_readout",
        "source_bundle": str(BUNDLE_PATH.resolve()),
        "diagnostic_boundary": "inverse effective CdA from displayed speed, altitude, mass, and derived flight-path angle; not a StatShark solver reproduction",
        "flight_path_angle_method": "finite difference of displayed altitude divided by displayed speed, then asin",
        "h4_shape_prior": {
            "path": str(H4_PATH.resolve()),
            "sha256": _sha256(H4_PATH),
            "mach_knots": list(h4_prior.mach_knots),
            "cda_knots_m2": list(h4_prior.cda_knots_m2),
            "m1p5_cda_effective_m2": h4_prior.cda_m2(1.5),
        },
        "rows": inverse_rows,
        "summary": inverse_summary,
        "status": "diagnostic_only_formal_fit_blocked",
    }
    fit_payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_label": "local_candidate_H5_body_alpha2_M1p5",
        "fit_parameterization": {
            "main_parameters": ["CdA0_1p5", "K_residual_1p5", "S_CxAoA_1p5"],
            "alpha_unit": "rad",
            "mach_shape": "frozen_H4_log_linear_delta",
            "sin_squared_model_enabled": False,
            "fin_drag_fit_enabled": False,
            "alpha_fin_body_cross_term_enabled": False,
        },
        "main_parameters": {
            "CdA0_1p5_m2": None,
            "K_residual_1p5_m2_per_rad2": None,
            "S_CxAoA_1p5_m2_per_rad2_per_field_unit": None,
            "K_alpha2_nominal_1p5_m2_per_rad2": None,
        },
        "fit_status": "blocked_missing_complete_3x3_matrix_and_nominal_alpha_coverage",
        "reason": "C18 formal cases returned frontend_error or empty_result, P0 did not sustain <=0.4 deg for five primary-window points, and P2 actual displayed alpha was about 1.2 deg rather than the declared 2.3-3.2 deg band.",
        "available_inverse_cda_summary": inverse_summary,
        "available_C0_C9_pairwise_diagnostic": _pairwise_c0_c9_diagnostic(all_case_rows),
        "no_partial_parameter_fit_reported": True,
    }
    acceptance_payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": bundle.get("action_budget", {}),
        "gates": gates,
        "formal_case_count": formal_case_count,
        "captured_case_count": captured_count,
        "formal_case_gate_count": formal_gate_count,
        "final_status": final_status,
        "parameters": fit_payload["main_parameters"],
        "stopping_conditions": coverage_payload["stopping_conditions"],
        "scope_boundaries": {
            "effective_cda_not_unique_Cx_or_area": True,
            "zero_active_fin_full_missile_not_bare_body": True,
            "no_full_mach_extrapolation": True,
            "no_fin_drag_or_cross_term_fit": True,
            "no_backend_solver_claim": True,
        },
    }
    source_manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_label": "local_candidate_H5_body_alpha2_M1p5",
        "statshark_new_calculation_performed_this_run": True,
        "statshark_new_calculation_authorized": True,
        "authorized_max_calculate_actions": int(bundle["action_budget"]["authorized_max_calculate_actions"]),
        "calculate_actions_used": int(bundle["action_budget"]["calculate_actions_used"]),
        "calculate_actions_remaining": int(bundle["action_budget"]["authorized_max_calculate_actions"]) - int(bundle["action_budget"]["calculate_actions_used"]),
        "game_files_modified": False,
        "h1_h4_5_outputs_overwritten": False,
        "raw_artifacts": [
            {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in [BUNDLE_PATH]
            + sorted(RAW_DIR.glob("model_snapshot_*.json"))
        ],
        "derived_artifacts": [
            "outputs/h5_body_alpha2/h5_normalized_samples.json",
            "outputs/h5_body_alpha2/h5_coverage_report.json",
            "outputs/h5_body_alpha2/h5_inverse_cda_diagnostics.json",
            "outputs/h5_body_alpha2/h5_fit_diagnostics.json",
            "outputs/h5_body_alpha2/h5_acceptance_report.json",
        ],
        "evidence_boundaries": {
            "statshark_reference_visible_readout": "formal H5 rows and failures preserved; visible values quantized",
            "inverse_cda_diagnostic": "local diagnostic only",
            "synthetic_test": "local identifiability only",
            "prior_reference_checkpoint": "frozen H4 shape and M1.5 effective node; not refit",
        },
        "status": final_status,
    }
    _write_json(OUTPUT_DIR / "h5_normalized_samples.json", normalized_payload)
    _write_json(OUTPUT_DIR / "h5_coverage_report.json", coverage_payload)
    _write_json(OUTPUT_DIR / "h5_inverse_cda_diagnostics.json", inverse_payload)
    _write_json(OUTPUT_DIR / "h5_fit_diagnostics.json", fit_payload)
    _write_json(OUTPUT_DIR / "h5_acceptance_report.json", acceptance_payload)
    _write_json(OUTPUT_DIR / "h5_formal_source_manifest.json", source_manifest)
    (OUTPUT_DIR / "H5_BODY_ALPHA2_FORMAL_REPORT.md").write_text(
        _build_report(bundle, coverage, acceptance_payload, inverse_summary),
        encoding="utf-8",
    )
    return {
        "status": final_status,
        "coverage": {
            "formal_case_count": formal_case_count,
            "captured_case_count": captured_count,
            "formal_case_gate_count": formal_gate_count,
        },
        "output_dir": str(OUTPUT_DIR.resolve()),
    }


def main() -> int:
    result = build_outputs()
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
