#!/usr/bin/env python3
"""Fit and report the low-g effective drag-area model hierarchy."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from aim120_model.config import load_model_config
from aim120_model.low_g_drag import (
    coefficient_names,
    evaluate_rows,
    fit_linear_bounded,
    model_basis,
    model_parameter_dict,
)
from aim120_model.sample_filters import LowGFilterSettings, apply_filter


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


def _settings_from_config(config: Mapping[str, Any]) -> LowGFilterSettings:
    values = config["low_g_drag"]["sample_filter"]
    return LowGFilterSettings(
        lateral_load_threshold_g=float(values["lateral_load_threshold_g"]),
        alpha_threshold_deg=float(values["alpha_threshold_deg"]),
        flight_path_threshold_deg=float(values["flight_path_threshold_deg"]),
        q_min_pa=float(values["q_min_pa"]),
        burn_stage_1_end_s=float(values["burn_stage_1_end_s"]),
        burn_end_s=float(values["burn_end_s"]),
        stage_1_exclusion_window_s=float(values["stage_1_exclusion_window_s"]),
        burn_end_exclusion_window_s=float(values["burn_end_exclusion_window_s"]),
    )


def _finite_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("accepted"):
            continue
        value = row.get("observed_cda_m2")
        if value is None or not math.isfinite(float(value)) or float(value) <= 0.0:
            continue
        result.append(dict(row))
    return result


def _split_rows(rows: Sequence[Mapping[str, Any]], modulo: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    safe_modulo = max(int(modulo), 2)
    train: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        (holdout if index % safe_modulo == 0 else train).append(dict(row))
    if not holdout and train:
        holdout.append(train.pop())
    if not train and holdout:
        train.append(holdout.pop())
    return train, holdout


def _anchor_row(config: Mapping[str, Any]) -> dict[str, Any]:
    anchor = config["reference"]["h2_high_mach_anchor"]
    return {
        "source_case": "h2_high_mach_local_anchor",
        "source_kind": "local_prior_not_statshark",
        "time_s": 7.0,
        "powered": False,
        "engine_stage": 0,
        "mach": float(anchor["mach"]),
        "alpha_rad": 0.0,
        "observed_cda_m2": float(anchor["cda0_m2"]),
        "fit_weight": float(config["low_g_drag"]["fit"]["anchor_prior_ridge_weight"]),
        "accepted": True,
        "rejection_reasons": [],
    }


def _metrics(rows: Sequence[Mapping[str, Any]], coefficients: Sequence[float], level: str, config: Mapping[str, Any]) -> dict[str, Any]:
    center = float(config["low_g_drag"]["mach_parameterization"]["center"])
    width = float(config["low_g_drag"]["mach_parameterization"]["width"])
    wave_width = float(config["low_g_drag"]["mach_parameterization"]["wave_width"])
    residuals: list[float] = []
    relative: list[float] = []
    powered_residuals: list[float] = []
    coast_residuals: list[float] = []
    evaluated = evaluate_rows(rows, coefficients, level, center, width, wave_width)
    for row in evaluated:
        residual = row.get("cda_residual_m2")
        target = row.get("observed_cda_m2")
        if residual is None or target is None:
            continue
        residual = float(residual)
        target = float(target)
        residuals.append(residual)
        relative.append(residual / max(abs(target), 1.0e-12))
        (powered_residuals if row.get("powered") else coast_residuals).append(residual)
    if not residuals:
        return {"count": 0, "rmse_m2": None, "mae_m2": None, "median_abs_m2": None, "relative_rmse": None}
    return {
        "count": len(residuals),
        "rmse_m2": math.sqrt(mean(value * value for value in residuals)),
        "mae_m2": mean(abs(value) for value in residuals),
        "median_abs_m2": median(abs(value) for value in residuals),
        "relative_rmse": math.sqrt(mean(value * value for value in relative)),
        "mean_signed_residual_m2": mean(residuals),
        "powered_rmse_m2": math.sqrt(mean(value * value for value in powered_residuals)) if powered_residuals else None,
        "coast_rmse_m2": math.sqrt(mean(value * value for value in coast_residuals)) if coast_residuals else None,
    }


def _uncertainty(rows: Sequence[Mapping[str, Any]], coefficients: Sequence[float], level: str, config: Mapping[str, Any]) -> dict[str, Any]:
    center = float(config["low_g_drag"]["mach_parameterization"]["center"])
    width = float(config["low_g_drag"]["mach_parameterization"]["width"])
    wave_width = float(config["low_g_drag"]["mach_parameterization"]["wave_width"])
    residuals: list[float] = []
    information = [0.0 for _ in coefficient_names(level)]
    for row in rows:
        target = row.get("observed_cda_m2")
        if target is None:
            continue
        basis = model_basis(
            float(row["mach"]),
            float(row.get("alpha_rad", 0.0)),
            bool(row.get("powered", False)),
            level,
            center,
            width,
            wave_width,
        )
        prediction = sum(float(a) * float(b) for a, b in zip(coefficients, basis))
        residuals.append(float(target) - prediction)
        row_weight = float(row.get("fit_weight", 1.0))
        for index, value in enumerate(basis):
            information[index] += row_weight * value * value
    if not residuals:
        return {"method": "diagonal_residual_scale", "standard_error": {}}
    sigma = math.sqrt(mean(value * value for value in residuals))
    return {
        "method": "diagonal_residual_scale",
        "warning": "approximate only; parameter correlation and local-model provenance dominate",
        "residual_scale_m2": sigma,
        "standard_error": {
            name: sigma / math.sqrt(max(information[index], 1.0e-18))
            for index, name in enumerate(coefficient_names(level))
        },
    }


def _positivity_check(coefficients: Sequence[float], level: str, config: Mapping[str, Any]) -> dict[str, Any]:
    center = float(config["low_g_drag"]["mach_parameterization"]["center"])
    width = float(config["low_g_drag"]["mach_parameterization"]["width"])
    wave_width = float(config["low_g_drag"]["mach_parameterization"]["wave_width"])
    violations: list[dict[str, Any]] = []
    for powered in (False, True):
        for mach_index in range(0, 31):
            mach = 0.2 + 0.1 * mach_index
            for alpha_rad in (0.0, math.radians(5.0)):
                basis = model_basis(mach, alpha_rad, powered, level, center, width, wave_width)
                value = sum(float(a) * float(b) for a, b in zip(coefficients, basis))
                if value <= 0.0 or not math.isfinite(value):
                    violations.append({"powered": powered, "mach": mach, "alpha_deg": math.degrees(alpha_rad), "cda_m2": value})
    return {"positive_over_check_grid": not violations, "violation_count": len(violations), "violations": violations[:20]}


def _fit_level(rows: Sequence[Mapping[str, Any]], train: Sequence[Mapping[str, Any]], holdout: Sequence[Mapping[str, Any]], level: str, config: Mapping[str, Any]) -> dict[str, Any]:
    mach_cfg = config["low_g_drag"]["mach_parameterization"]
    anchor = _anchor_row(config)
    coefficients = fit_linear_bounded(
        list(train) + [anchor],
        level=level,
        center=float(mach_cfg["center"]),
        width=float(mach_cfg["width"]),
        wave_width=float(mach_cfg["wave_width"]),
    )
    return {
        "level": level,
        "coefficient_names": list(coefficient_names(level)),
        "coefficients": coefficients,
        "parameters": model_parameter_dict(
            coefficients,
            level,
            float(mach_cfg["center"]),
            float(mach_cfg["width"]),
            float(mach_cfg["wave_width"]),
        ),
        "train_metrics": _metrics(train, coefficients, level, config),
        "holdout_metrics": _metrics(holdout, coefficients, level, config),
        "all_accepted_metrics": _metrics(rows, coefficients, level, config),
        "approximate_uncertainty": _uncertainty(list(train) + [anchor], coefficients, level, config),
        "positivity_check": _positivity_check(coefficients, level, config),
        "anchor_used": {
            "mach": anchor["mach"],
            "cda0_m2": anchor["observed_cda_m2"],
            "uncertainty_m2": config["reference"]["h2_high_mach_anchor"]["uncertainty_m2"],
            "source": anchor["source_kind"],
        },
    }


def _threshold_variants(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any], base: LowGFilterSettings) -> dict[str, Any]:
    values = config["low_g_drag"]["sensitivity"]
    variants: dict[str, Any] = {}
    for label, field, candidates in (
        ("lateral_load", "lateral_load_threshold_g", values["lateral_load_thresholds_g"]),
        ("alpha", "alpha_threshold_deg", values["alpha_thresholds_deg"]),
        ("flight_path", "flight_path_threshold_deg", values["flight_path_thresholds_deg"]),
    ):
        for candidate in candidates:
            settings = replace(base, **{field: float(candidate)})
            filtered: list[dict[str, Any]] = []
            for row in rows:
                item = apply_filter(row, settings)
                observed = row.get("observed_cda_m2")
                if observed is None or not math.isfinite(float(observed)) or float(observed) <= 0.0:
                    item["accepted"] = False
                    item.setdefault("rejection_reasons", []).append("inverse_drag_not_positive")
                filtered.append(item)
            selected = _finite_rows(filtered)
            if len(selected) < 8:
                variants[f"{label}_{float(candidate):g}"] = {"accepted_rows": len(selected), "fit": None}
                continue
            train, holdout = _split_rows(selected, int(config["low_g_drag"]["fit"]["holdout_modulo"]))
            fit = _fit_level(selected, train, holdout, "LG-0", config)
            variants[f"{label}_{float(candidate):g}"] = {
                "accepted_rows": len(selected),
                "a_sub_m2": fit["parameters"]["a_sub_m2"],
                "a_sup_m2": fit["parameters"]["a_sup_m2"],
                "a_wave_m2": fit["parameters"]["a_wave_m2"],
                "k_alpha_m2_per_rad2": fit["parameters"]["k_alpha_m2_per_rad2"],
                "holdout_rmse_m2": fit["holdout_metrics"]["rmse_m2"],
            }
    return variants


def _mach_range(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    values = [float(row["mach"]) for row in rows if row.get("mach") is not None and math.isfinite(float(row["mach"]))]
    return {"min": min(values) if values else None, "max": max(values) if values else None}


def _write_report(
    path: Path,
    generated_at: str,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    comparison: Mapping[str, Any],
    fit_report: Mapping[str, Any],
    selection: Mapping[str, Any],
    accepted_rows: Sequence[Mapping[str, Any]],
) -> None:
    summary = manifest["summary"]
    params = fit_report["parameters"]
    train = fit_report["train_metrics"]
    holdout = fit_report["holdout_metrics"]
    overlap = comparison.get("comparison", comparison)
    overlap_count = int(overlap.get("overlap_bin_count", 0))
    lines = [
        "# H3 低横向过载阻力辨识报告",
        "",
        "## 1. 一句话结论",
        "",
        "当前只能把 LG-0 共享 `CdA0(M)` 作为本地管道验证模型；没有独立的 StatShark 动力/滑行时间序列，不能证实动力段修正 `Delta_CdA_burn`，因此按停止条件冻结在 LG-0。",
        "",
        "## 2. 实际数据来源和 Mach 范围",
        "",
        f"- 生成时间：`{generated_at}`。",
        f"- 数据来源：`{manifest.get('source_kind')}`；输入文件均为本地 H2 输出，未标成 StatShark 参考。",
        f"- 使用样本：{summary.get('total_rows', 0)} 行，接受 {summary.get('accepted_rows', 0)} 行；接受样本 Mach 范围 `{_mach_range(accepted_rows)['min']}`–`{_mach_range(accepted_rows)['max']}`。",
        f"- 接受动力样本：{summary.get('accepted_by_power_state', {}).get('powered', 0)}；接受滑行样本：{summary.get('accepted_by_power_state', {}).get('coast', 0)}。",
        "- H2 7 s 高 Mach 锚点仅作为带不确定度的本地先验；它不是 StatShark 原始时间序列，也不能验证 Mach 曲线形状。",
        "",
        "## 3. 筛选阈值和微分设置",
        "",
        f"- 横向过载：`lateral_load_g <= {config['low_g_drag']['sample_filter']['lateral_load_threshold_g']} g`。",
        f"- 迎角：`abs(alpha) <= {config['low_g_drag']['sample_filter']['alpha_threshold_deg']}°`；飞行路径角：`abs(gamma) <= {config['low_g_drag']['sample_filter']['flight_path_threshold_deg']}°`。",
        f"- 动压下限：`q > {config['low_g_drag']['sample_filter']['q_min_pa']} Pa`。",
        f"- 发动机切换排除窗：1.7 s ± {config['low_g_drag']['sample_filter']['stage_1_exclusion_window_s']} s；7.0 s ± {config['low_g_drag']['sample_filter']['burn_end_exclusion_window_s']} s。",
        f"- 速度使用不跨越动力段/滑行段或 Stage 边界的局部二阶多项式，比较窗口 `{config['low_g_drag']['smoothing']['window_widths_s']}` s；窗口敏感性见 `sample_manifest.json`。",
        "",
        "## 4. 共享 CdA0(M) 参数和不确定度",
        "",
        f"- LG-0 参数：`A_sub={params['a_sub_m2']:.8g} m²`，`A_sup={params['a_sup_m2']:.8g} m²`，`A_wave={params['a_wave_m2']:.8g} m²`，`K_alpha={params['k_alpha_m2_per_rad2']:.8g} m²/rad²`。",
        f"- 固定形状参数：center `{params['center']}`，transition width `{params['width']}`，wave width `{params['wave_width']}`。",
        f"- 训练集误差：RMSE `{train.get('rmse_m2')}` m²；留出集误差：RMSE `{holdout.get('rmse_m2')}` m²。",
        f"- 不确定度是残差尺度下的对角近似：`{fit_report['approximate_uncertainty'].get('standard_error', {})}`；由于输入是本地模型生成数据，不能解释为 StatShark 参数置信区间。",
        "",
        "## 5. 动力与滑行残差比较",
        "",
        f"- 同 Mach/高度分箱同时含动力和滑行样本的数量：`{overlap_count}`。",
        f"- 组间均值差：`{overlap.get('aggregate', {}).get('mean_powered_minus_coast_cda_m2')}` m²；绝对差均值：`{overlap.get('aggregate', {}).get('mean_abs_powered_minus_coast_cda_m2')}` m²。",
        "- 这些差异来自同一套本地 H2 候选模型的轨迹反算，不能作为隐藏服务器动力段差异的证据。",
        "",
        "## 6. 是否需要 Delta_CdA_burn",
        "",
        f"- 当前选择：`{selection.get('selected_model')}`。",
        f"- 判定：{selection.get('reason')}",
        "- LG-1/LG-2 即使在本地数据上降低训练误差，也没有获得独立参考时间序列和重叠 Mach/高度条件的资格；不提升复杂度。",
        "",
        "## 7. 推力倍率退化风险",
        "",
        "动力段反算使用 `D = T - m*(dV/dt + g*sin(gamma))`。任何真实推力倍率 `K_thrust(M,h)` 的误差都会直接进入动力段阻力残差；因此当前不能把 powered residual 命名为喷流阻力或唯一的 `Delta_CdA_burn`。",
        "",
        "## 8. 训练、留出和阈值稳定性",
        "",
        f"- 数据划分：每 5 个样本留出 1 个，且动力/滑行组使用平衡权重。训练 RMSE `{train.get('rmse_m2')}`，留出 RMSE `{holdout.get('rmse_m2')}`。",
        "- 多阈值拟合结果保存在 `model_selection_report.json` 的 `threshold_sensitivity`；它只能说明本地管道的数量级稳定性，不能替代独立参考数据。",
        "",
        "## 9. 插值和外推范围",
        "",
        f"- 稀疏模型在 `{params['center'] - 2 * params['width']:.3g}`–`{params['center'] + 2 * params['width']:.3g}` 的跨声速形状由固定 transition/wave 宽度插值表达；实际样本 Mach 范围由第 2 节给出。",
        "- 低于样本最小 Mach、超过样本最大 Mach 的部分属于模型外推；不能称为已识别曲线。",
        "- 参考面积与 `Cx(M)` 的分解仍不可辨识，报告量是有效 `CdA`。",
        "",
        "## 10. 下一条最有价值的参考时间序列",
        "",
        "优先取得 Case B：高速无动力滑行，覆盖 Mach 3→1，且高度与动力段 Case A 重叠；其次取得 Case C：中低速无动力滑行，覆盖 Mach 1.2→0.3，避免大迎角下坠。两者都应保存 time、speed、Mach、height、flight-path angle、mass、thrust、AoA 和 lateral load。获得授权前不运行新的 StatShark Calculate。",
        "",
        "## 数据边界",
        "",
        "本报告没有新增 StatShark 计算，没有修改 War Thunder 游戏文件，也没有覆盖 H1/H2 输出。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=PROJECT_DIR / "outputs" / "h3_low_g_drag" / "inverse_cda_samples.json")
    parser.add_argument("--manifest", type=Path, default=PROJECT_DIR / "outputs" / "h3_low_g_drag" / "sample_manifest.json")
    parser.add_argument("--comparison", type=Path, default=PROJECT_DIR / "outputs" / "h3_low_g_drag" / "powered_coast_comparison.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs" / "h3_low_g_drag")
    args = parser.parse_args()

    config = load_model_config(PROJECT_DIR / "configs" / "aim120a_h3_low_g_drag.yaml")
    input_payload = json.loads(args.input.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    comparison = json.loads(args.comparison.read_text(encoding="utf-8")) if args.comparison.exists() else {}
    rows = _finite_rows(input_payload.get("samples", []))
    modulo = int(config["low_g_drag"]["fit"]["holdout_modulo"])
    train, holdout = _split_rows(rows, modulo)
    levels = {
        level: _fit_level(rows, train, holdout, level, config)
        for level in ("LG-0", "LG-1", "LG-2")
    }

    comparison_body = comparison.get("comparison", comparison)
    overlap_count = int(comparison_body.get("overlap_bin_count", 0))
    has_reference_time_series = bool(manifest.get("data_readiness", {}).get("powered_coast_overlap_is_reference_evidence", False))
    if not has_reference_time_series:
        selected_model = "LG-0"
        reason = "输入只有本地 H2 管道样本，没有独立 StatShark 动力/滑行时间序列；复杂度提升不可辨识。"
    elif overlap_count < int(config["low_g_drag"]["fit"]["minimum_overlap_bins"]):
        selected_model = "LG-0"
        reason = "动力/滑行重叠分箱不足，不能检验动力段修正。"
    else:
        holdout_scores = {
            level: result["holdout_metrics"]["rmse_m2"]
            for level, result in levels.items()
            if result["holdout_metrics"].get("rmse_m2") is not None
        }
        best = min(holdout_scores, key=holdout_scores.get) if holdout_scores else "LG-0"
        baseline = float(holdout_scores.get("LG-0", float("inf")))
        improved = baseline > 0.0 and float(holdout_scores.get(best, baseline)) < baseline * (1.0 - float(config["low_g_drag"]["fit"]["complexity_improvement_fraction"]))
        selected_model = best if improved else "LG-0"
        reason = "只有留出误差显著改善且存在独立重叠数据时才允许提升复杂度。"

    base_settings = _settings_from_config(config)
    threshold_sensitivity = _threshold_variants(input_payload.get("samples", []), config, base_settings)
    generated_at = datetime.now(timezone.utc).isoformat()
    lg0_report = dict(levels["LG-0"])
    lg0_report.update({
        "schema_version": 3,
        "generated_at_utc": generated_at,
        "model_label": config["model_label"],
        "source_kind": input_payload.get("source_kind", "unknown"),
        "statshark_new_calculation_performed_this_run": False,
        "accepted_sample_count": len(rows),
        "powered_sample_count": sum(1 for row in rows if row.get("powered")),
        "coast_sample_count": sum(1 for row in rows if not row.get("powered")),
        "mach_range": _mach_range(rows),
        "identifiability_boundary": (
            "The fit identifies an effective CdA0(M) plus a constant small-AoA term under a frozen sparse shape. "
            "It does not uniquely identify reference area, Cx, or true powered thrust correction."
        ),
    })
    selection_report = {
        "schema_version": 3,
        "generated_at_utc": generated_at,
        "model_label": config["model_label"],
        "source_kind": input_payload.get("source_kind", "unknown"),
        "statshark_new_calculation_performed_this_run": False,
        "accepted_sample_count": len(rows),
        "train_count": len(train),
        "holdout_count": len(holdout),
        "candidate_models": levels,
        "selected_model": selected_model,
        "reason": reason,
        "powered_coast_overlap_bin_count": overlap_count,
        "threshold_sensitivity": threshold_sensitivity,
        "complexity_policy": "promote LG-1/LG-2 only with independent overlap evidence and improved held-out error",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "lg0_fit_report.json").write_text(json.dumps(_json_safe(lg0_report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "model_selection_report.json").write_text(json.dumps(_json_safe(selection_report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_report(
        PROJECT_DIR / ".md" / "H3_LOW_G_DRAG_REPORT.md",
        generated_at,
        config,
        manifest,
        comparison,
        lg0_report,
        {"selected_model": selected_model, "reason": reason},
        rows,
    )
    print(
        f"LG0 train_rmse={levels['LG-0']['train_metrics']['rmse_m2']} "
        f"holdout_rmse={levels['LG-0']['holdout_metrics']['rmse_m2']} selected={selected_model}"
    )
    print(f"written: {args.output_dir / 'lg0_fit_report.json'}")
    print(f"written: {args.output_dir / 'model_selection_report.json'}")
    print(f"written: {PROJECT_DIR / '.md' / 'H3_LOW_G_DRAG_REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
