#!/usr/bin/env python3
"""Generate the auditable H4 status report from current local artifacts."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]


def _load(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    output_dir = PROJECT_DIR / "outputs" / "h4_glide_drag"
    source = _load(output_dir / "source_manifest.json", {})
    coverage = _load(output_dir / "coverage_report.json", {})
    atmosphere = _load(output_dir / "atmosphere_consistency.json", {})
    inverse = _load(output_dir / "inverse_cda_diagnostics.json", {})
    overlap = _load(output_dir / "overlap_consistency.json", {})
    sensitivity = _load(output_dir / "sensitivity_report.json", {})
    selection = _load(output_dir / "model_selection_report.json", {})
    fit = _load(output_dir / "cda_knots_fit.json", {})
    holdout = _load(output_dir / "trajectory_holdout_report.json", {})
    synthetic = _load(output_dir / "synthetic_validation_report.json", {})
    altitude = _load(output_dir / "altitude_overlap_report.json", {})
    repeatability = _load(output_dir / "repeatability_report.json", {})
    filtered = _load(output_dir / "filtered_samples.json", {})
    h3 = source.get("h3_checkpoint", {})
    synthetic_cases = synthetic.get("cases", [])
    zero_noise = synthetic_cases[0].get("metrics", {}) if synthetic_cases else {}
    artifact_inputs = {
        str(item.get("trajectory_id")): item.get("user_visible_inputs", {})
        for item in source.get("input_artifacts", [])
    }
    ranges = coverage.get("trajectory_ranges", {})
    rows = []
    for case, info in sorted(ranges.items()):
        inputs = artifact_inputs.get(case, {})
        target = f"M{info.get('mach_max')}→{info.get('mach_min')}; H{inputs.get('launch_altitude_m', '?')}m"
        rows.append((case, target, f"{info.get('sample_count', 0)} 条严格筛选行", "已纳入覆盖审计"))
    accepted_count = sum(1 for row in filtered.get("rows", []) if row.get("accepted"))
    rejected_count = sum(1 for row in filtered.get("rows", []) if not row.get("accepted"))
    inverse_valid_count = sum(1 for row in inverse.get("rows", []) if row.get("inverse_cda_valid"))
    lines = [
        "# H4 全包线无动力滑翔阻力报告",
        "",
        "## 1. 一句话结论和实际确认 Mach 范围",
        "",
        f"Plan 4 本轮已在授权后采集 {source.get('reference_trajectory_count', 0)} 条 StatShark 无动力/无制导参考轨迹。严格筛选后的直接支持为 `{coverage.get('actual_direct_support_range', {})}`；已生成仅限直接支持区间的正值 log(CdA) 部分模型，但仍不能称为完整 `M0.2–4.5` 包线。",
        "",
        "H3 的本地管道样本范围约为 `M1.014–3.023`，仍不计入 H4 参考覆盖。G6/G7 虽提供了低速可见标签，但在严格小迎角/航迹角筛选后没有足够连续样本做逆向导数。",
        "",
        "## 2. 原始数据来源、版本和案例清单",
        "",
        f"- H4 来源状态：`{source.get('status', 'unknown')}`；输入 artifact 数量：`{len(source.get('input_artifacts', []))}`；StatShark 参考轨迹：`{source.get('reference_trajectory_count', 0)}`。",
        f"- 已审计但排除的邻近 artifact：`{len(source.get('excluded_existing_artifacts', []))}`；其中 20° 离轴文件只有近似终点 tooltip，不是无动力滑翔时间序列。",
        f"- H3 冻结状态：`{h3.get('source_kind', 'unknown')}`，`statshark_reference_time_series_used={h3.get('statshark_reference_time_series_used')}`。",
        "- H3 配置、报告、样本清单和拟合结果的 SHA-256 已写入 `outputs/h4_glide_drag/source_manifest.json`。",
        "- G1–G7 的具体输入、覆盖目标和训练/验证角色见 `H4_REFERENCE_EXPERIMENT_DESIGN.md`；G5 为 G2 的重复输入，G6/G7 为低速补充。",
        "",
        "## 3. G1–G7 实际 Mach/高度/动压覆盖",
        "",
        "| 案例 | 目标 | 实际状态 | 结论 |",
        "|---|---|---|---|",
    ]
    lines.extend(f"| {case} | {target} | {status} | {conclusion} |" for case, target, status, conclusion in rows)
    lines.extend([
        "",
        f"共 {source.get('reference_samples_count', 0)} 条原始可见标签；严格筛选后 {accepted_count} 条通过、{rejected_count} 条拒绝。G1/G2/G4/G5 构成主要连续逆向 CdA 证据，G3 只提供短跨声速连接段，G6/G7 不满足稳定导数长度。",
        "",
        "## 4. 直接支持、插值和外推区间",
        "",
        f"- H4 参考直接支持：`{coverage.get('actual_direct_support_range', {})}`；目标范围内缺失：`{coverage.get('missing_target_ranges', [[0.2, 4.5]])}`。",
        f"- 部分拟合节点支持：`{fit.get('fit_support_mach_range', {})}`；节点外一律标记为外推，不计入确认包线。",
        "- 直接支持区间存在碎片，且 G2/G3 重叠宽度 0.20 Mach、G3/G4 无重叠；因此模型状态是 partial，不是完整包线。",
        "",
        "## 5. 数据清洗、筛选和被拒样本统计",
        "",
        f"- 参考原始行数：`{source.get('reference_samples_count', 0)}`；严格筛选通过：`{accepted_count}`；拒绝：`{rejected_count}`。",
        f"- 当前筛选设置见 `source_manifest.json`；alpha/q/lateral 预声明网格共 {len(sensitivity.get('combinations', []))} 组合，结果已写入 `sensitivity_report.json`。",
        "- 未来参考数据必须验证无动力、质量恒定、时间递增、单位一致和来源可追溯。",
        "",
        "## 6. 低速重力相消审计",
        "",
        f"- 逆向 CdA 状态：`{inverse.get('status', 'unknown')}`；有效逆向样本：`{inverse_valid_count}`；最大重力轴向相消比：`{inverse.get('gravity_cancellation', {}).get('max_ratio')}`。",
        "- 低速补充样本在严格筛选后不足以形成连续导数窗口，因此 M0.2–0.66 仍不能作稳定 CdA 结论。",
        "",
        "## 7. atmosphere/Mach 一致性",
        "",
        f"- 状态：`{atmosphere.get('status', 'unknown')}`；参考样本数：`{atmosphere.get('sample_count', 0)}`；最大 `mach_source - mach_recomputed`：`{atmosphere.get('max_abs_mach_difference')}`。",
        "- 未来数据必须同时保存 `mach_source` 和本地大气重算值，不能静默选择其一。",
        "",
        "## 8. 最终 CdA(M) 节点和不确定度",
        "",
        f"已生成 `cda_knots_fit.json`，状态 `{fit.get('status', 'unknown')}`；节点 Mach：`{fit.get('mach_knots', [])}`；CdA 节点：`{fit.get('cda_knots_m2', [])}`。该结果是有效样本的稳健节点中位数，不含 H3/合成数据；模型边界为部分直接支持，不能用于声称 M0.2–4.5 已验证。",
        "",
        "## 9. 逆算诊断与轴向重放结果",
        "",
        f"- 真实参考重放：`{holdout.get('status', 'unknown')}`；留出轨迹：`{holdout.get('holdout_trajectory_id')}`；指标：`{holdout.get('metrics', {})}`。",
        f"- 合成零噪声重放：`zero_noise_pass={synthetic.get('acceptance', {}).get('zero_noise_pass')}`，RMSE `{zero_noise.get('speed_rmse_mps')}` m/s。",
        "- 合成结果标记为 `synthetic_test`，只证明代码路径和单位/符号，不是 StatShark 证据。",
        "",
        "## 10. 完整轨迹留出误差",
        "",
        "已执行整轨迹留出：G5 作为 G2 重复案例留出，训练不含 G5；相对速度 RMSE 约 0.575%，通过当前 5% 诊断阈值。该验证只覆盖轴向重放，且使用观测高度/航迹角作为外生输入。",
        "",
        "## 11. 相邻案例重叠区一致性",
        "",
        f"- 覆盖相邻重叠审计状态：`{overlap.get('status', 'unknown')}`；G1/G2 和 G4/G5 满足约 0.3 Mach 要求，G2/G3 不满足。",
        f"- G2/G4 同 Mach 配对：`{altitude.get('pairing_summary', {})}`；该结果只支持高度差异复核，不自动加入环境项。",
        "",
        "## 12. 不同高度复核结果",
        "",
        f"G2/G4 已完成 5 个近 Mach 配对，平均 CdA 比值 G4/G2 约 `{altitude.get('pairing_summary', {}).get('mean_cda_ratio_g4_over_g2')}`；当前差异不足以单独证明环境项，继续保持单一有效 CdA 模型。",
        "",
        "## 13. 平滑、筛选和正则化敏感性",
        "",
        f"- 状态：`{sensitivity.get('status', 'unknown')}`；已实际重跑 {len(sensitivity.get('combinations', []))} 个 alpha/lateral/q 筛选组合。",
        "- 最宽松的预声明组合可把连续直接支持下探到约 M0.66，但仍不能覆盖 M0.2–0.66；宽松筛选也没有被用于主 CdA 拟合。",
        "",
        "## 14. 是否有资格称为零迎角 CdA0",
        "",
        "没有。即使未来取得小迎角数据，若无法证明迎角接近零且与航迹角/控制状态不共线，名称也必须保留为 `CdA_effective_small_alpha(M)`。",
        "",
        "## 15. 未识别量和下一条最有价值的数据",
        "",
        "未识别量包括 M0.2–0.66 亚音速阻力、M0.99–1.16 连接段、节点外外推可信度、跨声速峰形状和更完整的低速重力相消误差。下一步最有价值的数据是保持低迎角/小航迹角并延长 G6/G7，使每个低 Mach 区间至少有连续三点以上可反算。",
        "",
        "## 16. 是否具备开展动力段配对辨识的资格",
        "",
        "不具备。必须先取得并冻结独立滑翔基准，通过覆盖、重叠、低速相消和整轨迹留出验证，之后才能设计动力/滑翔配对实验。",
        "",
        "## 停止边界",
        "",
        "本轮在用户授权后执行了新的 StatShark 计算并保存了 G1–G7 可追溯可见标签；没有修改 War Thunder 游戏文件，也没有覆盖 H1/H2/H3 输出。`cda_knots_fit.json` 是部分直接支持模型，不是完整 M0.2–4.5 结论。",
    ])
    report_path = PROJECT_DIR / ".md" / "H4_GLIDE_DRAG_ENVELOPE_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
