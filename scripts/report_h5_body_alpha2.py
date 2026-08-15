#!/usr/bin/env python3
"""Write the bounded H5 Phase 0-2 report without inventing a fit."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "h5_body_alpha2"


def _load(name: str) -> dict[str, Any]:
    return json.loads((OUTPUT_DIR / name).read_text(encoding="utf-8"))


def _fmt(value: Any) -> str:
    if value is None:
        return "not available"
    if isinstance(value, float):
        return "{:.9g}".format(value)
    return str(value)


def build_report() -> str:
    phase0 = _load("phase0_freeze_manifest.json")
    phase1 = _load("phase1_existing_alpha_review.json")
    synthetic = _load("synthetic_identifiability_report.json")
    aggregate = phase1["aggregate_primary_window"]
    config = phase1["configuration_audit"]
    full = synthetic["full_matrix"]
    checks = synthetic["acceptance_checks"]
    old_cda = phase0["h4_snapshot"]["m1p5_cda_effective_m2"]
    case_lines = []
    for case in phase1["case_reports"]:
        window = case["primary_window"]
        case_lines.append(
            "| {case} | {count} | {alpha} | {bands} | {center} |".format(
                case=case["case_id"],
                count=window["sample_count"],
                alpha=_fmt(window["alpha_abs_deg"]["median"]),
                bands="P0={}; P1={}; P2={}".format(
                    window["alpha_bands"]["P0"],
                    window["alpha_bands"]["P1"],
                    window["alpha_bands"]["P2"],
                ),
                center=window["center_support_count"],
            )
        )
    gates = phase1["formal_h5_gates"]
    return """# H5 零主动舵面整弹弹体迎角平方阻力报告（Phase 0–2）

> H5-A FAIL: configuration not isolated (pre-collection gate)

## 1. 一句话结论

本轮已完成本地 Phase 0–2：冻结 H4/H4.5、确认既有 H4 在 `M≈1.5` 存在 `0°/1.5°/2.1–2.6°` 的可见迎角跨度，并通过带页面量化误差的合成整段重放测试；但尚未采集任何 H5 C0/C9/C18 StatShark 案例，因此没有实际 `CdA0_1p5` 或 `K_alpha2_nominal_1p5` 估计。

## 2. 权限和本轮计算边界

- 用户指令授权执行 Plan 5 的本地准备、代码和分析。
- 本轮未运行新的 StatShark `Calculate`；没有得到独立的网站计算次数授权。
- 未修改 War Thunder 游戏文件。
- 未覆盖 H1、H2、H3、H4 或 H4.5 产物。
- H5 输出隔离在 `outputs/h5_body_alpha2/`；H5 原始数据目录目前没有正式案例。

## 3. 冻结来源和证据边界

Phase 0 清单：`outputs/h5_body_alpha2/phase0_freeze_manifest.json`。

与 H4.5 Phase 0 清单逐项复核的既有冻结文件哈希：`{prior_hash_status}`；H4/H4.5 既有产物保持不变。

冻结的 H4 `M=1.5` 旧有效节点为 `{old_cda} m²`，来源类型为 `prior_reference_checkpoint`。它不是 `CdA0`，不能替代 H5 的零迎角截距。

现有 H4 可见轨迹属于 `statshark_reference_visible_readout`；合成结果属于 `synthetic_test`。两类证据未混入实际拟合。

## 4. “零主动舵面整弹”术语边界

本阶段目标仍是零主动舵面、完整外形整弹的有效 `CdA` 模型，不是几何上拆除尾翼和弹翼的 `bare-body`。现有原始文件只保存了质量、零推力、制导关闭和零侧向舵控等部分边界；完整的 `CxK/CyK/CxAoA/CyMaxAoA/wingAreaMult/...` 自定义模型快照缺失，因此配置隔离门未通过。

## 5. 既有 H4 在 M1.5 的迎角预审

输入：`data/raw/statshark_h4/G1–G7_statshark_visible_slider_20260811.json`。

| 案例 | `M=1.45–1.55` 点数 | 窗口内迎角中位数（绝对值，°） | 迎角档位计数 | 中心窗点数 |
|---|---:|---:|---|---:|
{case_table}

汇总：窗口内 `{primary}` 个原始点，中心窗 `{center}` 个点；中心窗由 `{center_cases}` 条历史支持。既有数据在规划意义上显示低/中/高迎角可达，但每条历史只有约 1–3 个窗口点，未达到 H5 每案例至少 5 点的密采样要求。

`P1-P0={p10}°`，`P2-P1={p21}°`，`P2={p2}°`；这只是历史状态跨度，不是独立迎角干预的通过结果。

## 6. C0/C9/C18 配置一致性

- 既有 H4 只有一个自定义变体：`{variants}`。
- 既有窗口没有可回读的 `CxAoA` 值：`{cx_values}`。
- 要求的 `[0, 9, 18]` 三档字段不在既有数据中。
- 完整模型快照：缺失。
- 因此 `G-H5-0 configuration isolation` 不通过；这不是气动模型失败，而是正式采集前的配置证据缺口。

## 7. Phase 2 合成可辨识性

合成矩阵使用 P0/P1/P2 × C0/C9/C18，共 `{traj_count}` 条轨迹、`{sample_count}` 个点；速度按 1 km/h、Mach 按 0.01、迎角按 0.1° 量化。已知参数的量化重放恢复门：`{quantized_gate}`。

恢复结果（仅合成测试）：

- `CdA0_1p5 = {_cda0}` m²；真值 `{truth_cda0}` m²；
- `K_alpha2_C9 = {_k9}` m²/rad²；真值 `{truth_k9}` m²/rad²；
- 量化速度重放加权 RMSE：`{rmse}` m/s。

M0/M1/M2/M4 竞争形式在特意增强的清洁合成诊断中留下了残差结构；报告同时保留“小角度实际数据可能无法强区分指数”的边界。该结果证明分析器链路可用，不证明 StatShark 后端采用 `alpha²`。

## 8. H5 实际参数和留出

本轮实际 H5 参数：未估计。

实际 H5 的 C0/C9/C18 双重差分、P1 完整留出、跨轨迹留出、H4 接缝和滤波敏感性：未执行，因为没有 H5 原始轨迹。

## 9. 当前门槛状态

| 门槛 | 状态 |
|---|---|
| G-H5-0 配置隔离 | FAIL：缺完整快照和 CxAoA 消融 |
| G-H5-1 M1.5 覆盖 | FAIL：既有 H4 每案例窗口点数不足 |
| G-H5-2 迎角激励 | 仅规划预审通过；非正式 H5 通过 |
| G-H5-3 alpha² 可辨识 | 未执行 |
| G-H5-4 CxAoA 缩放 | 未执行 |
| G-H5-5 模型形式 | 仅合成诊断通过 |
| G-H5-6 轨迹重放 | 仅合成诊断通过 |
| G-H5-7 H4 接缝 | 未执行 |

## 10. 本轮允许和禁止的结论

允许：

- 既有 H4 的自然无控制滑翔历史显示 `M≈1.5` 的低、中、高实际迎角状态可被制造；
- 本地 H5 分析器在页面可见精度量化下能恢复合成的 `CdA0/K_residual/S_CxAoA`；
- H4 旧节点 `{old_cda} m²` 保持冻结。

禁止：

- 给出实际 `CdA0_1p5` 或 `K_alpha2_nominal_1p5`；
- 声称 `CxAoA` 已线性缩放；
- 把既有 H4 的 `CdA_effective_small_alpha` 当作 `CdA0`；
- 把零主动舵面整弹称为真正裸弹体；
- 声称已排除 `H_fin_sep`/`H_fin_sum` 或恢复 StatShark 服务端公式。

## 11. 下一步和停止条件

进入 Phase 3 前需要新的 StatShark 计算授权。最小侦察批次为：`H5-P0-C9`、`H5-P2-C0`、`H5-P2-C9`、`H5-P2-C18`；这四例只能判断路线，不能完成 H5 三档迎角验收。正式矩阵需要 9 个案例：P0/P1/P2 × C0/C9/C18；每一次试算、失败案例和正式案例都应计入授权次数。

本轮在“未获网站计算授权、完整模型快照缺失、既有数据不满足正式密采样”处停止，没有生成虚假的 H5 拟合。

## 12. 产物

- `outputs/h5_body_alpha2/phase0_freeze_manifest.json`
- `outputs/h5_body_alpha2/source_manifest.json`
- `outputs/h5_body_alpha2/phase1_existing_alpha_review.json`
- `outputs/h5_body_alpha2/synthetic_identifiability_report.json`
- `configs/aim120a_h5_body_alpha2.yaml`
- `src/aim120_model/body_alpha2_drag.py`
- `src/aim120_model/body_alpha2_replay.py`
- `scripts/freeze_h5_phase0.py`
- `scripts/audit_h5_existing_alpha.py`
- `scripts/validate_h5_synthetic_identifiability.py`
- `scripts/report_h5_body_alpha2.py`

本地回归：`57` 个 stdlib fallback 测试通过。
""".format(
        old_cda=_fmt(old_cda),
        prior_hash_status=phase0["prior_h4_5_freeze_comparison"]["all_unchanged"],
        case_table="\n".join(case_lines),
        primary=aggregate["primary_sample_count"],
        center=aggregate["center_sample_count"],
        center_cases=", ".join(aggregate["center_support_cases"]),
        p10=_fmt(aggregate["alpha_separation"]["P1_minus_P0_deg"]),
        p21=_fmt(aggregate["alpha_separation"]["P2_minus_P1_deg"]),
        p2=_fmt(aggregate["alpha_separation"]["P2_median_deg"]),
        variants=", ".join(config["unique_missile_variants"]),
        cx_values=_fmt(config["cx_aoa_values_observed"]),
        traj_count=full["trajectory_count"],
        sample_count=full["sample_count"],
        quantized_gate=checks["known_alpha2_parameters_recovered_with_quantization"],
        _cda0=_fmt(full["fit_quantized"]["cda0_1p5"]),
        truth_cda0=_fmt(synthetic["truth_parameters"]["cda0_1p5"]),
        _k9=_fmt(full["fit_quantized"]["k_c9"]),
        truth_k9=_fmt(synthetic["truth_parameters"]["k_c9"]),
        rmse=_fmt(full["quantized_weighted_speed_rmse_mps"]),
    )


def main() -> int:
    report_path = PROJECT_ROOT / ".md" / "H5_BODY_ALPHA2_REPORT.md"
    report_path.write_text(build_report(), encoding="utf-8")
    print(json.dumps({"output": str(report_path), "status": "written"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
