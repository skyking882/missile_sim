"""Exploratory H4.5 transonic fit after the strict coverage gate failed.

The original Phase-3 blocked outputs are intentionally left unchanged.  This
script fits low-dimensional models to the saved visible-label trajectories and
writes a separately labelled provisional analysis.  It uses whole-trajectory
forward replay, not pointwise inverse-CdA regression.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from aim120_model.atmosphere import StandardAtmosphere
from aim120_model.axial_replay import replay_trajectory
from aim120_model.glide_drag_envelope import LogCdaEnvelope


RAW_H45 = PROJECT_DIR / "data" / "raw" / "statshark_h4_5"
RAW_H4 = PROJECT_DIR / "data" / "raw" / "statshark_h4"
H4_FIT_PATH = PROJECT_DIR / "outputs" / "h4_glide_drag" / "cda_knots_fit.json"
RM10_PATH = (
    PROJECT_DIR
    / "data"
    / "reference_external"
    / "rm10"
    / "rm10_figure13_composite_digitization.csv"
)
OUTPUT_DIR = PROJECT_DIR / "outputs" / "h4_5_transonic_drag" / "provisional_fit"
MASS_KG = 147.87
JOIN_MACH = 1.2


def _state_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(name) for name in ("speed_mps", "mach", "alpha_rad", "altitude_m", "target_distance_m"))


def _load_unique(path: Path, case_id: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    seen: set[tuple[Any, ...]] = set()
    rows: list[dict[str, Any]] = []
    for raw in payload["result"]["samples"]:
        key = _state_key(raw)
        if key in seen:
            continue
        seen.add(key)
        row = dict(raw)
        row.update(
            {
                "case_id": case_id,
                "trajectory_id": case_id,
                "source_kind": "statshark_reference",
                "mass_kg": float(row.get("mass_kg", MASS_KG)),
                "alpha_deg": math.degrees(float(row.get("alpha_rad", 0.0))),
            }
        )
        rows.append(row)
    return sorted(rows, key=lambda item: float(item["time_s"]))


def _dynamic_pressure(row: Mapping[str, Any], atmosphere: StandardAtmosphere) -> float:
    atm = atmosphere.sample(float(row["altitude_m"]))
    speed = float(row["speed_mps"])
    return 0.5 * atm.density_kg_m3 * speed * speed


def select_segment(
    rows: Sequence[Mapping[str, Any]],
    alpha_limit_deg: float,
    mach_min: float = 0.90,
    mach_max: float = 1.25,
    gamma_limit_deg: float = 5.0,
    q_min_pa: float = 1000.0,
) -> list[dict[str, Any]]:
    atmosphere = StandardAtmosphere()
    selected: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        if not (mach_min <= float(row["mach"]) <= mach_max):
            continue
        if abs(float(row["alpha_deg"])) > alpha_limit_deg + 1.0e-9:
            continue
        if abs(float(row["flight_path_angle_deg"])) > gamma_limit_deg + 1.0e-9:
            continue
        q = _dynamic_pressure(row, atmosphere)
        if q < q_min_pa:
            continue
        row["dynamic_pressure_pa"] = q
        selected.append(row)
    return selected


@dataclass(frozen=True)
class CompositeEnvelope:
    h4: LogCdaEnvelope
    below_join: Callable[[float], float]

    def cda_m2(self, mach: float) -> float:
        value = float(mach)
        if value >= JOIN_MACH:
            return self.h4.cda_m2(value)
        cda = float(self.below_join(value))
        if not math.isfinite(cda) or cda <= 0.0:
            raise ValueError("provisional CdA must remain positive")
        return cda


def load_inputs() -> tuple[dict[str, list[dict[str, Any]]], LogCdaEnvelope, np.ndarray, np.ndarray]:
    trajectories = {
        case_id: _load_unique(
            RAW_H45 / f"{case_id}_statshark_visible_slider_20260811.json", case_id
        )
        for case_id in ("T1", "T2", "T3", "T4")
    }
    trajectories["G3"] = _load_unique(
        RAW_H4 / "G3_statshark_visible_slider_20260811.json", "G3"
    )
    h4_payload = json.loads(H4_FIT_PATH.read_text(encoding="utf-8"))
    h4 = LogCdaEnvelope.from_cda_knots(
        h4_payload["mach_knots"], h4_payload["cda_knots_m2"]
    )
    rm_rows = list(csv.DictReader(RM10_PATH.open("r", encoding="utf-8", newline="")))
    rm_mach = np.array([float(row["mach"]) for row in rm_rows], dtype=float)
    rm_cd = np.array([float(row["cd_total"]) for row in rm_rows], dtype=float)
    return trajectories, h4, rm_mach, rm_cd


def make_rm_model(
    h4: LogCdaEnvelope,
    rm_mach: np.ndarray,
    rm_cd: np.ndarray,
    beta: float,
    slope: float = 0.0,
) -> CompositeEnvelope:
    join_cda = h4.cda_m2(JOIN_MACH)
    rm_join = float(np.interp(JOIN_MACH, rm_mach, rm_cd))

    def below(mach: float) -> float:
        ratio = float(np.interp(mach, rm_mach, rm_cd)) / rm_join
        return join_cda * ratio ** float(beta) * math.exp(float(slope) * (JOIN_MACH - mach))

    return CompositeEnvelope(h4=h4, below_join=below)


def make_k2_model(h4: LogCdaEnvelope, cda_m09: float, cda_m105: float) -> CompositeEnvelope:
    mach_knots = np.array([0.90, 1.05, 1.20], dtype=float)
    cda_knots = np.array([cda_m09, cda_m105, h4.cda_m2(JOIN_MACH)], dtype=float)

    def below(mach: float) -> float:
        return float(np.exp(np.interp(mach, mach_knots, np.log(cda_knots))))

    return CompositeEnvelope(h4=h4, below_join=below)


def replay_metrics(
    envelope: CompositeEnvelope,
    segments: Mapping[str, Sequence[Mapping[str, Any]]],
    max_step_s: float = 0.10,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    metrics: dict[str, dict[str, Any]] = {}
    replays: dict[str, list[dict[str, Any]]] = {}
    for case_id, rows in segments.items():
        if len(rows) < 2:
            metrics[case_id] = {
                "sample_count": len(rows),
                "speed_rmse_mps": None,
                "speed_relative_rmse": None,
                "terminal_speed_error_mps": None,
                "status": "insufficient_segment",
            }
            replays[case_id] = []
            continue
        replay = replay_trajectory(rows, envelope, max_step_s=max_step_s)
        # The first replay point is an imposed initial condition with exactly
        # zero residual.  Exclude it so short 2--5 point segments do not receive
        # an artificially optimistic RMSE.
        evaluated = replay[1:]
        residuals = [float(row["speed_residual_mps"]) for row in evaluated]
        observed = [float(row["observed_speed_mps"]) for row in evaluated]
        rmse = math.sqrt(sum(value * value for value in residuals) / len(residuals))
        relative_scale = max(sum(abs(value) for value in observed) / len(observed), 1.0e-9)
        item = {
            "sample_count": len(evaluated),
            "speed_rmse_mps": rmse,
            "speed_relative_rmse": rmse / relative_scale,
            "terminal_speed_error_mps": residuals[-1],
            "initial_condition_excluded_from_metrics": True,
        }
        item["status"] = "replayed"
        item["mach_min_observed"] = min(float(row["mach"]) for row in rows)
        item["mach_max_observed"] = max(float(row["mach"]) for row in rows)
        metrics[case_id] = item
        replays[case_id] = replay
    return metrics, replays


def training_loss(
    envelope: CompositeEnvelope,
    training_segments: Mapping[str, Sequence[Mapping[str, Any]]],
) -> float:
    metrics, _ = replay_metrics(envelope, training_segments)
    values = [
        float(item["speed_relative_rmse"]) ** 2
        for item in metrics.values()
        if item.get("speed_relative_rmse") is not None
    ]
    return float(np.mean(values)) if values else float("inf")


def _search_1d(
    builder: Callable[[float], CompositeEnvelope],
    segments: Mapping[str, Sequence[Mapping[str, Any]]],
    lower: float,
    upper: float,
) -> tuple[float, float]:
    best_x, best_loss = float("nan"), float("inf")
    lo, hi = lower, upper
    for points in (91, 61, 61):
        grid = np.linspace(lo, hi, points)
        losses = [training_loss(builder(float(value)), segments) for value in grid]
        index = int(np.argmin(losses))
        best_x, best_loss = float(grid[index]), float(losses[index])
        step = float(grid[1] - grid[0])
        lo, hi = best_x - 3.0 * step, best_x + 3.0 * step
    return best_x, best_loss


def _search_2d(
    builder: Callable[[float, float], CompositeEnvelope],
    segments: Mapping[str, Sequence[Mapping[str, Any]]],
    first_range: tuple[float, float],
    second_range: tuple[float, float],
) -> tuple[float, float, float]:
    a_lo, a_hi = first_range
    b_lo, b_hi = second_range
    best_a = best_b = float("nan")
    best_loss = float("inf")
    for points in (17, 13, 13):
        a_values = np.linspace(a_lo, a_hi, points)
        b_values = np.linspace(b_lo, b_hi, points)
        for a in a_values:
            for b in b_values:
                loss = training_loss(builder(float(a), float(b)), segments)
                if loss < best_loss:
                    best_a, best_b, best_loss = float(a), float(b), float(loss)
        a_step = float(a_values[1] - a_values[0])
        b_step = float(b_values[1] - b_values[0])
        a_lo, a_hi = best_a - 2.5 * a_step, best_a + 2.5 * a_step
        b_lo, b_hi = best_b - 2.5 * b_step, best_b + 2.5 * b_step
    return best_a, best_b, best_loss


def fit_suite(
    alpha_limit_deg: float,
    trajectories: Mapping[str, list[dict[str, Any]]],
    h4: LogCdaEnvelope,
    rm_mach: np.ndarray,
    rm_cd: np.ndarray,
) -> dict[str, Any]:
    segments = {
        case_id: select_segment(rows, alpha_limit_deg=alpha_limit_deg)
        for case_id, rows in trajectories.items()
    }
    training = {case_id: segments[case_id] for case_id in ("T1", "T2")}
    model_builders: dict[str, Callable[..., CompositeEnvelope]] = {
        "P0_beta_1": lambda: make_rm_model(h4, rm_mach, rm_cd, 1.0),
        "P0b_beta_1p669": lambda: make_rm_model(h4, rm_mach, rm_cd, 1.668993602463756),
    }

    beta, _ = _search_1d(
        lambda value: make_rm_model(h4, rm_mach, rm_cd, value), training, -1.0, 5.0
    )
    beta_p2, slope_p2, _ = _search_2d(
        lambda first, second: make_rm_model(h4, rm_mach, rm_cd, first, second),
        training,
        (-1.0, 5.0),
        (-5.0, 5.0),
    )
    log_cda09, log_cda105, _ = _search_2d(
        lambda first, second: make_k2_model(h4, math.exp(first), math.exp(second)),
        training,
        (math.log(0.006), math.log(0.035)),
        (math.log(0.012), math.log(0.040)),
    )
    fitted = {
        "P0_beta_1": {
            "complexity": 0,
            "parameters": {"beta": 1.0, "slope": 0.0},
            "envelope": model_builders["P0_beta_1"](),
        },
        "P0b_beta_1p669": {
            "complexity": 0,
            "parameters": {"beta": 1.668993602463756, "slope": 0.0},
            "envelope": model_builders["P0b_beta_1p669"](),
        },
        "P1_beta_free": {
            "complexity": 1,
            "parameters": {"beta": beta, "slope": 0.0},
            "envelope": make_rm_model(h4, rm_mach, rm_cd, beta),
        },
        "P2_beta_slope": {
            "complexity": 2,
            "parameters": {"beta": beta_p2, "slope": slope_p2},
            "envelope": make_rm_model(h4, rm_mach, rm_cd, beta_p2, slope_p2),
        },
        "K2_log_knots": {
            "complexity": 2,
            "parameters": {
                "cda_m09_m2": math.exp(log_cda09),
                "cda_m105_m2": math.exp(log_cda105),
                "cda_m12_m2": h4.cda_m2(JOIN_MACH),
            },
            "envelope": make_k2_model(h4, math.exp(log_cda09), math.exp(log_cda105)),
        },
    }

    model_results: dict[str, Any] = {}
    for name, item in fitted.items():
        metrics, replays = replay_metrics(item["envelope"], segments, max_step_s=0.02)
        train_values = [
            float(metrics[case_id]["speed_relative_rmse"])
            for case_id in ("T1", "T2")
            if metrics[case_id].get("speed_relative_rmse") is not None
        ]
        independent_values = [
            float(metrics[case_id]["speed_relative_rmse"])
            for case_id in ("T4", "G3")
            if metrics[case_id].get("speed_relative_rmse") is not None
        ]
        repeat_value = metrics["T3"].get("speed_relative_rmse")
        model_results[name] = {
            "complexity": item["complexity"],
            "parameters": item["parameters"],
            "training_mean_relative_rmse": float(np.mean(train_values)) if train_values else None,
            "independent_validation_mean_relative_rmse": float(np.mean(independent_values)) if independent_values else None,
            "repeat_T3_relative_rmse": repeat_value,
            "metrics_by_trajectory": metrics,
            "replays": replays,
        }

    ranking = sorted(
        model_results,
        key=lambda name: (
            float(model_results[name]["independent_validation_mean_relative_rmse"])
            if model_results[name]["independent_validation_mean_relative_rmse"] is not None
            else float("inf"),
            model_results[name]["complexity"],
        ),
    )
    best_validation = ranking[0]
    p1_score = float(model_results["P1_beta_free"]["independent_validation_mean_relative_rmse"])
    best_score = float(model_results[best_validation]["independent_validation_mean_relative_rmse"])
    if best_validation != "P1_beta_free" and best_score <= 0.85 * p1_score:
        recommended = best_validation
        rationale = "The more complex model improves independent validation by at least 15% over P1."
    else:
        recommended = "P1_beta_free"
        rationale = "P1 is retained because added complexity does not clear the predeclared 15% validation-improvement gate."

    # Compact curve table for exact lookup and plotting.
    curve_mach = np.round(np.arange(0.90, 1.2001, 0.025), 6)
    curve_rows = []
    for mach in curve_mach:
        row: dict[str, Any] = {"mach": float(mach)}
        for name, item in fitted.items():
            row[name] = item["envelope"].cda_m2(float(mach))
        curve_rows.append(row)

    return {
        "alpha_limit_deg": alpha_limit_deg,
        "segments": {
            case_id: {
                "sample_count": len(rows),
                "mach_min": min((float(row["mach"]) for row in rows), default=None),
                "mach_max": max((float(row["mach"]) for row in rows), default=None),
                "time_min_s": min((float(row["time_s"]) for row in rows), default=None),
                "time_max_s": max((float(row["time_s"]) for row in rows), default=None),
            }
            for case_id, rows in segments.items()
        },
        "models": model_results,
        "ranking_by_independent_validation": ranking,
        "recommended_provisional_model": recommended,
        "selection_rationale": rationale,
        "curve_rows": curve_rows,
    }


def p1_leave_one_out(
    alpha_limit_deg: float,
    trajectories: Mapping[str, list[dict[str, Any]]],
    h4: LogCdaEnvelope,
    rm_mach: np.ndarray,
    rm_cd: np.ndarray,
) -> list[dict[str, Any]]:
    segments = {
        case_id: select_segment(trajectories[case_id], alpha_limit_deg=alpha_limit_deg)
        for case_id in ("T1", "T2", "T4")
    }
    output = []
    for held_out in ("T1", "T2", "T4"):
        training = {case_id: rows for case_id, rows in segments.items() if case_id != held_out}
        beta, _ = _search_1d(
            lambda value: make_rm_model(h4, rm_mach, rm_cd, value), training, -1.0, 5.0
        )
        envelope = make_rm_model(h4, rm_mach, rm_cd, beta)
        metrics, _ = replay_metrics(envelope, {held_out: segments[held_out]}, max_step_s=0.02)
        output.append(
            {
                "held_out": held_out,
                "training_cases": sorted(training),
                "beta": beta,
                "held_out_metrics": metrics[held_out],
            }
        )
    return output


def _strip_replays(payload: dict[str, Any]) -> dict[str, Any]:
    clean = json.loads(json.dumps(payload))
    for sensitivity in clean["sensitivity_fits"].values():
        for model in sensitivity["models"].values():
            model.pop("replays", None)
    return clean


def write_chart(payload: dict[str, Any]) -> None:
    primary = payload["sensitivity_fits"]["alpha_3p0"]
    model_names = ["P0_beta_1", "P0b_beta_1p669", "P1_beta_free", "P2_beta_slope", "K2_log_knots"]
    colors = {
        "P0_beta_1": "#D97706",
        "P0b_beta_1p669": "#9A6B00",
        "P1_beta_free": "#1565C0",
        "P2_beta_slope": "#7C3AED",
        "K2_log_knots": "#4B5563",
    }
    styles = {
        "P0_beta_1": ":",
        "P0b_beta_1p669": "--",
        "P1_beta_free": "-",
        "P2_beta_slope": "-.",
        "K2_log_knots": "--",
    }
    labels = {
        "P0_beta_1": "RM-10 prior, beta=1",
        "P0b_beta_1p669": "RM-10 prior, beta=1.669",
        "P1_beta_free": "P1 fitted beta",
        "P2_beta_slope": "P2 beta + slope",
        "K2_log_knots": "K2 free log knots",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4), constrained_layout=True)
    ax = axes[0]
    rows = primary["curve_rows"]
    mach = [row["mach"] for row in rows]
    for name in model_names:
        ax.plot(
            mach,
            [row[name] for row in rows],
            color=colors[name],
            ls=styles[name],
            lw=2.7 if name == "K2_log_knots" else (2.1 if name in ("P1_beta_free", "P2_beta_slope") else 1.8),
            label=labels[name],
        )
    ax.scatter([1.2], [payload["h4_join_cda_m2"]], color="#111827", s=50, zorder=4, label="Frozen H4 join")
    ax.set_title("Provisional transonic CdA model comparison\nAlpha <= 3 deg; visible labels are 2-4 s apart", fontsize=12)
    ax.set_xlabel("Mach")
    ax.set_ylabel("Effective CdA (m²)")
    ax.set_xlim(0.90, 1.205)
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    selected = primary["recommended_provisional_model"]
    replay_map = primary["models"][selected]["replays"]
    trajectory_colors = {"T1": "#1565C0", "T2": "#D97706", "T3": "#7C3AED", "T4": "#6B7280", "G3": "#111827"}
    trajectory_styles = {"T1": "-", "T2": "-", "T3": "--", "T4": "-.", "G3": ":"}
    for case_id, replay in replay_map.items():
        if not replay:
            continue
        ax.plot(
            [row["mach_pred"] for row in replay],
            [row["speed_residual_mps"] for row in replay],
            color=trajectory_colors[case_id],
            ls=trajectory_styles[case_id],
            marker="o",
            lw=1.7,
            ms=4,
            label=case_id,
        )
    ax.axhline(0.0, color="#374151", lw=1.2)
    ax.set_title(
        f"Whole-trajectory speed residuals: {selected}\n"
        "Observed altitude and path angle are exogenous inputs",
        fontsize=12,
    )
    ax.set_xlabel("Predicted Mach")
    ax.set_ylabel("Predicted - observed speed (m/s)")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.savefig(OUTPUT_DIR / "h4_5_provisional_fit.png", dpi=180)
    plt.close(fig)


def write_report(payload: dict[str, Any]) -> None:
    primary = payload["sensitivity_fits"]["alpha_3p0"]
    strict = payload["sensitivity_fits"]["alpha_2p5"]
    selected = primary["recommended_provisional_model"]
    model = primary["models"][selected]
    p1 = primary["models"]["P1_beta_free"]
    strict_k2 = strict["models"]["K2_log_knots"]["parameters"]
    primary_k2 = primary["models"]["K2_log_knots"]["parameters"]
    endpoint_shift = (
        strict_k2["cda_m09_m2"] / primary_k2["cda_m09_m2"] - 1.0
    )
    center_shift = (
        strict_k2["cda_m105_m2"] / primary_k2["cda_m105_m2"] - 1.0
    )
    strict_beta = strict["models"]["P1_beta_free"]["parameters"]["beta"]
    primary_beta = primary["models"]["P1_beta_free"]["parameters"]["beta"]
    report = f"""# H4.5 跨音速临时拟合报告

## 结论

在不推翻原 coverage gate 的前提下，已对保存的可见标签做一次探索性前向重放拟合。主探索筛选使用 `alpha <= 3°`，实际支持约为 `M0.91–1.25`；推荐的临时候选为 `{selected}`。

该结果只能称为 `provisional effective small-alpha CdA`，不能升级为 Plan 4.5 覆盖成功。可见状态间隔约为 2–4 s，T1/T3 还是确定性重复而非独立信息。

## 推荐临时模型

推荐模型参数：

```json
{json.dumps(model['parameters'], ensure_ascii=False, indent=2)}
```

即：在 `M0.90 / 1.05 / 1.20` 使用 `0.010604 / 0.022313 / 0.023566 m²`，节点间对 `log(CdA)` 做线性插值；`M>=1.2` 继续使用冻结 H4。

训练轨迹 T1/T2 平均相对速度 RMSE：`{100.0 * float(model['training_mean_relative_rmse']):.3f}%`。
独立诊断轨迹 T4/G3 平均相对速度 RMSE：`{100.0 * float(model['independent_validation_mean_relative_rmse']):.3f}%`。
T3 重复轨迹相对速度 RMSE：`{100.0 * float(model['repeat_T3_relative_rmse']):.3f}%`。

## 单参数 P1 对照

`alpha <= 3°` 时 P1 的参数为：

```json
{json.dumps(p1['parameters'], ensure_ascii=False, indent=2)}
```

P1 的 T1/T2 平均相对 RMSE 为 `{100.0 * float(p1['training_mean_relative_rmse']):.3f}%`，T4/G3 平均相对 RMSE 为 `{100.0 * float(p1['independent_validation_mean_relative_rmse']):.3f}%`。

模型选择理由：{primary['selection_rationale']}

## 迎角敏感性

- `alpha <= 2.5°` 推荐模型：`{strict['recommended_provisional_model']}`。
- `alpha <= 3.0°` 推荐模型：`{primary['recommended_provisional_model']}`。
- K2 的 `M1.05` 节点从 2.5° 到 3° 只变化 `{100.0 * center_shift:+.1f}%`，该中心段相对稳定。
- K2 的 `M0.90` 节点变化 `{100.0 * endpoint_shift:+.1f}%`；2.5° 数据实际只到约 M0.99，因此这个端点没有稳定识别。
- P1 的自由指数从 2.5° 的 `{strict_beta:.3f}` 变为 3° 的 `{primary_beta:.3f}`，也说明单一 RM-10 指数不能视为已冻结参数。
- 2° 主门禁只有 5 个点且无法形成有效逆算，本报告没有把它伪装成拟合。

## 验证边界

- H4 的 `M>=1.2` 节点逐值未修改；所有模型在 `M1.2` 使用冻结锚点 `{payload['h4_join_cda_m2']:.9f} m²`。
- T3 与 T1 的共同可见状态完全相同，但时间表最多错开 2 s；T3 只能说明可见输出可重复，不能提供真正独立的物理验证。
- T4 和 G3 作为独立诊断，但轨迹数量仍少、采样仍粗。
- 速度 RMSE 排除了每段被强制精确匹配的初始点；否则短轨迹误差会被系统性压低。
- 低 Mach 样本的迎角随速度下降而增加，因此临时 CdA 可能包含迎角附加阻力。
- RM-10 只提供归一化形状；K2 模型用于检查无 RM-10 形状时数据能否给出相近结论。

## 决策

本结果适合用于查看曲线趋势和设计下一轮采样，不适合冻结成最终 `M0.90–1.20` 模型。总体可信度：`Share with caveats / provisional only`。
"""
    (OUTPUT_DIR / "PROVISIONAL_FIT_REPORT.md").write_text(report, encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trajectories, h4, rm_mach, rm_cd = load_inputs()
    fits = {
        "alpha_2p5": fit_suite(2.5, trajectories, h4, rm_mach, rm_cd),
        "alpha_3p0": fit_suite(3.0, trajectories, h4, rm_mach, rm_cd),
    }
    loo = p1_leave_one_out(3.0, trajectories, h4, rm_mach, rm_cd)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_label": "local_candidate_H4_5_transonic_drag_provisional",
        "status": "provisional_fit_despite_failed_coverage_gate",
        "original_gate_status": "coverage_gate_fail_sampling_or_support",
        "method": "whole-trajectory axial speed replay with observed altitude and flight-path angle; equal trajectory weighting",
        "training_cases": ["T1", "T2"],
        "repeat_case": "T3",
        "independent_diagnostic_cases": ["T4", "G3"],
        "h4_join_mach": JOIN_MACH,
        "h4_join_cda_m2": h4.cda_m2(JOIN_MACH),
        "h4_nodes_overwritten": False,
        "sensitivity_fits": fits,
        "p1_leave_one_trajectory_out_alpha_3p0": loo,
        "interpretation_boundary": (
            "Sparse 2-4 s visible-label data and alpha up to 3 deg make this an exploratory effective-small-alpha fit only. "
            "It is not a passed Plan 4.5 identification and does not reveal the StatShark backend formula."
        ),
    }
    write_chart(payload)
    write_report(payload)
    clean = _strip_replays(payload)
    (OUTPUT_DIR / "provisional_fit.json").write_text(
        json.dumps(clean, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    primary = fits["alpha_3p0"]
    print("recommended", primary["recommended_provisional_model"])
    for name, item in primary["models"].items():
        print(
            name,
            "params=", item["parameters"],
            "train_rmse=", item["training_mean_relative_rmse"],
            "validation_rmse=", item["independent_validation_mean_relative_rmse"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
