#!/usr/bin/env python3
"""Fit the recovered H6R yaw plant without fitting the controller.

``currentG`` is treated only as an observed effective fin input.  It is not
renamed to a physical fin force.  The command arrays remain outside this fit
except for recovering the sign of the effective input.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aim120_model.h6_utils import (  # noqa: E402
    matrix_rank,
    normal_equations,
    pearson,
    rms,
    solve_linear_system,
    utc_now_iso,
    write_json,
    wrap_angle,
)


G0 = 9.80665
TRAIN_PREFIX = "H6R_R2E_HIGH_EXCITATION_SCALE_"
HOLDOUT_PREFIX = "H6R_R2S_NONZERO_SCALE_"
LOW_Q_PREFIX = "H6R_R4_LOW_Q_SCALE_WDK_"
SIGN_HOLDOUT_PREFIX = "H6R_R5_NEGATIVE_YAW_HOLDOUT_"


def _load_groups(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in payload.get("normalized_rows", []):
        groups[str(row.get("case_id", "unknown"))].append(dict(row))
    return {
        case_id: sorted(rows, key=lambda row: float(row["time_s"]))
        for case_id, rows in groups.items()
    }


def _median(values: Iterable[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("median requires at least one value")
    center = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[center]
    return 0.5 * (ordered[center - 1] + ordered[center])


def _quadratic_at(
    times: Sequence[float],
    values: Sequence[float],
    center: int,
    window_s: float,
) -> Optional[Tuple[float, float]]:
    """Return first and second derivatives from a centered local quadratic."""

    half = 0.5 * float(window_s)
    t0 = float(times[center])
    design: List[List[float]] = []
    target: List[float] = []
    left = center
    while left > 0 and t0 - float(times[left - 1]) <= half:
        left -= 1
    right = center
    while right + 1 < len(times) and float(times[right + 1]) - t0 <= half:
        right += 1
    for index in range(left, right + 1):
        tau = float(times[index]) - t0
        design.append([1.0, tau, tau * tau])
        target.append(float(values[index]))
    if len(design) < 5:
        return None
    matrix, rhs = normal_equations(design, target, ridge=1.0e-14)
    coefficients = solve_linear_system(matrix, rhs)
    return float(coefficients[1]), 2.0 * float(coefficients[2])


def _prepared_rows(rows: Sequence[Mapping[str, Any]], window_s: float) -> List[Dict[str, float]]:
    times = [float(row["time_s"]) for row in rows]
    yaw = [float(row["yaw_rad"]) for row in rows]
    path_yaw = [float(row["flight_path_yaw_rad"]) for row in rows]
    output: List[Dict[str, float]] = []
    for index, row in enumerate(rows):
        time_s = times[index]
        if time_s < 0.80 or time_s > times[-1] - 0.5 * float(window_s):
            continue
        yaw_fit = _quadratic_at(times, yaw, index, window_s)
        path_fit = _quadratic_at(times, path_yaw, index, window_s)
        command = row.get("a_cmd_yaw_g")
        current_g = row.get("current_g_reported")
        if yaw_fit is None or path_fit is None or command is None or current_g is None:
            continue
        command_value = float(command)
        sign = 1.0 if command_value >= 0.0 else -1.0
        beta = wrap_angle(yaw[index] - path_yaw[index])
        q_pa = float(row.get("dynamic_pressure_pa", 0.0))
        speed = float(row.get("speed_mps", 0.0))
        output.append({
            "time_s": time_s,
            "yaw_rad": yaw[index],
            "path_yaw_rad": path_yaw[index],
            "beta_rad": beta,
            "yaw_rate_rad_s": yaw_fit[0],
            "yaw_accel_rad_s2": yaw_fit[1],
            "path_rate_rad_s": path_fit[0],
            "path_accel_g": speed * path_fit[0] / G0,
            "u_eff_g": sign * abs(float(current_g)),
            "q_pa": q_pa,
        })
    return output


def _fit_linear(design: Sequence[Sequence[float]], target: Sequence[float]) -> Dict[str, Any]:
    matrix, rhs = normal_equations(design, target, ridge=1.0e-12)
    coefficients = solve_linear_system(matrix, rhs)
    residuals = [
        sum(float(left) * float(right) for left, right in zip(row, coefficients)) - float(observed)
        for row, observed in zip(design, target)
    ]
    correlations: Dict[str, Optional[float]] = {}
    for left in range(len(design[0])):
        for right in range(left + 1, len(design[0])):
            correlations["{}:{}".format(left, right)] = pearson(
                [float(row[left]) for row in design],
                [float(row[right]) for row in design],
            )
    return {
        "coefficients": [float(value) for value in coefficients],
        "sample_count": len(target),
        "design_rank": matrix_rank(design),
        "design_column_correlations": correlations,
        "rmse": rms(residuals),
    }


def _angular_fit(groups: Sequence[Sequence[Mapping[str, float]]]) -> Dict[str, Any]:
    design: List[List[float]] = []
    target: List[float] = []
    for rows in groups:
        for row in rows:
            design.append([
                float(row["u_eff_g"]),
                -float(row["beta_rad"]),
                -float(row["yaw_rate_rad_s"]),
            ])
            target.append(float(row["yaw_accel_rad_s2"]))
    fit = _fit_linear(design, target)
    fit["parameters"] = {
        "B_u_rad_s2_per_g": fit["coefficients"][0],
        "K_beta_per_s2": fit["coefficients"][1],
        "C_r_per_s": fit["coefficients"][2],
    }
    return fit


def _path_fit(
    groups: Sequence[Sequence[Mapping[str, float]]],
    q_ref_pa: float,
    direct_term: bool,
) -> Dict[str, Any]:
    design: List[List[float]] = []
    target: List[float] = []
    for rows in groups:
        for row in rows:
            values = [(float(row["q_pa"]) / q_ref_pa) * float(row["beta_rad"])]
            if direct_term:
                values.append(float(row["u_eff_g"]))
            design.append(values)
            target.append(float(row["path_accel_g"]))
    fit = _fit_linear(design, target)
    fit["parameters"] = {"G_beta_g_per_rad_at_q_ref": fit["coefficients"][0]}
    if direct_term:
        fit["parameters"]["G_u_direct"] = fit["coefficients"][1]
    return fit


def _evaluate_linear(
    fit: Mapping[str, Any],
    groups: Sequence[Sequence[Mapping[str, float]]],
    q_ref_pa: float,
    direct_term: bool,
) -> Dict[str, Any]:
    coefficients = [float(value) for value in fit["coefficients"]]
    residuals: List[float] = []
    by_case: List[Dict[str, Any]] = []
    for rows in groups:
        case_residuals: List[float] = []
        for row in rows:
            design = [(float(row["q_pa"]) / q_ref_pa) * float(row["beta_rad"])]
            if direct_term:
                design.append(float(row["u_eff_g"]))
            predicted = sum(left * right for left, right in zip(design, coefficients))
            case_residuals.append(predicted - float(row["path_accel_g"]))
        residuals.extend(case_residuals)
        by_case.append({"sample_count": len(rows), "rmse_g": rms(case_residuals)})
    return {"sample_count": len(residuals), "rmse_g": rms(residuals), "by_case": by_case}


def _replay_angular(
    rows: Sequence[Mapping[str, float]],
    params: Mapping[str, float],
) -> Dict[str, Any]:
    if len(rows) < 2:
        return {"sample_count": 0, "yaw_rmse_rad": None, "rate_rmse_rad_s": None}
    b_u = float(params["B_u_rad_s2_per_g"])
    k_beta = float(params["K_beta_per_s2"])
    c_r = float(params["C_r_per_s"])
    psi = float(rows[0]["yaw_rad"])
    rate = float(rows[0]["yaw_rate_rad_s"])
    yaw_residuals: List[float] = [0.0]
    rate_residuals: List[float] = [rate - float(rows[0]["yaw_rate_rad_s"])]
    for left, right in zip(rows, rows[1:]):
        dt = float(right["time_s"]) - float(left["time_s"])
        if dt <= 0.0:
            continue
        u_mid = 0.5 * (float(left["u_eff_g"]) + float(right["u_eff_g"]))
        path_mid = 0.5 * (float(left["path_yaw_rad"]) + float(right["path_yaw_rad"]))

        def rhs(state_psi: float, state_rate: float) -> Tuple[float, float]:
            beta = wrap_angle(state_psi - path_mid)
            return state_rate, b_u * u_mid - k_beta * beta - c_r * state_rate

        k1 = rhs(psi, rate)
        k2 = rhs(psi + 0.5 * dt * k1[0], rate + 0.5 * dt * k1[1])
        k3 = rhs(psi + 0.5 * dt * k2[0], rate + 0.5 * dt * k2[1])
        k4 = rhs(psi + dt * k3[0], rate + dt * k3[1])
        psi += dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0
        rate += dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
        yaw_residuals.append(wrap_angle(psi - float(right["yaw_rad"])))
        rate_residuals.append(rate - float(right["yaw_rate_rad_s"]))
    return {
        "sample_count": len(yaw_residuals),
        "yaw_rmse_rad": rms(yaw_residuals),
        "yaw_rmse_deg": math.degrees(float(rms(yaw_residuals) or 0.0)),
        "rate_rmse_rad_s": rms(rate_residuals),
    }


def _replay_set(
    groups: Sequence[Sequence[Mapping[str, float]]],
    params: Mapping[str, float],
) -> Dict[str, Any]:
    reports = [_replay_angular(rows, params) for rows in groups]
    weighted_yaw = []
    weighted_rate = []
    for rows, report in zip(groups, reports):
        weighted_yaw.extend([float(report["yaw_rmse_rad"])] * len(rows))
        weighted_rate.extend([float(report["rate_rmse_rad_s"])] * len(rows))
    return {
        "trajectory_count": len(groups),
        "sample_count": sum(len(rows) for rows in groups),
        "yaw_rmse_rad_weighted": rms(weighted_yaw),
        "yaw_rmse_deg_weighted": math.degrees(float(rms(weighted_yaw) or 0.0)),
        "rate_rmse_rad_s_weighted": rms(weighted_rate),
        "trajectories": reports,
    }


def _q_scaled_parameters(
    high: Mapping[str, float],
    low: Mapping[str, float],
    q_high_pa: float,
    q_low_pa: float,
) -> Dict[str, float]:
    if q_high_pa <= 0.0 or q_low_pa <= 0.0 or q_high_pa == q_low_pa:
        raise ValueError("two distinct positive q references are required")

    def exponent(high_value: float, low_value: float) -> float:
        if high_value <= 0.0 or low_value <= 0.0:
            raise ValueError("q exponent inference requires positive coefficients")
        return math.log(low_value / high_value) / math.log(q_low_pa / q_high_pa)

    return {
        "q_ref_pa": q_high_pa,
        "B_u_ref": float(high["B_u_rad_s2_per_g"]),
        "K_beta_ref": float(high["K_beta_per_s2"]),
        "C_r_ref": float(high["C_r_per_s"]),
        "B_u_q_exponent": exponent(float(high["B_u_rad_s2_per_g"]), float(low["B_u_rad_s2_per_g"])),
        "K_beta_q_exponent": exponent(float(high["K_beta_per_s2"]), float(low["K_beta_per_s2"])),
        "C_r_q_exponent": exponent(float(high["C_r_per_s"]), float(low["C_r_per_s"])),
    }


def _replay_angular_q_scaled(
    rows: Sequence[Mapping[str, float]],
    params: Mapping[str, float],
) -> Dict[str, Any]:
    if len(rows) < 2:
        return {"sample_count": 0, "yaw_rmse_rad": None, "rate_rmse_rad_s": None}
    q_ref = float(params["q_ref_pa"])

    def scaled(name: str, exponent_name: str, q_pa: float) -> float:
        ratio = max(float(q_pa), 1.0) / max(q_ref, 1.0)
        return float(params[name]) * ratio ** float(params[exponent_name])

    psi = float(rows[0]["yaw_rad"])
    rate = float(rows[0]["yaw_rate_rad_s"])
    yaw_residuals: List[float] = [0.0]
    rate_residuals: List[float] = [0.0]
    for left, right in zip(rows, rows[1:]):
        dt = float(right["time_s"]) - float(left["time_s"])
        if dt <= 0.0:
            continue
        u_mid = 0.5 * (float(left["u_eff_g"]) + float(right["u_eff_g"]))
        path_mid = 0.5 * (float(left["path_yaw_rad"]) + float(right["path_yaw_rad"]))
        q_mid = 0.5 * (float(left["q_pa"]) + float(right["q_pa"]))
        b_u = scaled("B_u_ref", "B_u_q_exponent", q_mid)
        k_beta = scaled("K_beta_ref", "K_beta_q_exponent", q_mid)
        c_r = scaled("C_r_ref", "C_r_q_exponent", q_mid)

        def rhs(state_psi: float, state_rate: float) -> Tuple[float, float]:
            beta = wrap_angle(state_psi - path_mid)
            return state_rate, b_u * u_mid - k_beta * beta - c_r * state_rate

        k1 = rhs(psi, rate)
        k2 = rhs(psi + 0.5 * dt * k1[0], rate + 0.5 * dt * k1[1])
        k3 = rhs(psi + 0.5 * dt * k2[0], rate + 0.5 * dt * k2[1])
        k4 = rhs(psi + dt * k3[0], rate + dt * k3[1])
        psi += dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0
        rate += dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
        yaw_residuals.append(wrap_angle(psi - float(right["yaw_rad"])))
        rate_residuals.append(rate - float(right["yaw_rate_rad_s"]))
    return {
        "sample_count": len(yaw_residuals),
        "yaw_rmse_rad": rms(yaw_residuals),
        "yaw_rmse_deg": math.degrees(float(rms(yaw_residuals) or 0.0)),
        "rate_rmse_rad_s": rms(rate_residuals),
    }


def _replay_q_scaled_set(
    groups: Sequence[Sequence[Mapping[str, float]]],
    params: Mapping[str, float],
) -> Dict[str, Any]:
    reports = [_replay_angular_q_scaled(rows, params) for rows in groups]
    weighted_yaw: List[float] = []
    weighted_rate: List[float] = []
    for rows, report in zip(groups, reports):
        weighted_yaw.extend([float(report["yaw_rmse_rad"])] * len(rows))
        weighted_rate.extend([float(report["rate_rmse_rad_s"])] * len(rows))
    return {
        "trajectory_count": len(groups),
        "sample_count": sum(len(rows) for rows in groups),
        "yaw_rmse_rad_weighted": rms(weighted_yaw),
        "yaw_rmse_deg_weighted": math.degrees(float(rms(weighted_yaw) or 0.0)),
        "rate_rmse_rad_s_weighted": rms(weighted_rate),
        "trajectories": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "h6_fin_dynamics_recovery" / "h6_normalized_samples.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "h6_fin_dynamics_recovery" / "effective_yaw_plant_fit.json",
    )
    args = parser.parse_args()
    groups = _load_groups(args.input)
    train_raw = [rows for case_id, rows in sorted(groups.items()) if case_id.startswith(TRAIN_PREFIX)]
    holdout_raw = [rows for case_id, rows in sorted(groups.items()) if case_id.startswith(HOLDOUT_PREFIX)]
    sign_holdout_raw = [
        rows for case_id, rows in sorted(groups.items()) if case_id.startswith(SIGN_HOLDOUT_PREFIX)
    ]
    low_q_raw = [
        rows
        for case_id, rows in sorted(groups.items())
        if case_id.startswith(LOW_Q_PREFIX) and "_H6R_W2_" not in case_id
    ]
    q_fit_labels = (
        "H6R_SEM_F001",
        "H6R_SEM_F003",
        "H6R_SEM_F010",
        "H6R_SEM_F030",
        "H6R_BODY_C0_F100",
    )
    high_q_fit_raw = [
        rows
        for case_id, rows in sorted(groups.items())
        if case_id.startswith(TRAIN_PREFIX) and case_id.endswith(q_fit_labels)
    ]
    low_q_fit_raw = [
        rows
        for case_id, rows in sorted(groups.items())
        if case_id.startswith(LOW_Q_PREFIX) and case_id.endswith(q_fit_labels)
    ]
    high_q_validation_raw = [
        rows
        for case_id, rows in sorted(groups.items())
        if case_id.startswith(TRAIN_PREFIX) and not case_id.endswith(q_fit_labels)
    ]
    low_q_validation_raw = [
        rows
        for case_id, rows in sorted(groups.items())
        if case_id.startswith(LOW_Q_PREFIX)
        and "_H6R_W2_" not in case_id
        and not case_id.endswith(q_fit_labels)
    ]
    if not train_raw or not holdout_raw:
        report = {
            "schema_version": 1,
            "generated_at_utc": utc_now_iso(),
            "status": "blocked_missing_train_or_holdout",
            "train_trajectory_count": len(train_raw),
            "holdout_trajectory_count": len(holdout_raw),
            "parameters": None,
        }
        write_json(args.output, report)
        print(json.dumps(report, ensure_ascii=False))
        return 1

    windows = (0.10, 0.20, 0.30)
    window_reports: List[Dict[str, Any]] = []
    for window_s in windows:
        train = [_prepared_rows(rows, window_s) for rows in train_raw]
        holdout = [_prepared_rows(rows, window_s) for rows in holdout_raw]
        sign_holdout = [_prepared_rows(rows, window_s) for rows in sign_holdout_raw]
        low_q = [_prepared_rows(rows, window_s) for rows in low_q_raw]
        high_q_fit = [_prepared_rows(rows, window_s) for rows in high_q_fit_raw]
        low_q_fit = [_prepared_rows(rows, window_s) for rows in low_q_fit_raw]
        high_q_validation = [_prepared_rows(rows, window_s) for rows in high_q_validation_raw]
        low_q_validation = [_prepared_rows(rows, window_s) for rows in low_q_validation_raw]
        angular = _angular_fit(train)
        low_q_angular = _angular_fit(low_q)
        combined_angular = _angular_fit(train + low_q)
        q_high_fit = _angular_fit(high_q_fit)
        q_low_fit = _angular_fit(low_q_fit)
        q_high_ref = _median(row["q_pa"] for trajectory in high_q_fit for row in trajectory)
        q_low_ref = _median(row["q_pa"] for trajectory in low_q_fit for row in trajectory)
        q_scaled = _q_scaled_parameters(
            q_high_fit["parameters"],
            q_low_fit["parameters"],
            q_high_ref,
            q_low_ref,
        )
        q_ref = _median(row["q_pa"] for trajectory in train for row in trajectory)
        path_a = _path_fit(train, q_ref, direct_term=False)
        path_b = _path_fit(train, q_ref, direct_term=True)
        train_a = _evaluate_linear(path_a, train, q_ref, direct_term=False)
        train_b = _evaluate_linear(path_b, train, q_ref, direct_term=True)
        holdout_a = _evaluate_linear(path_a, holdout, q_ref, direct_term=False)
        holdout_b = _evaluate_linear(path_b, holdout, q_ref, direct_term=True)
        improvement = 1.0 - float(holdout_b["rmse_g"]) / max(float(holdout_a["rmse_g"]), 1.0e-12)
        window_reports.append({
            "window_s": window_s,
            "q_ref_pa": q_ref,
            "angular_derivative_fit": angular,
            "angular_replay_train": _replay_set(train, angular["parameters"]),
            "angular_replay_holdout": _replay_set(holdout, angular["parameters"]),
            "angular_replay_sign_holdout": _replay_set(sign_holdout, angular["parameters"]),
            "low_q_angular_derivative_fit": low_q_angular,
            "combined_q_angular_derivative_fit": combined_angular,
            "angular_replay_low_q_from_high_q_fit": _replay_set(low_q, angular["parameters"]),
            "angular_replay_low_q_from_low_q_fit": _replay_set(low_q, low_q_angular["parameters"]),
            "angular_replay_combined_q": _replay_set(train + low_q, combined_angular["parameters"]),
            "angular_replay_holdout_from_combined_q_fit": _replay_set(holdout, combined_angular["parameters"]),
            "q_scaled_candidate": {
                "fit_labels": list(q_fit_labels),
                "high_q_derivative_fit": q_high_fit,
                "low_q_derivative_fit": q_low_fit,
                "parameters": q_scaled,
                "replay_fit_trajectories": _replay_q_scaled_set(high_q_fit + low_q_fit, q_scaled),
                "replay_scale_validation": _replay_q_scaled_set(high_q_validation + low_q_validation, q_scaled),
                "replay_geometry_holdout": _replay_q_scaled_set(holdout, q_scaled),
                "replay_sign_holdout": _replay_q_scaled_set(sign_holdout, q_scaled),
            },
            "path_moment_only": {"fit": path_a, "train": train_a, "holdout": holdout_a},
            "path_moment_plus_direct": {"fit": path_b, "train": train_b, "holdout": holdout_b},
            "direct_term_holdout_relative_improvement": improvement,
        })

    canonical = window_reports[1]
    direct_improvements = [float(item["direct_term_holdout_relative_improvement"]) for item in window_reports]
    direct_coefficients = [
        float(item["path_moment_plus_direct"]["fit"]["parameters"]["G_u_direct"])
        for item in window_reports
    ]
    stable_direct_sign = all(value > 0.0 for value in direct_coefficients) or all(value < 0.0 for value in direct_coefficients)
    retain_direct = min(direct_improvements) >= 0.10 and stable_direct_sign
    angular_parameters = [item["angular_derivative_fit"]["parameters"] for item in window_reports]
    report = {
        "schema_version": 1,
        "generated_at_utc": utc_now_iso(),
        "source_kind": "statshark_backend_timeseries",
        "model_boundary": {
            "u_eff": "signed currentG observation in G; not physical fin force",
            "controller_command_excluded": "aCmdYaw is used only for sign and is not fitted",
            "angular_equation": "r_dot = B_u*u_eff - K_beta*beta - C_r*r",
            "path_candidate_A": "a_path/g = G_beta*(q/q_ref)*beta",
            "path_candidate_B": "candidate_A + G_u_direct*u_eff",
        },
        "train_case_prefix": TRAIN_PREFIX,
        "holdout_case_prefix": HOLDOUT_PREFIX,
        "train_trajectory_count": len(train_raw),
        "holdout_trajectory_count": len(holdout_raw),
        "sign_holdout_trajectory_count": len(sign_holdout_raw),
        "low_q_trajectory_count": len(low_q_raw),
        "window_reports": window_reports,
        "canonical_window_s": 0.20,
        "canonical_effective_parameters": canonical["angular_derivative_fit"]["parameters"],
        "canonical_low_q_effective_parameters": canonical["low_q_angular_derivative_fit"]["parameters"],
        "canonical_combined_q_effective_parameters": canonical["combined_q_angular_derivative_fit"]["parameters"],
        "window_parameter_sensitivity": angular_parameters,
        "path_structure_selection": {
            "selected": "moment_plus_direct" if retain_direct else "moment_only",
            "direct_term_retained": retain_direct,
            "holdout_relative_improvements": direct_improvements,
            "direct_coefficients": direct_coefficients,
            "stable_direct_sign": stable_direct_sign,
            "rule": "retain only if every derivative window improves holdout RMSE by at least 10% and sign is stable",
        },
        "canonical_train_replay": canonical["angular_replay_train"],
        "canonical_holdout_replay": canonical["angular_replay_holdout"],
        "canonical_sign_holdout_replay": canonical["angular_replay_sign_holdout"],
        "canonical_low_q_replay_from_high_q_fit": canonical["angular_replay_low_q_from_high_q_fit"],
        "canonical_low_q_replay_from_low_q_fit": canonical["angular_replay_low_q_from_low_q_fit"],
        "canonical_combined_q_replay": canonical["angular_replay_combined_q"],
        "canonical_holdout_replay_from_combined_q_fit": canonical["angular_replay_holdout_from_combined_q_fit"],
        "canonical_q_scaled_candidate": canonical["q_scaled_candidate"],
        "formal_parameter_status": "effective_candidate_pending_arm_wdk_q_holdout",
        "status": "fit_complete_effective_candidate",
    }
    write_json(args.output, report)
    print(json.dumps({
        "status": report["status"],
        "parameters": report["canonical_effective_parameters"],
        "path_structure": report["path_structure_selection"]["selected"],
        "holdout_yaw_rmse_deg": report["canonical_holdout_replay"]["yaw_rmse_deg_weighted"],
        "output": str(args.output.resolve()),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
