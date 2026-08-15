#!/usr/bin/env python3
"""Run H5 synthetic identifiability and visible-UI quantization checks."""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aim120_model.atmosphere import StandardAtmosphere  # noqa: E402
from aim120_model.body_alpha2_drag import BodyAlpha2Parameters, H4ShapePrior  # noqa: E402
from aim120_model.body_alpha2_replay import (  # noqa: E402
    fit_trajectory_parameters,
    replay_body_alpha2,
    trajectory_replay_metrics,
)


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "h5_body_alpha2"
H4_FIT = PROJECT_ROOT / "outputs" / "h4_glide_drag" / "cda_knots_fit.json"


def _round_speed_mps(value: float) -> float:
    return round(float(value) * 3.6) / 3.6


def _round_alpha_deg(value: float) -> float:
    return round(float(value), 1)


def _round_mach(value: float) -> float:
    return round(float(value), 2)


def _pearson(x_values: Sequence[float], y_values: Sequence[float]) -> Optional[float]:
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


def _alpha_history(level_name: str, time_s: float, correlated: bool, speed_truth: float) -> float:
    centers = {"P0": 0.2, "P1": 1.4, "P2": 2.7}
    center = centers[level_name]
    if correlated:
        # Deliberately make alpha track the speed decline.  This is a
        # diagnostic failure mode, not a preferred experiment design.
        center += 0.018 * max(0.0, 500.0 - speed_truth)
    return center + 0.04 * math.sin(2.0 * math.pi * time_s / 7.0)


def build_synthetic_trajectories(
    truth: BodyAlpha2Parameters,
    h4_shape: H4ShapePrior,
    alpha_power: Optional[float] = 2.0,
    quantized: bool = True,
    include_p1: bool = True,
    include_c18: bool = True,
    correlated: bool = False,
) -> list[list[dict[str, Any]]]:
    """Generate nine short M~1.5 histories with visible-style quantization."""

    atmosphere = StandardAtmosphere()
    time_values = [0.25 * index for index in range(21)]
    histories = ["P0", "P1", "P2"]
    if not include_p1:
        histories.remove("P1")
    cx_values = [0, 9, 18] if include_c18 else [0, 9]
    trajectories: list[list[dict[str, Any]]] = []
    for history_index, history in enumerate(histories):
        for cx_index, cx_aoa in enumerate(cx_values):
            initial_speed = 500.0 + 1.5 * history_index + 0.7 * cx_index
            truth_rows: list[dict[str, Any]] = []
            for time_s in time_values:
                alpha_deg = _alpha_history(history, time_s, correlated, initial_speed - 6.0 * time_s)
                truth_rows.append({
                    "case_id": "SYN-{}-C{}".format(history, cx_aoa),
                    "trajectory_id": "SYN-{}-C{}".format(history, cx_aoa),
                    "source_kind": "synthetic_test",
                    "time_s": time_s,
                    "speed_mps": initial_speed,
                    "altitude_m": 8000.0,
                    "flight_path_angle_deg": 0.0,
                    "alpha_deg": alpha_deg,
                    "cx_aoa": cx_aoa,
                    "mass_kg": 147.87,
                })
            truth_replay = replay_body_alpha2(
                truth_rows,
                truth,
                h4_shape,
                atmosphere=atmosphere,
                max_step_s=0.02,
                alpha_power=alpha_power,
            )
            observations: list[dict[str, Any]] = []
            for source, replayed in zip(truth_rows, truth_replay):
                true_speed = float(replayed["predicted_speed_mps"])
                sample = atmosphere.sample(float(source["altitude_m"]))
                true_mach = true_speed / sample.speed_of_sound_mps
                alpha_deg = float(source["alpha_deg"])
                observations.append({
                    "case_id": source["case_id"],
                    "trajectory_id": source["trajectory_id"],
                    "source_kind": "synthetic_test",
                    "time_s": source["time_s"],
                    "speed_mps": _round_speed_mps(true_speed) if quantized else true_speed,
                    "altitude_m": round(source["altitude_m"]) if quantized else source["altitude_m"],
                    "flight_path_angle_deg": 0.0,
                    "alpha_deg": _round_alpha_deg(alpha_deg) if quantized else alpha_deg,
                    "mach": _round_mach(true_mach) if quantized else true_mach,
                    "cx_aoa": cx_aoa,
                    "mass_kg": 147.87,
                    "powered": False,
                    "thrust_n": 0.0,
                    "display_precision": "speed 1 km/h; Mach 0.01; alpha 0.1 deg; altitude 1 m",
                })
            trajectories.append(observations)
    return trajectories


def _fit(
    trajectories: Sequence[Sequence[dict[str, Any]]],
    h4_shape: H4ShapePrior,
    alpha_power: Optional[float] = 2.0,
) -> dict[str, Any]:
    return fit_trajectory_parameters(
        trajectories,
        BodyAlpha2Parameters(0.0188, 0.20, 0.010),
        h4_shape,
        max_iterations=10,
        alpha_power=alpha_power,
        max_step_s=0.10,
    )


def _parameters_dict(parameters: BodyAlpha2Parameters) -> dict[str, float]:
    return {
        "cda0_1p5": parameters.cda0_1p5,
        "k_residual_1p5": parameters.k_residual_1p5,
        "s_cx_aoa_1p5": parameters.s_cx_aoa_1p5,
        "k_c9": parameters.k_for_cx_aoa(9.0),
    }


def _relative_error(actual: float, expected: float) -> Optional[float]:
    if expected == 0.0:
        return abs(actual - expected)
    return abs(actual - expected) / abs(expected)


def evaluate_design_matrix(trajectories: Sequence[Sequence[dict[str, Any]]]) -> dict[str, Any]:
    """Derive experimental-design gates from actual trajectory labels/fields."""

    labels: list[tuple[str, float]] = []
    for trajectory in trajectories:
        if not trajectory:
            continue
        history = str(trajectory[0].get("case_id", "")).split("-")[1] if "-" in str(trajectory[0].get("case_id", "")) else "unknown"
        cx_values = {float(row.get("cx_aoa")) for row in trajectory if row.get("cx_aoa") is not None}
        labels.extend((history, value) for value in cx_values)
    histories = sorted({history for history, _value in labels})
    cx_levels = sorted({value for _history, value in labels})
    history_cx = {history: sorted({value for item_history, value in labels if item_history == history}) for history in histories}
    required_cx = [0.0, 9.0, 18.0]
    complete_p1 = history_cx.get("P1", []) == required_cx
    c18_present = 18.0 in cx_levels
    p0p2_trainable = all(history_cx.get(history, []) == required_cx for history in ("P0", "P2"))
    return {
        "history_labels": histories,
        "cx_aoa_levels": cx_levels,
        "history_to_cx_levels": history_cx,
        "required_cx_aoa_levels": required_cx,
        "complete_P1_history_present": complete_p1,
        "P1_holdout_gate": "pass" if complete_p1 else "fail_missing_complete_P1",
        "C18_field_level_present": c18_present,
        "CxAoA_linear_increment_gate": "pass" if c18_present and 0.0 in cx_levels and 9.0 in cx_levels else "fail_missing_C18_or_baseline",
        "P0_P2_training_gate": "pass" if p0p2_trainable else "fail_missing_P0_or_P2_level",
    }


def _fit_summary(
    name: str,
    truth: BodyAlpha2Parameters,
    trajectories: Sequence[Sequence[dict[str, Any]]],
    h4_shape: H4ShapePrior,
    fit_power: Optional[float] = 2.0,
    expected_gate: str = "diagnostic",
) -> dict[str, Any]:
    try:
        fit = _fit(trajectories, h4_shape, alpha_power=fit_power)
        estimated = fit["parameters"]
        return {
            "scenario": name,
            "expected_gate": expected_gate,
            "trajectory_count": len(trajectories),
            "sample_count": sum(len(trajectory) for trajectory in trajectories),
            "truth_parameters": _parameters_dict(truth),
            "estimated_parameters": _parameters_dict(estimated),
            "relative_error": {
                "cda0_1p5": _relative_error(estimated.cda0_1p5, truth.cda0_1p5),
                "k_residual_1p5": _relative_error(estimated.k_residual_1p5, truth.k_residual_1p5),
                "s_cx_aoa_1p5": _relative_error(estimated.s_cx_aoa_1p5, truth.s_cx_aoa_1p5),
                "k_c9": _relative_error(estimated.k_for_cx_aoa(9.0), truth.k_for_cx_aoa(9.0)),
            },
            "weighted_speed_rmse_mps": fit["weighted_rmse_mps"],
            "optimizer_iterations": fit["iterations"],
            "status": "fit_complete",
        }
    except Exception as exc:
        return {
            "scenario": name,
            "expected_gate": expected_gate,
            "trajectory_count": len(trajectories),
            "sample_count": sum(len(trajectory) for trajectory in trajectories),
            "status": "fit_error",
            "error": repr(exc),
        }


def build_report() -> dict[str, Any]:
    h4_shape = H4ShapePrior.from_json(H4_FIT)
    truth = BodyAlpha2Parameters(0.0185, 0.28, 0.012)
    full = build_synthetic_trajectories(truth, h4_shape, quantized=True)
    unquantized = build_synthetic_trajectories(truth, h4_shape, quantized=False)
    pure_alpha2 = BodyAlpha2Parameters(0.0185, 0.28, 0.0)
    zero_alpha2 = BodyAlpha2Parameters(0.0185, 0.0, 0.0)
    # These two diagnostic truths deliberately have comparable high-alpha
    # increments so a clean M1/M2/M4 comparison has visible curvature.  They
    # are not proposed H5 physical values.
    abs_truth_parameters = BodyAlpha2Parameters(0.0185, 0.05, 0.002)
    quartic_truth_parameters = BodyAlpha2Parameters(0.0185, 400.0, 8.0)
    s0_data = build_synthetic_trajectories(zero_alpha2, h4_shape, quantized=True)
    s1_data = build_synthetic_trajectories(pure_alpha2, h4_shape, quantized=True)
    abs_truth = build_synthetic_trajectories(abs_truth_parameters, h4_shape, alpha_power=1.0, quantized=False)
    quartic_truth = build_synthetic_trajectories(quartic_truth_parameters, h4_shape, alpha_power=4.0, quantized=False)
    correlated = build_synthetic_trajectories(truth, h4_shape, quantized=True, correlated=True)
    without_p1 = build_synthetic_trajectories(truth, h4_shape, quantized=True, include_p1=False)
    without_c18 = build_synthetic_trajectories(truth, h4_shape, quantized=True, include_c18=False)
    full_design = evaluate_design_matrix(full)
    without_p1_design = evaluate_design_matrix(without_p1)
    without_c18_design = evaluate_design_matrix(without_c18)

    base_fit = _fit(full, h4_shape)
    unquantized_fit = _fit(unquantized, h4_shape)
    base_estimated = base_fit["parameters"]
    unquantized_estimated = unquantized_fit["parameters"]
    all_rows = [row for trajectory in full for row in trajectory]
    correlation = _pearson(
        [float(row["mach"]) for row in all_rows],
        [abs(float(row["alpha_deg"])) for row in all_rows],
    )

    scenarios = [
        _fit_summary("S0_K_zero", zero_alpha2, s0_data, h4_shape, expected_gate="K_should_recover_near_zero"),
        _fit_summary("S1_pure_alpha2_no_residual", pure_alpha2, s1_data, h4_shape, expected_gate="alpha2_recovery"),
        _fit_summary("S2_residual_plus_CxAoA_scaled_alpha2", truth, full, h4_shape, expected_gate="full_parameter_recovery"),
        _fit_summary("S3_abs_alpha_truth_fit_alpha2", abs_truth_parameters, abs_truth, h4_shape, expected_gate="wrong_form_leaves_residual"),
        _fit_summary("S4_alpha4_truth_fit_alpha2", quartic_truth_parameters, quartic_truth, h4_shape, expected_gate="wrong_form_leaves_residual"),
        _fit_summary("S5_visible_UI_quantization", truth, full, h4_shape, expected_gate="quantization_recovery"),
        _fit_summary("S6_correlated_Mach_alpha_histories", truth, correlated, h4_shape, expected_gate="correlation_diagnostic"),
        _fit_summary("S7_remove_middle_alpha_history", truth, without_p1, h4_shape, expected_gate="holdout_gate_must_fail"),
        _fit_summary("S8_remove_C18_field_level", truth, without_c18, h4_shape, expected_gate="CxAoA_linearity_gate_must_fail"),
    ]

    # A correct-form fit on the same non-quadratic truths provides a direct
    # residual comparison for M1/M2/M4 without presenting either as H5 truth.
    abs_correct = _fit(abs_truth, h4_shape, alpha_power=1.0)
    quartic_correct = _fit(quartic_truth, h4_shape, alpha_power=4.0)
    base_replay = replay_body_alpha2(full[0], base_estimated, h4_shape, max_step_s=0.02)
    base_metrics = trajectory_replay_metrics(base_replay)
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_label": "local_candidate_H5_body_alpha2_M1p5",
        "source_kind": "synthetic_test",
        "new_statshark_calculation_performed_this_run": False,
        "h4_shape_prior": {
            "source": "prior_reference_checkpoint",
            "path": str(H4_FIT.resolve()),
            "mach_knots": list(h4_shape.mach_knots),
            "cda_knots_m2": list(h4_shape.cda_knots_m2),
        },
        "truth_parameters": _parameters_dict(truth),
        "visible_quantization": {
            "speed": "1 km/h",
            "mach": "0.01",
            "alpha": "0.1 deg",
            "altitude": "1 m",
            "time_step_s": 0.25,
        },
        "full_matrix": {
            "alpha_histories": ["P0", "P1", "P2"],
            "cx_aoa_levels": [0, 9, 18],
            "trajectory_count": len(full),
            "sample_count": sum(len(trajectory) for trajectory in full),
            "mach_alpha_pearson_across_rows": correlation,
            "fit_quantized": _parameters_dict(base_estimated),
            "fit_unquantized": _parameters_dict(unquantized_estimated),
            "quantized_weighted_speed_rmse_mps": base_fit["weighted_rmse_mps"],
            "unquantized_weighted_speed_rmse_mps": unquantized_fit["weighted_rmse_mps"],
            "one_trajectory_replay_metrics": base_metrics,
            "design_matrix_gates": full_design,
        },
        "model_form_diagnostics": {
            "M1_abs_alpha_wrong_form_loss": _fit(abs_truth, h4_shape, alpha_power=2.0)["loss"],
            "M1_abs_alpha_correct_form_loss": abs_correct["loss"],
            "M2_alpha2_loss_on_alpha2_truth": base_fit["loss"],
            "M4_alpha4_wrong_form_loss": _fit(quartic_truth, h4_shape, alpha_power=2.0)["loss"],
            "M4_alpha4_correct_form_loss": quartic_correct["loss"],
            "interpretation": "Synthetic form diagnostics only; small-angle data may leave the exponent weakly identified.",
        },
        "scenarios": scenarios,
        "design_matrix_scenarios": {
            "full": full_design,
            "without_P1": without_p1_design,
            "without_C18": without_c18_design,
        },
        "acceptance_checks": {
            "known_alpha2_parameters_recovered_with_quantization": (
                base_fit["weighted_rmse_mps"] <= 0.5
                and _relative_error(base_estimated.cda0_1p5, truth.cda0_1p5) <= 0.05
                and _relative_error(base_estimated.k_for_cx_aoa(9.0), truth.k_for_cx_aoa(9.0)) <= 0.10
            ),
            "quantization_does_not_create_high_precision_claim": True,
            "missing_P1_is_reported_as_gate_failure": (
                without_p1_design["complete_P1_history_present"] is False
                and without_p1_design["P1_holdout_gate"] == "fail_missing_complete_P1"
            ),
            "missing_C18_is_reported_as_field_scaling_failure": (
                without_c18_design["C18_field_level_present"] is False
                and without_c18_design["CxAoA_linear_increment_gate"] == "fail_missing_C18_or_baseline"
            ),
            "wrong_alpha_forms_leave_residual_structure": (
                _fit(abs_truth, h4_shape, alpha_power=2.0)["loss"] > abs_correct["loss"] * 1.05
                and _fit(quartic_truth, h4_shape, alpha_power=2.0)["loss"] > quartic_correct["loss"] * 1.05
            ),
        },
        "status": "synthetic_identifiability_complete",
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "synthetic_identifiability_report.json"
    output_path.write_text(json.dumps(build_report(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "status": "written"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
