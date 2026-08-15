#!/usr/bin/env python3
"""Extract and invert low-g samples from local H2 trajectories.

This script intentionally accepts H2 output only as a local pipeline test.
It never labels local samples as StatShark reference telemetry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from aim120_model.axial_inverse import estimate_smoothed_speeds, estimate_speed_derivatives, invert_rows
from aim120_model.config import load_model_config
from aim120_model.sample_filters import (
    LowGFilterSettings,
    apply_filter,
    normalize_sample,
    summarize_filter_rows,
)


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


def _default_input() -> Path:
    candidates = sorted((PROJECT_DIR / "outputs" / "h2").glob("power_only_*.json"))
    if not candidates:
        raise FileNotFoundError("no local H2 power_only output found")
    return candidates[-1]


def _load_h2_samples(path: Path) -> tuple[str, str, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "result" in payload:
        result = payload["result"]
        case_name = str(result.get("case_name", path.stem))
        model_label = str(payload.get("metadata", {}).get("model_label", result.get("model_label", "unknown")))
        samples = result.get("samples", [])
    else:
        case_name = str(payload.get("case_name", path.stem))
        model_label = str(payload.get("model_label", "unknown"))
        samples = payload.get("samples", [])
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{path} contains no trajectory samples")
    return case_name, model_label, [dict(sample) for sample in samples]


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


def _window_summary(rows: Iterable[Mapping[str, Any]], primary_window_s: float, alternate_window_s: float) -> dict[str, Any]:
    primary_key = str(primary_window_s)
    alternate_key = str(alternate_window_s)
    differences: list[float] = []
    relative_differences: list[float] = []
    for row in rows:
        if not row.get("accepted"):
            continue
        primary = row.get("observed_cda_m2_by_window", {}).get(primary_key)
        alternate = row.get("observed_cda_m2_by_window", {}).get(alternate_key)
        if primary is None or alternate is None:
            continue
        primary_value = float(primary)
        difference = abs(primary_value - float(alternate))
        differences.append(difference)
        relative_differences.append(difference / max(abs(primary_value), 1.0e-12))
    if not differences:
        return {
            "paired_rows": 0,
            "mean_abs_cda_difference_m2": None,
            "p95_abs_cda_difference_m2": None,
            "mean_relative_difference": None,
            "max_relative_difference": None,
        }
    differences.sort()
    relative_differences.sort()
    p95_index = min(len(differences) - 1, int(math.ceil(0.95 * len(differences))) - 1)
    return {
        "paired_rows": len(differences),
        "mean_abs_cda_difference_m2": sum(differences) / len(differences),
        "p95_abs_cda_difference_m2": differences[p95_index],
        "mean_relative_difference": sum(relative_differences) / len(relative_differences),
        "max_relative_difference": relative_differences[-1],
    }


def _sensitivity_counts(rows: list[Mapping[str, Any]], base: LowGFilterSettings, config: Mapping[str, Any]) -> dict[str, Any]:
    sensitivity = config["low_g_drag"]["sensitivity"]
    result: dict[str, Any] = {}
    for lateral in sensitivity["lateral_load_thresholds_g"]:
        for alpha in sensitivity["alpha_thresholds_deg"]:
            for gamma in sensitivity["flight_path_thresholds_deg"]:
                settings = replace(
                    base,
                    lateral_load_threshold_g=float(lateral),
                    alpha_threshold_deg=float(alpha),
                    flight_path_threshold_deg=float(gamma),
                )
                filtered = [apply_filter(row, settings) for row in rows]
                key = f"lateral_{float(lateral):g}_alpha_{float(alpha):g}_gamma_{float(gamma):g}"
                result[key] = {
                    "lateral_load_threshold_g": float(lateral),
                    "alpha_threshold_deg": float(alpha),
                    "flight_path_threshold_deg": float(gamma),
                    "accepted_rows": sum(1 for row in filtered if row["accepted"]),
                    "total_rows": len(filtered),
                }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs" / "h3_low_g_drag")
    args = parser.parse_args()

    config_path = PROJECT_DIR / "configs" / "aim120a_h3_low_g_drag.yaml"
    config = load_model_config(config_path)
    settings = _settings_from_config(config)
    smoothing = config["low_g_drag"]["smoothing"]
    windows = [float(value) for value in smoothing["window_widths_s"]]
    primary_window_s = float(smoothing["primary_window_s"])
    if not windows:
        raise ValueError("at least one smoothing window is required")
    if str(primary_window_s) not in {str(window) for window in windows}:
        windows.insert(0, primary_window_s)

    inputs = [Path(path) for path in args.input] if args.input else [_default_input()]
    normalized_rows: list[dict[str, Any]] = []
    source_files: list[dict[str, Any]] = []
    source_labels: Counter[str] = Counter()
    for input_path in inputs:
        input_path = input_path.resolve()
        case_name, model_label, raw_samples = _load_h2_samples(input_path)
        source_labels[model_label] += 1
        source_files.append({
            "path": str(input_path),
            "sha256": _sha256(input_path),
            "case_name": case_name,
            "model_label": model_label,
            "source_kind": "local_pipeline_test",
            "sample_count": len(raw_samples),
        })
        for source_index, raw in enumerate(raw_samples):
            raw["source_case"] = case_name
            raw["source_kind"] = "local_pipeline_test"
            raw["source_time_index"] = source_index
            normalized_rows.append(
                normalize_sample(
                    raw,
                    settings=settings,
                    default_source_case=case_name,
                    default_source_kind="local_pipeline_test",
                )
            )

    derivative_by_window = {
        str(window): estimate_speed_derivatives(
            normalized_rows,
            window_s=window,
            polynomial_order=int(smoothing["polynomial_order"]),
        )
        for window in windows
    }
    smoothed_speed_by_window = {
        str(window): estimate_smoothed_speeds(
            normalized_rows,
            window_s=window,
            polynomial_order=int(smoothing["polynomial_order"]),
        )
        for window in windows
    }
    primary_key = str(primary_window_s)
    for index, row in enumerate(normalized_rows):
        raw_speed = row.get("speed_mps")
        smoothed_speed = smoothed_speed_by_window[primary_key][index]
        row["raw_speed_mps"] = raw_speed
        row["smoothed_speed_mps"] = smoothed_speed
        row["smoothed_speed_mps_by_window"] = {
            str(window): values[index]
            for window, values in smoothed_speed_by_window.items()
        }
        row["speed_smoothing_residual_mps"] = (
            float(raw_speed) - float(smoothed_speed)
            if raw_speed is not None and math.isfinite(float(raw_speed)) and math.isfinite(float(smoothed_speed))
            else float("nan")
        )
        row["residual"] = row["speed_smoothing_residual_mps"]
    inverse_rows = invert_rows(
        normalized_rows,
        derivative_by_window,
        primary_window_s=primary_window_s,
        gravity_mps2=float(config["atmosphere"]["gravity_mps2"]),
        settings=settings,
    )
    summary = summarize_filter_rows(inverse_rows)
    summary["accepted_by_power_state"] = {
        "powered": sum(1 for row in inverse_rows if row.get("accepted") and row.get("powered")),
        "coast": sum(1 for row in inverse_rows if row.get("accepted") and not row.get("powered")),
    }
    summary["accepted_by_engine_stage"] = {
        str(stage): sum(
            1 for row in inverse_rows if row.get("accepted") and int(row.get("engine_stage", 0)) == stage
        )
        for stage in (0, 1, 2)
    }

    now = datetime.now(timezone.utc)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 3,
        "generated_at_utc": now.isoformat(),
        "model_label": config["model_label"],
        "aero_model_version": config["aero_model_version"],
        "source_kind": "local_pipeline_test",
        "statshark_new_calculation_performed_this_run": False,
        "statshark_reference_time_series_used": False,
        "source_files": source_files,
        "source_model_labels": dict(source_labels),
        "filter_settings": settings.to_dict(),
        "smoothing": {
            "window_widths_s": windows,
            "primary_window_s": primary_window_s,
            "polynomial_order": int(smoothing["polynomial_order"]),
            "boundary_policy": "separate powered/coast and engine-stage segments; no cross-boundary smoothing",
            "saved_sample_fields": [
                "raw_speed_mps",
                "smoothed_speed_mps",
                "speed_derivative_mps2",
                "speed_smoothing_residual_mps",
                "observed_drag_n",
                "observed_cda_m2",
                "speed_derivative_mps2_by_window",
                "observed_cda_m2_by_window",
            ],
        },
        "summary": summary,
        "window_sensitivity": _window_summary(
            inverse_rows,
            primary_window_s,
            windows[1] if len(windows) > 1 else windows[0],
        ),
        "threshold_sensitivity": _sensitivity_counts(normalized_rows, settings, config),
        "data_readiness": {
            "powered_time_series": summary["accepted_by_power_state"]["powered"] > 0,
            "coast_time_series": summary["accepted_by_power_state"]["coast"] > 0,
            "powered_coast_overlap_is_reference_evidence": False,
            "reason": "all input trajectories are local H2 model outputs, not StatShark raw time series",
        },
    }
    (args.output_dir / "sample_manifest.json").write_text(
        json.dumps(_json_safe(manifest), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "inverse_cda_samples.json").write_text(
        json.dumps(_json_safe({
            "schema_version": 3,
            "generated_at_utc": now.isoformat(),
            "model_label": config["model_label"],
            "source_kind": "local_pipeline_test",
            "filter_settings": settings.to_dict(),
            "samples": inverse_rows,
        }), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"rows={len(inverse_rows)} accepted={summary['accepted_rows']} "
        f"powered={summary['accepted_by_power_state']['powered']} "
        f"coast={summary['accepted_by_power_state']['coast']}"
    )
    print(f"written: {args.output_dir / 'sample_manifest.json'}")
    print(f"written: {args.output_dir / 'inverse_cda_samples.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
