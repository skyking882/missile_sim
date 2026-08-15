#!/usr/bin/env python3
"""Create traceable H4 source/cleaning artifacts without inventing references."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
ROOT_DIR = PROJECT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from aim120_model.atmosphere import StandardAtmosphere
from aim120_model.config import load_model_config
from aim120_model.sample_filters import LowGFilterSettings, apply_filter, normalize_sample


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


def _h3_checkpoint() -> dict[str, Any]:
    paths = [
        PROJECT_DIR / ".md" / "H3_LOW_G_DRAG_REPORT.md",
        PROJECT_DIR / "configs" / "aim120a_h3_low_g_drag.yaml",
        PROJECT_DIR / "outputs" / "h3_low_g_drag" / "sample_manifest.json",
        PROJECT_DIR / "outputs" / "h3_low_g_drag" / "lg0_fit_report.json",
        PROJECT_DIR / "outputs" / "h3_low_g_drag" / "model_selection_report.json",
    ]
    artifacts = []
    for path in paths:
        if path.exists():
            artifacts.append({"path": str(path.resolve()), "sha256": _sha256(path), "size_bytes": path.stat().st_size})
    manifest_path = PROJECT_DIR / "outputs" / "h3_low_g_drag" / "sample_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    return {
        "model_label": "local_candidate_H3_low_g_drag",
        "source_kind": "local_pipeline_test",
        "direct_sample_range_mach": {"min": 1.0144726856870405, "max": 3.0225389575417285},
        "reference_time_series_available": False,
        "statshark_reference_time_series_used": False,
        "artifacts": artifacts,
        "sample_manifest_summary": manifest.get("summary", {}),
        "boundary": "H3 is a local pipeline validation only; its curve cannot fill H4 M0.2-1.0 or M3.0-4.5 reference gaps.",
    }


def _excluded_existing_artifacts() -> list[dict[str, Any]]:
    """Record nearby artifacts that are intentionally not H4 glide inputs."""

    path = PROJECT_DIR / "data" / "raw" / "statshark_off_axis_20deg_20260811.json"
    if not path.exists():
        return []
    return [{
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "source_kind": "statshark_reference",
        "inclusion_status": "excluded_from_h4_fit",
        "reason": "single off-axis terminal tooltip with approximate values; no unpowered glide time series or raw velocity history",
    }]


def _load_input(
    path: Path,
    source_kind: str,
    settings: LowGFilterSettings,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    result = payload.get("result", payload) if isinstance(payload, dict) else {}
    raw_rows = result.get("samples", []) if isinstance(result, dict) else []
    if not isinstance(raw_rows, list):
        raise ValueError(f"{path} has no samples list")
    trajectory_id = str(result.get("case_name", path.stem)) if isinstance(result, dict) else path.stem
    artifact = {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "original_filename": path.name,
        "acquisition_time_utc": datetime.now(timezone.utc).isoformat(),
        "source_kind": source_kind,
        "trajectory_id": trajectory_id,
        "source_url_or_artifact": metadata.get("source_url_or_artifact", str(path.resolve())),
        "missile_variant": metadata.get("missile_variant", "AIM-120A"),
        "statshark_version_identifier": metadata.get("statshark_version_identifier"),
        "user_visible_inputs": metadata.get("user_visible_inputs", {}),
        "raw_field_origin": metadata.get("raw_field_origin", "artifact_json_samples"),
    }
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_rows):
        item = dict(raw)
        if "mass_kg" not in item and metadata.get("static_mass_kg") is not None:
            item["mass_kg"] = metadata["static_mass_kg"]
        item["trajectory_id"] = trajectory_id
        item["case_id"] = trajectory_id
        item["source_kind"] = source_kind
        item["source_file"] = str(path.resolve())
        item["source_time_index"] = index
        normalized = normalize_sample(item, settings=settings, default_source_case=trajectory_id, default_source_kind=source_kind)
        normalized["trajectory_id"] = trajectory_id
        normalized["case_id"] = trajectory_id
        normalized["source_file"] = str(path.resolve())
        normalized["source_kind"] = source_kind
        normalized["source_time_index"] = index
        normalized["engine_state"] = "coast" if not normalized.get("powered") else "powered"
        filtered = apply_filter(normalized, settings)
        if source_kind == "statshark_reference" and (
            bool(normalized.get("powered")) or abs(float(normalized.get("thrust_n", 0.0))) > 1.0e-6
        ):
            filtered["accepted"] = False
            filtered.setdefault("rejection_reasons", []).append("powered_or_nonzero_thrust_reference")
        rows.append(filtered)
    return artifact, rows


def _atmosphere_consistency(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    atmosphere = StandardAtmosphere()
    differences: list[float] = []
    records: list[dict[str, Any]] = []
    for row in rows:
        speed = row.get("speed_mps")
        altitude = row.get("altitude_m")
        mach_source = row.get("mach")
        if not all(value is not None and math.isfinite(float(value)) for value in (speed, altitude, mach_source)):
            continue
        sound = atmosphere.sample(float(altitude)).speed_of_sound_mps
        mach_recomputed = float(speed) / sound
        difference = float(mach_source) - mach_recomputed
        differences.append(difference)
        records.append({
            "trajectory_id": row.get("trajectory_id"),
            "time_s": row.get("time_s"),
            "mach_source": float(mach_source),
            "mach_recomputed": mach_recomputed,
            "mach_difference": difference,
        })
    return {
        "model": "standard_atmosphere_existing_local_v1",
        "sample_count": len(records),
        "mean_mach_difference": sum(differences) / len(differences) if differences else None,
        "max_abs_mach_difference": max((abs(value) for value in differences), default=None),
        "records": records,
        "status": "audited" if records else "blocked_missing_reference_samples",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="*", type=Path, default=[])
    parser.add_argument("--source-kind", choices=["statshark_reference", "local_pipeline_test", "synthetic_test"], default="statshark_reference")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs" / "h4_glide_drag")
    parser.add_argument("--lateral-load-threshold-g", type=float, default=2.0)
    parser.add_argument("--alpha-threshold-deg", type=float, default=2.0)
    parser.add_argument("--flight-path-threshold-deg", type=float, default=8.0)
    parser.add_argument("--q-min-pa", type=float, default=1000.0)
    parser.add_argument(
        "--statshark-calculation-performed-this-run",
        action="store_true",
        help="Record that new StatShark calculator cases were authorized and acquired in this run.",
    )
    args = parser.parse_args()

    config = load_model_config(PROJECT_DIR / "configs" / "aim120a_h4_glide_drag_envelope.yaml")
    now = datetime.now(timezone.utc).isoformat()
    settings = LowGFilterSettings(
        lateral_load_threshold_g=args.lateral_load_threshold_g,
        alpha_threshold_deg=args.alpha_threshold_deg,
        flight_path_threshold_deg=args.flight_path_threshold_deg,
        q_min_pa=args.q_min_pa,
    )
    artifacts: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for input_path in args.input:
        artifact, input_rows = _load_input(input_path.resolve(), args.source_kind, settings)
        artifacts.append(artifact)
        rows.extend(input_rows)

    h3 = _h3_checkpoint()
    source_manifest = {
        "schema_version": 4,
        "generated_at_utc": now,
        "model_label": config["model_label"],
        "statshark_new_calculation_performed_this_run": bool(args.statshark_calculation_performed_this_run),
        "filter_settings": settings.to_dict(),
        "reference_source_kind": "statshark_reference",
        "input_artifacts": artifacts,
        "excluded_existing_artifacts": _excluded_existing_artifacts(),
        "reference_trajectory_count": len({row["trajectory_id"] for row in rows if row.get("source_kind") == "statshark_reference"}),
        "reference_samples_count": sum(1 for row in rows if row.get("source_kind") == "statshark_reference"),
        "h3_checkpoint": h3,
        "source_policy": config["source_policy"],
        "status": "ready_for_coverage_audit" if artifacts else "blocked_missing_statshark_reference_artifacts",
    }
    filtered = {
        "schema_version": 4,
        "generated_at_utc": now,
        "model_label": config["model_label"],
        "source_kind_policy": config["source_policy"],
        "filter_settings": settings.to_dict(),
        "rows": rows,
        "status": "reference_samples_present" if rows else "empty_reference_dataset",
    }
    inverse = {
        "schema_version": 4,
        "generated_at_utc": now,
        "model_label": config["model_label"],
        "source_kind": args.source_kind,
        "filter_settings": settings.to_dict(),
        "statshark_reference_samples_available": bool(rows),
        "status": "pending_inverse_diagnostic" if rows else "blocked_missing_statshark_reference_samples",
        "samples": [],
        "note": "Inverse diagnostics are not run on H3 local samples as if they were H4 reference evidence.",
    }
    atmosphere = _atmosphere_consistency(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "source_manifest.json").write_text(json.dumps(_json_safe(source_manifest), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "filtered_samples.json").write_text(json.dumps(_json_safe(filtered), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "inverse_cda_diagnostics.json").write_text(json.dumps(_json_safe(inverse), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "atmosphere_consistency.json").write_text(json.dumps(_json_safe(atmosphere), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"artifacts={len(artifacts)} reference_trajectories={source_manifest['reference_trajectory_count']} "
        f"reference_samples={source_manifest['reference_samples_count']} status={source_manifest['status']}"
    )
    print(f"written: {args.output_dir / 'source_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
