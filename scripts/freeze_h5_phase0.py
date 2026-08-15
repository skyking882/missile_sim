#!/usr/bin/env python3
"""Create the read-only H5 Phase 0 freeze manifest."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GAME_ROOT = PROJECT_ROOT.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "h5_body_alpha2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, label: str | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        return {"label": label or str(path), "path": str(resolved), "exists": False}
    stat = resolved.stat()
    return {
        "label": label or str(path),
        "path": str(resolved),
        "exists": True,
        "sha256": sha256_file(resolved),
        "size_bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


def partial_model_snapshot() -> dict[str, Any]:
    required = [
        "CxK", "CyK", "CxAoA", "CyMaxAoA", "wingAreaMult", "caliber", "mass",
        "finsLatAccel", "finsAoaHor", "finsAoaVer", "tvAng", "guidance", "loft",
        "engine force", "engine duration", "mass lost",
    ]
    observed = {
        "mass": {
            "value_kg": 147.87,
            "source": "H4/H4.5 visible custom-model metadata",
        },
        "engine force": {
            "value_n": 0.0,
            "source": "H4/H4.5 visible custom-model metadata",
        },
        "mass lost": {
            "value_kg": 0.0,
            "source": "H4/H4.5 visible custom-model metadata",
        },
        "guidance": {
            "enabled": False,
            "source": "H4/H4.5 visible custom-model metadata",
        },
        "finsLatAccel": {
            "value_g": 0.0,
            "source": "H4/H4.5 visible custom-model metadata",
        },
    }
    missing_fields = [field for field in required if field not in observed]
    return {
        "status": "missing_full_field_snapshot",
        "required_fields": required,
        "observed_partial_fields": observed,
        "missing_fields": missing_fields,
        "source_artifacts": [
            "data/raw/statshark_h4/G1_statshark_visible_slider_20260811.json",
            "data/raw/statshark_h4/G2_statshark_visible_slider_20260811.json",
            "data/raw/statshark_h4/G3_statshark_visible_slider_20260811.json",
            "data/raw/statshark_h4/G4_statshark_visible_slider_20260811.json",
        ],
        "interpretation": "H4/H4.5 metadata is a partial operational boundary, not a full C0/C9/C18 model snapshot.",
    }


def compare_prior_h4_5_freeze() -> dict[str, Any]:
    prior_path = PROJECT_ROOT / "outputs" / "h4_5_transonic_drag" / "phase0_freeze_manifest.json"
    if not prior_path.is_file():
        return {"status": "prior_manifest_missing", "all_unchanged": None, "checks": []}
    payload = json.loads(prior_path.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []
    for item in payload.get("frozen_files", []):
        path = Path(str(item.get("path", "")))
        if path.is_file():
            actual = sha256_file(path)
            checks.append({
                "path": str(path),
                "expected_sha256": item.get("sha256"),
                "actual_sha256": actual,
                "unchanged": actual == item.get("sha256"),
            })
        else:
            checks.append({"path": str(path), "expected_sha256": item.get("sha256"), "actual_sha256": None, "unchanged": False})
    return {
        "prior_manifest": str(prior_path.resolve()),
        "checks": checks,
        "all_unchanged": all(item["unchanged"] for item in checks) if checks else None,
        "status": "checked",
    }


def build_manifest() -> dict[str, Any]:
    frozen_paths: list[tuple[str, Path]] = [
        ("plan4", GAME_ROOT / "plan4.md"),
        ("plan4_5", GAME_ROOT / "plan4.5.md"),
        ("h4_report", PROJECT_ROOT / ".md" / "H4_GLIDE_DRAG_ENVELOPE_REPORT.md"),
        ("h4_5_report", PROJECT_ROOT / "outputs" / "h4_5_transonic_drag" / "H4_5_TRANSONIC_DRAG_REPORT.md"),
        ("h4_knots_fit", PROJECT_ROOT / "outputs" / "h4_glide_drag" / "cda_knots_fit.json"),
        ("h4_5_phase0_manifest", PROJECT_ROOT / "outputs" / "h4_5_transonic_drag" / "phase0_freeze_manifest.json"),
        ("h5_config", PROJECT_ROOT / "configs" / "aim120a_h5_body_alpha2.yaml"),
    ]
    for case_id in ("G1", "G2", "G3", "G4"):
        frozen_paths.append((
            "h4_raw_" + case_id,
            PROJECT_ROOT / "data" / "raw" / "statshark_h4" / (case_id + "_statshark_visible_slider_20260811.json"),
        ))
    records = [file_record(path, label) for label, path in frozen_paths]
    missing = [item["label"] for item in records if not item.get("exists")]
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_label": "local_candidate_H5_body_alpha2_M1p5",
        "new_statshark_calculation_performed_this_run": False,
        "new_statshark_calculation_authorized": False,
        "game_files_modified": False,
        "h4_h4_5_outputs_overwritten": False,
        "frozen_files": records,
        "missing_frozen_files": missing,
        "h4_snapshot": {
            "mach_knots": [1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5],
            "cda_knots_m2": [
                0.023565906388471745,
                0.019197192082694305,
                0.016345883182943697,
                0.014068372157138211,
                0.012615392885219368,
                0.011196041388527839,
                0.010661208195557412,
                0.010238512396048966,
            ],
            "reference_node_status": "prior_reference_checkpoint",
            "m1p5_cda_effective_m2": 0.019197192082694305,
        },
        "current_custom_model_snapshot": partial_model_snapshot(),
        "prior_h4_5_freeze_comparison": compare_prior_h4_5_freeze(),
        "evidence_boundaries": {
            "statshark_visible_readout": "reference only; no new H5 rows this run",
            "synthetic_test": "local identifiability only",
            "prior_reference_checkpoint": "H4 M>=1.2 shape and M1.5 node; not refit",
        },
        "output_dir": str(OUTPUT_DIR.resolve()),
        "status": "phase0_frozen_partial_model_snapshot_gap",
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "phase0_freeze_manifest.json"
    manifest = build_manifest()
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    source_manifest = {
        "schema_version": 1,
        "generated_at_utc": manifest["generated_at_utc"],
        "model_label": manifest["model_label"],
        "statshark_new_calculation_performed_this_run": False,
        "authorization": "local Plan 5 execution only; no new website Calculate authorization",
        "source_kinds": {
            "statshark_reference_visible_readout": "frozen H4 rows only; not H5 formal data",
            "synthetic_test": "local Phase 2 only",
            "prior_reference_checkpoint": "frozen H4 M>=1.2 shape and M1.5 node",
        },
        "input_artifacts": manifest["frozen_files"],
        "current_custom_model_snapshot": manifest["current_custom_model_snapshot"],
        "h5_formal_raw_case_count": 0,
        "status": "phase0_to_phase2_only_no_formal_h5_reference_rows",
    }
    source_path = OUTPUT_DIR / "source_manifest.json"
    source_path.write_text(json.dumps(source_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "source_manifest": str(source_path), "status": "written"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
