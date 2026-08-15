#!/usr/bin/env python3
"""Create the isolated H6 provenance, authorization ledger, and clone plan."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GAME_ROOT = PROJECT_ROOT.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aim120_model.h6_utils import sha256_file, utc_now_iso, write_json  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "h6_fin_dynamics"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "statshark_h6_fin_dynamics"


def file_record(label: str, path: Path) -> Dict[str, Any]:
    path = Path(path)
    exists = path.exists() and path.is_file()
    record: Dict[str, Any] = {
        "label": label,
        "path": str(path.resolve()),
        "exists": exists,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size if exists else None,
        "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat() if exists else None,
    }
    return record


def _statshark_identifiers() -> Dict[str, Any]:
    path = PROJECT_ROOT / "data" / "raw" / "statshark_h5_body_alpha2" / "formal_capture_bundle.json"
    if not path.exists():
        return {"source_url": None, "asset_identifiers": [], "status": "not_available"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"source_url": None, "asset_identifiers": [], "status": "unreadable"}
    identifiers: List[Dict[str, Any]] = []

    def walk(value: Any, prefix: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                path_key = "{}.{}".format(prefix, key) if prefix else str(key)
                if any(token in str(key).lower() for token in ("asset", "chunk", "build", "version")):
                    if not isinstance(child, (dict, list)):
                        identifiers.append({"key": path_key, "value": child})
                walk(child, path_key)
        elif isinstance(value, list):
            for index, child in enumerate(value[:20]):
                walk(child, "{}[{}]".format(prefix, index))

    walk(payload)
    return {
        "source_url": payload.get("source_url"),
        "asset_identifiers": identifiers[:100],
        "status": "observed_from_prior_capture_only",
        "note": "H5 was visible-slider evidence; no H6 backend asset identifiers are claimed yet.",
    }


def clone_matrix() -> Dict[str, Any]:
    nominal = {
        "mass_kg": 147.87,
        "caliber_m": 0.1778,
        "length_m": 3.66,
        "wingAreaMult": 1.275,
        "CxK": 1.425,
        "CyMaxAoA": 1.0,
        "tvAng": 0.0,
        "finsLatAccel": 42.2579,
        "finsAoaHor": 0.268941,
        "finsAoaVer": 0.268941,
        "distCmStab": 0.175,
        "WdK": [1.0, 1.0, 1.0],
        "CyK": 0.0,
        "CxAoA": 0.0,
        "guidance_enabled": True,
        "loft_enabled": False,
        "tvc": 0.0,
        "thrust_n": 0.0,
        "mass_loss_kg": 0.0,
        "time_step_s": 0.02,
        "time_life_s": 15.0,
    }
    def with_updates(model_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        item = dict(nominal)
        item.update(updates)
        item["model_id"] = model_id
        return item

    models: List[Dict[str, Any]] = []
    models.extend([
        with_updates("H6_AX0", {"finsLatAccel": 0.0}),
        with_updates("H6_AX_H", {"finsAoaVer": 0.0}),
        with_updates("H6_AX_V", {"finsAoaHor": 0.0}),
        with_updates("H6_AX_HV", {}),
    ])
    for label, value in (("F000", 0.0), ("F050", 21.12895), ("F100", 42.2579), ("F150", 63.38685)):
        models.append(with_updates("H6_" + label, {"finsLatAccel": value}))
    for label, value in (("L050", 0.0875), ("L100", 0.175), ("L150", 0.2625)):
        models.append(with_updates("H6_" + label, {"distCmStab": value}))
    for label, value in (("W050", 0.5), ("W100", 1.0), ("W200", 2.0)):
        models.append(with_updates("H6_" + label, {"WdK": [value, 1.0, 1.0]}))
    for label, value in (("A050", 7.704592), ("A100", 15.409184), ("A150", 23.113776)):
        models.append(with_updates("H6_" + label, {"relevant_finsAoa_deg": value}))
    for label, value in (("F000", 0.0), ("F050", 21.12895), ("F100", 42.2579), ("F150", 63.38685)):
        models.append(with_updates("H6_BODY_" + label, {"CyK": 2.2, "CxAoA": 9.0, "finsLatAccel": value}))
    return {
        "schema_version": 1,
        "nominal_fields": nominal,
        "models": models,
        "status": "planned_local_clone_matrix_not_saved_to_statshark",
        "save_and_reopen_required": True,
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "model_snapshots").mkdir(parents=True, exist_ok=True)

    frozen_candidates = [
        ("plan5", GAME_ROOT / "plan5.md"),
        ("plan6", GAME_ROOT / "plan6.md"),
        ("h2_dynamics", PROJECT_ROOT / "src" / "aim120_model" / "h2_dynamics.py"),
        ("control", PROJECT_ROOT / "src" / "aim120_model" / "control.py"),
        ("h2_report", PROJECT_ROOT / ".md" / "H2_REPORT.md"),
        ("h3_report", PROJECT_ROOT / ".md" / "H3_LOW_G_DRAG_REPORT.md"),
        ("h4_report", PROJECT_ROOT / ".md" / "H4_GLIDE_DRAG_ENVELOPE_REPORT.md"),
        ("h4_5_report", PROJECT_ROOT / "outputs" / "h4_5_transonic_drag" / "H4_5_TRANSONIC_DRAG_REPORT.md"),
        ("h4_shape_fit", PROJECT_ROOT / "outputs" / "h4_glide_drag" / "cda_knots_fit.json"),
        ("h5_report", PROJECT_ROOT / ".md" / "H5_BODY_ALPHA2_REPORT.md"),
        ("h5_formal_report", PROJECT_ROOT / "outputs" / "h5_body_alpha2" / "H5_BODY_ALPHA2_FORMAL_REPORT.md"),
        ("h5_phase0_manifest", PROJECT_ROOT / "outputs" / "h5_body_alpha2" / "phase0_freeze_manifest.json"),
        ("h5_formal_source_manifest", PROJECT_ROOT / "outputs" / "h5_body_alpha2" / "h5_formal_source_manifest.json"),
        ("h2_convergence", PROJECT_ROOT / "outputs" / "h2" / "convergence_h2.json"),
        ("h2_reference_comparison", PROJECT_ROOT / "outputs" / "h2" / "reference_comparison_h2.json"),
        ("rm10_report", PROJECT_ROOT / ".md" / "RM10_H4_SHAPE_FIT_REPORT.md"),
    ]
    frozen_files = [file_record(label, path) for label, path in frozen_candidates]
    game_container = GAME_ROOT / "aces.vromfs.bin"
    datamine_git_candidates = list(GAME_ROOT.glob("**/.git"))
    datamine_status = {
        "commit": None,
        "checkout_found": bool(datamine_git_candidates),
        "status": "not_found_in_workspace" if not datamine_git_candidates else "checkout_present_but_not_resolved",
        "note": "No standalone datamine checkout was discoverable from the current workspace; no commit is inferred from the game container.",
    }
    manifest = {
        "schema_version": 1,
        "generated_at_utc": utc_now_iso(),
        "model_label": "local_candidate_H6_fin_plant_v1",
        "phase": "phase0_freeze",
        "new_statshark_calculate_performed_this_run": False,
        "new_statshark_calculate_authorized": True,
        "authorization": {
            "user_text": "我授权一切采集",
            "planned_formal_actions": 6,
            "contingency_actions": 0,
            "failure_empty_retry_counts": True,
            "interpretation": "execute the six planned actions in plan6; no unbounded extra retries",
        },
        "game_files_modified": False,
        "prior_outputs_overwritten": False,
        "frozen_files": frozen_files,
        "datamine_provenance": datamine_status,
        "aim120a_source_provenance": {
            "standalone_extracted_file_found": False,
            "source_file_sha256": None,
            "game_container": file_record("aces_vromfs_bin_observation", game_container),
            "note": "The current workspace exposes the game container, not a standalone editable AIM-120A source file; H6 will not modify it.",
        },
        "statshark_asset_provenance": _statshark_identifiers(),
        "evidence_boundaries": {
            "statshark_backend_timeseries": "reserved for raw F1-F6 POST captures",
            "statshark_reference_visible_readout": "prior H5/H4 only; not used as H6 derivative data",
            "synthetic_test": "local analyzer validation only",
        },
        "h6_paths": {
            "output_dir": str(OUTPUT_DIR.resolve()),
            "raw_dir": str(RAW_DIR.resolve()),
        },
        "status": "phase0_frozen_authorized_capture_pending",
    }
    write_json(OUTPUT_DIR / "phase0_freeze_manifest.json", manifest)
    ledger = {
        "schema_version": 1,
        "model_label": "local_candidate_H6_fin_plant_v1",
        "authorized_max_calculate_actions": 6,
        "contingency_actions": 0,
        "calculate_actions_used": 0,
        "calculate_actions_remaining": 6,
        "failure_empty_retry_counts": True,
        "authorization_text": "我授权一切采集",
        "status": "authorized_not_started",
        "actions": [],
    }
    write_json(RAW_DIR / "action_ledger.json", ledger)
    matrix = clone_matrix()
    write_json(RAW_DIR / "clone_matrix.json", matrix)
    write_json(OUTPUT_DIR / "source_manifest.json", {
        "schema_version": 1,
        "generated_at_utc": utc_now_iso(),
        "phase0_manifest": str((OUTPUT_DIR / "phase0_freeze_manifest.json").resolve()),
        "action_ledger": str((RAW_DIR / "action_ledger.json").resolve()),
        "clone_matrix": str((RAW_DIR / "clone_matrix.json").resolve()),
        "raw_backend_bundle": None,
        "status": "authorized_capture_pending",
    })
    (RAW_DIR / "README.md").write_text(
        "# H6 raw capture directory\n\n"
        "Raw F1-F6 request/response pairs belong here. Each Calculate action is\n"
        "counted in `action_ledger.json`; failures and empty results are retained.\n"
        "No raw backend bundle exists yet. The local clone matrix is declarative\n"
        "until each model is saved and reopened in StatShark.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": manifest["status"],
        "phase0_manifest": str((OUTPUT_DIR / "phase0_freeze_manifest.json").resolve()),
        "authorized_actions": ledger["authorized_max_calculate_actions"],
        "models_planned": len(matrix["models"]),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
