#!/usr/bin/env python3
"""Freeze the zero-Calculate H6 recovery setup.

This creates a new recovery evidence tree and deliberately does not call the
StatShark calculator.  The current in-app browser exposes UI/DOM evidence and
page assets, but no request/response-body capture capability, so the R0 gate
is recorded as blocked rather than silently promoted to an R1-ready state.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GAME_ROOT = PROJECT_ROOT.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aim120_model.fin_force_inverse import validate_response_arrays  # noqa: E402
from aim120_model.h6_utils import utc_now_iso, write_json  # noqa: E402


RAW_DIR = PROJECT_ROOT / "data" / "raw" / "statshark_h6_fin_dynamics_recovery"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "h6_fin_dynamics_recovery"
OLD_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "statshark_h6_fin_dynamics"
OLD_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "h6_fin_dynamics"


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_observation(path: Path) -> Dict[str, Any]:
    exists = path.exists() and path.is_file()
    return {
        "path": str(path.resolve()),
        "exists": exists,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size if exists else None,
    }


def write_once(path: Path, payload: Any) -> Path:
    """Write a new artifact without overwriting a previous recovery artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    candidate = path
    index = 0
    while candidate.exists():
        index += 1
        candidate = path.with_name("{}_retry{:02d}{}".format(path.stem, index, path.suffix))
    write_json(candidate, payload)
    return candidate


def snapshot() -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "model_id": "H6R_BODY_C0_F100",
        "source": "StatShark UI standard-clone -> save -> reopen readback",
        "capture_kind": "ui_snapshot_only",
        "captured_utc": utc_now_iso(),
        "raw_request_payload_available": False,
        "raw_response_payload_available": False,
        "exact_custom_server_id_available": False,
        "verification": {
            "created_in_ui": True,
            "appeared_in_selected_missiles": True,
            "reopened_in_edit_dialog": True,
            "all_changed_tabs_read_back": True,
            "calculate_performed_during_r0": False,
        },
        "basic": {
            "name": "H6R_BODY_C0_F100",
            "version": "local",
            "mass_kg": 147.87,
            "caliber_m": 0.1778,
            "length_m": 3.66,
            "wingAreaMult": 1.275,
        },
        "aerodynamics": {
            "CxK": 1.425,
            "CyK": 2.2,
            "CxAoA": 0.0,
            "CyMaxAoA": 1.0,
            "finsLatAccel_g": 42.2579,
            "finsAoaHor_deg": 15.409184238027873,
            "finsAoaVer_deg": 15.409184238027873,
            "distCmStab_m": 0.17500001,
            "tvAng_deg": 0.0,
            "WdK": [1.0, 1.0, 1.0],
            "axis_mapping_status": "not_yet_tested_in_H6R",
        },
        "engine": {
            "stage_count": 2,
            "all_activation_conditions_enabled": False,
            "stages": [
                {
                    "stage": 1,
                    "pulse_count": 1,
                    "pulse_time_s": 1.7,
                    "thrust_n": 0.0,
                    "mass_loss_kg": 0.0,
                    "factor_index": 0,
                    "engine_type": "solid rocket",
                },
                {
                    "stage": 2,
                    "pulse_count": 1,
                    "pulse_time_s": 5.3,
                    "thrust_n": 0.0,
                    "mass_loss_kg": 0.0,
                    "factor_index": 0,
                    "engine_type": "solid rocket",
                },
            ],
            "final_mass_kg": 147.87,
        },
        "performance": {
            "maxSpeed_mps": 2500.0,
            "maxDistance_m": 100000.0,
            "timeLife_s": 15.0,
        },
        "guidance": {
            "enabled": True,
            "type": "Radar",
            "lockRange_m": 100000.0,
            "maxG": 35.0,
            "maxRate_deg_s": 60.0,
            "PN": 4.0,
            "timeout_s": 0.6,
            "limit_aoa": False,
            "proximity_fuze": True,
            "proximity_radius_m": 12.0,
            "high_loft": False,
        },
        "advanced": {
            "pid": {
                "switch_time_s": 3.4028234663852886e38,
                "P": 0.0086,
                "I": 0.0565,
                "D": 0.00025,
                "integral_limit": 1.0,
            },
            "time_to_gain": [[0.0, 1.0]],
            "hit_time_to_gain": [[10.0, 1.0], [25.0, 0.8], [50.0, 0.5]],
        },
        "interpretation": {
            "CyK_2_2": "StatShark clone/schema readback used to restore body-lift observability; not claimed as AIM-120A source truth.",
            "CxAoA_0": "R1-R5 recovery configuration; CxAoA=9 is reserved for R6 energy pairing.",
            "lockRange_100000": "H6R excitation configuration to remove the old 16 km lock-range confound; not a real seeker-range claim.",
        },
    }


def synthetic_schema_precheck() -> Dict[str, Any]:
    count = 51
    times = [round(index * 0.02, 10) for index in range(count)]
    response = {
        "times": times,
        "missileX": [float(index) for index in range(count)],
        "missileY": [3000.0 for _ in range(count)],
        "missileZ": [0.0 for _ in range(count)],
        "missileSpeedMs": [493.0555556 for _ in range(count)],
        "angle": [0.0 for _ in range(count)],
        "yaw": [0.0 for _ in range(count)],
        "Cd": [0.018 for _ in range(count)],
        "aCmd": [0.0 for _ in range(count)],
        "aCmdYaw": [0.0 for _ in range(count)],
    }
    report = validate_response_arrays(response, model_id="synthetic_schema_fixture")
    return {
        "kind": "synthetic_test",
        "status": "pass" if report["status"] == "pass" else "fail",
        "report": report,
        "fixture_not_statshark_capture": True,
        "required_arrays": ["times", "missileX", "missileY", "missileZ", "missileSpeedMs", "angle", "yaw"],
        "optional_arrays_checked": ["Cd", "aCmd", "aCmdYaw"],
        "sample_count": count,
        "sample_dt_s": 0.02,
    }


def write_r0_report(payload: Mapping[str, Any]) -> Path:
    path = OUTPUT_DIR / "H6R_R0_REPORT.md"
    lines = [
        "# H6R R0 Recovery Setup Report",
        "",
        "- Status: `{}`".format(payload["status"]),
        "- Calculate actions during R0: `0`",
        "- Old H6 artifacts modified: `false`",
        "- War Thunder game files modified: `false`",
        "",
        "## Gate results",
        "",
        "| Gate | Result | Evidence |",
        "|---|---|---|",
    ]
    for name, item in payload["gates"].items():
        lines.append("| `{}` | `{}` | {} |".format(name, item["status"], item.get("detail", "")))
    lines.extend([
        "",
        "## H6R model smoke snapshot",
        "",
        "`H6R_BODY_C0_F100` was rebuilt from AIM-120A and reopened in the same StatShark page context. The readback confirms `CyK=2.2`, `CxAoA=0`, zero thrust/mass loss, `timeLife=15 s`, `lockRange=100000 m`, and high-loft off.",
        "",
        "## Capture boundary",
        "",
        "The connected browser advertises DOM/SVG, console-log, and page-asset inspection, but no Network/response-body export capability. Consequently R1 must not be started: a UI-only Calculate would reproduce the old failure mode and would not satisfy H6R provenance.",
        "",
        "## New recovery paths",
        "",
        "- `data/raw/statshark_h6_fin_dynamics_recovery/session_manifest.json`",
        "- `data/raw/statshark_h6_fin_dynamics_recovery/calculate_ledger.json`",
        "- `data/raw/statshark_h6_fin_dynamics_recovery/model_snapshots/H6R_BODY_C0_F100.json`",
        "- `outputs/h6_fin_dynamics_recovery/schema_report.json`",
        "- `outputs/h6_fin_dynamics_recovery/H6R_R0_REPORT.md`",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "model_snapshots").mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "requests").mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "responses").mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "network_evidence").mkdir(parents=True, exist_ok=True)

    model = snapshot()
    schema = synthetic_schema_precheck()
    model_path = write_once(RAW_DIR / "model_snapshots" / "H6R_BODY_C0_F100.json", model)
    schema_path = write_once(OUTPUT_DIR / "schema_report.json", {
        "schema_version": 1,
        "generated_at_utc": utc_now_iso(),
        "source_kind": "synthetic_schema_precheck",
        "status": schema["status"],
        "raw_statshark_response_present": False,
        "normalized_rows": 0,
        "precheck": schema,
        "note": "This validates the local parser contract only; it is not a StatShark response.",
    })

    old_observations = {
        "old_h6_output_report": file_observation(OLD_OUTPUT_DIR / "H6_FIN_DYNAMICS_REPORT.md"),
        "old_h6_schema_report": file_observation(OLD_OUTPUT_DIR / "schema_report.json"),
        "old_h6_capture_bundle": file_observation(OLD_RAW_DIR / "formal_capture_bundle.json"),
        "old_h6_snapshot_manifest": file_observation(OLD_RAW_DIR / "model_snapshot_manifest.json"),
    }
    assets = {
        "page_url": "https://statshark.net/missilecalculator",
        "build_assets_observed": [
            "main-VM5HRLQY.js",
            "chunk-BlND7Q1o.js",
            "polyfills-OP2ZPRWU.js",
            "styles-2SGSLABJ.css",
        ],
        "page_asset_inventory_summary": {
            "script": 31,
            "font": 2,
            "image": 19,
            "stylesheet": 1,
            "inline_svg": 15,
            "total": 64,
        },
        "hashes": "not computed; pageAssets exposed observed URLs/build names only",
    }
    gates = {
        "fixed_browser_environment": {
            "status": "pass_with_profile_label_caveat",
            "detail": "Codex In-app Browser, same page origin and current session; profile name is not exposed.",
        },
        "network_can_identify_CalcMissileRange": {
            "status": "blocked_capability_unavailable",
            "detail": "Available tab capabilities: pageAssets only; no Network/request/response capture surface.",
        },
        "request_response_save_rehearsal_without_send": {
            "status": "blocked_not_rehearsed",
            "detail": "No raw transport capture surface exists in this browser connection.",
        },
        "new_directory_writable_and_old_H6_untouched": {
            "status": "pass",
            "detail": "New recovery tree created; old H6 hashes observed and no old file was written.",
        },
        "local_synthetic_schema_precheck": {
            "status": schema["status"],
            "detail": "51-sample synthetic response passed required-array validation.",
        },
        "minimal_clone_save_reopen": {
            "status": "pass",
            "detail": "H6R_BODY_C0_F100 appeared in selected missiles and all changed tabs were read back.",
        },
    }
    status = "h6r_blocked_capture_setup"
    session = {
        "schema_version": 1,
        "session_label": "H6R_attempt_01_R0",
        "generated_at_utc": utc_now_iso(),
        "status": status,
        "calculate_actions_performed": 0,
        "browser": {
            "name": "Codex In-app Browser",
            "family": "iab",
            "profile_label": "current connected session; human profile label unavailable",
            "origin": "https://statshark.net",
            "calculator_url": "https://statshark.net/missilecalculator",
            "cookies_saved": False,
            "authorization_headers_saved": False,
            "unrelated_storage_saved": False,
        },
        "page_assets": assets,
        "capture_capabilities": {
            "dom_snapshot": True,
            "rendered_svg": True,
            "console_logs": True,
            "page_assets": True,
            "network_request_body": False,
            "network_response_body": False,
            "http_status_capture": False,
        },
        "old_h6_preservation": old_observations,
        "gates": gates,
        "model_snapshot": str(model_path.resolve()),
        "schema_precheck": str(schema_path.resolve()),
        "interpretation": "R0 stops before R1 because raw CalcMissileRange provenance cannot be captured in the available browser surface.",
    }
    session_path = write_once(RAW_DIR / "session_manifest.json", session)
    ledger_path = write_once(RAW_DIR / "calculate_ledger.json", {
        "schema_version": 1,
        "session_label": "H6R_attempt_01_R0",
        "authorization": {
            "user_texts": [
                "我授权一切采集",
                "我允许你执行尽可能多的calculate，不用被plan束缚，直到完成",
            ],
            "scope_interpretation": "unlimited Calculate is available after the current plan's prerequisite capture gates pass; R0 itself forbids Calculate.",
        },
        "calculate_actions_used": 0,
        "calculate_actions_remaining": None,
        "status": status,
        "actions": [],
        "stop_rule": "R0 capture setup failed; do not execute R1-R7 through UI-only Calculate.",
    })
    report_path = write_r0_report({"status": status, "gates": gates})
    write_json(RAW_DIR / "README.json", {
        "status": status,
        "raw_request_response_dirs": ["requests", "responses"],
        "network_evidence_dir": "network_evidence",
        "calculate_ledger": str(ledger_path.resolve()),
        "note": "Empty requests/responses directories are intentional because R0 stopped before Calculate.",
    })
    print(json.dumps({
        "status": status,
        "calculate_actions": 0,
        "model_snapshot": str(model_path.resolve()),
        "session_manifest": str(session_path.resolve()),
        "schema_report": str(schema_path.resolve()),
        "report": str(report_path.resolve()),
        "old_h6_modified": False,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
