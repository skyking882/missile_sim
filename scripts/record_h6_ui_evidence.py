#!/usr/bin/env python3
"""Materialize the H6 StatShark UI audit evidence.

The browser run exposed rendered plots and client-side diagnostics, but the
available browser capability did not expose response bodies or network
payloads.  This script therefore records request reconstructions and visible
outcomes without inventing backend time-series arrays.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aim120_model.h6_utils import sha256_file, utc_now_iso, write_json  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "h6_fin_dynamics"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "statshark_h6_fin_dynamics"
SNAPSHOT_DIR = RAW_DIR / "model_snapshots"
CAPTURE_DATE = "2026-08-12"
TIME_STEP_S = 0.02


def request_reconstruction(
    start_speed_kmh: float,
    launch_altitude_m: float,
    target_speed_kmh: float,
    target_altitude_m: float,
    distance_m: float,
    target_azimuth_deg: float,
) -> Dict[str, Any]:
    """Record the UI fields that map to the public calculator request.

    Custom server IDs and the exact serialized body are intentionally marked
    unavailable: they were not exposed by the permitted browser capability.
    """

    return {
        "capture_kind": "ui_request_fields_reconstructed",
        "raw_request_body_exported": False,
        "custom_server_ids_exported": False,
        "fields": {
            "Missiles": "selected UI models; exact custom server IDs unavailable",
            "StartSpeed": start_speed_kmh,
            "LaunchAltitude": launch_altitude_m,
            "LaunchAngle": 0.0,
            "ClosureRate": target_speed_kmh,
            "InitialTargetDistance": distance_m,
            "TargetAltitude": target_altitude_m,
            "Timestep": TIME_STEP_S,
            "LaunchYaw": 0.0,
            "TargetAzimuth": target_azimuth_deg,
            "TargetCourse": 0.0,
            "TargetConstantGTurn": 0.0,
            "TargetVerticalCourse": 0.0,
            "version": "local UI selection; exact serialized version field unavailable",
        },
        "units": {
            "StartSpeed": "km/h",
            "ClosureRate": "km/h",
            "LaunchAltitude": "m",
            "TargetAltitude": "m",
            "InitialTargetDistance": "m",
            "TargetAzimuth": "deg",
            "Timestep": "s",
        },
    }


def scenario(
    start_speed_kmh: float,
    launch_altitude_m: float,
    target_speed_kmh: float,
    target_altitude_m: float,
    distance_m: float,
    target_azimuth_deg: float,
) -> Dict[str, Any]:
    return request_reconstruction(
        start_speed_kmh=start_speed_kmh,
        launch_altitude_m=launch_altitude_m,
        target_speed_kmh=target_speed_kmh,
        target_altitude_m=target_altitude_m,
        distance_m=distance_m,
        target_azimuth_deg=target_azimuth_deg,
    )


def plot_evidence(
    visible_status: str,
    summary: str,
    **facts: Any,
) -> Dict[str, Any]:
    return {
        "visible_status": visible_status,
        "summary": summary,
        "facts": facts,
    }


FRONTEND_LOGS = [
    "TypeError: t.toLowerCase is not a function at filterCustomMissiles",
    "Data for missile 'undefined' not found in the response.",
]


def action(
    action_id: str,
    batch: str,
    models: Sequence[str],
    request: Mapping[str, Any],
    evidence: Mapping[str, Any],
    notes: str,
    frontend_logs: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    return {
        "action_id": action_id,
        "batch": batch,
        "case_id": action_id,
        "model_id": batch,
        "models": list(models),
        "status": "ui_only_no_raw_response",
        "ui_calculate_submitted": True,
        "request": copy.deepcopy(dict(request)),
        "response": {
            "raw_response_available": False,
            "raw_response_retained": False,
            "expected_top_level_schema": ["missileIds", "results"],
            "expected_result_arrays": [
                "times",
                "missileX",
                "missileY",
                "missileZ",
                "missileSpeedKmh/missileSpeedMs",
                "angle",
                "yaw",
                "Cd",
                "aCmd",
                "aCmdYaw",
            ],
            "capture_limitation": (
                "The permitted browser capability exposed rendered DOM/SVG and "
                "client diagnostics, not response bodies or network captures."
            ),
        },
        "visible_plot_evidence": copy.deepcopy(dict(evidence)),
        "frontend_logs": list(frontend_logs or []),
        "notes": notes,
    }


def build_actions() -> List[Dict[str, Any]]:
    f1_formal = scenario(1775.0, 3000.0, 0.0, 3000.0, 50000.0, 15.0)
    f1_lock = scenario(1775.0, 3000.0, 0.0, 3000.0, 10000.0, 15.0)
    f1_responsive = scenario(1775.0, 3000.0, 1200.0, 3000.0, 10000.0, 15.0)
    f1_vertical = scenario(1620.0, 8000.0, 0.0, 12000.0, 30000.0, 0.0)
    f2_responsive = scenario(1775.0, 3000.0, 1200.0, 3000.0, 10000.0, 25.0)
    f2_small_az = scenario(1775.0, 3000.0, 1200.0, 3000.0, 10000.0, 5.0)
    f3_formal = scenario(1620.0, 10000.0, 0.0, 10000.0, 50000.0, 25.0)
    f4_formal = scenario(1775.0, 3000.0, 0.0, 3000.0, 50000.0, -25.0)
    f5_formal = scenario(1775.0, 3000.0, 0.0, 3000.0, 50000.0, 25.0)
    f6_formal = scenario(1620.0, 8000.0, 0.0, 12000.0, 30000.0, 0.0)

    axis_models = ["H6_AX0", "H6_AX_H", "H6_AX_V", "H6_AX_HV"]
    force_models = ["H6_F000", "H6_F050", "H6_F100", "H6_F150"]
    f2_models = force_models + ["H6_L050", "H6_L150", "H6_W050", "H6_W200"]
    f3_models = force_models + ["H6_L100", "H6_W100"]
    f4_models = ["H6_F000", "H6_F100", "H6_A050", "H6_A100", "H6_A150"]
    f5_models = ["H6_BODY_F000", "H6_BODY_F050", "H6_BODY_F100", "H6_BODY_F150"]
    f6_models = ["H6_BODY_F000", "H6_BODY_F100", "H6_PITCH_W100"]

    return [
        action(
            "H6_UI_001",
            "F1_formal_high_q_az15",
            axis_models,
            f1_formal,
            plot_evidence(
                "success_toast_and_mapped_plot",
                "Top/yaw rendered paths were horizontal at zero lateral displacement for all four clones.",
                top_paths_horizontal_zero=True,
                yaw_paths_horizontal_zero=True,
                series_count=4,
            ),
            "First formal F1 run; H6_AX0 still had the pre-correction high-loft state at this point.",
        ),
        action(
            "H6_UI_002",
            "F1_lock_diagnostic_10km",
            axis_models,
            f1_lock,
            plot_evidence(
                "success_toast_and_mapped_plot",
                "Shorter lock diagnostic still rendered zero lateral/top-view paths.",
                top_paths_horizontal_zero=True,
                yaw_paths_horizontal_zero=True,
                series_count=4,
            ),
            "Repeated with a 10 km target distance to increase visible terminal response.",
        ),
        action(
            "H6_UI_003",
            "F1_lock_diagnostic_repeat",
            axis_models,
            f1_lock,
            plot_evidence(
                "success_toast_and_mapped_plot",
                "Repeat of the lock diagnostic reproduced the zero lateral/top-view result.",
                top_paths_horizontal_zero=True,
                yaw_paths_horizontal_zero=True,
                series_count=4,
            ),
            "Failure/empty-result repetition is retained as evidence rather than deduplicated away.",
        ),
        action(
            "H6_UI_004",
            "F1_responsive_high_q_az15",
            axis_models,
            f1_responsive,
            plot_evidence(
                "success_toast_and_mapped_plot",
                "aCmdYaw became nonzero for AX_H/AX_V/AX_HV, while rendered attitude/top paths remained zero.",
                aCmdYaw_nonzero_for=["H6_AX_H", "H6_AX_V", "H6_AX_HV"],
                aCmdYaw_AX0_zero_on_pre_fix_model=True,
                attitude_paths_zero=True,
                command_to_attitude_disconnect=True,
            ),
            "Responsive diagnosis was run before the H6_AX0 high-loft UI correction was applied.",
        ),
        action(
            "H6_UI_005",
            "F1_vertical_axis_diagnostic",
            axis_models,
            f1_vertical,
            plot_evidence(
                "success_toast_and_mapped_plot",
                "Vertical-axis diagnostic gave identical aCmd and Cd curves across AX0/H/V/HV.",
                aCmd_pitch_identical_across_axis_clones=True,
                Cd_identical_across_axis_clones=True,
                axis_mapping_unresolved=True,
                attitude_paths_not_excited=True,
            ),
            "The finsAoa axis intervention did not create an identifiable vertical/horizontal response split.",
        ),
        action(
            "H6_UI_006",
            "F2_responsive_high_q_az25",
            f2_models,
            f2_responsive,
            plot_evidence(
                "success_toast_and_mapped_plot",
                "All eight clone top/yaw paths were horizontal at zero; aCmdYaw was nonzero and effectively identical.",
                top_paths_horizontal_zero=True,
                yaw_paths_horizontal_zero=True,
                aCmdYaw_nonzero=True,
                aCmdYaw_identical_across_models=True,
                command_to_attitude_disconnect=True,
            ),
            "Responsive F2 intervention used to test force/arm/damping sensitivity before formal fit.",
        ),
        action(
            "H6_UI_007",
            "F2_responsive_high_q_az5",
            f2_models,
            f2_small_az,
            plot_evidence(
                "success_toast_and_mapped_plot",
                "Smaller target azimuth reproduced horizontal zero top-view paths.",
                top_paths_horizontal_zero=True,
                yaw_paths_horizontal_zero=True,
                command_sensitivity_not_observed_in_attitude=True,
            ),
            "Small-angle repeat checks whether the zero-path result is only a large-angle saturation artifact.",
        ),
        action(
            "H6_UI_008",
            "F3_formal_low_q_az25",
            f3_models,
            f3_formal,
            plot_evidence(
                "success_toast_then_empty_plot",
                "The UI reported calculation success, then the chart mapping emitted a missing undefined-missile warning and no usable plot.",
                data_for_missile_undefined=True,
                mapped_plot_available=False,
                raw_arrays_available=False,
            ),
            "Preserved as a schema/mapping anomaly; it is not treated as a physical zero response.",
            FRONTEND_LOGS,
        ),
        action(
            "H6_UI_009",
            "F4_formal_high_q_az_minus25",
            f4_models,
            f4_formal,
            plot_evidence(
                "submitted_no_mapped_plot",
                "Calculate was submitted; after the observation window no mapped plot or durable success evidence was available.",
                mapped_plot_available=False,
                raw_arrays_available=False,
            ),
            "The absent mapped plot is recorded as an observation boundary, not as a backend failure claim.",
            FRONTEND_LOGS,
        ),
        action(
            "H6_UI_010",
            "F5_formal_body_coupled_high_q",
            f5_models,
            f5_formal,
            plot_evidence(
                "submitted_no_mapped_plot",
                "Calculate was submitted; no mapped body-coupled plot was available for extraction.",
                mapped_plot_available=False,
                raw_arrays_available=False,
            ),
            "Body-coupled rows cannot be used for force/drag fitting without response arrays.",
            FRONTEND_LOGS,
        ),
        action(
            "H6_UI_011",
            "F6_formal_holdout",
            f6_models,
            f6_formal,
            plot_evidence(
                "submitted_no_mapped_plot",
                "The F6 holdout was submitted, but its mapped side plot remained unavailable.",
                mapped_plot_available=False,
                raw_arrays_available=False,
                holdout_used_for_refit=False,
            ),
            "F6 is retained as a holdout attempt and was not used for refitting.",
            FRONTEND_LOGS,
        ),
        action(
            "H6_UI_012",
            "control_standard_AIM120A_vs_H6_F150",
            ["AIM-120A", "H6_F150"],
            f2_responsive,
            plot_evidence(
                "submitted_no_mapped_plot",
                "A standard AIM-120A control plus H6_F150 was submitted, but no mapped comparison plot was available.",
                control_series_present_in_selection=True,
                mapped_plot_available=False,
                raw_arrays_available=False,
            ),
            "Control run was kept separate from the formal H6 fit and does not repair the missing response schema.",
            FRONTEND_LOGS,
        ),
    ]


def base_snapshot(model_id: str) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "model_id": model_id,
        "provenance": {
            "source": "StatShark custom-missile UI save/edit/save/reopen",
            "capture_date": CAPTURE_DATE,
            "ui_reopened": True,
            "raw_backend_payload_exported": False,
            "readback_status": "verified_ui_values",
            "note": "This is a UI field snapshot, not a raw server model object.",
        },
        "units": {
            "mass": "kg",
            "caliber": "m",
            "length": "m",
            "area_multiplier": "dimensionless",
            "angle": "deg in UI readback snapshot",
            "fin_accel": "g",
            "arm": "m",
            "wdk": "dimensionless",
            "speed": "m/s",
            "distance": "m",
            "time": "s",
        },
        "basic": {
            "mass_kg": 147.87,
            "caliber_m": 0.1778,
            "length_m": 3.66,
            "wingAreaMult": 1.275,
            "version": "local",
        },
        "aerodynamics": {
            "CxK": 1.425,
            "CyMaxAoA": 1.0,
            "tvAng": 0.0,
            "finsLatAccel_g": 42.2579,
            "finsAoaHor_deg": 15.409184,
            "finsAoaVer_deg": 15.409184,
            "distCmStab_m": 0.17500001,
            "WdK": [1.0, 1.0, 1.0],
            "CyK": 0.0,
            "CxAoA": 0.0,
        },
        "engine": {
            "stage_count": 2,
            "stages": [
                {
                    "stage": 1,
                    "pulse_time_s": 1.7,
                    "thrust_n": 0.0,
                    "mass_loss_kg": 0.0,
                    "factor_index": 0,
                    "engine_type": "solid rocket",
                    "activation_conditions_enabled": False,
                },
                {
                    "stage": 2,
                    "pulse_time_s": 5.3,
                    "thrust_n": 0.0,
                    "mass_loss_kg": 0.0,
                    "factor_index": 0,
                    "engine_type": "solid rocket",
                    "activation_conditions_enabled": False,
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
            "lockRange_m": 16000.0,
            "maxG": 35.0,
            "maxRate_deg_s": 60.0,
            "PN": 4.0,
            "timeout_s": 0.6,
            "proximity_fuze": True,
            "proximity_radius_m": 12.0,
            "high_loft": False,
            "limit_aoa": False,
        },
        "advanced": {
            "pid": [
                {
                    "switch_time_s": 3.4028234663852886e38,
                    "P": 0.0086,
                    "I": 0.0565,
                    "D": 0.00025,
                    "integral_limit": 1.0,
                }
            ],
            "time_to_gain": [[0.0, 1.0]],
            "hit_time_to_gain": [[10.0, 1.0], [25.0, 0.8], [50.0, 0.5]],
        },
        "verification": {
            "model_saved": True,
            "model_reopened": True,
            "ui_custom_search_used": True,
            "unresolved_frontend_mapping_warning": True,
        },
    }


def model_snapshots() -> List[Dict[str, Any]]:
    values: Dict[str, Dict[str, Any]] = {}
    for model_id in [
        "H6_AX0", "H6_AX_H", "H6_AX_V", "H6_AX_HV",
        "H6_F000", "H6_F050", "H6_F100", "H6_F150",
        "H6_L050", "H6_L100", "H6_L150",
        "H6_W050", "H6_W100", "H6_W200",
        "H6_A050", "H6_A100", "H6_A150",
        "H6_BODY_F000", "H6_BODY_F050", "H6_BODY_F100", "H6_BODY_F150",
        "H6_PITCH_W100",
    ]:
        values[model_id] = {}

    values["H6_AX0"].update({"finsLatAccel_g": 0.0})
    values["H6_AX_H"].update({"finsAoaVer_deg": 0.0})
    values["H6_AX_V"].update({"finsAoaHor_deg": 0.0})
    values["H6_AX_HV"].update({})
    for model_id, value in (("H6_F000", 0.0), ("H6_F050", 21.12895), ("H6_F100", 42.2579), ("H6_F150", 63.38685)):
        values[model_id]["finsLatAccel_g"] = value
    values["H6_L050"].update({"distCmStab_m": 0.0875, "requested_distCmStab_m": 0.0875})
    values["H6_L100"].update({"distCmStab_m": 0.17500001, "requested_distCmStab_m": 0.175})
    values["H6_L150"].update({"distCmStab_m": 0.2625, "requested_distCmStab_m": 0.2625})
    values["H6_W050"]["WdK"] = [1.0, 1.0, 0.5]
    values["H6_W100"]["WdK"] = [1.0, 1.0, 1.0]
    values["H6_W200"]["WdK"] = [1.0, 1.0, 2.0]
    for model_id, value in (("H6_A050", 7.704592), ("H6_A100", 15.409184), ("H6_A150", 23.113776)):
        values[model_id].update({
            "finsAoaHor_deg": value,
            "finsAoaVer_deg": value,
            "requested_relevant_finsAoa_deg": value,
            "axis_mapping_status": "unresolved_symmetric_limit_probe",
        })
    values["H6_BODY_F000"].update({"CyK": 2.2, "CxAoA": 9.0, "finsLatAccel_g": 0.0})
    values["H6_BODY_F050"].update({"CyK": 2.2, "CxAoA": 9.0, "finsLatAccel_g": 21.12895})
    values["H6_BODY_F100"].update({"CyK": 2.2, "CxAoA": 9.0, "finsLatAccel_g": 42.2579})
    values["H6_BODY_F150"].update({"CyK": 2.2, "CxAoA": 9.0, "finsLatAccel_g": 63.38685})
    values["H6_PITCH_W100"].update({"WdK": [1.0, 1.0, 1.0]})

    snapshots: List[Dict[str, Any]] = []
    for model_id, updates in values.items():
        item = base_snapshot(model_id)
        for section in ("aerodynamics",):
            for key, value in updates.items():
                if key in item[section]:
                    item[section][key] = value
        item.update({key: value for key, value in updates.items() if key not in item["aerodynamics"]})
        if model_id.startswith("H6_BODY_"):
            item["aerodynamics"]["CyK"] = 2.2
            item["aerodynamics"]["CxAoA"] = 9.0
        if model_id in ("H6_A050", "H6_A100", "H6_A150"):
            item["interpretation"] = {
                "formal_use": "blocked_until_axis_mapping_gate",
                "reason": "UI stores both horizontal and vertical AOA fields at the same observed value in this probe.",
            }
        if model_id == "H6_AX0":
            item["calculation_history_caveat"] = "High-loft was corrected to false and the model was reopened; early F1 actions predate that correction."
        snapshots.append(item)
    return snapshots


def update_phase0_and_manifests(actions: Sequence[Mapping[str, Any]], snapshots: Sequence[Mapping[str, Any]]) -> None:
    phase0_path = OUTPUT_DIR / "phase0_freeze_manifest.json"
    if phase0_path.exists():
        payload = json.loads(phase0_path.read_text(encoding="utf-8"))
        payload["new_statshark_calculate_performed_this_run"] = True
        payload["new_statshark_calculate_authorized"] = True
        payload["authorization"] = {
            "user_texts": ["我授权一切采集", "我允许你执行尽可能多的calculate，不用被plan束缚，直到完成"],
            "planned_formal_actions": 6,
            "additional_diagnostic_actions": len(actions) - 6,
            "calculate_actions_counted": len(actions),
            "failure_empty_retry_counts": True,
            "interpretation": "User expanded authorization to continue Calculate actions until the evidence boundary was reached.",
        }
        payload["status"] = "phase0_frozen_capture_complete"
        write_json(phase0_path, payload)

    ledger = {
        "schema_version": 2,
        "model_label": "local_candidate_H6_fin_plant_v1",
        "authorization_texts": ["我授权一切采集", "我允许你执行尽可能多的calculate，不用被plan束缚，直到完成"],
        "planned_formal_actions": 6,
        "calculate_actions_used": len(actions),
        "calculate_actions_remaining": None,
        "unbounded_until_evidence_boundary": True,
        "failure_empty_retry_counts": True,
        "capture_quality_counts": {
            "mapped_plot_observations": 7,
            "empty_or_unmapped_plot_observations": 5,
            "diagnostic_repeat_actions": 1,
            "frontend_warning_types": 2,
        },
        "status": "capture_complete_at_evidence_boundary",
        "actions": list(actions),
        "note": "The count is the observed UI Calculate sequence in this audit bundle; non-Calculate model edits are excluded.",
    }
    write_json(RAW_DIR / "action_ledger.json", ledger)
    write_json(OUTPUT_DIR / "ui_capture_ledger.json", ledger)

    matrix_path = RAW_DIR / "clone_matrix.json"
    if matrix_path.exists():
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    else:
        matrix = {"schema_version": 1}
    matrix["capture_status"] = "saved_and_reopened_ui_verified"
    matrix["planned_model_count"] = len(matrix.get("models", []))
    matrix["actual_saved_reopened_model_count"] = len(snapshots)
    matrix["additional_holdout_model"] = "H6_PITCH_W100"
    matrix["axis_mapping_status"] = "unresolved_for_A050_A100_A150_and_not_excited_in_AX_vertical_probe"
    write_json(matrix_path, matrix)

    source_path = OUTPUT_DIR / "source_manifest.json"
    source = json.loads(source_path.read_text(encoding="utf-8")) if source_path.exists() else {}
    source.update({
        "generated_at_utc": utc_now_iso(),
        "raw_backend_bundle": str((RAW_DIR / "formal_capture_bundle.json").resolve()),
        "action_ledger": str((RAW_DIR / "action_ledger.json").resolve()),
        "model_snapshot_dir": str(SNAPSHOT_DIR.resolve()),
        "status": "ui_capture_bundle_complete_raw_backend_arrays_unavailable",
    })
    write_json(source_path, source)


def write_capture_bundle(actions: Sequence[Mapping[str, Any]]) -> Path:
    bundle = {
        "schema_version": 2,
        "generated_at_utc": utc_now_iso(),
        "source_kind": "statshark_ui_rendered_evidence",
        "raw_backend_response_available": False,
        "raw_backend_response_retained": False,
        "raw_response_limitation": (
            "The in-app browser capability allowed UI interaction and rendered DOM/SVG inspection, "
            "but did not expose network response bodies. No direct API or hidden transport was used."
        ),
        "expected_backend_schema_observed_from_client": {
            "top_level": ["missileIds", "results"],
            "result_arrays": [
                "times", "missileX", "missileY", "missileZ", "targetX", "targetY", "targetZ",
                "missileSpeedKmh/missileSpeedMs", "machNumber", "angle", "yaw", "Cd", "aCmd", "aCmdYaw",
            ],
        },
        "captures": list(actions),
        "status": "ui_plot_only_or_frontend_mapping_blocked",
    }
    path = RAW_DIR / "formal_capture_bundle.json"
    write_json(path, bundle)
    return path


def write_snapshots(snapshots: Sequence[Mapping[str, Any]]) -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    for snapshot in snapshots:
        model_id = str(snapshot["model_id"])
        write_json(SNAPSHOT_DIR / (model_id + ".json"), snapshot)
    write_json(RAW_DIR / "model_snapshot_manifest.json", {
        "schema_version": 1,
        "generated_at_utc": utc_now_iso(),
        "model_count": len(snapshots),
        "saved_reopened_verified": True,
        "raw_backend_payload_exported": False,
        "models": [str(item["model_id"]) for item in snapshots],
        "sha256_by_model": {
            str(item["model_id"]): sha256_file(SNAPSHOT_DIR / (str(item["model_id"]) + ".json"))
            for item in snapshots
        },
    })


def write_evidence_markdown(actions: Sequence[Mapping[str, Any]], snapshots: Sequence[Mapping[str, Any]]) -> Path:
    path = OUTPUT_DIR / "ui_capture_evidence.md"
    lines = [
        "# H6 StatShark UI Capture Evidence",
        "",
        "- Capture date: `2026-08-12`",
        "- Saved/reopened model snapshots: `{}`".format(len(snapshots)),
        "- Counted UI Calculate submissions: `{}`".format(len(actions)),
        "- Raw backend arrays exported: `false`",
        "- Evidence status: `ui_plot_only_or_frontend_mapping_blocked`",
        "",
        "## Boundary",
        "",
        "The available browser capability permitted normal UI interaction and rendered SVG/DOM inspection. It did not expose network response bodies or the exact serialized custom-missile request. The bundle therefore retains UI field reconstructions, visible plot facts, and frontend diagnostics without fabricating `times`/trajectory arrays.",
        "",
        "## Model readback",
        "",
        "All 22 snapshots under `data/raw/statshark_h6_fin_dynamics/model_snapshots/` were saved in the custom UI and reopened for field verification. `H6_A050/A100/A150` remain axis-mapping probes, not formal deflection-limit measurements.",
        "",
        "## Calculate ledger",
        "",
        "| ID | Batch | UI observation | Backend arrays |",
        "|---|---|---|---|",
    ]
    for item in actions:
        evidence = item["visible_plot_evidence"]
        lines.append("| `{}` | `{}` | {} | unavailable |".format(
            item["action_id"], item["batch"], evidence["summary"].replace("|", "\\|")))
    lines.extend([
        "",
        "## Repeated diagnostics",
        "",
        "- F1 lock and responsive runs showed nonzero command labels in some interventions while actual rendered attitude/top paths stayed at zero.",
        "- The vertical-axis diagnostic produced identical `aCmd` and `Cd` curves for AX0/AX_H/AX_V/AX_HV.",
        "- F3 produced the frontend warning `Data for missile 'undefined' not found in the response.` after a calculation-success indication.",
        "- The custom-model filter repeatedly logged `TypeError: t.toLowerCase is not a function at filterCustomMissiles`.",
        "",
        "## Conservative conclusion",
        "",
        "The H6 formal force/moment/drag fit is blocked by response-schema/mapping evidence and insufficient measured attitude excitation. The local synthetic identifiability tests pass, but they do not validate StatShark or replace the missing raw arrays.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    actions = build_actions()
    snapshots = model_snapshots()
    update_phase0_and_manifests(actions, snapshots)
    bundle_path = write_capture_bundle(actions)
    write_snapshots(snapshots)
    evidence_path = write_evidence_markdown(actions, snapshots)
    result = {
        "status": "ui_capture_bundle_complete_raw_backend_arrays_unavailable",
        "calculate_actions": len(actions),
        "model_snapshots": len(snapshots),
        "bundle": str(bundle_path.resolve()),
        "evidence": str(evidence_path.resolve()),
    }
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
