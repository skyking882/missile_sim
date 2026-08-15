#!/usr/bin/env python3
"""Assemble a conservative H6 status report from phase outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aim120_model.h6_utils import utc_now_iso  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "h6_fin_dynamics"


def _read(name: str) -> Optional[Dict[str, Any]]:
    path = OUTPUT_DIR / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "H6_FIN_DYNAMICS_REPORT.md")
    args = parser.parse_args()
    phase0 = _read("phase0_freeze_manifest.json")
    schema = _read("schema_report.json")
    force = _read("force_scale_report.json")
    fit = _read("moment_damping_fit.json")
    direction = _read("fin_drag_model_comparison.json")
    holdout = _read("holdout_report.json")
    ledger_path = PROJECT_ROOT / "data" / "raw" / "statshark_h6_fin_dynamics" / "action_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else None
    ui_only = bool(schema and schema.get("status") == "blocked_schema_ui_only")
    if ui_only:
        status = "h6_blocked_schema"
        stop_reason = (
            "F1-F6 and diagnostic Calculate actions were submitted, but the available UI/browser "
            "evidence did not expose raw response arrays or a stable custom-model result mapping. "
            "Rendered command labels did not produce measured attitude excitation."
        )
    elif not schema or schema.get("status", "").startswith("blocked"):
        status = "h6_blocked_schema"
        stop_reason = "No valid F1-F6 backend timeseries is available for H6 identification."
    elif holdout and holdout.get("holdout_pass"):
        status = "h6_pass_projection_only"
        stop_reason = "Frozen effective plant passed the available whole-trajectory holdout."
    elif fit and fit.get("status") == "fit_complete":
        status = "h6_partial_dynamics_pass_drag_blocked"
        stop_reason = "Effective dynamics fit exists, but the full holdout or extra-drag gate is unresolved."
    else:
        status = "h6_blocked_identifiability"
        stop_reason = "The available trajectories do not support a stable effective plant fit."
    lines = [
        "# H6 Fin Dynamics Status Report",
        "",
        "> This is an isolated effective fin-plant report. It does not identify real fin deflection, hinge moment, inertia tensor, or StatShark source code.",
        "",
        "- Generated UTC: `{}`".format(utc_now_iso()),
        "- Model label: `local_candidate_H6_fin_plant_v1`",
        "- Evidence class: `statshark_ui_rendered_evidence`; backend rows would be `statshark_backend_timeseries`; local tests are `synthetic_test`",
        "- Status: `{}`".format(status),
        "- Stop reason: {}".format(stop_reason),
        "- Secondary gates: `h6_blocked_excitation`, `h6_blocked_identifiability`",
        "",
        "## Authorization and capture ledger",
        "",
        "- User authorization recorded: `{}`".format((ledger or {}).get("authorization_texts", (ledger or {}).get("authorization_text", "not found"))),
        "- Planned formal Calculate actions: `{}`".format((ledger or {}).get("planned_formal_actions", (ledger or {}).get("authorized_max_calculate_actions", "unknown"))),
        "- UI Calculate actions counted: `{}`; remaining: `{}`".format((ledger or {}).get("calculate_actions_used", "unknown"), "not applicable after expanded authorization" if (ledger or {}).get("unbounded_until_evidence_boundary") else (ledger or {}).get("calculate_actions_remaining", "unknown")),
        "- Capture quality counts: `{}`".format(json.dumps((ledger or {}).get("capture_quality_counts", {}), ensure_ascii=False)),
        "",
        "## Gate summary",
        "",
        "| Gate | Status |",
        "|---|---|",
        "| Phase 0 provenance freeze | {} |".format((phase0 or {}).get("status", "missing")),
        "| Backend schema | {} |".format((schema or {}).get("status", "not run")),
        "| UI request/plot ledger | {} |".format("captured" if ledger else "missing"),
        "| Raw response arrays | {} |".format("available" if (schema or {}).get("raw_response_available") else "not exported"),
        "| UI command/attitude diagnostic | {} |".format("captured; excitation/attitude gate blocked" if ui_only else (force or {}).get("status", "not run")),
        "| Moment/damping fit | {} |".format("blocked by raw schema" if ui_only else (fit or {}).get("status", "not run")),
        "| Force direction / drag | {} |".format("blocked by raw schema" if ui_only else (direction or {}).get("status", "not run")),
        "| F6 whole-trajectory holdout | {} |".format("submitted; mapped plot unavailable" if ui_only else (holdout or {}).get("status", "not run")),
        "",
        "## Frozen interpretation",
        "",
        "- `aCmd/aCmdYaw` remain excitation labels, not measured fin force.",
        "- Effective fin force is `mass * (trajectory-curvature normal acceleration - declared body normal acceleration)`.",
        "- The initial drag hypothesis is projection-only (`H6-D0`); an independent `K_fin` remains null until the two-trajectory and holdout gates pass.",
        "- F6 must remain a complete holdout and must not be used for refitting.",
        "- The observed `aCmd/aCmdYaw` versus zero/identical rendered attitude paths is a command-to-attitude disconnect, not a fitted zero fin force.",
        "- The repeated `undefined` model warning and `filterCustomMissiles` type error are retained as frontend mapping evidence, not silently classified as physical behavior.",
        "",
        "## Observed batch outcome",
        "",
        "- `F1`: mapped plots were available in the early diagnostics; lateral/top paths stayed at zero while command labels could be nonzero.",
        "- `F2`: force/arm/damping responsive batch retained the same command-to-attitude disconnect; no parameter fit is claimed.",
        "- `F3`: calculation-success indication was followed by an `undefined` missile mapping warning and an unusable chart.",
        "- `F4-F6`: submissions and missing mapped-plot observations are preserved; they are not reinterpreted as physical zero trajectories.",
        "- Control: standard AIM-120A versus H6_F150 was submitted separately; it did not provide a mapped comparison plot.",
        "",
        "## Artifacts",
        "",
        "- `phase0_freeze_manifest.json`",
        "- `source_manifest.json`",
        "- `schema_report.json`",
        "- `ui_capture_ledger.json`",
        "- `ui_capture_evidence.md`",
        "- `h6_normalized_samples.json`",
        "- `model_snapshot_manifest.json`",
        "- `synthetic_identifiability_report.json` (local analyzer only)",
        "- `formal_capture_bundle.json` (UI evidence; no raw arrays)",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "output": str(args.output.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
