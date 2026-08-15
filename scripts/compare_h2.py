#!/usr/bin/env python3
"""Compare H2 local runs against already recorded approximate references."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from aim120_model.config import find_case, load_cases, load_data_file, load_model_config
from aim120_model.h2_simulator import H2Simulator
from aim120_model.metrics import compare_result_to_reference


def main() -> int:
    config = load_model_config(PROJECT_DIR / "configs" / "aim120a_h2.yaml")
    cases = load_cases(PROJECT_DIR / "configs" / "cases.yaml")
    reference_data = load_data_file(PROJECT_DIR / "data" / "reference_cases.json")
    references = reference_data["cases"]
    simulator = H2Simulator(config)
    comparisons: list[dict[str, object]] = []
    for case in cases:
        result = simulator.run(case)
        reference_key = case.get("reference_case")
        if reference_key is None or reference_key not in references:
            row = {
                "case_name": case["name"],
                "status": "blocker_missing_reference",
                "reason": "No existing numeric reference was available; no value was guessed.",
                "local": {
                    "event_type": result["event_type"],
                    "terminal_time_s": result["terminal_time_s"],
                },
            }
            print(f"{case['name']}: BLOCKER missing existing reference")
        else:
            row = {
                "case_name": case["name"],
                "status": "compared_to_existing_approximate_reference",
                "comparison": compare_result_to_reference(result, references[reference_key]),
            }
            comparison = row["comparison"]
            local = comparison["local"]
            print(
                f"{case['name']}: event_match={comparison['event_match']} "
                f"local_event={local['event_type']} "
                f"t={local['terminal_time_s']:.3f}s"
            )
        comparisons.append(row)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_label": config["model_label"],
        "aero_model_version": config["aero_model_version"],
        "force_geometry_version": config["force_geometry_version"],
        "control_model_version": config["control_model_version"],
        "statshark_new_calculation_performed_this_run": False,
        "reference_data_includes_prior_authorized_readout": reference_data["source"].get(
            "new_calculation_performed_for_this_project", False
        ),
        "comparisons": comparisons,
    }
    output_path = PROJECT_DIR / "outputs" / "h2" / "reference_comparison_h2.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"written: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
