#!/usr/bin/env python3
"""Run local cases and compare only against already recorded references."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from aim120_model.config import find_case, load_cases, load_data_file, load_model_config
from aim120_model.metrics import compare_result_to_reference, terminal_summary
from aim120_model.simulator import H1Simulator


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "outputs" / "reference_comparison.json")
    args = parser.parse_args()

    config = load_model_config(PROJECT_DIR / "configs" / "aim120a_statshark.yaml")
    cases = load_cases(PROJECT_DIR / "configs" / "cases.yaml")
    reference_data = load_data_file(PROJECT_DIR / "data" / "reference_cases.json")
    references = reference_data["cases"]
    selected = [find_case(cases, args.case)] if args.case else cases
    simulator = H1Simulator(config)
    comparisons: list[dict[str, object]] = []
    for case in selected:
        result = simulator.run(case)
        reference_key = case.get("reference_case")
        if reference_key is None or reference_key not in references:
            row = {
                "case_name": case["name"],
                "status": "blocker_missing_reference",
                "reason": "Existing StatShark result was not numerically read; no value was guessed.",
                "local": terminal_summary(result),
            }
        else:
            row = {
                "case_name": case["name"],
                "status": "compared_to_existing_approximate_reference",
                "comparison": compare_result_to_reference(result, references[reference_key]),
            }
        comparisons.append(row)
        if row["status"] == "blocker_missing_reference":
            print(f"{case['name']}: BLOCKER missing existing reference")
        else:
            comparison = row["comparison"]
            local = comparison["local"]
            print(
                f"{case['name']}: event_match={comparison['event_match']} "
                f"local_event={local['event_type']} "
                f"t={local['terminal_time_s']:.3f}s"
            )
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_label": "local_candidate_H1",
        "statshark_new_calculation_performed": reference_data["source"].get("new_calculation_performed_for_this_project", False),
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
