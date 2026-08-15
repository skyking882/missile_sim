#!/usr/bin/env python3
"""Run one or all local H1 cases and save auditable output files."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from aim120_model.config import find_case, load_cases, load_data_file, load_model_config
from aim120_model.metrics import terminal_summary
from aim120_model.simulator import H1Simulator


def git_commit(project_dir: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--case")
    group.add_argument("--all", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config_path = PROJECT_DIR / "configs" / "aim120a_statshark.yaml"
    cases_path = PROJECT_DIR / "configs" / "cases.yaml"
    config = load_model_config(config_path)
    cases = load_cases(cases_path)
    selected = cases if args.all else [find_case(cases, args.case)]
    simulator = H1Simulator(config)
    output_dir = PROJECT_DIR / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for case in selected:
        result = simulator.run(case)
        now = datetime.now(timezone.utc)
        stamp = now.strftime("%Y%m%dT%H%M%SZ")
        output_path = args.output if args.output and len(selected) == 1 else output_dir / f"{case['name']}_{stamp}.json"
        payload = {
            "metadata": {
                "generated_at_utc": now.isoformat(),
                "model_label": "local_candidate_H1",
                "git_commit": git_commit(PROJECT_DIR),
                "config_path": str(config_path),
                "case_path": str(cases_path),
                "new_statshark_calculation_performed": False,
            },
            "config": config,
            "case": case,
            "result": result,
            "terminal_summary": terminal_summary(result),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written.append(output_path)
        summary = payload["terminal_summary"]
        print(
            f"{case['name']}: event={summary['event_type']} "
            f"t={summary['terminal_time_s']:.3f}s "
            f"speed={summary['terminal_speed_kmh']:.1f}km/h "
            f"range={summary['terminal_distance_to_target_m']:.1f}m"
        )
    print("written:")
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

