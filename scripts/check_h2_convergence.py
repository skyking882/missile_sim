#!/usr/bin/env python3
"""Check H2 terminal-result sensitivity to the fixed RK4 time step."""

from __future__ import annotations

import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from aim120_model.config import load_cases, load_model_config
from aim120_model.h2_simulator import H2Simulator
from aim120_model.metrics import terminal_summary


def main() -> int:
    config = load_model_config(PROJECT_DIR / "configs" / "aim120a_h2.yaml")
    cases = load_cases(PROJECT_DIR / "configs" / "cases.yaml")
    rows: list[dict[str, object]] = []
    for case in cases:
        runs: dict[str, dict[str, object]] = {}
        for dt_s in (0.02, 0.01):
            candidate = copy.deepcopy(config)
            candidate["numerics"]["dt_s"] = dt_s
            candidate["numerics"]["max_steps"] = max(
                int(candidate["numerics"]["max_steps"]),
                int(candidate["performance"]["lifetime_s"] / dt_s) + 10,
            )
            result = H2Simulator(candidate).run(case)
            runs[str(dt_s)] = terminal_summary(result)
        coarse = runs["0.02"]
        fine = runs["0.01"]
        rows.append({
            "case_name": case["name"],
            "runs": runs,
            "absolute_difference": {
                "terminal_time_s": abs(coarse["terminal_time_s"] - fine["terminal_time_s"]),
                "terminal_speed_kmh": abs(coarse["terminal_speed_kmh"] - fine["terminal_speed_kmh"]),
                "terminal_altitude_m": abs(coarse["terminal_altitude_m"] - fine["terminal_altitude_m"]),
                "terminal_distance_to_target_m": abs(
                    coarse["terminal_distance_to_target_m"] - fine["terminal_distance_to_target_m"]
                ),
            },
        })
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_label": config["model_label"],
        "dt_s_compared": [0.02, 0.01],
        "interpretation": "This is a numerical convergence check, not a validation against the hidden game solver.",
        "cases": rows,
    }
    output_path = PROJECT_DIR / "outputs" / "h2" / "convergence_h2.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        diff = row["absolute_difference"]
        print(
            f"{row['case_name']}: dt0.02-vs-dt0.01 "
            f"dt={diff['terminal_time_s']:.6f}s "
            f"speed={diff['terminal_speed_kmh']:.6f}km/h"
        )
    print(f"written: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
