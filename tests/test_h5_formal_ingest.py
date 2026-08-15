"""Regression checks for the bounded H5 formal-capture ingestion."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "h5_body_alpha2"
RAW_DIR = ROOT / "data" / "raw" / "statshark_h5_body_alpha2"


def _load(name: str) -> dict:
    return json.loads((OUTPUT_DIR / name).read_text(encoding="utf-8"))


def test_h5_formal_matrix_preserves_failures_and_budget() -> None:
    raw = json.loads((RAW_DIR / "formal_capture_bundle.json").read_text(encoding="utf-8"))
    coverage = _load("h5_coverage_report.json")
    assert raw["action_budget"]["authorized_max_calculate_actions"] == 15
    assert raw["action_budget"]["calculate_actions_used"] == 15
    assert len(coverage["cases"]) == 9
    assert coverage["matrix"]["captured_case_count"] == 6
    assert coverage["matrix"]["full_3x3_capture"] is False
    assert {case["case_id"] for case in coverage["cases"] if case["capture_status"] != "captured"} == {
        "H5-P0-C18",
        "H5-P1-C18",
        "H5-P2-C18",
    }


def test_h5_formal_alpha_gates_use_displayed_window_values() -> None:
    coverage = _load("h5_coverage_report.json")
    by_id = {case["case_id"]: case for case in coverage["cases"]}
    assert by_id["H5-P0-C0"]["primary_window"]["nominal_history_sample_count"] == 2
    assert by_id["H5-P1-C0"]["formal_case_gate"] is True
    assert by_id["H5-P1-C9"]["formal_case_gate"] is True
    assert by_id["H5-P2-C0"]["primary_window"]["nominal_history_sample_count"] == 0
    assert by_id["H5-P2-C9"]["primary_window"]["nominal_history_sample_count"] == 0


def test_h5_formal_fit_remains_blocked_without_fabricated_parameters() -> None:
    acceptance = _load("h5_acceptance_report.json")
    fit = _load("h5_fit_diagnostics.json")
    normalized = _load("h5_normalized_samples.json")
    assert acceptance["final_status"] == "h5_formal_fit_blocked_missing_C18_and_alpha_coverage"
    assert acceptance["gates"]["three_by_three_capture_gate"] is False
    assert acceptance["gates"]["C18_field_scaling_gate"] is False
    assert all(value is None for value in fit["main_parameters"].values())
    assert normalized["raw_labels_preserved"] is True
    assert normalized["failed_or_empty_formal_cases"]
