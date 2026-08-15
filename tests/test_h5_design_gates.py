import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from validate_h5_synthetic_identifiability import (
    build_synthetic_trajectories,
    evaluate_design_matrix,
)
from aim120_model.body_alpha2_drag import BodyAlpha2Parameters, H4ShapePrior


def _shape():
    return H4ShapePrior((1.2, 1.5, 2.0), (0.024, 0.019, 0.016))


def test_missing_p1_is_a_real_holdout_gate_failure():
    full = build_synthetic_trajectories(BodyAlpha2Parameters(0.0185, 0.28, 0.012), _shape())
    without_p1 = build_synthetic_trajectories(
        BodyAlpha2Parameters(0.0185, 0.28, 0.012), _shape(), include_p1=False
    )
    assert evaluate_design_matrix(full)["P1_holdout_gate"] == "pass"
    missing = evaluate_design_matrix(without_p1)
    assert missing["complete_P1_history_present"] is False
    assert missing["P1_holdout_gate"] == "fail_missing_complete_P1"


def test_missing_c18_is_a_real_field_scaling_gate_failure():
    without_c18 = build_synthetic_trajectories(
        BodyAlpha2Parameters(0.0185, 0.28, 0.012), _shape(), include_c18=False
    )
    missing = evaluate_design_matrix(without_c18)
    assert missing["C18_field_level_present"] is False
    assert missing["CxAoA_linear_increment_gate"] == "fail_missing_C18_or_baseline"
