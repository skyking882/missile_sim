import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from freeze_h5_phase0 import partial_model_snapshot


def test_partial_model_snapshot_returns_observed_fields_and_true_missing_set():
    snapshot = partial_model_snapshot()
    assert snapshot is not None
    assert snapshot["status"] == "missing_full_field_snapshot"
    assert snapshot["observed_partial_fields"]["mass"]["value_kg"] == 147.87
    assert snapshot["observed_partial_fields"]["engine force"]["value_n"] == 0.0
    assert snapshot["observed_partial_fields"]["guidance"]["enabled"] is False
    assert snapshot["observed_partial_fields"]["finsLatAccel"]["value_g"] == 0.0
    assert "mass" not in snapshot["missing_fields"]
    assert "engine force" not in snapshot["missing_fields"]
    assert "guidance" not in snapshot["missing_fields"]
    assert "finsLatAccel" not in snapshot["missing_fields"]
