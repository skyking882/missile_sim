import json
from pathlib import Path


def test_frozen_h4_visible_rows_have_required_alpha_and_source_boundary():
    root = Path(__file__).resolve().parents[1]
    path = root / "data" / "raw" / "statshark_h4" / "G4_statshark_visible_slider_20260811.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload["metadata"]
    assert metadata["missile_variant"].startswith("AIM120A_H4_GLIDE_NOPWR_NOGUIDE")
    assert metadata["static_mass_kg"] == 147.87
    assert metadata["custom_model_boundary"]["guidance_enabled"] is False
    assert metadata["custom_model_boundary"]["fin_lateral_acceleration"] == 0
    samples = payload["result"]["samples"]
    near_m15 = [row for row in samples if 1.45 <= row["mach"] <= 1.55]
    assert near_m15
    assert all("alpha_rad" in row for row in near_m15)
