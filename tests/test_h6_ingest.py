import math

from aim120_model.fin_force_inverse import normalize_backend_result, normalize_capture_bundle, validate_response_arrays


def _response(offset=0.0):
    return {
        "times": [0.0, 0.1, 0.2],
        "missileX": [offset, offset + 10.0, offset + 20.0],
        "missileY": [0.0, 0.0, 0.0],
        "missileZ": [0.0, 0.5, 2.0],
        "missileSpeedMs": [100.0, 100.0, 100.0],
        "angle": [0.0, 0.0, 0.0],
        "yaw": [0.0, 1.0, 2.0],
        "currentMass": [147.87, 147.87, 147.87],
        "currentThrust": [0.0, 0.0, 0.0],
        "drag": [0.0, 0.0, 0.0],
    }


def test_backend_arrays_are_normalized_and_units_are_explicit():
    normalized = normalize_backend_result(_response(), "H6_TEST", "CASE", angle_unit="deg", body={"cy_k": 0.0})
    assert normalized["status"] == "normalized"
    assert normalized["angle_unit_input"] == "deg"
    assert len(normalized["rows"]) == 3
    assert normalized["rows"][1]["raw_index"] == 1
    assert normalized["rows"][1]["mass_kg"] == 147.87
    assert normalized["rows"][1]["source_kind"] == "statshark_backend_timeseries"


def test_multi_result_mapping_uses_missile_ids_and_preserves_failures():
    payload = {
        "schema_version": 1,
        "captures": [
            {"case_id": "A", "model_id": "M_A", "status": "success", "response": {"missileIds": ["M_A", "M_B"], "results": [_response(), _response(1.0)]}},
            {"case_id": "B", "model_id": "M_B", "status": "empty_result", "failure": "empty"},
        ],
    }
    normalized = normalize_capture_bundle(payload)
    assert normalized["record_count"] == 2
    assert len(normalized["normalized_rows"]) == 3
    assert normalized["normalized_records"][1]["status"] == "preserved_non_success"


def test_array_length_and_time_failures_are_rejected():
    bad = _response()
    bad["missileY"] = [0.0]
    report = validate_response_arrays(bad, model_id="BAD")
    assert report["status"] == "invalid"
    assert any(issue.startswith("length_mismatch:y_m") for issue in report["issues"])
    bad_time = _response()
    bad_time["times"] = [0.0, 0.2, 0.1]
    assert "time_not_strictly_increasing" in validate_response_arrays(bad_time)["issues"]
