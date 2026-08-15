import math

from aim120_model.fin_force_inverse import normalize_backend_result
from aim120_model.h6_utils import unwrap_angles


def _circle_response(omega):
    radius = 100.0
    speed = radius * abs(omega)
    times = [0.0, 0.05, 0.10, 0.15, 0.20]
    x = [radius * math.sin(omega * time) for time in times]
    z = [radius * (1.0 - math.cos(omega * time)) for time in times]
    yaw = [math.degrees(omega * time) for time in times]
    return {
        "times": times,
        "missileX": x,
        "missileY": [0.0] * len(times),
        "missileZ": z,
        "missileSpeedMs": [speed] * len(times),
        "angle": [0.0] * len(times),
        "yaw": yaw,
        "currentMass": [147.87] * len(times),
    }


def test_horizontal_curvature_preserves_positive_and_negative_sign():
    positive = normalize_backend_result(_circle_response(0.2), "P", "P", angle_unit="deg", body={"cy_k": 0.0})
    negative = normalize_backend_result(_circle_response(-0.2), "N", "N", angle_unit="deg", body={"cy_k": 0.0})
    assert sum(row["normal_accel_yaw_mps2"] for row in positive["rows"]) > 0.0
    assert sum(row["normal_accel_yaw_mps2"] for row in negative["rows"]) < 0.0
    assert sum(row["fin_normal_accel_mps2"] for row in positive["rows"]) > 0.0
    assert sum(row["fin_normal_accel_mps2"] for row in negative["rows"]) < 0.0


def test_angle_unwrap_does_not_create_pi_spike():
    values = [math.radians(value) for value in (179.0, -179.0, -177.0)]
    unwrapped = unwrap_angles(values)
    assert max(abs(value) for value in unwrapped) < 4.0 * math.pi
    assert abs(unwrapped[1] - unwrapped[0]) < math.radians(5.0)


def test_backend_aoa_uses_declared_angle_unit():
    response = _circle_response(0.2)
    response["aoa"] = [10.0] * len(response["times"])
    normalized = normalize_backend_result(response, "A", "A", angle_unit="deg", body={"cy_k": 0.0})
    assert abs(normalized["rows"][2]["aoa_rad"] - math.radians(10.0)) < 1.0e-12
