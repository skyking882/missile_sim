import math

from aim120_model.axial_inverse import estimate_smoothed_speeds, estimate_speed_derivatives, inverse_drag_sample
from aim120_model.sample_filters import LowGFilterSettings


def _settings():
    return LowGFilterSettings(q_min_pa=100.0)


def test_powered_axial_inverse_recovers_known_drag():
    rows = []
    for index in range(11):
        time_s = index * 0.1
        rows.append({
            "time_s": time_s,
            "speed_mps": 100.0 + 4.0 * time_s,
            "powered": True,
            "engine_stage": 1,
            "mass_kg": 10.0,
            "thrust_n": 60.0,
            "mach": 0.5,
            "altitude_m": 3000.0,
            "flight_path_angle_rad": 0.0,
            "flight_path_angle_deg": 0.0,
            "alpha_rad": 0.0,
            "alpha_total_deg": 0.0,
            "lateral_load_g": 0.0,
            "dynamic_pressure_pa": 2000.0,
        })
    derivatives = estimate_speed_derivatives(rows, window_s=0.2, polynomial_order=2)
    smoothed = estimate_smoothed_speeds(rows, window_s=0.2, polynomial_order=2)
    result = inverse_drag_sample(rows[5], derivatives[5], gravity_mps2=9.80665, settings=_settings())
    assert result["accepted"]
    assert abs(result["speed_derivative_mps2"] - 4.0) < 1.0e-8
    assert abs(smoothed[5] - rows[5]["speed_mps"]) < 1.0e-8
    assert abs(result["observed_drag_n"] - 20.0) < 1.0e-8
    assert abs(result["observed_cda_m2"] - 0.01) < 1.0e-8


def test_coast_axial_inverse_keeps_gravity_term():
    row = {
        "time_s": 10.0,
        "speed_mps": 200.0,
        "powered": False,
        "engine_stage": 0,
        "mass_kg": 10.0,
        "thrust_n": 0.0,
        "mach": 0.6,
        "altitude_m": 3000.0,
        "flight_path_angle_rad": math.radians(5.0),
        "flight_path_angle_deg": 5.0,
        "alpha_rad": 0.0,
        "alpha_total_deg": 0.0,
        "lateral_load_g": 0.0,
        "dynamic_pressure_pa": 2000.0,
    }
    # Choose dV/dt so that D=20 N after the gravity projection.
    derivative = -2.0 - 9.80665 * math.sin(math.radians(5.0))
    result = inverse_drag_sample(row, derivative, gravity_mps2=9.80665, settings=_settings())
    assert result["accepted"]
    assert abs(result["observed_drag_n"] - 20.0) < 1.0e-8
    assert abs(result["observed_cda_m2"] - 0.01) < 1.0e-8


def test_negative_inferred_drag_is_retained_as_rejection():
    row = {
        "time_s": 2.0,
        "speed_mps": 200.0,
        "powered": False,
        "engine_stage": 0,
        "mass_kg": 10.0,
        "thrust_n": 0.0,
        "mach": 0.6,
        "altitude_m": 3000.0,
        "flight_path_angle_rad": 0.0,
        "flight_path_angle_deg": 0.0,
        "alpha_rad": 0.0,
        "alpha_total_deg": 0.0,
        "lateral_load_g": 0.0,
        "dynamic_pressure_pa": 2000.0,
    }
    result = inverse_drag_sample(row, speed_derivative_mps2=1.0, settings=_settings())
    assert not result["accepted"]
    assert "observed_drag_non_positive" in result["rejection_reasons"]
