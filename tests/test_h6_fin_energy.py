from aim120_model.fin_energy import augment_force_direction, compare_force_directions, fit_fin_drag_candidates


def _rows():
    return [
        {
            "time_s": float(index),
            "vx": 200.0,
            "vy": 0.0,
            "vz": 0.0,
            "pitch_rad": 0.0,
            "yaw_rad": 0.05,
            "fin_force_n": 100.0 + 10.0 * index,
            "dynamic_pressure_pa": 50000.0,
            "body_normal_accel_mps2": 0.0,
            "body_force_n": 0.0,
            "extra_drag_residual_n": 0.25 * (100.0 + 10.0 * index) ** 2 / 50000.0,
        }
        for index in range(4)
    ]


def test_flow_normal_force_has_zero_axial_projection():
    rows = augment_force_direction(_rows(), "flow_normal")
    assert max(abs(row["fin_axial_projection_n"]) for row in rows) < 1.0e-10


def test_body_normal_force_can_have_axial_projection_at_beta():
    rows = augment_force_direction(_rows(), "body_normal")
    assert max(abs(row["fin_axial_projection_n"]) for row in rows) > 0.1
    comparison = compare_force_directions(_rows())
    assert comparison["models"]["body_normal"]["projection_rms_n"] > comparison["models"]["flow_normal"]["projection_rms_n"]


def test_quadratic_fin_drag_candidate_recovers_declared_coefficient():
    fit = fit_fin_drag_candidates(_rows())
    coefficient = fit["candidate_fits"]["fin_load_squared"]["coefficients"][0]
    assert abs(coefficient - 0.25) < 1.0e-6
