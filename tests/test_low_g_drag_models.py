import math

from aim120_model.low_g_drag import (
    SparseCdaParameters,
    fit_linear_bounded,
    model_basis,
    prediction_from_coefficients,
    sparse_cda0,
    total_cda,
)


def test_sparse_cda_is_positive_and_continuous_over_check_range():
    params = SparseCdaParameters(0.01, 0.014, 0.003, 0.02)
    values = [sparse_cda0(0.2 + 0.1 * index, params) for index in range(31)]
    assert all(value > 0.0 for value in values)
    assert max(abs(values[index + 1] - values[index]) for index in range(len(values) - 1)) < 0.01


def test_alpha_term_is_even_and_burn_correction_is_powered_only():
    params = SparseCdaParameters(0.01, 0.014, 0.003, 0.5)
    plus = total_cda(1.2, math.radians(3.0), False, params)
    minus = total_cda(1.2, math.radians(-3.0), False, params)
    powered = total_cda(1.2, 0.0, True, params, burn_delta_m2=0.002)
    coast = total_cda(1.2, 0.0, False, params, burn_delta_m2=0.002)
    assert abs(plus - minus) < 1.0e-12
    assert abs(powered - coast - 0.002) < 1.0e-12


def test_bounded_linear_fit_recovers_synthetic_lg0_coefficients():
    coefficients = [0.008, 0.016, 0.004, 0.35]
    rows = []
    for powered in (False, True):
        for index in range(12):
            mach = 0.4 + 0.25 * index
            alpha = math.radians((index % 4) * 0.8)
            rows.append({
                "mach": mach,
                "alpha_rad": alpha,
                "powered": powered,
                "observed_cda_m2": prediction_from_coefficients(mach, alpha, powered, coefficients, "LG-0"),
            })
    fitted = fit_linear_bounded(rows, "LG-0")
    assert max(abs(a - b) for a, b in zip(fitted, coefficients)) < 1.0e-7
