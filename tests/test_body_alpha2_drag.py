import math

from aim120_model.body_alpha2_drag import (
    BodyAlpha2Parameters,
    H4ShapePrior,
    alpha_deg_to_rad,
    alpha_rad_from_row,
    body_alpha2_cda,
    fit_cda_observations,
)


def test_alpha_is_converted_to_radians_before_squared_term():
    assert abs(alpha_deg_to_rad(180.0) - math.pi) < 1.0e-12
    row = {"alpha_deg": 2.0}
    assert abs(alpha_rad_from_row(row) - math.radians(2.0)) < 1.0e-12
    assert abs(alpha_rad_from_row({"alpha_rad": math.radians(2.0), "alpha_deg": 99.0}) - math.radians(2.0)) < 1.0e-12


def test_alpha2_is_zero_at_zero_and_even_in_sign():
    parameters = BodyAlpha2Parameters(0.019, 0.2, 0.01)
    zero = body_alpha2_cda(1.5, 0.0, 9.0, parameters)
    plus = body_alpha2_cda(1.5, math.radians(2.0), 9.0, parameters)
    minus = body_alpha2_cda(1.5, math.radians(-2.0), 9.0, parameters)
    assert abs(zero - parameters.cda0_1p5) < 1.0e-12
    assert abs(plus - minus) < 1.0e-12
    assert plus > zero


def test_cx_aoa_levels_have_linear_coefficient_intervention():
    parameters = BodyAlpha2Parameters(0.019, 0.1, 0.02)
    alpha = math.radians(2.0)
    c0 = body_alpha2_cda(1.5, alpha, 0.0, parameters)
    c9 = body_alpha2_cda(1.5, alpha, 9.0, parameters)
    c18 = body_alpha2_cda(1.5, alpha, 18.0, parameters)
    assert abs((c9 - c0) - (c18 - c9)) < 1.0e-12


def test_h4_shape_is_only_a_known_local_offset():
    shape = H4ShapePrior((1.2, 1.5, 2.0), (0.024, 0.019, 0.016))
    parameters = BodyAlpha2Parameters(0.019, 0.0, 0.0)
    assert abs(body_alpha2_cda(1.5, 0.0, 9.0, parameters, shape) - 0.019) < 1.0e-12
    assert body_alpha2_cda(1.2, 0.0, 9.0, parameters, shape) > 0.019


def test_direct_cda_fit_recovers_three_h5_parameters():
    truth = BodyAlpha2Parameters(0.019, 0.25, 0.015)
    rows = []
    for alpha_deg in (0.2, 1.4, 2.7):
        for cx_aoa in (0.0, 9.0, 18.0):
            rows.append({
                "mach": 1.5,
                "alpha_rad": math.radians(alpha_deg),
                "cx_aoa": cx_aoa,
                "observed_cda_m2": body_alpha2_cda(1.5, math.radians(alpha_deg), cx_aoa, truth),
            })
    fit = fit_cda_observations(rows)
    estimated = fit["parameters"]
    assert abs(estimated.cda0_1p5 - truth.cda0_1p5) < 1.0e-12
    assert abs(estimated.k_residual_1p5 - truth.k_residual_1p5) < 1.0e-12
    assert abs(estimated.s_cx_aoa_1p5 - truth.s_cx_aoa_1p5) < 1.0e-12
