import math

from aim120_model.body_alpha2_drag import BodyAlpha2Parameters, body_alpha2_cda


def test_three_alpha_levels_and_three_cx_levels_are_full_rank():
    rows = []
    for alpha_deg in (0.2, 1.4, 2.7):
        for cx_aoa in (0.0, 9.0, 18.0):
            alpha_squared = math.radians(alpha_deg) ** 2
            rows.append((1.0, alpha_squared, cx_aoa * alpha_squared))
    determinant = (
        rows[0][0] * (rows[4][1] * rows[8][2] - rows[4][2] * rows[8][1])
        - rows[0][1] * (rows[4][0] * rows[8][2] - rows[4][2] * rows[8][0])
        + rows[0][2] * (rows[4][0] * rows[8][1] - rows[4][1] * rows[8][0])
    )
    assert determinant != 0.0


def test_alpha2_model_does_not_use_degree_squared_units():
    parameters = BodyAlpha2Parameters(0.019, 1.0, 0.0)
    alpha_deg = 2.0
    increment = body_alpha2_cda(1.5, math.radians(alpha_deg), 0.0, parameters) - parameters.cda0_1p5
    assert abs(increment - math.radians(alpha_deg) ** 2) < 1.0e-12
    assert abs(increment - alpha_deg ** 2) > 1.0
