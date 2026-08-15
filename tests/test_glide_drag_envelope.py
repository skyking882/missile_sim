import math

from aim120_model.glide_drag_envelope import (
    LogCdaEnvelope,
    cda_physical_checks,
    classify_mach_support,
)


def test_log_linear_interpolation_is_geometric_and_positive():
    envelope = LogCdaEnvelope.from_cda_knots([0.2, 1.0, 2.0], [0.01, 0.04, 0.01])
    midpoint = envelope.evaluate(0.6)
    expected = math.sqrt(0.01 * 0.04)
    assert abs(midpoint["cda_m2"] - expected) < 1.0e-12
    assert midpoint["model_support_label"] == "interpolation"
    assert cda_physical_checks(envelope, 0.2, 2.0, 0.1)["positive_and_finite"]


def test_outside_model_range_is_labeled_extrapolation_not_direct_support():
    envelope = LogCdaEnvelope.from_cda_knots([0.2, 4.5], [0.01, 0.02])
    assert envelope.evaluate(0.1)["model_support_label"] == "extrapolation"
    assert envelope.evaluate(4.5)["model_support_label"] == "direct_support"
    assert classify_mach_support(0.3, [(0.2, 0.8), (1.2, 2.0)], (0.2, 4.5)) == "direct_support"
    assert classify_mach_support(1.0, [(0.2, 0.8), (1.2, 2.0)], (0.2, 4.5)) == "interpolation"
    assert classify_mach_support(3.0, [(0.2, 0.8), (1.2, 2.0)], (0.2, 4.5)) == "extrapolation"
