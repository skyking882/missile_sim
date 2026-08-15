import math

from aim120_model.units import deg_to_rad, kmh_to_mps, mps2_to_g, mps_to_kmh, rad_to_deg


def test_speed_round_trip():
    assert kmh_to_mps(1200.0) == 1200.0 / 3.6
    assert mps_to_kmh(kmh_to_mps(1200.0)) == 1200.0


def test_angle_round_trip():
    assert math.isclose(rad_to_deg(deg_to_rad(22.5)), 22.5, abs_tol=1e-12)


def test_g_conversion():
    assert math.isclose(mps2_to_g(9.80665), 1.0, abs_tol=1e-12)

