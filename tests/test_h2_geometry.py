import math
from pathlib import Path

from aim120_model.aerodynamics import compute_aerodynamics_h2
from aim120_model.config import load_model_config
from aim120_model.math3d import dot


CONFIG = load_model_config(Path(__file__).parents[1] / "configs" / "aim120a_h2.yaml")


def test_zero_aoa_has_no_natural_lift():
    sample = compute_aerodynamics_h2(3000.0, (300.0, 0.0, 0.0), 0.0, 0.0, CONFIG)
    assert abs(sample.pitch_alpha_rad) < 1e-12
    assert abs(sample.yaw_alpha_rad) < 1e-12
    assert max(abs(value) for value in sample.natural_lift_force_n) < 1e-9
    assert dot(sample.drag_force_n, sample.air_velocity_mps) < 0.0


def test_pitch_lift_is_perpendicular_and_has_expected_sign():
    sample = compute_aerodynamics_h2(3000.0, (300.0, 0.0, 0.0), math.radians(5.0), 0.0, CONFIG)
    assert sample.pitch_alpha_rad > 0.0
    assert sample.natural_lift_force_n[1] > 0.0
    assert abs(dot(sample.natural_lift_force_n, sample.air_velocity_mps)) < 1e-6


def test_yaw_lift_is_perpendicular_and_sign_symmetric():
    positive = compute_aerodynamics_h2(3000.0, (300.0, 0.0, 0.0), 0.0, math.radians(5.0), CONFIG)
    negative = compute_aerodynamics_h2(3000.0, (300.0, 0.0, 0.0), 0.0, math.radians(-5.0), CONFIG)
    assert positive.yaw_alpha_rad > 0.0
    assert negative.yaw_alpha_rad < 0.0
    assert positive.natural_lift_force_n[2] > 0.0
    assert negative.natural_lift_force_n[2] < 0.0
    assert abs(dot(positive.natural_lift_force_n, positive.air_velocity_mps)) < 1e-6


def test_flow_normal_lift_has_no_axial_component():
    sample = compute_aerodynamics_h2(3000.0, (250.0, -60.0, 35.0), math.radians(4.0), math.radians(-3.0), CONFIG)
    assert abs(dot(sample.natural_lift_force_n, sample.air_velocity_hat)) < 1e-8
