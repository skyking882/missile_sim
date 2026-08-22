import math
from pathlib import Path

from aim120_model.aerodynamics import StandardAtmosphere
from aim120_model.drag_models import area_basis, drag_force_from_cda, drag_power_w, effective_cda0, effective_cda_alpha
from aim120_model.math3d import dot

from aim120_model.config import load_model_config


CONFIG = load_model_config(Path(__file__).parents[1] / "configs" / "aim120a_h2.yaml")


def test_effective_cda_is_positive_and_alpha_term_is_even():
    for mach in (0.0, 0.8, 1.0, 2.0, 5.0):
        assert effective_cda0(mach, CONFIG) > 0.0
    plus = effective_cda_alpha(0.1, -0.2, 1.2, CONFIG)
    minus = effective_cda_alpha(-0.1, 0.2, 1.2, CONFIG)
    assert plus > 0.0
    assert abs(plus - minus) < 1e-12


def test_drag_force_opposes_air_velocity_and_power_is_non_positive():
    velocity = (400.0, -30.0, 20.0)
    force = drag_force_from_cda(50000.0, 0.02, (1.0, 0.0, 0.0))
    assert dot(force, (1.0, 0.0, 0.0)) < 0.0
    assert drag_power_w(force, velocity) < 0.0
    zero = drag_force_from_cda(0.0, 0.02, (1.0, 0.0, 0.0))
    assert max(abs(value) for value in zero) == 0.0


def test_area_basis_matches_explicit_caliber_multiplier():
    base = math.pi * CONFIG["geometry"]["caliber_m"] ** 2 / 4.0
    expected = base * CONFIG["geometry"]["wing_area_multiplier"]
    assert abs(area_basis(CONFIG) - expected) < 1e-12


def test_supersonic_decay_shape_is_opt_in_and_floored():
    import copy

    baseline = effective_cda0(1.0, CONFIG)
    decayed_config = copy.deepcopy(CONFIG)
    decayed_config["drag_model"]["shape_mode"] = "scaled_h1_shape_supersonic_decay"
    decayed_config["drag_model"]["decay_start_mach"] = 1.2
    decayed_config["drag_model"]["decay_exponent"] = 0.5
    decayed_config["drag_model"]["decay_floor"] = 0.6
    below = effective_cda0(1.0, decayed_config)
    assert abs(below - baseline) < 1e-12
    high = effective_cda0(5.0, decayed_config)
    high_base = effective_cda0(5.0, CONFIG)
    assert high < high_base
    assert high >= 0.6 * high_base - 1e-12
    assert CONFIG["drag_model"]["shape_mode"] == "scaled_h1_shape"
