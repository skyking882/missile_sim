import copy
import json
import math
from pathlib import Path

from aim120_model.aerodynamics import StandardAtmosphere
from aim120_model.config import load_model_config
from aim120_model.drag_models import (
    CX_1943_X1_10_TABLE,
    INTERPOLATED_CX_1943_X1_10,
    area_basis,
    drag_force_from_cda,
    drag_power_w,
    effective_cda0,
    effective_cda_alpha,
    interpolate_cx_vs_mach,
)
from aim120_model.math3d import dot


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


def test_1943_cx_table_interpolates_and_holds_endpoints() -> None:
    assert interpolate_cx_vs_mach(1.0) == 0.330
    assert interpolate_cx_vs_mach(1.2) == 0.413
    assert interpolate_cx_vs_mach(4.0) == 0.231
    assert interpolate_cx_vs_mach(0.0) == 0.173
    assert interpolate_cx_vs_mach(5.0) == 0.231
    mid = interpolate_cx_vs_mach(1.5)
    expected = 0.396 + (1.5 - 1.4) / (1.6 - 1.4) * (0.380 - 0.396)
    assert abs(mid - expected) < 1e-12
    assert abs(interpolate_cx_vs_mach(1.5) - 0.388) < 1e-12
    assert CX_1943_X1_10_TABLE[0] == (0.8, 0.173)
    assert CX_1943_X1_10_TABLE[-1] == (4.0, 0.231)


def test_profile_cda0_uses_interpolated_1943_table_times_cxk() -> None:
    from aim120_model.profile_adapter import build_h2_candidate_config, load_runtime_defaults

    root = Path(__file__).resolve().parents[1]
    profile = json.loads((root / "missiles" / "us_aim_120a.json").read_text(encoding="utf-8"))
    defaults = load_runtime_defaults(str((root / "config" / "profile_h2_runtime_defaults.json").resolve()))
    config, _ = build_h2_candidate_config(profile, defaults)
    assert config["drag_model"]["shape_mode"] == INTERPOLATED_CX_1943_X1_10
    area = area_basis(config)
    cx_k = float(config["aerodynamics"]["cx_k"])
    cda = effective_cda0(1.2, config)
    assert abs(cda - area * cx_k * 0.413) < 1e-12
    gaussian = copy.deepcopy(config)
    gaussian["drag_model"]["shape_mode"] = "scaled_h1_shape"
    assert effective_cda0(1.2, config) < effective_cda0(1.2, gaussian)


def test_profile_cda_alpha_uses_caliber_area_without_mach_shape() -> None:
    from aim120_model.profile_adapter import build_h2_candidate_config, load_runtime_defaults

    root = Path(__file__).resolve().parents[1]
    profile = json.loads((root / "missiles" / "cn_pl12.json").read_text(encoding="utf-8"))
    defaults = load_runtime_defaults(str((root / "config" / "profile_h2_runtime_defaults.json").resolve()))
    config, _ = build_h2_candidate_config(profile, defaults)
    caliber = area_basis(config, "caliber_area")
    wing = area_basis(config)
    assert wing > caliber
    cx_aoa = float(config["aerodynamics"]["cx_vs_aoa"])
    alpha = 0.2
    expected = caliber * cx_aoa * alpha * alpha
    assert abs(effective_cda_alpha(alpha, 0.0, 1.2, config) - expected) < 1e-12
    assert abs(effective_cda_alpha(alpha, 0.0, 3.1, config) - expected) < 1e-12
    capped = effective_cda_alpha(2.0, 0.0, 1.2, config)
    assert abs(capped - caliber * cx_aoa * 1.2 * 1.2) < 1e-12
    frozen = effective_cda_alpha(alpha, 0.0, 1.2, CONFIG)
    assert frozen != expected
