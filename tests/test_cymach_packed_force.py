from __future__ import annotations

import copy
import math
from pathlib import Path

from aim120_model.control import base_indicated_speed_schedule
from aim120_model.dynamics import SimState
from aim120_model.drag_models import mach_multiplier
from aim120_model.h2_dynamics import capture_alpha_envelope_g, forces_for_state_h2
from aim120_model.profile_adapter import build_h2_candidate_config, load_runtime_defaults
from aim120_model.propulsion import PiecewisePropulsion
from missile_gui.library import scan_library


ROOT = Path(__file__).resolve().parents[1]


def _aim7_config():
    defaults = load_runtime_defaults(str(ROOT / "config/profile_h2_runtime_defaults.json"))
    profiles, errors = scan_library(ROOT / "data/profile_sets/2.59.0.27/missiles", ROOT)
    assert errors == []
    profile = next(p for p in profiles if p["missile_id"] == "us_aim7e2_dogfight_sparrow")
    config, _ = build_h2_candidate_config(profile, defaults)
    config["aerodynamics"]["natural_lift_enabled"] = False
    return config


def test_explicit_cymach_mult_scales_actual_packed_force_and_capture_envelope():
    config = _aim7_config()
    table = config["aerodynamics"]["mach_lift"]
    assert table["model"] == "lookup_table"
    speed = 1.4 * 340.294
    state = SimState(
        (0.0, 6500.0, 0.0), (speed, 0.0, 0.0), 0.12, 0.0, 0.0, 0.0,
        config["geometry"]["initial_mass_kg"],
    )
    propulsion = PiecewisePropulsion.from_config(config)
    actual = forces_for_state_h2(state, 0.0, config, propulsion, powered=False)
    fallback = copy.deepcopy(config)
    fallback["aerodynamics"]["mach_lift"] = {
        "base": 1.0, "transonic_peak": 0.1, "center": 1.0, "width": 0.6,
    }
    no_table = forces_for_state_h2(state, 0.0, fallback, propulsion, powered=False)
    multiplier = mach_multiplier(actual.aero.mach, table)
    actual_g = actual.pitch_tail_force_n / (state.mass * config["atmosphere"]["gravity_mps2"])
    fallback_g = no_table.pitch_tail_force_n / (state.mass * config["atmosphere"]["gravity_mps2"])
    assert abs(actual_g - fallback_g * multiplier) < 1e-9

    base_indicated_speed_schedule(actual.aero.dynamic_pressure_pa, config)
    alpha_max = math.radians(config["aerodynamics"]["horizontal_fin_aoa_limit_deg"])
    capture_with = capture_alpha_envelope_g(
        actual.aero.dynamic_pressure_pa,
        state.mass,
        actual.propulsion.thrust_n,
        alpha_max,
        config,
        mach=actual.aero.mach,
    )
    capture_without = capture_alpha_envelope_g(
        actual.aero.dynamic_pressure_pa,
        state.mass,
        actual.propulsion.thrust_n,
        alpha_max,
        fallback,
        mach=actual.aero.mach,
    )
    thrust_g = actual.propulsion.thrust_n * math.sin(alpha_max) / (
        state.mass * config["atmosphere"]["gravity_mps2"]
    )
    assert abs((capture_with - thrust_g) - (capture_without - thrust_g) * multiplier) < 1e-9

def test_no_table_packed_force_path_remains_bitwise_unchanged():
    config = _aim7_config()
    table_config = copy.deepcopy(config)
    table_config["aerodynamics"]["mach_lift"] = {"base": 1.0, "transonic_peak": 0.1, "center": 1.0, "width": 0.6}
    state = SimState(
        (0.0, 6500.0, 0.0), (1.4 * 340.294, 0.0, 0.0), 0.12, 0.0, 0.0, 0.0,
        config["geometry"]["initial_mass_kg"],
    )
    propulsion = PiecewisePropulsion.from_config(config)
    result = forces_for_state_h2(state, 0.0, table_config, propulsion, powered=False)
    fin_limit = math.radians(table_config["aerodynamics"]["horizontal_fin_aoa_limit_deg"])
    schedule = base_indicated_speed_schedule(result.aero.dynamic_pressure_pa, table_config)
    expected = (
        table_config["aerodynamics"]["fins_lateral_acceleration_g"]
        * schedule.fin_force_scale
        * table_config["aerodynamics"]["packed_lift_slope_scale"]
        * result.aero.pitch_alpha_rad / fin_limit
    )
    result_g = result.pitch_tail_force_n / (state.mass * config["atmosphere"]["gravity_mps2"])
    assert abs(result_g - expected) < 1e-9
