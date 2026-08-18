from __future__ import annotations

import copy
import json
import math
from dataclasses import replace
from pathlib import Path

from aim120_model.config import find_case, load_cases
from aim120_model.dynamics import SimState, state_is_finite
from aim120_model.h2_dynamics import forces_for_state_h2, rk4_step_h2
from aim120_model.h2_simulator import H2Simulator
from aim120_model.profile_adapter import (
    GENERALIZED_AERO_MOMENT_PLANT,
    LEGACY_CRITICAL_DAMPED_PLANT,
    build_h2_candidate_config,
)
from aim120_model.propulsion import PiecewisePropulsion


ROOT = Path(__file__).resolve().parents[1]


def _profile(missile_id: str = "us_aim_120a") -> dict:
    return json.loads(
        (ROOT / "missiles" / f"{missile_id}.json").read_text(encoding="utf-8")
    )


def _defaults(plant_model: str) -> dict:
    defaults = json.loads(
        (ROOT / "config" / "profile_h2_runtime_defaults.json").read_text(
            encoding="utf-8"
        )
    )
    defaults["plant_model"] = plant_model
    return defaults


def _config() -> dict:
    config, _ = build_h2_candidate_config(
        _profile(), _defaults(GENERALIZED_AERO_MOMENT_PLANT)
    )
    return config


def _state(config: dict, **kwargs: float) -> SimState:
    return SimState(
        position=(0.0, 3000.0, 0.0),
        velocity=(300.0, 0.0, 0.0),
        pitch=float(kwargs.get("pitch", 0.0)),
        yaw=float(kwargs.get("yaw", 0.0)),
        pitch_rate=float(kwargs.get("pitch_rate", 0.0)),
        yaw_rate=float(kwargs.get("yaw_rate", 0.0)),
        mass=float(config["geometry"]["initial_mass_kg"]),
        actual_pitch_fin_angle_rad=float(kwargs.get("pitch_fin", 0.0)),
        actual_yaw_fin_angle_rad=float(kwargs.get("yaw_fin", 0.0)),
    )


def _diag(config: dict, state: SimState):
    return forces_for_state_h2(
        state,
        0.0,
        config,
        PiecewisePropulsion.from_config(config),
        powered=False,
    )


def _central(plus, minus, field: str, step: float) -> float:
    return (float(getattr(plus, field)) - float(getattr(minus, field))) / (2.0 * step)


def test_generalized_candidate_is_explicit_and_does_not_change_legacy_default() -> None:
    legacy, _ = build_h2_candidate_config(_profile(), _defaults(LEGACY_CRITICAL_DAMPED_PLANT))
    candidate = _config()

    assert legacy["control"]["plant_semantics"] == "fin_torque_body_aoa"
    assert candidate["control"]["plant_semantics"] == "generalized_aero_moment"
    assert candidate["runtime_adapter"] == "profile_h2_generalized_aero_moment_v1_candidate"
    assert "Unsupported local reduced-order generalized CN/Cm" in candidate["reference"]["runtime_boundary"]
    assert candidate["control"]["pid"] == legacy["control"]["pid"]
    assert candidate["control"]["pid"]["p"] == _profile()["control"]["pid"]["p"]
    assert candidate["drag_model"] == legacy["drag_model"]
    assert "generalized_aero_moment_candidate" in candidate["aerodynamics"]
    assert "fixed_lifting_surface_candidate" not in candidate["aerodynamics"]
    assert "tail_station_x_m" not in candidate["aerodynamics"]
    assert candidate["control"]["base_indicated_speed_mode"] == "none"
    generalized = candidate["aerodynamics"]["generalized_aero_moment_candidate"]
    assert generalized["cm_alpha_dot"] == 0.0
    assert generalized["cm_alpha_dot_runtime_status"] == (
        "frozen_zero_unsupported_no_independent_alpha_dot_state"
    )


def test_generalized_candidate_rejects_out_of_boundary_coefficient() -> None:
    defaults = _defaults(GENERALIZED_AERO_MOMENT_PLANT)
    defaults["generalized_aero_moment_candidate"]["cm_delta_per_rad"] = 1001.0
    try:
        build_h2_candidate_config(_profile(), defaults)
    except ValueError as exc:
        assert "within +/-1000" in str(exc)
    else:
        raise AssertionError("out-of-bound generalized coefficient must be rejected")


def test_generalized_candidate_freezes_requested_alpha_dot_coefficient() -> None:
    defaults = _defaults(GENERALIZED_AERO_MOMENT_PLANT)
    defaults["generalized_aero_moment_candidate"]["cm_alpha_dot"] = 12.0
    config, assumptions = build_h2_candidate_config(_profile(), defaults)
    candidate = config["aerodynamics"]["generalized_aero_moment_candidate"]
    assert candidate["cm_alpha_dot_requested"] == 12.0
    assert candidate["cm_alpha_dot"] == 0.0
    assert any("runtime freezes Cm_alpha_dot=0" in item for item in assumptions)


def test_generalized_telemetry_declares_cm_q_only_rate_identification() -> None:
    config = _config()
    config["performance"]["lifetime_s"] = 0.01
    config["numerics"]["max_steps"] = 2
    case = find_case(load_cases(ROOT / "configs" / "cases.yaml"), "power_only")
    sample = H2Simulator(config).run(case)["samples"][0]

    assert sample["generalized_cm_alpha_dot_runtime_status"] == (
        "frozen_zero_unsupported_no_independent_alpha_dot_state"
    )
    assert sample["generalized_identified_rate_term"] == "Cm_q only"
    assert sample["generalized_cm_alpha_dot_runtime_enabled"] is False
    assert sample["generalized_cm_alpha_dot_per_rad"] == 0.0


def test_generalized_force_and_moment_derivatives_are_independent() -> None:
    config = _config()
    candidate = config["aerodynamics"]["generalized_aero_moment_candidate"]
    zero = _diag(config, _state(config))
    epsilon = 1.0e-5
    rate_epsilon = 1.0e-4
    alpha_plus = _diag(config, _state(config, pitch=epsilon))
    alpha_minus = _diag(config, _state(config, pitch=-epsilon))
    delta_plus = _diag(config, _state(config, pitch_fin=epsilon))
    delta_minus = _diag(config, _state(config, pitch_fin=-epsilon))
    rate_plus = _diag(config, _state(config, pitch_rate=rate_epsilon))
    rate_minus = _diag(config, _state(config, pitch_rate=-rate_epsilon))
    q_s = zero.aero.normal_force_dynamic_pressure_pa * zero.body_reference_area_m2
    d = zero.body_reference_length_m

    assert math.isclose(
        _central(alpha_plus, alpha_minus, "pitch_body_normal_force_n", epsilon),
        q_s * candidate["cn_alpha_per_rad"],
        rel_tol=1e-10,
    )
    assert math.isclose(
        _central(alpha_plus, alpha_minus, "pitch_body_static_moment_nm", epsilon),
        q_s * d * candidate["cm_alpha_per_rad"],
        rel_tol=1e-10,
    )
    assert math.isclose(
        _central(delta_plus, delta_minus, "pitch_tail_force_n", epsilon),
        q_s * candidate["cn_delta_per_rad"],
        rel_tol=1e-10,
    )
    assert math.isclose(
        _central(delta_plus, delta_minus, "pitch_tail_moment_nm", epsilon),
        q_s * d * candidate["cm_delta_per_rad"],
        rel_tol=1e-10,
    )
    expected_rate = (
        q_s
        * d
        * candidate["cm_q"]
        * d
        / (2.0 * zero.aero.speed_mps)
    )
    assert math.isclose(
        _central(rate_plus, rate_minus, "pitch_body_rate_moment_nm", rate_epsilon),
        expected_rate,
        rel_tol=1e-10,
    )
    assert zero.generalized_cm_alpha_dot_per_rad == 0.0
    assert zero.generalized_pitch_alpha_dot_hat == 0.0
    assert zero.generalized_yaw_alpha_dot_hat == 0.0
    assert zero.generalized_cm_alpha_dot_runtime_enabled is False

    # Changing Cm_delta changes only the moment branch; changing CN_delta
    # changes only the force branch.  This is the structural identifiability
    # property that a single force station cannot provide.
    cm_changed = copy.deepcopy(config)
    cm_changed["aerodynamics"]["generalized_aero_moment_candidate"]["cm_delta_per_rad"] *= 2.0
    cm_plus = _diag(cm_changed, _state(cm_changed, pitch_fin=epsilon))
    cm_minus = _diag(cm_changed, _state(cm_changed, pitch_fin=-epsilon))
    assert _central(cm_plus, cm_minus, "pitch_tail_force_n", epsilon) == _central(
        delta_plus, delta_minus, "pitch_tail_force_n", epsilon
    )
    assert math.isclose(
        _central(cm_plus, cm_minus, "pitch_tail_moment_nm", epsilon),
        2.0 * _central(delta_plus, delta_minus, "pitch_tail_moment_nm", epsilon),
        rel_tol=1e-12,
    )

    cn_changed = copy.deepcopy(config)
    cn_changed["aerodynamics"]["generalized_aero_moment_candidate"]["cn_delta_per_rad"] *= 2.0
    cn_plus = _diag(cn_changed, _state(cn_changed, pitch_fin=epsilon))
    cn_minus = _diag(cn_changed, _state(cn_changed, pitch_fin=-epsilon))
    assert math.isclose(
        _central(cn_plus, cn_minus, "pitch_tail_force_n", epsilon),
        2.0 * _central(delta_plus, delta_minus, "pitch_tail_force_n", epsilon),
        rel_tol=1e-12,
    )
    assert _central(cn_plus, cn_minus, "pitch_tail_moment_nm", epsilon) == _central(
        delta_plus, delta_minus, "pitch_tail_moment_nm", epsilon
    )


def test_generalized_candidate_keeps_aft_tail_nonminimum_phase_signs_explicit() -> None:
    config = _config()
    positive = _diag(config, _state(config, pitch_fin=0.1))
    assert positive.pitch_tail_force_n < 0.0
    assert positive.pitch_tail_moment_nm > 0.0
    assert positive.pitch_body_static_moment_nm == 0.0
    assert positive.pitch_tail_moment_nm != positive.pitch_tail_force_n * 0.0


def test_generalized_candidate_short_integration_is_finite_and_damped() -> None:
    config = _config()
    propulsion = PiecewisePropulsion.from_config(config)
    state = _state(config, pitch_fin=0.02)
    max_rate = 0.0
    for index in range(100):
        state = rk4_step_h2(state, index * 0.001, 0.001, config, propulsion, False)
        assert state_is_finite(state)
        max_rate = max(max_rate, abs(state.pitch_rate), abs(state.yaw_rate))
    assert max_rate < 10.0
    assert state.pitch_rate > 0.0
