from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import replace
from pathlib import Path

from aim120_model.config import load_cases
from aim120_model.aerodynamics import (
    body_axes,
    body_axes_for_state,
    body_axes_from_quaternion,
    cg_wind_normal_basis,
    normalize_quaternion,
    quaternion_from_pitch_yaw,
    quaternion_multiply,
)
from aim120_model.control import base_indicated_speed_schedule, update_control_feedback
from aim120_model.dynamics import SimState, state_is_finite
from aim120_model.h2_dynamics import (
    _flow_at_station,
    combined_empirical_tail_force_reference,
    forces_for_state_h2,
    rk4_step_h2,
    split_empirical_tail_force,
)
from aim120_model.h2_simulator import H2Simulator
from aim120_model.guidance import guidance_command
from aim120_model.metrics import continuous_closest_approach
from aim120_model.math3d import cross, dot, scale
from aim120_model.profile_adapter import (
    BODY_CM_TAIL_FORCE_PLANT,
    LEGACY_CRITICAL_DAMPED_PLANT,
    build_h2_candidate_config,
)
from aim120_model.propulsion import PiecewisePropulsion
from aim120_model.target import TargetState


ROOT = Path(__file__).resolve().parents[1]


def _profile(missile_id: str = "us_aim_120a") -> dict:
    return json.loads((ROOT / "missiles" / f"{missile_id}.json").read_text(encoding="utf-8"))


def _defaults(plant_model: str) -> dict:
    defaults = json.loads(
        (ROOT / "config" / "profile_h2_runtime_defaults.json").read_text(encoding="utf-8")
    )
    defaults["plant_model"] = plant_model
    return defaults


def _candidate_config(missile_id: str = "us_aim_120a") -> dict:
    config, _assumptions = build_h2_candidate_config(
        _profile(missile_id), _defaults(BODY_CM_TAIL_FORCE_PLANT)
    )
    return config


def _legacy_config() -> dict:
    config, _assumptions = build_h2_candidate_config(
        _profile(), _defaults(LEGACY_CRITICAL_DAMPED_PLANT)
    )
    return config


def _diagnostics(config: dict, state: SimState):
    return forces_for_state_h2(
        state,
        0.0,
        config,
        PiecewisePropulsion.from_config(config),
        powered=False,
    )


def _state(
    *,
    pitch: float = 0.0,
    yaw: float = 0.0,
    pitch_rate: float = 0.0,
    yaw_rate: float = 0.0,
    pitch_fin: float = 0.0,
    yaw_fin: float = 0.0,
) -> SimState:
    return SimState(
        (0.0, 3000.0, 0.0),
        (300.0, 0.0, 0.0),
        pitch,
        yaw,
        pitch_rate,
        yaw_rate,
        147.87,
        actual_pitch_fin_angle_rad=pitch_fin,
        actual_yaw_fin_angle_rad=yaw_fin,
    )


def test_candidate_adapter_is_selectable_without_changing_frozen_control_inputs() -> None:
    profile = _profile()
    legacy = _legacy_config()
    candidate = _candidate_config()

    assert legacy["control"]["plant_semantics"] == "fin_torque_body_aoa"
    assert candidate["control"]["plant_semantics"] == "body_cm_tail_force_moment"
    assert candidate["runtime_adapter"] == "profile_h2_body_cm_split_tail_fixed_lift_v3_candidate"
    assert candidate["model_label"] == "us_aim_120a_profile_h2_body_cm_split_tail_fixed_lift_v3_candidate"
    assert "Unsupported local reduced-order" in candidate["reference"]["runtime_boundary"]
    assert legacy["runtime_adapter"] == "profile_h2_fin_torque_aoa_v12"
    assert legacy["model_label"] == "us_aim_120a_profile_h2_fin_torque_aoa_v12"
    assert candidate["control"]["pid_error_scale"] == 1.0
    assert legacy["control"]["pid_output_semantics"] == "fin_angle_rad"
    assert candidate["control"]["pid_output_semantics"] == "body_rate_command_rad_s"
    assert "candidate_rate_inner_loop" not in legacy["control"]
    assert candidate["control"]["candidate_rate_inner_loop"] == {
        "time_constant_s": 0.1,
        "outer_pid_output_semantics": "body_rate_command_rad_s",
        "inner_loop_semantics": "first_order_rate_target_with_local_angular_acceleration_and_tail_effectiveness_inversion",
        "source": "shared_unsupported_candidate_assumption",
    }
    assert candidate["control"]["pid"] == legacy["control"]["pid"]
    assert candidate["control"]["pid"]["p"] == profile["control"]["pid"]["p"]
    assert candidate["control"]["actuator_time_constant_s"] == legacy["control"]["actuator_time_constant_s"]
    assert candidate["control"]["derivative_filter_time_constant_s"] == legacy["control"]["derivative_filter_time_constant_s"]
    assert candidate["guidance"] == legacy["guidance"]
    assert candidate["aerodynamics"]["tail_station_x_m"] == -abs(
        profile["aerodynamics"]["fin_moment_arm_m"]
    )
    assert candidate["aerodynamics"]["tail_station_semantics"]["interpretation"] == (
        "unverified_empirical_arm_provisionally_metres_aft"
    )
    assert candidate["aerodynamics"]["empirical_fin_authority"]["radial_allocation"] == (
        "unit_disk_empirical_authority_allocation_not_stall_model"
    )
    assert candidate["control"]["fin_actuator_travel"]["source_field"] == (
        "finsAoaHor/finsAoaVer"
    )
    assert candidate["attitude_candidate"]["primary_orientation"] == (
        "unit_quaternion_body_to_inertial_wxyz"
    )
    split = candidate["aerodynamics"]["split_tail_candidate"]
    assert split["tail_alpha_force_multiplier"] == 1.0
    assert split["tail_delta_force_multiplier"] == 1.0
    assert split["fin_mechanical_limit_pitch_rad"] == candidate["control"]["fin_actuator_travel"]["pitch_limit_rad"]
    assert split["fin_authority_angle_reference_pitch_rad"] == candidate["aerodynamics"]["empirical_fin_authority"]["pitch_incidence_reference_rad"]
    assert "unsupported fixed-tail" in split["boundary"]
    fixed = candidate["aerodynamics"]["fixed_lifting_surface_candidate"]
    assert fixed["fixed_lifting_surface_multiplier"] == 0.0
    assert fixed["station_x_m"] == 0.0
    assert fixed["fixed_lifting_surface_area_slope_m2_per_rad"] == 0.0
    assert "not derived from wingAreaMult" in fixed["boundary"]


def _with_fixed_lifting_multiplier(config: dict, multiplier: float) -> dict:
    changed = copy.deepcopy(config)
    fixed = changed["aerodynamics"]["fixed_lifting_surface_candidate"]
    fixed["fixed_lifting_surface_multiplier"] = multiplier
    fixed["fixed_lifting_surface_area_slope_m2_per_rad"] = (
        multiplier * fixed["body_normal_force_area_slope_m2_per_rad"]
    )
    return changed


def test_fixed_lifting_k0_is_pointwise_identical_to_pre_source_candidate() -> None:
    config = _candidate_config("cn_pl12")
    previous = copy.deepcopy(config)
    previous["aerodynamics"].pop("fixed_lifting_surface_candidate")
    rng = random.Random(0x3F17ED)
    states = [
        _state(),
        _state(pitch=0.2, yaw=-0.1, pitch_rate=0.7, yaw_rate=-0.4, pitch_fin=0.08, yaw_fin=-0.03),
        replace(_state(pitch=-0.3, yaw=0.2), velocity=(760.0, 40.0, -80.0), mass=100.0),
    ] + [
        replace(
            _state(
                pitch=rng.uniform(-0.5, 0.5), yaw=rng.uniform(-0.5, 0.5),
                pitch_rate=rng.uniform(-4.0, 4.0), yaw_rate=rng.uniform(-4.0, 4.0),
                pitch_fin=rng.uniform(-0.4, 0.4), yaw_fin=rng.uniform(-0.4, 0.4),
            ),
            position=(0.0, rng.uniform(0.0, 15_000.0), 0.0),
            velocity=(rng.uniform(100.0, 1_200.0), rng.uniform(-150.0, 150.0), rng.uniform(-150.0, 150.0)),
            mass=rng.uniform(70.0, 200.0),
        )
        for _ in range(100)
    ]
    for state in states:
        old = _diagnostics(previous, state)
        new = _diagnostics(config, state)
        assert new.total_force_n == old.total_force_n
        assert new.non_gravity_acceleration_mps2 == old.non_gravity_acceleration_mps2
        assert new.pitch_angular_acceleration_rad_s2 == old.pitch_angular_acceleration_rad_s2
        assert new.yaw_angular_acceleration_rad_s2 == old.yaw_angular_acceleration_rad_s2
        assert new.pitch_tail_force_n == old.pitch_tail_force_n
        assert new.yaw_tail_force_n == old.yaw_tail_force_n
        assert new.pitch_body_total_moment_nm == old.pitch_body_total_moment_nm
        assert new.yaw_body_total_moment_nm == old.yaw_body_total_moment_nm
        assert new.fixed_lifting_surface_force_n == (0.0, 0.0, 0.0)


def test_fixed_lifting_surface_is_linear_wind_normal_force_with_zero_moment() -> None:
    base = _candidate_config("cn_pl12")
    state_pitch = _state(pitch=0.1)
    state_yaw = _state(yaw=0.1)
    one_pitch = _diagnostics(_with_fixed_lifting_multiplier(base, 1.0), state_pitch)
    two_pitch = _diagnostics(_with_fixed_lifting_multiplier(base, 2.0), state_pitch)
    one_yaw = _diagnostics(_with_fixed_lifting_multiplier(base, 1.0), state_yaw)
    diagonal = _diagnostics(_with_fixed_lifting_multiplier(base, 1.0), _state(pitch=1e-6, yaw=1e-6))
    faster = _diagnostics(
        _with_fixed_lifting_multiplier(base, 1.0),
        replace(state_pitch, velocity=(600.0, 0.0, 0.0)),
    )
    assert one_pitch.fixed_lifting_surface_station_x_m == 0.0
    assert one_pitch.pitch_fixed_lifting_surface_force_n > 0.0
    assert one_yaw.yaw_fixed_lifting_surface_force_n > 0.0
    assert one_pitch.pitch_fixed_lifting_surface_force_n == one_yaw.yaw_fixed_lifting_surface_force_n
    assert math.isclose(
        diagonal.pitch_fixed_lifting_surface_force_n,
        diagonal.yaw_fixed_lifting_surface_force_n,
        rel_tol=1e-6,
    )
    assert two_pitch.pitch_fixed_lifting_surface_force_n == 2.0 * one_pitch.pitch_fixed_lifting_surface_force_n
    assert math.isclose(
        faster.pitch_fixed_lifting_surface_force_n,
        4.0 * one_pitch.pitch_fixed_lifting_surface_force_n,
        rel_tol=2e-15,
    )
    assert one_pitch.pitch_fixed_lifting_surface_moment_nm == 0.0
    assert one_pitch.yaw_fixed_lifting_surface_moment_nm == 0.0
    assert two_pitch.pitch_angular_acceleration_rad_s2 == one_pitch.pitch_angular_acceleration_rad_s2
    assert two_pitch.yaw_angular_acceleration_rad_s2 == one_pitch.yaw_angular_acceleration_rad_s2
    assert abs(dot(one_pitch.fixed_lifting_surface_force_n, one_pitch.aero.air_velocity_mps)) < 1e-9


def test_fixed_lifting_surface_off_axis_station_adds_cross_product_moment() -> None:
    base = _candidate_config("cn_pl12")
    base["aerodynamics"]["fins_lateral_acceleration_g"] = 0.0
    state = _state(pitch=0.1, yaw=-0.03)
    at_cg = _with_fixed_lifting_multiplier(base, 2.0)
    at_cg["aerodynamics"]["fixed_lifting_surface_candidate"]["station_x_m"] = 0.0
    off_axis = copy.deepcopy(at_cg)
    off_axis["aerodynamics"]["fixed_lifting_surface_candidate"]["station_x_m"] = 0.25
    negative = copy.deepcopy(at_cg)
    negative["aerodynamics"]["fixed_lifting_surface_candidate"]["station_x_m"] = -0.25

    cg = _diagnostics(at_cg, state)
    plus = _diagnostics(off_axis, state)
    minus = _diagnostics(negative, state)
    axes = body_axes_for_state(state)
    expected_moment = cross(
        scale(axes.forward, 0.25), plus.fixed_lifting_surface_force_n
    )
    expected_pitch = dot(expected_moment, axes.right)
    expected_yaw = -dot(expected_moment, axes.up)
    inertia = state.mass * base["geometry"]["length_m"] ** 2 / 12.0

    assert plus.fixed_lifting_surface_station_x_m == 0.25
    assert minus.fixed_lifting_surface_station_x_m == -0.25
    assert math.isclose(
        plus.pitch_fixed_lifting_surface_moment_nm,
        expected_pitch,
        rel_tol=2e-14,
        abs_tol=2e-12,
    )
    assert math.isclose(
        plus.yaw_fixed_lifting_surface_moment_nm,
        expected_yaw,
        rel_tol=2e-14,
        abs_tol=2e-12,
    )
    assert cg.pitch_fixed_lifting_surface_moment_nm == 0.0
    assert cg.yaw_fixed_lifting_surface_moment_nm == 0.0
    assert math.isclose(
        plus.pitch_fixed_lifting_surface_moment_nm,
        -minus.pitch_fixed_lifting_surface_moment_nm,
        rel_tol=2e-12,
        abs_tol=2e-10,
    )
    assert math.isclose(
        plus.yaw_fixed_lifting_surface_moment_nm,
        -minus.yaw_fixed_lifting_surface_moment_nm,
        rel_tol=2e-12,
        abs_tol=2e-10,
    )
    assert math.isclose(
        plus.pitch_angular_acceleration_rad_s2 - cg.pitch_angular_acceleration_rad_s2,
        plus.pitch_fixed_lifting_surface_moment_nm / inertia,
        rel_tol=2e-12,
        abs_tol=2e-10,
    )
    assert math.isclose(
        plus.yaw_angular_acceleration_rad_s2 - cg.yaw_angular_acceleration_rad_s2,
        plus.yaw_fixed_lifting_surface_moment_nm / inertia,
        rel_tol=2e-12,
        abs_tol=2e-10,
    )


def test_fixed_lifting_surface_zero_alpha_or_q_is_zero_and_only_enters_specific_force() -> None:
    base = _candidate_config("cn_pl12")
    zero_alpha = _diagnostics(_with_fixed_lifting_multiplier(base, 2.0), _state())
    zero_q = _diagnostics(
        _with_fixed_lifting_multiplier(base, 2.0),
        replace(_state(pitch=0.2), velocity=(0.0, 0.0, 0.0)),
    )
    assert zero_alpha.fixed_lifting_surface_force_n == (0.0, 0.0, 0.0)
    assert zero_q.fixed_lifting_surface_force_n == (0.0, 0.0, 0.0)

    state = _state(pitch=0.1)
    without = _diagnostics(base, state)
    with_fixed = _diagnostics(_with_fixed_lifting_multiplier(base, 1.0), state)
    assert with_fixed.aero.cda0_m2 == without.aero.cda0_m2
    assert with_fixed.aero.cda_alpha_m2 == without.aero.cda_alpha_m2
    assert with_fixed.pitch_tail_force_n == without.pitch_tail_force_n
    assert with_fixed.pitch_tail_moment_nm == without.pitch_tail_moment_nm
    assert with_fixed.wind_normal_pitch_acceleration_g > without.wind_normal_pitch_acceleration_g
    assert with_fixed.pitch_angular_acceleration_rad_s2 == without.pitch_angular_acceleration_rad_s2
    basis = cg_wind_normal_basis(state, base)
    gravity = base["atmosphere"]["gravity_mps2"]
    expected_increment_g = dot(with_fixed.fixed_lifting_surface_force_n, basis.up) / (state.mass * gravity)
    assert math.isclose(
        with_fixed.wind_normal_pitch_acceleration_g - without.wind_normal_pitch_acceleration_g,
        expected_increment_g,
        rel_tol=2e-14,
        abs_tol=2e-15,
    )
    for index in range(3):
        assert math.isclose(
            with_fixed.total_force_n[index] - without.total_force_n[index],
            with_fixed.fixed_lifting_surface_force_n[index],
            rel_tol=2e-14,
            abs_tol=2e-10,
        )


def test_split_tail_k1_matches_frozen_combined_chi_pointwise() -> None:
    rng = random.Random(0x51A17)
    deterministic = [
        (147.87, 20_000.0, 0.01, -0.015, 0.002, -0.003),
        (80.0, 180_000.0, 0.5, -0.4, -0.5, 0.4),
    ]
    random_states = [
        (
            rng.uniform(60.0, 220.0),
            rng.uniform(100.0, 250_000.0),
            rng.uniform(-0.7, 0.7),
            rng.uniform(-0.7, 0.7),
            rng.uniform(-0.6, 0.6),
            rng.uniform(-0.6, 0.6),
        )
        for _ in range(500)
    ]
    saw_capped = False
    saw_uncapped = False
    for mass, q_inf, alpha_p, alpha_y, delta_p, delta_y in deterministic + random_states:
        kwargs = {
            "mass_kg": mass,
            "gravity_mps2": 9.80665,
            "dynamic_pressure_pa": q_inf,
            "q_base_pa": 153125.00225,
            "acceleration_authority_g": 32.0,
            "pitch_alpha_rad": alpha_p,
            "yaw_alpha_rad": alpha_y,
            "pitch_delta_rad": delta_p,
            "yaw_delta_rad": delta_y,
            "pitch_authority_reference_rad": math.radians(20.0),
            "yaw_authority_reference_rad": math.radians(20.0),
        }
        old = combined_empirical_tail_force_reference(**kwargs)
        new = split_empirical_tail_force(**kwargs)
        saw_capped |= new.cap_active
        saw_uncapped |= not new.cap_active
        for field in (
            "pitch_alpha_slope_n_per_rad", "yaw_alpha_slope_n_per_rad",
            "pitch_delta_slope_n_per_rad", "yaw_delta_slope_n_per_rad",
            "pitch_alpha_force_n", "yaw_alpha_force_n",
            "pitch_delta_force_n", "yaw_delta_force_n",
            "pitch_pre_cap_force_n", "yaw_pre_cap_force_n",
            "pitch_force_n", "yaw_force_n", "force_cap_n", "cap_scale",
        ):
            assert math.isclose(
                getattr(new, field), getattr(old, field), rel_tol=3e-15, abs_tol=2e-11
            ), field
        arm = -0.12
        assert math.isclose(new.pitch_force_n * arm, old.pitch_force_n * arm, rel_tol=3e-15, abs_tol=2e-12)
        assert math.isclose(new.yaw_force_n * arm, old.yaw_force_n * arm, rel_tol=3e-15, abs_tol=2e-12)
        if new.force_cap_n > 0.0:
            assert math.isclose(-new.pitch_force_n / new.force_cap_n, -old.pitch_force_n / old.force_cap_n, rel_tol=3e-15, abs_tol=2e-15)
            assert math.isclose(-new.yaw_force_n / new.force_cap_n, -old.yaw_force_n / old.force_cap_n, rel_tol=3e-15, abs_tol=2e-15)
        assert new.cap_active == old.cap_active
    assert saw_capped and saw_uncapped


def test_split_tail_candidate_path_matches_combined_reference_for_random_rates_and_states() -> None:
    config = _candidate_config("cn_pl12")
    rng = random.Random(0xB0D7)
    for _ in range(100):
        state = replace(
            _state(
                pitch=rng.uniform(-0.35, 0.35),
                yaw=rng.uniform(-0.35, 0.35),
                pitch_rate=rng.uniform(-4.0, 4.0),
                yaw_rate=rng.uniform(-4.0, 4.0),
                pitch_fin=rng.uniform(-0.5, 0.5),
                yaw_fin=rng.uniform(-0.5, 0.5),
            ),
            position=(0.0, rng.uniform(0.0, 15_000.0), 0.0),
            velocity=(rng.uniform(150.0, 1_200.0), rng.uniform(-100.0, 100.0), rng.uniform(-100.0, 100.0)),
            mass=rng.uniform(70.0, 200.0),
        )
        observed = _diagnostics(config, state)
        empirical = config["aerodynamics"]["empirical_fin_authority"]
        old = combined_empirical_tail_force_reference(
            mass_kg=state.mass,
            gravity_mps2=config["atmosphere"]["gravity_mps2"],
            dynamic_pressure_pa=observed.aero.dynamic_pressure_pa,
            q_base_pa=config["aerodynamics"]["split_tail_candidate"]["q_base_pa"],
            acceleration_authority_g=config["aerodynamics"]["fins_lateral_acceleration_g"],
            pitch_alpha_rad=state.actual_pitch_fin_angle_rad - observed.pitch_tail_effective_incidence_rad,
            yaw_alpha_rad=state.actual_yaw_fin_angle_rad - observed.yaw_tail_effective_incidence_rad,
            pitch_delta_rad=state.actual_pitch_fin_angle_rad,
            yaw_delta_rad=state.actual_yaw_fin_angle_rad,
            pitch_authority_reference_rad=empirical["pitch_incidence_reference_rad"],
            yaw_authority_reference_rad=empirical["yaw_incidence_reference_rad"],
        )
        assert math.isclose(observed.pitch_tail_force_n, old.pitch_force_n, rel_tol=4e-15, abs_tol=2e-10)
        assert math.isclose(observed.yaw_tail_force_n, old.yaw_force_n, rel_tol=4e-15, abs_tol=2e-10)
        assert observed.tail_force_cap_active == old.cap_active
        assert math.isclose(observed.tail_force_cap_scale, old.cap_scale, rel_tol=4e-15, abs_tol=2e-15)
        axes = body_axes_for_state(state)
        _, _, _, pitch_normal, yaw_normal = _flow_at_station(
            state, axes, observed.tail_station_x_m, config
        )
        arm_vector = scale(axes.forward, observed.tail_station_x_m)
        expected_pitch_moment = dot(
            cross(arm_vector, scale(pitch_normal, old.pitch_force_n)), axes.right
        )
        expected_yaw_moment = -dot(
            cross(arm_vector, scale(yaw_normal, old.yaw_force_n)), axes.up
        )
        assert math.isclose(observed.pitch_tail_moment_nm, expected_pitch_moment, rel_tol=4e-15, abs_tol=2e-10)
        assert math.isclose(observed.yaw_tail_moment_nm, expected_yaw_moment, rel_tol=4e-15, abs_tol=2e-10)
        assert math.isclose(
            observed.pitch_tail_alpha_moment_nm,
            dot(cross(arm_vector, scale(pitch_normal, old.pitch_alpha_force_n)), axes.right),
            rel_tol=4e-15, abs_tol=2e-10,
        )
        assert math.isclose(
            observed.yaw_tail_delta_moment_nm,
            -dot(cross(arm_vector, scale(yaw_normal, old.yaw_delta_force_n)), axes.up),
            rel_tol=4e-15, abs_tol=2e-10,
        )
        if old.force_cap_n > 0.0:
            assert math.isclose(observed.pitch_tail_authority_fraction, -old.pitch_force_n / old.force_cap_n, rel_tol=4e-15, abs_tol=2e-15)
            assert math.isclose(observed.yaw_tail_authority_fraction, -old.yaw_force_n / old.force_cap_n, rel_tol=4e-15, abs_tol=2e-15)


def test_split_tail_sign_and_radial_cap() -> None:
    common = {
        "mass_kg": 100.0,
        "gravity_mps2": 9.80665,
        "dynamic_pressure_pa": 100_000.0,
        "q_base_pa": 150_000.0,
        "acceleration_authority_g": 30.0,
        "pitch_authority_reference_rad": 0.2,
        "yaw_authority_reference_rad": 0.2,
    }
    signed = split_empirical_tail_force(
        **common,
        pitch_alpha_rad=0.01,
        yaw_alpha_rad=0.0,
        pitch_delta_rad=0.02,
        yaw_delta_rad=0.0,
    )
    assert signed.pitch_alpha_force_n > 0.0
    assert signed.pitch_delta_force_n < 0.0
    assert signed.pitch_force_n < 0.0
    capped = split_empirical_tail_force(
        **common,
        pitch_alpha_rad=1.0,
        yaw_alpha_rad=1.0,
        pitch_delta_rad=-1.0,
        yaw_delta_rad=-1.0,
    )
    assert capped.cap_active
    assert math.isclose(
        math.hypot(capped.pitch_force_n, capped.yaw_force_n),
        capped.force_cap_n,
        rel_tol=2e-16,
    )


def test_split_tail_k_alpha_monotonically_reduces_local_static_equilibrium() -> None:
    roots = []
    epsilon = 1e-5
    fin_angle = 0.005
    for k_alpha in (1.0, 1.4, 1.8, 2.5):
        config = _candidate_config("cn_pl12")
        config["aerodynamics"]["split_tail_candidate"]["tail_alpha_force_multiplier"] = k_alpha
        zero = _diagnostics(config, _state(pitch_fin=fin_angle))
        plus = _diagnostics(config, _state(pitch=epsilon, pitch_fin=fin_angle))
        minus = _diagnostics(config, _state(pitch=-epsilon, pitch_fin=fin_angle))
        stiffness = (plus.pitch_total_moment_nm - minus.pitch_total_moment_nm) / (2.0 * epsilon)
        assert stiffness < 0.0
        roots.append(abs(-zero.pitch_total_moment_nm / stiffness))
    assert all(left > right for left, right in zip(roots, roots[1:]))

def test_body_normal_force_uses_circle_area_cn2_and_ignores_wing_multiplier() -> None:
    config = _candidate_config()
    config["aerodynamics"]["fins_lateral_acceleration_g"] = 0.0
    state = _state(pitch=0.1)
    observed = _diagnostics(config, state)
    diameter = config["geometry"]["caliber_m"]
    expected_area = math.pi * diameter * diameter / 4.0
    expected_force = (
        observed.aero.dynamic_pressure_pa
        * expected_area
        * 2.0
        * observed.aero.pitch_alpha_rad
    )

    assert observed.aero.reference_area_m2 == expected_area
    assert observed.body_reference_area_m2 == expected_area
    assert observed.aero.lift_coefficient_pitch == 2.0 * observed.aero.pitch_alpha_rad
    assert abs(observed.pitch_body_normal_force_n - expected_force) < 1e-9

    changed = copy.deepcopy(config)
    changed["geometry"]["wing_area_multiplier"] *= 100.0
    changed["drag_model"]["lift_area_scale"] *= 17.0
    changed_observed = _diagnostics(changed, state)
    assert changed_observed.aero.reference_area_m2 == expected_area
    assert changed_observed.pitch_body_normal_force_n == observed.pitch_body_normal_force_n


def test_body_cp_local_flow_generates_nominal_cnq_and_cmq() -> None:
    config = _candidate_config()
    config["aerodynamics"]["fins_lateral_acceleration_g"] = 0.0
    pitch_alpha = _diagnostics(config, _state(pitch=0.1))
    yaw_alpha = _diagnostics(config, _state(yaw=0.1))
    pitch_rate = _diagnostics(config, _state(pitch_rate=1.0))
    yaw_rate = _diagnostics(config, _state(yaw_rate=1.0))
    combined = _diagnostics(config, _state(pitch=0.1, pitch_rate=1.0))

    assert pitch_alpha.body_cn_alpha_per_rad == 2.0
    assert pitch_alpha.body_cp_cg_arm_over_diameter == 3.0
    assert pitch_alpha.body_cm_alpha_per_rad == 6.0
    assert pitch_alpha.body_cn_q == -12.0
    assert pitch_alpha.body_cm_q == -36.0
    assert pitch_alpha.pitch_body_static_moment_nm > 0.0
    assert pitch_alpha.pitch_body_rate_moment_nm == 0.0
    assert pitch_rate.pitch_body_static_moment_nm == 0.0
    assert pitch_rate.pitch_body_rate_moment_nm < 0.0
    assert pitch_alpha.pitch_body_static_moment_nm == yaw_alpha.yaw_body_static_moment_nm
    assert pitch_rate.pitch_body_rate_moment_nm == yaw_rate.yaw_body_rate_moment_nm
    assert combined.pitch_body_normal_force_n < pitch_alpha.pitch_body_normal_force_n

    diameter = config["geometry"]["caliber_m"]
    area = math.pi * diameter * diameter / 4.0
    speed = 300.0
    epsilon = 1e-5
    plus = _diagnostics(config, _state(pitch_rate=epsilon))
    minus = _diagnostics(config, _state(pitch_rate=-epsilon))
    force_rate_derivative = (
        plus.pitch_body_normal_force_n - minus.pitch_body_normal_force_n
    ) / (2.0 * epsilon)
    moment_rate_derivative = (
        plus.pitch_body_total_moment_nm - minus.pitch_body_total_moment_nm
    ) / (2.0 * epsilon)
    force_scale = plus.aero.dynamic_pressure_pa * area * diameter / (2.0 * speed)
    moment_scale = force_scale * diameter
    assert abs(force_rate_derivative / force_scale + 12.0) < 1e-9
    assert abs(moment_rate_derivative / moment_scale + 36.0) < 1e-9


def test_candidate_body_cp_force_and_cross_product_moment_share_one_action() -> None:
    config = _candidate_config()
    config["aerodynamics"]["fins_lateral_acceleration_g"] = 0.0
    diagnostics = _diagnostics(config, _state(pitch=1e-5))
    expected_arm = (
        diagnostics.body_cp_cg_arm_over_diameter
        * config["geometry"]["caliber_m"]
    )
    observed_arm = (
        diagnostics.pitch_body_total_moment_nm
        / diagnostics.pitch_body_normal_force_n
    )
    assert abs(observed_arm - expected_arm) < 1e-9


def test_candidate_zero_rate_and_pure_pitch_rate_body_terms() -> None:
    config = _candidate_config()
    config["aerodynamics"]["fins_lateral_acceleration_g"] = 0.0
    zero_rate = _diagnostics(config, _state(pitch=0.05))
    pure_rate = _diagnostics(config, _state(pitch_rate=0.2))
    zero_speed = _diagnostics(
        config,
        replace(_state(), velocity=(0.0, 0.0, 0.0)),
    )

    assert zero_rate.pitch_body_rate_moment_nm == 0.0
    assert pure_rate.pitch_body_normal_force_n < 0.0
    assert pure_rate.pitch_body_total_moment_nm < 0.0
    assert zero_speed.pitch_body_normal_force_n == 0.0
    assert zero_speed.pitch_body_total_moment_nm == 0.0


def test_candidate_quaternion_composes_body_pitch_yaw_and_stays_orthonormal() -> None:
    config = _candidate_config()
    config["aerodynamics"]["natural_lift_enabled"] = False
    config["aerodynamics"]["fins_lateral_acceleration_g"] = 0.0
    propulsion = PiecewisePropulsion.from_config(config)
    scalar_axes = body_axes(0.2, -0.3)
    quaternion_axes = body_axes_from_quaternion(quaternion_from_pitch_yaw(0.2, -0.3))
    assert max(
        abs(a - b)
        for scalar_vector, quaternion_vector in zip(
            (scalar_axes.forward, scalar_axes.up, scalar_axes.right),
            (quaternion_axes.forward, quaternion_axes.up, quaternion_axes.right),
        )
        for a, b in zip(scalar_vector, quaternion_vector)
    ) < 1e-15
    state = replace(
        _state(pitch_rate=0.4),
        velocity=(0.0, 0.0, 0.0),
        orientation_quaternion=quaternion_from_pitch_yaw(0.0, 0.0),
    )
    dt_s = 0.005
    for index in range(100):
        state = rk4_step_h2(state, index * dt_s, dt_s, config, propulsion, powered=False)
    state = replace(state, pitch_rate=0.0, yaw_rate=0.3)
    for index in range(100, 200):
        state = rk4_step_h2(state, index * dt_s, dt_s, config, propulsion, powered=False)

    quaternion = state.orientation_quaternion
    assert quaternion is not None
    assert abs(math.sqrt(sum(value * value for value in quaternion)) - 1.0) < 1e-14
    expected = quaternion_multiply(
        (math.cos(0.1), 0.0, 0.0, math.sin(0.1)),
        (math.cos(0.075), 0.0, -math.sin(0.075), 0.0),
    )
    if sum(a * b for a, b in zip(quaternion, expected)) < 0.0:
        expected = tuple(-value for value in expected)
    assert max(abs(a - b) for a, b in zip(quaternion, normalize_quaternion(expected))) < 1e-10
    axes = body_axes_from_quaternion(quaternion)
    vectors = (axes.forward, axes.up, axes.right)
    for vector in vectors:
        assert abs(sum(value * value for value in vector) - 1.0) < 1e-14
    assert abs(sum(a * b for a, b in zip(axes.forward, axes.up))) < 1e-14
    assert abs(sum(a * b for a, b in zip(axes.forward, axes.right))) < 1e-14
    assert abs(sum(a * b for a, b in zip(axes.up, axes.right))) < 1e-14


def test_candidate_pitch_yaw_diagonal_uses_empirical_radial_authority_allocation() -> None:
    config = _candidate_config()
    pitch_limit = config["control"]["fin_actuator_travel"]["pitch_limit_rad"]
    yaw_limit = config["control"]["fin_actuator_travel"]["yaw_limit_rad"]
    state = _state(pitch_fin=pitch_limit, yaw_fin=yaw_limit)
    diagnostics = _diagnostics(config, state)
    schedule = base_indicated_speed_schedule(
        diagnostics.aero.dynamic_pressure_pa,
        config,
    )
    full_force = (
        config["aerodynamics"]["fins_lateral_acceleration_g"]
        * schedule.fin_force_scale
        * state.mass
        * config["atmosphere"]["gravity_mps2"]
    )
    pitch_fraction = -diagnostics.pitch_tail_force_n / full_force
    yaw_fraction = -diagnostics.yaw_tail_force_n / full_force
    assert abs(pitch_fraction - yaw_fraction) < 1e-12
    assert abs(math.hypot(pitch_fraction, yaw_fraction) - 1.0) < 1e-12


def test_candidate_cg_wind_normal_basis_is_orthonormal_and_positive_yaw_right() -> None:
    config = _candidate_config()
    state = replace(
        _state(),
        velocity=(300.0, 20.0, 10.0),
        orientation_quaternion=quaternion_from_pitch_yaw(0.2, -0.1),
    )
    basis = cg_wind_normal_basis(state, config)
    vectors = (basis.forward, basis.up, basis.right)
    for vector in vectors:
        assert abs(sum(value * value for value in vector) - 1.0) < 1e-14
    for left, right in ((basis.forward, basis.up), (basis.forward, basis.right), (basis.up, basis.right)):
        assert abs(sum(a * b for a, b in zip(left, right))) < 1e-14
    zero_basis = cg_wind_normal_basis(_state(), config)
    assert zero_basis.right == (0.0, 0.0, 1.0)


def test_candidate_wind_normal_feedback_excludes_drag_but_includes_thrust_at_aoa() -> None:
    config = _candidate_config()
    config["aerodynamics"]["natural_lift_enabled"] = False
    config["aerodynamics"]["fins_lateral_acceleration_g"] = 0.0
    level = _diagnostics(config, _state())
    assert abs(level.wind_normal_pitch_acceleration_g) < 1e-14
    assert abs(level.wind_normal_yaw_acceleration_g) < 1e-14

    state = replace(
        _state(pitch=0.1),
        orientation_quaternion=quaternion_from_pitch_yaw(0.1, 0.0),
    )
    powered = forces_for_state_h2(
        state,
        0.0,
        config,
        PiecewisePropulsion.from_config(config),
        powered=True,
    )
    assert powered.wind_normal_pitch_acceleration_g > 0.0


def test_candidate_zero_pn_horizontal_command_adds_one_g_gravity_compensation() -> None:
    config = _candidate_config()
    state = replace(_state(), orientation_quaternion=quaternion_from_pitch_yaw(0.0, 0.0))
    target = TargetState((1000.0, 3000.0, 0.0), (0.0, 0.0, 0.0))
    output = guidance_command(state, target, 0.0, config, enabled=True)
    assert output.commanded_acceleration_mps2 == (0.0, 0.0, 0.0)
    assert abs(output.wind_normal_specific_force_command_g[0] - 1.0) < 1e-14
    assert output.wind_normal_specific_force_command_g[1] == 0.0
    assert output.controller_specific_force_command_g == output.wind_normal_specific_force_command_g


def test_continuous_closest_approach_uses_segment_interior() -> None:
    samples = [
        {
            "time_s": 0.0,
            "position_m": [0.0, 0.0, 0.0],
            "target_position_m": [-1.0, 1.0, 0.0],
            "velocity_mps": [0.0, 0.0, 0.0],
            "target_velocity_mps": [2.0, 0.0, 0.0],
            "distance_to_target_m": math.sqrt(2.0),
            "closing_speed_mps": 0.0,
        },
        {
            "time_s": 1.0,
            "position_m": [0.0, 0.0, 0.0],
            "target_position_m": [1.0, 1.0, 0.0],
            "velocity_mps": [0.0, 0.0, 0.0],
            "target_velocity_mps": [2.0, 0.0, 0.0],
            "distance_to_target_m": math.sqrt(2.0),
            "closing_speed_mps": 0.0,
        },
    ]
    closest = continuous_closest_approach(samples)
    assert closest["continuous_minimum_distance_m"] == 1.0
    assert closest["time_at_minimum_distance_s"] == 0.5
    assert closest["closing_speed_at_minimum_distance_mps"] == 0.0


def test_signed_aft_tail_force_and_moment_share_one_physical_force() -> None:
    config = _candidate_config()
    positive_pitch = _diagnostics(config, _state(pitch_fin=0.1))
    positive_yaw = _diagnostics(config, _state(yaw_fin=0.1))

    assert positive_pitch.tail_station_x_m < 0.0
    assert positive_pitch.pitch_tail_force_n < 0.0
    assert positive_pitch.pitch_tail_moment_nm > 0.0
    assert positive_pitch.pitch_angular_acceleration_rad_s2 > 0.0
    assert abs(
        positive_pitch.pitch_tail_moment_nm / positive_pitch.pitch_tail_force_n
        - positive_pitch.tail_station_x_m
    ) < 1e-12
    assert positive_pitch.pitch_tail_force_n == positive_yaw.yaw_tail_force_n
    assert positive_pitch.pitch_tail_moment_nm == positive_yaw.yaw_tail_moment_nm
    assert positive_pitch.pitch_angular_acceleration_rad_s2 == positive_yaw.yaw_angular_acceleration_rad_s2


def test_candidate_tail_static_and_rate_incidence_remain_traceable() -> None:
    config = _candidate_config()
    alpha = _diagnostics(config, _state(pitch=0.1))
    rate = _diagnostics(config, _state(pitch_rate=1.0))

    assert alpha.pitch_tail_effective_incidence_rad < 0.0
    assert alpha.pitch_tail_force_n > 0.0
    assert alpha.pitch_tail_moment_nm < 0.0
    assert rate.pitch_tail_rate_incidence_rad > 0.0
    assert rate.pitch_tail_effective_incidence_rad < 0.0
    assert rate.pitch_tail_moment_nm < 0.0
    assert rate.pitch_tail_rate_damping_per_s > 0.0


def test_candidate_has_no_synthetic_critical_residual_and_exports_audit_fields() -> None:
    config = _candidate_config()
    diagnostics = _diagnostics(config, _state(pitch=0.05, pitch_rate=0.2, pitch_fin=0.03))
    assert diagnostics.pitch_residual_rate_damping_per_s == 0.0
    assert diagnostics.yaw_residual_rate_damping_per_s == 0.0
    assert diagnostics.pitch_residual_damping_moment_nm == 0.0
    assert diagnostics.yaw_residual_damping_moment_nm == 0.0
    assert diagnostics.pitch_total_moment_nm == (
        diagnostics.pitch_body_static_moment_nm
        + diagnostics.pitch_body_rate_moment_nm
        + diagnostics.pitch_tail_moment_nm
    )

    config["performance"]["lifetime_s"] = 0.02
    config["numerics"]["max_steps"] = 10
    case = load_cases(ROOT / "configs" / "cases.yaml")[0]
    sample = H2Simulator(config).run(case)["samples"][-1]
    required = {
        "pitch_body_normal_force_n",
        "pitch_body_static_moment_nm",
        "pitch_body_rate_moment_nm",
        "pitch_tail_force_n",
        "pitch_tail_moment_nm",
        "pitch_total_moment_nm",
        "body_cp_cg_arm_over_diameter",
        "body_cm_alpha_per_rad",
        "body_cm_q",
        "pitch_residual_rate_damping_per_s",
    }
    assert required.issubset(sample)
    assert sample["pitch_residual_rate_damping_per_s"] == 0.0


def test_candidate_adapter_rejects_nonfinite_lambda() -> None:
    defaults = _defaults(BODY_CM_TAIL_FORCE_PLANT)
    defaults["body_cm_candidate"]["cp_cg_arm_over_diameter"] = float("nan")
    try:
        build_h2_candidate_config(_profile(), defaults)
    except ValueError as exc:
        assert "positive and finite" in str(exc)
    else:
        raise AssertionError("non-finite lambda must be rejected")


def test_candidate_adapter_rejects_nonfinite_fixed_lift_station() -> None:
    defaults = _defaults(BODY_CM_TAIL_FORCE_PLANT)
    defaults["body_cm_candidate"]["fixed_lifting_surface_station_x_m"] = float("nan")
    try:
        build_h2_candidate_config(_profile(), defaults)
    except ValueError as exc:
        assert "must be finite" in str(exc)
    else:
        raise AssertionError("non-finite fixed-lift station must be rejected")


def test_candidate_actuator_keeps_fin_angle_without_direct_g_feedthrough() -> None:
    config = _candidate_config()
    config["control"]["pid"].update({"p": 1.0, "i": 0.0, "d": 0.0})
    updates = update_control_feedback(
        _state(),
        (100.0, 0.0),
        config,
        config["control"]["actuator_time_constant_s"],
        enabled=True,
        plant_diagnostics=_diagnostics(config, _state()),
    )
    assert updates["actual_pitch_fin_angle_rad"] > 0.0
    assert updates["commanded_pitch_rate_rad_s"] > 0.0
    assert updates["pitch_rate_error_rad_s"] > 0.0
    assert updates["actual_pitch_acceleration_g"] == 0.0
    assert updates["actual_yaw_acceleration_g"] == 0.0


def test_candidate_rate_inner_loop_uses_body_rate_feedback_with_aft_tail_sign() -> None:
    config = _candidate_config()
    config["control"]["pid"].update({"p": 0.02, "i": 0.0, "d": 0.0})
    positive = update_control_feedback(
        _state(),
        (5.0, 0.0),
        config,
        0.01,
        enabled=True,
        plant_diagnostics=_diagnostics(config, _state()),
    )
    opposing = update_control_feedback(
        _state(pitch_rate=0.2),
        (0.0, 0.0),
        config,
        0.01,
        enabled=True,
        plant_diagnostics=_diagnostics(config, _state(pitch_rate=0.2)),
    )

    assert positive["commanded_pitch_rate_rad_s"] == 0.1
    assert positive["actual_pitch_fin_angle_rad"] > 0.0
    assert opposing["pitch_rate_error_rad_s"] < 0.0
    assert opposing["actual_pitch_fin_angle_rad"] < 0.0

    positive_tail = _diagnostics(
        config,
        replace(_state(), actual_pitch_fin_angle_rad=positive["actual_pitch_fin_angle_rad"]),
    )
    assert positive_tail.pitch_tail_force_n < 0.0
    assert positive_tail.pitch_tail_moment_nm > 0.0
    assert positive_tail.pitch_angular_acceleration_rad_s2 > 0.0


def test_candidate_acceleration_step_records_initial_reverse_then_positive_response() -> None:
    config = _candidate_config()
    propulsion = PiecewisePropulsion.from_config(config)
    state = _state()
    dt_s = 0.002
    initial_tail_force_n = None
    maximum_positive_body_g = 0.0
    maximum_positive_total_g = 0.0
    for index in range(750):
        before = _diagnostics(config, state)
        state = replace(
            state,
            measured_pitch_normal_g=before.wind_normal_pitch_acceleration_g,
            measured_yaw_normal_g=before.wind_normal_yaw_acceleration_g,
        )
        updates = update_control_feedback(
            state,
            (5.0, 0.0),
            config,
            dt_s,
            enabled=True,
            plant_diagnostics=before,
        )
        state_for_step = replace(state, **updates)
        controlled = _diagnostics(config, state_for_step)
        if initial_tail_force_n is None:
            initial_tail_force_n = controlled.pitch_tail_force_n
        maximum_positive_body_g = max(
            maximum_positive_body_g, controlled.pitch_body_aoa_force_g
        )
        maximum_positive_total_g = max(
            maximum_positive_total_g, controlled.pitch_normal_acceleration_g
        )
        state = rk4_step_h2(
            state_for_step,
            index * dt_s,
            dt_s,
            config,
            propulsion,
            powered=False,
        )
        assert state_is_finite(state)

    assert initial_tail_force_n is not None and initial_tail_force_n < 0.0
    assert maximum_positive_body_g > 0.0
    assert maximum_positive_total_g > 0.0
    assert state.pitch_rate > 0.0


def test_candidate_local_linearization_has_restoring_stiffness_and_damping() -> None:
    config = _candidate_config()
    epsilon = 1e-5
    stiffness = (
        _diagnostics(config, _state(pitch=epsilon)).pitch_total_moment_nm
        - _diagnostics(config, _state(pitch=-epsilon)).pitch_total_moment_nm
    ) / (2.0 * epsilon)
    rate_derivative = (
        _diagnostics(config, _state(pitch_rate=epsilon)).pitch_total_moment_nm
        - _diagnostics(config, _state(pitch_rate=-epsilon)).pitch_total_moment_nm
    ) / (2.0 * epsilon)

    assert stiffness < 0.0
    assert rate_derivative < 0.0


def test_candidate_short_fixed_fin_integration_is_finite_and_bounded() -> None:
    config = _candidate_config()
    propulsion = PiecewisePropulsion.from_config(config)
    state = _state(pitch_fin=0.05)
    dt_s = 0.001
    maximum_abs_pitch = 0.0
    maximum_abs_pitch_rate = 0.0
    for index in range(500):
        state = rk4_step_h2(
            state, index * dt_s, dt_s, config, propulsion, powered=False
        )
        assert state_is_finite(state)
        maximum_abs_pitch = max(maximum_abs_pitch, abs(state.pitch))
        maximum_abs_pitch_rate = max(maximum_abs_pitch_rate, abs(state.pitch_rate))

    assert maximum_abs_pitch < math.pi / 2.0
    assert maximum_abs_pitch_rate < 10.0
