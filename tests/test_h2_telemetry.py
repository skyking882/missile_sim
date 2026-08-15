import copy
import math
from pathlib import Path

from aim120_model.config import find_case, load_cases, load_model_config
from aim120_model.dynamics import SimState
from aim120_model.h2_dynamics import forces_for_state_h2
from aim120_model.h2_simulator import H2Simulator
from aim120_model.math3d import norm
from aim120_model.propulsion import PiecewisePropulsion


ROOT = Path(__file__).parents[1]
CONFIG = load_model_config(ROOT / "configs" / "aim120a_h2.yaml")


def test_physical_normal_load_excludes_axial_drag_from_lateral_load():
    propulsion = PiecewisePropulsion.from_config(CONFIG)
    state = SimState((0.0, 3000.0, 0.0), (300.0, 0.0, 0.0), 0.0, 0.0, 0.0, 0.0, 147.87)
    diagnostics = forces_for_state_h2(state, 0.0, CONFIG, propulsion, powered=False)
    assert diagnostics.axial_specific_force_g < 0.0
    assert abs(diagnostics.pitch_normal_acceleration_g) < 1e-12
    assert abs(diagnostics.yaw_normal_acceleration_g) < 1e-12
    assert abs(diagnostics.lateral_load_g) < 1e-12
    assert diagnostics.total_specific_force_g > 0.0
    assert diagnostics.drag_power_w < 0.0


def test_free_fall_has_gravity_in_acceleration_but_no_non_gravity_specific_force():
    propulsion = PiecewisePropulsion.from_config(CONFIG)
    state = SimState((0.0, 3000.0, 0.0), (0.0, 0.0, 0.0), 0.0, 0.0, 0.0, 0.0, 147.87)
    diagnostics = forces_for_state_h2(state, 0.0, CONFIG, propulsion, powered=False)
    assert norm(diagnostics.specific_force_mps2) < 1e-12
    assert abs(diagnostics.acceleration_mps2[1] + CONFIG["atmosphere"]["gravity_mps2"]) < 1e-12


def test_positive_body_aoa_generates_restoring_fin_moment_without_cyk():
    config = copy.deepcopy(CONFIG)
    config["aerodynamics"]["natural_lift_enabled"] = False
    config["aerodynamics"].pop("cy_k", None)
    config["control"]["fin_aoa_moment_enabled"] = True
    propulsion = PiecewisePropulsion.from_config(config)
    state = SimState((0.0, 3000.0, 0.0), (300.0, 0.0, 0.0), 0.1, 0.0, 0.0, 0.0, 147.87)
    diagnostics = forces_for_state_h2(state, 0.0, config, propulsion, powered=False)

    assert diagnostics.aero.pitch_alpha_rad > 0.0
    assert diagnostics.pitch_angular_acceleration_rad_s2 < 0.0


def test_body_axis_pid_feedback_is_distinct_from_trajectory_normal_telemetry_at_aoa():
    config = copy.deepcopy(CONFIG)
    config["aerodynamics"]["natural_lift_enabled"] = False
    propulsion = PiecewisePropulsion.from_config(config)
    state = SimState((0.0, 3000.0, 0.0), (300.0, 0.0, 0.0), 0.2, 0.0, 0.0, 0.0, 147.87)
    diagnostics = forces_for_state_h2(state, 0.0, config, propulsion, powered=False)

    assert diagnostics.pitch_normal_acceleration_g > 0.0
    assert abs(diagnostics.trajectory_pitch_normal_acceleration_g) < 1e-12
    assert diagnostics.lateral_load_g > diagnostics.trajectory_lateral_load_g


def test_fin_torque_plant_applies_one_tail_force_to_translation_and_moment():
    config = copy.deepcopy(CONFIG)
    config["aerodynamics"]["natural_lift_enabled"] = False
    config["control"]["plant_semantics"] = "fin_torque_body_aoa"
    config["control"]["base_indicated_speed_mode"] = "none"
    propulsion = PiecewisePropulsion.from_config(config)

    fin_only = SimState(
        (0.0, 3000.0, 0.0),
        (300.0, 0.0, 0.0),
        0.0, 0.0, 0.0, 0.0, 147.87,
        actual_pitch_fin_angle_rad=0.1,
    )
    fin_only_diagnostics = forces_for_state_h2(fin_only, 0.0, config, propulsion, powered=False)
    assert fin_only_diagnostics.trajectory_pitch_normal_acceleration_g > 0.0
    assert fin_only_diagnostics.pitch_angular_acceleration_rad_s2 > 0.0
    assert fin_only_diagnostics.pitch_fin_moment_equivalent_g > 0.0
    assert fin_only_diagnostics.pitch_body_aoa_force_g == 0.0

    body_aoa = SimState(
        (0.0, 3000.0, 0.0),
        (300.0, 0.0, 0.0),
        0.1, 0.0, 0.0, 0.0, 147.87,
    )
    aoa_diagnostics = forces_for_state_h2(body_aoa, 0.0, config, propulsion, powered=False)
    assert aoa_diagnostics.pitch_body_aoa_force_g == 0.0
    assert aoa_diagnostics.trajectory_pitch_normal_acceleration_g < 0.0
    assert aoa_diagnostics.pitch_angular_acceleration_rad_s2 < 0.0


def test_thin_plate_normal_force_uses_two_pi_slope_without_imputed_cap():
    config = copy.deepcopy(CONFIG)
    config["aerodynamics"].update({
        "natural_lift_enabled": True,
        "normal_force_model": "thin_plate_2pi",
        "cn_alpha_per_rad": 2.0 * math.pi,
        "normal_force_cap_enabled": False,
        "max_cy_at_aoa": 1.0,
    })
    propulsion = PiecewisePropulsion.from_config(config)

    linear = SimState(
        (0.0, 3000.0, 0.0),
        (300.0, 0.0, 0.0),
        0.1, 0.0, 0.0, 0.0, 147.87,
    )
    linear_diagnostics = forces_for_state_h2(linear, 0.0, config, propulsion, powered=False)
    assert abs(
        linear_diagnostics.aero.lift_coefficient_pitch
        - 2.0 * math.pi * linear_diagnostics.aero.pitch_alpha_rad
    ) < 1e-12
    assert linear_diagnostics.trajectory_pitch_normal_acceleration_g > 0.0

    high_alpha = SimState(
        (0.0, 3000.0, 0.0),
        (300.0, 0.0, 0.0),
        0.3, 0.0, 0.0, 0.0, 147.87,
    )
    high_alpha_diagnostics = forces_for_state_h2(
        high_alpha, 0.0, config, propulsion, powered=False
    )
    assert high_alpha_diagnostics.aero.lift_coefficient_pitch > 1.0
    assert abs(
        high_alpha_diagnostics.aero.lift_coefficient_pitch
        - 2.0 * math.pi * high_alpha_diagnostics.aero.pitch_alpha_rad
    ) < 1e-12


def test_fin_torque_plant_body_rate_reduces_tail_effective_incidence():
    config = copy.deepcopy(CONFIG)
    config["aerodynamics"]["natural_lift_enabled"] = False
    config["control"]["plant_semantics"] = "fin_torque_body_aoa"
    config["control"]["base_indicated_speed_mode"] = "none"
    propulsion = PiecewisePropulsion.from_config(config)
    common = {
        "position": (0.0, 3000.0, 0.0),
        "velocity": (300.0, 0.0, 0.0),
        "pitch": 0.0,
        "yaw": 0.0,
        "yaw_rate": 0.0,
        "mass": 147.87,
        "actual_pitch_fin_angle_rad": 0.1,
    }
    still = SimState(pitch_rate=0.0, **common)
    rotating = SimState(pitch_rate=50.0, **common)

    still_diagnostics = forces_for_state_h2(still, 0.0, config, propulsion, powered=False)
    rotating_diagnostics = forces_for_state_h2(rotating, 0.0, config, propulsion, powered=False)

    expected_rate_incidence = rotating.pitch_rate * config["aerodynamics"]["distance_cm_to_stabilizer_m"] / 300.0
    assert abs(rotating_diagnostics.pitch_tail_rate_incidence_rad - expected_rate_incidence) < 1e-12
    assert rotating_diagnostics.pitch_tail_effective_incidence_rad < still_diagnostics.pitch_tail_effective_incidence_rad
    assert rotating_diagnostics.pitch_fin_moment_equivalent_g < still_diagnostics.pitch_fin_moment_equivalent_g
    inertia_per_mass = config["geometry"]["length_m"] ** 2 / 12.0
    fin_limit = math.radians(config["aerodynamics"]["horizontal_fin_aoa_limit_deg"])
    expected_stiffness = (
        config["aerodynamics"]["fins_lateral_acceleration_g"]
        * config["atmosphere"]["gravity_mps2"]
        * config["aerodynamics"]["distance_cm_to_stabilizer_m"]
        / (inertia_per_mass * fin_limit)
    )
    expected_frequency = math.sqrt(expected_stiffness)
    expected_tail_damping = (
        expected_stiffness * config["aerodynamics"]["distance_cm_to_stabilizer_m"] / 300.0
    )
    assert abs(rotating_diagnostics.pitch_natural_frequency_rad_s - expected_frequency) < 1e-12
    assert abs(rotating_diagnostics.pitch_tail_rate_damping_per_s - expected_tail_damping) < 1e-12
    assert abs(
        rotating_diagnostics.pitch_residual_rate_damping_per_s
        - (2.0 * expected_frequency - expected_tail_damping)
    ) < 1e-12


def test_h2_output_uses_lateral_load_for_actual_overload():
    config = copy.deepcopy(CONFIG)
    config["performance"]["lifetime_s"] = 0.04
    config["numerics"]["max_steps"] = 10
    case = find_case(load_cases(ROOT / "configs" / "cases.yaml"), "power_only")
    result = H2Simulator(config).run(case)
    assert result["event_type"] == "lifetime"
    assert result["samples"]
    sample = result["samples"][-1]
    assert sample["actual_overload_g"] == sample["lateral_load_g"]
    assert sample["model_label"] == "local_candidate_H2"
    assert sample["control_model_version"] == "physical_feedback_v1"
