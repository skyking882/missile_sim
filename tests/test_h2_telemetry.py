import copy
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
