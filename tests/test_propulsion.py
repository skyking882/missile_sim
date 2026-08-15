from pathlib import Path

from aim120_model.config import load_model_config
from aim120_model.propulsion import PiecewisePropulsion


CONFIG = Path(__file__).parents[1] / "configs" / "aim120a_statshark.yaml"


def test_piecewise_mass_boundaries():
    propulsion = PiecewisePropulsion.from_config(load_model_config(CONFIG))
    assert abs(propulsion.mass_at(0.0) - 147.87) < 1e-9
    assert abs(propulsion.mass_at(1.7) - (147.87 - 16.119995)) < 1e-9
    assert abs(propulsion.mass_at(7.0) - (147.87 - 16.119995 - 30.419998)) < 1e-9
    assert abs(propulsion.mass_at(80.0) - propulsion.mass_at(7.0)) < 1e-9


def test_stage_boundary_selection():
    propulsion = PiecewisePropulsion.from_config(load_model_config(CONFIG))
    assert propulsion.sample(1.699999).stage_name == "stage_1"
    assert propulsion.sample(1.7).stage_name == "stage_2"
    assert propulsion.sample(7.0).stage_name is None


def test_unpowered_variant_preserves_mass_and_force():
    propulsion = PiecewisePropulsion.from_config(load_model_config(CONFIG))
    sample = propulsion.sample(2.0, powered=False)
    assert sample.thrust_n == 0.0
    assert sample.mass_flow_kg_s == 0.0
    assert sample.mass_kg == propulsion.initial_mass_kg

