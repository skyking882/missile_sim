from pathlib import Path

from aim120_model import EffectiveControllerEnvelope, H2Simulator, __version__, load_cases, load_model_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_model_config(ROOT / "configs" / "aim120a_v1.json")
CASES = load_cases(ROOT / "configs" / "aim120a_v1_cases.json")


def test_frozen_release_version_is_consistent():
    assert __version__ == "1.0.1"
    assert CONFIG["release_version"] == "1.0.0"


def test_frozen_trajectory_regression_cases():
    expected_times = {
        "head_on_10deg_12km": 10.553325,
        "head_on_38deg_15km": 17.242967,
    }
    simulator = H2Simulator(CONFIG)
    for case in CASES:
        result = simulator.run(case)
        assert result["event_type"] == "fuse"
        assert abs(result["terminal_time_s"] - expected_times[case["name"]]) < 0.001
        assert abs(result["samples"][-1]["distance_to_target_m"] - 12.0) < 0.001


def test_frozen_effective_controller_mapping():
    parameters = CONFIG["effective_controller"]
    controller = EffectiveControllerEnvelope(
        gain=parameters["gain"],
        authority_fraction=parameters["authority_fraction"],
    )
    authority = CONFIG["aerodynamics"]["fins_lateral_acceleration_g"]
    expected = parameters["gain"] * 10.0
    assert abs(controller.predict_current_g_magnitude(10.0, authority) - expected) < 1e-12
    assert abs(controller.predict_signed_effective_output(-10.0, authority) + expected) < 1e-12
