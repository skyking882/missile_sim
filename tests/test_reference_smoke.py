from pathlib import Path

from aim120_model.config import load_cases, load_model_config
from aim120_model.simulator import H1Simulator


ROOT = Path(__file__).parents[1]


def test_all_local_cases_run_without_numerical_failure():
    config = load_model_config(ROOT / "configs" / "aim120a_statshark.yaml")
    cases = load_cases(ROOT / "configs" / "cases.yaml")
    simulator = H1Simulator(config)
    for case in cases:
        result = simulator.run(case)
        assert result["model_label"] == "local_candidate_H1"
        assert result["samples"]
        assert result["event_type"] in {"fuse", "ground", "lifetime", "max_distance"}
        required = {
            "time_s", "position_m", "velocity_mps", "mass_kg", "thrust_n", "drag_n",
            "mach", "angle_of_attack_rad", "commanded_acceleration_mps2", "actual_overload_g",
            "pitch_rad", "yaw_rad", "distance_to_target_m", "current_gain",
        }
        assert required.issubset(result["samples"][-1])

