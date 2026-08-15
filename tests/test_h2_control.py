from pathlib import Path

from aim120_model.config import load_model_config
from aim120_model.control import update_control_feedback
from aim120_model.dynamics import SimState


CONFIG = load_model_config(Path(__file__).parents[1] / "configs" / "aim120a_h2.yaml")


def test_physical_normal_feedback_and_zero_fin_authority_are_separate():
    state = SimState(
        (0.0, 3000.0, 0.0),
        (300.0, 0.0, 0.0),
        0.0, 0.0, 0.0, 0.0, 147.87,
        measured_pitch_normal_g=5.0,
        measured_yaw_normal_g=-2.0,
    )
    updates = update_control_feedback(
        state,
        (0.0, 0.0),
        CONFIG,
        0.02,
        enabled=True,
        authority_scale=0.0,
        feedback_measurement="physical_normal_g",
    )
    assert updates["pitch_pid_output"] < 0.0
    assert updates["yaw_pid_output"] > 0.0
    assert updates["pitch_requested_fin_command"] < 0.0
    assert updates["yaw_requested_fin_command"] > 0.0
    assert updates["pitch_fin_command"] == 0.0
    assert updates["yaw_fin_command"] == 0.0


def test_disabled_controller_clears_actuator_and_pid_telemetry():
    state = SimState((0.0, 3000.0, 0.0), (300.0, 0.0, 0.0), 0.0, 0.0, 0.0, 0.0, 147.87)
    updates = update_control_feedback(state, (10.0, -10.0), CONFIG, 0.02, enabled=False)
    assert updates["pitch_pid_output"] == 0.0
    assert updates["yaw_pid_output"] == 0.0
    assert updates["pitch_fin_command"] == 0.0
    assert updates["yaw_fin_command"] == 0.0
