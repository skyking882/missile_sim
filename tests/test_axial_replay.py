import math

from aim120_model.atmosphere import StandardAtmosphere
from aim120_model.axial_replay import replay_trajectory, trajectory_replay_metrics
from aim120_model.glide_drag_envelope import LogCdaEnvelope


def test_constant_cda_replay_recovers_closed_form_speed_history():
    atmosphere = StandardAtmosphere()
    envelope = LogCdaEnvelope.from_cda_knots([0.2, 4.5], [0.01, 0.01])
    altitude = 3000.0
    mass = 100.0
    initial_speed = 700.0
    coefficient = 0.5 * atmosphere.sample(altitude).density_kg_m3 * 0.01 / mass
    rows = []
    for index in range(21):
        time_s = index * 0.25
        speed = initial_speed / (1.0 + coefficient * initial_speed * time_s)
        rows.append({
            "trajectory_id": "closed_form",
            "source_kind": "synthetic_test",
            "time_s": time_s,
            "speed_mps": speed,
            "altitude_m": altitude,
            "flight_path_angle_rad": 0.0,
            "mass_kg": mass,
        })
    replayed = replay_trajectory(rows, envelope, max_step_s=0.02)
    metrics = trajectory_replay_metrics(replayed)
    assert metrics["speed_rmse_mps"] < 1.0e-7
    assert metrics["terminal_speed_error_mps"] < 1.0e-7
    assert max(float(row["gravity_cancellation_ratio"]) for row in replayed) == 0.0
    assert all(float(row["axial_drag_accel_mps2"]) > 0.0 for row in replayed)
