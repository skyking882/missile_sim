import math

from aim120_model.body_alpha2_drag import BodyAlpha2Parameters
from aim120_model.body_alpha2_replay import balanced_weighted_rmse, replay_body_alpha2, trajectory_replay_metrics


def _rows(gamma_deg=0.0, alpha_deg=0.0):
    return [
        {
            "case_id": "TEST",
            "trajectory_id": "TEST",
            "time_s": float(index),
            "speed_mps": 450.0,
            "altitude_m": 8000.0,
            "flight_path_angle_deg": gamma_deg,
            "alpha_deg": alpha_deg,
            "cx_aoa": 9.0,
            "mass_kg": 147.87,
        }
        for index in range(5)
    ]


def test_zero_thrust_axial_replay_decelerates_and_keeps_mass_constant():
    rows = _rows()
    replayed = replay_body_alpha2(rows, BodyAlpha2Parameters(0.019, 0.0, 0.0), max_step_s=0.05)
    assert replayed[-1]["predicted_speed_mps"] < replayed[0]["predicted_speed_mps"]
    assert all(row["mass_kg"] == 147.87 for row in replayed)
    assert all(row["cda_m2"] > 0.0 for row in replayed)


def test_climb_and_descent_gravity_terms_have_expected_sign():
    parameters = BodyAlpha2Parameters(0.019, 0.0, 0.0)
    climb = replay_body_alpha2(_rows(gamma_deg=5.0), parameters, max_step_s=0.05)
    descent = replay_body_alpha2(_rows(gamma_deg=-5.0), parameters, max_step_s=0.05)
    assert climb[0]["dVdt_pred_mps2"] < descent[0]["dVdt_pred_mps2"]


def test_known_alpha2_increment_is_recovered_in_replay_diagnostics():
    parameters = BodyAlpha2Parameters(0.019, 0.2, 0.01)
    plus = replay_body_alpha2(_rows(alpha_deg=2.0), parameters, max_step_s=0.05)
    minus = replay_body_alpha2(_rows(alpha_deg=-2.0), parameters, max_step_s=0.05)
    assert abs(plus[0]["cda_m2"] - minus[0]["cda_m2"]) < 1.0e-12
    assert plus[0]["cda_m2"] > parameters.cda0_1p5


def test_replay_metrics_report_zero_residual_for_matching_observations():
    parameters = BodyAlpha2Parameters(0.019, 0.0, 0.0)
    truth = replay_body_alpha2(_rows(), parameters, max_step_s=0.05)
    observed = []
    for source, row in zip(_rows(), truth):
        item = dict(source)
        item["speed_mps"] = row["predicted_speed_mps"]
        observed.append(item)
    replayed = replay_body_alpha2(observed, parameters, max_step_s=0.05)
    metrics = trajectory_replay_metrics(replayed)
    assert metrics["speed_rmse_mps"] < 1.0e-9


def test_weighted_rmse_balances_trajectories_before_sample_count():
    # First trajectory has two samples; the second has one.  The intended
    # value is sqrt((mean([1, 9]) + mean([4])) / 2) = sqrt(4.5).
    assert abs(balanced_weighted_rmse([[1.0, 3.0], [2.0]]) - math.sqrt(4.5)) < 1.0e-12
