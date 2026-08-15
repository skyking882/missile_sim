import math
import sys
from pathlib import Path

from aim120_model.fin_dynamics_replay import fit_full_trajectory, replay_attitude
from aim120_model.h6_utils import rms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from validate_h6_synthetic_identifiability import ARM_M, DT_S, build_synthetic_trajectory  # noqa: E402


def test_complete_trajectory_holdout_is_not_used_for_refit():
    train = [build_synthetic_trajectory(0.5, 1.0, 45000.0, "TRAIN")]
    holdout = build_synthetic_trajectory(1.5, -1.0, 90000.0, "HOLDOUT")
    fit = fit_full_trajectory(train, distance_cm_to_stabilizer_m=ARM_M, max_step_s=DT_S)
    replayed = replay_attitude(holdout, fit["parameters"], distance_cm_to_stabilizer_m=ARM_M, max_step_s=DT_S)
    angle_rmse = math.degrees(rms(row["psi_residual_rad"] for row in replayed))
    assert angle_rmse <= 0.5
