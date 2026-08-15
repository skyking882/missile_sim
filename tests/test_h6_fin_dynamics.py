import math
import sys
from pathlib import Path

from aim120_model.fin_dynamics import FinDynamicsParams, angular_acceleration, rk4_step
from aim120_model.fin_dynamics_replay import fit_full_trajectory, replay_attitude


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from validate_h6_synthetic_identifiability import (  # noqa: E402
    ARM_M,
    DT_S,
    TRUTH,
    build_synthetic_trajectory,
)


def test_zero_fin_force_has_zero_fin_moment_at_zero_state():
    params = FinDynamicsParams(0.5, 0.2, 0.4, q_ref_pa=1000.0)
    assert angular_acceleration(0.0, 0.0, 0.0, 0.0, ARM_M, 1000.0, params) == 0.0


def test_moment_arm_scales_force_term_without_changing_force_input():
    params = FinDynamicsParams(0.5, 0.0, 0.0, q_ref_pa=1000.0)
    short = angular_acceleration(0.0, 0.0, 0.0, 10.0, 0.1, 1000.0, params)
    long = angular_acceleration(0.0, 0.0, 0.0, 10.0, 0.2, 1000.0, params)
    assert abs(long - 2.0 * short) < 1.0e-12


def test_full_replay_fits_synthetic_effective_dynamics():
    trajectories = [
        build_synthetic_trajectory(0.5, 1.0, 45000.0, "T0"),
        build_synthetic_trajectory(1.0, -1.0, 55000.0, "T1"),
    ]
    fit = fit_full_trajectory(trajectories, distance_cm_to_stabilizer_m=ARM_M, max_step_s=DT_S)
    assert fit["identifiability"]["design_full_rank"]
    assert abs(fit["parameters"].b_f_ref - TRUTH.b_f_ref) / TRUTH.b_f_ref < 0.10
    assert abs(fit["parameters"].k_beta_ref - TRUTH.k_beta_ref) / TRUTH.k_beta_ref < 0.10
    assert abs(fit["parameters"].c_r_ref - TRUTH.c_r_ref) / TRUTH.c_r_ref < 0.10
