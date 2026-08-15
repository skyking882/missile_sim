import math

from aim120_model.sample_filters import LowGFilterSettings, apply_filter, normalize_sample


def _raw(time_s=3.0, lateral=0.0, alpha_deg=0.0, gamma_deg=0.0):
    speed = 300.0
    gamma = math.radians(gamma_deg)
    return {
        "time_s": time_s,
        "velocity_mps": [speed * math.cos(gamma), speed * math.sin(gamma), 0.0],
        "position_m": [0.0, 3000.0, 0.0],
        "mass_kg": 100.0,
        "thrust_n": 1000.0,
        "mach": 1.0,
        "angle_of_attack_rad": math.radians(alpha_deg),
        "lateral_load_g": lateral,
    }


def test_normalization_derives_pressure_gamma_and_alpha():
    row = normalize_sample(_raw(gamma_deg=-4.0, alpha_deg=3.0))
    assert abs(row["flight_path_angle_deg"] + 4.0) < 1.0e-10
    assert abs(row["alpha_total_deg"] - 3.0) < 1.0e-10
    assert row["dynamic_pressure_pa"] > 0.0


def test_default_low_g_filter_rejects_each_boundary_condition():
    settings = LowGFilterSettings(q_min_pa=100.0)
    for row, reason in (
        (normalize_sample(_raw(time_s=1.7)), "near_stage_1_boundary"),
        (normalize_sample(_raw(time_s=7.0)), "near_burn_end_boundary"),
        (normalize_sample(_raw(lateral=2.1)), "lateral_load_above_threshold"),
        (normalize_sample(_raw(alpha_deg=5.1)), "alpha_above_threshold"),
        (normalize_sample(_raw(gamma_deg=5.1)), "flight_path_above_threshold"),
    ):
        filtered = apply_filter(row, settings)
        assert not filtered["accepted"]
        assert reason in filtered["rejection_reasons"]


def test_threshold_edges_are_inclusive_except_q_min():
    settings = LowGFilterSettings(q_min_pa=100.0)
    row = apply_filter(normalize_sample(_raw(lateral=2.0, alpha_deg=5.0, gamma_deg=5.0)), settings)
    assert row["accepted"]
