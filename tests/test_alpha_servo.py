import math
import sys
from dataclasses import replace
from pathlib import Path

import aim120_model.alpha_servo as model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_alpha_servo_candidate import candidate_inputs, compare_samples
from gen0_alpha_servo_model import CASES


def _parameters(**overrides):
    values = dict(lift_area_per_rad=0.6, horizontal_fin_limit_rad=0.5,
                  vertical_fin_limit_rad=0.5, stability_distance_m=0.175,
                  fin_force_area_per_rad=0.4, fin_lateral_acceleration_limit_g=50.0, length_m=3.71,
                  navigation_gain=4.0, command_limit_g=50.0,
                  fuse_radius_m=1.0, fuse_arming_time_s=0.0,
                  capture_time_constant_s=0.3, acceleration_pid_kp=0.0071,
                  acceleration_pid_ki=0.005, acceleration_pid_kd=0.00015,
                  acceleration_pid_integral_limit=1.0,
                  initial_alpha_deg=0.0, guidance_delay_s=10.0)
    values.update(overrides)
    return model.AlphaServoParameters(**values)


def test_target_course_is_independent_of_initial_bearing():
    for azimuth in (0., 30., 75.):
        position, velocity = model.planar_target_motion(dict(
            target_speed_kmh=1200., initial_distance_m=8000.,
            target_azimuth_deg=azimuth, target_heading_deg=180.))
        assert math.isclose(math.hypot(*position), 8000.)
        assert math.isclose(velocity[0], -1200./3.6)
        assert abs(velocity[1]) < 1e-10
    legacy = dict(target_speed_kmh=1200., target_azimuth_deg=30.)
    _, velocity = model.planar_target_motion(legacy)
    assert velocity == (-1200./3.6*math.cos(math.pi/6), -1200./3.6*math.sin(math.pi/6))


def test_confirmed_r77_target_follows_parallel_opposite_course():
    cfg, params = candidate_inputs("su_r_77_1")
    scenario = CASES[0][3]
    assert scenario['target_heading_deg'] == 180
    assert scenario['target_course_reference'] == 'absolute_world'
    result = model.run_alpha_servo(cfg, params, scenario, max_time_s=.1)
    for row in result.samples:
        assert math.isclose(row['target_x_m'], 8000*math.cos(math.pi/6)-1200/3.6*row['time_s'], abs_tol=1e-9)
        assert math.isclose(row['target_z_m'], 4000., abs_tol=1e-9)


def test_planar_target_course_conventions_match_three_dimensional_target():
    from aim120_model.target import TargetModel
    for reference, course in (('absolute_world', 180.), ('sensorwhale_launch_axis', 0.),
                              ('statshark_relative_to_los', 0.)):
        scenario = dict(target_speed_kmh=1200., initial_distance_m=8000.,
                        target_azimuth_deg=30., target_heading_deg=course,
                        target_course_reference=reference)
        position, velocity = model.planar_target_motion(scenario)
        target = TargetModel(dict(initial_target_distance_m=8000., target_altitude_m=6300.,
            target_azimuth_deg=30., target_speed_kmh=1200., target_course_deg=course,
            target_course_reference=reference), 9.80665).initial_state
        assert position == (target.position[0], target.position[2])
        assert velocity == (target.velocity[0], target.velocity[2])


def test_axial_integration_respects_burn_boundaries(monkeypatch):
    monkeypatch.setattr(model, "effective_cda0", lambda *args: 0.0)
    monkeypatch.setattr(model, "effective_cda_alpha", lambda *args: 0.0)
    cfg = dict(geometry={"initial_mass_kg": 100.0},
               propulsion={"stages": [
                   dict(name="boost", duration_s=0.5, thrust_n=1000.0, mass_lost_kg=0.0, isp_s=1.0),
                   dict(name="sustain", duration_s=0.5, thrust_n=500.0, mass_lost_kg=0.0, isp_s=1.0)]},
               performance={"lifetime_s": 2.0}, atmosphere={"gravity_mps2": 0.0})
    p = _parameters()
    scenario = dict(launch_speed_kmh=1080.0, launch_altitude_m=6300.0,
                    target_speed_kmh=0.0)
    result = model.run_alpha_servo(cfg, p, scenario, max_time_s=1.5, plant_dt_s=0.007)
    assert result.event == "lifetime"
    assert math.isclose(result.samples[-1]["speed_kmh"], 307.5 * 3.6, abs_tol=1e-7)
    # Integral of 300 + (10, then 5, then 0) m/s² over three half seconds.
    assert math.isclose(result.samples[-1]["x_m"], 458.125, abs_tol=1e-7)


def test_swept_fuse_respects_arming_boundary(monkeypatch):
    monkeypatch.setattr(model, "effective_cda0", lambda *args: 0.0)
    monkeypatch.setattr(model, "effective_cda_alpha", lambda *args: 0.0)
    cfg = dict(geometry={"initial_mass_kg": 100.0},
               propulsion={"stages": [dict(name="coast", duration_s=1.0, thrust_n=0.0,
                                            mass_lost_kg=0.0, isp_s=1.0)]},
               performance={"lifetime_s": 1.0}, atmosphere={"gravity_mps2": 0.0})
    p = _parameters(fuse_arming_time_s=0.01)
    scenario = dict(launch_speed_kmh=1080.0, launch_altitude_m=6300.0,
                    target_speed_kmh=1080.0, target_azimuth_deg=0.0, initial_distance_m=10.0)
    result = model.run_alpha_servo(cfg, p, scenario, max_time_s=0.03, plant_dt_s=0.02)
    assert result.event == "fuse"
    assert math.isclose(result.samples[-1]["time_s"], 0.015, abs_tol=1e-12)
    assert math.isclose(result.minimum_distance_m, 1.0, abs_tol=1e-9)


def test_free_candidate_converges_with_fixed_controller_period():
    cfg, params = candidate_inputs("su_r_77_1")
    scenario = CASES[0][3]
    coarse = model.run_alpha_servo(cfg, params, scenario, max_time_s=3.5)
    fine = model.run_alpha_servo(cfg, params, scenario, max_time_s=3.5, plant_dt_s=0.001)
    assert coarse.event == fine.event
    assert abs(coarse.samples[-1]["time_s"] - fine.samples[-1]["time_s"]) < 0.002
    for a, b in zip(coarse.samples[:-1], fine.samples[:-1]):
        assert abs(a["time_s"] - b["time_s"]) < 1e-8
        assert abs(a["speed_kmh"] - b["speed_kmh"]) < 0.2
        assert abs(a["alpha_deg"] - b["alpha_deg"]) < 0.1
        assert all(math.isfinite(value) for value in a.values())
    # The actual launch controller operates before the 1.7 s fuse arming time.
    assert any(abs(row["horizontal_fin_deflection_rad"]) > 0.01
               for row in coarse.samples if 0.5 <= row["time_s"] < 1.7)
    assert all(abs(row["horizontal_fin_deflection_rad"]) <= params.horizontal_fin_limit_rad
               for row in coarse.samples)


def test_control_angle_limit_preserves_raw_radians():
    _, params = candidate_inputs("su_r_77_1")
    assert params.horizontal_fin_limit_rad == 0.460812
    assert params.vertical_fin_limit_rad == 0.460812


def test_tail_moment_requires_pressure_and_lever_arm():
    p = _parameters()
    def sample(parameters=p, pressure=40000.0, mass=190.0):
        return model.fin_moment_sample(400.0, 0.0, 0.0, pressure, mass, 0.1, parameters)
    base = sample()
    assert base.angular_acceleration_rad_s2 > 0.0
    assert sample(pressure=0.0).angular_acceleration_rad_s2 == 0.0
    assert sample(replace(p, stability_distance_m=0.0)).angular_acceleration_rad_s2 == 0.0
    assert math.isclose(sample(replace(p, stability_distance_m=0.35)).moment_nm, 2.0 * base.moment_nm)
    assert math.isclose(sample(mass=380.0).angular_acceleration_rad_s2, base.angular_acceleration_rad_s2 / 2.0)


def test_force_limit_leaves_low_pressure_slope_and_passive_moments_unchanged():
    low_cap = _parameters(fin_lateral_acceleration_limit_g=10.0)
    high_cap = replace(low_cap, fin_lateral_acceleration_limit_g=80.0)
    for pressure in (100.0, 1000.0):
        low = model.fin_moment_sample(400.0, 0.0, 0.0, pressure, 190.0, 0.2, low_cap)
        high = model.fin_moment_sample(400.0, 0.0, 0.0, pressure, 190.0, 0.2, high_cap)
        assert low.control_force_n == high.control_force_n == low.control_force_raw_n
    low = model.fin_moment_sample(1200.0, 0.1, 0.2, 1e7, 190.0, 0.2, low_cap)
    high = model.fin_moment_sample(1200.0, 0.1, 0.2, 1e7, 190.0, 0.2, high_cap)
    assert low.passive_moment_nm == high.passive_moment_nm
    assert abs(low.control_force_n) < abs(high.control_force_n)


def test_saturated_force_uses_current_mass_and_acceleration_cap():
    accelerations = []
    for limit in (10.0, 20.0, 40.0, 80.0):
        p = _parameters(fin_lateral_acceleration_limit_g=limit)
        per_mass = []
        for mass in (190.0, 119.0):
            for sign in (-1.0, 1.0):
                sample = model.fin_moment_sample(1200.0, 0.0, 0.0, 1e7, mass, sign * 0.2, p)
                assert math.isclose(abs(sample.control_force_n) / mass, 9.80665 * limit)
                assert math.isclose(sample.control_moment_nm, -sample.control_force_n * p.stability_distance_m)
                assert sample.control_moment_nm * sign > 0.0
                per_mass.append(abs(sample.angular_acceleration_rad_s2))
        assert all(math.isclose(value, per_mass[0]) for value in per_mass)
        accelerations.append(per_mass[0])
    assert all(math.isclose(b, 2.0 * a) for a, b in zip(accelerations, accelerations[1:]))


def test_zero_control_force_cap_keeps_passive_stability():
    p = _parameters(fin_lateral_acceleration_limit_g=0.0)
    sample = model.fin_moment_sample(400.0, 0.1, 0.0, 40000.0, 190.0, 0.2, p)
    assert sample.control_force_n == 0.0
    assert sample.moment_nm < 0.0


def test_candidate_accepts_independent_fin_force_slope():
    _, baseline = candidate_inputs("su_r_77_1")
    _, changed = candidate_inputs("su_r_77_1", fin_force_area_per_rad=0.8)
    assert changed.fin_force_area_per_rad == 0.8
    assert baseline.fin_lateral_acceleration_limit_g == changed.fin_lateral_acceleration_limit_g == 53.5864
    assert baseline.horizontal_fin_limit_rad == changed.horizontal_fin_limit_rad


def test_independent_restoring_slope_separates_control_and_static_trim():
    p = _parameters(fin_force_area_per_rad=0.8, restoring_force_area_per_rad=0.4)
    def sample(parameters, alpha=0.1, delta=0.05, omega=0.0):
        return model.fin_moment_sample(400.0, alpha, omega, 40000.0, 190.0, delta, parameters)
    trimmed = sample(p)
    assert abs(trimmed.moment_nm) < 1e-10
    stronger_control = sample(replace(p, fin_force_area_per_rad=1.6))
    stronger_restoring = sample(replace(p, restoring_force_area_per_rad=0.8))
    assert stronger_control.passive_moment_nm == trimmed.passive_moment_nm
    assert stronger_control.moment_nm > 0
    assert stronger_restoring.control_moment_nm == trimmed.control_moment_nm
    assert stronger_restoring.moment_nm < 0
    # Zero-control rotation must dissipate energy for either restoring slope.
    for parameters in (p, replace(p, restoring_force_area_per_rad=0.8)):
        assert sample(parameters, alpha=0., delta=0., omega=.2).moment_nm * .2 < 0


def test_vertical_trim_uses_independent_restoring_ratio_and_force_cap():
    cfg, base = candidate_inputs("su_r_77_1")
    p = replace(base, restoring_force_area_per_rad=base.fin_force_area_per_rad / 2)
    result = model.run_alpha_servo(cfg, p, CASES[0][3], max_time_s=0.02)
    for row in result.samples:
        assert math.isclose(row["vertical_fin_deflection_rad"], row["vertical_alpha_rad"] / 2)
    low_travel = replace(p, restoring_force_area_per_rad=base.fin_force_area_per_rad * 2,
                         vertical_fin_limit_rad=0.05)
    assert model.run_alpha_servo(cfg, low_travel, CASES[0][3], max_time_s=.02).event == "vertical_trim_unavailable"
    assert model.run_alpha_servo(cfg, replace(p, fin_lateral_acceleration_limit_g=0),
                                 CASES[0][3], max_time_s=.02).event == "vertical_trim_unavailable"


def test_restoring_slope_rejects_nonpositive_and_nonfinite_values():
    import pytest
    cfg, p = candidate_inputs("su_r_77_1")
    for value in (0., -1., math.nan, math.inf):
        with pytest.raises(ValueError, match="restoring"):
            candidate_inputs("su_r_77_1", restoring_force_area_per_rad=value)
        with pytest.raises(ValueError, match="positive"):
            model.run_alpha_servo(cfg, replace(p, restoring_force_area_per_rad=value),
                                  CASES[0][3], max_time_s=.02)


def test_uncommanded_tail_restores_alpha_and_damps_body_rotation():
    p = _parameters()
    for sign in (-1.0, 1.0):
        displaced = model.fin_moment_sample(400.0, sign * 0.1, 0.0, 40000.0, 190.0, 0.0, p)
        rotating = model.fin_moment_sample(400.0, 0.0, sign * 0.2, 40000.0, 190.0, 0.0, p)
        assert displaced.moment_nm * sign < 0.0
        assert rotating.moment_nm * (sign * 0.2) < 0.0


def test_independent_rate_damping_is_dissipative_and_preserves_static_forces():
    base = _parameters()
    damped = replace(base, rate_damping_derivative_m4=8.0)
    for omega in (-0.3, 0.0, 0.3):
        old = model.fin_moment_sample(400.0, 0.1, omega, 40000.0, 190.0, 0.2, base)
        new = model.fin_moment_sample(400.0, 0.1, omega, 40000.0, 190.0, 0.2, damped)
        assert new.rate_damping_moment_nm * omega <= 0.0
        assert new.equivalent_normal_force_n == old.equivalent_normal_force_n
        assert new.control_moment_nm == old.control_moment_nm
        assert new.passive_moment_nm == old.passive_moment_nm
        assert math.isclose(new.moment_nm, old.moment_nm + new.rate_damping_moment_nm)
        assert math.isclose(new.angular_acceleration_rad_s2,
                            old.angular_acceleration_rad_s2 - new.rate_damping_per_s * omega)
        if omega == 0.0:
            assert new.moment_nm == old.moment_nm
    vacuum = model.fin_moment_sample(400.0, 0.1, 0.3, 0.0, 190.0, 0.2, damped)
    assert vacuum.rate_damping_moment_nm == 0.0


def test_independent_rate_damping_rejects_negative_and_nonfinite_values():
    import pytest
    cfg, p = candidate_inputs("su_r_77_1")
    for value in (-1.0, math.nan, math.inf):
        with pytest.raises(ValueError, match="rate"):
            candidate_inputs("su_r_77_1", rate_damping_derivative_m4=value)
        with pytest.raises(ValueError, match="rate"):
            model.run_alpha_servo(cfg, replace(p, rate_damping_derivative_m4=value),
                                  CASES[0][3], max_time_s=0.02)


def test_fin_command_changes_rotation_before_aoa():
    cfg, params = candidate_inputs("su_r_77_1")
    params = replace(params, initial_alpha_deg=0.0)
    result = model.run_alpha_servo(cfg, params, CASES[0][3], max_time_s=0.54)
    enabled = next(row for row in result.samples if math.isclose(row["time_s"], 0.5))
    following = result.samples[-1]
    assert enabled["horizontal_fin_deflection_rad"] > 0.0
    assert enabled["body_angular_acceleration_rad_s2"] > 0.0
    assert abs(enabled["horizontal_alpha_rad"]) < 1e-12
    assert following["body_rate_rad_s"] > 0.0
    assert following["horizontal_alpha_rad"] > 0.0


def test_body_aoa_is_geometry_and_is_not_clipped_to_fin_travel():
    cfg, params = candidate_inputs("su_r_77_1")
    params = replace(params, horizontal_fin_limit_rad=math.radians(5.0),
                     initial_alpha_deg=30.0, guidance_delay_s=10.0)
    result = model.run_alpha_servo(cfg, params, CASES[0][3], max_time_s=0.1)
    assert all(abs(row["horizontal_alpha_rad"]) > params.horizontal_fin_limit_rad
               for row in result.samples)
    for row in result.samples:
        expected = (row["body_heading_rad"] - row["heading_rad"] + math.pi) % (2 * math.pi) - math.pi
        assert math.isclose(row["horizontal_alpha_rad"], expected, abs_tol=1e-12)


def test_comparison_excludes_both_ends_without_extrapolation():
    import numpy as np
    samples = [dict(time_s=t, speed_kmh=1000.0, alpha_deg=2.0,
                    wind_normal_specific_force_g=3.0) for t in (1.0, 2.0)]
    observations = np.array([[0.5, 1000, 2, 3], [1.5, 1000, 2, 3], [2.5, 1000, 2, 3]])
    rows, rmse, count = compare_samples(samples, observations)
    assert count == 1
    assert np.all(rmse == 0.0)
    assert math.isnan(rows[0]["predicted_speed_kmh"])
    assert math.isnan(rows[-1]["predicted_speed_kmh"])
