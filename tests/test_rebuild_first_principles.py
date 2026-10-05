from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from aim120_model.dynamics import SimState
from aim120_model.h2_dynamics import forces_for_state_h2
from aim120_model.h2_simulator import H2Simulator
from aim120_model.profile_adapter import build_h2_candidate_config, load_runtime_defaults
from aim120_model.propulsion import PiecewisePropulsion


ROOT = Path(__file__).resolve().parents[1]


def _profile(missile_id: str = "su_r_77_1") -> dict:
    return json.loads((ROOT / "missiles" / f"{missile_id}.json").read_text())


def _config(missile_id: str = "su_r_77_1") -> dict:
    defaults = load_runtime_defaults(
        str(ROOT / "config" / "profile_h2_rebuild_defaults.json")
    )
    config, _ = build_h2_candidate_config(_profile(missile_id), defaults)
    return config


def _state(config: dict, *, pitch: float = 0.0, fin: float = 0.0, mass: float | None = None) -> SimState:
    return SimState(
        position=(0.0, 1000.0, 0.0),
        velocity=(500.0, 0.0, 0.0),
        pitch=pitch,
        yaw=0.0,
        pitch_rate=0.0,
        yaw_rate=0.0,
        mass=float(config["geometry"]["initial_mass_kg"] if mass is None else mass),
        actual_pitch_fin_angle_rad=fin,
    )


def _diag(config: dict, state: SimState):
    return forces_for_state_h2(
        state,
        0.0,
        config,
        PiecewisePropulsion.from_config(config),
        powered=False,
    )


def test_rebuild_selects_shared_force_moment_law_and_removes_legacy_capture() -> None:
    config = _config()
    assert config["control"]["plant_semantics"] == "rebuild_first_principles"
    assert config["control"]["pid_output_semantics"] == "fin_angle_rad"
    assert config["control"]["feedback_measurement"] == "cg_wind_normal_specific_force_g"
    assert config["control"]["derivative_filter_time_constant_s"] == 0.0
    assert config["guidance"]["midcourse"]["enabled"] is False
    assert config["guidance"]["angle_to_acceleration_multiplier"] == 17.5
    assert config["guidance"]["loft_omega_max_deg_s"] == 0.5
    assert config["guidance"]["target_elevation_deg"] == -3.5
    assert "packed_lift_force_eta_law" not in config["aerodynamics"]
    assert config["guidance"]["lofting_elevation_deg"] == 24.0
    candidate = config["aerodynamics"]["generalized_aero_moment_candidate"]
    assert candidate["cm_q"] == 0.0
    assert candidate["normalized_shared_gains"]["delta_force_gain"] != 0.0
    assert candidate["normalized_shared_gains"]["delta_moment_gain"] != 0.0

    pl12 = _config("cn_pl12")
    assert pl12["guidance"]["lofting_elevation_deg"] == 20.0
    assert pl12["guidance"]["target_elevation_deg"] == -2.25
    assert pl12["guidance"]["angle_to_acceleration_multiplier"] == 20.0
    assert pl12["guidance"]["loft_omega_max_deg_s"] == 0.5
    assert config["guidance"]["target_elevation_deg"] == -3.5
    assert config["guidance"]["angle_to_acceleration_multiplier"] == 17.5
    assert config["guidance"]["loft_omega_max_deg_s"] == 0.5


def test_rebuild_fin_deflection_has_direct_force_and_moment_at_zero_aoa() -> None:
    config = _config()
    zero = _diag(config, _state(config))
    positive = _diag(config, _state(config, fin=0.1))
    negative = _diag(config, _state(config, fin=-0.1))
    assert math.isclose(zero.aero.pitch_alpha_rad, 0.0, abs_tol=1.0e-12)
    assert positive.pitch_tail_force_n != 0.0
    assert positive.pitch_tail_moment_nm != 0.0
    assert math.isclose(
        positive.pitch_tail_force_n,
        -negative.pitch_tail_force_n,
        rel_tol=1.0e-12,
    )
    assert math.isclose(
        positive.pitch_tail_moment_nm,
        -negative.pitch_tail_moment_nm,
        rel_tol=1.0e-12,
    )


def test_rebuild_has_non_degenerate_trim_and_independent_moment_gain() -> None:
    config = _config()
    h = 1.0e-5
    alpha_plus = _diag(config, _state(config, pitch=h))
    alpha_minus = _diag(config, _state(config, pitch=-h))
    delta_plus = _diag(config, _state(config, fin=h))
    delta_minus = _diag(config, _state(config, fin=-h))
    alpha_force = (alpha_plus.pitch_body_normal_force_n - alpha_minus.pitch_body_normal_force_n) / (2.0 * h)
    delta_force = (delta_plus.pitch_tail_force_n - delta_minus.pitch_tail_force_n) / (2.0 * h)
    alpha_moment = (alpha_plus.pitch_body_static_moment_nm - alpha_minus.pitch_body_static_moment_nm) / (2.0 * h)
    delta_moment = (delta_plus.pitch_tail_moment_nm - delta_minus.pitch_tail_moment_nm) / (2.0 * h)
    assert alpha_force != 0.0 and delta_force != 0.0
    assert alpha_moment != 0.0 and delta_moment != 0.0

    # Trim the static moment with the fin while holding a small nonzero AoA.
    # A rank-one M=-arm*F map would force the corresponding total force to
    # zero; the rebuild hypothesis must leave a nonzero maneuver load.
    trim_alpha = 0.03
    trim_delta = -alpha_moment / delta_moment * trim_alpha
    trimmed = _diag(config, _state(config, pitch=trim_alpha, fin=trim_delta))
    trim_force = trimmed.pitch_body_normal_force_n + trimmed.pitch_tail_force_n
    trim_moment = trimmed.pitch_body_static_moment_nm + trimmed.pitch_tail_moment_nm
    assert abs(trim_force) > 1.0
    assert math.isclose(trim_moment, 0.0, abs_tol=1.0e-8)

    # Change one moment gain only.  Force derivatives and the other moment
    # derivative must remain independent of that change.
    defaults = copy.deepcopy(
        load_runtime_defaults(str(ROOT / "config" / "profile_h2_rebuild_defaults.json"))
    )
    defaults["rebuild_force_moment_candidate"]["alpha_moment_gain"] = -2.0
    changed_config, _ = build_h2_candidate_config(_profile(), defaults)
    changed_alpha_plus = _diag(changed_config, _state(changed_config, pitch=h))
    changed_alpha_minus = _diag(changed_config, _state(changed_config, pitch=-h))
    changed_delta_plus = _diag(changed_config, _state(changed_config, fin=h))
    changed_delta_minus = _diag(changed_config, _state(changed_config, fin=-h))
    changed_alpha_force = (
        changed_alpha_plus.pitch_body_normal_force_n
        - changed_alpha_minus.pitch_body_normal_force_n
    ) / (2.0 * h)
    changed_delta_force = (
        changed_delta_plus.pitch_tail_force_n - changed_delta_minus.pitch_tail_force_n
    ) / (2.0 * h)
    changed_alpha_moment = (
        changed_alpha_plus.pitch_body_static_moment_nm
        - changed_alpha_minus.pitch_body_static_moment_nm
    ) / (2.0 * h)
    changed_delta_moment = (
        changed_delta_plus.pitch_tail_moment_nm - changed_delta_minus.pitch_tail_moment_nm
    ) / (2.0 * h)
    assert math.isclose(changed_alpha_force, alpha_force, rel_tol=1.0e-12)
    assert math.isclose(changed_delta_force, delta_force, rel_tol=1.0e-12)
    assert math.isclose(changed_delta_moment, delta_moment, rel_tol=1.0e-12)
    assert not math.isclose(changed_alpha_moment, alpha_moment, rel_tol=1.0e-3)


def test_rebuild_force_uses_initial_mass_reference_before_load_cap() -> None:
    config = _config()
    light = _diag(config, _state(config, pitch=0.03, fin=0.05, mass=119.0))
    launch = _diag(config, _state(config, pitch=0.03, fin=0.05, mass=190.0))
    assert math.isclose(
        light.pitch_body_normal_force_n + light.pitch_tail_force_n,
        launch.pitch_body_normal_force_n + launch.pitch_tail_force_n,
        rel_tol=1.0e-12,
    )


def test_rebuild_common_smoke_runs_r77_and_pl12_without_capture_path() -> None:
    case = {
        "name": "rebuild_common_smoke",
        "model_variant": "full",
        "observation_mode": "ideal_truth",
        "loft_enabled": False,
        "initial_conditions": {
            "start_speed_kmh": 1200.0,
            "launch_altitude_m": 6200.0,
            "launch_angle_deg": 0.0,
            "launch_yaw_deg": 0.0,
            "target_speed_kmh": 1550.0,
            "target_altitude_m": 6200.0,
            "initial_target_distance_m": 8000.0,
            "target_azimuth_deg": 30.0,
            "target_course_deg": 0.0,
            "target_course_reference": "relative_to_los",
            "target_constant_g_turn": 0.0,
            "target_vertical_course_deg": 0.0,
        },
    }
    for missile_id in ("su_r_77_1", "cn_pl12"):
        config = _config(missile_id)
        config["performance"]["lifetime_s"] = 0.5
        result = H2Simulator(config).run(case)
        assert result["event_type"] != "numerical_failure"
        assert result["samples"][-1]["pcc_capture_mode"] == "off"
        early = [
            sample
            for sample in result["samples"]
            if any(abs(value) > 1.0e-9 for value in sample["controller_specific_force_command_g"])
        ]
        assert early, "the common smoke case must produce a finite early PN command"
        assert all(abs(sample["midcourse_weight"]) <= 1.0e-12 for sample in early)
        assert any(
            abs(sample["pitch_pid_output"]) + abs(sample["yaw_pid_output"]) > 1.0e-12
            for sample in early
        )
        assert any(
            abs(sample["pitch_requested_fin_angle_rad"])
            + abs(sample["yaw_requested_fin_angle_rad"])
            > 1.0e-12
            for sample in early
        )
        assert any(
            abs(sample["actual_pitch_fin_angle_rad"])
            + abs(sample["actual_yaw_fin_angle_rad"])
            > 1.0e-12
            for sample in early
        )


def test_mixed_flow_frame_is_orthonormal_including_body_up_flow() -> None:
    from aim120_model.aerodynamics import body_axes, wind_normal_basis
    from aim120_model.math3d import dot, norm

    axes = body_axes(0.3, -0.4)
    for velocity in ((430.0, -180.0, 210.0), axes.up, tuple(-x for x in axes.up)):
        basis = wind_normal_basis(axes, velocity)
        assert math.isclose(norm(basis.up), 1.0, abs_tol=1e-12)
        assert math.isclose(norm(basis.right), 1.0, abs_tol=1e-12)
        assert abs(dot(basis.up, basis.right)) < 1e-12
        assert abs(dot(basis.up, basis.forward)) < 1e-12
        assert abs(dot(basis.right, basis.forward)) < 1e-12


def test_exact_cg_aoa_and_drag_use_wind_and_quaternion_not_local_incidence() -> None:
    from aim120_model.aerodynamics import (
        body_axes_from_quaternion, compute_aerodynamics_h2, quaternion_from_pitch_yaw,
    )
    from aim120_model.math3d import dot, norm, sub

    config = _config()
    config["atmosphere"]["wind_mps"] = [35.0, -15.0, 20.0]
    axes = body_axes_from_quaternion(quaternion_from_pitch_yaw(0.4, -0.35))
    velocity = (420.0, -80.0, 130.0)
    relative = sub(velocity, tuple(config["atmosphere"]["wind_mps"]))
    expected = math.acos(dot(relative, axes.forward) / norm(relative))
    sample = compute_aerodynamics_h2(3000.0, velocity, 0.0, 0.0, config,
        axes_override=axes, normal_force_velocity_mps=(420.0, 90.0, -20.0))
    assert math.isclose(sample.angle_of_attack_rad, expected, abs_tol=1e-12)
    expected_drag_area = math.pi * 0.2**2 / 4 * 9 * min(expected, 1.2)**2
    assert math.isclose(sample.cda_alpha_m2, expected_drag_area, abs_tol=1e-12)
    cg_sample = compute_aerodynamics_h2(3000.0, velocity, 0.0, 0.0, config, axes_override=axes)
    assert not math.isclose(sample.pitch_alpha_rad, cg_sample.pitch_alpha_rad)
    assert math.isclose(sample.angle_of_attack_rad, cg_sample.angle_of_attack_rad, abs_tol=1e-12)


def test_rebuild_trajectory_g_matches_cg_normal_projection_and_feedback() -> None:
    from dataclasses import replace
    from aim120_model.aerodynamics import quaternion_from_pitch_yaw
    from aim120_model.math3d import dot, norm, normalize, scale, sub

    config = _config()
    config["atmosphere"]["wind_mps"] = [20.0, -30.0, 15.0]
    state = replace(_state(config), velocity=(520.0, -50.0, 80.0),
        orientation_quaternion=quaternion_from_pitch_yaw(0.3, -0.25),
        actual_pitch_fin_angle_rad=0.07, actual_yaw_fin_angle_rad=-0.03)
    diagnostics = forces_for_state_h2(state, 0.0, config,
        PiecewisePropulsion.from_config(config), powered=True)
    tangent = normalize(sub(state.velocity, tuple(config["atmosphere"]["wind_mps"])))
    force = diagnostics.specific_force_mps2
    expected = norm(sub(force, scale(tangent, dot(force, tangent)))) / config["atmosphere"]["gravity_mps2"]
    assert math.isclose(diagnostics.trajectory_lateral_load_g, expected, abs_tol=1e-10)
    assert math.isclose(diagnostics.trajectory_lateral_load_g,
        math.hypot(diagnostics.wind_normal_pitch_acceleration_g,
                   diagnostics.wind_normal_yaw_acceleration_g), abs_tol=1e-10)
    assert diagnostics.propulsion.thrust_n > 0
    # Specific force excludes gravity: using kinematic acceleration would
    # yield a different normal load in this mixed-angle powered state.
    acceleration = diagnostics.acceleration_mps2
    kinematic = norm(sub(acceleration, scale(tangent, dot(acceleration, tangent)))) / config["atmosphere"]["gravity_mps2"]
    assert not math.isclose(expected, kinematic, abs_tol=1e-5)
