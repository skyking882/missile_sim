from aim120_model.dynamics import SimState
from aim120_model.guidance import guidance_command
from aim120_model.observation import IdealTruthTrackProvider, KinematicTrackProvider, SensorTrackProvider
from aim120_model.target import TargetState
from aim120_model.tracking import TrackMode


def _missile() -> SimState:
    return SimState((0.0, 1000.0, 0.0), (300.0, 0.0, 0.0), 0.0, 0.0, 0.0, 0.0, 100.0)


def _model(active_radar=False, reconnect=False):
    return {
        "active_radar": active_radar,
        "inertial_navigation": True,
        "use_target_velocity": True,
        "datalink": True,
        "reconnect_datalink": reconnect,
        "inertial_drift_speed_mps": 2.0,
        "break_lock_max_time_s": 100.0,
        "radar_seeker": {
            "active": active_radar,
            "lock_angle_max_deg": 55.0,
            "angle_max_deg": 55.0,
            "rate_max_deg_s": 60.0,
            "prolongation_time_max_s": 1.0,
            "doppler_speed_gate": {"filter_alpha": 0.8, "filter_beta": 0.05, "search_range_mps": 300.0},
            "dist_gate": {"filter_alpha": 0.8, "filter_beta": 0.05, "search_range_m": 5000.0},
            "receiver": {"range_max_m": 25000.0},
            "doppler_speed": {"min_mps": -3000.0, "max_mps": 3000.0, "width_mps": 20.0, "ref_width_mps": 80.0, "signal_width_min_mps": 5.0},
        },
    }


def test_ideal_truth_provider_returns_a_copied_ideal_solution():
    target = TargetState((10000.0, 1000.0, 0.0), (-300.0, 0.0, 0.0))
    solution = IdealTruthTrackProvider().update(0.0, 0.1, _missile(), target)
    assert solution.mode is TrackMode.IDEAL_TRUTH
    assert solution.valid is True
    assert solution.source == "target_truth"


def test_datalink_updates_next_tick_then_ins_uses_last_velocity_after_disconnect():
    provider = SensorTrackProvider(
        _model(),
        lock_range_m=20000.0,
        datalink_disconnect_time_s=0.5,
        inertial_drift_direction=(0.0, 0.0, 1.0),
    )
    first_target = TargetState((10000.0, 1000.0, 0.0), (-300.0, 0.0, 0.0))
    first = provider.update(0.0, 1.0, _missile(), first_target)
    assert first.mode is TrackMode.DATALINK
    second_target = TargetState((9900.0, 1000.0, 0.0), (0.0, 200.0, 0.0))
    second = provider.update(0.25, 0.25, _missile(), second_target)
    assert second.mode is TrackMode.DATALINK
    assert second.velocity == (0.0, 200.0, 0.0)

    changed_truth = TargetState((9800.0, 1000.0, 0.0), (500.0, 0.0, 0.0))
    third = provider.update(1.25, 1.0, _missile(), changed_truth)
    assert third.mode is TrackMode.INERTIAL
    assert third.velocity == (0.0, 200.0, 0.0)
    assert third.position[2] > 1.9
    assert third.position[0] < first.position[0]


def test_pitbull_closes_datalink_when_reconnect_is_false():
    provider = SensorTrackProvider(_model(active_radar=True), lock_range_m=20000.0)
    tracked = provider.update(0.0, 0.1, _missile(), TargetState((10000.0, 1000.0, 0.0), (-300.0, 0.0, 0.0)))
    assert tracked.mode is TrackMode.RADAR_TRACK
    assert provider.radar_has_tracked_once is True
    assert provider.datalink_connected is False
    coasted = provider.update(0.5, 0.5, _missile(), TargetState((10000.0, 900.0, 0.0), (0.0, 0.0, 0.0)))
    assert coasted.mode is TrackMode.INS_SEARCH


def test_radar_lock_loss_uses_ins_search_then_reacquires_track():
    provider = SensorTrackProvider(
        _model(active_radar=True),
        lock_range_m=20000.0,
        inertial_drift_direction=(0.0, 0.0, 1.0),
    )
    tracked = provider.update(0.0, 0.1, _missile(), TargetState((10000.0, 1000.0, 0.0), (-300.0, 0.0, 0.0)))
    assert tracked.mode is TrackMode.RADAR_TRACK
    assert provider.seeker_display_state == "TRK"

    short_loss = provider.update(0.5, 0.5, _missile(), TargetState((10000.0, 900.0, 0.0), (0.0, 0.0, 0.0)))
    assert short_loss.mode is TrackMode.INS_SEARCH
    assert short_loss.valid is True
    assert provider.seeker_display_state == "INS+SRC"

    long_loss = provider.update(1.1, 0.6, _missile(), TargetState((10000.0, 900.0, 0.0), (0.0, 0.0, 0.0)))
    assert long_loss.mode is TrackMode.INS_SEARCH
    assert long_loss.valid is True
    assert provider.radar.distance_gate.initialized is False
    assert provider.seeker_display_state == "INS+SRC"

    reacquired = provider.update(1.2, 0.1, _missile(), TargetState((10000.0, 1000.0, 0.0), (-300.0, 0.0, 0.0)))
    assert reacquired.mode is TrackMode.RADAR_TRACK
    assert provider.seeker_display_state == "TRK"
    assert provider.first_reacquire_time_s == 1.2


def test_generic_profile_track_also_reacquires_after_deterministic_lock_loss():
    provider = KinematicTrackProvider(
        lock_range_m=20000.0,
        maximum_angular_rate_deg_s=60.0,
        seeker_type="optical",
    )
    tracked = provider.update(0.0, 0.1, _missile(), TargetState((10000.0, 1000.0, 0.0), (-300.0, 0.0, 0.0)))
    assert tracked.mode is TrackMode.PROFILE_KINEMATIC
    assert provider.seeker_display_state == "TRK"

    lost = provider.update(0.5, 0.5, _missile(), TargetState((30000.0, 1000.0, 0.0), (-300.0, 0.0, 0.0)))
    assert lost.mode is TrackMode.INS_SEARCH
    assert lost.valid is True
    assert provider.seeker_display_state == "INS+SRC"

    reacquired = provider.update(1.0, 0.5, _missile(), TargetState((10000.0, 1000.0, 0.0), (-300.0, 0.0, 0.0)))
    assert reacquired.mode is TrackMode.PROFILE_KINEMATIC
    assert provider.seeker_display_state == "TRK"
    assert provider.first_reacquire_time_s == 1.0


def test_reconnect_true_allows_datalink_after_radar_track():
    provider = SensorTrackProvider(_model(active_radar=True, reconnect=True), lock_range_m=20000.0)
    provider.update(0.0, 0.1, _missile(), TargetState((10000.0, 1000.0, 0.0), (-300.0, 0.0, 0.0)))
    assert provider.datalink_connected is True
    solution = provider.update(0.5, 0.5, _missile(), TargetState((10000.0, 900.0, 0.0), (0.0, 200.0, 0.0)))
    assert solution.mode is TrackMode.DATALINK


def test_guidance_uses_last_track_after_observation_disconnect():
    provider = SensorTrackProvider(_model(), lock_range_m=20000.0, datalink_disconnect_time_s=0.1)
    initial = TargetState((10000.0, 1000.0, 1000.0), (-300.0, 0.0, 0.0))
    first = provider.update(0.0, 0.1, _missile(), initial)
    changed = TargetState((10000.0, 1000.0, 1000.0), (0.0, 0.0, -300.0))
    after = provider.update(1.0, 0.9, _missile(), changed)
    assert after.mode is TrackMode.INERTIAL
    assert after.velocity == initial.velocity


def test_public_api_sensor_track_uses_256_hz_and_exposes_track_telemetry():
    from pathlib import Path
    import sys

    root = Path(__file__).resolve().parents[1]
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))
    from aim120_model.public_api import simulate
    from missile_gui.library import scan_library

    profiles, errors = scan_library(root / "missiles", root)
    assert errors == []
    profile = next(item for item in profiles if item.get("missile_id") == "us_aim_120a")
    scenario = {
        "launch_speed_kmh": 1200.0,
        "launch_altitude_m": 6500.0,
        "launch_pitch_deg": 0.0,
        "launch_heading_deg": 0.0,
        "target_speed_kmh": 1200.0,
        "target_altitude_m": 6500.0,
        "initial_distance_m": 12000.0,
        "target_azimuth_deg": 0.0,
        "target_heading_deg": 0.0,
        "target_vertical_heading_deg": 0.0,
        "target_constant_turn_g": 0.0,
        "max_simulation_time_s": 0.1,
        "observation_mode": "sensor_track",
    }
    result = simulate(profile, scenario)
    assert result["model"]["observation_mode"] == "sensor_track"
    assert result["model"]["guidance_update_hz"] == 256.0
    assert result["model"]["observation_provider"] == "radar_datalink_ins_v1"
    assert result["model"]["lock_state_machine"] == "TRK->INS+SRC->TRK"
    assert result["model"]["random_measurement_noise"] is False
    assert result["samples"][0]["track_mode"] == "radar_track"
    assert "radar_reject_reason" in result["samples"][0]


def test_public_api_sensor_track_is_available_for_profile_without_sensor_datamine():
    from pathlib import Path
    import sys

    root = Path(__file__).resolve().parents[1]
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))
    from aim120_model.public_api import simulate
    from missile_gui.library import scan_library

    profiles, errors = scan_library(root / "missiles", root)
    assert errors == []
    profile = next(item for item in profiles if item.get("missile_id") == "su_r_27t")
    scenario = {
        "launch_speed_kmh": 1200.0,
        "launch_altitude_m": 6500.0,
        "launch_pitch_deg": 0.0,
        "launch_heading_deg": 0.0,
        "target_speed_kmh": 1200.0,
        "target_altitude_m": 6500.0,
        "initial_distance_m": 12000.0,
        "target_azimuth_deg": 0.0,
        "target_heading_deg": 0.0,
        "target_vertical_heading_deg": 0.0,
        "target_constant_turn_g": 0.0,
        "max_simulation_time_s": 0.1,
        "observation_mode": "sensor_track",
    }
    result = simulate(profile, scenario)
    assert result["model"]["observation_mode"] == "sensor_track"
    assert result["model"]["observation_provider"] == "profile_kinematic_v1"
    assert result["model"]["radar_model"] == "not_applicable"
    assert result["samples"][0]["track_mode"] == "profile_kinematic_track"
    assert result["samples"][0]["track_source"] == "profile_kinematic_v1"

    ideal = simulate(profile, {**scenario, "observation_mode": "ideal_truth"})
    assert ideal["model"]["observation_mode"] == "ideal_truth"
    assert ideal["model"]["observation_provider"] == "ideal_truth"
