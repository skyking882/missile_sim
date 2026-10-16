from aim120_model.dynamics import SimState
from aim120_model.guidance import guidance_command
import math
import copy

import pytest

from aim120_model.observation import IdealTruthTrackProvider, InertialTrackPropagator, KinematicTrackProvider, SensorTrackProvider
from aim120_model.target import TargetState
from aim120_model.tracking import TrackMode, TrackSolution


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
    assert coasted.mode is TrackMode.RADAR_COAST


@pytest.mark.parametrize('filter_kind', ['none', 'radar', 'unified'])
@pytest.mark.parametrize('branch', ['single', 'decoys', 'track_gate'])
def test_provider_keeps_last_accepted_estimate_through_missed_echoes(filter_kind, branch):
    provider = SensorTrackProvider(_model(active_radar=True), lock_range_m=20000.)
    provider.enable_launcher_support(lambda t, truth: ('', (1000., 1000., 0.), (10., 0., 0.)))
    if filter_kind == 'radar':
        provider.enable_alpha_beta_observer(.8, .05)
    elif filter_kind == 'unified':
        provider.enable_unified_target_observer()
    if branch == 'track_gate':
        provider.radar.enable_track_gate_model({'notch_track_hazard': [], 'datalink_after_lock': False})
    kwargs = {} if branch == 'single' else {'decoys': []}
    provider.update(0., 0., _missile(), TargetState((30000., 1000., 0.), (0., 0., 0.)), **kwargs)
    accepted = provider.update(1., 1., _missile(), TargetState((1020., 1000., 0.), (40., 0., 0.)), **kwargs)
    assert accepted.mode is TrackMode.RADAR_TRACK
    if filter_kind != 'none':
        assert accepted.velocity[0] != 40.
    observer = provider.alpha_beta_observer
    count = None if observer is None else observer.update_count
    missing = TargetState((0., 1000., 10000.), (999., 0., 0.))
    for t in (1.25, 1.75, 2.):
        held = provider.update(t, .25, _missile(), missing, **kwargs)
        assert not provider.last_detection.valid
        assert held.mode is TrackMode.RADAR_COAST and held.valid
        assert held.sample_time_s == accepted.sample_time_s
        assert held.position == pytest.approx(tuple(p+v*(t-1.) for p,v in zip(accepted.position,accepted.velocity)))
        assert held.velocity == accepted.velocity
        assert provider.radar._last_detection_time_s == 1.
        assert provider.first_lock_loss_time_s is None
        assert provider.seeker_display_state == 'TRK'
        if t == 1.25:
            recovering = copy.deepcopy(provider)
    if observer is not None:
        assert observer.update_count == count
    fresh = TargetState(tuple(p+v*.5 for p,v in zip(accepted.position,accepted.velocity)), accepted.velocity)
    resumed = recovering.update(1.5, .25, _missile(), fresh, **kwargs)
    assert resumed.mode is TrackMode.RADAR_TRACK
    assert resumed.sample_time_s == 1.5
    assert recovering.radar._last_detection_time_s == 1.5
    assert recovering.first_lock_loss_time_s is None
    if recovering.alpha_beta_observer is not None:
        assert recovering.alpha_beta_observer.update_count == count+1
        assert recovering.alpha_beta_observer.last_dt_s == .5
    expired = provider.update(2.000001, .000001, _missile(), missing, **kwargs)
    assert expired.mode is TrackMode.INS_SEARCH
    assert provider.first_lock_loss_time_s == 2.000001


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
    assert short_loss.mode is TrackMode.RADAR_COAST
    assert short_loss.valid is True
    assert provider.seeker_display_state == "TRK"

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
    assert solution.mode is TrackMode.RADAR_COAST
    after_timeout = provider.update(1.1, .6, _missile(), TargetState((10000., 900., 0.), (0., 200., 0.)))
    assert after_timeout.mode is TrackMode.DATALINK


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


def _inertial_seed(position=(10000.0, 2500.0, 1000.0), velocity=(-300.0, -100.0, 20.0), time=0.0):
    return TrackSolution(position, velocity, time, time, TrackMode.DATALINK, True, "datalink")


def test_ins_floor_clips_y_velocity_and_preserves_horizontal_propagation():
    floor = InertialTrackPropagator(min_altitude_m=50.0)
    full = InertialTrackPropagator()
    for propagator in (floor, full):
        propagator.reset(_inertial_seed())
    for time in (20.0, 25.0, 30.0):
        bounded = floor.update(time, 5.0)
        unbounded = full.update(time, 5.0)
        assert bounded.position[0::2] == unbounded.position[0::2]
        assert bounded.velocity[0::2] == unbounded.velocity[0::2]
        assert bounded.position[1] == (500.0 if time == 20.0 else 50.0)
        assert bounded.velocity[1] == (-100.0 if time == 20.0 else 0.0)
        assert bounded.sample_time_s == 0.0
    assert unbounded.position[1] == -500.0


def test_ins_floor_includes_drift_and_reset_restores_observed_velocity():
    floor = InertialTrackPropagator(drift_speed_mps=200.0, drift_direction=(1.0, -1.0, 1.0),
                                   min_altitude_m=50.0)
    full = InertialTrackPropagator(drift_speed_mps=200.0, drift_direction=(1.0, -1.0, 1.0))
    for propagator in (floor, full):
        propagator.reset(_inertial_seed())
    for time in (20.0, 30.0, 40.0):
        bounded = floor.update(time, 10.0)
        unbounded = full.update(time, 10.0)
        assert bounded.position[1] == 50.0
        assert bounded.velocity[1] == 0.0
        assert bounded.position[0::2] == unbounded.position[0::2]
    observed = _inertial_seed(position=(5000.0, 20.0, 1000.0), velocity=(-100.0, 100.0, 0.0), time=40.0)
    floor.reset(observed)
    assert floor._position_estimate == observed.position
    assert floor._velocity_estimate == observed.velocity
    # Fresh observations are unmodified; the next coast applies the floor again.
    assert floor.update(41.0, 1.0).position[1] == 50.0
    floor.reset(_inertial_seed(time=41.0))
    solution = floor.update(42.0, 1.0)
    assert solution.position[1] > 50.0
    assert solution.velocity[1] == -100.0
    assert solution.sample_time_s == 41.0


def test_sensor_floor_uses_last_observation_and_does_not_clip_direct_updates():
    provider = SensorTrackProvider(_model(), min_altitude_m=50.0, datalink_disconnect_time_s=0.1)
    first = TargetState((10000.0, 2500.0, 1000.0), (-300.0, -100.0, 20.0))
    assert provider.update(0.0, 0.1, _missile(), first).position == first.position
    changed = TargetState((20000.0, 3000.0, 4000.0), (100.0, 200.0, 300.0))
    coast = provider.update(25.0, 25.0, _missile(), changed)
    assert coast.mode is TrackMode.INERTIAL
    assert coast.position == (2500.0, 50.0, 1550.0)
    assert coast.velocity == (-300.0, 0.0, 20.0)
    provider.datalink_disconnect_time_s = None
    provider.datalink_connected = True
    observed = TargetState((8000.0, 20.0, 0.0), (-100.0, -20.0, 0.0))
    fresh = provider.update(26.0, 1.0, _missile(), observed)
    assert fresh.mode is TrackMode.DATALINK
    assert fresh.position == observed.position
    assert fresh.velocity == observed.velocity
    assert provider.inertial._error == (0.0, 0.0, 0.0)


def test_radar_floor_does_not_prevent_reacquisition_or_clip_radar_observation():
    provider = SensorTrackProvider(_model(active_radar=True), lock_range_m=20000.0, min_altitude_m=50.0)
    tracked = TargetState((10000.0, 2500.0, 0.0), (-300.0, -100.0, 0.0))
    assert provider.update(0.0, 0.1, _missile(), tracked).mode is TrackMode.RADAR_TRACK
    lost = TargetState((30000.0, 2500.0, 0.0), (0.0, 0.0, 0.0))
    coast = provider.update(25.0, 25.0, _missile(), lost)
    assert coast.mode is TrackMode.INS_SEARCH
    assert coast.position[1] == 50.0
    # Reacquisition needs a causal pointing prior as well as the altitude floor.
    provider.inertial.reset(TrackSolution((10300., 50., 0.), (-300., 0., 0.),
        25., 25., TrackMode.INERTIAL, True, 'available_search_prior'))
    observed = TargetState((10000.0, 20.0, 0.0), (-300.0, 100.0, 0.0))
    fresh = provider.update(26.0, 1.0, _missile(), observed)
    assert fresh.mode is TrackMode.RADAR_TRACK
    assert fresh.position[1] == 20.0
    assert fresh.velocity[1] == 100.0
    assert provider.first_reacquire_time_s == 26.0
    assert provider.inertial._error == (0.0, 0.0, 0.0)


def test_kinematic_floor_and_reacquisition_keep_direct_observation():
    provider = KinematicTrackProvider(lock_range_m=20000.0, min_altitude_m=50.0)
    first = TargetState((10000.0, 2500.0, 0.0), (-300.0, -100.0, 0.0))
    provider.update(0.0, 0.1, _missile(), first)
    lost = provider.update(25.0, 25.0, _missile(), TargetState((30000.0, 2400.0, 0.0), (0.0, 0.0, 0.0)))
    assert lost.mode is TrackMode.INS_SEARCH
    assert lost.position == (2500.0, 50.0, 0.0)
    observed = TargetState((10000.0, 20.0, 0.0), (-300.0, 100.0, 0.0))
    fresh = provider.update(26.0, 1.0, _missile(), observed)
    assert fresh.position == observed.position
    assert fresh.velocity == observed.velocity
    assert provider.first_reacquire_time_s == 26.0


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, True, "50"])
def test_ins_floor_requires_finite_numeric_altitude(value):
    with pytest.raises(ValueError, match="min_altitude_m"):
        InertialTrackPropagator(min_altitude_m=value)
