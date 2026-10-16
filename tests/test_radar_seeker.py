import pytest

from aim120_model.dynamics import SimState
from aim120_model.radar_seeker import RadarSeekerObserver, SeekerState
from aim120_model.target import TargetState
from aim120_model.tracking import TrackMode


def _missile() -> SimState:
    return SimState((0.0, 1000.0, 0.0), (300.0, 0.0, 0.0), 0.0, 0.0, 0.0, 0.0, 100.0)


def _sensor_model(**overrides):
    model = {
        "active_radar": True,
        "use_target_velocity": True,
        "radar_seeker": {
            "active": True,
            "angle_gate_rate_deg_s": 30.0,
            "prolongation_time_max_s": 1.0,
            "lock_angle_max_deg": 55.0,
            "angle_max_deg": 55.0,
            "rate_max_deg_s": 60.0,
            "doppler_speed_gate": {"filter_alpha": 0.8, "filter_beta": 0.05, "search_range_mps": 300.0},
            "dist_gate": {"filter_alpha": 0.8, "filter_beta": 0.05, "search_range_m": 5000.0},
            "receiver": {"range_m": 16000.0, "range_max_m": 25000.0, "rcs_m2": 1., "antenna": {"angle_half_sens_deg": 15., "side_lobes_sensitivity_db": -30.}},
            "transmitter": {"antenna": {"angle_half_sens_deg": 15., "side_lobes_sensitivity_db": -30.}},
            "doppler_speed": {"min_mps": -3000.0, "max_mps": 3000.0, "width_mps": 20.0, "ref_width_mps": 80.0, "signal_width_min_mps": 5.0},
        },
    }
    model.update(overrides)
    return model


def _target(y=1000.0, vx=-300.0):
    return TargetState((10000.0, y, 0.0), (vx, 0.0, 0.0))


def test_head_on_target_inside_gates_tracks():
    observer = RadarSeekerObserver(_sensor_model(), lock_range_m=20000.0)
    detection, solution = observer.update(0.0, 1.0 / 256.0, _missile(), _target())
    assert detection.valid is True
    assert solution is not None
    assert solution.mode is TrackMode.RADAR_TRACK
    assert observer.state is SeekerState.TRACK


def test_off_boresight_target_is_rejected():
    observer = RadarSeekerObserver(_sensor_model(), lock_range_m=20000.0)
    target = TargetState((0.0, 1000.0, 10000.0), (0.0, 0.0, -300.0))
    detection, solution = observer.update(0.0, 1.0 / 256.0, _missile(), target)
    assert detection.valid is False
    assert detection.reason == "beam_range_gate"
    assert solution is None


def test_lock_loss_coasts_then_enters_search_and_reacquires_after_gate_reset():
    observer = RadarSeekerObserver(_sensor_model(), lock_range_m=20000.0)
    first, solution = observer.update(0.0, 0.1, _missile(), _target())
    assert first.valid and solution is not None
    assert observer.distance_gate.initialized is True
    short_loss_detection, short_loss = observer.update(0.5, 0.5, _missile(), _target(y=900.0, vx=0.0))
    assert short_loss_detection.reason == "ground_clutter_notch"
    assert short_loss.mode is TrackMode.RADAR_COAST
    assert short_loss.sample_time_s == 0.0
    assert short_loss.position == (9850.0, 1000.0, 0.0)
    assert observer.state is SeekerState.COAST
    assert observer.distance_gate.initialized is True
    lost_detection, lost = observer.update(1.1, 0.6, _missile(), _target(y=900.0, vx=0.0))
    assert lost_detection.reason == "ground_clutter_notch"
    assert lost is None
    assert observer.state is SeekerState.SEARCH
    assert observer.distance_gate.initialized is False
    recapture_detection, recapture = observer.update(1.2, 0.1, _missile(), _target(y=1000.0, vx=-300.0))
    assert recapture_detection.valid is True
    assert recapture is not None and recapture.mode is TrackMode.RADAR_TRACK
    assert observer.state is SeekerState.TRACK


@pytest.mark.parametrize('decoys', [None, []])
def test_coast_clock_is_last_echo_not_last_failed_tick(decoys):
    observer = RadarSeekerObserver(_sensor_model())
    start = 31.895833333333332
    observer.update(start, .02, _missile(), _target(), decoys=decoys)
    missing = _target(y=900., vx=0.)
    for t in (start+.25, start+.75, 32.895833333333336):
        detection, solution = observer.update(t, .02, _missile(), missing, decoys=decoys)
        assert not detection.valid
        assert solution.mode is TrackMode.RADAR_COAST
        assert solution.sample_time_s == start
        assert solution.position[0] == pytest.approx(10000.-300.*(t-start))
    assert observer.update(start+1.000001, .02, _missile(), missing, decoys=decoys)[1] is None
    assert observer._last_detection_time_s == start
    assert observer.update(start+1.1, .02, _missile(), _target(), decoys=decoys)[1].mode is TrackMode.RADAR_TRACK


def test_no_echo_history_cannot_coast():
    observer = RadarSeekerObserver(_sensor_model())
    detection, solution = observer.update(.5, .02, _missile(), _target(y=900., vx=0.))
    assert not detection.valid and solution is None
    assert observer.state is SeekerState.SEARCH


def test_look_down_notch_boundary_is_deterministic_and_look_up_is_not_notched():
    for radial_speed, expected_reason in ((0.0, "ground_clutter_notch"), (49.0, "ground_clutter_notch"), (51.0, "")):
        observer = RadarSeekerObserver(_sensor_model(), lock_range_m=20000.0)
        detection, _ = observer.update(0.0, 0.1, _missile(), _target(y=900.0, vx=radial_speed))
        assert detection.reason == expected_reason

    observer = RadarSeekerObserver(_sensor_model(), lock_range_m=20000.0)
    detection, _ = observer.update(0.0, 0.1, _missile(), _target(y=1100.0, vx=0.0))
    assert detection.valid is True
    assert detection.reason == ""


@pytest.mark.parametrize('path', ['ordinary', 'empty_decoys', 'decoys', 'search', 'track_gate', 'vg_capture'])
def test_launch_startup_blocks_all_radar_paths_without_measurement_history(path):
    from aim120_model.chaff import Reflector
    from aim120_model.radar_seeker import SeekerSearch
    observer = RadarSeekerObserver(_sensor_model(lock_timeout_s=.75))
    kwargs = {}
    if path == 'empty_decoys': kwargs['decoys'] = []
    elif path == 'decoys': kwargs['decoys'] = [Reflector('chaff', (10000., 1000., 0.), (-300., 0., 0.), .001, 'startup')]
    elif path == 'search': observer.search = SeekerSearch(observer, {'scan_half_angle_deg': 0.})
    elif path == 'track_gate': observer.enable_track_gate_model({'notch_track_hazard': []})
    elif path == 'vg_capture': observer.enable_chaff_capture({'p_max': 1., 'v0_mps': 10000.})
    for time in (0., .5, .75-2e-12):
        detection, solution = observer.update(time, 1/48, _missile(), _target(), **kwargs)
        assert detection.reason == 'startup_delay' and not detection.valid and solution is None
        assert not observer.distance_gate.initialized and not observer.doppler_gate.initialized
        assert observer._last_detection_time_s is None
        assert observer._last_measurement_time_s is None
        if observer.search is not None:
            assert observer.search.search_start_s is None and observer.search.candidate is None
    detection, solution = observer.update(.75, 1/48, _missile(), _target(), **kwargs)
    assert detection.valid and solution.mode is TrackMode.RADAR_TRACK
    assert observer._last_detection_time_s == .75


def test_startup_is_launch_age_not_time_since_first_call_or_reacquisition():
    observer = RadarSeekerObserver(_sensor_model(lock_timeout_s=.75))
    assert observer.update(3., .02, _missile(), _target())[0].valid
    missing = _target(y=900., vx=0.)
    assert observer.update(3.5, .02, _missile(), missing)[1].mode is TrackMode.RADAR_COAST
    assert observer.update(4.000000000000001, .02, _missile(), missing)[1].mode is TrackMode.RADAR_COAST
    assert observer.update(4.000001, .02, _missile(), missing)[1] is None
    assert observer.update(4.01, .02, _missile(), _target())[0].valid


def test_startup_and_explicit_scan_dwell_are_distinct_clocks():
    from aim120_model.radar_seeker import SeekerSearch
    observer = RadarSeekerObserver(_sensor_model(lock_timeout_s=.75))
    observer.search = SeekerSearch(observer, {'scan_half_angle_deg': 0., 'dwell_s': .1})
    assert not observer.update(.5, .02, _missile(), _target())[0].valid
    assert not observer.update(.75, .02, _missile(), _target())[0].valid
    assert observer.update(.85, .02, _missile(), _target())[0].valid
    assert observer.search.first_lock_time_s == .85
    default = RadarSeekerObserver(_sensor_model(lock_timeout_s=.75))
    default.search = SeekerSearch(default, {'scan_half_angle_deg': 0.})
    assert default.search.dwell_s == 0.
    assert default.update(.75-2e-16, .02, _missile(), _target())[0].valid


def test_startup_uses_each_profile_delay_and_preserves_missing_or_inactive():
    observer = RadarSeekerObserver(_sensor_model(lock_timeout_s=1.5))
    assert observer.update(.75, .02, _missile(), _target())[0].reason == 'startup_delay'
    assert observer.update(1.5, .02, _missile(), _target())[0].valid
    assert RadarSeekerObserver(_sensor_model()).update(0., .02, _missile(), _target())[0].valid
    inactive = _sensor_model(lock_timeout_s=.75); inactive['radar_seeker']['active'] = False
    assert RadarSeekerObserver(inactive).update(0., .02, _missile(), _target())[0].reason == 'inactive'


@pytest.mark.parametrize('filter_kind', ['none', 'radar', 'unified'])
def test_provider_uses_datalink_then_ins_during_startup(filter_kind):
    from aim120_model.observation import SensorTrackProvider
    model = _sensor_model(lock_timeout_s=.75, datalink=True, inertial_navigation=True,
                          inertial_drift_speed_mps=0., break_lock_max_time_s=100.)
    provider = SensorTrackProvider(model, datalink_disconnect_time_s=.5)
    provider.enable_launcher_support(lambda time, truth: ("", _target().position, _target().velocity))
    if filter_kind == 'radar': provider.enable_alpha_beta_observer(.8, .05)
    elif filter_kind == 'unified': provider.enable_unified_target_observer()
    first = provider.update(0., 0., _missile(), _target())
    assert first.mode is TrackMode.DATALINK
    assert not provider.radar_has_tracked_once
    second = provider.update(.5, .5, _missile(), _target())
    assert second.mode in (TrackMode.INERTIAL, TrackMode.INS_SEARCH)
    assert provider.radar._last_measurement_time_s is None
    if filter_kind == 'radar': assert provider.alpha_beta_observer.update_count == 0
    acquired = provider.update(.75, .25, _missile(), _target())
    assert acquired.mode is TrackMode.RADAR_TRACK
    assert acquired.sample_time_s == .75
