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
            "receiver": {"range_m": 16000.0, "range_max_m": 25000.0},
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
    assert detection.reason == "angle_gate"
    assert solution is None


def test_lock_loss_enters_search_and_reacquires_after_gate_reset():
    observer = RadarSeekerObserver(_sensor_model(), lock_range_m=20000.0)
    first, solution = observer.update(0.0, 0.1, _missile(), _target())
    assert first.valid and solution is not None
    assert observer.distance_gate.initialized is True
    short_loss_detection, short_loss = observer.update(0.5, 0.5, _missile(), _target(y=900.0, vx=0.0))
    assert short_loss_detection.reason == "ground_clutter_notch"
    assert short_loss is None
    assert observer.state is SeekerState.SEARCH
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


def test_look_down_notch_boundary_is_deterministic_and_look_up_is_not_notched():
    for radial_speed, expected_reason in ((0.0, "ground_clutter_notch"), (49.0, "ground_clutter_notch"), (51.0, "")):
        observer = RadarSeekerObserver(_sensor_model(), lock_range_m=20000.0)
        detection, _ = observer.update(0.0, 0.1, _missile(), _target(y=900.0, vx=radial_speed))
        assert detection.reason == expected_reason

    observer = RadarSeekerObserver(_sensor_model(), lock_range_m=20000.0)
    detection, _ = observer.update(0.0, 0.1, _missile(), _target(y=1100.0, vx=0.0))
    assert detection.valid is True
    assert detection.reason == ""
