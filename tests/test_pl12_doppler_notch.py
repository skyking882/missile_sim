import json
from pathlib import Path

import pytest

from aim120_model.dynamics import SimState
from aim120_model.radar_seeker import RadarSeekerObserver
from aim120_model.target import TargetState


PROFILE_PATH = Path(__file__).resolve().parents[1] / "missiles" / "cn_pl12.json"


def _sensor_model() -> dict:
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    return profile["guidance"]["sensor_model"]


def _missile_state() -> SimState:
    return SimState(
        position=(0.0, 10_000.0, 0.0),
        velocity=(500.0, 0.0, 0.0),
        pitch=0.0,
        yaw=0.0,
        pitch_rate=0.0,
        yaw_rate=0.0,
        mass=160.0,
    )


def test_pl12_profile_exposes_datamine_radar_sensor_model() -> None:
    sensor = _sensor_model()
    radar = sensor["radar_seeker"]

    assert sensor["active_radar"] is True
    assert sensor["inertial_navigation"] is True
    assert sensor["datalink"] is True
    assert radar["active"] is True
    assert radar["doppler_speed"]["width_mps"] == pytest.approx(20.0)
    assert radar["doppler_speed"]["ref_width_mps"] == pytest.approx(80.0)


def test_pl12_notch_half_width_is_50_mps() -> None:
    observer = RadarSeekerObserver(_sensor_model(), lock_range_m=16_000.0)
    assert observer.notch_half_width_mps == pytest.approx(50.0)


def test_pl12_look_down_beaming_target_is_rejected_by_ground_clutter_notch() -> None:
    observer = RadarSeekerObserver(_sensor_model(), lock_range_m=16_000.0)
    missile = _missile_state()
    # Target is below the missile and moves perpendicular to the missile-target LOS,
    # so its ground-referenced radial velocity is approximately zero: a notch case.
    target = TargetState(
        position=(10_000.0, 9_000.0, 0.0),
        velocity=(0.0, 0.0, 300.0),
    )

    detection, solution = observer.update(0.0, 0.02, missile, target)

    assert detection.valid is False
    assert detection.reason == "ground_clutter_notch"
    assert abs(detection.ground_radial_speed_mps) <= detection.notch_half_width_mps
    assert solution is None


def test_pl12_non_notching_look_down_target_can_be_detected() -> None:
    observer = RadarSeekerObserver(_sensor_model(), lock_range_m=16_000.0)
    missile = _missile_state()
    target = TargetState(
        position=(10_000.0, 9_000.0, 0.0),
        velocity=(-300.0, 0.0, 0.0),
    )

    detection, solution = observer.update(0.0, 0.02, missile, target)

    assert detection.valid is True
    assert detection.reason == ""
    assert abs(detection.ground_radial_speed_mps) > detection.notch_half_width_mps
    assert solution is not None
