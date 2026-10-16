import math
from types import SimpleNamespace

import pytest

from aim120_model.observation import SensorTrackProvider
from aim120_model.surface_runtime import SurfaceRuntime
from aim120_model.target import TargetState
from aim120_model.tracking import AlphaBetaStateObserver, TrackMode, TrackSolution


def track(p, v, t, mode=TrackMode.RADAR_TRACK, source="radar"):
    return TrackSolution(p, v, t, t, mode, True, source)


def test_three_dimensional_position_update_ignores_measurement_velocity():
    observer = AlphaBetaStateObserver(.8, .05)
    observer.initialize(track((0, 10, 20), (10, 20, 30), 0, TrackMode.DATALINK, "datalink"))
    filtered = observer.update(track((7, 18, 40), (999, -999, 999), .5))
    assert filtered.position == pytest.approx((6.6, 18.4, 39))
    assert filtered.velocity == pytest.approx((10.2, 19.8, 30.5))
    assert observer.last_prediction == (5, 20, 35)
    assert observer.last_innovation == (2, -2, 5)
    assert observer.update_count == 1
    with pytest.raises(RuntimeError, match="only be initialized once"):
        observer.initialize(filtered)


def test_prediction_gap_does_not_create_measurements_or_reset_velocity():
    observer = AlphaBetaStateObserver(.8, .05)
    observer.initialize(track((0, 0, 0), (10, 0, 0), 0))
    assert observer.predict(1) == (10, 0, 0)
    assert observer.predict(2) == (20, 0, 0)
    assert observer.update_count == 0
    assert observer.update(track((22, 0, 0), (999, 0, 0), 2)).velocity == pytest.approx((10.05, 0, 0))
    assert observer.last_dt_s == 2


def provider_with_stub_radar(monkeypatch, enabled):
    provider = SensorTrackProvider({"datalink": True, "inertial_navigation": True})
    if enabled:
        provider.enable_alpha_beta_observer(.8, .05)
    provider.enable_launcher_support(lambda t, truth: ("", (1000+10*t, 1000, 0), (10, 0, 0)))
    positions = {1: (1020, 1000, 0), 3: (1100, 1000, 0)}
    def observe(t, dt, state, truth):
        valid = t in positions
        return (SimpleNamespace(valid=valid, reason="" if valid else "no_measurement"),
                track(positions[t], (999, 0, 0), t) if valid else None)
    monkeypatch.setattr(provider.radar, "update", observe)
    return provider


def test_provider_initializes_once_from_support_and_ins_coasts_filtered_state(monkeypatch):
    provider = provider_with_stub_radar(monkeypatch, True)
    truth = TargetState((1000, 1000, 0), (999, 0, 0))
    initial = provider.update(0, 0, None, truth)
    assert initial.source == "datalink"
    first = provider.update(1, 1, None, truth)
    assert first.source == "radar_alpha_beta"
    assert first.position == pytest.approx((1018, 1000, 0))
    assert first.velocity == pytest.approx((10.5, 0, 0))
    assert provider.alpha_beta_observer.initial_solution == initial
    coast = provider.update(2, 1, None, truth)
    assert coast.position == pytest.approx((1028.5, 1000, 0))
    assert coast.velocity == first.velocity
    assert provider.alpha_beta_observer.update_count == 1
    reacquired = provider.update(3, 1, None, truth)
    assert reacquired.velocity == pytest.approx((12.025, 0, 0))
    assert provider.alpha_beta_observer.last_dt_s == 2


def test_default_provider_still_copies_original_radar_velocity(monkeypatch):
    provider = provider_with_stub_radar(monkeypatch, False)
    truth = TargetState((1000, 1000, 0), (999, 0, 0))
    provider.update(0, 0, None, truth)
    measured = provider.update(1, 1, None, truth)
    assert provider.alpha_beta_observer is None
    assert measured.position == (1020, 1000, 0)
    assert measured.velocity == (999, 0, 0)
    assert measured.source == "radar"


def test_surface_pn_consumes_filtered_track_position_and_velocity():
    observer = AlphaBetaStateObserver(.8, .05)
    observer.initialize(track((10000, 1000, 1000), (-300, 0, 0), 0, TrackMode.DATALINK))
    measurement = track((9970, 1000, 1010), (0, 0, -999), .1)
    filtered = observer.update(measurement)
    model = SurfaceRuntime.__new__(SurfaceRuntime)
    model.track = filtered
    model.autopilot = {"propNavMult": 3., "_hit_gain": [[0., 1.]], "_age_gain": [[0., 1.]],
                       "loftEnabled": False, "reqAccelMax": 40.}
    model.config = {"pn_candidate": "ideal_relative_velocity_normal", "time_to_hit_convention": "linear_closing"}
    state = (0., 1000., 0., 300., 0., 0., 0., 0., 0., 1., 0., 0., 0.)
    request = model.guidance(.1, state)
    assert model.target_velocity == filtered.velocity
    assert tuple(a+b*.1 for a,b in zip(model.target0,model.target_velocity)) == pytest.approx(filtered.position)
    model.track = measurement
    raw_request = model.guidance(.1, state)
    assert math.dist(request, raw_request) > 1.
