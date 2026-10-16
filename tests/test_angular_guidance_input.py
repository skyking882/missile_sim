"""Public angular-input integration and causal replay boundaries."""
import copy
import json
import math
import struct
from dataclasses import replace
from pathlib import Path

import pytest

from aim120_model.angular_guidance import (AngularObservation, RecordedSeekerHistory,
                                         RecordedSeekerSample, native_tick_time)
from aim120_model.effective_surface_core import G, axes
from aim120_model.math3d import cross, dot, norm, scale, sub
from aim120_model.public_api import create_surface_missile, simulate
from aim120_model.surface_runtime import simulate_surface
from aim120_model.target import TargetState
from aim120_model.tracking import TrackMode, TrackSolution

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def profile():
    return json.loads((ROOT/'data/profile_sets/2.59.0.27/missiles/su_r_77_1.json').read_text())


class Truth:
    def state_at(self, t):
        # Deliberately differs from the seeker input.
        return TargetState((30000., 10000., -20000.), (0., 0., 0.))


def angular():
    return AngularObservation((1., 0., 0.), (.05, .02, -.03), 10000., 100.,
                              (0., 8000., 0.), (600., 0., 0.))


def runtime(profile, source=None):
    return create_surface_missile(profile, launch_position_m=(0., 8000., 0.),
        launch_velocity_mps=(600., 0., 0.), launch_pitch_deg=0., launch_heading_deg=0.,
        target=Truth(), observation_mode='ideal_truth', loft=False, end_time_s=.4,
        guidance_input=source)


def test_full_rate_native_parity_and_same_target_after_observer_change():
    a = angular()
    native = AngularObservation.from_native(direction=a.direction, full_rate_rad_s=(-.05, -.02, -.03),
        range_m=a.range_m, closing_mps=a.closing_mps,
        observer_position_m=a.observer_position_m, observer_velocity_mps=a.observer_velocity_mps)
    assert native == a
    p, v = a.target_motion()
    assert p == pytest.approx((10000., 8000., 0.))
    assert v == pytest.approx((500., 300., 200.))
    moved = a.at_observer((150., 8050., 20.), (620., 8., -15.))
    pp, vv = moved.target_motion()
    assert pp == pytest.approx(p)
    assert vv == pytest.approx(v)
    assert moved.axial_rate_rad_s == pytest.approx(a.axial_rate_rad_s)
    assert a.at_observer(a.observer_position_m, a.observer_velocity_mps).full_rate_rad_s == pytest.approx(a.full_rate_rad_s)


def test_actual_controller_consumes_full_rate_without_truth_or_axial_removal(profile):
    original = copy.deepcopy(profile)
    a = angular()
    calls = []
    def source(t, p, v):
        calls.append((t, p, v))
        return a.track_for(t, p, v)
    rt = runtime(profile, source)
    assert rt.autopilot['propNavMult'] == 6.
    assert rt.config['pid_error_frame'] == 'velocity_normal_projected_body'
    assert rt.config['pid_pressure_gain_convention'] == 'floor_one'
    # A unit hit-gain synthetic boundary isolates the full-rate PN mapping.
    rt.autopilot['_hit_gain'] = [[0., 1.]]
    assert rt.guidance(0., rt.state) == pytest.approx((0., -42., 102.))
    rt.track = replace(rt.track, full_rate_rad_s=(0., .02, -.03))
    assert rt.guidance(0., rt.state) == pytest.approx((0., 18., 12.))
    rt.track = source(0., rt.state[:3], rt.state[3:6])
    while not rt.done:
        rt.step()
    assert rt.last_request != (0., 0., 0.)
    assert any(x != 0. for x in rt.control)
    assert all(row['observed_target_velocity_mps'] == pytest.approx((500., 300., 200.)) for row in rt._rows)
    assert all(row['target_velocity_mps'] == (0., 0., 0.) for row in rt._rows)
    assert calls[-1][1] == rt.state[:3]
    assert profile == original  # No profile/PID/airframe coefficient mutation.
    json.dumps(rt.result(), allow_nan=False)


@pytest.mark.parametrize('entry', [simulate, simulate_surface])
def test_full_rate_reaches_batch_public_entries(profile, entry):
    a = angular()
    s = dict(launch_speed_kmh=2160., launch_altitude_m=8000., launch_pitch_deg=0., launch_heading_deg=0.,
        target_speed_kmh=0., target_altitude_m=8000., initial_distance_m=30000., target_azimuth_deg=0.,
        target_heading_deg=0., target_vertical_heading_deg=0., target_constant_turn_g=0.,
        max_simulation_time_s=.4, observation_mode='ideal_truth')
    result = entry(profile, s, guidance_input=lambda t, p, v: a.track_for(t, p, v))
    assert result['model']['pn_gain'] == 6.
    assert result['model']['observation_provider'] == 'external_guidance_input'
    assert result['samples'][-1]['guidance_full_rate_rad_s'] is not None
    assert result['samples'][-1]['commanded_acceleration_g'] != (0., 0., 0.)
    json.dumps(result, allow_nan=False)


def test_native_clock_and_no_future_trk_leak_on_each_capture():
    def f32(x):
        return struct.unpack('f', struct.pack('f', x))[0]
    a = angular()
    ins = RecordedSeekerSample(0., TrackMode.INERTIAL, (9000., 7500., 600.), (400., 10., -5.))
    samples = [ins,
        RecordedSeekerSample(f32(107/48), TrackMode.RADAR_TRACK, (1., 2., 3.), (999.,)*3, a),
        replace(ins, time_s=f32(148/48)),
        RecordedSeekerSample(f32(161/48), TrackMode.RADAR_TRACK, (1., 2., 3.), (999.,)*3, a)]
    history = RecordedSeekerHistory(samples)
    for i in (1, 3):
        start, end = history.samples[i-1].time_s, history.samples[i].time_s
        for t in ((start+end)/2, end-1e-8):
            track = history(t, a.observer_position_m, a.observer_velocity_mps)
            assert track.full_rate_rad_s is None
            assert track.position == pytest.approx(tuple(p+v*(t-start) for p, v in zip(ins.position_m, ins.velocity_mps)))
            assert track.velocity == ins.velocity_mps
        track = history(math.nextafter(end, -math.inf), a.observer_position_m, a.observer_velocity_mps)
        assert track.mode == TrackMode.RADAR_TRACK
        assert track.velocity == pytest.approx(a.target_motion()[1])
        assert track.full_rate_rad_s == pytest.approx(a.full_rate_rad_s)
    assert native_tick_time(f32(107/48)) == 107/48
    with pytest.raises(ValueError, match='native physics tick'):
        native_tick_time(2.23)


def test_acquisition_bridge_uses_prior_velocity_and_no_pre_capture_correction():
    # Constant reference missile motion; both target endpoint velocities known.
    own = lambda t: ((600*t, 8000., 0.), (600., 0., 0.))
    a = RecordedSeekerSample(0., TrackMode.INERTIAL, (9000., 7500., 600.), (400., 10., -5.))
    end = 1.
    angular_end = replace(angular(), observer_position_m=own(end)[0])
    b = RecordedSeekerSample(end, TrackMode.RADAR_TRACK, (0.,)*3, (0.,)*3, angular_end)
    history = RecordedSeekerHistory([a, b], capture_times_s={1.: .5}, reference_motion=own)
    for t in (.25, .5-1e-8):
        tr = history(t, *own(t))
        assert tr.mode == TrackMode.INERTIAL
        assert tr.velocity == a.velocity_mps
    for t in (.5, .75, 1.):
        tr = history(t, *own(t))
        u = (t-.5)/.5
        expected_v = tuple((1-u)*v0+u*v1 for v0, v1 in zip(a.velocity_mps, angular_end.target_motion()[1]))
        assert tr.velocity == pytest.approx(expected_v)
        n = sub(tr.position, own(t)[0]); n = scale(n, 1/norm(n))
        assert dot(n, tr.full_rate_rad_s) == pytest.approx(angular_end.axial_rate_rad_s)
    # A different second TRK cannot change the missing acquisition interval.
    c = replace(b, time_s=2., angular=replace(angular_end, full_rate_rad_s=(4., 5., 6.)))
    extended = RecordedSeekerHistory([a, b, c], capture_times_s={1.: .5}, reference_motion=own)
    assert extended(.75, *own(.75)) == history(.75, *own(.75))


def test_project_before_cap_and_feedback_in_same_plane(profile):
    rt = runtime(profile)
    assert rt.limit_acceleration_request((10000., 0., 100.), rt.state) == (0., 0., 100.)
    assert rt.limit_acceleration_request((10000., 0., 1000.), rt.state) == pytest.approx((0., 0., 50*G))
    # Achieved normal acceleration gives zero error even with longitudinal drag
    # and an incidence angle between the body and velocity frame.
    q = (0., 0., math.sin(.2), math.cos(.2))
    rt.state = rt.state[:6]+q+rt.state[10:]
    rt.guidance = lambda t, state: (0., 1., 2.)
    rt.previous_acceleration = (-90., 1., 2.)
    rt.update_control(.5, rt.state)
    assert rt.last_error == pytest.approx((0., 0.), abs=1e-12)


@pytest.mark.parametrize('ias_ratio,expected_ratio', [(.5, 1.), (2., 4.)])
def test_pressure_floor_keeps_low_speed_boost_only_removes_high_speed_reduction(profile, ias_ratio, expected_ratio):
    fixed, old = runtime(profile), runtime(profile)
    old.config['pid_pressure_gain_convention'] = 'inverse_square'
    for rt in (fixed, old):
        density = rt.atmosphere.sample(8000.).density_kg_m3
        speed = rt.autopilot['baseIndSpeed']*ias_ratio/(3.6*math.sqrt(density/1.225))
        rt.state = rt.state[:3]+(speed, 0., 0.)+rt.state[6:]
        rt.guidance = lambda t, state: (0., 1., 1.)
        rt.previous_acceleration = (0., 0., 0.)
    new_output = fixed.update_control(.5, fixed.state)
    old_output = old.update_control(.5, old.state)
    assert new_output == pytest.approx(tuple(expected_ratio*x for x in old_output))
    assert fixed.integral == old.integral
    assert fixed.pid_source == old.pid_source


def test_invalid_external_source_is_not_silently_replaced_with_truth(profile):
    with pytest.raises(ValueError, match='return TrackSolution'):
        runtime(profile, lambda t, p, v: None)
    with pytest.raises(ValueError, match='requested time'):
        runtime(profile, lambda t, p, v: TrackSolution((0.,)*3, (0.,)*3, 1., 1., TrackMode.INERTIAL, True, 'bad'))


@pytest.mark.parametrize('target_speed', [0., 200.])
def test_timed_hold_does_not_drag_last_measurement_with_moving_observer(target_speed):
    own = lambda t: ((500*t, 8000., 0.), (500., 0., 0.))
    initial = AngularObservation((1., 0., 0.), (.03, 0., 0.), 1000., 500.-target_speed, *own(0.))
    cached = replace(initial, observer_position_m=own(1.)[0])
    fresh = RecordedSeekerSample(0., TrackMode.RADAR_TRACK, (0.,)*3, (9999.,)*3,
                                 initial, last_measurement_time_s=0.)
    held = replace(fresh, time_s=1., angular=cached)
    history = RecordedSeekerHistory([fresh, held], reference_motion=own)
    for t in (.5, 1., 1.5):
        tr = history(t, *own(t))
        assert tr.position == pytest.approx((1000.+target_speed*t, 8000., 0.))
        assert tr.velocity == pytest.approx((target_speed, 0., 0.))
        n = sub(tr.position, own(t)[0]); n = scale(n, 1/norm(n))
        assert dot(n, tr.full_rate_rad_s) == pytest.approx(.03)
        assert tr.sample_time_s == 0.
    # Omitted timing keeps the old recorded-state interpolation unchanged.
    legacy = RecordedSeekerHistory([replace(fresh, last_measurement_time_s=None),
                                    replace(held, last_measurement_time_s=None)], reference_motion=own)
    assert legacy(1., *own(1.)).position == pytest.approx((1500., 8000., 0.))


def test_first_saved_hold_reconstructs_hidden_measurement_and_waits_for_fresh_resume():
    own = lambda t: ((500*t, 8000., 0.), (500., 0., 0.))
    # The cached range was measured at .5s; direction/closing are saved at1s.
    held_angle = AngularObservation((1., 0., 0.), (.03, 0., 0.), 750., 300., *own(1.))
    held = RecordedSeekerSample(1., TrackMode.RADAR_TRACK, (99.,)*3, (9999.,)*3,
                                held_angle, last_measurement_time_s=.5)
    fresh_angle = replace(held_angle, range_m=2000., observer_position_m=own(2.)[0])
    fresh = replace(held, time_s=2., angular=fresh_angle, last_measurement_time_s=2.)
    ins = RecordedSeekerSample(3., TrackMode.INERTIAL, (7000., 8000., 0.), (100., 0., 0.))
    history = RecordedSeekerHistory([held, fresh, ins], reference_motion=own)
    assert history(1., *own(1.)).position == pytest.approx((1100., 8000., 0.))
    assert history(1.5, *own(1.5)).position == pytest.approx((1200., 8000., 0.))
    assert history(2., *own(2.)).position == pytest.approx((3000., 8000., 0.))
    assert history(2.5, *own(2.5)).mode == TrackMode.RADAR_TRACK
    assert history(3., *own(3.)).position == ins.position_m
    assert history(3., *own(3.)).full_rate_rad_s is None


def test_fresh_angular_interpolation_is_unchanged_with_explicit_measurement_times():
    a = RecordedSeekerSample(0., TrackMode.RADAR_TRACK, (0.,)*3, (0.,)*3, angular())
    b = replace(a, time_s=1., angular=replace(a.angular, range_m=9000.,
                                            observer_position_m=(600., 8000., 0.)))
    legacy = RecordedSeekerHistory([a, b])
    timed = RecordedSeekerHistory([replace(a, last_measurement_time_s=0.),
                                   replace(b, last_measurement_time_s=1.)])
    for t in (0., .5, 1.):
        assert timed(t, (300., 8000., 0.), (600., 0., 0.)) == legacy(t, (300., 8000., 0.), (600., 0., 0.))


def test_invalid_held_timing_is_not_accepted_without_reference_motion():
    with pytest.raises(ValueError, match='not after'):
        RecordedSeekerSample(0., TrackMode.RADAR_TRACK, (0.,)*3, (0.,)*3, angular(),
                             last_measurement_time_s=1.)
    held = RecordedSeekerSample(1., TrackMode.RADAR_TRACK, (0.,)*3, (0.,)*3, angular(),
                                last_measurement_time_s=0.)
    with pytest.raises(ValueError, match='reference_motion'):
        RecordedSeekerHistory([held])
