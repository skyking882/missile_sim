"""Causal complete-rate approximation, native parity and public integration."""
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from aim120_model.aerodynamics import body_axes, body_axes_for_state
from aim120_model.angular_guidance import _rotate
from aim120_model.math3d import add, cross, dot, norm, scale, sub
from aim120_model.observation import SensorTrackProvider
from aim120_model.public_api import create_surface_missile, simulate
from aim120_model.q_angular_observer import QCaptureAngularObserver
from aim120_model.surface_runtime import simulate_surface
from aim120_model.target import TargetState
from aim120_model.tracking import TrackMode, TrackSolution

ROOT = Path(__file__).resolve().parents[1]


def missile(pitch=.3, yaw=.7):
    return SimpleNamespace(position=(0., 5000., 0.), velocity=(600., 0., 0.), pitch=pitch, yaw=yaw)


def solution(t, position=(10000., 5500., 3000.), velocity=(0., 100., 200.),
             mode=TrackMode.DATALINK, sample=None, valid=True):
    return TrackSolution(position, velocity, t if sample is None else sample, t, mode, valid, 'received')


def advance(o, s, own, received=False, identity=('target',), closing=100.):
    return o.update(s, own, received=received, identity=identity, closing_mps=closing)


def test_q_once_uses_previous_full_body_posture_and_native_axial_parity():
    o = QCaptureAngularObserver()
    old = missile()
    prior = solution(0.)
    advance(o, prior, old)
    n = scale(sub(prior.position, old.position), 1 / norm(sub(prior.position, old.position)))
    a = scale(cross(sub(prior.velocity, old.velocity), n), 1 / norm(sub(prior.position, old.position)))
    axes = body_axes_for_state(old)
    expected = add(add(scale(axes.forward, a[0]), scale(axes.up, a[1])), scale(axes.right, a[2]))
    captured = advance(o, solution(.02, mode=TrackMode.RADAR_TRACK), missile(-.6, -1.), True)
    assert captured.full_rate_rad_s == pytest.approx(expected)
    assert abs(dot(n, expected)) > .001
    assert o.capture_count == 1
    # Independent native construction: C flips polar z, axial = -C.
    native_a = (-a[0], -a[1], a[2])
    C = lambda v: (v[0], v[1], -v[2])
    # Q_native = C Q_runtime C, never merely rotate about the velocity axis.
    native_cols = (C(axes.forward), C(axes.up), scale(C(axes.right), -1.))
    native_q_a = tuple(sum(native_cols[j][i] * native_a[j] for j in range(3)) for i in range(3))
    assert expected == pytest.approx(scale(C(native_q_a), -1.))
    # Next echo is precisely the prediction. Q must not be repeated.
    predicted = _rotate(n, expected, -.02)
    p = add(missile().position, scale(predicted, 10000.))
    next_track = advance(o, solution(.04, p, mode=TrackMode.RADAR_TRACK), missile(), True)
    assert next_track.full_rate_rad_s == pytest.approx(expected)
    assert o.capture_count == 1


def test_innovation_sign_duplicate_coast_and_actual_expiry():
    o = QCaptureAngularObserver()
    own = missile(0., 0.)
    p = (10000., 5000., 0.)
    first = advance(o, solution(0., p, mode=TrackMode.RADAR_TRACK), own, True)
    assert first.full_rate_rad_s == (0., 0., 0.)  # No prior: explicit causal zero.
    angle = .01
    m = (math.cos(angle), math.sin(angle), 0.)
    corrected = advance(o, solution(.02, add(own.position, scale(m, 10000.)), mode=TrackMode.RADAR_TRACK), own, True)
    assert corrected.full_rate_rad_s == pytest.approx((0., 0., -11.23 * angle))
    count = o.correction_count
    # A repeated accepted packet does not inject its different direction.
    duplicate = advance(o, solution(.04, (0., 5000., 10000.), mode=TrackMode.RADAR_TRACK, sample=.02), own, True)
    assert o.correction_count == count
    assert duplicate.full_rate_rad_s == pytest.approx(corrected.full_rate_rad_s)
    held = advance(o, solution(.06, (1e8, 1e8, 1e8), mode=TrackMode.RADAR_COAST, sample=.02), own)
    assert held.mode is TrackMode.RADAR_COAST
    assert held.sample_time_s == .02
    assert held.full_rate_rad_s == pytest.approx(corrected.full_rate_rad_s)
    assert norm(sub(held.position, own.position)) == pytest.approx(9996.)
    assert o.correction_count == count
    # Genuine expiry resumes the Cartesian prior and permits one new Q reset.
    ins = solution(.08, (10000., 5000., 1000.), mode=TrackMode.INS_SEARCH)
    advance(o, ins, own)
    reacquired = advance(o, solution(.1, p, mode=TrackMode.RADAR_TRACK), own, True)
    assert o.capture_count == 2
    assert reacquired.full_rate_rad_s != corrected.full_rate_rad_s


def test_different_reflector_does_not_inherit_rate_even_after_loss():
    o = QCaptureAngularObserver()
    own = missile()
    advance(o, solution(0.), own)
    a = advance(o, solution(.02, mode=TrackMode.RADAR_TRACK), own, True)
    assert norm(a.full_rate_rad_s) > 0
    b = advance(o, solution(.04, mode=TrackMode.RADAR_TRACK), own, True, ('decoy', 3))
    assert b.full_rate_rad_s == pytest.approx((0., 0., 0.), abs=1e-15)
    advance(o, solution(.06, mode=TrackMode.INS_SEARCH), own)
    c = advance(o, solution(.08, mode=TrackMode.RADAR_TRACK), own, True)
    assert c.full_rate_rad_s == pytest.approx((0., 0., 0.), abs=1e-15)


@pytest.fixture
def profile():
    return json.loads((ROOT / 'data/profile_sets/2.59.0.27/missiles/su_r_77_1.json').read_text())


def scenario():
    return dict(launch_speed_kmh=2160., launch_altitude_m=8000., launch_pitch_deg=4.,
        launch_heading_deg=12., target_speed_kmh=1200., target_altitude_m=8200.,
        initial_distance_m=10000., target_azimuth_deg=20., target_heading_deg=90.,
        target_vertical_heading_deg=0., target_constant_turn_g=0., max_simulation_time_s=1.1,
        observation_mode='sensor_track', loft_enabled=False)


@pytest.mark.parametrize('entry', [simulate, simulate_surface])
def test_public_batch_observer_is_causal_opt_in_and_feeds_full_rate_to_pn(profile, entry):
    original = copy.deepcopy(profile)
    baseline = entry(profile, scenario())
    explicit_none = entry(profile, scenario(), angular_observer=None)
    assert baseline == explicit_none
    result = entry(profile, scenario(), angular_observer='q_capture')
    assert result['model']['angular_observer']['rate_correction_gain_per_s'] == 11.23
    assert result['model']['angular_observer']['kind'] == 'assumed'
    tracks = [r for r in result['samples'] if r.get('guidance_full_rate_rad_s') is not None]
    assert tracks
    assert any(abs(dot(scale(sub(r['observed_target_position_m'], r['position_m']),
        1/norm(sub(r['observed_target_position_m'], r['position_m']))), r['guidance_full_rate_rad_s'])) > 1e-6 for r in tracks)
    assert any(norm(r['commanded_acceleration_g']) > 0 for r in tracks)
    assert profile == original
    json.dumps(result, allow_nan=False)


def test_step_api_matches_batch_and_unsupported_options_fail(profile):
    class Target:
        def state_at(self, t):
            return TargetState((10000., 8200., 1000. + 300*t), (0., 0., 300.))
    kwargs = dict(launch_position_m=(0., 8000., 0.), launch_velocity_mps=scale(body_axes(math.radians(4.), math.radians(12.)).forward, 600.),
        launch_pitch_deg=4., launch_heading_deg=12., target=Target(), loft=False, end_time_s=1.1)
    rt = create_surface_missile(profile, **kwargs, angular_observer='q_capture')
    while not rt.done:
        rt.step()
    result = rt.result()
    batch = simulate_surface(profile, scenario(), target_factory=lambda _: Target(), angular_observer='q_capture')
    assert len(result['samples']) == len(batch['samples'])
    for stepped, batched in zip(result['samples'], batch['samples']):
        assert stepped['track_mode'] == batched['track_mode']
        for key in ('position_m', 'velocity_mps', 'observed_target_position_m',
                    'observed_target_velocity_mps', 'guidance_full_rate_rad_s'):
            if stepped[key] is None:
                assert batched[key] is None
            else:
                assert stepped[key] == pytest.approx(batched[key], abs=1e-9)
    assert result['angular_observer']['capture_count'] >= 1
    assert result['angular_observer']['correction_count'] >= 1
    assert rt.track.full_rate_rad_s is not None
    assert rt.guidance_kinematics(rt.time_s, rt.state)[2] == pytest.approx(scale(rt.track.full_rate_rad_s, -1.))
    with pytest.raises(ValueError, match='cannot alter external'):
        create_surface_missile(profile, **kwargs, angular_observer='q_capture', guidance_input=lambda *a: None)
    with pytest.raises(ValueError, match='mutually exclusive'):
        create_surface_missile(profile, **kwargs, angular_observer='q_capture', target_observer='unified_alpha_beta')
    with pytest.raises(ValueError, match='active radar'):
        create_surface_missile(profile, **kwargs, observation_mode='ideal_truth', angular_observer='q_capture')


def test_provider_uses_available_estimate_and_accepted_echo_only(profile):
    providers = []
    for unseen_velocity in ((1234., -999., 456.), (-444., 888., -777.)):
        p = SensorTrackProvider(profile['guidance']['sensor_model'])
        p.enable_q_angular_observer()
        p.enable_launcher_support(lambda t, truth: ('', (10000., 5500., 3000.), (0., 100., 200.)))
        own = missile()
        prior = p.update(0., 0., own, TargetState((1e7, 1e7, 1e7), unseen_velocity))
        assert prior.mode is TrackMode.DATALINK
        assert prior.velocity == (0., 100., 200.)
        # Receive the same valid echo independently of the unobservable truth.
        echo = solution(1., mode=TrackMode.RADAR_TRACK)
        p.radar.measurement_identity = ('target',)
        detection = SimpleNamespace(valid=True, reason='', relative_radial_speed_mps=-100.)
        p.radar.update = lambda *args, **kwargs: (detection, echo)
        track = p.update(1., 1., own, TargetState((1e7, 1e7, 1e7), unseen_velocity))
        providers.append((track, p.angular_observer.prior_rate))
    assert providers[0] == providers[1]
