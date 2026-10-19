"""State/identity/time contracts without fitting a flight or observer gain."""
from dataclasses import FrozenInstanceError, replace
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from aim120_model.angular_guidance import _rotate
from aim120_model.math3d import add, dot, norm, scale, sub
from aim120_model.observation import SensorTrackProvider
from aim120_model.public_api import simulate, create_surface_missile
from aim120_model.surface_runtime import simulate_surface
from aim120_model.target import TargetState
from aim120_model.tracking import DatalinkObservation, TrackIdentity, TrackMode, TrackSolution
from test_observation_modes import _model, _missile

H = 1 / 48
PLANE = TrackIdentity('aircraft', 'plane-1')
OTHER = TrackIdentity('missile', 'missile-2')
TRUTH = TargetState((10000., 1000., 0.), (300., 0., 0.))


def test_strict_support_only_cannot_become_truth_while_legacy_stays_compatible():
    for strict in (False, True):
        p = SensorTrackProvider(_model(), strict_autonomous=strict)
        p.enable_launcher_support(lambda t, truth: '')
        track = p.update(0., H, _missile(), TRUTH)
        assert track.valid is not strict
        if strict:
            assert track.mode is TrackMode.LOST
            assert not p.inertial.initialized
            assert p.radar.require_search_estimate
        else:
            assert track.position == TRUTH.position
            assert track.estimate_source == 'legacy_truth_datalink'
        assert p.stage_snapshots == ()  # No implicit accumulating log.


def test_strict_packet_gap_repeat_stale_future_and_expiry_use_owned_history():
    p = SensorTrackProvider(_model(), strict_autonomous=True, target_identity=PLANE)
    p.inertial.break_lock_max_time_s = 1.
    packet = DatalinkObservation((1000., 1000., 0.), (10., 0., 0.), 0., 'launcher_filter', PLANE)
    reply = ['', packet]
    p.enable_launcher_support(lambda t, truth: reply[0] if len(reply) == 1 else tuple(reply))
    first = p.update(.2, .2, _missile(), TRUTH)
    assert first.position == (1002., 1000., 0.)
    assert first.sample_time_s == 0. and first.solution_time_s == first.publication_time_s == .2
    assert first.identity == PLANE and first.estimate_source == 'launcher_filter'
    reply[:] = ['']
    gap = p.update(.4, .2, _missile(), TRUTH)
    assert gap.position == pytest.approx((1004., 1000., .4))  # Existing INS drift retained.
    assert gap.identity == PLANE and gap.sample_time_s == 0.
    reply[:] = ['', replace(packet, position=(99999.,)*3)]  # Same stamp is not a new correction.
    repeated = p.update(.6, .2, _missile(), TRUTH)
    assert repeated.position[0] == pytest.approx(1006.)
    reply[:] = ['', replace(packet, sample_time_s=-1.)]
    stale = p.update(.8, .2, _missile(), TRUTH)
    assert stale.position[0] == pytest.approx(1008.)
    assert stale.sample_time_s == 0.
    reply[:] = ['']
    assert not p.update(1.01, .21, _missile(), TRUTH).valid
    reply[:] = ['', replace(packet, sample_time_s=2.)]
    with pytest.raises(ValueError, match='future'):
        p.update(1.02, .01, _missile(), TRUTH)


def echo(t, own, identity=PLANE, direction=(1., 0., 0.)):
    return TrackSolution(add(own.position, scale(direction, 10000.)), (50., 0., 100.),
                         t, t, TrackMode.RADAR_TRACK, True, 'synthetic_radar', identity=identity)


def install_echo(p, track, accepted=True, identity_key=('target',)):
    p.radar.measurement_identity = identity_key
    detection = SimpleNamespace(valid=accepted, reason='' if accepted else 'missing_echo',
                                relative_radial_speed_mps=-100., range_m=10000.)
    p.radar.update = lambda *args, **kwargs: (detection, track)


def test_connected_stages_seed_from_current_prior_then_hold_without_cache_rewrite():
    own = _missile()
    p = SensorTrackProvider(_model(active_radar=True), strict_autonomous=True,
                            observer_diagnostics=True, target_identity=PLANE)
    p.enable_q_angular_observer()
    full = (.04, -.03, .02)
    # A current legitimate prior arrives on the capture tick, before association.
    p.enable_launcher_support(lambda t, truth: ('', DatalinkObservation(
        (10000., 1500., 1000.), (10., 0., 100.), t, 'external_estimator', PLANE, full)))
    install_echo(p, echo(H, own))
    first = p.update(H, H, own, TRUTH)
    o = p.angular_observer
    assert o.capture_rate_seed == pytest.approx(full)  # Own Q is identity.
    assert o.capture_count == o.correction_count == 1
    assert first.identity == PLANE
    stages = p.stage_snapshots
    assert [s.stage for s in stages] == ['ins_refresh', 'prior_refresh', 'radar_association',
        'capture_handover', 'accepted_measurement', 'publish_guidance']
    handover = stages[3]
    assert handover.seed_source == 'available_full_prior'
    assert handover.seed_rate_world_rad_s == pytest.approx(full)
    assert handover.accepted_measurement_time_s is None
    assert handover.measurement_cache.sample_time_s == H
    assert stages[4].accepted_measurement_time_s == H
    assert stages[-1].publication_time_s == H
    assert stages[2].measurement_source == 'synthetic_radar_echo_instantaneous'
    assert stages[-1].radial_source == 'instantaneous_sensor_relative_radial_speed'
    with pytest.raises(FrozenInstanceError):
        handover.stage = 'changed'
    raw_cache = p.measurement_cache
    angular_cache = o.measurement_cache
    rate = first.full_rate_rad_s
    direction = o.direction
    # The Cartesian coast is intentionally absurd: it must not overwrite angle/rate/cache.
    held = replace(first, position=(9e8,)*3, mode=TrackMode.RADAR_COAST,
                   solution_time_s=3*H, publication_time_s=None)
    install_echo(p, held, False)
    result = p.update(3*H, 2*H, own, TRUTH)
    assert result.full_rate_rad_s == pytest.approx(rate)
    assert o.direction == pytest.approx(_rotate(direction, rate, -2*H))
    assert o.measurement_cache is angular_cache and p.measurement_cache is raw_cache
    assert o.sample_time_s == result.sample_time_s == H
    assert o.correction_count == 1
    assert o.range_m == pytest.approx(10000.-100.*2*H)
    assert [s.stage for s in p.stage_snapshots][-2:] == ['held_propagation', 'publish_guidance']
    assert stages[3] == handover  # Earlier snapshots were not mutated by later updates.
    assert p.stage_snapshots[-1].previous_accepted_time_s == H
    assert p.stage_snapshots[-1].accepted_measurement_time_s == H
    assert p.stage_snapshots[-1].angular_state_time_s == 3*H
    # Same target and a new sensor source is not a reflector change/Q reset.
    fresh = replace(echo(4*H, own, direction=o.direction), source='another_radar_transport')
    install_echo(p, fresh)
    p.update(4*H, H, own, TRUTH)
    assert o.capture_count == 1 and o.correction_count == 2
    # A known different physical object gets an explicit experimental handover.
    install_echo(p, echo(5*H, own, OTHER), identity_key=('decoy', 2))
    transferred = p.update(5*H, H, own, TRUTH)
    assert transferred.identity == p.estimator_identity == p.selected_reflector_identity == OTHER
    assert p.intended_target_identity == PLANE
    assert transferred.full_rate_rad_s == (0., 0., 0.)
    handover2 = next(x for x in p.stage_snapshots if x.stage == 'capture_handover')
    assert handover2.reason == 'reflector_changed'
    assert handover2.seed_source == 'experimental_zero_without_same_object_prior'


def test_ins_target_direction_is_distinct_from_gimbal_candidate():
    p = SensorTrackProvider(_model(active_radar=True), strict_autonomous=True,
                           observer_diagnostics=True, target_identity=PLANE)
    # 70 degree yaw exceeds per-axis 55 degree travel, but is a target estimate.
    n = (math.cos(math.radians(70.)), 0., math.sin(math.radians(70.)))
    position = add(_missile().position, scale(n, 1000.))
    p.enable_launcher_support(lambda t, truth: ('', DatalinkObservation(position, (0.,)*3, t, identity=PLANE)))
    result = p.update(1., H, _missile(), TargetState(position, (0.,)*3))
    assert p.target_los_estimate_world == pytest.approx(n)
    axis = p.antenna_axis_world
    assert axis is not None
    assert math.degrees(math.acos(dot(axis, n))) == pytest.approx(15.)
    assert result.identity == PLANE
    assert p.ins_estimate.position == position
    assert p.ins_estimate.identity == PLANE


def test_radar_owns_raw_cache_and_inertial_keeps_identity_and_full_state():
    p = SensorTrackProvider(_model(active_radar=True), target_identity=PLANE)
    p.enable_q_angular_observer()
    p.enable_launcher_support(lambda t, truth: ('', (10000., 1000., 0.), (0., 0., 100.)))
    p.update(0., H, _missile(), TargetState((30000., 1000., 0.), (0.,)*3))
    track = p.update(49*H, H, _missile(), TRUTH)
    assert track.valid and track.identity == PLANE
    raw = p.radar.measurement_cache
    assert raw.source == 'radar' and raw.identity == PLANE
    assert raw.full_rate_rad_s is None
    assert track.full_rate_rad_s is not None
    assert p.radar.last_solution.full_rate_rad_s is not None
    held = p.inertial.update(1.1, .1)
    assert held.identity == PLANE
    assert held.full_rate_rad_s is None
    assert p.inertial.estimate.full_rate_rad_s == track.full_rate_rad_s
    assert held.sample_time_s == track.sample_time_s


ROOT = Path(__file__).resolve().parents[1]


def test_production_options_are_connected_bounded_and_do_not_change_full_rate_pn():
    profile = json.loads((ROOT/'data/profile_sets/2.59.0.27/missiles/su_r_77_1.json').read_text())
    original = copy.deepcopy(profile)
    class Target:
        def state_at(self, t):
            return TargetState((10000., 8000., 1000.+100.*t), (0., 0., 100.))
    args = dict(launch_position_m=(0., 8000., 0.), launch_velocity_mps=(600., 0., 0.),
                launch_pitch_deg=0., launch_heading_deg=0., target=Target(), end_time_s=1., loft=False)
    support = lambda t, truth: ('', DatalinkObservation((10000., 8000., 1000.+100.*t),
        (0., 0., 100.), t, identity=PLANE, full_rate_rad_s=(.02, -.01, .03)))
    rt = create_surface_missile(profile, **args, launcher_support=support, strict_autonomous=True,
        observer_diagnostics=True, target_identity=PLANE, angular_observer='q_capture')
    while not rt.done:
        rt.step()
        assert len(rt.provider.stage_snapshots) <= 7
    result = rt.result()
    assert result['observer_state_flow']['strict_autonomous'] is True
    assert rt.track.full_rate_rad_s is not None
    assert rt.guidance_kinematics(rt.time_s, rt.state)[2] == pytest.approx(scale(rt.track.full_rate_rad_s, -1.))
    json.dumps(result, allow_nan=False)
    assert profile == original
    with pytest.raises(ValueError, match='state-flow'):
        create_surface_missile(profile, **args, observation_mode='ideal_truth', strict_autonomous=True)
    with pytest.raises(ValueError, match='state-flow'):
        create_surface_missile(profile, **args, strict_autonomous=True, guidance_input=lambda *args: None)


@pytest.mark.parametrize('entry', [simulate, simulate_surface])
def test_batch_strict_contract_reaches_the_provider(entry):
    from test_q_angular_observer import scenario
    profile = json.loads((ROOT/'data/profile_sets/2.59.0.27/missiles/su_r_77_1.json').read_text())
    result = entry(profile, scenario(), strict_autonomous=True,
                   observer_diagnostics=True, target_identity=PLANE)
    flow = result['model']['observer_state_flow']
    assert flow['strict_autonomous'] is True
    assert flow['identity'] == {'object_type': 'aircraft', 'object_id': 'plane-1'}
    assert flow['stages'][-1]['stage'] == 'publish_guidance'
    assert all(s['track_mode'] in ('lost', 'radar_search') for s in result['samples'])
    json.dumps(result, allow_nan=False)


def test_next_call_early_stages_do_not_republish_previous_echo_as_current():
    own = _missile()
    p = SensorTrackProvider(_model(active_radar=True), observer_diagnostics=True,
                           target_identity=PLANE)
    install_echo(p, echo(H, own))
    p.update(H, H, own, TRUTH)
    previous_cache = p.measurement_cache
    assert p.last_detection.valid
    assert p.stage_snapshots[-1].measurement_range_m == 10000.
    # Next evaluation accepts another echo. The preceding INS stage must still
    # show no current echo, while retaining the previous accepted cache/time.
    install_echo(p, echo(2*H, own))
    p.update(2*H, H, own, TRUTH)
    early = next(x for x in p.stage_snapshots if x.stage == 'ins_refresh')
    association = next(x for x in p.stage_snapshots if x.stage == 'radar_association')
    assert early.measurement_range_m is None
    assert early.measurement_relative_radial_mps is None
    assert early.measurement_source is None
    assert early.measurement_cache is previous_cache
    assert early.previous_accepted_time_s == H
    assert association.measurement_range_m == 10000.
    assert association.measurement_relative_radial_mps == -100.
    assert association.measurement_source == 'synthetic_radar_echo_instantaneous'
    assert association.measurement_cache.sample_time_s == 2*H
