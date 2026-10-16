"""Information boundary tests using real detector branches and packet cadence."""
import copy
from pathlib import Path

import pytest

from aim120_model.chaff import Reflector
from aim120_model.dynamics import SimState
from aim120_model.observation import SensorTrackProvider
from aim120_model.radar_seeker import SeekerSearch
from aim120_model.target import TargetState
from aim120_model.tracking import DatalinkObservation, TrackMode


def missile():
    return SimState((0., 1000., 0.), (300., 0., 0.), 0., 0., 0., 0., 100.)


def provider(branch="normal", datalink=False):
    sensor = dict(active_radar=True, datalink=datalink, inertial_navigation=True,
        inertial_drift_speed_mps=0., break_lock_max_time_s=100.,
        radar_seeker=dict(active=True, rate_max_deg_s=180., angle_gate_rate_deg_s=180.,
            lock_angle_max_deg=90., angle_max_deg=90., receiver=dict(range_max_m=20000.,range_m=20000.,rcs_m2=1.,
                antenna=dict(angle_half_sens_deg=15.,side_lobes_sensitivity_db=-30.)),
            transmitter=dict(antenna=dict(angle_half_sens_deg=15.,side_lobes_sensitivity_db=-30.)),
            distance=dict(width_m=300.)))
    p = SensorTrackProvider(sensor, lock_range_m=20000.)
    p.enable_unified_target_observer()
    if branch == "track_gate":
        p.radar.enable_track_gate_model(dict(notch_track_hazard=[], range_gate_m=1000., speed_gate_mps=1000.))
    if branch == "vg":
        p.radar.enable_chaff_capture(dict(p_max=1., v0_mps=10000., delay_s=0.))
    return p


def echo(p, t, transverse=0., decoys=None):
    return p.update(t, .1, missile(), TargetState((1000.+10*t, 1000., 0.), (10., transverse, 0.)), decoys=decoys)


@pytest.mark.parametrize("branch", ["normal", "decoy", "track_gate", "vg"])
def test_full_truth_velocity_cannot_seed_update_or_predict(branch):
    a, b = provider(branch), provider(branch)
    for t in (0., .1, .2):
        # Radial velocity/range are identical. Opposite transverse velocities
        # give equal LOS-rate magnitudes and pass the same detector conditions.
        decoys = [] if branch != "normal" else None
        aa, bb = echo(a, t, 100., decoys), echo(b, t, -100., decoys)
        assert a.last_detection.valid and b.last_detection.valid
        assert aa == bb
        assert a.radar.last_solution == aa
        assert a.radar._last_velocity == aa.velocity
        if branch == "track_gate":
            assert a.radar.track_gate_prediction == b.radar.track_gate_prediction
            assert a.radar.track_gate_prediction[2] == aa.velocity
    assert a.alpha_beta_observer.initial_solution.velocity == (0., 0., 0.)
    assert aa.velocity != (10., 100., 0.)


def test_timestamped_datalink_coasts_without_fake_updates():
    p = provider(datalink=True)
    p.radar.active = False
    packets = [DatalinkObservation((1000., 1000., 0.), (10., 0., 0.), 0.),
               DatalinkObservation((1011., 1000., 0.), (999., 0., 0.), 1.)]
    p.enable_launcher_support(lambda t, truth: ("", packets[int(t >= 1.)]))
    outputs = [echo(p, t) for t in (0., .25, .5, .75)]
    assert [o.position[0] for o in outputs] == [1000., 1002.5, 1005., 1007.5]
    assert p.alpha_beta_observer.update_count == 0
    new = echo(p, 1.)
    assert new.velocity == pytest.approx((10.05, 0., 0.))
    assert p.alpha_beta_observer.update_count == 1
    assert p.alpha_beta_observer.last_dt_s == 1.
    echo(p, 1.1)
    assert p.alpha_beta_observer.update_count == 1


def test_no_external_prior_means_no_datalink_truth_fallback():
    p = provider(datalink=True)
    p.radar.active = False
    p.enable_launcher_support(lambda t, truth: "")
    assert not echo(p, 0.).valid
    assert not echo(p, 1.).valid
    assert p.alpha_beta_observer.solution is None


def test_datalink_roundoff_equal_time_preserves_raw_stamp_and_deduplicates():
    p = provider(datalink=True)
    p.radar.active = False
    packet = DatalinkObservation((1000., 1000., 0.), (10., 0., 0.), 0.)
    p.enable_launcher_support(lambda t, truth: ("", packet))
    echo(p, 0.)
    now = 0.
    for _ in range(6):
        now += 1/48
    raw_stamp = 219.729-219.604
    assert raw_stamp > now
    packet = DatalinkObservation((1002., 1000., 0.), (999., 0., 0.), raw_stamp)
    solution = echo(p, now)
    observer = p.alpha_beta_observer
    assert observer.update_count == 1
    assert observer.solution.solution_time_s == now
    assert observer.solution.sample_time_s == raw_stamp
    assert p.datalink_observation.sample_time_s == raw_stamp
    assert p._last_datalink_sample_s == raw_stamp
    later = echo(p, now+1/48)
    assert observer.update_count == 1
    assert later.position[0] == pytest.approx(solution.position[0]+solution.velocity[0]/48)


@pytest.mark.parametrize('entry', ['callback', 'consume'])
def test_datalink_real_future_is_rejected(entry):
    p = provider(datalink=True)
    packet = DatalinkObservation((1000., 1000., 0.), (10., 0., 0.), .125+1e-9)
    if entry == 'callback':
        p.enable_launcher_support(lambda t, truth: ('', packet))
        with pytest.raises(ValueError, match='in the future'):
            echo(p, .125)
    else:
        p.datalink_observation = packet
        p.datalink_estimate = (packet.position, packet.velocity)
        with pytest.raises(ValueError, match='in the future'):
            p._consume_unified_datalink(.125)


def test_unstamped_legacy_tuple_is_only_an_initial_prior():
    p = provider(datalink=True)
    p.radar.active = False
    p.enable_launcher_support(lambda t, truth: ("", (1000.+100*t, 1000., 0.), (10.+100*t, 0., 0.)))
    echo(p, 0.)
    assert echo(p, 1.).position == (1010., 1000., 0.)
    assert p.alpha_beta_observer.update_count == 0


def test_loss_and_same_object_reacquisition_keep_history():
    p = provider()
    echo(p, 0.)
    first = echo(p, .1)
    observer = p.alpha_beta_observer
    p.radar.receiver_range_max_m = 500.
    coast = echo(p, .2)
    assert coast.position[0] == pytest.approx(first.position[0] + first.velocity[0]*.1)
    p.radar.receiver_range_max_m = 20000.
    reacquired = echo(p, .5, transverse=100.)
    assert p.alpha_beta_observer is observer
    assert observer.last_dt_s == .4
    assert reacquired.velocity[1] == 0.
    assert not any(e['event'] == 'identity_reset_zero_velocity' for e in p.observer_events)


@pytest.mark.parametrize("branch", ["decoy", "track_gate", "vg"])
def test_decoy_transfer_uses_zero_velocity_and_commits_filtered_prediction(branch):
    p = provider(branch)
    echo(p, 0., decoys=[])
    old = p.alpha_beta_observer
    def decoy(t, vy):
        return Reflector('chaff', (950.+10*t, 1000., 0.), (10., vy, 0.), 100., 'bundle',
                         release_identity='release', release_time_s=.1,
                         release_position=(1000., 1000., 0.), release_velocity=(10., 0., 0.))
    a = echo(p, .1, decoys=[decoy(.1, 500.)])
    assert p.radar.last_reflector == 'chaff'
    assert a.velocity == (0., 0., 0.)
    assert p.alpha_beta_observer is not old
    b = echo(p, .2, decoys=[decoy(.2, -500.)])
    assert b.velocity[1] == 0.
    assert p.radar._last_velocity == b.velocity
    if branch == 'track_gate':
        assert p.radar.track_gate_prediction[2] == b.velocity


def test_simultaneous_external_prior_and_echo_keep_prior_velocity():
    p = provider(datalink=True)
    p.enable_launcher_support(lambda t, truth: ('', DatalinkObservation((990., 1000., 0.), (5., 0., 0.), 0.)))
    solution = echo(p, 0., 100.)
    assert solution.position == (998., 1000., 0.)
    assert solution.velocity == (5., 0., 0.)
    assert p.alpha_beta_observer.last_dt_s == 0.


def test_search_center_and_gate_prediction_use_estimate_not_truth_velocity():
    a, b = provider('track_gate', True), provider('track_gate', True)
    for p in (a, b):
        p.enable_launcher_support(lambda t, truth: ('', DatalinkObservation((1000.,1000.,0.),(10.,0.,0.),0.)))
        p.radar.search = SeekerSearch(p.radar, dict(scan_half_angle_deg=0., dwell_s=0.))
    for t in (0., .1):
        assert echo(a,t,100.,[]) == echo(b,t,-100.,[])
        assert a.radar.search.beam == b.radar.search.beam
        assert a.radar.track_gate_prediction == b.radar.track_gate_prediction


def test_existing_radar_only_ab_is_not_silently_replaced():
    p = SensorTrackProvider({})
    p.enable_alpha_beta_observer(.8, .05)
    with pytest.raises(ValueError, match='exclusive'):
        p.enable_unified_target_observer()


@pytest.mark.parametrize('api', ['simulate', 'simulate_surface', 'create_surface_missile'])
def test_public_api_selects_unified_mode(api):
    from aim120_model.public_api import simulate
    from aim120_model.surface_runtime import simulate_surface, create_surface_missile
    from missile_gui.library import scan_library
    root = Path(__file__).resolve().parents[1]
    profiles, errors = scan_library(root/'missiles', root)
    assert not errors
    profile = next(p for p in profiles if p['missile_id'] == 'su_r_77_1')
    support = lambda t, truth: ('', DatalinkObservation((50000.,9000.,2000.),(-100.,0.,0.),0.))
    scene = dict(launch_speed_kmh=1200.,launch_altitude_m=8000.,launch_pitch_deg=0.,launch_heading_deg=0.,
        target_speed_kmh=1000.,target_altitude_m=8000.,initial_distance_m=50000.,target_azimuth_deg=0.,
        target_heading_deg=180.,target_vertical_heading_deg=0.,target_constant_turn_g=0.,
        max_simulation_time_s=.1,observation_mode='sensor_track',loft_enabled=True)
    options = dict(target_observer='unified_alpha_beta',launcher_support=support)
    if api == 'create_surface_missile':
        class Target:
            def state_at(self,t):return TargetState((50000.,8000.,0.),(-300.,0.,0.))
        runtime = create_surface_missile(profile,launch_position_m=(0.,8000.,0.),launch_velocity_mps=(300.,0.,0.),
            launch_pitch_deg=0.,launch_heading_deg=0.,target=Target(),end_time_s=.1,**options)
        assert runtime.provider.unified_target_observer
        assert runtime.track.velocity == (-100.,0.,0.)
    else:
        result = (simulate if api == 'simulate' else simulate_surface)(profile,scene,**options)
        assert result['model']['target_observer']['mode'] == 'unified_alpha_beta'
        assert result['samples'][0]['observed_target_velocity_mps'] == (-100.,0.,0.)
