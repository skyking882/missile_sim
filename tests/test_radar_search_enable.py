"""Causal INS search enable; receiver range and launch startup are separate."""
import copy

import pytest

from aim120_model.dynamics import SimState
from aim120_model.observation import SensorTrackProvider
from aim120_model.radar_seeker import RadarSeekerObserver, SeekerSearch
from aim120_model.target import TargetState
from aim120_model.tracking import TrackMode


def model():
    antenna=dict(angle_half_sens_deg=15.,side_lobes_sensitivity_db=-30.)
    return dict(active_radar=True,datalink=True,inertial_navigation=True,
        lock_timeout_s=.75,inertial_drift_speed_mps=0.,break_lock_max_time_s=100.,
        radar_seeker=dict(active=True,lock_angle_max_deg=55.,angle_max_deg=55.,
            rate_max_deg_s=60.,angle_gate_rate_deg_s=30.,prolongation_time_max_s=1.,
            receiver=dict(range_m=16000.,range_max_m=25000.,rcs_m2=1.,antenna=antenna),
            transmitter=dict(antenna=antenna),
            doppler_speed=dict(min_mps=-3000.,max_mps=3000.,width_mps=20.,ref_width_mps=80.)))


def own(x=0.):return SimState((x,1000.,0.),(300.,0.,0.),0.,0.,0.,0.,100.)
def target(x=10000.,y=1000.,vx=-300.):return TargetState((x,y,0.),(vx,0.,0.))


@pytest.mark.parametrize('branch',['ordinary','empty_decoys','search','track_gate','vg'])
def test_causal_range_blocks_every_path_without_consuming_history(branch):
    radar=RadarSeekerObserver(model(),lock_range_m=16000.);kwargs={}
    if branch=='empty_decoys':kwargs['decoys']=[]
    elif branch=='search':radar.search=SeekerSearch(radar,dict(scan_half_angle_deg=0.))
    elif branch=='track_gate':radar.enable_track_gate_model(dict(notch_track_hazard=[]))
    elif branch=='vg':radar.enable_chaff_capture(dict(p_max=1.,v0_mps=10000.))
    radar.search_estimate_position=(16100.,1000.,0.);radar.require_search_estimate=True
    if radar.search is not None:radar.search.estimate=radar.search_estimate_position
    assert radar.update(.5,.02,own(),target(),**kwargs)[0].reason=='startup_delay'
    for t in (.75,.9):
        detection,solution=radar.update(t,.02,own(),target(),**kwargs)
        assert detection.reason=='INS_range_search_disabled'and solution is None
        assert radar._last_detection_time_s is None and not radar.distance_gate.initialized
        assert not radar.doppler_gate.initialized and radar._last_measurement_time_s is None
        if radar.search is not None:assert radar.search.search_start_s is None
        if branch=='vg':assert radar.capture_previous is None and not radar.capture_trials
        if branch=='track_gate':assert radar.track_gate_last_time is None
    # Current own position matters: the same target estimate is now at16km.
    detection,solution=radar.update(1.,.02,own(100.),target(),**kwargs)
    assert detection.valid and solution.mode is TrackMode.RADAR_TRACK
    assert radar._last_detection_time_s==1.
    if radar.search is not None:assert radar.search.search_start_s==1.


def test_estimate_enables_search_but_true_receiver_range_still_limits_echo():
    radar=RadarSeekerObserver(model(),lock_range_m=16000.)
    radar.search_estimate_position=(15900.,1000.,0.)
    # True target outside16km is observable after estimated-distance enable.
    assert radar.update(.75,.02,own(),target(17000.), target_rcs_m2=2.)[0].valid
    other=RadarSeekerObserver(model(),lock_range_m=16000.)
    other.search_estimate_position=(15900.,1000.,0.)
    assert other.update(.75,.02,own(),target(26000.))[0].reason=='range_gate'
    legacy=RadarSeekerObserver(model(),lock_range_m=16000.)
    assert legacy.update(.75,.02,own(),target(17000.))[0].reason=='range_gate'
    assert legacy.update(.75,.02,own(),target())[0].valid


def test_hold_does_not_reapply_enable_but_expired_reacquisition_does():
    radar=RadarSeekerObserver(model(),lock_range_m=16000.)
    radar.search_estimate_position=(15900.,1000.,0.)
    assert radar.update(.75,.02,own(),target())[0].valid
    radar.search_estimate_position=(20000.,1000.,0.)
    assert radar.update(1.,.25,own(),target())[0].valid
    missing=target(y=900.,vx=0.)
    assert radar.update(1.5,.5,own(),missing)[1].mode is TrackMode.RADAR_COAST
    assert radar.update(2.,.5,own(),missing)[1].mode is TrackMode.RADAR_COAST
    assert radar._last_detection_time_s==1.
    assert radar.update(2.000001,.02,own(),target())[0].reason=='INS_range_search_disabled'
    assert radar._last_detection_time_s==1.
    radar.search_estimate_position=(15900.,1000.,0.)
    # No new.75s timer starts at the range crossing or reacquisition.
    assert radar.update(2.01,.02,own(),target())[0].valid


@pytest.mark.parametrize('filter_kind',['none','radar','unified'])
def test_provider_uses_external_DL_then_one_causal_INS_propagation(filter_kind):
    p=SensorTrackProvider(model(),lock_range_m=16000.,datalink_disconnect_time_s=.8)
    p.enable_launcher_support(lambda t,truth:('',(16200.-200.*t,1000.,0.),(-200.,0.,0.)))
    if filter_kind=='radar':p.enable_alpha_beta_observer(.8,.05)
    elif filter_kind=='unified':p.enable_unified_target_observer()
    p.update(0.,0.,own(),target())
    assert p.update(.75,.75,own(),target()).mode is TrackMode.DATALINK
    assert p.last_detection.reason=='INS_range_search_disabled'
    assert p.radar.search_estimate_position==(16050.,1000.,0.)
    calls=[];update=p.inertial.update
    def counted(*args):calls.append(args);return update(*args)
    p.inertial.update=counted
    before=copy.deepcopy(p.alpha_beta_observer)
    p.update(.9,.15,own(),target())
    assert p.last_detection.reason=='INS_range_search_disabled'
    assert p.radar.search_estimate_position==pytest.approx((16020.,1000.,0.))
    if filter_kind!='unified':assert len(calls)==1
    if filter_kind=='radar':assert p.alpha_beta_observer.update_count==before.update_count==0
    if filter_kind=='unified':assert p.alpha_beta_observer.update_count==before.update_count
    assert p.radar._last_detection_time_s is None
    accepted=p.update(1.,.1,own(),target())
    assert accepted.mode is TrackMode.RADAR_TRACK and p.radar._last_detection_time_s==1.
    assert p.radar.search_estimate_position==pytest.approx((16000.,1000.,0.))


def test_explicit_provider_without_available_prior_cannot_substitute_truth():
    p=SensorTrackProvider(model(),lock_range_m=16000.,datalink_enabled=False)
    p.enable_launcher_support(lambda t,truth:'not available')
    assert not p.update(.75,.02,own(),target()).valid
    assert p.last_detection.reason=='search_estimate_unavailable'
    assert p.radar._last_detection_time_s is None


def test_ordinary_coast_preserves_INS_memory_until_search_resumes():
    p=SensorTrackProvider(model(),lock_range_m=16000.)
    p.enable_launcher_support(lambda t,truth:('',(10000.,1000.,0.),(-300.,0.,0.)))
    assert p.update(.75,.02,own(),target()).mode is TrackMode.RADAR_TRACK
    memory=copy.deepcopy(p.inertial.__dict__)
    for t in (1.,1.5,1.75):
        assert p.update(t,.25,own(),target(26000.)).mode is TrackMode.RADAR_COAST
        assert p.inertial.__dict__==memory
    assert p.update(1.750001,.000001,own(),target(26000.)).mode is TrackMode.INS_SEARCH
    assert p.inertial._last_update_time_s==1.750001
    assert p.radar.search_estimate_position==pytest.approx((9699.9997,1000.,0.))


def test_search_pointing_and_enable_use_the_same_available_estimate():
    p=SensorTrackProvider(model(),lock_range_m=16000.)
    p.enable_launcher_support(lambda t,truth:('',(0.,1000.,15000.),(0.,0.,0.)))
    # The estimate is in range but points90deg away from the true target.
    p.update(.75,.02,own(),target())
    assert p.last_detection.reason=='beam_range_gate'
    assert p.radar.search_estimate_position==(0.,1000.,15000.)
    assert p.radar._last_detection_time_s is None
