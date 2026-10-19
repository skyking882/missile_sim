"""Capture event timing and expired angular publication contracts."""
import copy
import inspect
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from aim120_model.capture_retry import RetryClock
from aim120_model.radar_seeker import RadarSeekerObserver, SeekerSearch
from aim120_model.observation import InertialTrackPropagator, SensorTrackProvider
from aim120_model.tracking import TrackMode, TrackSolution
from aim120_model.target import TargetState
from aim120_model.public_api import simulate, create_surface_missile
from aim120_model.surface_runtime import simulate_surface
from test_observation_modes import _model, _missile

H=1/48


def test_immediate_first_then_global_phase_and_duplicate():
    calls=[]
    c=RetryClock()
    def fail(searching):
        calls.append(searching)
        return False
    assert c.step(100,enabled=True,search_ready=True,evaluate=fail).attempted
    assert c.step(100,enabled=True,search_ready=True,evaluate=fail).attempted
    assert len(calls)==1
    assert not c.step(101,enabled=True,search_ready=True,evaluate=fail).attempted
    assert c.step(145,enabled=True,search_ready=True,evaluate=fail).attempted
    assert calls==[True,True]


def test_disable_and_range_exit_preserve_failure_skip_missed_slot():
    c=RetryClock();calls=[]
    f=lambda searching:calls.append(searching) or False
    c.step(10,enabled=True,search_ready=True,evaluate=f)
    c.step(49,enabled=False,search_ready=True,evaluate=f)
    c.step(50,enabled=True,search_ready=False,evaluate=f)
    assert not c.step(51,enabled=True,search_ready=True,evaluate=f).attempted
    assert c.step(97,enabled=True,search_ready=True,evaluate=f).attempted
    assert len(calls)==2


def test_hold_recovery_not_delayed_and_expiry_returns_retry():
    c=RetryClock();c.step(100,enabled=True,search_ready=True,evaluate=lambda _:True)
    assert c.step(101,enabled=True,search_ready=False,evaluate=lambda searching:not searching).accepted
    assert c.step(149,enabled=True,search_ready=False,evaluate=lambda searching:False).state=='HOLD'
    assert c.step(150,enabled=True,search_ready=True,evaluate=lambda searching:True).attempted is False
    assert c.step(193,enabled=True,search_ready=True,evaluate=lambda searching:searching).accepted


def test_global_launch_epochs_have_distinct_relative_retry_ages():
    for epoch,expected in ((11415,58),(14630,59)):
        c=RetryClock();c.step(epoch+36,enabled=True,search_ready=True,evaluate=lambda _:False)
        assert c.next_search_tick-epoch==expected


def test_beam_single_leg_half_power_and_no_hard_edge():
    cfg=_model(True)
    for leg in ('receiver','transmitter'):
        cfg['radar_seeker'].setdefault(leg,{})['antenna']={
            'angle_half_sens_deg':15.,'side_lobes_sensitivity_db':-30.}
    o=RadarSeekerObserver(cfg)
    theta=math.radians(15.)
    assert o.beam_half_rad==pytest.approx(theta)
    assert o.illumination_half_rad==pytest.approx(theta)
    assert SeekerSearch.gain(theta,o.receiver_antenna)==pytest.approx(.5)
    assert o.return_gain((math.cos(theta),math.sin(theta),0.),(1.,0.,0.))==pytest.approx(.25)
    assert o.return_gain((math.cos(2*theta),math.sin(2*theta),0.),(1.,0.,0.))>0
    assert o.lock_angle_max_rad==pytest.approx(math.radians(55.))


def test_real_radar_common_entry_schedules_and_does_not_reaccept_duplicate(monkeypatch):
    cfg=_model(True);cfg['lock_timeout_s']=.75
    o=RadarSeekerObserver(cfg,launch_global_tick=11415)
    calls=[]
    def evaluate(t,*args):
        calls.append(t)
        return o._detection(False,'beam_range_gate',10000.,0.,0.,0.,0.,False),None
    monkeypatch.setattr(o,'_update',evaluate)
    for t in (0.,.75,.75,.8,58*H):
        d,_=o.update(t,H,_missile(),TargetState((10000.,1000.,0.),(0.,)*3),decoys=[])
        if t==.8:assert not d.attempted and d.reason=='await_global_phase'
    assert calls==[.75,58*H]
    assert o.capture_retry.attempt_count==2
    o.reset();assert not o.capture_retry.first_attempted and o.launch_global_tick==11415


def test_expired_trk_rate_is_memory_only_but_external_ins_rate_is_preserved():
    full=(.02,-.03,.04)
    source=TrackSolution((10000.,1000.,0.),(50.,0.,100.),0.,0.,TrackMode.RADAR_TRACK,True,'radar',full)
    ins=InertialTrackPropagator();ins.reset(source)
    out=ins.update(2.,H)
    assert out.full_rate_rad_s is None
    assert ins.estimate is source and ins.estimate.full_rate_rad_s==full
    ins.reset(replace(source,mode=TrackMode.DATALINK,source='external'))
    assert ins.update(3.,H).full_rate_rad_s==full


def test_provider_keeps_trk_hold_then_clears_rate_after_timeout():
    cfg=_model(True);cfg['inertial_drift_speed_mps']=0.
    p=SensorTrackProvider(cfg,datalink_enabled=False,launch_global_tick=100)
    p.enable_q_angular_observer()
    own=_missile();truth=TargetState((10000.,1000.,100.),(50.,0.,100.))
    first=p.update(0.,H,own,truth)
    assert first.mode is TrackMode.RADAR_TRACK and first.full_rate_rad_s is not None
    miss=TargetState((1e9,1000.,0.),(0.,)*3)
    held=p.update(H,H,own,miss)
    assert held.mode is TrackMode.RADAR_COAST and held.full_rate_rad_s==first.full_rate_rad_s
    expired=p.update(1.+H,H,own,miss)
    assert expired.mode is TrackMode.INS_SEARCH and expired.full_rate_rad_s is None
    assert p.inertial.estimate.full_rate_rad_s==first.full_rate_rad_s


def test_global_epoch_reaches_runtime_and_public_signatures():
    for f in (simulate,simulate_surface,create_surface_missile):
        assert inspect.signature(f).parameters['launch_global_tick'].default==0
    profile=json.loads((Path(__file__).resolve().parents[1]/'data/profile_sets/2.59.0.27/missiles/su_r_77_1.json').read_text())
    class Target:
        def state_at(self,t):return TargetState((10000.,8000.,0.),(0.,0.,0.))
    rt=create_surface_missile(profile,launch_position_m=(0.,8000.,0.),launch_velocity_mps=(300.,0.,0.),launch_pitch_deg=0.,launch_heading_deg=0.,target=Target(),end_time_s=0.,launch_global_tick=14630)
    assert rt.provider.radar.launch_global_tick==14630
    assert rt.config['capture_retry']['launch_global_tick']==14630
