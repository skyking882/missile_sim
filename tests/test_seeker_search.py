"""Opt-in search geometry, state transitions and the entire active catalog."""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path
import pytest
from aim120_model.chaff import Reflector,ChaffingTarget,ChaffProgram,ChaffSpec
from aim120_model.dynamics import SimState
from aim120_model.observation import SensorTrackProvider
from aim120_model.public_api import simulate,create_surface_missile,validate_scenario,_case_from_scenario,SimulationInputError,UnsupportedPhysicsError
from aim120_model.radar_seeker import RadarSeekerObserver,SeekerSearch,validate_seeker_search
from aim120_model.surface_runtime import simulate_surface,SurfaceRuntime
from aim120_model.profile_catalog import load_profile_catalog
from aim120_model.target import TargetState
from aim120_model.tracking import TrackMode,TrackSolution
from missile_gui.library import scan_library,public_profile
ROOT=Path(__file__).resolve().parents[1]

def missile():return SimState((0.,1000.,0.),(300.,0.,0.),0.,0.,0.,0.,100.)
def target(distance=1000.,z=0.,vx=-100.):return TargetState((distance,1000.,z),(vx,0.,0.))
def sensor(active=True,dwell=.2):
 return dict(active_radar=active,inertial_navigation=True,datalink=True,lock_timeout_s=0.,
  inertial_drift_speed_mps=2.,radar_seeker=dict(active=active,prolongation_time_max_s=.2,
   lock_angle_max_deg=55.,angle_max_deg=55.,rate_max_deg_s=60.,angle_gate_rate_deg_s=60.,
   receiver=dict(rcs_m2=2.,range_m=16000.,range_max_m=25000.,antenna=dict(angle_half_sens_deg=15.,side_lobes_sensitivity_db=-30.)),
   transmitter=dict(antenna=dict(angle_half_sens_deg=15.,side_lobes_sensitivity_db=-30.))))
def observer(options=None,model=None):
 o=RadarSeekerObserver(model or sensor(),lock_range_m=20000.)
 o.search=SeekerSearch(o,{'scan_half_angle_deg':0.,'dwell_s':.2,**(options or {})})
 return o

def test_gain_half_power_sidelobes_rcs_and_distinct_tx_rx():
 o=observer({'dwell_s':0.}); s=o.search
 angle=math.radians(7.5);s.beam=(1.,0.,0.);los=(math.cos(angle),0.,math.sin(angle))
 assert s.return_gain(los)==pytest.approx(.25)
 assert s.detection_range(los,2.)==pytest.approx(16000.*.25**.25)
 assert s.detection_range((1.,0.,0.),32.)==pytest.approx(32000.)
 assert s.return_gain((0.,0.,1.))==pytest.approx(1e-6)
 m=sensor();m['radar_seeker']['transmitter']['antenna']['angle_half_sens_deg']=6.
 m['radar_seeker']['receiver']['antenna']['angle_half_sens_deg']=9.
 other=observer(model=m)
 assert other.search.tx[0]!=other.search.rx[0]
 assert other.search.return_gain(los)<s.return_gain(los)

@pytest.mark.parametrize('pattern',['spiral','raster'])
def test_scan_phase_and_body_axis_bounds(pattern):
 o=observer({'scan_half_angle_deg':70.,'pattern':pattern});s=o.search
 s.estimate=(0.,1000.,1000.)
 for i in range(101):
  s.prepare(i*.015,missile(),target(),True)
  angle=math.degrees(math.acos(s.beam[0]))
  from aim120_model.aerodynamics import body_axes_for_state
  from aim120_model.math3d import dot
  axes=body_axes_for_state(missile())
  x,y,z=(dot(s.beam,a) for a in (axes.forward,axes.up,axes.right))
  assert abs(math.degrees(math.atan2(z,x)))<=55.+1e-10
  assert abs(math.degrees(math.atan2(y,math.hypot(x,z))))<=55.+1e-10
  if i==0:assert angle==pytest.approx(55.) and s.scan_offset_deg==pytest.approx(0.)
 assert math.degrees(s.half_rad)==pytest.approx(55.)


def test_dwell_resets_on_miss_and_prolongation_boundary():
 o=observer();m=missile();t=target()
 assert not o.update(0.,.1,m,t)[0].valid
 assert not o.distance_gate.initialized
 bad=target(z=100000.)
 assert not o.update(.1,.1,m,bad)[0].valid
 assert not o.update(.2,.1,m,t)[0].valid
 assert not o.update(.3,.1,m,t)[0].valid
 assert o.update(.4,.1,m,t)[0].valid
 assert o.search.first_lock_time_s==.4
 assert not o.update(.5,.1,m,bad)[0].valid
 # A short gap uses track gates and no new dwell.
 assert o.update(.6,.1,m,t)[0].valid
 assert o.search.lock_loss_count==0
 assert not o.update(.81,.21,m,bad)[0].valid
 assert o.search.lock_loss_count==1
 assert not o.update(.9,.09,m,t)[0].valid
 assert o.update(1.1,.2,m,t)[0].valid
 assert o.search.reacquire_times_s==[1.1]
 assert o.search.reacquire_delays_s==pytest.approx([.29])
 o.reset()
 assert o.search.first_lock_time_s is None and o.search.lock_loss_count==0


def test_identity_dwell_cannot_mix_target_or_chaff_salvos():
 o=observer();m=missile();t=target()
 assert not o.update(0.,.1,m,t,decoys=[])[0].valid
 # Stronger co-located chaff wins; it starts an independent dwell.
 def chaff(identity):return Reflector('chaff',t.position,t.velocity,100.,identity)
 assert not o.update(.1,.1,m,t,decoys=[chaff('salvo1')])[0].valid
 assert not o.update(.2,.1,m,t,decoys=[chaff('salvo2')])[0].valid
 assert not o.update(.3,.1,m,t,decoys=[chaff('salvo2')])[0].valid
 assert o.update(.4,.1,m,t,decoys=[chaff('salvo2')])[0].valid
 assert o.last_reflector=='chaff'
 assert Reflector('chaff',t.position,t.velocity,1.).identity is None
 missing=observer()
 for time in [0.,.2,.4]:
  detection,_=missing.update(time,.2,m,t,decoys=[chaff(None)])
  assert detection.reason=='acquisition_identity_missing'


def test_rebuilt_chaff_has_stable_unique_identity():
 class Static:
  def state_at(self,t):return target()
 wrapper=ChaffingTarget(Static(),ChaffSpec(10.),ChaffProgram(start_s=0.,interval_s=.1,per_salvo=2,total=4))
 first=wrapper.decoys_at(.06);second=wrapper.decoys_at(.07)
 assert len(first)==2
 assert [x.identity for x in first]==[x.identity for x in second]
 assert len({x.identity for x in wrapper.decoys_at(.16)})==4


def test_current_datalink_and_ins_pointing_without_truth_leak_or_double_drift():
 p=SensorTrackProvider(sensor(),lock_range_m=20000.)
 p.radar.search=SeekerSearch(p.radar,{'scan_half_angle_deg':0.,'dwell_s':100.})
 t=target(z=300.)
 p.enable_launcher_support(lambda time,truth:('',t.position,t.velocity))
 p.update(0.,0.,missile(),t)
 assert p.radar.search.estimate==t.position
 p.datalink_connected=False
 false_truth=target(z=50000.)
 p.update(.5,.5,missile(),false_truth)
 assert p.radar.search.estimate==(950.,1000.,301.)
 assert p.inertial._last_update_time_s==.5
 assert p.last_solution.position==(950.,1000.,301.)

@pytest.mark.parametrize('bad',[False,[],{'frame_time_s':0},{'dwell_s':None},{'dwell_s':True},{'pattern':'bad'},
 {'scan_half_angle_deg':math.inf},{'dwell_s':-1},{'extra':1}])
def test_option_validation(bad):
 with pytest.raises(ValueError):validate_seeker_search(bad)

def scenario():
 return dict(launch_speed_kmh=1200.,launch_altitude_m=8000.,launch_pitch_deg=0.,launch_heading_deg=0.,
 target_speed_kmh=1000.,target_altitude_m=8000.,initial_distance_m=2000.,target_azimuth_deg=0.,target_heading_deg=0.,
 target_course_reference='relative_to_los',target_vertical_heading_deg=0.,target_constant_turn_g=0.,
 max_simulation_time_s=.95,observation_mode='sensor_track',loft_enabled=False)

_,CATALOG=load_profile_catalog(ROOT)
PROFILES,ERRORS=scan_library(CATALOG['profiles_dir'],ROOT)
assert not ERRORS
RUNNABLE=[p for p in PROFILES if public_profile(p)['runnable']]
assert len(RUNNABLE)==118

@pytest.mark.parametrize('profile',RUNNABLE,ids=lambda p:p['missile_id'])
def test_all_runnable_three_entries_and_supported_radar_acquisition(profile):
 sc=scenario();norm=validate_scenario(sc);defaults=json.loads((ROOT/'config/effective_surface_defaults.json').read_text())
 ref=SurfaceRuntime(profile,norm,_case_from_scenario(norm),defaults)
 options=dict(seeker_search={"scan_half_angle_deg":0.})
 active=bool(getattr(getattr(ref.provider,'radar',None),'active',False))
 a=simulate(profile,sc,**options);b=simulate_surface(profile,sc,**options)
 assert a==b
 runtime=create_surface_missile(profile,launch_position_m=ref.creation['position_xyz_m'],launch_velocity_mps=ref.creation['velocity_xyz_mps'],
  launch_pitch_deg=0.,launch_heading_deg=0.,target=ref.target,loft=False,end_time_s=.95,**options)
 while not runtime.done:runtime.step()
 result=runtime.result()
 assert a['samples']==result['samples']
 assert a['summary']['first_lock_time_s']==result['summary']['first_lock_time_s']
 if active:
  assert a['summary']['first_lock_time_s'] is not None
 else:
  assert a['model']['seeker_search']['status']=='not_applicable'
  assert all(row['seeker_beam_target_deg'] is None for row in a['samples'])
 # Omitted and explicit None keep the old output schema and values.
 old=simulate(profile,sc)
 assert old==simulate(profile,sc,seeker_search=None)
 assert 'seeker_search' not in old['model']

@pytest.mark.parametrize('profile',[p for p in PROFILES if not public_profile(p)['runnable']],ids=lambda p:p['missile_id'])
def test_existing_unsupported_stays_rejected(profile):
 with pytest.raises(UnsupportedPhysicsError):simulate(profile,scenario(),seeker_search={})
 with pytest.raises(ValueError):simulate_surface(profile,scenario(),seeker_search={})
 with pytest.raises(ValueError):create_surface_missile(profile,launch_position_m=(0.,8000.,0.),launch_velocity_mps=(300.,0.,0.),
  launch_pitch_deg=0.,launch_heading_deg=0.,target=object(),seeker_search={})

@pytest.mark.parametrize('profile',[p for p in RUNNABLE if ((p['guidance'].get('sensor_model') or {}).get('radar_seeker') or {}).get('active') is True],ids=lambda p:p['missile_id'])
def test_every_active_profile_reacquires_with_own_parameters(profile):
 o=RadarSeekerObserver(profile['guidance']['sensor_model'],lock_range_m=profile['guidance']['lock_range_m'])
 o.search=SeekerSearch(o,{"scan_half_angle_deg":0.})
 m=missile();t=target(distance=500.)
 for i in range(37):o.update(i/48.,1/48.,m,t)
 assert o.search.first_lock_time_s==pytest.approx(.75)
 assert o.startup_delay_s==profile['guidance']['sensor_model']['lock_timeout_s']
 assert o.search.dwell_s==0.
 o.update(1.,.25,m,TargetState((500.,999.,0.),(0.,0.,0.)))
 o.update(1.76,.76,m,TargetState((500.,999.,0.),(0.,0.,0.)))
 assert o.search.lock_loss_count==1
 for i in range(37):detection,_=o.update(2.+i/48.,1/48.,m,t)
 assert detection.valid
 assert o.search.reacquire_times_s==pytest.approx([2.])


def test_rcs_reaches_nondecoy_path_and_empty_decoys_share_acquisition():
 profile=next(p for p in RUNNABLE if p['missile_id']=='su_r_77_1')
 class Target:
  rcs_m2=16.
  def __init__(self,base):self.base=base
  def state_at(self,t):return self.base.state_at(t)
 class Empty(Target):
  def decoys_at(self,t):return []
 sc=scenario();sc.update(initial_distance_m=14000.,max_simulation_time_s=.77)
 options=dict(seeker_search={'scan_half_angle_deg':0.,'dwell_s':0.})
 no=simulate(profile,sc,**options)
 high=simulate(profile,sc,target_factory=Target,**options)
 empty=simulate(profile,sc,target_factory=Empty,**options)
 assert no['summary']['first_lock_time_s'] is None
 assert high['summary']['first_lock_time_s']==pytest.approx(.75)
 assert high['samples']==empty['samples']
 sources=high['model']['seeker_search']['parameter_sources']
 assert all(item['kind']=='assumed' and item['grade']=='D' for item in sources.values())


def test_within_tick_fuse_requires_completed_search_lock():
 profile=next(p for p in RUNNABLE if p['missile_id']=='su_r_77_1')
 class Near:
  def state_at(self,t):return TargetState((2.,8000.,0.),(0.,0.,0.))
 options=dict(launch_position_m=(0.,8000.,0.),launch_velocity_mps=(333.,0.,0.),
  launch_pitch_deg=0.,launch_heading_deg=0.,target=Near(),loft=False,end_time_s=.107,require_seeker_lock=True)
 pending=create_surface_missile(profile,**options,seeker_search={'dwell_s':100.})
 while not pending.done:pending.step()
 assert pending.result()['event_type']=='lifetime'
 assert not pending.provider.radar_has_tracked_once
 assert pending.result()['summary']['first_lock_time_s'] is None
 # With the real startup, zero dwell still cannot fuse immediately.
 delayed=create_surface_missile(profile,**options,seeker_search={'dwell_s':0.})
 assert not delayed.done and not delayed.provider.radar_has_tracked_once
 # Isolate the existing immediate-fuse contract using a zero-startup fixture.
 immediate=copy.deepcopy(profile);immediate['guidance']['sensor_model']['lock_timeout_s']=0.
 acquired=create_surface_missile(immediate,**options,seeker_search={'dwell_s':0.})
 assert acquired.result()['event_type']=='fuse'
 assert acquired.provider.radar_has_tracked_once


@pytest.mark.parametrize('switch_candidate',[False,True])
def test_cw_clear_beam_preserves_stable_chaff_dwell(switch_candidate):
 o=observer();o.cw_on_clear_beam=True;o.clutter_model='look_down'
 m=missile();t=target(vx=0.)
 # Rebuild the reflector each tick; clear-beam CW must retain its identity.
 times=[0.,.1,.2,.3] if switch_candidate else [0.,.1,.2]
 for index,time in enumerate(times):
  identity='salvo1' if not switch_candidate or index==0 else 'salvo2'
  decoy=Reflector('chaff',t.position,t.velocity,100.,identity)
  detection,solution=o.update(time,.1,m,t,decoys=[decoy])
  if index<len(times)-1:
   assert not detection.valid and detection.reason=='acquisition_dwell'
   assert o.search.candidate==('decoy',identity)
  else:
   assert detection.valid and solution.valid
   assert o.last_reflector=='chaff'
 assert o.cw_ticks==len(times)
 assert o.search.first_lock_time_s==times[-1]
