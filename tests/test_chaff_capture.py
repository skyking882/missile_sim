"""Replay-calibrated opt-in capture: release units, timing, gates and API parity."""
import copy
import json
from pathlib import Path
import pytest
from aim120_model.chaff import Reflector, ChaffingTarget, ChaffSpec, ChaffProgram, chaffing_factory
from aim120_model.dynamics import SimState
from aim120_model.target import TargetState
from aim120_model.radar_seeker import RadarSeekerObserver, SeekerSearch, validate_chaff_capture
from aim120_model.public_api import simulate, create_surface_missile, SimulationInputError
from aim120_model.surface_runtime import simulate_surface
from missile_gui.library import scan_library
ROOT=Path(__file__).resolve().parents[1]

def missile():return SimState((0.,1000.,0.),(300.,0.,0.),0.,0.,0.,0.,100.)
def target(vg=100.,down=False):return TargetState((1000.,900. if down else 1000.,0.),(vg,0.,0.))
def observer(**options):
 s=dict(active_radar=True,radar_seeker=dict(active=True,prolongation_time_max_s=1.,
  receiver=dict(range_max_m=20000.,range_m=16000.,rcs_m2=1.,antenna=dict(angle_half_sens_deg=15.,side_lobes_sensitivity_db=-30.)),
  transmitter=dict(antenna=dict(angle_half_sens_deg=15.,side_lobes_sensitivity_db=-30.)),
  doppler_speed=dict(width_mps=20.,ref_width_mps=80.), distance=dict(width_m=300.)))
 o=RadarSeekerObserver(s,lock_range_m=20000.);o.enable_chaff_capture({'p_max':1.,'v0_mps':10000.,**options});return o

def decoy(t=.1,identity='a',release='r',position=(950.,900.,0.),vg=100.,rcs=1.):
 return Reflector('chaff',position,(0.,0.,0.),rcs,identity,release,t,(1000.,900.,0.),(vg,0.,0.))

def test_one_release_delay_vg_at_release_and_bundle_order():
 o=observer();m=missile();t=target(down=True)
 assert o.update(0.,.1,m,t,[])[0].valid
 bundles=[decoy(identity='b'),decoy(identity='a',position=(1050.,900.,0.),rcs=.001)]
 o.update(.1,.1,m,t,bundles)
 trial=next(iter(o.capture_trials.values()));vg=trial['vg_mps']
 for age in (.2,.5,1.39):o.update(age,.1,m,target(300.,True),list(reversed(bundles)))
 assert len(o.capture_trials)==1 and trial['vg_mps']==vg and o.last_reflector=='target'
 assert o.update(1.4,.01,m,target(300.,True),bundles)[0].valid
 assert o.last_reflector=='chaff' and o.last_solution.position==(1050.,900.,0.)
 assert trial['captured_time_s']==1.4 # alphabetic identity, despite weaker/farther return
 for age in (1.5,1.6):o.update(age,.1,m,t,bundles)
 assert len(o.capture_trials)==1


def test_target_notch_stays_and_chaff_bypasses_notch_and_hard_doppler():
 o=observer();m=missile()
 assert not o.update(0.,.1,m,target(0.,True),[])[0].valid
 assert o.last_detection.reason=='ground_clutter_notch'
 assert o.update(.05,.05,m,target(100.,True),[])[0].valid
 d=decoy(vg=100.)
 o.update(.1,.05,m,target(100.,True),[d])
 assert not o.update(1.3,1.2,m,target(0.,True),[d])[0].valid
 assert o.update(1.4,.1,m,target(0.,True),[d])[0].valid
 assert o.last_reflector=='chaff'


def test_geometry_blocks_actual_capture_without_redraw():
 o=observer();m=missile();t=target()
 o.update(0.,.1,m,t,[])
 d=decoy(position=(400.,1000.,0.));o.update(.1,.1,m,t,[d]);o.update(1.4,1.3,m,t,[d])
 trial=next(iter(o.capture_trials.values()))
 assert trial['selected'] and trial['captured_time_s'] is None and o.last_reflector=='target'
 old=trial['draw'];o.update(1.5,.1,m,t,[d]);assert trial['draw']==old


def test_unlocked_release_never_gets_later_trial_and_missing_identity_surfaces():
 o=observer();m=missile();t=target()
 d=decoy(t=0.);o.update(0.,.1,m,t,[d]);o.update(1.4,1.4,m,t,[d])
 assert not next(iter(o.capture_trials.values()))['eligible'] and o.last_reflector=='target'
 with pytest.raises(ValueError,match='identity'):o.update(1.5,.1,m,t,[Reflector('chaff',t.position,t.velocity,1.)])


def test_seed_snapshot_repeat_and_expiration_reacquisition():
 o=observer(p_max=.5);m=missile();t=target();o.update(0.,.1,m,t,[])
 d=decoy();o.update(.1,.1,m,t,[d]);restored=copy.deepcopy(o)
 for x in (o,restored):x.update(1.4,1.3,m,t,[d])
 assert o.capture_summary()==restored.capture_summary() and o.last_solution==restored.last_solution
 repeat=observer(p_max=.5);repeat.update(0.,.1,m,t,[]);repeat.update(.1,.1,m,t,[d])
 assert next(iter(o.capture_trials.values()))['draw']==next(iter(repeat.capture_trials.values()))['draw']
 yes=observer();yes.update(0.,.1,m,t,[]);yes.update(.1,.1,m,t,[d]);yes.update(1.4,1.3,m,t,[d])
 yes.update(1.5,.1,m,t,[]);assert yes.last_reflector=='target'


def test_search_dwell_and_no_random_search_capture():
 o=observer();o.search=SeekerSearch(o,dict(scan_half_angle_deg=0.,dwell_s=.2))
 m=missile();t=target()
 assert not o.update(0.,.1,m,t,[])[0].valid
 assert o.update(.2,.2,m,t,[])[0].valid
 d=decoy(t=.3);o.update(.3,.1,m,t,[d]);o.update(1.6,1.3,m,t,[d])
 later=decoy(t=1.7,identity='later',release='later')
 o.update(1.7,.1,m,t,[d,later])
 assert not next(t for t in o.capture_trials.values() if t['release_identity']=='later')['eligible']
 o.update(1.8,.1,m,t,[d,later])
 assert o.last_reflector=='chaff' and o.search.first_lock_time_s==.2

@pytest.mark.parametrize('bad',[{'model':'bad'},{'seed':True},{'p_max':1.01},{'scale_mps':0},{'delay_s':-1},{'x':2}])
def test_invalid_options(bad):
 with pytest.raises(ValueError):validate_chaff_capture(bad)


def test_program_salvo_release_identity_and_api_results():
 p=next(p for p in scan_library(ROOT/'missiles',ROOT)[0] if p['missile_id']=='cn_pl12')
 scene=dict(launch_speed_kmh=1000.,launch_altitude_m=1000.,launch_pitch_deg=0.,launch_heading_deg=0.,target_speed_kmh=300.,target_altitude_m=1000.,
  initial_distance_m=7000.,target_azimuth_deg=0.,target_heading_deg=180.,target_vertical_heading_deg=0.,
  target_constant_turn_g=0.,max_simulation_time_s=2.,observation_mode='sensor_track',loft_enabled=False)
 factory=chaffing_factory(None,ChaffSpec(1.),ChaffProgram(.1,per_salvo=3,total=3))
 options=dict(p_max=1.,v0_mps=10000.)
 a=simulate(p,scene,target_factory=factory,chaff_capture=options)
 b=simulate_surface(p,scene,target_factory=factory,chaff_capture=options)
 assert a==b and len(a['summary']['chaff_capture']['trials'])==1
 # This salvo was released before radar startup: it had no accepted target track.
 assert a['summary']['chaff_capture']['trials'][0]['captured_time_s'] is None
 assert not a['summary']['chaff_capture']['trials'][0]['eligible']
 normal=simulate(p,scene,target_factory=factory)
 assert simulate(p,scene,target_factory=factory,chaff_capture=None)==normal
 class T:
  def state_at(self,t):return TargetState((7000.-100.*t,1000.,0.),(-100.,0.,0.))
 target_model=ChaffingTarget(T(),ChaffSpec(1.),ChaffProgram(.1,per_salvo=2,total=2))
 rt=create_surface_missile(p,launch_position_m=(0.,1000.,0.),launch_velocity_mps=(300.,0.,0.),
                          target=target_model,launch_pitch_deg=0.,launch_heading_deg=0.,end_time_s=2.,chaff_capture=options,require_seeker_lock=True)
 while not rt.done:rt.step()
 assert len(rt.result()['chaff_capture']['trials'])==1
 with pytest.raises(SimulationInputError):simulate(p,dict(scene,observation_mode='ideal_truth'),chaff_capture={})
