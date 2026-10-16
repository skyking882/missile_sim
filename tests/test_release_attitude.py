"""Independent release attitude and world observer information boundaries."""
import json,math,sys
from pathlib import Path

import pytest

from aim120_model.dynamics import SimState
from aim120_model.effective_surface_core import axes,dot,G
from aim120_model.observation import SensorTrackProvider
from aim120_model.public_api import (simulate,SimulationInputError,create_surface_missile,
                                   launch_quaternion_xyzw_from_velocity_normal,
                                   launch_quaternion_xyzw_from_dagor)
from aim120_model.surface_runtime import simulate_surface
from aim120_model.target import TargetState

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def profile():return json.loads((ROOT/'missiles/su_r_77_1.json').read_text())

def scene():
    return dict(launch_speed_kmh=1200,launch_altitude_m=8000,launch_pitch_deg=0,launch_heading_deg=0,
        target_speed_kmh=1000,target_altitude_m=8000,initial_distance_m=10000,target_azimuth_deg=0,
        target_heading_deg=180,target_course_reference='absolute_world',target_vertical_heading_deg=0,
        target_constant_turn_g=0,max_simulation_time_s=.25,observation_mode='ideal_truth',loft_enabled=False)

class Target:
    def state_at(self,t):return TargetState((10000.,8000.,0.),(-300.,0.,0.))

def create(profile,q=None):
    return create_surface_missile(profile,launch_position_m=(0.,8000.,0.),launch_velocity_mps=(400.,0.,0.),
        launch_pitch_deg=0.,launch_heading_deg=0.,target=Target(),observation_mode='ideal_truth',loft=False,
        launch_quaternion_xyzw=q,end_time_s=.25)

def test_api_override_keeps_velocity_and_synchronizes_creation_and_t0(profile):
    q=launch_quaternion_xyzw_from_velocity_normal((400.,0.,0.),(0.,1.,0.),7.)
    runtime=create(profile,tuple(3*x for x in q))
    assert runtime.state[3:6]==(400.,0.,0.)
    assert runtime.state[6:10]==pytest.approx(q)
    assert runtime.creation['quaternion_xyzw_raw']==runtime.state[6:10]
    assert runtime.state[10:13]==(0.,0.,0.)
    # The retained pre-first-step feedback convention remains the gravity prior.
    assert runtime.previous_acceleration==(0.,-G,0.)
    initial_q=runtime.state[6:10]
    while not runtime.done:runtime.step()
    assert runtime.result()['samples'][0]['quaternion_xyzw']==initial_q
    assert runtime.result()['samples'][0]['alpha_deg']==pytest.approx(7.)
    for entry in (simulate,simulate_surface):
        result=entry(profile,scene(),launch_quaternion_xyzw=q)
        assert result['samples'][0]['velocity_mps']==(1200/3.6,0.,0.)
        assert result['samples'][0]['alpha_deg']==pytest.approx(7.)

def test_native_creation_attitude_preserves_all_axes_and_independent_velocity(profile):
    native=(.4622223973274231,-.4326050281524658,-.34564638137817383,-.692626953125)
    q=launch_quaternion_xyzw_from_dagor(native)
    # Reflect both bases independently. A nose-only reconstruction would lose roll.
    c=(1.,1.,-1.)
    native_axes=axes(native)
    expected=tuple(tuple(c[i]*column[i]*c[j] for i in range(3))
                   for j,column in enumerate(native_axes))
    for actual,reference in zip(axes(q),expected):
        assert actual==pytest.approx(reference)
    runtime=create(profile,q)
    assert runtime.state[3:6]==(400.,0.,0.)
    for actual,reference in zip(axes(runtime.state[6:10]),expected):
        assert actual==pytest.approx(reference)
    result=simulate(profile,scene(),launch_quaternion_xyzw=q)
    assert result['samples'][0]['quaternion_xyzw']==pytest.approx(q)
    assert result['samples'][0]['velocity_mps']==(1200/3.6,0.,0.)

def test_replay_creation_attitude_selection_and_clock_boundary():
    sys.path.insert(0,str(ROOT.parent/'WT_overlay'))
    from wt_overlay.sandbox_replay import ReplayTrack
    track=ReplayTrack.__new__(ReplayTrack)
    track.launch_by_uid={118:dict(wt_eid=2439170,t=237.812)}
    rocket=dict(eid=2439170,creation_raw_fields=dict(creation_time=237.8125,
        u16_1=(.4622223973,-.4326050282,-.3456463814,-.6926269531)))
    track.native_launch_rockets={2439170:rocket}
    q,source=track.launch_attitude(118)
    assert q==launch_quaternion_xyzw_from_dagor(rocket['creation_raw_fields']['u16_1'])
    assert source=='native_missile_creation.u16_1'
    # No later sync attitude can silently become a launch attitude.
    missing=dict(eid=2439170,sync=[dict(t_ms=237958,quaternion=(0.,0.,0.,1.))])
    assert track.launch_attitude(118,missing)==(None,'estimated_carrier_velocity_normal_aoa')
    with pytest.raises(ValueError,match='same replay missile'):
        track.launch_attitude(118,dict(rocket,eid=9))
    with pytest.raises(ValueError,match='creation time'):
        track.launch_attitude(118,dict(rocket,creation_raw_fields=dict(creation_time=238.,u16_1=(0,0,0,1))))
    with pytest.raises(ValueError,match='launch_quaternion_xyzw'):
        track.launch_attitude(118,dict(rocket,creation_raw_fields=dict(u16_1=(0,0,0,0))))

def test_nonzero_incidence_moves_with_zero_control_before_delay(profile):
    q=launch_quaternion_xyzw_from_velocity_normal((400.,0.,0.),(0.,1.,0.),7.)
    runtime=create(profile,q);f=axes(q)[0]
    derivative,diagnostic=runtime.plant(0.,runtime.state,(0.,0.))
    assert abs(derivative[12])>0.
    assert diagnostic['lift_n']>0.
    while not runtime.done:runtime.step()
    assert all(row['control']==(0.,0.) for row in runtime.result()['samples'])
    assert math.dist(axes(runtime.state[6:10])[0],f)>1e-3
    assert math.dist(runtime.previous_acceleration,runtime.plant(runtime.time_s,runtime.state,(0.,0.))[1]['acceleration_xyz_mps2'])<1e-10

@pytest.mark.parametrize('normal',[(0.,1.,0.),(0.,0.,1.),(0.,-1.,0.),(0.,0.,-1.)])
def test_full_roll_and_incidence_geometry(normal):
    alpha=math.radians(9.);q=launch_quaternion_xyzw_from_velocity_normal((400.,0.,0.),normal,9.)
    f,up,right=axes(q)
    assert f==pytest.approx((math.cos(alpha),normal[1]*math.sin(alpha),normal[2]*math.sin(alpha)))
    assert up==pytest.approx((-math.sin(alpha),normal[1]*math.cos(alpha),normal[2]*math.cos(alpha)))
    assert dot(f,up)==pytest.approx(0.,abs=1e-12)
    assert right==pytest.approx((0.,-normal[2],normal[1]))

@pytest.mark.parametrize('q',[(0,0,0,0),(0,0,1),('bad',0,0,1),(math.nan,0,0,1),(0,0,math.inf,1)])
def test_invalid_release_attitude_is_rejected(profile,q):
    with pytest.raises(ValueError,match='launch_quaternion_xyzw'):create(profile,q)
    with pytest.raises(SimulationInputError,match='launch_quaternion_xyzw'):simulate(profile,scene(),launch_quaternion_xyzw=q)

def test_release_normal_has_no_silent_roll_fallback():
    with pytest.raises(ValueError,match='transverse'):launch_quaternion_xyzw_from_velocity_normal((400,0,0),(1,0,0),0.)
    with pytest.raises(ValueError,match='nonzero'):launch_quaternion_xyzw_from_velocity_normal((0,0,0),(0,1,0),0.)

def sensor():
    return dict(active_radar=True,datalink=True,inertial_navigation=True,inertial_drift_speed_mps=0.,
        radar_seeker=dict(active=True,rate_max_deg_s=180.,angle_gate_rate_deg_s=180.,
            lock_angle_max_deg=90.,angle_max_deg=90.,receiver=dict(range_max_m=20000.,range_m=20000.,rcs_m2=1.,
                antenna=dict(angle_half_sens_deg=15.,side_lobes_sensitivity_db=-30.)),
            transmitter=dict(antenna=dict(angle_half_sens_deg=15.,side_lobes_sensitivity_db=-30.))))

def test_real_accepted_radar_world_filter_is_invariant_to_own_rotation():
    outputs=[]
    for rotating in (False,True):
        p=SensorTrackProvider(sensor(),lock_range_m=20000.);p.enable_alpha_beta_observer(.8,.05)
        p.enable_launcher_support(lambda t,truth:('',(1000+10*t,1000.,0.),(10.,0.,0.)))
        rows=[]
        for i in range(11):
            t=i*.05;angle=8*math.sin(i*.4) if rotating else 0.
            q=launch_quaternion_xyzw_from_velocity_normal((300.,0.,0.),(0.,1.,0.),angle)
            state=SimState((300*t,1000.,0.),(300.,0.,0.),0.,0.,0.,0.,190.,orientation_quaternion=(q[3],*q[:3]))
            rows.append(p.update(t,.05,state,TargetState((1000+10*t,1000.,0.),(10.,0.,0.))))
            assert p.last_detection.valid
        assert not p.unified_target_observer
        outputs.append(rows)
    assert outputs[0]==outputs[1]
    assert all(row.velocity==(10.,0.,0.) for row in outputs[0])

def test_radar_only_filter_leaves_existing_datalink_estimates_unfiltered():
    p=SensorTrackProvider(sensor(),lock_range_m=20000.);p.enable_alpha_beta_observer(.8,.05);p.radar.active=False
    p.enable_launcher_support(lambda t,truth:('',(1000+100*t*t,1000.,0.),(17+30*t,2.,3.)))
    for t in (0.,.1,.2):
        q=launch_quaternion_xyzw_from_velocity_normal((300.,0.,0.),(0.,0.,1.),20*t)
        state=SimState((300*t,1000.,0.),(300.,0.,0.),0.,0.,0.,0.,190.,orientation_quaternion=(q[3],*q[:3]))
        solution=p.update(t,.1,state,TargetState((9999.,9999.,9999.),(999.,999.,999.)))
        assert solution.position==(1000+100*t*t,1000.,0.)
        assert solution.velocity==(17+30*t,2.,3.)
        assert solution.source=='datalink'
    assert p.alpha_beta_observer.solution is None

def test_wt_caller_preserves_enu_roll_and_aoa():
    sys.path.insert(0,str(ROOT.parent/'WT_overlay'))
    from wt_overlay.engagement import release_quaternion_xyzw
    from wt_overlay.flight import FlightState
    from wt_overlay.escape import from_enu
    s=FlightState((0.,0.,8000.),(300.,0.,0.),(0.,1.,0.),9.)
    q=release_quaternion_xyzw(s);f,up,right=axes(q);alpha=math.radians(9.)
    assert f==pytest.approx(from_enu((math.cos(alpha),math.sin(alpha),0.)))
    assert right==pytest.approx(from_enu((0.,0.,1.)))
    assert s.velocity==(300.,0.,0.)


def test_real_wt_fire_transmits_carrier_frame_without_rotating_velocity():
    sys.path.insert(0,str(ROOT.parent/'WT_overlay'))
    from wt_overlay.engagement import Engagement,PlaneSpec,default_library
    from wt_overlay.flight import FlightState
    from wt_overlay.escape import from_enu
    specs=[PlaneSpec('f_15c_golden_eagle',0,(0.,0.,8000.),(300.,0.,0.),None,'su_r_77_1',1,0),
           PlaneSpec('f_15c_golden_eagle',1,(10000.,0.,8000.),(-300.,0.,0.),None,'su_r_77_1',1,0)]
    eng=Engagement(specs,library=default_library(ROOT));plane,target=eng.planes
    plane.flight.state=FlightState((0.,0.,8000.),(300.,0.,0.),(0.,1.,0.),9.)
    missile=eng.fire(plane,target);rt=missile.runtime;alpha=math.radians(9.)
    assert rt.state[3:6]==from_enu((300.,0.,0.))
    assert axes(rt.state[6:10])[0]==pytest.approx(from_enu((math.cos(alpha),math.sin(alpha),0.)))
    assert axes(rt.state[6:10])[2]==pytest.approx(from_enu((0.,0.,1.)))
    assert rt._rows[0]['alpha_deg']==pytest.approx(9.)
    assert rt.state[10:13]==(0.,0.,0.)
