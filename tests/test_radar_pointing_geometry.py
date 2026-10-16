"""Behavioral radar geometry checks with independent analytic boundary cases."""
import math
import sys
from pathlib import Path
import pytest
from aim120_model.aerodynamics import body_axes_for_state
from aim120_model.dynamics import SimState
from aim120_model.observation import SensorTrackProvider
from aim120_model.radar_seeker import RadarSeekerObserver
from aim120_model.target import TargetState
from test_seeker_search import sensor


def state(roll=0.):
    return SimState((0.,1000.,0.),(0.,0.,0.),0.,0.,0.,0.,100.,
                    orientation_quaternion=(math.cos(roll/2),math.sin(roll/2),0.,0.))


def point(ms,yaw,pitch,distance=1000.):
    a,b=math.radians(yaw),math.radians(pitch)
    axes=body_axes_for_state(ms)
    weights=(math.cos(b)*math.cos(a),math.sin(b),math.cos(b)*math.sin(a))
    return tuple(ms.position[i]+distance*sum(w*axis[i]for w,axis in zip(weights,(axes.forward,axes.up,axes.right)))for i in range(3))


@pytest.mark.parametrize('path',['single','decoys','track_gate','scan'])
@pytest.mark.parametrize('roll',[0.,math.pi/2])
def test_diagonal_mechanical_pointing_and_continuous_beam(path,roll):
    ms=state(roll)
    def check(yaw,pitch):
        o=RadarSeekerObserver(sensor(),lock_range_m=20000.)
        o.clutter_model="look_down_angle"
        o.clutter_min_depression_rad=math.pi/2
        p=point(ms,yaw,pitch)
        if path=='track_gate':
            o.enable_track_gate_model({'notch_track_hazard':[]})
            o.track_gate_estimate=p
        if path=='scan':
            from aim120_model.radar_seeker import SeekerSearch
            o.search=SeekerSearch(o,{'scan_half_angle_deg':0.,'dwell_s':0.})
            o.search.estimate=p
        kw={}if path=='single'else{'decoys':[]}
        return o.update(0.,.02,ms,TargetState(p,(0.,0.,0.)),**kw)[0]
    assert check(55.,55.).valid  # Total nose angle70.79; both axes attainable.
    assert check(0.,62.5).valid  # Half power is not a visibility edge.
    assert check(0.,62.501).valid
    assert check(62.5,62.5).valid  # Close reflectors remain detectable off beam.
    assert check(3.,4.).valid


def test_profile_beam_width_is_generic_and_transmitter_fallback_is_not_halved_twice():
    model=sensor()
    model['radar_seeker']['receiver']['antenna']['angle_half_sens_deg']=10.
    del model['radar_seeker']['transmitter']
    o=RadarSeekerObserver(model)
    assert math.degrees(o.beam_half_rad)==pytest.approx(5.)
    assert math.degrees(o.illumination_half_rad)==pytest.approx(5.)
    assert math.degrees(o.lock_angle_max_rad)==pytest.approx(55.)


def test_association_window_uses_designation_before_mechanical_clipping():
    ms=state();o=RadarSeekerObserver(sensor(),lock_range_m=20000.)
    o.enable_track_gate_model({'notch_track_hazard':[]})
    p=point(ms,54.,62.)
    o.track_gate_estimate=p
    assert o.update(0.,.02,ms,TargetState(p,(0.,0.,0.)))[0].valid
    # Same beam geometry, but a genuinely wrong designation still fails5°.
    o=RadarSeekerObserver(sensor(),lock_range_m=20000.)
    o.enable_track_gate_model({'notch_track_hazard':[]})
    o.track_gate_estimate=point(ms,45.,55.)
    detection=o.update(0.,.02,ms,TargetState(point(ms,55.,55.),(0.,0.,0.)))[0]
    assert detection.reason=='acquire_window'


def test_launcher_explicit_aircraft_beam_and_scalar_approximation():
    ms=state();p=SensorTrackProvider(sensor())
    g=dict(yaw_limits_deg=(-30.,30.),pitch_limits_deg=(-20.,20.),beam_half_angle_deg=4.,
           orientation_quaternion_xyzw=(0.,0.,0.,1.))
    p.enable_launcher_radar(ms.position,ms.velocity,geometry=g)
    for pitch,expected in [(24.,''),(24.001,'launcher_gimbal')]:
        assert p._launcher_loses_target(0.,TargetState(point(ms,0.,pitch),(100.,100.,0.)))==expected
    legacy=SensorTrackProvider(sensor())
    legacy.enable_launcher_radar(ms.position,(300.,0.,0.),55.)
    assert legacy._launcher_loses_target(0.,TargetState(point(ms,55.,55.),(100.,100.,0.)))==''
    assert legacy._launcher_loses_target(0.,TargetState(point(ms,55.,55.01),(100.,100.,0.)))=='launcher_gimbal'


sys.path.insert(0,str(Path('/Users/skyking/Documents/WT_overlay')))
from wt_overlay.sensors import RadarSensor,OwnState
from wt_overlay.units import Radar,ScanPattern,Waveform


def aircraft_radar(ha=3.,he=3.):
    pat=ScanPattern('fixed',(-30.,30.),(-20.,20.),0.,1,1.,1.)
    wf=Waveform('tr','sig',100000.,1.,None,None,None,False,100000.,beam_azimuth_deg=ha,beam_elevation_deg=he)
    r=RadarSensor(Radar('test','test',(pat,),(wf,),None,1.))
    r.set_mode('search')
    return r


def test_aircraft_uses_own_limits_and_elliptical_beam_with_roll():
    r=aircraft_radar(3.,5.)
    for roll in (0.,90.):
        own=OwnState((0.,0.,1000.),(0.,0.,0.),0.,0.,roll)
        d=lambda az,el:r._ray(own.axes,az,el)
        assert r._beam_contains(own,d(30.,25.),d(30.,25.),r._for)
        assert not r._beam_contains(own,d(30.,25.001),d(30.,25.001),r._for)
        assert r._beam_contains(own,d(33.,0.),d(33.,0.),r._for)
        assert not r._beam_contains(own,d(33.001,0.),d(33.001,0.),r._for)
        assert not r._beam_contains(own,d(33.,25.),d(33.,25.),r._for)


def test_aircraft_scan_rejects_rectangle_corner_and_covers_circle_boundary():
    r=aircraft_radar();own=OwnState((0.,0.,1000.),(0.,0.,0.),0.)
    pos=lambda az,el:tuple(own.position[i]+1000.*v for i,v in enumerate(r._ray(own.axes,az,el)))
    assert r.beam_covers(0.,own,pos(3.,0.))
    assert not r.beam_covers(0.,own,pos(3.,3.))
    from wt_overlay.sensors import TargetTruth
    assert len(r.update(.02,.02,own,[TargetTruth('inside',pos(2.,0.),(0.,0.,0.)),
                                      TargetTruth('corner',pos(3.,3.),(0.,0.,0.))]).hits)==1

@pytest.mark.parametrize('entry',['simulate','simulate_surface','create_surface_missile'])
def test_explicit_launcher_geometry_reaches_all_public_entries(entry):
    from test_seeker_search import RUNNABLE,scenario
    from aim120_model.public_api import simulate
    from aim120_model.surface_runtime import simulate_surface,create_surface_missile
    profile=next(p for p in RUNNABLE if p['missile_id']=='su_r_77_1')
    g=dict(yaw_limits_deg=(-30.,30.),pitch_limits_deg=(-20.,20.),beam_half_angle_deg=4.,
           orientation_quaternion_xyzw=(0.,0.,0.,1.))
    if entry!='create_surface_missile':
        sc=scenario();sc['max_simulation_time_s']=.1
        result={'simulate':simulate,'simulate_surface':simulate_surface}[entry](profile,sc,launcher_radar_geometry=g)
        assert result['model']['launcher_radar_geometry']==g
    else:
        class Target:
            def state_at(self,t):return TargetState((1000.,1000.,0.),(-100.,0.,0.))
        runtime=create_surface_missile(profile,launch_position_m=(0.,1000.,0.),launch_velocity_mps=(300.,0.,0.),
                       launch_pitch_deg=0.,launch_heading_deg=0.,target=Target(),end_time_s=.02,launcher_radar_geometry=g)
        assert runtime.provider.launcher_radar[-1]==pytest.approx(math.radians(4.))


def test_aircraft_stt_and_fast_refresh_use_reachable_center_not_target_box():
    from wt_overlay.sensors import TargetTruth
    from wt_overlay.units import Tws
    own=OwnState((0.,0.,1000.),(0.,0.,0.),0.)
    r=aircraft_radar(3.,5.)
    def tg(ident,az,el):
        p=tuple(own.position[i]+1000.*v for i,v in enumerate(r._ray(own.axes,az,el)))
        return TargetTruth(ident,p,(0.,0.,0.))
    r.set_mode('stt',stt_truth='edge',t=0.)
    assert len(r.update(.02,.02,own,[tg('edge',30.,25.)]).hits)==1
    assert r.stt_state=='tracking'
    assert not r.update(.04,.02,own,[tg('edge',30.,26.)]).hits
    # A fast revisit uses the held track's prediction with its own fast limits.
    pat=r.radar.search_patterns[0];wf=r.radar.search_waveforms[0]
    fast=ScanPattern('fast',(-30.,30.),(-20.,20.),30.,1,1.,.02)
    tws=Tws((pat,),(wf,),1.,10,0.,fast,.5)
    fast_sensor=RadarSensor(Radar('fast','fast',(pat,),(wf,),tws,1.))
    fast_sensor.set_mode('tws',t=0.)
    d=tuple(tg('edge',30.,25.).position[i]-own.position[i]for i in range(3))
    hit=fast_sensor._try(own,tg('edge',30.,25.),d,1000.,wf,fast_sensor._fast_for,'fast',.02,pointing=d)
    assert hit is not None
    fast_sensor._tick=1;fast_sensor._ingest(hit,.02)
    fast_sensor._tick=2
    assert len(fast_sensor._fast_refresh(.1,own,[tg('edge',30.,25.)]))==1

@pytest.mark.parametrize('path', ['single', 'decoys', 'track_gate', 'scan', 'vg_capture'])
def test_gaussian_echo_has_no_half_power_cutoff_and_rejects_weak_far_return(path):
    def check(distance):
        ms=state();o=RadarSeekerObserver(sensor(),lock_range_m=25000.)
        o.clutter_min_depression_rad=math.pi/2
        p=point(ms,0.,65.,distance)
        if path=='track_gate':
            o.enable_track_gate_model({'notch_track_hazard':[]})
            o.track_gate_estimate=p
        if path=='scan':
            from aim120_model.radar_seeker import SeekerSearch
            o.search=SeekerSearch(o,{'scan_half_angle_deg':0.,'dwell_s':0.})
            o.search.estimate=p
        if path=='vg_capture':
            o.enable_chaff_capture({'p_max':1.})
        kw={} if path=='single' else {'decoys':[]}
        return o.update(0.,.02,ms,TargetState(p,(0.,0.,0.)),target_rcs_m2=2.,**kw)[0]
    assert check(1000.).valid
    far=check(15000.)
    assert not far.valid and far.reason=='beam_range_gate'


def test_ordinary_gaussian_power_and_rcs_match_optional_search():
    from aim120_model.radar_seeker import SeekerSearch
    o=RadarSeekerObserver(sensor());s=SeekerSearch(o,{'scan_half_angle_deg':0.})
    s.beam=(1.,0.,0.);theta=math.radians(7.5);los=(math.cos(theta),math.sin(theta),0.)
    assert o.return_gain(los,s.beam)==pytest.approx(.25)
    assert o.detection_range(los,s.beam,2.)==pytest.approx(16000.*.25**.25)
    assert o.detection_range(los,s.beam,2.)==pytest.approx(s.detection_range(los,2.))


def test_plain_provider_forwards_target_rcs_to_echo_gate():
    def detect(rcs):
        provider=SensorTrackProvider(sensor(),lock_range_m=25000.)
        provider.enable_launcher_support(lambda t,truth:('',(15000.,1000.,0.),(0.,0.,0.)))
        provider.update(0.,.02,state(),TargetState((17000.,1000.,0.),(-100.,0.,0.)),target_rcs_m2=rcs)
        return provider.last_detection
    assert not detect(1.).valid
    assert detect(4.).valid


def test_ordinary_runtime_passes_target_rcs_without_search_or_decoys():
    import json
    from aim120_model.public_api import create_surface_missile
    profile=json.loads((Path(__file__).resolve().parents[1]/'data/profile_sets/2.59.0.27/missiles/su_r_77_1.json').read_text())
    class Target:
        def __init__(self,rcs):self.rcs_m2=rcs
        def state_at(self,t):return TargetState((17500.,1000.,0.),(-300.,0.,0.))
    def detect(rcs):
        runtime=create_surface_missile(profile,launch_position_m=(0.,1000.,0.),launch_velocity_mps=(300.,0.,0.),launch_pitch_deg=0.,launch_heading_deg=0.,target=Target(rcs),end_time_s=.75,loft=False,launcher_support=lambda t,truth:('',(15000.,1000.,0.),(-300.,0.,0.)))
        while not runtime.done:runtime.step()
        return runtime.provider.last_detection
    assert not detect(1.).valid
    assert detect(4.).valid
