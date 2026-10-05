from __future__ import annotations
import copy
import json
import math
from pathlib import Path

from aim120_model.public_api import simulate, validate_scenario, _case_from_scenario
from aim120_model.surface_runtime import SurfaceRuntime
from aim120_model.effective_surface_core import EffectiveSurfaceGame
from aim120_model.tracking import TrackMode, TrackSolution
from missile_gui.library import scan_library
from aim120_model.profile_catalog import load_profile_catalog

ROOT=Path(__file__).resolve().parents[1]

def inputs():
    _,catalog=load_profile_catalog(ROOT)
    profiles,errors=scan_library(catalog['profiles_dir'],ROOT)
    assert not errors
    return {p['missile_id']:p for p in profiles}

def scenario():
    return dict(launch_speed_kmh=1200,launch_altitude_m=6300,launch_pitch_deg=0,launch_heading_deg=0,
        target_speed_kmh=1200,target_altitude_m=6300,initial_distance_m=8000,target_azimuth_deg=30,
        target_heading_deg=180,target_course_reference='absolute_world',target_vertical_heading_deg=0,
        target_constant_turn_g=0,max_simulation_time_s=3,observation_mode='sensor_track',loft_enabled=True)

def runtime(profile,s):
    s=validate_scenario(s)
    return SurfaceRuntime(profile,s,_case_from_scenario(s),json.loads((ROOT/'config/effective_surface_defaults.json').read_text()))

def test_general_uses_single_surface_navigation_and_per_profile_stages():
    for name in ['su_r_77_1','us_aim_120a','cn_pl12','atam_mistral']:
        p=inputs()[name];s=scenario();r=simulate(p,s)
        assert r['model']['runtime_adapter']=='effective_single_surface_v1'
        assert r['model']['observation_mode']=='sensor_track'
        assert r['model']['control']=='raw_parallel_pid'
        assert r['samples'][0]['mass_kg']==p['geometry']['initial_mass_kg']
        json.dumps(r,allow_nan=False)

def test_reference_formula_preserved_for_loft_and_large_offaxis():
    profile=inputs()['su_r_77_1']
    for azimuth in [0,60]:
        s=scenario();s.update(observation_mode='ideal_truth',target_azimuth_deg=azimuth)
        model=runtime(profile,s)
        assert model.creation["quaternion_xyzw_raw"] == (0.,0.,0.,1.)
        raw=copy.deepcopy(model.raw)
        stage=profile['propulsion']['stages'][0]
        raw.update(timeFire=stage['duration_s'],force=stage['thrust_n'],massEnd=raw['mass']-stage['mass_lost_kg'])
        # This is the retained reference forward path, with the same launch,
        # ordinary controller and coefficient values, before terminal events.
        reference=EffectiveSurfaceGame(raw,copy.deepcopy(model.config),model.creation).run(3)
        rows=model.simulate(3)['samples']
        assert len(rows)==len(reference)
        assert max(math.dist(a['position_xyz_m'],b['position_m']) for a,b in zip(reference,rows))<1e-8
        assert max(math.dist(a['velocity_xyz_mps'],b['velocity_mps']) for a,b in zip(reference,rows))<1e-8

def test_drag_continuation_is_positive_and_smooth_and_zero_speed_is_finite():
    s=scenario();s['launch_speed_kmh']=0
    model=runtime(inputs()['su_r_77_1'],s)
    c0,c1,c2=model.p['cd_base_inverse_mach'];join=-2*c2/c1
    below=model.coefficients(join-1e-7,0)[0];above=model.coefficients(join+1e-7,0)[0]
    assert abs(below-above)<1e-10
    assert abs((above-model.coefficients(join,0)[0])/1e-7)<1e-5
    state=tuple(model.creation['position_xyz_m'])+(0.,0.,0.)+tuple(model.creation['quaternion_xyzw_raw'])+(0.,0.,0.)
    derivative,diagnostic=model.plant(0,state,(0.,0.))
    assert all(math.isfinite(v) for v in derivative)
    assert diagnostic['drag_base_n']==diagnostic['drag_aoa_n']==diagnostic['lift_n']==0.
    r=simulate(model.profile,s)
    json.dumps(r,allow_nan=False)
    assert r['samples'][1]['velocity_mps']!=(0.,0.,0.)

def test_guidance_uses_provider_solution_and_lost_track_has_no_request():
    model=runtime(inputs()['su_r_77_1'],scenario())
    state=tuple(model.creation['position_xyz_m'])+tuple(model.creation['velocity_xyz_mps'])+tuple(model.creation['quaternion_xyzw_raw'])+(0.,0.,0.)
    model.track=TrackSolution((1000.,6500.,500.),(-100.,0.,0.),0.,0.,TrackMode.INERTIAL,True,'test_track')
    command=model.guidance(1.,state)
    model.target=object() # Navigation never queries this truth-only object.
    model.target0=(1e9,1e9,1e9);model.target_velocity=(1e9,1e9,1e9)
    assert model.guidance(1.,state)==command
    model.track=TrackSolution((1e9,1e9,1e9),(1e9,1e9,1e9),0.,0.,TrackMode.LOST,False,'lost')
    assert model.update_control(1.,state)==(0.,0.)
    assert model.last_request==(0.,0.,0.)

def test_multistage_delay_and_mass_are_consumed_without_raw_checkout():
    profile=copy.deepcopy(inputs()['su_r_77_1'])
    profile['propulsion']['stages']=[dict(name='first',duration_s=1.,thrust_n=100.,mass_lost_kg=2.,fire_delay_s=0.),dict(name='second',duration_s=2.,thrust_n=50.,mass_lost_kg=3.,fire_delay_s=.5)]
    model=runtime(profile,scenario())
    assert model.mass_thrust(.5)==(189.,100.)
    assert model.mass_thrust(1.25)==(188.,0.)
    assert model.mass_thrust(2.5)==(186.5,50.)
    assert model.mass_thrust(4.)==(185.,0.)


def test_sensor_quaternion_bridge_preserves_all_three_body_axes():
    from aim120_model.aerodynamics import body_axes_from_quaternion
    from aim120_model.effective_surface_core import axes
    s=scenario();s.update(launch_pitch_deg=37,launch_heading_deg=-42)
    model=runtime(inputs()['su_r_77_1'],s)
    state=tuple(model.creation['position_xyz_m'])+tuple(model.creation['velocity_xyz_mps'])+tuple(model.creation['quaternion_xyzw_raw'])+(0.,0.,0.)
    expected=axes(state[6:10]);actual=body_axes_from_quaternion(model.missile_state(state).orientation_quaternion)
    for a,b in zip(expected,[actual.forward,actual.up,actual.right]):
        assert math.dist(a,b)<1e-12
    assert math.dist(state[3:6],tuple(v*1200/3.6 for v in actual.forward))<1e-12
