#!/usr/bin/env python3
"""Conditional TRK-window game simulation, not autonomous capture or full launch.

Only guidance_kinematics is overridden. The replay input adapter and target
motion remain identical across candidates; no feedback from current WT states.
"""
from __future__ import annotations
import argparse,copy,json,math,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import compare as A
from aim120_model.surface_runtime import SurfaceRuntime
from aim120_model.effective_surface_core import EffectiveSurfaceGame
from aim120_model.angular_guidance import AngularObservation,RecordedSeekerSample,RecordedSeekerHistory
from aim120_model.tracking import TrackMode
from aim120_model.target import TargetState

class ComparedRuntime(SurfaceRuntime):
    def guidance_kinematics(self,t,state):
        r,v,omega=super().guidance_kinematics(t,state)
        if self.compare_kind=='full':return r,v,omega
        n=A.unit(r);o=np.asarray(omega,float)
        if self.compare_kind=='tangent':o=o-float(n@o)*n
        else:o,_=A.spherical(n,o,self.paper_pole)
        return r,v,tuple(o)

class InitialTarget:
    def __init__(self,position,velocity):self.p=np.asarray(position);self.v=np.asarray(velocity)
    def state_at(self,t):return TargetState(tuple(self.p+self.v*t),tuple(self.v))

def history_for(records,epoch):
    samples=[]
    for r in records:
        assert r['code']==31 and r['timer']==0
        age=(r['tick']-epoch)/48
        obs=AngularObservation.from_native(direction=r['n'],full_rate_rad_s=r['w'],range_m=r['R'],closing_mps=r['closing'],observer_position_m=r['p'],observer_velocity_mps=r['vm'])
        samples.append(RecordedSeekerSample(age,TrackMode.RADAR_TRACK,tuple(A.C@np.asarray(r['Ps'])),tuple(A.C@np.asarray(r['Vs'])),obs))
    return RecordedSeekerHistory(samples,restore_native_clock=False)

def simulate(all_records,case,kind,substeps):
    selected=[r for r in all_records if case['start']<=r['tick']<=case['end']]
    assert selected[0]['tick']==case['start'] and selected[-1]['tick']==case['end']
    assert all(b['tick']>a['tick'] for a,b in zip(selected,selected[1:]))
    assert all(r['code']==31 and r['timer']==0 and r['flag']==0 for r in selected)
    seed=selected[0];previous=max((r for r in all_records if r['tick']<seed['tick']),key=lambda r:r['tick'])
    assert 0<seed['tick']-previous['tick']<=8
    p=A.C@np.asarray(seed['p']);v=A.C@np.asarray(seed['vm']);q=A.C@np.asarray(seed['Q'])@A.C;qp=A.C@np.asarray(previous['Q'])@A.C
    body_omega=Rotation.from_matrix(qp.T@q).as_rotvec()/((seed['tick']-previous['tick'])/48)
    quat=Rotation.from_matrix(q).as_quat();start=(seed['tick']-case['epoch'])/48;end=(selected[-1]['tick']-case['epoch'])/48
    hist=history_for(selected,case['epoch']);obs=hist.samples[0].angular;tp,tv=obs.target_motion()
    profile=A.read(A.ROOT/'data/profile_sets/2.59.0.27/missiles/su_r_77_1.json');defaults=A.read(A.ROOT/'config/effective_surface_defaults.json')
    scenario=dict(launch_pitch_deg=0.,launch_heading_deg=0.,launch_altitude_m=float(p[1]),launch_speed_kmh=float(np.linalg.norm(v)*3.6),simulation_dt_s=1/48,observation_mode='ideal_truth',loft_enabled=True)
    target=InitialTarget(np.asarray(tp)-np.asarray(tv)*start,tv)
    model=ComparedRuntime(profile,scenario,{'initial_conditions':{}},defaults,launch_position_m=tuple(p),launch_velocity_mps=tuple(v),launch_quaternion_xyzw=tuple(quat),guidance_input=hist,target=target)
    model.compare_kind=kind;model.paper_pole=A.UP.copy()
    if kind=='paper_capture_body_up':model.paper_pole=q[:,1].copy()
    elif kind=='paper_initial_los_chart':
        n0=A.unit(np.asarray(tp)-p);model.paper_pole=A.unit(A.UP-float(A.UP@n0)*n0)
    model.integration_substeps=substeps
    state=tuple(p)+tuple(v)+tuple(quat)+tuple(body_omega)
    model.state=state;model.time_s=start
    _,dg=model.plant(start,state,model.control);model.previous_acceleration=dg['acceleration_xyz_mps2']
    reference={r['tick']:r for r in selected};trace=[];scored=[]
    # Integrate a common requested horizon with terminal fuse/ground shortcuts
    # disabled by the diagnostic loop, not by changing production code/config.
    for tick in range(case['start'],case['end']+1):
        t=(tick-case['epoch'])/48;model.track=hist(t,state[:3],state[3:6]);rate=model.track.full_rate_rad_s
        _,dg=model.plant(t,state,model.control)
        row=dict(tick=tick,age=t,position=state[:3],velocity=state[3:6],quaternion_xyzw=state[6:10],omega_body=state[10:],alpha_deg=dg['geometric_incidence_unsigned_deg'],speed_mps=float(np.linalg.norm(state[3:6])),input_full_rate=rate,track_position=model.track.position,track_velocity=model.track.velocity)
        if tick in reference:
            r=reference[tick];dp=np.asarray(state[:3])-A.C@np.asarray(r['p']);dv=np.asarray(state[3:6])-A.C@np.asarray(r['vm']);tan=A.unit(A.C@np.asarray(r['vm']));cross=dp-float(dp@tan)*tan
            sc=dict(tick=tick,age=t,position_error_m=float(np.linalg.norm(dp)),position_error_xyz_runtime_m=dp,cross_track_m=float(np.linalg.norm(cross)),speed_error_mps=float(np.linalg.norm(state[3:6])-np.linalg.norm(r['vm'])),velocity_vector_error_mps=float(np.linalg.norm(dv)),attitude_error_deg=Rotation.from_matrix(Rotation.from_quat(state[6:10]).as_matrix()@(A.C@np.asarray(r['Q'])@A.C).T).magnitude()*A.D,raw_sha=r['raw_sha'])
            scored.append(sc);row['score']=sc
        if tick==case['end']:
            trace.append(row);break
        model.control=model.update_control(t,state);row['command_mps2']=model.last_request;row['control']=model.control;trace.append(row)
        cuts=[t,t+1/48];clock=0.
        for stage in model.stages:
            fire=clock+float(stage.get('fire_delay_s') or 0.);clock=fire+stage['duration_s']
            cuts.extend(z for z in (fire,clock) if t<z<t+1/48)
        cuts=sorted(set(cuts))
        for left,right in zip(cuts,cuts[1:]):
            h=(right-left)/substeps
            for j in range(substeps):state=EffectiveSurfaceGame.step(model,left+j*h,state,model.control,h)
        _,dg=model.plant(t+1/48,state,model.control);model.previous_acceleration=dg['acceleration_xyz_mps2']
        if not all(math.isfinite(x) for x in state):raise ValueError('Nonfinite flight state')
    summary=dict(kind=kind,integration_substeps=substeps,tick0=case['start'],tick1=case['end'],start_age=start,end_age=end,duration_s=end-start,scored_original_nodes=len(scored),initial_body_omega_runtime=body_omega,paper_pole_runtime=model.paper_pole,maximum_position_error_m=max(r['position_error_m'] for r in scored),maximum_abs_speed_error_mps=max(abs(r['speed_error_mps']) for r in scored),maximum_velocity_vector_error_mps=max(r['velocity_vector_error_mps'] for r in scored),maximum_cross_track_m=max(r['cross_track_m'] for r in scored),rms_position_error_m=A.stats([r['position_error_m'] for r in scored])['rms'],terminal=scored[-1],minimum_altitude_m=min(r['position'][1] for r in trace),initial_controller='zero integral/previous_error, previous_acceleration initialized using frozen plant and zero control',gain_tables=profile['guidance'].get('time_to_hit_gain_table'),pn_gain=profile['guidance']['pn_gain'])
    return summary,trace

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    verified=A.verify(a.input);anchors=A.read(a.input/'anchors.json.gz');summary=dict(scope='Same replay adapter, frozen production plant/control, guidance-consumed rate only changed. Starts at continuous TRK boundary with identical unknown-controller-memory zero initialization. Not autonomous radar, original full-launch validation, actual hits, or identification of game intent.',files_verified=verified,cases={})
    for uid,case in A.CASES.items():
        results=[];traces={}
        kinds=('full','tangent','paper_worldup','paper_capture_body_up','paper_initial_los_chart')
        for kind in kinds:
            s,tr=simulate(anchors[uid],case,kind,4);results.append(s);traces[kind]=tr;A.save(a.out/f'{uid}_{kind}_4.json.gz',tr);print('FLIGHT',uid,json.dumps(A.clean(s)),flush=True)
        sensitivity=[]
        for kind in ('full','tangent','paper_worldup'):
            s,tr=simulate(anchors[uid],case,kind,8);base=traces[kind]
            assert [r['tick'] for r in base]==[r['tick'] for r in tr]
            s['max_position_change_from_4substeps_m']=max(float(np.linalg.norm(np.asarray(x['position'])-y['position'])) for x,y in zip(tr,base));sensitivity.append(s);A.save(a.out/f'{uid}_{kind}_8.json.gz',tr);print('FLIGHT_SENSITIVITY',uid,json.dumps(A.clean(s)),flush=True)
        base=traces['full'];divergence={}
        for kind,tr in traces.items():
            divergence[kind]=dict(max_position_difference_from_full_m=max(float(np.linalg.norm(np.asarray(x['position'])-y['position'])) for x,y in zip(tr,base)),terminal_difference_from_full_runtime_xyz_m=np.asarray(tr[-1]['position'])-base[-1]['position'])
        summary['cases'][uid]=dict(models=results,sensitivity=sensitivity,intermodel=divergence)
    A.save(a.out/'summary.json',summary)
if __name__=='__main__':main()
