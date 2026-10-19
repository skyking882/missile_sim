#!/usr/bin/env python3
"""No-fit War Thunder held recovery diagnosis on frozen source records."""
from __future__ import annotations
import argparse,copy,hashlib,inspect,json,math,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'capture_retry_20261019'))
import replay_check as R
DEG=180/math.pi

def unit(a):
    a=np.asarray(a,float);return a/np.linalg.norm(a)
def angle(a,b):return R.angle(a,b)
def ray(a):
    y,p=a;return np.array([math.cos(p)*math.cos(y),math.sin(p),-math.cos(p)*math.sin(y)])
def angles(a):
    a=unit(a);return np.array([math.atan2(-a[2],a[0]),math.asin(np.clip(a[1],-1,1))])
def clipray(n,q):return q@ray(np.clip(angles(q.T@unit(n)),-55/DEG,55/DEG))
def rot(n,w,t):return Rotation.from_rotvec(-np.array(w)*t).apply(n)
def metric(a):
    a=np.asarray(a,float)
    if not len(a):return {'n':0}
    return dict(n=len(a),rms=float(np.sqrt(np.mean(a*a))),median=float(np.median(a)),maximum=float(np.max(a)))
def beam(n,g,r,width,rcs):
    d=angle(n,g);single=max(math.exp(-math.log(2)*(d/width)**2),.001)
    det=16000*(rcs/2*single*single)**.25
    return dict(offset_deg=d,range_m=r,detection_range_m=det,margin_db=40*math.log10(det/r),pass_signal=det>=r)

def held_episodes(rs,epoch):
    out=[];i=1
    while i<len(rs):
        a,b=rs[i-1:i+1]
        if (b['tick']-epoch)/48>6:break
        if a['code']==b['code']==31 and a['timer']==0 and b['timer']>0 and a['flag']==b['flag']==0:
            seed=a;held=[];j=i;prev=a
            while j<len(rs):
                r=rs[j];dt=(r['tick']-prev['tick'])/48
                if dt<=0 or dt>8/48 or r['code']!=31 or r['flag']!=0 or r['timer']<=0 or abs(r['timer']-prev['timer']-dt)>2*10/32767:break
                held.append(r);prev=r;j+=1
            if held:out.append((seed,held));i=max(i+1,j);continue
        i+=1
    return out

def snapshots(rs,frames,epoch):
    fr={r['tick']:r for r in frames};eps=[]
    for seed,held in held_episodes(rs,epoch):
        q0=np.array(seed['Q']);g0=q0@ray(seed['angles']);n0=np.array(seed['n']);w0=np.array(seed['w']);p0=np.array(seed['Ps']);v0=np.array(seed['Vs']);rows=[]
        for r in held:
            if r['tick'] not in fr:continue
            f=fr[r['tick']];dt=(r['tick']-seed['tick'])/48;q=np.array(r['Q']);own=np.array(r['p']);target=R.C@np.array(f['target_position']);ntrue=unit(target-own);distance=float(np.linalg.norm(target-own));grec=q@ray(r['angles'])
            propagated=rot(n0,w0,dt)
            candidates=dict(PV_CV=clipray(p0+v0*dt-own,q),fixed_cached_Ps=clipray(p0-own,q),angular_const_rate=clipray(propagated,q),fixed_body_angles=q@ray(seed['angles']),fixed_world_axis=clipray(g0,q),recorded_axis=grec,current_saved_direction=clipray(np.array(r['n']),q))
            rec=dict(tick=r['tick'],age=(r['tick']-epoch)/48,timer=r['timer'],raw_sha=r['raw_sha'],index=r['index'],angle_int=r['angle_int'],angles_deg=np.array(r['angles'])*DEG,angle_change_deg=(np.array(r['angles'])-seed['angles'])*DEG,rate_same=r['w']==seed['w'],Ps_same=r['Ps']==seed['Ps'],Vs_same=r['Vs']==seed['Vs'],direction_prediction_error_deg=angle(propagated,r['n']),Ps_to_target_m=float(np.linalg.norm(np.array(r['Ps'])-target)),body_turn_deg=float(Rotation.from_matrix(q@q0.T).magnitude()*DEG),candidates={})
            for name,g in candidates.items():
                rec['candidates'][name]=dict(axis_error_deg=angle(g,grec),beam={f'{width:g}_{rcs:g}':beam(ntrue,g,distance,width,rcs) for width in (7.5,15.) for rcs in (1.,2.)})
            rows.append(rec)
        summary={name:dict(pointing_error=metric([r['candidates'][name]['axis_error_deg'] for r in rows]),signal_passes_15_rcs2=sum(r['candidates'][name]['beam']['15_2']['pass_signal'] for r in rows),first_signal_pass_tick=next((r['tick'] for r in rows if r['candidates'][name]['beam']['15_2']['pass_signal']),None)) for name in rows[0]['candidates']}
        sf=fr[seed['tick']];seed_target=R.C@np.array(sf['target_position'])
        eps.append(dict(seed_tick=seed['tick'],seed_age=(seed['tick']-epoch)/48,seed=seed,seed_Ps_to_independent_target_m=float(np.linalg.norm(p0-seed_target)),held_rows=len(rows),summary=summary,rows=rows))
    return eps

class GateAudit(R.LoggedRadar):
    instances=[]
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw);self.audit=[];self.instances.append(self)
    def all_gates(self,t,h,own,target,rcs):
        rel=np.array(target.position)-own.position;r=float(np.linalg.norm(rel));n=rel/r;u=np.array(target.velocity)-own.velocity;rr=float(u@n);vg=float(np.array(target.velocity)@n);rate=float(np.linalg.norm(np.cross(rel,u))/(r*r));hold=self._within_track_hold(t);dt=self._effective_dt(t,h);g=np.array(self._antenna_center(own,tuple(n),not hold,t));det=self.detection_range(tuple(n),tuple(g),rcs)
        ds=self.distance_gate;vs=self.doppler_gate;ar=angle(n,self._last_accepted_los)/dt if hold and self._last_accepted_los is not None else None
        checks=dict(receiver_range=r<=self._receiver_range_limit(not hold),beam_range=r<=det,rate=rate<=self.rate_max_rad_s,angle_rate=ar is None or ar<=self.angle_gate_rate_rad_s*DEG,doppler_bounds=self.doppler_min_mps<=rr<=self.doppler_max_mps,clutter=not(self._on_clutter(target.position[1]<own.position[1],own.position,r,tuple(rel)) and abs(vg)<=self.notch_half_width_mps),distance_filter=ds.accepts(r,dt),doppler_filter=vs.accepts(rr,dt))
        scalars={}
        for name,f,value in [('distance',ds,r),('doppler',vs,rr)]:
            pred=f.predict(dt);scalars[name]=dict(initialized=f.initialized,value=f.value,rate=f.rate,predicted=pred,measurement=value,innovation=value-pred,half_range=f.search_range,margin=f.search_range-abs(value-pred))
        # Verify power margin using logarithmic formula independent of range-root.
        offset=angle(g,n);log_gain=sum(max(-math.log(2)*(offset/DEG/a[0])**2,math.log(a[1])) for a in (self.receiver_antenna,self.transmitter_antenna));md=10/math.log(10)*(math.log(rcs/self.reference_rcs_m2)+4*math.log(self.receiver_range_m/r)+log_gain)
        assert abs(md-40*math.log10(det/r))<1e-8
        return dict(age=t,last_accepted_age=self._last_detection_time_s,holding=hold,dt_since_observation=dt,range_m=r,beam_offset_deg=offset,antenna_axis=g,margin_db=md,range_rate_mps=rr,ground_radial_mps=vg,geometric_rate_deg_s=rate*DEG,angle_since_last_per_s=ar,angle_gate_rate_deg_s=self.angle_gate_rate_rad_s*DEG,checks=checks,scalar_gates=scalars,cache_position=self._last_position,cache_velocity=self._last_velocity,estimated_track=repr(self.estimated_track),receiver_range_max=self._receiver_range_limit(not hold))
    def update(self,t,h,own,target,*a,**kw):
        before=self.all_gates(t,h,own,target,kw.get('target_rcs_m2',1.))
        result=super().update(t,h,own,target,*a,**kw);before['accepted']=result[0].valid;before['first_rejection']=result[0].reason;self.audit.append(before);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    manifest=R.read(a.input/'MANIFEST.json')
    for row in manifest:
        f=a.input/row['path'];assert f.stat().st_size==row['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==row['sha256']
    hist=R.read(a.input/'input/anchor_full_histories.json.gz');profile=R.read(R.ROOT/'data/profile_sets/2.59.0.27/missiles/su_r_77_1.json');base=R.read(a.input/'results/replay/summary.json');out=dict(manifest_verified=len(manifest),scope='Conditional held restart diagnostics and unchanged detector gate inspection, no fit, no production patch',cases={},code_contract={})
    for method in ('set_estimated_track','_antenna_center','_coast_solution'):
        out['code_contract'][method]=inspect.getsource(getattr(R.RadarSeekerObserver,method))
    for uid in ('118','148'):
        frames=R.read(a.input/f'results/replay/frames_{uid}_packet.json.gz');condition=next(c for c in base['conditions'] if c['uid']==int(uid) and c['target_clock']=='packet' and c['width']=='off_axis_half_power' and c['rcs_m2']==2 and c['policy']=='retry');epoch=condition['inputs']['epoch_tick']
        eps=snapshots(hist[uid],frames,epoch)
        fresh=[r for r in hist[uid] if r['tick'] in {f['tick'] for f in frames} and r['code']==31 and r['timer']==0]
        fresh_axis=metric([angle(np.array(r['Q'])@ray(r['angles']),clipray(r['n'],np.array(r['Q']))) for r in fresh])
        R.LoggedRadar=GateAudit;s,trace=R.simulate(frames,profile,2.,'off_axis_half_power','retry');old=R.read(a.input/f'results/replay/{uid}_packet_off_axis_half_power_2_retry.json.gz');assert trace==old,uid
        audit=GateAudit.instances[-1].audit;recovery=[x for x in audit if 2.2<=x['age']<=3.35]
        out['cases'][uid]=dict(epoch=epoch,episodes=eps,fresh_axis_clipped_n_error=fresh_axis,unchanged_prior_trace=True,all_gate_records=len(audit),state_changes=s['state_changes'],key_gate_records=[x for x in audit if any(abs(x['age']-t)<.012 for t in (2.2708333333,2.2916666667,2.4375,2.5,3.25,4.2083333333))])
        R.save(a.out/f'all_gates_{uid}.json.gz',audit)
        print('EPISODES',uid,json.dumps(R.clean([{'seed':e['seed_tick'],'rows':e['held_rows'],'seed_Ps_target_error':e['seed_Ps_to_independent_target_m'],'summary':e['summary']} for e in eps])),flush=True)
        if eps:
            rows=eps[0]['rows'];key=[r for r in rows if r['tick']==eps[0]['seed_tick']+8 or abs(r['age']-2.4375)<.025 or r in (rows[0],rows[-1])];print('HELD_ROWS',uid,json.dumps(R.clean(key)),flush=True)
        print('KEY_GATES',uid,json.dumps(R.clean(out['cases'][uid]['key_gate_records'])),flush=True)
    R.save(a.out/'summary.json',out);print('CONTRACTS',json.dumps(out['code_contract']),flush=True)
if __name__=='__main__':main()
