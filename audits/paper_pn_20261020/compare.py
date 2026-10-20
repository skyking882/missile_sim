#!/usr/bin/env python3
"""Fixed-input game replay comparison with Yang & Yang Eq8c/9a.

This does not infer a deployed weapon design, the game's private source, or
actual acceleration from a requested command. No fitted coefficients.
"""
from __future__ import annotations
import argparse,gzip,hashlib,json,math,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from aim120_model.effective_surface_core import G,table
from aim120_model.angular_guidance import AngularObservation
D=180/math.pi
C=np.diag([1.,1.,-1.])
UP=np.array([0.,1.,0.])
KINDS=('full','tangent','paper_worldup')
CASES={'118':dict(replay='2026.10.08_11.49.38',eid=2439170,epoch=11415,start=11617,end=12016),
       '148':dict(replay='2026.10.08_12.11.04',eid=1612803,epoch=14630,start=14739,end=15049)}

def read(p):
    with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return json.load(f)
def clean(x):
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    if isinstance(x,float) and not math.isfinite(x):raise ValueError('Nonfinite output')
    return x

def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with (gzip.open(p,'wt') if p.suffix=='.gz' else p.open('w')) as f:json.dump(clean(x),f,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def unit(x):
    x=np.asarray(x,float);s=np.linalg.norm(x)
    if s<1e-14:raise ValueError('Zero vector')
    return x/s

def angle(a,b):
    if np.linalg.norm(a)<1e-12 or np.linalg.norm(b)<1e-12:return None
    return math.atan2(np.linalg.norm(np.cross(a,b)),float(np.dot(a,b)))*D

def stats(x):
    x=np.asarray([v for v in x if v is not None],float)
    if not len(x):return {'n':0}
    return dict(n=len(x),rms=float(np.sqrt(np.mean(x*x))),median=float(np.median(x)),p95_abs=float(np.percentile(abs(x),95)),max_abs=float(np.max(abs(x))),minimum=float(np.min(x)),maximum=float(np.max(x)))

def spherical(n,w,k=UP):
    """Same n_dot as w, but complete fixed-pole spherical-frame spin.

    Works for either w sign convention: the map is linear in w. The pole is
    part of the specified chart and transforms with coordinates, not with time.
    """
    n=unit(n);k=unit(k);w=np.asarray(w,float);u=w-float(w@n)*n;s=float(k@n);den=1-s*s
    if den<=1e-8:raise ValueError('Spherical chart is singular')
    lam=s*float(k@u)/den
    return u+lam*n,lam

def frame(n,k=UP):
    n=unit(n);k=unit(k);e=unit(k-float(n@k)*n)
    return np.column_stack((n,e,np.cross(n,e)))

def projected(a,vm,maximum):
    v=unit(vm);p=a-float(a@v)*v;l=np.linalg.norm(p)
    return p*min(1.,maximum/l) if l>0 else p

def tests():
    rng=np.random.default_rng(20261020);mx=0.;checks=0
    for _ in range(300):
        n=unit(rng.normal(size=3));w=rng.normal(size=3)*.05
        if abs(n@UP)>.95:continue
        u=w-float(n@w)*n;p,lam=spherical(n,w);dt=1e-5
        nplus=Rotation.from_rotvec(-w*dt).apply(n);nminus=Rotation.from_rotvec(w*dt).apply(n)
        numerical=-Rotation.from_matrix(frame(nplus)@frame(nminus).T).as_rotvec()/(2*dt)
        err=np.linalg.norm(numerical-p);mx=max(mx,err);assert err<1e-7
        R=float(rng.uniform(100,20000));c=float(rng.uniform(-100,1600));v=R*np.cross(n,w)-c*n
        np.testing.assert_allclose(np.cross(n,w),np.cross(n,p),atol=2e-14,rtol=0)
        np.testing.assert_allclose(np.cross(p,v)-np.cross(w,v),(lam-float(n@w))*np.cross(n,v),atol=1e-10,rtol=1e-13)
        if np.linalg.norm(u)>1e-9:
            s=np.linalg.norm(u);t=np.cross(n,u)/s;b=np.cross(n,t)
            expansion=R*s*s*n+c*s*t+lam*R*s*b
            np.testing.assert_allclose(np.cross(p,v),expansion,atol=1e-10,rtol=1e-13)
        q=Rotation.random(random_state=rng).as_matrix();pp,ll=spherical(q@n,q@w,q@UP)
        np.testing.assert_allclose(pp,q@p,atol=1e-12,rtol=0)
        rp,rl=spherical(C@n,-C@w,C@UP)
        np.testing.assert_allclose(rp,-C@p,atol=1e-12,rtol=0)
        np.testing.assert_allclose(np.cross(-C@w,C@v),C@np.cross(w,v),atol=1e-10,rtol=1e-13)
        obs=AngularObservation.from_native(direction=n,full_rate_rad_s=w,range_m=R,closing_mps=c,observer_position_m=(1,2,3),observer_velocity_mps=(200,3,4))
        pv=obs.target_motion();obs2=AngularObservation.from_native(direction=n,full_rate_rad_s=p,range_m=R,closing_mps=c,observer_position_m=(1,2,3),observer_velocity_mps=(200,3,4))
        np.testing.assert_allclose(obs2.target_motion(),pv,atol=1e-9,rtol=1e-13)
        checks+=1
    return dict(random_geometry_checks=checks,finite_difference_frame_rate_max_error_rad_s=mx,parity_covariance_and_target_motion_passed=True)

def verify(root):
    rows=read(root/'MANIFEST.json')
    for r in rows:
        p=root/r['path'];assert p.is_file() and p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],str(p)
    return len(rows)

def command_record(meta,r,profile):
    n=unit(r['n']);w=np.asarray(r['w']);u=w-float(w@n)*n;paper,lam=spherical(n,w)
    R=float(r['R']);c=float(r['closing']);v=R*np.cross(n,w)-c*n;vm=np.asarray(r['vm']);N=profile['guidance']['pn_gain']
    h=table(profile['guidance'].get('time_to_hit_gain_table') or [[0.,1.]],R/c if c>0 else math.inf)
    rates={'full':w,'tangent':u,'paper_worldup':paper}
    commands={k:N*h*np.cross(z,v) for k,z in rates.items()}
    limited={k:projected(z,vm,profile['guidance']['maximum_lateral_acceleration_g']*G) for k,z in commands.items()}
    raw=commands['full'];limit=limited['full'];delta=commands['paper_worldup']-raw
    np.testing.assert_allclose(delta,N*h*(lam-float(n@w))*np.cross(n,v),atol=1e-8,rtol=1e-12)
    rv=np.asarray(r['Vs'])-vm;rp=np.asarray(r['Ps'])-r['p'];nr=unit(rp);wr=np.cross(rv,nr)/np.linalg.norm(rp);pvr,lvr=spherical(nr,wr)
    planar_normal=unit(u) if np.linalg.norm(u)>1e-12 else np.zeros(3)
    return dict(replay=meta['replay'],eid=meta['eid'],uid=meta['uid'],tick=r['tick'],index=r['index'],raw_sha=r['raw_sha'],lambda_WT_deg_s=float(n@w)*D,lambda_paper_deg_s=lam*D,lambda_difference_deg_s=(lam-float(n@w))*D,transverse_rate_deg_s=float(np.linalg.norm(u))*D,range_m=R,closing_mps=c,relative_speed_mps=float(np.linalg.norm(v)),gain=N*h,native_n=n,native_w=w,relative_velocity=v,full_command_g=float(np.linalg.norm(raw))/G,paper_command_g=float(np.linalg.norm(commands['paper_worldup']))/G,tangent_command_g=float(np.linalg.norm(commands['tangent']))/G,paper_difference_g=float(np.linalg.norm(delta))/G,tangent_difference_g=float(np.linalg.norm(commands['tangent']-raw))/G,paper_angle_deg=angle(raw,commands['paper_worldup']),paper_relative_error=float(np.linalg.norm(delta)/np.linalg.norm(raw)) if np.linalg.norm(raw)>1e-9 else None,paper_limited_difference_g=float(np.linalg.norm(limited['paper_worldup']-limit))/G,tangent_limited_difference_g=float(np.linalg.norm(limited['tangent']-limit))/G,paper_limited_angle_deg=angle(limit,limited['paper_worldup']),full_out_of_plane_g=float(raw@planar_normal)/G,paper_out_of_plane_g=float(commands['paper_worldup']@planar_normal)/G,command_full_native=raw,command_paper_native=commands['paper_worldup'],command_tangent_native=commands['tangent'],geometric_PV_paper_rate_deg_s=pvr*D,geometric_PV_paper_difference_g=float(np.linalg.norm(N*h*np.cross(pvr,rv)-raw))/G)

METRICS=('lambda_WT_deg_s','lambda_paper_deg_s','lambda_difference_deg_s','paper_difference_g','tangent_difference_g','paper_angle_deg','paper_relative_error','paper_limited_difference_g','tangent_limited_difference_g','paper_limited_angle_deg','full_command_g','paper_command_g','full_out_of_plane_g','paper_out_of_plane_g','geometric_PV_paper_difference_g')
def summarize(rows):return dict(n=len(rows),entities=len({(r['replay'],r['eid']) for r in rows}),metrics={k:stats([r[k] for r in rows]) for k in METRICS})

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    verified=verify(a.input);fresh=read(a.input/'fresh.json.gz');profile=read(ROOT/'data/profile_sets/2.59.0.27/missiles/su_r_77_1.json');rows=[];excluded={}
    for x in fresh:
        r=x['record']
        if r['flag']!=0 or r['timer']!=0 or r['code']!=31:excluded['not_fresh_flag0']=excluded.get('not_fresh_flag0',0)+1;continue
        if r['R']<=0:excluded['nonpositive_range']=excluded.get('nonpositive_range',0)+1;continue
        try:rows.append(command_record(x,r,profile))
        except ValueError as exc:
            if str(exc)!='Spherical chart is singular':raise
            excluded['singular_chart']=excluded.get('singular_chart',0)+1
    groups={'all_fresh':rows}
    for uid,cc in CASES.items():groups[uid+'_continuous_TRK']=[r for r in rows if r['replay']==cc['replay'] and r['eid']==cc['eid'] and cc['start']<=r['tick']<=cc['end']]
    cc=CASES['118'];groups['118_first_fresh']=[r for r in rows if r['replay']==cc['replay'] and r['eid']==cc['eid'] and 11521<=r['tick']<=11524]
    summary=dict(scope='Same-input commanded PN difference, not measured game acceleration or proof of game guidance consumption. Fresh data only, h and clipping preserved from frozen config.',source_sha256=sha(a.input/'fresh.json.gz'),files_verified=verified,mathematical_tests=tests(),excluded=excluded,groups={k:summarize(v) for k,v in groups.items()},points=[r for r in rows if any(r['replay']==cc['replay'] and r['eid']==cc['eid'] and r['tick'] in (cc['start'],cc['end'],11524) for cc in CASES.values())])
    save(a.out/'summary.json',summary);save(a.out/'rows.json.gz',rows)
    for name,v in summary['groups'].items():print('COMMAND_GROUP',name,json.dumps(clean(v)),flush=True)
    for v in summary['points']:print('COMMAND_POINT',json.dumps(clean(v)),flush=True)
    print('COMMAND_VERIFICATION',json.dumps(clean({k:summary[k] for k in ('files_verified','mathematical_tests','excluded')})),flush=True)
if __name__=='__main__':main()
