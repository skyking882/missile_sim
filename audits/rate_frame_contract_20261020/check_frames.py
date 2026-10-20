#!/usr/bin/env python3
"""War Thunder serialized-state frame tests. No fitted coefficients or runtime edits."""
from __future__ import annotations
import argparse,gzip,hashlib,json,math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
D=180/math.pi
QW=15/32767
MODELS=('world','inertial_body','relative_body','zero_axial_world','flip_axial_world')
ANCHORS={('2026.10.08_11.49.38',2439170):118,('2026.10.08_12.11.04',1612803):148}

def read(p):
    with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return json.load(f)
def clean(x):
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    if isinstance(x,float) and not math.isfinite(x):raise ValueError('Nonfinite result')
    return x

def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with (gzip.open(p,'wt') if p.suffix=='.gz' else p.open('w')) as f:json.dump(clean(x),f,ensure_ascii=False,separators=(',',':'),allow_nan=False)
def verify(root):
    rows=read(root/'MANIFEST.json')
    for r in rows:
        p=root/r['path'];assert p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],str(p)
    return len(rows)
def vec(x):return np.asarray(x,float)
def unit(x):
    x=vec(x);m=np.linalg.norm(x)
    if m<1e-14 or not np.isfinite(m):raise ValueError('Invalid vector')
    return x/m

def angle(a,b):
    a,b=unit(a),unit(b)
    return math.atan2(np.linalg.norm(np.cross(a,b)),float(a@b))*D

def exp(x):return Rotation.from_rotvec(vec(x)).as_matrix()
def ray(a):
    yaw,pitch=a
    return np.array([math.cos(pitch)*math.cos(yaw),math.sin(pitch),-math.cos(pitch)*math.sin(yaw)])

def frame(n):
    n=unit(n);up=np.array([0.,1.,0.]);up-=n*float(up@n)
    if np.linalg.norm(up)<1e-6:return None
    up=unit(up);return np.column_stack((n,up,np.cross(n,up)))

def stats(x):
    x=np.asarray(x,float)
    if not len(x):return {'n':0}
    return dict(n=len(x),rms=float(np.sqrt(np.mean(x*x))),median=float(np.median(x)),p95=float(np.percentile(x,95)),maximum=float(np.max(x)),minimum=float(np.min(x)))
def near(e):return e['tick1']==((e['tick0']-1)//48+1)*48+1 and e['new']['timer']==0 and e['old']['flag']==e['new']['flag']==0

def scalar_cell(a,w):
    k=np.rint(vec(w)/QW);assert np.max(np.abs(k-vec(w)/QW))<1e-8
    lo=(k-np.where(k<=0,1.,0.))*QW;hi=(k+np.where(k>=0,1.,0.))*QW;a=vec(a)
    return float(np.minimum(a*lo,a*hi).sum()),float(np.maximum(a*lo,a*hi).sum())

def contract(captures,episodes):
    rows=[]
    for e in captures:
        a,b=e['old'],e['new'];q0,q1=vec(a['Q']),vec(b['Q']);w0,w1=vec(a['w']),vec(b['w']);n0,n1=unit(a['n']),unit(b['n'])
        r=vec(a['Ps'])-a['p'];g=np.cross(vec(a['Vs'])-a['vm'],r)/float(r@r);np1=unit(vec(b['Ps'])-b['p']);loc=ray(a['angles']);wrong=loc.copy();wrong[2]*=-1
        target=scalar_cell(n1,w1);hypotheses={'unchanged':np.eye(3),'Q':q1,'QT':q1.T,'relative_Q':q1@q0.T}
        errs={};compatible={}
        for name,q in hypotheses.items():
            errs[name]=float(n1@(q@w0-w1))*D;l,h=scalar_cell(q.T@n1,w0);compatible[name]=max(l,target[0])<=min(h,target[1])+1e-14
        rows.append(dict(replay=e['replay'],eid=e['eid'],uid=e['uid'],family=e['family'],tick0=e['tick0'],tick1=e['tick1'],near=near(e),first=e['original_capture'],fresh=b['timer']==0,raw_sha0=a['raw_sha'],raw_sha1=b['raw_sha'],INS_pointing_errors_deg={'Q_native':angle(q0@loc,n0),'QT_native':angle(q0.T@loc,n0),'Q_wrong_z':angle(q0@wrong,n0)},TRK_direction_errors_deg={'world':angle(n1,np1),'body':angle(q1@n1,np1)},INS_geometric_rate_error_deg_s={'world':np.linalg.norm(w0-g)*D,'body':np.linalg.norm(q0@w0-g)*D,'inverse_Q_control':np.linalg.norm(q0.T@w0-g)*D},capture_axial_errors_deg_s=errs,capture_axial_compatible=compatible,lambda_world_deg_s=float(n1@w1)*D,lambda_after_Q_deg_s=float(n1@q1@w1)*D,old_world_geo_lambda_deg_s=float(unit(r)@w0)*D))
    groups={}
    for name,rr in [('all931',rows),('R77',[r for r in rows if r['family']=='R-77-1']),('near_R77_79',[r for r in rows if r['family']=='R-77-1' and r['near']])]:
        fresh=[r for r in rr if r['fresh']]
        groups[name]=dict(n=len(rr),fresh_n=len(fresh),INS_pointing={k:stats([r['INS_pointing_errors_deg'][k] for r in rr]) for k in ('Q_native','QT_native','Q_wrong_z')},TRK_direction={k:stats([r['TRK_direction_errors_deg'][k] for r in fresh]) for k in ('world','body')},INS_rate={k:stats([r['INS_geometric_rate_error_deg_s'][k] for r in rr]) for k in ('world','body','inverse_Q_control')},INS_rate_within_01={k:sum(r['INS_geometric_rate_error_deg_s'][k]<.1 for r in rr) for k in ('world','body','inverse_Q_control')},capture_axial={k:stats([r['capture_axial_errors_deg_s'][k] for r in rr]) for k in hypotheses},capture_axial_compatible={k:sum(r['capture_axial_compatible'][k] for r in rr) for k in hypotheses})
    # Corrected native gimbal triad: this supersedes only the old actual_gimbal results.
    spin=[]
    for e in episodes:
        if not e['strict']:continue
        s=e['seed'];prev=s
        for r in e['held']:
            dt=(r['tick']-prev['tick'])/48;fa,fb=frame(ray(prev['angles'])),frame(ray(r['angles']))
            if fa is not None and fb is not None:
                f0,f1=vec(prev['Q'])@fa,vec(r['Q'])@fb;wf=-Rotation.from_matrix(f1@f0.T).as_rotvec()/dt;n=unit(unit(prev['n'])+unit(r['n']));w=vec(s['w'])
                spin.append(dict(episode_id=e['episode_id'],replay=e['replay'],eid=e['eid'],uid=e['uid'],tick=r['tick'],axis_error_deg=angle(f1[:,0],r['n']),axial_error_deg_s=float(n@(wf-w))*D,full_error_deg_s=np.linalg.norm(wf-w)*D))
            prev=r
    spin_groups={}
    for name,rr in [('all',spin),('axis_aligned_005deg',[r for r in spin if r['axis_error_deg']<=.05])]:spin_groups[name]=dict(n=len(rr),axis_error=stats([r['axis_error_deg'] for r in rr]),axial_error=stats([r['axial_error_deg_s'] for r in rr]),full_error=stats([r['full_error_deg_s'] for r in rr]))
    return rows,spin,dict(groups=groups,corrected_gimbal=spin_groups,anchors=[r for r in rows if (r['replay'],r['eid']) in ANCHORS],correction='Prior actual_gimbal scores used wrong native z sign and are superseded. No production parser was changed.')

def exact_step(n,w,q0,q1,dt,kind):
    if kind=='world':return exp(-w*dt)@n
    if kind=='relative_body':return q1@exp(-w*dt)@q0.T@n
    if kind=='inertial_body':
        om=Rotation.from_matrix(q1@q0.T).as_rotvec()/dt;omb=q0.T@om
        return q1@exp(-(w+omb)*dt)@q0.T@n
    raise ValueError(kind)

def propagate(episodes):
    rows=[];ers=[]
    for e in episodes:
        if not e['strict']:continue
        s=e['seed'];n0=unit(s['n']);w=vec(s['w']);lam=float(n0@w);u=w-lam*n0
        state={k:n0.copy() for k in MODELS};prev=s;path=0.;part=[]
        for r in e['held']:
            dt=(r['tick']-prev['tick'])/48;assert 0<dt<=8/48 and r['w']==s['w'];q0,q1=vec(prev['Q']),vec(r['Q']);path+=Rotation.from_matrix(q1@q0.T).magnitude()*D
            for kind in MODELS:
                if kind=='zero_axial_world':state[kind]=exp(-u*dt)@state[kind]
                elif kind=='flip_axial_world':state[kind]=exp(-(w-2*lam*n0)*dt)@state[kind]
                else:state[kind]=exact_step(state[kind],w,q0,q1,dt,kind)
            T=(r['tick']-s['tick'])/48
            np.testing.assert_allclose(state['world'],exp(-w*T)@n0,atol=3e-13,rtol=0)
            np.testing.assert_allclose(state['relative_body'],q1@exp(-w*T)@vec(s['Q']).T@n0,atol=3e-13,rtol=0)
            part.append(dict(episode_id=e['episode_id'],replay=e['replay'],eid=e['eid'],uid=e['uid'],seed_tick=s['tick'],tick=r['tick'],T=T,rotation_path_deg=path,errors_deg={k:angle(x,r['n']) for k,x in state.items()},raw_sha=r['raw_sha'],seed_sha=s['raw_sha']))
            prev=r
        strong=path>=10 and np.linalg.norm(w)*D>=.1
        ers.append(dict(episode_id=e['episode_id'],replay=e['replay'],eid=e['eid'],uid=e['uid'],seed_tick=s['tick'],last_tick=prev['tick'],records=len(part),rotation_path_deg=path,rate_norm_deg_s=np.linalg.norm(w)*D,lambda_deg_s=lam*D,strong_rotation=strong,RMS_deg={k:stats([r['errors_deg'][k] for r in part])['rms'] for k in MODELS},endpoint=part[-1]))
        rows+=part
    stats_groups={}
    for name,ee in [('all',ers),('strong_rotation',[e for e in ers if e['strong_rotation']])]:
        ids={e['episode_id'] for e in ee};rr=[r for r in rows if r['episode_id'] in ids]
        stats_groups[name]=dict(episodes=len(ee),entities=len({(e['replay'],e['eid']) for e in ee}),records=len(rr),errors={k:stats([r['errors_deg'][k] for r in rr]) for k in MODELS},world_better_by_episode={k:sum(e['RMS_deg']['world']<e['RMS_deg'][k] for e in ee) for k in MODELS if k!='world'},endpoint_errors={k:stats([e['endpoint']['errors_deg'][k] for e in ee]) for k in MODELS})
    return rows,ers,dict(groups=stats_groups,anchors=[e for e in ers if (e['replay'],e['eid']) in ANCHORS])

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['contract','propagation'],required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    checked=verify(a.input);eps=read(a.input/'episodes.json.gz');caps=read(a.input/'capture_events.json.gz');assert len(caps)==931 and sum(e['strict'] for e in eps)==139
    if a.mode=='contract':
        rows,spin,s=contract(caps,eps);save(a.out/'rows.json.gz',rows);save(a.out/'gimbal.json.gz',spin)
    else:
        rows,ers,s=propagate(eps);save(a.out/'rows.json.gz',rows);save(a.out/'episodes.json.gz',ers)
    s.update(mode=a.mode,input_files_verified=checked,scope='Read-only WT state diagnostics. No fitted gains; current own Q is conditional input. No production or autonomous flight update.')
    save(a.out/'summary.json',s)
    for name,g in s['groups'].items():print('GROUP',a.mode,name,json.dumps(clean(g)),flush=True)
    for r in s['anchors']:
        if a.mode=='propagation' or (r['uid']==118 and r['tick1']<=11617) or (r['uid']==148 and r['tick1']==14739):print('ANCHOR',a.mode,json.dumps(clean(r)),flush=True)
    if 'corrected_gimbal' in s:print('CORRECTED_GIMBAL',json.dumps(clean(s['corrected_gimbal'])),flush=True)
if __name__=='__main__':main()
