#!/usr/bin/env python3
"""Fixed game replay mechanism tests, not a fitted autonomous observer."""
from __future__ import annotations
import argparse,copy,gzip,hashlib,json,math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

DEG=180/math.pi
QW=15/32767
ANCHORS={('2026.10.08_11.49.38',2439170):118,('2026.10.08_12.11.04',1612803):148}

def read(p):
    with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return json.load(f)
def clean(x):
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    if isinstance(x,float) and not math.isfinite(x):return None
    return x

def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with (gzip.open(p,'wt') if p.suffix=='.gz' else p.open('w')) as f:json.dump(clean(x),f,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(root):
    rows=read(root/'MANIFEST.json')
    for r in rows:
        p=root/r['path'];assert p.is_file() and p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],str(p)
    return len(rows)
def v(x):return np.asarray(x,float)
def unit(x):
    x=v(x);a=np.linalg.norm(x)
    if not np.isfinite(a) or a<1e-15:raise ValueError('Invalid direction')
    return x/a

def ang(a,b):
    a,b=unit(a),unit(b)
    return math.atan2(np.linalg.norm(np.cross(a,b)),np.dot(a,b))*DEG

def propagate(n,w,t):return Rotation.from_rotvec(-v(w)*t).apply(unit(n))
def manual(n,w,t):
    n=unit(n);a=np.linalg.norm(w)
    if a<1e-15:return n.copy()
    e=v(w)/a;z=a*t
    return math.cos(z)*n+math.sin(z)*np.cross(n,e)+(1-math.cos(z))*np.dot(e,n)*e

def frame(n,up=(0.,1.,0.)):
    n=unit(n);up=v(up);e=up-n*np.dot(n,up)
    if np.linalg.norm(e)<1e-6:return None
    e=unit(e);f=np.column_stack((n,e,np.cross(n,e)))
    np.testing.assert_allclose(f.T@f,np.eye(3),atol=2e-12,rtol=0)
    assert np.linalg.det(f)>0
    return f

def ray(angles):
    a,e=angles
    return np.array([math.cos(e)*math.cos(a),math.sin(e),math.cos(e)*math.sin(a)])

def frames(r):
    n=unit(r['n']);q=v(r['Q']);bn=unit(q.T@n);fb=frame(bn)
    fg=frame(ray(r['angles']))
    return {'world_up':frame(n),'body_up_target':q@fb if fb is not None else None,'actual_gimbal':q@fg if fg is not None else None,'body_only':q}

def metric(values):
    x=np.asarray([a for a in values if a is not None and math.isfinite(a)],float)
    if not len(x):return {'n':0}
    return dict(n=len(x),rms=float(np.sqrt(np.mean(x*x))),median=float(np.median(x)),p95=float(np.percentile(x,95)),max=float(np.max(x)),min=float(np.min(x)))

def math_checks():
    rng=np.random.default_rng(902019);maximum=0.
    for _ in range(200):
        n=unit(rng.normal(size=3));w=rng.normal(size=3);lam=float(n@w);u=w-lam*n;t=float(rng.uniform(0,1))
        nf=propagate(n,w,t);m=manual(n,w,t);maximum=max(maximum,float(np.linalg.norm(nf-m)))
        np.testing.assert_allclose(np.cross(n,w),np.cross(n,u),atol=1e-14,rtol=0)
        second=np.cross(np.cross(n,w),w)
        np.testing.assert_allclose(second,lam*u-float(u@u)*n,atol=3e-14,rtol=0)
        ut=Rotation.from_rotvec(-w*t).apply(u)
        np.testing.assert_allclose(ut,w-lam*nf,atol=1e-14,rtol=0)
        np.testing.assert_allclose(np.cross(nf,ut),np.cross(nf,w),atol=1e-14,rtol=0)
        q=Rotation.random(random_state=rng).as_matrix();np.testing.assert_allclose((q@n)@(q@w),lam,atol=1e-14,rtol=0)
    a0=.2;e=.6;adot=.3;dt=1e-5
    n0,n1=ray([a0,e]),ray([a0+adot*dt,e]);f0,f1=frame(n0),frame(n1)
    wf=-Rotation.from_matrix(f1@f0.T).as_rotvec()/dt
    assert abs(wf@unit(n0+n1)-adot*math.sin(e))<1e-9
    return dict(random_cases=200,manual_rotation_max=maximum,frame_sign_sanity_passed=True,instantaneous_vs_second_derivative_passed=True,transported_tangent_equivalence_passed=True)

def held_audit(episodes):
    out=[];rows=[];frame_rows=[];guards=0
    for e in episodes:
        if not e['strict']:continue
        s=e['seed'];n0=unit(s['n']);w0=v(s['w']);lam=float(n0@w0);u0=w0-lam*n0;flip=w0-2*lam*n0
        start=s;previous=s;previous_frames=frames(s);accum={k:0. for k in previous_frames};valid_duration={k:0. for k in previous_frames};local=[];localframes=[]
        for r in e['held']:
            T=(r['tick']-s['tick'])/48;dt=(r['tick']-previous['tick'])/48
            assert 0<dt<=8/48 and r['w']==s['w']
            n=unit(r['n']);nf=propagate(n0,w0,T);nz=propagate(n0,u0,T);nr=propagate(n0,flip,T)
            mt=manual(n0,w0,T);np.testing.assert_allclose(nf,mt,atol=1e-14,rtol=0)
            transported=Rotation.from_rotvec(-w0*T).apply(u0)
            np.testing.assert_allclose(np.cross(nf,transported),np.cross(nf,w0),atol=1e-14,rtol=0)
            bad=copy.deepcopy(r)
            for k in ('n','w','Ps','Vs','angles'):bad[k]=float('nan')
            nt=propagate(n0,w0,(bad['tick']-s['tick'])/48);np.testing.assert_array_equal(nt,nf);guards+=1
            x=dict(episode_id=e['episode_id'],replay=e['replay'],eid=e['eid'],uid=e['uid'],tick=r['tick'],seed_tick=s['tick'],T=T,lambda0_deg_s=lam*DEG,transverse0_deg_s=float(np.linalg.norm(u0))*DEG,full_error_deg=ang(nf,n),zero_once_error_deg=ang(nz,n),flip_once_error_deg=ang(nr,n),full_vs_zero_deg=ang(nf,nz),full_vs_flip_deg=ang(nf,nr),lambda_observed_deg_s=float(n@v(r['w']))*DEG,lambda_drift_deg_s=float(n@w0-lam)*DEG,raw_sha=r['raw_sha'],seed_sha=s['raw_sha'])
            local.append(x);rows.append(x)
            ff=frames(r);nav=unit(unit(previous['n'])+n);ww=(v(previous['w'])+v(r['w']))/2;lambda_ref=float(ww@nav)*DEG
            for name,f1 in ff.items():
                f0=previous_frames[name]
                if f0 is None or f1 is None:continue
                wf=-Rotation.from_matrix(f1@f0.T).as_rotvec()/dt
                lv=float(wf@nav)*DEG;accum[name]+=lv*dt;valid_duration[name]+=dt
                z=dict(episode_id=e['episode_id'],uid=e['uid'],replay=e['replay'],eid=e['eid'],tick0=previous['tick'],tick1=r['tick'],dt=dt,kind=name,lambda_frame_deg_s=lv,lambda_recorded_deg_s=lambda_ref,axial_error_deg_s=lv-lambda_ref,full_vector_error_deg_s=float(np.linalg.norm(wf-ww))*DEG,axis_vs_n_deg=ang(f1[:,0],n),raw_sha=r['raw_sha'])
                frame_rows.append(z);localframes.append(z)
            previous=r;previous_frames=ff
        end=local[-1];duration=end['T'];fr={}
        for name in accum:
            zz=[x for x in localframes if x['kind']==name]
            fr[name]=dict(mean_lambda_deg_s=accum[name]/valid_duration[name] if valid_duration[name]>0 else None,integrated_spin_deg=accum[name],recorded_spin_deg=lam*DEG*valid_duration[name],axial_error=metric([x['axial_error_deg_s'] for x in zz]),full_vector_error=metric([x['full_vector_error_deg_s'] for x in zz]),axis_error=metric([x['axis_vs_n_deg'] for x in zz]))
        out.append(dict(episode_id=e['episode_id'],uid=e['uid'],replay=e['replay'],eid=e['eid'],seed_tick=s['tick'],seed_sha=s['raw_sha'],last_tick=end['tick'],n=len(local),duration_s=duration,lambda0_deg_s=lam*DEG,transverse0_deg_s=float(np.linalg.norm(u0))*DEG,end=end,resolvable=end['full_vs_zero_deg']>.01,errors={k:metric([r[k] for r in local]) for k in ('full_error_deg','zero_once_error_deg','flip_once_error_deg','lambda_drift_deg_s')},frame_comparison=fr))
    groups={}
    sels={'all':out,'resolvable':[e for e in out if e['resolvable']],'lambda_abs_above_1':[e for e in out if abs(e['lambda0_deg_s'])>1]}
    for name,ee in sels.items():
        ids={e['episode_id'] for e in ee};rr=[r for r in rows if r['episode_id'] in ids]
        groups[name]=dict(episodes=len(ee),entities=len({(e['replay'],e['eid']) for e in ee}),records=len(rr),errors={k:metric([r[k] for r in rr]) for k in ('full_error_deg','zero_once_error_deg','flip_once_error_deg')},full_better_than_zero_episodes=sum(e['errors']['full_error_deg']['rms']<e['errors']['zero_once_error_deg']['rms'] for e in ee),frames={kind:dict(axial_error=metric([r['axial_error_deg_s'] for r in frame_rows if r['episode_id'] in ids and r['kind']==kind]),full_error=metric([r['full_vector_error_deg_s'] for r in frame_rows if r['episode_id'] in ids and r['kind']==kind])) for kind in ('world_up','body_up_target','actual_gimbal','body_only')})
    return out,rows,frame_rows,dict(groups=groups,poison_guards=guards,anchors=[e for e in out if (e['replay'],e['eid']) in ANCHORS])

def near(e):return e['tick1']==((e['tick0']-1)//48+1)*48+1 and e['new']['timer']==0 and e['old']['flag']==e['new']['flag']==0

def interval(coeff,w):
    k=np.rint(v(w)/QW);assert np.max(np.abs(v(w)/QW-k))<1e-8
    lo=(k-np.where(k<=0,1.,0.))*QW;hi=(k+np.where(k>=0,1.,0.))*QW;c=v(coeff)
    return float(np.sum(np.where(c>=0,c*lo,c*hi))),float(np.sum(np.where(c>=0,c*hi,c*lo)))

def capture_audit(events):
    rows=[]
    for e in events:
        a,b=e['old'],e['new'];n0,n1=unit(a['n']),unit(b['n']);w0,w1=v(a['w']),v(b['w']);q0,q1=v(a['Q']),v(b['Q']);np0=unit(v(a['Ps'])-v(a['p']));nprior=unit(v(a['Ps'])+v(a['Vs'])*((e['tick1']-e['tick0'])/48)-v(b['p']))
        actual=float(n1@w1);old=float(n0@w0);term_n=float((n1-n0)@w0);term_w=float(n1@(w1-w0));assert abs(actual-old-term_n-term_w)<1e-12
        candidates={'no_rate_change':np.eye(3),'Q_new':q1,'Q_old':q0,'relative_body_rotation':q1@q0.T,'inverse_Q_new':q1.T}
        preds={};errs={};compatible={};target=interval(n1,w1)
        for name,q in candidates.items():
            value=float(n1@q@w0);preds[name]=value*DEG;errs[name]=(value-actual)*DEG;l,h=interval(q.T@n1,w0);compatible[name]=max(l,target[0])<=min(h,target[1])+1e-14
        full=.8*(q1@w0)+.2*n1*float(n1@q1@w0)
        # Same orthogonal transform on a true old LOS and old rate preserves dot.
        np.testing.assert_allclose((q1@np0)@(q1@w0),np0@w0,atol=2e-14,rtol=0)
        rows.append(dict(replay=e['replay'],eid=e['eid'],uid=e['uid'],family=e['family'],tick0=e['tick0'],tick1=e['tick1'],original=e['original_capture'],near=near(e),new_timer=b['timer'],flags=[a['flag'],b['flag']],old_lambda_saved_deg_s=old*DEG,old_lambda_Ps_deg_s=float(np0@w0)*DEG,actual_lambda_deg_s=actual*DEG,lambda_change_deg_s=(actual-old)*DEG,direction_only_contribution_deg_s=term_n*DEG,rate_change_contribution_deg_s=term_w*DEG,predictions_deg_s=preds,errors_deg_s=errs,scalar_quantization_compatible=compatible,full_baseline_error_deg_s=float(np.linalg.norm(full-w1))*DEG,axis_mismatch_QoldLOS_to_newLOS_deg=ang(q1@np0,n1),oldPV_prediction_to_newLOS_deg=ang(nprior,n1),old_raw_sha=a['raw_sha'],new_raw_sha=b['raw_sha']))
    sels={'all931':rows,'all_R77':[r for r in rows if r['family']=='R-77-1'],'near_R77_79':[r for r in rows if r['family']=='R-77-1' and r['near']],'near_R77_first':[r for r in rows if r['family']=='R-77-1' and r['near'] and r['original']],'near_R77_reacquisition':[r for r in rows if r['family']=='R-77-1' and r['near'] and not r['original']]}
    assert len(sels['near_R77_79'])==79
    groups={name:dict(n=len(rr),errors={k:metric([r['errors_deg_s'][k] for r in rr]) for k in candidates},scalar_compatible={k:sum(r['scalar_quantization_compatible'][k] for r in rr) for k in candidates},actual_lambda=metric([r['actual_lambda_deg_s'] for r in rr]),old_Ps_lambda=metric([r['old_lambda_Ps_deg_s'] for r in rr])) for name,rr in sels.items()}
    return rows,dict(groups=groups,anchors=[r for r in rows if (r['replay'],r['eid']) in ANCHORS])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--memory',type=Path,required=True);ap.add_argument('--offaxis',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);z=ap.parse_args();z.out.mkdir(parents=True,exist_ok=False)
    verified={'memory':verify(z.memory),'offaxis':verify(z.offaxis)}
    epath=z.memory/'input/episodes.json.gz';cpaths=sorted(z.offaxis.rglob('capture_events.json.gz'));assert cpaths
    caps=read(cpaths[0]);episodes=read(epath);assert len(caps)==931 and sum(e['strict'] for e in episodes)==139
    print('INPUT_COUNTS',len(caps),len(episodes),sum(e['strict'] for e in episodes),flush=True)
    ee,rr,fr,hs=held_audit(episodes);cr,cs=capture_audit(caps)
    result=dict(scope='Original WT game-state diagnostics without new fits. Same saved held seed is an input; new capture n is conditional. No autonomous PN or complete-flight ablation performed.',verification=dict(manifest_files=verified,math=math_checks(),held_count=len(ee),held_records=len(rr),capture_records=len(cr)),sources=dict(episodes_sha256=sha(epath),capture_path=str(cpaths[0].relative_to(z.offaxis)),capture_sha256=sha(cpaths[0])),held=hs,captures=cs)
    save(z.out/'summary.json',result);save(z.out/'held_episodes.json.gz',ee);save(z.out/'held_rows.json.gz',rr);save(z.out/'frame_rows.json.gz',fr);save(z.out/'capture_rows.json.gz',cr)
    for name,g in hs['groups'].items():print('HELD_GROUP',name,json.dumps(clean(g)),flush=True)
    for e in hs['anchors']:print('HELD_ANCHOR',json.dumps(clean(e)),flush=True)
    for name,g in cs['groups'].items():print('CAPTURE_GROUP',name,json.dumps(clean(g)),flush=True)
    for e in cs['anchors']:print('CAPTURE_ANCHOR',json.dumps(clean(e)),flush=True)
    print('VERIFIED',json.dumps(clean(result['verification'])),flush=True)
if __name__=='__main__':main()
