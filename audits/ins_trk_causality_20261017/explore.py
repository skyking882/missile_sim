#!/usr/bin/env python3
"""War Thunder replay field identification. No production simulator imports."""
from __future__ import annotations
import argparse,collections,copy,gzip,json,math,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize_scalar
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'capture_held_20261017'))
import audit as A
H=1/48;TAU=.078525

def load(p):
    with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return json.load(f)

def arr(r):
    r=r.copy()
    for k in ('n','nraw','w','Q','p','vm','Ps','Vs','angles','m'):
        if k in r:r[k]=np.asarray(r[k],float)
    if 'm' not in r:r['m']=r['Q']@A.ray(r['angles'])
    return r

def norm(a):return np.linalg.norm(a,axis=-1)
def un(a):return a/norm(a)[...,None]
def dot(a,b):return np.sum(a*b,axis=-1)
def vang(a,b):return np.arctan2(norm(np.cross(a,b)),dot(a,b))
def sphere(p,m,alpha):
    v=np.cross(m,p);vlen=norm(v);th=np.arctan2(vlen,dot(m,p));c=v*np.divide(th,vlen,out=np.ones_like(th),where=vlen>1e-14)[...,None]
    return un(Rotation.from_rotvec(-alpha*c).apply(p))
def step(n,w,m,T,alpha):
    p=Rotation.from_rotvec(-w*T).apply(n);nn=sphere(p,m,alpha);c=A.correction(p,nn)
    return nn,w+c/(TAU+T/2)

def stats(rows,keys):return {k:A.metric([r[k] for r in rows]) for k in keys}

def frame_task(caps,series,out):
    rows=[]
    for e in caps:
        for stage,name in [('INS','old'),('TRK','new')]:
            r=arr(e[name]);rad=r['Ps']-r['p'];R=norm(rad);n=A.unit(rad);v=r['Vs']-r['vm'];geo=np.cross(v,n)/R
            candidates={'raw':r['w'],'Qraw':r['Q']@r['w'],'QTraw':r['Q'].T@r['w'],'minus_raw':-r['w'],'minus_Qraw':-r['Q']@r['w'],'minus_QTraw':-r['Q'].T@r['w']}
            scores={}
            for key,w in candidates.items():
                error=w-geo;parallel=float(error@n);transverse=error-parallel*n
                scores[key]=dict(full=norm(error)*A.RAD,transverse=norm(transverse)*A.RAD,parallel=parallel*A.RAD)
            rows.append(dict(replay=e['replay'],uid=e['uid'],family=e['family'],stage=stage,fresh=r['timer']==0,original=e['original_capture'],R_geom=R,R_record=r['R'],rate_geo=geo*A.RAD,rate_raw=r['w']*A.RAD,Ps=r['Ps'],Vs=r['Vs'],Ps_closure_m=norm(r['p']+r['R']*r['n']-r['Ps']),radial_error_mps=float(r['Vs']@n-r['d']),n_Ps_deg=A.angle(r['n'],n)*A.RAD,scores=scores))
    summary={}
    for stage in ('INS','TRK'):
        for family in ('all','R-77-1'):
            rr=[r for r in rows if r['stage']==stage and (family=='all' or r['family']==family) and (stage=='INS' or r['fresh'])]
            summary[stage+'/'+family]=dict(count=len(rr),scores={name:{k:A.metric([r['scores'][name][k] for r in rr]) for k in ('full','transverse','parallel')} for name in rows[0]['scores']},closure=stats(rr,['Ps_closure_m','radial_error_mps','n_Ps_deg']))
    # Complete-vector global copy vs scalar transverse correction is diagnostic only.
    shrink=[]
    for e in caps:
        a,b=arr(e['old']),arr(e['new']);n=b['n'];delta={}
        for label,q in [('old',a['Q']),('new',b['Q'])]:
            s=q@a['w'];sp=s-(s@n)*n;wp=b['w']-(b['w']@n)*n;scale=float(sp@wp/(sp@sp)) if sp@sp>1e-12 else None
            floor=norm(wp-scale*sp)*A.RAD if scale is not None else None
            delta[label]=dict(axial_error=float((s-b['w'])@n)*A.RAD,scalar_transverse=scale,unexplained_transverse=floor,copy_error=norm(s-b['w'])*A.RAD)
        tc=((a['tick']-1)//48+1)*48+1;k=b['tick']-tc
        shrink.append(dict(replay=e['replay'],uid=e['uid'],family=e['family'],original=e['original_capture'],phase_in_bracket=a['tick']<tc<=b['tick'],post_ticks=k,details=delta))
    summary['shrink_by_phase']={}
    for k in (-1,0,1,2,3):
        rr=[r for r in shrink if r['phase_in_bracket'] and r['post_ticks']==k and r['family']=='R-77-1']
        if rr:summary['shrink_by_phase'][str(k)]={q:{key:A.metric([r['details'][q][key] for r in rr if r['details'][q][key] is not None]) for key in ('axial_error','scalar_transverse','unexplained_transverse','copy_error')} for q in ('old','new')}
    summary['anchors']=[r for r in rows if any(e['uid']==r['uid'] and e['replay']==r['replay'] and (e['replay'],e['eid']) in A.ANCHORS for e in caps)]
    summary['anchor_shrink']=[r for r in shrink if r['uid'] in (118,148) and r['original']]
    A.save(out/'frame_rows.json.gz',rows);A.save(out/'shrink_rows.json.gz',shrink);A.save(out/'frame_summary.json',summary)
    for k,v in summary.items():
        if k!='anchors':print('FRAME',k,json.dumps(A.serial(v)),flush=True)
    for uid in (118,148):
        rr=[r for r in rows if r['uid']==uid and r['original'] and r['family']=='R-77-1']
        print('FRAME_ANCHOR',uid,json.dumps(A.serial(rr)),flush=True)

def capture_predict(a,b,phase,path,qsource,init,alpha):
    t0,t1=a['tick'],b['tick'];gap=t1-t0
    if not(t0<phase<=t1):raise ValueError('Capture phase not bracketed')
    def own(t):
        f=(t-t0)/gap;return (1-f)*a['p']+f*b['p']
    def measured(t):
        if path=='past_PV':pos=a['Ps']+a['Vs']*((t-t0)/48)
        elif path=='endpoint_PV_DIAGNOSTIC':pos=b['Ps']+b['Vs']*((t-t1)/48)
        else:raise ValueError(path)
        return A.unit(pos-own(t))
    f=(phase-t0)/gap
    q=a['Q'] if qsource=='Qold' else Rotation.from_rotvec(Rotation.from_matrix(b['Q']@a['Q'].T).as_rotvec()*f).as_matrix()@a['Q']
    n=measured(phase);w=q@a['w']
    if init=='self_h':n,w=step(n,w,n,H,alpha)
    for t in range(phase+1,t1+1):n,w=step(n,w,measured(t),H,alpha)
    return n,w

def capture_task(caps,out):
    rows=[];guard=0;skipped=[]
    for e in caps:
        a,b=arr(e['old']),arr(e['new']);phase=((a['tick']-1)//48+1)*48+1
        if not(a['tick']<phase<=b['tick']):skipped.append({k:e[k] for k in ('replay','eid','uid','gap','tick0','tick1','original_capture')});continue
        scores={}
        for path in ('past_PV','endpoint_PV_DIAGNOSTIC'):
            for q in ('Qold','Qphase'):
                for init in ('reset','self_h'):
                    for alpha in (.926,1.):
                        name=f'{path}/{q}/{init}/a{alpha}'
                        n,w=capture_predict(a,b,phase,path,q,init,alpha);scores[name]=dict(**A.errors(n,w,b),n=n,w=w)
                        if path=='past_PV':
                            poisoned=copy.deepcopy(b)
                            for key in ('Ps','Vs','n','w'):poisoned[key]=np.full(3,np.nan)
                            nn,ww=capture_predict(a,poisoned,phase,path,q,init,alpha)
                            np.testing.assert_array_equal(n,nn);np.testing.assert_array_equal(w,ww);guard+=1
        base=e['predictions']['Qold/self_h'];scores['prior_Qold_self_h_recorded_angles']={k:base[k] for k in ('n_deg','w_deg_s','lambda_error_deg_s','transverse_deg_s','P_m')}
        scores['copy_Qold'] = A.errors(b['m'],a['Q']@a['w'],b)
        rows.append({k:e[k] for k in ('replay','eid','uid','family','original_capture','fresh','limited','gap','age0','age1')}|dict(phase=phase,post_ticks=b['tick']-phase,scores=scores))
    def summ(rr):return {model:{k:A.metric([r['scores'][model][k] for r in rr]) for k in ('w_deg_s','n_deg','lambda_error_deg_s','transverse_deg_s','P_m')} for model in rows[0]['scores']}
    groups={'all_phase_eligible':rows,'original_phase_eligible':[r for r in rows if r['original_capture']],'R77_fresh_phase':[r for r in rows if r['family']=='R-77-1' and r['fresh']],'R77_fresh_unlimited_phase':[r for r in rows if r['family']=='R-77-1' and r['fresh'] and not r['limited']]}
    summary=dict(total=len(caps),eligible=len(rows),skipped=len(skipped),past_only_poison_checks=guard,parameters=dict(alphas=[.926,1.],tau=TAU,step=H),groups={k:summ(rr) for k,rr in groups.items()},anchors=[r for r in rows if (r['replay'],r['eid']) in A.ANCHORS])
    A.save(out/'capture_timing_events.json.gz',rows);A.save(out/'capture_phase_exclusions.json',skipped);A.save(out/'capture_timing_summary.json',summary)
    print('CAPTURE_COUNTS',json.dumps({k:summary[k] for k in ('total','eligible','skipped','past_only_poison_checks')}),flush=True)
    for name,v in summary['groups'].items():print('CAPTURE_GROUP',name,json.dumps({m:{k:z[k] for k in ('w_deg_s','n_deg','lambda_error_deg_s')} for m,z in v.items()}),flush=True)
    for r in summary['anchors']:print('CAPTURE_ANCHOR',json.dumps(A.serial({k:v for k,v in r.items() if k!='scores'}|{'scores':{m:{k:z[k] for k in ('w_deg_s','n_deg','lambda_error_deg_s','P_m')} for m,z in r['scores'].items()}})),flush=True)

def ordinary_pairs(series):
    rows=[];rejected=collections.Counter()
    for s in series:
        rr=s['rows']
        for aa,bb in zip(rr,rr[1:]):
            if aa is None or bb is None:rejected['unsupported']+=1;continue
            a,b=arr(aa),arr(bb);gap=b['tick']-a['tick'];T=gap/48
            if not(0<gap<=3):rejected['gap']+=1;continue
            if not(a['code']==b['code']==31 and a['timer']==b['timer']==0 and a['flag']==b['flag']==0):rejected['mode_timer_flag']+=1;continue
            ownerr=norm(b['p']-a['p']-(b['vm']+a['vm'])*T/2)
            if ownerr>=.5:rejected['own_motion_time']+=1;continue
            if b['R']<=0:rejected['range']+=1;continue
            r=dict(replay=s['replay'],eid=s['eid'],uid=s['info']['uid'],family=s['info']['family'],tick0=a['tick'],tick1=b['tick'],T=T,own_residual=ownerr,maxangle=max(abs(a['angles']).max(),abs(b['angles']).max())*A.RAD,p=b['p'],target=b['Ps'],R=b['R'],m=b['m'],n=b['n'],dw=b['w']-a['w'],vrel=b['Vs']-b['vm'],old_vrel=a['Vs']-b['vm'],prior_PV=A.unit(a['Ps']+a['Vs']*T-b['p']),prior_currentVs_DIAGNOSTIC=A.unit(a['Ps']+b['Vs']*T-b['p']),prior_angular=Rotation.from_rotvec(-a['w']*T).apply(a['n']))
            rows.append(r)
    return rows,dict(rejected)

def causality_task(series,out):
    rr,reject=ordinary_pairs(series);sets={'all_family_45':[r for r in rr if r['maxangle']<45],'R77_54p9':[r for r in rr if r['family']=='R-77-1' and r['maxangle']<54.9]};result=dict(rejected=reject,scope='five-scene endpoint field reconstruction. Models labelled DIAGNOSTIC use contemporaneous outputs. None identifies measurement causality.',groups={});predrows=[]
    for label,rows in sets.items():
        if not rows:continue
        arrs={k:np.array([r[k] for r in rows]) for k in ('p','target','R','m','n','dw','vrel','old_vrel','prior_PV','prior_currentVs_DIAGNOSTIC','prior_angular')};rep=np.array([r['replay'] for r in rows]);m=arrs['m'];R=arrs['R'];target=arrs['target'];p=arrs['p']
        priors={k:arrs[k] for k in ('prior_PV','prior_currentVs_DIAGNOSTIC','prior_angular')}
        models=['direct_pointing','prior_PV','prior_angular','prior_currentVs_DIAGNOSTIC','current_velocity_shift_DIAGNOSTIC','old_velocity_shift','delta_rate_output_relation_DIAGNOSTIC']
        def prediction(model,c):
            if model=='direct_pointing':n=m
            elif model in priors:n=sphere(priors[model],m,c)
            elif model=='delta_rate_output_relation_DIAGNOSTIC':n=Rotation.from_rotvec(-c*arrs['dw']).apply(m)
            else:
                v=arrs['vrel'] if model=='current_velocity_shift_DIAGNOSTIC' else arrs['old_vrel'];v=v-dot(v,m)[:,None]*m;n=un(m+c*v/R[:,None])
            return p+R[:,None]*n
        group=dict(count=len(rows),folds=[],models={});tot={model:np.zeros(len(rows)) for model in models};params=collections.defaultdict(list)
        for scene in sorted(set(rep)):
            test=rep==scene;train=~test
            if not train.any() or not test.any():continue
            fold=dict(test_scene=scene,train_count=int(train.sum()),test_count=int(test.sum()),models={})
            for model in models:
                if model=='direct_pointing':c=0.
                else:
                    bounds=(0.,1.5) if model in priors else (-.2,.2)
                    def objective(c):return float(np.mean(np.sum((prediction(model,c)[train]-target[train])**2,axis=-1)))
                    fit=minimize_scalar(objective,bounds=bounds,method='bounded',options={'xatol':1e-11});c=float(fit.x)
                er=norm(prediction(model,c)-target);tot[model][test]=er[test];params[model].append(c);fold['models'][model]=dict(parameter=c,P_error_m=A.metric(er[test]))
            group['folds'].append(fold)
        for model in models:group['models'][model]=dict(P_error_m=A.metric(tot[model]),fold_parameters=params[model])
        group['current_field_closure']=dict(P_Rn_m=A.metric(norm(p+R[:,None]*arrs['n']-target)))
        for i,r in enumerate(rows):predrows.append({k:r[k] for k in ('replay','eid','uid','family','tick0','tick1','T','own_residual','maxangle')}|dict(group=label,P_errors_m={model:tot[model][i] for model in models}))
        result['groups'][label]=group;print('CAUSALITY_GROUP',label,json.dumps(A.serial(group)),flush=True)
    A.save(out/'causality_summary.json',result);A.save(out/'causality_oof_rows.json.gz',predrows)

def verification_task(caps,out):
    rng=np.random.default_rng(4817);mx=0.;cases=0
    for _ in range(2000):
        n=A.unit(rng.normal(size=3));w=rng.normal(size=3);vm=rng.normal(size=3)*300;rad=rng.uniform(100,20000);d=rng.normal()*100
        v=vm-(vm@n)*n+d*n+rad*np.cross(n,w)
        wg=np.cross(v-vm,n)/rad
        mx=max(mx,float(np.max(np.abs(wg-(w-(w@n)*n)))));cases+=1
        a=.8;nn,ww=step(n,w,n,H,a)
        np.testing.assert_allclose(nn,A.blend(A.rotate(n,w,H),n,a),atol=1e-14,rtol=0)
    result=dict(random_geometric_identity_tests=cases,max_abs_rate_error=mx,warning='Algebra and interface tests do not identify a game mechanism.')
    assert mx<1e-12;A.save(out/'verification.json',result);print('VERIFICATION',json.dumps(result),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('task',choices=['frame','capture','causality','verification']);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--cohort',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    caps=load(args.evidence/'core/capture_events.json.gz');series=load(args.cohort/'cohort.json.gz')
    if args.task=='frame':frame_task(caps,series,args.out)
    elif args.task=='capture':capture_task(caps,args.out)
    elif args.task=='causality':causality_task(series,args.out)
    else:verification_task(caps,args.out)
if __name__=='__main__':main()
