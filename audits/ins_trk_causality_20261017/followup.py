#!/usr/bin/env python3
"""Exploratory follow-up to four completed replay-identification tracks."""
from __future__ import annotations
import argparse,copy,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import explore as E
A=E.A;H=E.H;TAU=E.TAU

def kphase(a):return ((a['tick']-1)//48+1)*48+1

def reset_predict(a,b,kappa,rate_source,qsource,direction_source):
    T=(b['tick']-a['tick'])/48
    prior=A.unit(a['Ps']+a['Vs']*T-b['p'])
    if rate_source=='stored':w=a['w']
    elif rate_source=='old_geometry':
        r=a['Ps']-a['p'];w=np.cross(a['Vs']-a['vm'],A.unit(r))/np.linalg.norm(r)
    elif rate_source=='geometry_at_endpoint':
        r=a['Ps']+a['Vs']*T-b['p'];w=np.cross(a['Vs']-b['vm'],A.unit(r))/np.linalg.norm(r)
    else:raise ValueError(rate_source)
    q=a['Q'] if qsource=='Qold' else b['Q'];s=q@w
    if direction_source=='current_n_DIAGNOSTIC':n=b['n']
    elif direction_source=='pointing_CONDITIONAL':n=b['Q']@A.ray(b['angles'])
    elif direction_source=='past_PV':n=prior
    else:raise ValueError(direction_source)
    para=(s@n)*n;perp=s-para
    return n,para+kappa*perp,para,perp

def contracted_capture(a,b,phase,path,qsource,kappa):
    gap=b['tick']-a['tick'];t0,t1=a['tick'],b['tick']
    def m(t):
        f=(t-t0)/gap;own=(1-f)*a['p']+f*b['p']
        target=a['Ps']+a['Vs']*((t-t0)/48) if path=='past_PV' else b['Ps']+b['Vs']*((t-t1)/48)
        return A.unit(target-own)
    f=(phase-t0)/gap
    q=a['Q'] if qsource=='Qold' else Rotation.from_rotvec(Rotation.from_matrix(b['Q']@a['Q'].T).as_rotvec()*f).as_matrix()@a['Q']
    n=m(phase);s=q@a['w'];w=kappa*s+(1-kappa)*(s@n)*n
    for t in range(phase+1,t1+1):n,w=E.step(n,w,m(t),H,.926)
    return n,w

def reset_task(root,out):
    events=E.load(root/'evidence/capture_events.json.gz');kap={'copy':1.,'retention_0p8':.8,'self_feedback_a0p926':1-.926*H/(TAU+H/2),'self_feedback_a1':1-H/(TAU+H/2)}
    rows=[];paired=[];poisonchecks=0
    for e in events:
        a,b=E.arr(e['old']),E.arr(e['new'])
        phase=kphase(a)
        if b['tick']!=phase or b['timer']!=0 or a['flag']!=0 or b['flag']!=0:continue
        scores={};basis={}
        for src in ('stored','old_geometry','geometry_at_endpoint'):
            for q in ('Qold','Qnew'):
                for ns in ('current_n_DIAGNOSTIC','pointing_CONDITIONAL','past_PV'):
                    key=f'{src}/{q}/{ns}'
                    n,w,para,perp=reset_predict(a,b,.8,src,q,ns);basis[key]=(para,perp)
                    for label,k in kap.items():scores[key+'/'+label]=A.errors(n,para+k*perp,b)
                    if ns!='current_n_DIAGNOSTIC':
                        poisoned=copy.deepcopy(b)
                        for k in ('Ps','Vs','n','w'):poisoned[k]=np.full(3,np.nan)
                        n2,w2,_,_=reset_predict(a,poisoned,.8,src,q,ns)
                        np.testing.assert_array_equal(n,n2);np.testing.assert_array_equal(w,w2);poisonchecks+=1
        base={k:e[k] for k in ('replay','eid','uid','family','original_capture','fresh','limited','gap','age0','age1')}
        rows.append(dict(**base,scores=scores));paired.append(dict(**base,basis=basis,target=b['w']))
    summaries={};fitrows=[]
    for name,indices in [('all',list(range(len(rows)))),('R77',[i for i,r in enumerate(rows) if r['family']=='R-77-1']),('R77_unlimited',[i for i,r in enumerate(rows) if r['family']=='R-77-1' and not r['limited']])]:
        selected=[rows[i] for i in indices]
        if not selected:continue
        model_summary={model:{k:A.metric([r['scores'][model][k] for r in selected]) for k in ('w_deg_s','lambda_error_deg_s','transverse_deg_s','n_deg')} for model in selected[0]['scores']}
        fits={}
        for src in ('stored','old_geometry','geometry_at_endpoint'):
            for q in ('Qold','Qnew'):
                model=f'{src}/{q}/current_n_DIAGNOSTIC';es=[];ks=[];folds=[]
                for scene in sorted({rows[i]['replay'] for i in indices}):
                    train=[i for i in indices if rows[i]['replay']!=scene];test=[i for i in indices if rows[i]['replay']==scene]
                    den=sum(float(paired[i]['basis'][model][1]@paired[i]['basis'][model][1]) for i in train)
                    if den<=1e-15:continue
                    k=sum(float(paired[i]['basis'][model][1]@(paired[i]['target']-paired[i]['basis'][model][0])) for i in train)/den
                    err=[]
                    for i in test:
                        para,perp=paired[i]['basis'][model];er=float(np.linalg.norm(para+k*perp-paired[i]['target']))*A.RAD;es.append(er);err.append(er)
                        fitrows.append(dict(group=name,model=model,test_scene=scene,replay=rows[i]['replay'],eid=rows[i]['eid'],uid=rows[i]['uid'],age1=rows[i]['age1'],kappa_train=k,error_deg_s=er))
                    ks.append(k);folds.append(dict(scene=scene,train=len(train),test=len(test),kappa=k,error=A.metric(err)))
                fits[model]=dict(kappa_folds=ks,error_deg_s=A.metric(es),folds=folds)
        summaries[name]=dict(n=len(selected),models=model_summary,fitted=fits)
        picked={k:v for k,v in model_summary.items() if '/current_n_DIAGNOSTIC/' in k and ('/Qnew/' in k or '/stored/' in k)}
        print('RESET_SUMMARY',name,json.dumps(A.serial(dict(n=len(selected),models=picked,fitted=fits))),flush=True)
    mult=[]
    for e in events:
        a,b=E.arr(e['old']),E.arr(e['new']);phase=kphase(a)
        if not a['tick']<phase<=b['tick']:continue
        scores={}
        for path in ('past_PV','endpoint_PV_DIAGNOSTIC'):
            for q in ('Qold','Qphase'):
                n,w=contracted_capture(a,b,phase,path,q,.8);scores[path+'/'+q+'/retention_0p8']=A.errors(n,w,b)
        mult.append({k:e[k] for k in ('replay','eid','uid','family','original_capture','fresh','limited','gap','age0','age1')}|dict(post_ticks=b['tick']-phase,scores=scores))
    multi_summary={}
    for name,rr in [('all',mult),('R77_fresh',[r for r in mult if r['family']=='R-77-1' and r['fresh']])]:
        multi_summary[name]={model:{k:A.metric([r['scores'][model][k] for r in rr]) for k in ('w_deg_s','n_deg','lambda_error_deg_s','P_m')} for model in mult[0]['scores']}
    summary=dict(scope='Exploratory contraction test, current-n projections explicitly diagnostic; zero-post-step phase is a hypothesis',fixed_kappas=kap,zero_phase_events=len(rows),past_or_pointing_label_isolation_checks=poisonchecks,groups=summaries,multistep=multi_summary,anchors_zero=[r for r in rows if (r['replay'],r['eid']) in A.ANCHORS],anchors_multi=[r for r in mult if (r['replay'],r['eid']) in A.ANCHORS])
    A.save(out/'reset_summary.json',summary);A.save(out/'reset_events.json.gz',rows);A.save(out/'reset_fitted_oof.json.gz',fitrows);A.save(out/'contracted_multistep_events.json.gz',mult)
    print('MULTISTEP_RESET',json.dumps(A.serial(multi_summary)),flush=True)
    for r in summary['anchors_multi'][:3]:print('CONTRACTED_ANCHOR',json.dumps(A.serial(r)),flush=True)
    for r in summary['anchors_zero'][:2]:
        picked={k:v for k,v in r['scores'].items() if '/current_n_DIAGNOSTIC/retention_0p8' in k or '/past_PV/retention_0p8' in k}
        print('ZERO_ANCHOR',json.dumps(A.serial({k:r[k] for k in ('uid','age1','gap')}|dict(scores=picked))),flush=True)

def update_from_n(n,pang,wold,vm,R,d,T):
    v=np.cross(n,pang);vl=E.norm(v);th=np.arctan2(vl,E.dot(n,pang));c=v*np.divide(th,vl,out=np.ones_like(th),where=vl>1e-14)[:,None]
    w=wold+c/(TAU+T/2)[:,None]
    V=vm-E.dot(vm,n)[:,None]*n+d[:,None]*n+R[:,None]*np.cross(n,w)
    return w,V

def solve(inputs,alpha):
    P0,n0,w0,p,vm,R,d,T,m=(inputs[k] for k in ('P0','n0','w0','p','vm','R','d','T','m'))
    pang=Rotation.from_rotvec(-w0*T[:,None]).apply(n0);n=E.sphere(pang,m,.926)
    for _ in range(50):
        w,V=update_from_n(n,pang,w0,vm,R,d,T)
        n=E.sphere(E.un(P0+T[:,None]*V-p),m,alpha[:,None])
    w,V=update_from_n(n,pang,w0,vm,R,d,T)
    rhs=E.sphere(E.un(P0+T[:,None]*V-p),m,alpha[:,None]);res=E.vang(rhs,n)
    return p+R[:,None]*n,n,w,V,res

def implicit_task(root,out):
    series=E.load(root/'input/cohort.json.gz');cs=E.load(next((root/'causality').rglob('causality_summary.json')))
    pairs=[]
    for s in series:
        for old,new in zip(s['rows'],s['rows'][1:]):
            if old is None or new is None:continue
            a,b=E.arr(old),E.arr(new);gap=b['tick']-a['tick'];T=gap/48
            if not(0<gap<=3 and a['code']==b['code']==31 and a['timer']==b['timer']==0 and a['flag']==b['flag']==0 and b['R']>0):continue
            ownerr=np.linalg.norm(b['p']-a['p']-(b['vm']+a['vm'])*T/2)
            if ownerr>=.5:continue
            maxangle=max(abs(a['angles']).max(),abs(b['angles']).max())*A.RAD
            inp=dict(P0=a['Ps'],n0=a['n'],w0=a['w'],p=b['p'],vm=b['vm'],R=b['R'],d=b['d'],T=T,m=b['m'])
            target=dict(P=b['Ps'],n=b['n'],w=b['w'],V=b['Vs'])
            pairs.append(dict(replay=s['replay'],eid=s['eid'],uid=s['info']['uid'],family=s['info']['family'],tick0=a['tick'],tick1=b['tick'],maxangle=maxangle,inputs=inp,labels=target))
    result=dict(scope='Endpoint-conditioned implicit state reconstruction, not autonomous measurement generation; alpha fixed from other-scene fits, current Ps/Vs/n/w not input to solver',groups={});allrows=[]
    for group,rr in [('all_family_45',[r for r in pairs if r['maxangle']<45]),('R77_54p9',[r for r in pairs if r['family']=='R-77-1' and r['maxangle']<54.9])]:
        assert len(rr)==cs['groups'][group]['count']
        inputs={k:np.array([r['inputs'][k] for r in rr]) for k in rr[0]['inputs']};labels={k:np.array([r['labels'][k] for r in rr]) for k in rr[0]['labels']}
        params={f['test_scene']:f['models']['prior_currentVs_DIAGNOSTIC']['parameter'] for f in cs['groups'][group]['folds']};alpha=np.array([params[r['replay']] for r in rr])
        P,n,w,V,res=solve(inputs,alpha)
        # Poisoned output object is deliberately not passed across the solver's input boundary.
        poisoned={k:np.full_like(v,np.nan) for k,v in labels.items()};P2,n2,w2,V2,res2=solve({k:v.copy() for k,v in inputs.items()},alpha.copy())
        for x,y in zip((P,n,w,V,res),(P2,n2,w2,V2,res2)):np.testing.assert_array_equal(x,y)
        errs=dict(P_m=E.norm(P-labels['P']),n_deg=E.vang(n,labels['n'])*A.RAD,w_deg_s=E.norm(w-labels['w'])*A.RAD,V_mps=E.norm(V-labels['V']),fixed_point_residual_rad=res)
        g=dict(count=len(rr),alpha_by_test_scene=params,metrics={k:A.metric(v) for k,v in errs.items()},not_converged_1e_10_rad=int((res>1e-10).sum()),same_sample_baselines={k:cs['groups'][group]['models'][k] for k in ('direct_pointing','prior_PV','prior_angular','prior_currentVs_DIAGNOSTIC')})
        result['groups'][group]=g
        for i,r in enumerate(rr):allrows.append({k:r[k] for k in ('replay','eid','uid','family','tick0','tick1','maxangle')}|dict(group=group,alpha=alpha[i],**{k:v[i] for k,v in errs.items()}))
        print('IMPLICIT_GROUP',group,json.dumps(A.serial(g)),flush=True)
    A.save(out/'implicit_summary.json',result);A.save(out/'implicit_rows.json.gz',allrows)

def main():
    p=argparse.ArgumentParser();p.add_argument('task',choices=['reset','implicit']);p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    if args.task=='reset':reset_task(args.input,args.out)
    else:implicit_task(args.input,args.out)
if __name__=='__main__':main()
