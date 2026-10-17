#!/usr/bin/env python3
"""Exploratory, bounded War Thunder state-field tests and independent numerical checks."""
from __future__ import annotations
import argparse,collections,hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import linprog,minimize_scalar
from scipy.spatial.transform import Rotation
import refine as F
E,A=F.E,F.A

def read(root,rel):return E.load(root/rel)
def stats(rows,field):return {name:A.metric([r[field][name] for r in rows]) for name in rows[0][field]} if rows else {}

def equal_lp(M,k0,k1,kind):
    # Independently expressed as six absolute variables and exact M*x-y=0.
    bounds=[]
    for k in list(k0)+list(k1):
        if kind=='nearest':bounds.append((k-.5,k+.5))
        elif k>0:bounds.append((k,k+1))
        elif k<0:bounds.append((k-1,k))
        else:bounds.append((-1,1))
    z=linprog(np.zeros(6),A_eq=np.column_stack((M,-np.eye(3))),b_eq=np.zeros(3),bounds=bounds,method='highs')
    if z.success:assert np.linalg.norm(M@z.x[:3]-z.x[3:],ord=np.inf)<1e-6
    elif z.status!=2:raise RuntimeError(z.message)
    return bool(z.success)

def first_capture(caps,out):
    rows=[]
    for e in caps:
        if not e['original_capture']:continue
        a,b=E.arr(e['old']),E.arr(e['new']);covered=F.phase_after(a['tick'])<=b['tick']
        base=F.ident(e)|dict(phase1_in_bracket=covered,flag0=a['flag'],flag1=b['flag'],eligible=b['timer']==0 and a['flag']==b['flag']==0)
        if not base['eligible']:
            rows.append(base);continue
        n=b['n'];k0,k1=F.rateints(a['w']),F.rateints(b['w']);P=F.projector(n,.8)
        maps={'Qnew_copy':b['Q'],'Qnew_k08':P@b['Q'],'Qold_k08':P@a['Q']}
        errs={key:float(np.linalg.norm(M@k0-k1))*F.QSTEP*A.RAD for key,M in maps.items()}
        feas={key+'/'+enc:equal_lp(M,k0,k1,enc) for key,M in maps.items() for enc in ('truncate','nearest')}
        rows.append(base|dict(errors=errs,feasible=feas))
    assert len(rows)==501
    result={}
    for group,rr in [('all_first',rows),('phase1_first',[r for r in rows if r['phase1_in_bracket']]),('other_phase_first',[r for r in rows if not r['phase1_in_bracket']]),('R77_first',[r for r in rows if r['family']=='R-77-1']),('R77_other_phase_first',[r for r in rows if r['family']=='R-77-1' and not r['phase1_in_bracket']])]:
        eligible=[r for r in rr if r['eligible']]
        result[group]=dict(total=len(rr),scored=len(eligible),errors=stats(eligible,'errors'),feasible_counts={key:sum(r['feasible'][key] for r in eligible) for key in eligible[0]['feasible']} if eligible else {})
    A.save(out/'first_capture_summary.json',result);A.save(out/'first_capture_rows.json.gz',rows)
    print('FIRST_CAPTURE',json.dumps(A.serial(result)),flush=True)

def predict(n0,w0,m,T,model,value):
    g=1/(F.TAU+T/2)
    if model=='repeat_endpoint_ray':
        n=n0.copy();w=w0.copy();gaps=np.rint(48*T).astype(int)
        for j in range(int(gaps.max())):
            ix=gaps>j;p=Rotation.from_rotvec(-F.H*w[ix]).apply(n[ix]);nn=E.sphere(p,m[ix],value)
            c=np.cross(nn,p);l=E.norm(c);theta=np.arctan2(l,E.dot(nn,p));c*=np.divide(theta,l,out=np.ones_like(theta),where=l>1e-14)[:,None]
            w[ix]+=c/(F.TAU+F.H/2);n[ix]=nn
        return n,w
    p=Rotation.from_rotvec(-w0*T[:,None]).apply(n0)
    if model=='constant_alpha':a=np.full(len(T),value)
    elif model=='fractional_fixed_lag':a=1-value/T
    elif model=='rational_lag':a=T/(T+value)
    elif model=='implicit_velocity':a=value/(1-(1-value)*T*g)
    else:raise ValueError(model)
    nn=E.sphere(p,m,a[:,None]);c=np.cross(nn,p);l=E.norm(c);theta=np.arctan2(l,E.dot(nn,p));c*=np.divide(theta,l,out=np.ones_like(theta),where=l>1e-14)[:,None]
    return nn,w0+c*g[:,None]

def lag_test(series,out):
    pp,_=E.ordinary_pairs(series);rr=[r for r in pp if r['family']=='R-77-1' and r['maxangle']<54.9];assert len(rr)==5484
    lut={}
    for s in series:
        for a,b in zip(s['rows'],s['rows'][1:]):
            if a is not None and b is not None:lut[(s['replay'],s['eid'],a['tick'],b['tick'])]=(a,b)
    ab=[lut[(r['replay'],r['eid'],r['tick0'],r['tick1'])] for r in rr]
    n0=np.array([a['n'] for a,b in ab]);w0=np.array([a['w'] for a,b in ab]);nt=np.array([b['n'] for a,b in ab]);wt=np.array([b['w'] for a,b in ab]);T=np.array([r['T'] for r in rr]);m=np.array([r['m'] for r in rr]);p=np.array([r['p'] for r in rr]);R=np.array([r['R'] for r in rr]);Pt=np.array([r['target'] for r in rr]);reps=np.array([r['replay'] for r in rr]);gaps=np.rint(48*T).astype(int)
    models=['constant_alpha','fractional_fixed_lag','rational_lag','implicit_velocity','repeat_endpoint_ray'];scores={key:dict(n_deg=np.full(len(rr),np.nan),w_deg_s=np.full(len(rr),np.nan),P_m=np.full(len(rr),np.nan)) for key in models};folds=[]
    for rep in sorted(set(reps)):
        train=reps!=rep;test=~train;fold=dict(test_scene=rep,train=int(train.sum()),test=int(test.sum()),models={})
        for name in models:
            def loss(c):nn,_=predict(n0,w0,m,T,name,c);return float(np.mean(np.sum((nn[train]-nt[train])**2,axis=1)))
            if name=='repeat_endpoint_ray':
                grid=np.linspace(.3,1.,29);cost=np.array([loss(c) for c in grid]);j=int(np.argmin(cost));lo,hi=grid[max(0,j-1)],grid[min(28,j+1)]
                if lo==hi:c=float(grid[j])
                else:
                    res=minimize_scalar(loss,bounds=(lo,hi),method='bounded',options={'xatol':1e-10});c=float(min([(loss(float(grid[j])),float(grid[j])),(float(res.fun),float(res.x))],key=lambda x:x[0])[1])
                fold['models'][name]=dict(value=c,grid_best=float(grid[j]),grid_losses=cost.tolist())
            else:
                bounds=(0,.01) if 'lag' in name else (.3,.999999)
                res=minimize_scalar(loss,bounds=bounds,method='bounded',options={'xatol':1e-10});assert res.success;c=float(res.x)
                fold['models'][name]=dict(value=c)
            nn,ww=predict(n0,w0,m,T,name,c)
            scores[name]['n_deg'][test]=E.vang(nn[test],nt[test])*A.RAD;scores[name]['w_deg_s'][test]=E.norm(ww[test]-wt[test])*A.RAD;scores[name]['P_m'][test]=E.norm(p[test]+R[test,None]*nn[test]-Pt[test])
        folds.append(fold)
    strata={}
    for name,mask in [('all',np.ones(len(rr),bool))]+[(str(g),gaps==g) for g in (1,2,3)]:strata[name]={model:{key:A.metric(v[mask]) for key,v in vals.items()} for model,vals in scores.items()}
    result=dict(scope='Exploratory common-lag innovation weights with leave-scene-out scoring; pointing-conditioned, not independently measured target angles.',n=len(rr),folds=folds,strata=strata)
    rows=[{k:r[k] for k in ('replay','eid','uid','tick0','tick1')}|dict(gap=int(gaps[j]),scores={model:{key:float(v[j]) for key,v in vals.items()} for model,vals in scores.items()}) for j,r in enumerate(rr)]
    A.save(out/'lag_summary.json',result);A.save(out/'lag_rows.json.gz',rows)
    print('LAG_PARAMETERS',json.dumps([{k:f[k] for k in ('test_scene','train','test')}|{'models':{m:z['value'] for m,z in f['models'].items()}} for f in folds]),flush=True)
    print('LAG_STRATA',json.dumps(A.serial(strata)),flush=True)

def review(caps,hist,results,out):
    qrows=read(results/'quant','quantization_rows.json.gz');ph=read(results/'phase','phase_summary.json')
    checks=[];lpchecked=0
    for r in qrows:
        n,q=np.array(r['n']),np.array(r['Q']);M=.8*q+.2*np.outer(n,n)@q
        for enc in ('truncate','nearest'):
            expected=r['slack_cells']['post_Q_k08/'+enc]<=1e-7;actual=equal_lp(M,np.array(r['rate0_int']),np.array(r['rate1_int']),enc)
            assert actual==expected,(r['replay'],r['eid'],r['tick1'],enc);lpchecked+=1
    for group,pred in ph['groups'].items():
        selected=[]
        for e in caps:
            if group=='original' and not e['original_capture']:continue
            if group=='repeated' and e['original_capture']:continue
            if group=='R77_original' and not(e['family']=='R-77-1' and e['original_capture']):continue
            if group=='R77_repeated' and not(e['family']=='R-77-1' and not e['original_capture']):continue
            selected.append(e)
        assert len(selected)==pred['n']
        for phase in range(48):
            cnt=sum(any(t%48==phase for t in range(e['tick0']+1,e['tick1']+1)) for e in selected)
            assert cnt==pred['phase_coverage'][str(phase)];checks.append((group,phase,cnt))
    # Correct the diagnostic loss bookkeeping, including transitions via code29.
    fixed={}
    for uid,rr in hist.items():
        exits=[];last_exit=None
        for a,b in zip(rr,rr[1:]):
            if a['code']==31 and b['code']!=31:last_exit=b['tick'];exits.append(dict(kind='leave31',tick0=a['tick'],tick1=b['tick'],code1=b['code'],timer0=a['timer'],timer1=b['timer']))
            if a['code']!=31 and b['code']==31:exits.append(dict(kind='return31',tick0=a['tick'],tick1=b['tick'],code0=a['code'],timer0=a['timer'],timer1=b['timer'],ticks_since_last_leave31=b['tick']-last_exit if last_exit is not None else None))
        fixed[uid]=exits
    # Six quantization exceptions: independent own-clock diagnostic, no exclusion.
    bykey={(e['replay'],e['eid'],e['tick1']):e for e in caps};bad=[]
    for r in qrows:
        if r['family']!='R-77-1' or r['slack_cells']['post_Q_k08/truncate']<=1e-7:continue
        e=bykey[(r['replay'],r['eid'],r['tick1'])];a,b=E.arr(e['old']),E.arr(e['new']);T=e['gap']/48;dp=b['p']-a['p'];v=(a['vm']+b['vm'])/2
        bad.append(F.ident(e)|dict(extra_output_cells=r['slack_cells']['post_Q_k08/truncate'],own_displacement_error_T_m=float(np.linalg.norm(dp-T*v)),own_time_ratio=float(dp@v/(v@v*T)),Q_rotation_deg=float(np.linalg.norm(Rotation.from_matrix(b['Q']@a['Q'].T).as_rotvec()))*A.RAD))
    report=dict(independent_equality_LPs=lpchecked,phase_count_comparisons=len(checks),all_checks_passed=True,conditional_target_projection=True,rate_encoding_hypotheses_not_source_identification=True,six_R77_exceptions=bad,loss_bookkeeping_correction='Original phase-summary ticks_since_previous_loss omitted intermediate code29; corrected file records each departure from code31. Capture phase histograms and map scores are unaffected.')
    A.save(out/'independent_review.json',report);A.save(out/'anchor_loss_boundaries_corrected.json',fixed)
    print('INDEPENDENT_REVIEW',json.dumps(A.serial(report)),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--results',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    base=a.input/'data/main';caps=read(base,'evidence/capture_events.json.gz');series=read(base,'input/cohort.json.gz');hist=read(base,'evidence/anchor_full_histories.json.gz');assert len(caps)==931
    for root in ('quant','phase','temporal'):
        ss=read(a.results/root,'sources.json')
        for rel,h in ss.items():assert hashlib.sha256((a.input/rel).read_bytes()).hexdigest()==h,(root,rel)
    review(caps,hist,a.results,a.out);first_capture(caps,a.out);lag_test(series,a.out)
    A.save(a.out/'sources.json',{str(p.relative_to(a.input)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [base/'evidence/capture_events.json.gz',base/'evidence/anchor_full_histories.json.gz',base/'input/cohort.json.gz']})

if __name__=='__main__':main()
