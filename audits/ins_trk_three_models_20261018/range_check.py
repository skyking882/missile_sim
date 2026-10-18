#!/usr/bin/env python3
"""Offline comparison of serialized War Thunder replay vectors; no runtime edits.
Current n and range are expressly conditional inputs. Current target PV and
response rate are excluded from candidate construction. See PROTOCOL.md.
"""
from __future__ import annotations
import argparse, gzip, hashlib, json, math, sys
from pathlib import Path
import numpy as np
from scipy.optimize import linprog

QSTEP=15/32767
DEG=180/math.pi
TOL=1e-7

def read(path):
    with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as f:return json.load(f)

def serial(x):
    if isinstance(x,np.ndarray):return serial(x.tolist())
    if isinstance(x,np.generic):return serial(x.item())
    if isinstance(x,dict):return {str(k):serial(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [serial(v) for v in x]
    if isinstance(x,float) and not math.isfinite(x):return None
    return x

def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    with (gzip.open(path,'wt') if path.suffix=='.gz' else path.open('w')) as f:json.dump(serial(obj),f,ensure_ascii=False,separators=(',',':'),allow_nan=False)

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def verify(root):
    rows=read(root/'MANIFEST.json')
    for r in rows:
        p=root/r['path']
        assert p.stat().st_size==r['bytes'] and digest(p)==r['sha256'],str(p)
    return len(rows)

def vec(x):return np.asarray(x,dtype=float)
def D(n,k=.8):return k*np.eye(3)+(1-k)*np.outer(n,n)
def near(e):return e['tick1']==((e['tick0']-1)//48+1)*48+1 and e['new']['timer']==0 and e['old']['flag']==e['new']['flag']==0

def cells(k):
    k=vec(k)
    return k-np.where(k<=0,1.,0.),k+np.where(k>=0,1.,0.)
def ints(w):
    x=vec(w)/QSTEP;k=np.rint(x)
    assert np.max(np.abs(x-k))<1e-8
    return k

def lp(c,**kwargs):
    fit=linprog(c,method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9},**kwargs)
    if fit.status not in (0,2,3):raise RuntimeError(fit.message)
    return fit

def slack(M,b,k0,k1):
    l,u=cells(k0);ll,uu=cells(k1)
    fit=lp([0,0,0,1],A_ub=np.vstack([np.c_[M,-np.ones(3)],np.c_[-M,-np.ones(3)]]),b_ub=np.r_[uu-b,-ll+b],bounds=list(zip(l,u))+[(0,None)])
    if not fit.success:raise RuntimeError(fit.message)
    y=M@fit.x[:3]+b;t=max(0.,float(fit.x[3]))
    assert max(float(np.max(ll-y)),float(np.max(y-uu)))<=t+1e-6
    return t

def scale_constraints(M,k0,k1,equality=False):
    l,u=cells(k0);ll,uu=cells(k1)
    if not equality:
        mat=np.vstack([np.c_[np.eye(3),-u],np.c_[-np.eye(3),l],np.c_[M,np.zeros(3)],np.c_[-M,np.zeros(3)]])
        return dict(A_ub=mat,b_ub=np.r_[np.zeros(6),uu,-ll],bounds=[(None,None)]*3+[(0,None)])
    mat=np.zeros((6,7));mat[:3,:3]=np.eye(3);mat[:3,3]=-u;mat[3:,:3]=-np.eye(3);mat[3:,3]=l
    eq=np.zeros((3,7));eq[:,:3]=M;eq[:,4:]=-np.eye(3)
    return dict(A_ub=mat,b_ub=np.zeros(6),A_eq=eq,b_eq=np.zeros(3),bounds=[(None,None)]*3+[(0,None)]+list(zip(ll,uu)))

def scaling_diagnostic(M,k0,k1):
    arg=scale_constraints(M,k0,k1);fit=lp([0,0,0,0],**arg)
    other=lp(np.zeros(7),**scale_constraints(M,k0,k1,True))
    assert fit.success==other.success,(fit.message,other.message)
    if not fit.success:
        assert fit.status==other.status==2
        return dict(feasible=False,rho_min=None,rho_max=None,independent_equality_match=True)
    mn=lp([0,0,0,1],**arg);mx=lp([0,0,0,-1],**arg)
    assert mn.success and (mx.success or mx.status==3)
    out=dict(feasible=True,rho_min=float(mn.x[3]),rho_max=float(mx.x[3]) if mx.success else None,rho_upper_unbounded=mx.status==3,independent_equality_match=True)
    if mx.success and mx.x[3]<=0:out['feasible']=False
    return out

def public_new(new):
    return {k:new[k] for k in ('p','vm','Q','n','R','angles')}

def range_candidates(old,new,T):
    n=vec(new['n']);Q=vec(new['Q']);base=D(n)@Q
    r0=float(old['R']);r1=float(new['R']);rg=float(np.linalg.norm(vec(old['Ps'])-vec(old['p'])))
    rp=float(np.linalg.norm(vec(old['Ps'])+vec(old['Vs'])*T-vec(new['p'])))
    pairs={'cache_new':(r0,r1),'geometry_new':(rg,r1),'predicted_new':(rp,r1),'cache_geometry':(r0,rg)}
    out={};invalid={};ratios={}
    for label,(a,b) in pairs.items():
        ok=min(a,b)>1e-6 and math.isfinite(a+b)
        for direction,rho in [('ratio',a/b if ok else 1.),('reciprocal',b/a if ok else 1.)]:
            key=label+'/'+direction;ratios[key]=rho
            out[key+'/whole']=(rho*base,np.zeros(3))
            out[key+'/transverse']=(D(n,.8*rho)@Q,np.zeros(3))
            for suffix in ('whole','transverse'):invalid[key+'/'+suffix]=None if ok else 'invalid_range_baseline_fallback'
    return out,invalid,ratios

def metric(x):
    a=np.asarray(x,float)
    if not len(a):return {'n':0}
    return dict(n=len(a),rms=float(np.sqrt(np.mean(a*a))),median=float(np.median(a)),p95=float(np.percentile(a,95)),max=float(np.max(a)))

def cohorts(events):
    n=np.array([near(e) for e in events]);r=np.array([e['family']=='R-77-1' for e in events]);fresh=np.array([e['new']['timer']==0 and e['old']['flag']==e['new']['flag']==0 for e in events]);first=np.array([e['original_capture'] for e in events])
    return {'near244':n,'near_R77_79':n&r,'all931':np.ones(len(events),bool),'all_R77':r,'first_all':first,'reacquisition_all':~first,'fresh_flag00':fresh,'fresh_flag00_R77':fresh&r,'later_than_phase1':np.array([e['tick1']>((e['tick0']-1)//48+1)*48+1 for e in events])&fresh,'near_first_R77':n&r&first,'near_reacquisition_R77':n&r&~first}

def evaluate_family(events,builder,name,fixed_weight=1.):
    N=len(events);groups=cohorts(events);primary=groups['near244'];scenes=np.array([e['replay'] for e in events]);families=np.array([e['family'] for e in events]);n=vec([e['new']['n'] for e in events]);x=vec([e['old']['w'] for e in events]);y=vec([e['new']['w'] for e in events]);M0=np.array([D(nn)@vec(e['new']['Q']) for nn,e in zip(n,events)]);b0=np.zeros((N,3));p0=np.einsum('nij,nj->ni',M0,x)
    matrices={};biases={};invalid={};aux=[];guards=0
    for i,e in enumerate(events):
        old=e['old'];new=public_new(e['new']);T=(e['tick1']-e['tick0'])/48
        candidates,bad,extra=builder(old,new,T)
        poison=dict(new,Ps=np.full(3,np.nan),Vs=np.full(3,np.nan),w=np.full(3,np.nan))
        check,_,_=builder(old,poison,T)
        for key,(m,b) in candidates.items():
            np.testing.assert_array_equal(m,check[key][0]);np.testing.assert_array_equal(b,check[key][1])
            if key not in matrices:matrices[key]=np.empty((N,3,3));biases[key]=np.empty((N,3));invalid[key]=[]
            matrices[key][i]=m;biases[key][i]=b;invalid[key].append(bad.get(key))
        aux.append(extra);guards+=1
    names=sorted(matrices);rawpred={key:np.einsum('nij,nj->ni',matrices[key],x)+biases[key] for key in names}
    fitted={key:p0.copy() for key in names};beta_by={key:np.zeros(N) for key in names};chosen=p0.copy();chosenM=M0.copy();chosenb=b0.copy();choice=['baseline']*N;weights=np.zeros(N);folds=[]
    for scene in sorted(set(scenes)):
        for fam in sorted(set(families[scenes==scene])):
            test=(scenes==scene)&(families==fam);train=primary&(scenes!=scene)&(families==fam);fallback=int(train.sum())<10
            if fallback:train=primary&(scenes!=scene)
            assert train.any() and not np.any(train&(scenes==scene))
            loss0=float(np.sum((p0[train]-y[train])**2));best=(loss0,'baseline',0.);fits={}
            for key in names:
                delta=rawpred[key]-p0;den=float(np.sum(delta[train]**2));beta=float(np.clip(np.sum(delta[train]*(y[train]-p0[train]))/den,0,1)) if den>1e-30 else 0.
                pred=p0+beta*delta;loss=float(np.sum((pred[train]-y[train])**2));fits[key]=dict(beta=beta,training_loss=loss)
                fitted[key][test]=pred[test];beta_by[key][test]=beta
                if loss<best[0]-1e-20:best=(loss,key,beta)
            _,key,beta=best
            if key!='baseline':
                chosen[test]=fitted[key][test];chosenM[test]=M0[test]+beta*(matrices[key][test]-M0[test]);chosenb[test]=beta*biases[key][test]
            ids=np.where(test)[0]
            for i in ids:choice[i]=key
            weights[test]=beta
            folds.append(dict(held_scene=scene,target_family=fam,training_near_rows=int(train.sum()),test_rows=int(test.sum()),other_family_training_fallback=fallback,selected=key,selected_beta=beta,baseline_training_loss=loss0,candidates=fits))
    models={'baseline':(p0,M0,b0,np.ones(N)),'selected_LOSO':(chosen,chosenM,chosenb,weights)}
    for key in names:
        models['fixed/'+key]=(p0+fixed_weight*(rawpred[key]-p0),M0+fixed_weight*(matrices[key]-M0),fixed_weight*biases[key],np.full(N,fixed_weight))
        beta=beta_by[key];models['fit/'+key]=(fitted[key],M0+beta[:,None,None]*(matrices[key]-M0),beta[:,None]*biases[key],beta)
    k0=vec([ints(a) for a in x]);k1=vec([ints(a) for a in y]);data={};base_sl=np.array([slack(M0[i],np.zeros(3),k0[i],k1[i]) for i in range(N)])
    for key,(pred,mm,bb,ww) in models.items():
        error=pred-y;full=np.linalg.norm(error,axis=1)*DEG;ax=np.sum(error*n,axis=1)*DEG;trans=np.linalg.norm(error-(ax/DEG)[:,None]*n,axis=1)*DEG
        calc=np.ones(N,bool) if key in ('baseline','selected_LOSO') else primary
        sl=np.full(N,np.nan)
        for i in np.where(calc)[0]:sl[i]=base_sl[i] if key=='baseline' else slack(mm[i],bb[i]/QSTEP,k0[i],k1[i])
        data[key]=dict(prediction_rad_s=pred,rate_deg_s=full,axial_deg_s=ax,transverse_deg_s=trans,slack_cells=sl,coefficient=ww)
    summary={}
    for group,mask in groups.items():
        summary[group]={}
        for key,r in data.items():
            good=mask&np.isfinite(r['slack_cells']);entry={v:metric(r[v][mask]) for v in ('rate_deg_s','axial_deg_s','transverse_deg_s')}
            entry.update(cell_checked=int(good.sum()),cell_feasible=int(np.sum(r['slack_cells'][good]<=TOL)),repaired_baseline_failures=int(np.sum(good&(base_sl>TOL)&(r['slack_cells']<=TOL))),introduced_failures=int(np.sum(good&(base_sl<=TOL)&(r['slack_cells']>TOL))))
            summary[group][key]=entry
    pinned=groups['near_R77_79']&(base_sl>TOL)
    assert pinned.sum()==6 and np.sum(groups['near_R77_79']&(base_sl<=TOL))==73
    assert abs(metric(data['baseline']['rate_deg_s'][groups['near_R77_79']])['rms']-.028389140426290316)<1e-10
    rows=[]
    for i,e in enumerate(events):
        row={k:e[k] for k in ('replay','eid','uid','family','tick0','tick1','gap','fresh','limited','original_capture')}
        row.update(near=bool(primary[i]),selected=choice[i],selected_weight=weights[i],auxiliary_inputs=aux[i],domain_fallbacks={key:invalid[key][i] for key in names if invalid[key][i]},old_record_sha=e['old']['raw_sha'],new_record_sha=e['new']['raw_sha'],scores={key:{field:value[i] for field,value in r.items()} for key,r in data.items()})
        rows.append(row)
    result=dict(family=name,scope='Conditional current n and range; no current target Ps/Vs or response rate in candidate builders. Fitting uses other scenes only; repeated historical data, not new blind replays.',counts=dict(events=N,near=int(primary.sum()),R77_near=int(groups['near_R77_79'].sum()),input_isolation_checks=guards),groups=summary,folds=folds,domain_fallback_counts={key:sum(v is not None for v in invalid[key]) for key in names},six_cases=[rows[i] for i in np.where(pinned)[0]])
    return result,rows,base_sl

def run(root,out):
    out.mkdir(parents=True,exist_ok=False);checked=verify(root)
    source=root/'prior/evidence/data/main/evidence/capture_events.json.gz';events=read(source);assert len(events)==931
    result,rows,base_sl=evaluate_family(events,range_candidates,'range',1.)
    diagnostics=[]
    for i,e in enumerate(events):
        n=vec(e['new']['n']);M=D(n)@vec(e['new']['Q']);x=vec(e['old']['w']);y=vec(e['new']['w']);p=M@x;s=float(p@p)
        rho=max(0.,float(p@y/s)) if s>1e-30 else 0.;shape=float(np.linalg.norm(y-rho*p))*DEG
        sc=scaling_diagnostic(M,ints(x),ints(y))
        diagnostics.append({k:e[k] for k in ('replay','eid','uid','family','tick0','tick1','original_capture')}|dict(near=near(e),baseline_feasible=base_sl[i]<=TOL,decoded_best_rho=rho,decoded_shape_error_deg_s=shape,**sc))
    groups=cohorts(events);ds={}
    for group,mask in groups.items():
        rr=[r for r,b in zip(diagnostics,mask) if b];ds[group]=dict(n=len(rr),baseline_feasible=sum(r['baseline_feasible'] for r in rr),arbitrary_scale_feasible=sum(r['feasible'] for r in rr),rescued=sum(not r['baseline_feasible'] and r['feasible'] for r in rr),shape_error=metric([r['decoded_shape_error_deg_s'] for r in rr]))
    result['scaling_diagnostic']=ds;result['six_scaling_diagnostics']=[r for r in diagnostics if r['near'] and r['family']=='R-77-1' and not r['baseline_feasible']]
    result['provenance']=dict(input_manifest_files_verified=checked,input_capture_sha256=digest(source),code_sha256=digest(Path(__file__)),python=sys.version,numpy=np.__version__)
    save(out/'summary.json',result);save(out/'rows.json.gz',rows);save(out/'scaling_diagnostics.json.gz',diagnostics)
    compact=dict(counts=result['counts'],scaling=ds,six_scaling=result['six_scaling_diagnostics'],selected_scores={g:{k:v for k,v in ss.items() if k in ('baseline','selected_LOSO')} for g,ss in result['groups'].items()},six_selected=[{k:r[k] for k in ('replay','uid','tick0','tick1','selected','selected_weight','auxiliary_inputs')}|{'scores':{k:r['scores'][k] for k in ('baseline','selected_LOSO')}} for r in result['six_cases']],R77_folds=[r for r in result['folds'] if r['target_family']=='R-77-1'],fixed_near_R77=result['groups']['near_R77_79'])
    save(out/'compact.json',compact)
    for key,value in compact.items():print('RANGE_RESULT',key,json.dumps(serial(value),ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.input,a.out)
