#!/usr/bin/env python3
"""Offline War Thunder replay refinements. No production simulator edits/imports."""
from __future__ import annotations
import argparse,collections,hashlib,json,math,sys,unittest
from pathlib import Path
import numpy as np
from scipy.optimize import linprog,minimize_scalar
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ins_trk_causality_20261017'))
import explore as E
A=E.A;QSTEP=15/32767;H=1/48;TAU=.078525

def phase_after(t,phase=1):return ((t-phase)//48+1)*48+phase

def cell(k,kind):
    k=np.asarray(k,float)
    if kind=='nearest':return k-.5,k+.5
    if kind!='truncate':raise ValueError(kind)
    return k-np.where(k<=0,1.,0.),k+np.where(k>=0,1.,0.)

def midpoint(k,kind):
    l,u=cell(k,kind);return (l+u)/2

def rateints(w):
    z=np.asarray(w)/QSTEP;k=np.rint(z)
    if np.max(np.abs(z-k))>1e-8:raise ValueError('Rate not on declared integer grid')
    return k

def projector(n,k):return k*np.eye(3)+(1-k)*np.outer(n,n)

def slack_lp(M,k0,k1,kind):
    l0,u0=cell(k0,kind);l1,u1=cell(k1,kind)
    r=M@k0-k1
    B=np.vstack([np.column_stack([M,-np.ones(3)]),np.column_stack([-M,-np.ones(3)])])
    rhs=np.r_[u1-k1-r,-(l1-k1)+r]
    fit=linprog([0,0,0,1],A_ub=B,b_ub=rhs,bounds=list(zip(l0-k0,u0-k0))+[(0,None)],method='highs')
    if not fit.success:raise RuntimeError(fit.message)
    z=fit.x[:3];t=float(fit.x[3]);v=M@(k0+z)
    assert np.max(np.maximum(l1-v,v-u1))<=t+1e-6
    return max(t,0.),k0+z

def ident(e):return {k:e[k] for k in ('replay','eid','uid','family','tick0','tick1','gap','original_capture','fresh','limited')}

def summary_scores(rows,field):
    names=sorted(set(k for r in rows for k in r[field]))
    return {name:A.metric([r[field][name] for r in rows if name in r[field]]) for name in names}

def quant_task(caps,out):
    selected=[e for e in caps if e['new']['tick']==phase_after(e['old']['tick']) and e['new']['timer']==0 and e['old']['flag']==e['new']['flag']==0]
    assert len(selected)==244 and sum(e['family']=='R-77-1' for e in selected)==79
    rows=[];basis={'raw':[],'centered':[]};grid=np.round(np.arange(.77,.83001,.002),3)
    for e in selected:
        a,b=E.arr(e['old']),E.arr(e['new']);n=b['n'];q=b['Q'];k0,k1=rateints(a['w']),rateints(b['w']);P=projector(n,.8)
        nps=A.unit(b['Ps']-b['p']);ng=b['m'];npast=A.unit(a['Ps']+a['Vs']*(e['gap']/48)-b['p'])
        models={'post_Q_k08':P@q,'pre_Q_k08':q@P,'no_Q_k08':P,'post_QT_k08':P@q.T,'Q_copy':q,'self_a0926':projector(n,1-.926*H/(TAU+H/2))@q,'self_a1':projector(n,1-H/(TAU+H/2))@q,'post_Qold_k08':P@a['Q'],'post_Q_nPs_k08':projector(nps,.8)@q,'post_Q_pointing_k08':projector(ng,.8)@q,'post_Q_pastPV_k08':projector(npast,.8)@q}
        rec=ident(e)|dict(rate0_int=k0,rate1_int=k1,n=n,Q=q,raw_errors={},centered_errors={},slack_cells={},grid_slack_cells={})
        for label,M in models.items():
            rec['raw_errors'][label]=float(np.linalg.norm(M@k0-k1))*QSTEP*A.RAD
            rec['centered_errors'][label]=float(np.linalg.norm(M@midpoint(k0,'truncate')-midpoint(k1,'truncate')))*QSTEP*A.RAD
            for kind in ('truncate','nearest'):
                sl,_=slack_lp(M,k0,k1,kind);rec['slack_cells'][label+'/'+kind]=sl
        if e['family']=='R-77-1':
            for k in grid:rec['grid_slack_cells'][str(k)]=slack_lp(projector(n,float(k))@q,k0,k1,'truncate')[0]
        for label,x,y in [('raw',k0,k1),('centered',midpoint(k0,'truncate'),midpoint(k1,'truncate'))]:
            s=q@x;par=(s@n)*n;per=s-par
            basis[label].append(dict(scene=e['replay'],family=e['family'],parallel=par,perp=per,target=y))
        rows.append(rec)
    groups={}
    for name,inds in [('all',list(range(len(rows)))),('R77',[i for i,r in enumerate(rows) if r['family']=='R-77-1']),('R77_limited',[i for i,r in enumerate(rows) if r['family']=='R-77-1' and r['limited']])]:
        rr=[rows[i] for i in inds];fits={}
        for enc in basis:
            z=[basis[enc][i] for i in inds];folds=[];errs=[]
            for scene in sorted(set(r['scene'] for r in z)):
                train=[r for r in z if r['scene']!=scene];test=[r for r in z if r['scene']==scene]
                den=sum(float(r['perp']@r['perp']) for r in train)
                if den<1e-12:continue
                k=sum(float(r['perp']@(r['target']-r['parallel'])) for r in train)/den
                ea=[float(np.linalg.norm(r['parallel']+k*r['perp']-r['target']))*QSTEP*A.RAD for r in test];errs+=ea
                folds.append(dict(scene=scene,kappa_train=k,n_train=len(train),n_test=len(test),error=A.metric(ea)))
            fits[enc]=dict(folds=folds,error=A.metric(errs))
        sl=summary_scores(rr,'slack_cells')
        for key in sl:sl[key]['feasible_count']=sum(r['slack_cells'][key]<=1e-7 for r in rr)
        groups[name]=dict(n=len(rr),raw=summary_scores(rr,'raw_errors'),centered=summary_scores(rr,'centered_errors'),slack=sl,fits=fits)
    r77=[r for r in rows if r['family']=='R-77-1'];gr={k:dict(feasible=sum(r['grid_slack_cells'][k]<=1e-7 for r in r77),slack=A.metric([r['grid_slack_cells'][k] for r in r77])) for k in r77[0]['grid_slack_cells']}
    worst=sorted(r77,key=lambda r:r['slack_cells']['post_Q_k08/truncate'],reverse=True)[:12]
    result=dict(scope='Conditional reset mapping, fixed decoded Q/n, rate encoding alternatives. LP witnesses are not predicted states.',q_rad_s=QSTEP,q_deg_s=QSTEP*A.RAD,groups=groups,grid_R77=gr,worst_R77=worst)
    A.save(out/'quantization_summary.json',result);A.save(out/'quantization_rows.json.gz',rows)
    for k,v in groups.items():print('QUANT_GROUP',k,json.dumps(A.serial(v)),flush=True)
    print('QUANT_GRID',json.dumps(A.serial(gr)),flush=True)
    print('QUANT_WORST',json.dumps(A.serial([{k:r[k] for k in ('replay','eid','uid','tick0','tick1','gap','limited')}|{'slack':r['slack_cells']['post_Q_k08/truncate'],'error':r['raw_errors']['post_Q_k08']} for r in worst])),flush=True)

def phase_task(caps,hist,out):
    rows=[]
    for e in caps:
        a,b=E.arr(e['old']),E.arr(e['new']);n=b['n'];s=b['Q']@a['w'];wp=projector(n,.8)@s
        rows.append(ident(e)|dict(before_phase=a['tick']%48,after_phase=b['tick']%48,flag0=a['flag'],flag1=b['flag'],timer0=a['timer'],timer1=b['timer'],raw_reset_error_deg_s=float(np.linalg.norm(wp-b['w']))*A.RAD,contained_phases=[p for p in range(48) if a['tick']<phase_after(a['tick'],p)<=b['tick']]))
    subsets={'all':rows,'original':[r for r in rows if r['original_capture']],'repeated':[r for r in rows if not r['original_capture']],'R77_original':[r for r in rows if r['family']=='R-77-1' and r['original_capture']],'R77_repeated':[r for r in rows if r['family']=='R-77-1' and not r['original_capture']]}
    groups={}
    for name,rr in subsets.items():
        byphase={}
        for ph in range(48):
            zz=[r for r in rr if r['after_phase']==ph and r['fresh'] and r['flag0']==r['flag1']==0]
            if zz:byphase[str(ph)]=dict(n=len(zz),gap_counts=dict(collections.Counter(r['gap'] for r in zz)),reset_error=A.metric([r['raw_reset_error_deg_s'] for r in zz]))
        groups[name]=dict(n=len(rr),before_hist=dict(collections.Counter(r['before_phase'] for r in rr)),after_hist=dict(collections.Counter(r['after_phase'] for r in rr)),phase_coverage={str(p):sum(p in r['contained_phases'] for r in rr) for p in range(48)},fresh_flag0_endpoint=byphase)
    boundaries={}
    for uid,rs in hist.items():
        bnd=[];lastcapture=None;lastloss=None
        for a,b in zip(rs,rs[1:]):
            if a['code']==b['code']:continue
            r=dict(uid=uid,index0=a['index'],index1=b['index'],tick0=a['tick'],tick1=b['tick'],phase0=a['tick']%48,phase1=b['tick']%48,code0=a['code'],code1=b['code'],flag0=a['flag'],flag1=b['flag'],timer0=a['timer'],timer1=b['timer'],gap=b['tick']-a['tick'],Ps_change=float(np.linalg.norm(np.array(b['Ps'])-a['Ps'])),Vs_change=float(np.linalg.norm(np.array(b['Vs'])-a['Vs'])))
            if a['code']==25 and b['code']==31:
                r['ticks_since_previous_capture']=b['tick']-lastcapture if lastcapture is not None else None;r['ticks_since_previous_loss']=b['tick']-lastloss if lastloss is not None else None;lastcapture=b['tick']
            if a['code']==31 and b['code']==25:lastloss=b['tick']
            bnd.append(r)
        boundaries[uid]=bnd
    result=dict(scope='All48 candidate phases; coverage is temporal compatibility, not proof of event time or echo acceptance.',groups=groups,anchor_boundaries=boundaries)
    A.save(out/'phase_summary.json',result);A.save(out/'phase_rows.json.gz',rows)
    for name,g in groups.items():
        print('PHASE_GROUP',name,json.dumps(A.serial(dict(n=g['n'],after_hist=g['after_hist'],best_coverage=sorted(g['phase_coverage'].items(),key=lambda x:x[1],reverse=True)[:8],endpoint=g['fresh_flag0_endpoint']))),flush=True)
    for uid,rr in boundaries.items():print('BOUNDARIES',uid,json.dumps(A.serial(rr)),flush=True)

def temporal_task(series,out):
    pairs,_=E.ordinary_pairs(series);sets={'R77':[r for r in pairs if r['family']=='R-77-1' and r['maxangle']<54.9],'all45':[r for r in pairs if r['maxangle']<45]}
    assert len(sets['R77'])==5484 and len(sets['all45'])==32189
    lookup={}
    for ss in series:
        for a,b in zip(ss['rows'],ss['rows'][1:]):
            if a is not None and b is not None:lookup[(ss['replay'],ss['eid'],a['tick'],b['tick'])]=(a,b)
    result=dict(scope='Pointing-conditioned ordinary state reconstruction; not independently identified radar observations. Other scenes fit only.',groups={});outrows=[]
    for group,rr in sets.items():
        ab=[lookup[(r['replay'],r['eid'],r['tick0'],r['tick1'])] for r in rr]
        n0=np.array([a['n'] for a,b in ab]);wraw=np.array([a['w'] for a,b in ab]);wt=np.array([b['w'] for a,b in ab]);nt=np.array([b['n'] for a,b in ab]);m=np.array([r['m'] for r in rr]);p=np.array([r['p'] for r in rr]);R=np.array([r['R'] for r in rr]);Pt=np.array([r['target'] for r in rr]);T=np.array([r['T'] for r in rr]);g=1/(TAU+T/2);gap=np.rint(T*48).astype(int);reps=np.array([r['replay'] for r in rr]);names=['constant_alpha','constant_output_offset','implicit_velocity_alpha','repeated_current_pointing']
        for encoding in (['raw','centered'] if group=='R77' else ['raw']):
            w0=wraw if encoding=='raw' else (wraw/QSTEP+.5*np.sign(wraw))*QSTEP
            pred0=Rotation.from_rotvec(-w0*T[:,None]).apply(n0)
            def predict(model,c):
                if model=='repeated_current_pointing':
                    n=n0.copy();w=w0.copy()
                    for j in range(int(gap.max())):
                        ids=gap>j;pp=Rotation.from_rotvec(-w[ids]*H).apply(n[ids]);nn=E.sphere(pp,m[ids],c)
                        cv=np.cross(nn,pp);vl=E.norm(cv);th=np.arctan2(vl,E.dot(nn,pp));corr=cv*np.divide(th,vl,out=np.ones_like(th),where=vl>1e-14)[:,None]
                        w[ids]+=corr/(TAU+H/2);n[ids]=nn
                    return n,w
                if model=='constant_alpha':alpha=c
                elif model=='constant_output_offset':alpha=(1/(1+c*g))[:,None]
                elif model=='implicit_velocity_alpha':alpha=(c/(1-(1-c)*T*g))[:,None]
                else:raise ValueError(model)
                n=E.sphere(pred0,m,alpha);cv=np.cross(n,pred0);vl=E.norm(cv);th=np.arctan2(vl,E.dot(n,pred0));corr=cv*np.divide(th,vl,out=np.ones_like(th),where=vl>1e-14)[:,None]
                return n,w0+corr*g[:,None]
            scores={name:dict(n_deg=np.full(len(rr),np.nan),w_deg_s=np.full(len(rr),np.nan),P_m=np.full(len(rr),np.nan)) for name in names};folds=[]
            for rep in sorted(set(reps)):
                test=reps==rep;train=~test
                fold=dict(test_scene=rep,train_count=int(train.sum()),test_count=int(test.sum()),parameters={})
                for name in names:
                    bounds=(0,.04) if name=='constant_output_offset' else (.3,.999999)
                    def objective(c):n,_=predict(name,c);return float(np.mean(np.sum((n[train]-nt[train])**2,axis=1)))
                    fit=minimize_scalar(objective,bounds=bounds,method='bounded',options={'xatol':1e-10});c=float(fit.x);n,w=predict(name,c)
                    fold['parameters'][name]=dict(value=c,success=bool(fit.success))
                    scores[name]['n_deg'][test]=E.vang(n[test],nt[test])*A.RAD;scores[name]['w_deg_s'][test]=E.norm(w[test]-wt[test])*A.RAD;scores[name]['P_m'][test]=E.norm(p[test]+R[test,None]*n[test]-Pt[test])
                folds.append(fold)
            strat={}
            for label,mask in [('all',np.ones(len(rr),bool))]+[(str(i),gap==i) for i in sorted(set(gap))]:
                strat[label]={name:{k:A.metric(v[mask]) for k,v in fields.items()} for name,fields in scores.items()}
            diagnostic={}
            for i in sorted(set(gap)):
                ids=gap==i
                fit=minimize_scalar(lambda c:float(np.mean(np.sum((predict('constant_alpha',c)[0][ids]-nt[ids])**2,axis=1))),bounds=(.3,.999999),method='bounded',options={'xatol':1e-11})
                diagnostic[str(i)]=dict(n=int(ids.sum()),alpha_all_scene_fit=float(fit.x),scope='Diagnostic fit within gap, not held-out validation')
            result['groups'][group+'/'+encoding]=dict(n=len(rr),folds=folds,strata=strat,alpha_per_gap_diagnostic=diagnostic)
            for j,r in enumerate(rr):outrows.append({k:r[k] for k in ('replay','eid','uid','tick0','tick1')}|dict(group=group,encoding=encoding,gap=int(gap[j]),scores={name:{k:float(v[j]) for k,v in z.items()} for name,z in scores.items()}))
            print('TEMPORAL_GROUP',group,encoding,json.dumps(A.serial(result['groups'][group+'/'+encoding])),flush=True)
    A.save(out/'temporal_summary.json',result);A.save(out/'temporal_rows.json.gz',outrows)

class Checks(unittest.TestCase):
    def test_cells(self):
        l,u=cell([-2,0,3],'truncate');np.testing.assert_array_equal(l,[-3,-1,3]);np.testing.assert_array_equal(u,[-2,1,4])
    def test_lp(self):
        self.assertAlmostEqual(slack_lp(np.eye(3),np.array([2.,-3,0]),np.array([2.,-3,0]),'truncate')[0],0)
        self.assertAlmostEqual(slack_lp(np.eye(3),np.zeros(3),np.array([3.,0,0]),'nearest')[0],2)
    def test_projection(self):
        n=A.unit([1,2,3]);P=projector(n,.8);np.testing.assert_allclose(P@n,n,atol=1e-14);v=np.cross(n,[1,0,0]);np.testing.assert_allclose(P@v,.8*v,atol=1e-14)
    def test_order(self):
        n=np.array([1.,0,0]);q=Rotation.from_euler('z',45,degrees=True).as_matrix();P=projector(n,.8);self.assertGreater(np.linalg.norm(P@q-q@P),.1)
    def test_implicit_weight(self):
        for T in (H,2*H,3*H):
            a=.876;f=a/(1-(1-a)*T/(TAU+T/2));self.assertAlmostEqual(f,a+(1-a)*T/(TAU+T/2)*f)

def main():
    p=argparse.ArgumentParser();p.add_argument('task',choices=['quant','phase','temporal','tests']);p.add_argument('--input',type=Path);p.add_argument('--out',type=Path);args=p.parse_args()
    if args.task=='tests':
        run=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks));raise SystemExit(not run.wasSuccessful())
    args.out.mkdir(parents=True,exist_ok=False)
    base=args.input/'data/main';cp=base/'evidence/capture_events.json.gz';sp=base/'input/cohort.json.gz';hp=base/'evidence/anchor_full_histories.json.gz'
    A.save(args.out/'sources.json',{str(x.relative_to(args.input)):hashlib.sha256(x.read_bytes()).hexdigest() for x in (cp,sp,hp)})
    caps=E.load(cp);assert len(caps)==931
    if args.task=='quant':quant_task(caps,args.out)
    elif args.task=='phase':phase_task(caps,E.load(hp),args.out)
    else:temporal_task(E.load(sp),args.out)

if __name__=='__main__':main()
