#!/usr/bin/env python3
"""Numerical review on identical observations, not an independent AI review."""
import argparse,json,math
from pathlib import Path
import numpy as np
from scipy.optimize import lsq_linear,nnls
import range_check as C
import other_checks as O

def key(e):return (e['replay'],e['eid'],e['tick0'],e['tick1'])

def main(inp,saved,out):
    out.mkdir(parents=True,exist_ok=False);checked=C.verify(inp);events=C.read(inp/'prior/evidence/data/main/evidence/capture_events.json.gz')
    near=np.array([e['new']['timer']==0 and e['old']['flag']==0 and e['new']['flag']==0 and (e['tick1']-1)%48==0 for e in events]);assert near.sum()==244
    scenes=np.array([e['replay'] for e in events]);fams=np.array([e['family'] for e in events]);n=C.vec([e['new']['n'] for e in events]);x=C.vec([e['old']['w'] for e in events]);y=C.vec([e['new']['w'] for e in events]);base=np.array([C.D(nn)@C.vec(e['new']['Q'])@xx for nn,e,xx in zip(n,events,x)])
    scalar_checks=[];error_checks=0;max_error_difference=0.;comparison={};six={};weight_checks=0
    for family,builder,fix in [('range',C.range_candidates,1.),('mixture',O.mixtures,.2),('representation',O.representations,1.)]:
        summary=C.read(saved/family/'results/summary.json');rows=C.read(saved/family/'results/rows.json.gz');assert len(rows)==931
        pure={};invalid={}
        for i,(e,r) in enumerate(zip(events,rows)):
            assert key(e)==key(r);assert bool(near[i])==r['near']
            mods,_,_=builder(e['old'],C.public_new(e['new']),(e['tick1']-e['tick0'])/48)
            for label,(M,b) in mods.items():
                pure.setdefault(label,np.empty((len(events),3)))[i]=M@x[i]+b
            for label,z in r['scores'].items():
                pred=C.vec(z['prediction_rad_s']);error=pred-y[i];actual=math.sqrt(math.fsum(float(v*v) for v in error))*C.DEG
                delta=abs(actual-z['rate_deg_s']);max_error_difference=max(max_error_difference,delta);error_checks+=1
                assert delta<=1e-9+1e-12*abs(actual)
        for fold in summary['folds']:
            train=near&(scenes!=fold['held_scene'])
            if not fold['other_family_training_fallback']:train&=fams==fold['target_family']
            test=(scenes==fold['held_scene'])&(fams==fold['target_family'])
            assert train.sum()==fold['training_near_rows'] and test.sum()==fold['test_rows']
            assert not np.any(train&test)
            for label,fit in fold['candidates'].items():
                delta=pure[label]-base;d=delta[train].ravel();v=(y-base)[train].ravel();size=float(np.linalg.norm(d));beta=fit['beta']
                if size>1e-15:
                    check=lsq_linear((d/size)[:,None],v/size,bounds=(0,1),method='bvls',tol=1e-12)
                    assert check.success
                    diff=abs(float(check.x[0])-beta);assert diff<1e-8,(family,label,diff)
                    scalar_checks.append(diff)
                target=base+beta*delta
                for i in np.where(test)[0]:
                    np.testing.assert_allclose(target[i],rows[i]['scores']['fit/'+label]['prediction_rad_s'],atol=1e-12,rtol=1e-12);weight_checks+=1
            choice=fold['selected'];beta=fold['selected_beta']
            for i in np.where(test)[0]:
                assert rows[i]['selected']==choice and rows[i]['selected_weight']==beta
        comparison[family]={g:{name:{'rms':z['rate_deg_s']['rms'],'maximum':z['rate_deg_s']['max'],'feasible':z['cell_feasible'],'repaired':z['repaired_baseline_failures'],'introduced':z['introduced_failures']} for name,z in ss.items() if name in ('baseline','selected_LOSO')} for g,ss in summary['groups'].items()}
        six[family]=summary['six_cases']
    diagnostics=[]
    for e,xx,yy,nn in zip(events,x,y,n):
        s=C.vec(e['new']['Q'])@xx;a=(s@nn)*nn;b=s-a
        aa=max(0.,float(a@yy/(a@a))) if a@a>1e-25 else 0.;bb=max(0.,float(b@yy/(b@b))) if b@b>1e-25 else 0.;pred=aa*a+bb*b
        coef,err=nnls(np.column_stack([a,b]),yy);assert abs(np.linalg.norm(pred-yy)-err)<1e-10
        diagnostics.append({k:e[k] for k in ('replay','eid','uid','tick0','tick1','family')}|dict(diagnostic_only=True,axial_coefficient=aa,transverse_coefficient=bb,decoded_two_component_error_deg_s=float(np.linalg.norm(pred-yy))*C.DEG))
    si={key(r) for r in six['range']};scaling=C.read(saved/'range/results/scaling_diagnostics.json.gz')
    out_six=[]
    for k in sorted(si):
        scale=next(r for r in scaling if key(r)==k);diag=next(r for r in diagnostics if key(r)==k)
        record=dict(identity=k,scale=scale,two_component=diag,families={})
        for family in six:
            r=next(r for r in six[family] if key(r)==k)
            record['families'][family]=dict(selected=r['selected'],weight=r['selected_weight'],rate_error=r['scores']['selected_LOSO']['rate_deg_s'],slack=r['scores']['selected_LOSO']['slack_cells'])
        rr=next(r for r in six['range'] if key(r)==k);ratios=list(rr['auxiliary_inputs'].values());record['observed_ratio_minmax']=[min(ratios),max(ratios)]
        out_six.append(record)
    review=dict(input_manifest_verified=checked,rate_error_recalculations=error_checks,maximum_error_recalculation_difference=max_error_difference,independent_bounded_lsq_fits=len(scalar_checks),max_coefficient_difference=max(scalar_checks,default=0),held_scene_prediction_checks=weight_checks,independent_two_component_nnls_checks=len(diagnostics),scope='Independent arithmetic/optimizer formulations on same frozen observations; not independent new replay validation or subagents.',comparison=comparison,six_cases=out_six)
    C.save(out/'review.json',review);C.save(out/'two_component_diagnostics.json.gz',diagnostics)
    for k in ('input_manifest_verified','rate_error_recalculations','maximum_error_recalculation_difference','independent_bounded_lsq_fits','max_coefficient_difference','held_scene_prediction_checks','independent_two_component_nnls_checks'):print('REVIEW_CHECK',k,review[k],flush=True)
    print('REVIEW_TABLE',json.dumps(comparison,ensure_ascii=False),flush=True)
    print('REVIEW_SIX',json.dumps(C.serial(out_six),ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--saved',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.input,a.saved,a.out)
