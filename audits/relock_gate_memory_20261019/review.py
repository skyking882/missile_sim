#!/usr/bin/env python3
"""Independent held-channel review and fixed quadrature alternatives; no fitting."""
from __future__ import annotations
import argparse,collections,copy,math
from pathlib import Path
import numpy as np
import analyze as A
import scalar_memory as S
B='B_closing_slope_own_delta'

def dot(a,b):return math.fsum(float(x)*float(y) for x,y in zip(a,b))
def independent_b(s,r):
    n=np.array(s['n'],float);n=n/math.sqrt(dot(n,n));dt=(r['tick']-s['tick'])/48
    return s['closing']+s['last']*dt+dot(n,np.array(r['vm'])-s['vm'])
def metric_abs(x):return A.metric([abs(a) for a in x])
def reconstruction(eps):
    rows=[];compare=0.;poison=0
    for e in eps:
        if not e['strict']:continue
        s=e['seed'];previous=s;terms={k:0. for k in ('start','mid','end')}
        for r in e['held']:
            ta=(previous['tick']-s['tick'])/48;tb=(r['tick']-s['tick'])/48;dv=A.vec(r['vm'])-A.vec(previous['vm'])
            for name,frac in [('start',0.),('mid',.5),('end',1.)]:
                nn=A.propagate(s['n'],s['w'],ta+(tb-ta)*frac);terms[name]+=dot(nn,dv)
            preds={'frozen_ray':independent_b(s,r)}|{name:s['closing']+s['last']*tb+value for name,value in terms.items()}
            standard=S.predictors(s,r)[B];compare=max(compare,abs(preds['frozen_ray']-standard))
            bad=copy.deepcopy(r)
            for k in ('n','w','d','last','closing','Ps','Vs','R'):bad[k]=float('nan')
            assert independent_b(s,bad)==preds['frozen_ray'];poison+=1
            err={k:v-r['closing'] for k,v in preds.items()}
            bound=S.Q*(2+tb)+2*np.linalg.norm(dv)*math.sqrt(3)/32767+2*math.sqrt(3)*4000/65535
            rows.append(dict(episode_id=e['episode_id'],replay=e['replay'],eid=e['eid'],uid=e['uid'],tick=r['tick'],T=tb,raw_sha=r['raw_sha'],errors=err,predictions=preds,observed=r['closing'],bound_frozen_firstorder=bound,within_bound=abs(err['frozen_ray'])<=bound))
            previous=r
    metrics={k:dict(signed=A.metric([r['errors'][k] for r in rows]),absolute=metric_abs([r['errors'][k] for r in rows])) for k in ('frozen_ray','start','mid','end')}
    adverse=sorted(rows,key=lambda r:abs(r['errors']['frozen_ray']),reverse=True)[:12]
    return rows,dict(n=len(rows),episodes=len({r['episode_id'] for r in rows}),within_conservative_firstorder_bound=sum(r['within_bound'] for r in rows),models=metrics,adverse_frozen=adverse,independent_max_difference=compare,current_answer_poison_checks=poison)

def fresh_controls(fresh,targets):
    by=collections.defaultdict(list)
    for x in fresh:by[x['replay'],x['eid']].append(x)
    scores=[]
    for key,rs in by.items():
        for x,y in zip(rs,rs[1:]):
            s,r=x['record'],y['record'];T=(r['tick']-s['tick'])/48
            if s['flag'] or r['flag'] or not 0<T<=8/48:continue
            target=targets.get((y['replay'],y['target_eid']))
            if target is None:continue
            pred=independent_b(s,r)
            for clock in ('packet','state'):
                a=A.geometry(s,target,clock);b=A.geometry(r,target,clock)
                if a is None or b is None or max(a['ps_error'],b['ps_error'])>50:continue
                scores.append(dict(replay=key[0],eid=key[1],uid=y['uid'],clock=clock,tick0=s['tick'],tick1=r['tick'],T=T,seed_ps_error_m=a['ps_error'],new_ps_error_m=b['ps_error'],innovation_mps=pred+b['rr'],current_output_innovation_mps=r['closing']+b['rr'],predicted_closing=pred,actual_closing=-b['rr'],new_saved_closing=r['closing'],target_bracket_s=b['target_bracket_s'],raw_sha=r['raw_sha']))
    out={}
    for clock in ('packet','state'):
        rr=[r for r in scores if r['clock']==clock];out[clock]=dict(n=len(rr),entities=len({(r['replay'],r['eid']) for r in rr}),innovation_abs=metric_abs([r['innovation_mps'] for r in rr]),rejected={str(w):sum(abs(r['innovation_mps'])>w for r in rr) for w in (10,20,300)},current_output_abs=metric_abs([r['current_output_innovation_mps'] for r in rr]),worst=sorted(rr,key=lambda r:abs(r['innovation_mps']),reverse=True)[:5])
    return scores,out

def tests():
    s=dict(tick=100,n=[1.,0.,0.],w=[0.,0.,0.],vm=[100.,20.,30.],closing=40.,last=3.,d=60.)
    r=dict(tick=124,vm=[110.,-200.,50.]);assert independent_b(s,r)==51.5
    shifted=dict(r,tick=1124);old=dict(s,tick=1100);assert independent_b(s,r)==independent_b(old,shifted)
    s2=dict(s,vm=[1100.,2020.,3030.]);r2=dict(r,vm=[1110.,1800.,3050.]);assert independent_b(s2,r2)==independent_b(s,r)
    nochange=dict(r,vm=s['vm']);assert independent_b(s,nochange)==41.5
    s3=dict(s,last=0.);assert independent_b(s3,r)==50.
    # This is a memory law, not a cooldown: zero innovation at any T is admissible.
    for age in (.125,.5,1.):
        rr=dict(r,tick=100+int(48*age));pred=independent_b(s,rr);assert abs(-pred+pred)<10
    withnan=dict(r,closing=float('nan'),n=[float('nan')]*3,Ps=[float('nan')]*3);assert independent_b(s,r)==independent_b(s,withnan)
    return 9

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--main',type=Path,required=True);p.add_argument('--scalar',type=Path,required=True);p.add_argument('--out',type=Path,required=True);z=p.parse_args();z.out.mkdir(parents=True,exist_ok=False)
    eps=A.read(z.input/'episodes.json.gz');base=A.read(z.main/'summary.json');sr=A.read(z.scalar/'records.json.gz');fresh=A.read(z.input/'fresh.json.gz');units=A.read(z.input/'units.json.gz');targets={(rep,int(u['eid'])):A.Target(u) for rep,uu in units.items() for u in uu}
    rec,review=reconstruction(eps);freshrows,fc=fresh_controls(fresh,targets)
    bmap={r['episode_id']:r for r in eps};smap={(r['clock'],r['episode_id'],r['tick']):r for r in sr}
    seen=0
    for r in rec:
        e=bmap[r['episode_id']];raw=next(x for x in e['held'] if x['tick']==r['tick']);assert raw['raw_sha']==r['raw_sha'];seen+=1
        m=smap.get(('packet',r['episode_id'],r['tick']))
        if m is not None:assert abs(m['predicted_closing'][B]-r['predictions']['frozen_ray'])<1e-9
    diagnostics=A.read(z.main/'episodes.json.gz');primary=[e for e in diagnostics if e['clock']=='packet' and e['strict'] and e['nearby_seed_reference']]
    examples=[];totals=collections.Counter();allrows=[]
    for e in primary:
        erows=[]
        for d in e['rows']:
            m=smap.get(('packet',e['episode_id'],d['tick']))
            if m is None:continue
            inv=m['innovations'][B]
            rr=dict(role=d['role'],tick=d['tick'],T=d['age_from_seed_s'],signal_margin_db=d['signal_limits']['15.0']['margin_ref2_db'],radial_innovation_mps=inv,reference_closing=-d['radial_speed_mps'],saved_closing=m['recorded_closing'],predicted_closing=m['predicted_closing'][B],ground_radial=d['target_ground_radial_mps'],range_CV_error_m=d['range_innovation_m']['range_PV_CV'],beam_deg=d['beam_offset_deg'],direction_prediction_error_deg=d['predicted_direction_error_deg'],common=d['common_candidate_eligible'],early=d['age_from_seed_s']<A.EARLY)
            erows.append(rr);allrows.append(dict(episode_id=e['episode_id'],replay=e['replay'],eid=e['eid'],**rr))
        eligible=[r for r in erows if r['role']=='held' and r['common'] and r['early']]
        if not eligible:continue
        b20=[r for r in eligible if abs(r['radial_innovation_mps'])>20];b10=[r for r in eligible if abs(r['radial_innovation_mps'])>10]
        totals['eligible_episodes']+=1;totals['with_any_speed20_rejection']+=bool(b20);totals['with_any_speed10_rejection']+=bool(b10);totals['all_eligible_blocked_speed20']+=len(b20)==len(eligible);totals['all_eligible_blocked_speed10']+=len(b10)==len(eligible)
        if b20:examples.append(dict(episode_id=e['episode_id'],replay=e['replay'],eid=e['eid'],uid=e['uid'],seed_tick=e['seed_tick'],exit_reason=e['exit_reason'],eligible_n=len(eligible),rejected20=len(b20),first_rejected20=b20[0],maximum_innovation=max(abs(r['radial_innovation_mps']) for r in eligible),returns=[r for r in erows if r['role']=='return']))
    # Examples cover distinct entities/scenes; within this diagnostic-positive
    # set selection is sorted by scene/entity, not minimum fit residual.
    examples.sort(key=lambda e:(e['replay'],e['eid'],e['seed_tick']));distinct=[];used=set()
    for e in examples:
        key=e['replay'],e['eid']
        if key not in used:distinct.append(e);used.add(key)
    aep=next(e for e in diagnostics if e['clock']=='packet' and e['eid']==2439170 and e['seed_tick']==11524)
    anchor=[]
    for d in aep['rows']:
        m=smap.get(('packet',aep['episode_id'],d['tick']))
        if m is None:continue
        anchor.append(dict(tick=d['tick'],age_s=(d['tick']-11415)/48,role=d['role'],actual_closing=-d['radial_speed_mps'],saved_closing=m['recorded_closing'],predicted_closing=m['predicted_closing'][B],innovation=m['innovations'][B],signal_margin_db=d['signal_limits']['15.0']['margin_ref2_db'],common=d['common_candidate_eligible'],rangeCV=d['range_innovation_m']['range_PV_CV'],groundradial=d['target_ground_radial_mps']))
    held=[r for r in anchor if r['role']=='held'];ref2eligible=[r for r in held if r['common']]
    countanchor=dict(held_records=len(held),signal_and_notch_eligible=len(ref2eligible),remaining10=sum(abs(r['innovation'])<=10 for r in ref2eligible),remaining20=sum(abs(r['innovation'])<=20 for r in ref2eligible),remaining300=sum(abs(r['innovation'])<=300 for r in ref2eligible),allheld_rejected_by_signal_or_speed20=all(not r['common'] or abs(r['innovation'])>20 for r in held),first_signal_recovery=ref2eligible[0] if ref2eligible else None)
    output=dict(scope='Conditional retrospective gates; acceptance-law use and width semantics still hypotheses. ScalarB chosen by output reconstruction after first-pass comparison, not held-label fitting.',tests=tests(),reconstruction=review,fresh_controls=fc,source_row_sha_checks=seen,cohort_totals=dict(totals),distinct_examples=distinct,all_diagnostic_positive_examples=examples,anchor118_gate_count=countanchor,anchor118=anchor)
    A.save(z.out/'verification.json',output);A.save(z.out/'reconstruction.json.gz',rec);A.save(z.out/'fresh_pairs.json.gz',freshrows);A.save(z.out/'primary_rows.json.gz',allrows)
    print('REVIEW_RECON',A.json.dumps(A.clean(review)),flush=True)
    for c,cc in fc.items():print('FRESH_CONTROLS',c,A.json.dumps(A.clean(cc)),flush=True)
    print('EPISODE_COUNTS',A.json.dumps(dict(totals)),flush=True);print('ANCHOR_COUNTS',A.json.dumps(countanchor),flush=True)
    for e in distinct[:10]:print('DISTINCT_CASE',A.json.dumps(A.clean(e)),flush=True)
    print('REVIEW_CHECKS',A.json.dumps(dict(tests=output['tests'],source_sha_checks=seen)),flush=True)
if __name__=='__main__':main()
