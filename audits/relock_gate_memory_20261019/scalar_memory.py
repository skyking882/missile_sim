#!/usr/bin/env python3
"""No-fit scalar-memory hypotheses. current closing is only a response label."""
from __future__ import annotations
import argparse,copy,math
from pathlib import Path
import numpy as np
import analyze as A
Q=30000/32767
MODELS=('A_closing_plus_slope','B_closing_slope_own_delta','C_frozen_ray_d_slope','D_predicted_ray_d_slope','E_frozen_ray_d','F_predicted_ray_d')

def midpoint(x):return x+math.copysign(Q/2,x) if x else 0.
def predictors(seed,current,mid=False):
    T=(current['tick']-seed['tick'])/48
    if T<0:raise ValueError('Negative prediction age')
    n0=A.unit(seed['n']);nt=A.propagate(n0,seed['w'],T);v=A.vec(current['vm']);v0=A.vec(seed['vm'])
    d,l,c=[midpoint(seed[k]) if mid else seed[k] for k in ('d','last','closing')]
    return dict(zip(MODELS,[c+l*T,c+l*T+float(n0@(v-v0)),float(n0@v)-d+l*T,float(nt@v)-d+l*T,float(n0@v)-d,float(nt@v)-d]))

def main():
    p=A.argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--main',type=Path,required=True);p.add_argument('--out',type=Path,required=True);z=p.parse_args();z.out.mkdir(parents=True,exist_ok=False)
    episodes=A.read(z.input/'episodes.json.gz');main=A.read(z.main/'episodes.json.gz');by={(e['clock'],e['episode_id']):e for e in main};anchors=A.read(z.input/'anchors.json.gz');records=[];guards=0;independent=0.;all_recon=[]
    for ep in episodes:
        if not ep['strict']:continue
        seed=ep['seed'];constant=all(r['d']==seed['d'] and r['last']==seed['last'] for r in ep['held'])
        output={}
        for role,r in [('held',r) for r in ep['held']]+([] if ep['exit'] is None or ep['exit_reason']!='fresh_return' else [('return',ep['exit'])]):
            pred=predictors(seed,r);mid=predictors(seed,r,True);T=(r['tick']-seed['tick'])/48
            scalar=math.fsum(float(x)*float(y) for x,y in zip(A.unit(seed['n']),r['vm']))-seed['d']+seed['last']*T
            independent=max(independent,abs(scalar-pred['C_frozen_ray_d_slope']))
            bad=copy.deepcopy(r)
            for k in ('n','w','Ps','Vs','R','d','closing','last'):bad[k]=float('nan')
            assert predictors(seed,bad)==pred;guards+=1
            err={k:v-r['closing'] for k,v in pred.items()};merr={k:v-midpoint(r['closing']) for k,v in mid.items()}
            bound=Q*(2+T)+2*np.linalg.norm(r['vm'])*math.sqrt(3)/32767+math.sqrt(3)*4000/65535
            rr=dict(episode_id=ep['episode_id'],uid=ep['uid'],replay=ep['replay'],eid=ep['eid'],tick=r['tick'],role=role,T=T,constant_d_last=constant,raw_sha=r['raw_sha'],seed_sha=seed['raw_sha'],predicted_closing=pred,recorded_closing=r['closing'],prediction_error=err,midpoint_error=merr,conservative_C_error_bound_mps=bound,C_within_conservative_bound=abs(err['C_frozen_ray_d_slope'])<=bound)
            if role=='held':all_recon.append(rr)
            for clock in ('packet','state'):
                previous=by.get((clock,ep['episode_id']))
                match=None if previous is None else next((x for x in previous['rows'] if x['index']==r['index']),None)
                if match is None:continue
                inv={k:match['radial_speed_mps']+v for k,v in pred.items()}
                records.append(rr|dict(clock=clock,nearby_seed_reference=previous['nearby_seed_reference'],independent_relative_radial_mps=match['radial_speed_mps'],reference_target_ground_mps=match['target_ground_radial_mps'],common_candidate_eligible=match['common_candidate_eligible'],current_ps_error_m=match['current_ps_error_m'],innovations=inv,target_bracket_s=match['target_bracket_s'],angle_prediction_error_deg=match['predicted_direction_error_deg'],range_CV_innovation_m=match['range_innovation_m']['range_PV_CV']))
    recon={}
    for group,rs in [('all_strict',all_recon),('constant_d_last',[r for r in all_recon if r['constant_d_last']])]:
        recon[group]=dict(records=len(rs),episodes=len({r['episode_id'] for r in rs}),raw={k:A.metric([r['prediction_error'][k] for r in rs]) for k in MODELS},midpoint={k:A.metric([r['midpoint_error'][k] for r in rs]) for k in MODELS},C_within_conservative_bound=sum(r['C_within_conservative_bound'] for r in rs))
    tables={};false=[]
    for clock in ('packet','state'):
        rs=[r for r in records if r['clock']==clock and r['nearby_seed_reference']]
        held=[r for r in rs if r['role']=='held' and r['T']<A.EARLY and r['common_candidate_eligible']]
        ret=[r for r in rs if r['role']=='return' and r['T']<A.EARLY]
        tables[clock]={}
        for model in MODELS:
            for width in (10.,20.,300.):
                bad=[r for r in ret if abs(r['innovations'][model])>width]
                tables[clock][f'{model}/{width:g}']=dict(held_n=len(held),held_rejected=sum(abs(r['innovations'][model])>width for r in held),held_still_pass=sum(abs(r['innovations'][model])<=width for r in held),returns_n=len(ret),returns_rejected=len(bad),episodes_with_held_still_pass=len({r['episode_id'] for r in held if abs(r['innovations'][model])<=width}))
                if clock=='packet' and model=='C_frozen_ray_d_slope' and width==10:false=bad
    first118=[]
    seed=next(r for r in anchors['118'] if r['tick']==11524)
    for r in anchors['118']:
        if 11524<=r['tick']<=11617:
            pred=predictors(seed,r)
            first118.append(dict(tick=r['tick'],age_s=(r['tick']-11415)/48,code=r['code'],timer=r['timer'],closing=r['closing'],d=r['d'],last=r['last'],predicted_closing=pred,primary_error_mps=pred['C_frozen_ray_d_slope']-r['closing']))
    summary=dict(scope='Reconstruction of a held output scalar plus necessary-gate tests. Inferring acceptance use from output fit is not automatic; recovered measurement is independent aircraft history, not game lock ID.',reconstruction=recon,gates=tables,false_early_rejections_primary10=false,anchor118=first118,verification=dict(current_answer_poison_checks=guards,independent_scalar_max_difference_mps=independent),all_records=len(records))
    A.save(z.out/'summary.json',summary);A.save(z.out/'records.json.gz',records);A.save(z.out/'reconstruction.json.gz',all_recon)
    print('RECONSTRUCTION',A.json.dumps(A.clean(recon)),flush=True)
    for clock,t in tables.items():print('MEMORY_GATES',clock,A.json.dumps(t),flush=True)
    for r in first118:
        if r['tick'] in (11524,11526,11532,11571,11573,11575,11615,11617):print('MEMORY118',A.json.dumps(r),flush=True)
    for r in false:print('ADVERSE_RETURN',A.json.dumps(A.clean(r)),flush=True)
    print('CHECKS',A.json.dumps(summary['verification']),flush=True)
if __name__=='__main__':main()
