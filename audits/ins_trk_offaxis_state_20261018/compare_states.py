#!/usr/bin/env python3
"""Descriptive audit of War Thunder serialized states; no production-model updates."""
from __future__ import annotations
import argparse,copy,csv,hashlib,json,math,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'capture_held_20261017'))
sys.path.insert(0,str(ROOT/'ins_trk_three_models_20261018'))
import audit as A
import range_check as C
DEG=180/math.pi
TAU=.078525
PINNED={('2026.10.08_11.49.38',2191362,11279,11281),('2026.10.08_11.49.38',2260994,11423,11425),('2026.10.08_11.49.38',2439170,11519,11521),('2026.10.08_11.49.38',2826242,12335,12337),('2026.10.08_12.03.42',1524739,16991,16993),('2026.10.08_12.11.04',1941506,9695,9697)}

def ident(e):return (e['replay'],int(e['eid']),int(e['tick0']),int(e['tick1']))
def arr(r):
    r=r.copy()
    for k in ('n','w','Q','p','vm','Ps','Vs','angles'):r[k]=np.asarray(r[k],float)
    return r

def ang(a,b):return A.angle(A.unit(a),A.unit(b))*DEG

def geometry(r):
    d=r['Ps']-r['p'];R=float(np.linalg.norm(d))
    return np.cross(r['Vs']-r['vm'],d)/(R*R)

def classify(e):
    a=arr(e['old']);p=A.unit(a['Ps']-a['p']);beta=ang(p,a['Q'][:,0])
    g='small' if beta<=20 else ('large' if beta>=50 else 'middle')
    b='0-20' if beta<=20 else ('20-40' if beta<=40 else ('40-60' if beta<=60 else '60-180'))
    rr=float(np.linalg.norm(a['Ps']-a['p']));rb='below1km' if rr<1000 else ('1-5km' if rr<5000 else 'above5km')
    phase=((e['tick0']-1)//48+1)*48+1
    lag=e['tick1']-phase if not e['original_capture'] and phase<=e['tick1'] else None
    return dict(offaxis_deg=beta,angle_group=g,angle_bin=b,range0_m=rr,range_bin=rb,original=bool(e['original_capture']),candidate_reacq_lag=lag,near=C.near(e),fresh00=e['new']['timer']==0 and e['old']['flag']==e['new']['flag']==0)

def diagnose(e):
    a,b=arr(e['old']),arr(e['new']);T=(e['tick1']-e['tick0'])/48
    n0,n1=a['n'],b['n'];ng0=a['Q']@A.ray(a['angles']);ng1=b['Q']@A.ray(b['angles'])
    np0=A.unit(a['Ps']-a['p']);np1=A.unit(b['Ps']-b['p'])
    pp=a['Ps']+a['Vs']*T;npred=A.unit(pp-b['p'])
    s=b['Q']@a['w'];baseline=C.D(n1)@s
    axis_pointing=C.D(ng1)@s;axis_prior=C.D(npred)@s
    err=baseline-b['w'];ax=float(err@n1);tr=err-ax*n1
    dc=n1-n0;pieces=[np0-n0,npred-np0,n1-npred]
    np.testing.assert_allclose(sum(pieces),dc,rtol=0,atol=5e-16)
    rhs=.2*abs(math.sin(ang(n1,ng1)/DEG))*np.linalg.norm(a['w'])
    actual=float(np.linalg.norm(axis_pointing-baseline))
    assert actual<=rhs+1e-13,(ident(e),actual,rhs)
    sing=np.linalg.svd(C.D(n1)@b['Q'],compute_uv=False)
    np.testing.assert_allclose(sing,[1.,.8,.8],atol=2e-14,rtol=0)
    M=C.D(n1)@b['Q'];Msplit=.8*b['Q']+.2*np.outer(n1,n1@b['Q'])
    np.testing.assert_allclose(M,Msplit,atol=1e-14,rtol=0)
    sl=C.slack(M,np.zeros(3),C.ints(a['w']),C.ints(b['w']))
    predn=A.rotate(n0,a['w'],T);corr=A.correction(predn,n1)
    wrong=a['w']+corr/(TAU+T/2)
    oldbody=A.angles(a['Q'].T@np0)*DEG
    ownmean=(a['vm']+b['vm'])/2;ownmove=b['p']-a['p'];v2=float(ownmean@ownmean)
    r=dict(replay=e['replay'],eid=e['eid'],uid=e['uid'],family=e['family'],tick0=e['tick0'],tick1=e['tick1'],index0=a['index'],index1=b['index'],gap=e['gap'],timer0=a['timer'],timer1=b['timer'],flag0=a['flag'],flag1=b['flag'],raw_sha0=a['raw_sha'],raw_sha1=b['raw_sha'],**classify(e))
    r.update(mechanical_demand0=bool(np.max(np.abs(oldbody))>55),old_target_body_angles_deg=oldbody,pointing_at_limit1=bool(np.max(np.abs(b['angles']))*DEG>=54.99),offaxis_velocity_deg=ang(np0,a['vm']),offaxis1_deg=ang(np1,b['Q'][:,0]),rate0_deg_s=float(np.linalg.norm(a['w']))*DEG,rate1_deg_s=float(np.linalg.norm(b['w']))*DEG,
        baseline_error_deg_s=float(np.linalg.norm(err))*DEG,baseline_axial_deg_s=ax*DEG,baseline_transverse_deg_s=float(np.linalg.norm(tr))*DEG,baseline_slack_cells=sl,baseline_feasible=sl<=C.TOL,
        pure_Q_error_deg_s=float(np.linalg.norm(s-b['w']))*DEG,no_Q_error_deg_s=float(np.linalg.norm(C.D(n1)@a['w']-b['w']))*DEG,old_Q_error_deg_s=float(np.linalg.norm(C.D(n1)@a['Q']@a['w']-b['w']))*DEG,
        pointing_axis_error_deg_s=float(np.linalg.norm(axis_pointing-b['w']))*DEG,prior_axis_error_deg_s=float(np.linalg.norm(axis_prior-b['w']))*DEG,
        pointing_substitution_change_deg_s=actual*DEG,pointing_substitution_bound_deg_s=rhs*DEG,matrix_condition=float(sing[0]/sing[-1]),
        n0_vs_pointing_deg=ang(n0,ng0),n0_vs_Ps_deg=ang(n0,np0),n1_vs_pointing_deg=ang(n1,ng1),n1_vs_Ps_deg=ang(n1,np1),n_temporal_jump_deg=ang(n0,n1),prior_propagation_angle_deg=ang(np0,npred),prior_vs_new_direction_deg=ang(npred,n1),Ps_CV_innovation_m=float(np.linalg.norm(b['Ps']-pp)),
        direction_source_chord=pieces[0],direction_propagation_chord=pieces[1],direction_innovation_chord=pieces[2],direction_jump_chord=dc,
        old_geo_error_deg_s=float(np.linalg.norm(a['w']-geometry(a)))*DEG,old_geo_lambda_deg_s=float(a['w']@np0)*DEG,
        crossing_ordinary_negative_control_deg_s=float(np.linalg.norm(wrong-b['w']))*DEG,crossing_innovation_deg=ang(predn,n1),Q_body_change_deg=float(np.linalg.norm(Rotation.from_matrix(b['Q']@a['Q'].T).as_rotvec()))*DEG,
        own_motion_time_ratio=float(ownmove@ownmean/v2/T) if v2>1 else None,own_motion_residual_m=float(np.linalg.norm(ownmove-ownmean*T)),
        lambda1_deg_s=float(b['w']@n1)*DEG,lambda_Qseed_deg_s=float(s@n1)*DEG,old_geo_world=geometry(a),baseline_vector=baseline)
    return r

METRICS=('baseline_error_deg_s','baseline_axial_deg_s','baseline_transverse_deg_s','pure_Q_error_deg_s','no_Q_error_deg_s','pointing_axis_error_deg_s','prior_axis_error_deg_s','rate0_deg_s','n0_vs_Ps_deg','n_temporal_jump_deg','prior_vs_new_direction_deg','n1_vs_pointing_deg','Ps_CV_innovation_m','old_geo_error_deg_s','crossing_ordinary_negative_control_deg_s','Q_body_change_deg','range0_m','own_motion_time_ratio')
def stats(rows):
    r={'n':len(rows),'unique_entities':len({(x['replay'],x['eid']) for x in rows}),'scenes':len({x['replay'] for x in rows}),'first_count':sum(x['original'] for x in rows),'feasible':sum(x['baseline_feasible'] for x in rows),'mechanical_demand_count':sum(x['mechanical_demand0'] for x in rows),'pointing_limit_count':sum(x['pointing_at_limit1'] for x in rows),'metrics':{k:C.metric([x[k] for x in rows if x[k] is not None]) for k in METRICS}}
    den=math.fsum(x['rate1_deg_s']**2 for x in rows)
    r['relative_error_energy']=math.sqrt(math.fsum(x['baseline_error_deg_s']**2 for x in rows)/den) if den>1e-20 else None
    return r

def make_groups(rows):
    rr=[x for x in rows if x['family']=='R-77-1']
    sels={'all931':rows,'R77_all':rr,'R77_first':[x for x in rr if x['original']],'R77_reacq':[x for x in rr if not x['original']],'R77_near':[x for x in rr if x['near']],'R77_fresh00':[x for x in rr if x['fresh00']],'R77_first_fresh00':[x for x in rr if x['original'] and x['fresh00']]}
    for lag in (0,1,2):sels['R77_reacq_fresh00_lag'+str(lag)]=[x for x in rr if x['fresh00'] and x['candidate_reacq_lag']==lag]
    return {name:{g:stats([x for x in rr if g=='all' or x['angle_group']==g]) for g in ('all','small','middle','large')} for name,rr in sels.items()}

def standardized(rows,keys):
    dd={}
    for r in rows:
        if r['angle_group'] not in ('small','large'):continue
        key=tuple(str(r[k]) for k in keys);dd.setdefault(key,{'small':[],'large':[]})[r['angle_group']].append(r)
    cells=[]
    for key,r in sorted(dd.items()):
        if not r['small'] or not r['large']:continue
        w=min(len(r['small']),len(r['large']))
        cells.append(dict(stratum=list(key),weight=w,small=stats(r['small']),large=stats(r['large'])))
    total=sum(c['weight'] for c in cells)
    pooled={}
    for g in ('small','large'):
        pooled[g]={k:math.sqrt(sum(c['weight']*c[g]['metrics'][k]['rms']**2 for c in cells)/total) if total else None for k in ('baseline_error_deg_s','baseline_axial_deg_s','baseline_transverse_deg_s','n0_vs_Ps_deg','prior_vs_new_direction_deg')}
    return dict(scope='Common-support descriptive standardization, not causal identification or coefficient fitting; each cell weighted by min(group counts)',keys=keys,cells=cells,total_weight=total,pooled_rms=pooled)

def windows(root,events,rows):
    p=sorted((root/'input').rglob('anchor_full_histories.json.gz'));assert p
    hist=C.read(p[0]);out=[]
    for uid in ('118','148'):
        rr=[arr(r) for r in hist[uid]]
        ee=[e for e in events if (e['replay'],e['eid']) in A.ANCHORS and str(e['uid'])==uid][:2 if uid=='118' else 1]
        for e in ee:
            j=next(i for i,x in enumerate(rr) if x['index']==e['new']['index'] and x['tick']==e['tick1'])
            win=[]
            for i in range(max(0,j-4),min(len(rr),j+6)):
                b=rr[i];ng=b['Q']@A.ray(b['angles']);npb=A.unit(b['Ps']-b['p'])
                r=dict(index=b['index'],tick=b['tick'],relative_tick=b['tick']-e['tick1'],code=b['code'],timer=b['timer'],flag=b['flag'],target_offaxis_deg=ang(npb,b['Q'][:,0]),n_vs_pointing_deg=ang(b['n'],ng),n_vs_Ps_deg=ang(b['n'],npb),rate_deg_s=b['w']*DEG,old_geo_error_deg_s=float(np.linalg.norm(b['w']-geometry(b)))*DEG,raw_sha=b['raw_sha'])
                if i:
                    a=rr[i-1];T=(b['tick']-a['tick'])/48
                    if 0<T<=8/48:
                        c=A.correction(A.rotate(a['n'],a['w'],T),b['n'])
                        r.update(gap=b['tick']-a['tick'],mode_pair=[a['code'],b['code']],direction_jump_deg=ang(a['n'],b['n']),delta_rate_deg_s=float(np.linalg.norm(b['w']-a['w']))*DEG,ordinary_update_residual_deg_s=float(np.linalg.norm(b['w']-a['w']-c/(TAU+T/2)))*DEG,Ps_CV_innovation_m=float(np.linalg.norm(b['Ps']-a['Ps']-a['Vs']*T)))
                win.append(r)
            out.append(dict(identity=list(ident(e)),uid=int(uid),original=e['original_capture'],rows=win))
    return out

def math_checks():
    rng=np.random.default_rng(31901)
    for _ in range(200):
        n=A.unit(rng.normal(size=3));m=A.unit(rng.normal(size=3));q=Rotation.random(random_state=rng).as_matrix()
        np.testing.assert_allclose(np.linalg.svd(C.D(n)@q,compute_uv=False),[1,.8,.8],rtol=0,atol=1e-14)
        np.testing.assert_allclose(np.linalg.norm(C.D(n)-C.D(m),ord=2),.2*np.linalg.norm(np.cross(n,m)),rtol=0,atol=1e-14)
    return 200

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    verified=C.verify(a.input);nested=C.verify(a.input/'input');source=a.input/'input/prior/evidence/data/main/evidence/capture_events.json.gz';ee=C.read(source);assert len(ee)==931
    assert len({ident(e) for e in ee})==931
    oldrows=C.read(a.input/'studies/range/results/rows.json.gz');oldby={ident(e):e for e in oldrows};results=[];difference=0.;guards=0
    for e in ee:
        r=diagnose(e);v=oldby[ident(e)]['scores']['baseline'];difference=max(difference,abs(v['rate_deg_s']-r['baseline_error_deg_s']),abs(v['slack_cells']-r['baseline_slack_cells']))
        bad=copy.deepcopy(e)
        for k in ('n','w','Ps','Vs'):bad['new'][k]=[float('nan')]*3
        assert classify(e)==classify(bad);guards+=1;results.append(r)
    assert difference<1e-8,difference
    near=[x for x in results if x['near'] and x['family']=='R-77-1'];assert len(near)==79 and sum(x['baseline_feasible'] for x in near)==73
    assert abs(C.metric([x['baseline_error_deg_s'] for x in near])['rms']-.02838914042629035)<1e-12
    groups=make_groups(results);r77=[r for r in results if r['family']=='R-77-1']
    strata={name:standardized([r for r in r77 if r['fresh00']],keys) for name,keys in [('stage_gap_range',['original','candidate_reacq_lag','gap','range_bin']),('scene_stage_gap',['replay','original','candidate_reacq_lag','gap'])]}
    byscene={s:{g:stats([r for r in r77 if r['replay']==s and r['angle_group']==g]) for g in ('small','middle','large')} for s in sorted({r['replay'] for r in r77})}
    tablebins={b:stats([r for r in r77 if r['angle_bin']==b]) for b in ('0-20','20-40','40-60','60-180')}
    pins=[r for r in results if ident(r) in PINNED];assert len(pins)==6
    anchor_events=[r for r in results if (r['replay'],r['eid']) in A.ANCHORS]
    reps=[]
    for g in ('small','large'):
        for first in (True,False):
            cand=sorted([r for r in r77 if r['angle_group']==g and r['original']==first and r['fresh00']],key=lambda r:(r['replay'],r['eid'],r['tick1']))
            if cand:reps.append(cand[0])
    win=windows(a.input,ee,results)
    summary=dict(scope='Descriptive fixed-input game replay state audit; no fitted model or new physical integration; off-axis measured from old estimated Ps, not truth',source_capture_sha256=C.digest(source),input_files_verified=verified,nested_input_files_verified=nested,counts=dict(events=len(results),R77=len(r77),near_R77=len(near),classification_isolation_checks=guards),prior_result_max_difference=difference,geometry_tests=math_checks(),groups=groups,angle_bins=tablebins,by_scene=byscene,standardized=strata,pinned=pins,anchor_events=anchor_events,representatives=reps,anchor_windows=win,parameters=dict(kappa=.8,tau=TAU,small_max_deg=20,large_min_deg=50))
    C.save(a.out/'summary.json',summary);C.save(a.out/'events.json.gz',results);C.save(a.out/'anchor_windows.json',win)
    flatkeys=('replay','eid','uid','tick0','tick1','index0','index1','family','original','candidate_reacq_lag','gap','timer0','offaxis_deg','angle_group','range0_m','mechanical_demand0','baseline_error_deg_s','baseline_axial_deg_s','baseline_transverse_deg_s','baseline_feasible','n0_vs_Ps_deg','prior_vs_new_direction_deg','n1_vs_pointing_deg','old_geo_error_deg_s','Ps_CV_innovation_m','raw_sha0','raw_sha1')
    with (a.out/'EVENT_INDEX.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=flatkeys);w.writeheader();w.writerows({k:r[k] for k in flatkeys} for r in results)
    cols=('baseline_error_deg_s','baseline_axial_deg_s','baseline_transverse_deg_s','n0_vs_Ps_deg','prior_vs_new_direction_deg','n1_vs_pointing_deg','old_geo_error_deg_s','pointing_axis_error_deg_s','crossing_ordinary_negative_control_deg_s')
    compact={name:{g:dict(n=v['n'],feasible=v['feasible'],limit=v['pointing_limit_count'],relative=v['relative_error_energy'],rms={k:v['metrics'][k].get('rms') for k in cols},median={k:v['metrics'][k].get('median') for k in ('baseline_error_deg_s','n0_vs_Ps_deg','prior_vs_new_direction_deg')}) for g,v in gg.items()} for name,gg in groups.items()}
    C.save(a.out/'compact.json',dict(groups=compact,standardized={k:{x:v[x] for x in ('keys','total_weight','pooled_rms')} for k,v in strata.items()}))
    for name in ('R77_all','R77_first','R77_reacq','R77_near','R77_reacq_fresh00_lag0','R77_reacq_fresh00_lag1','R77_reacq_fresh00_lag2'):
        print('ANGLE_GROUP',name,json.dumps(C.serial(compact[name])),flush=True)
    for k,v in strata.items():print('STANDARDIZED',k,json.dumps(C.serial({x:v[x] for x in ('keys','total_weight','pooled_rms')})),flush=True)
    for r in pins+anchor_events[:3]:print('EVENT',json.dumps(C.serial({k:r[k] for k in flatkeys})),flush=True)
    for r in win:print('WINDOW',json.dumps(C.serial(r)),flush=True)
    print('VALIDATION',json.dumps({k:summary[k] for k in ('counts','input_files_verified','nested_input_files_verified','prior_result_max_difference','geometry_tests')}),flush=True)
if __name__=='__main__':main()
