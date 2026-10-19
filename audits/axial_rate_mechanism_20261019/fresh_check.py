#!/usr/bin/env python3
"""No fitted gains: discriminate named transport contracts in game output."""
from __future__ import annotations
import argparse,collections,json,math,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import analyze as A
DIR_BOUND=2*math.sqrt(3)/(32767-math.sqrt(3))


def uncertainty(r):return DIR_BOUND*(np.linalg.norm(r['w'])+math.sqrt(3)*A.QW)
def overlap(a,b):return max(a[0],b[0])<=min(a[1],b[1])+1e-14

def necessary_pair(s,r):
    n0,n1=A.unit(s['n']),A.unit(r['n']);w0,w1=A.v(s['w']),A.v(r['w']);T=(r['tick']-s['tick'])/48
    p=A.propagate(n0,w0,T);cross=np.cross(n1,p);sn=float(np.linalg.norm(cross));co=float(n1@p)
    if sn<1e-14 and co<0:raise ValueError('Antipodal correction')
    c=cross*(math.atan2(sn,co)/sn) if sn>1e-14 else np.zeros(3)
    g=1/(.078525+T/2);no=w0+g*c;transport=Rotation.from_rotvec(-c).apply(w0)+g*c
    preds={'add_without_transport':float(n1@w0),'transport_old_rate':float(n0@w0)}
    actual=float(n1@w1);target=A.interval(n1,w1);bands={'add_without_transport':A.interval(n1,w0),'transport_old_rate':A.interval(n0,w0)}
    residuals={k:(p-actual)*A.DEG for k,p in preds.items()};expanded={}
    u0,u1=uncertainty(s),uncertainty(r)
    for k,b in bands.items():expanded[k]=overlap((b[0]-u0,b[1]+u0),(target[0]-u1,target[1]+u1))
    return dict(tick0=s['tick'],tick1=r['tick'],index0=s['index'],index1=r['index'],raw_sha0=s['raw_sha'],raw_sha1=r['raw_sha'],gap=r['tick']-s['tick'],lambda0_deg_s=float(n0@w0)*A.DEG,lambda1_deg_s=actual*A.DEG,lambda_prediction_deg_s={k:v*A.DEG for k,v in preds.items()},axial_residuals_deg_s=residuals,scalar_compatible={k:overlap(b,target) for k,b in bands.items()},expanded_scalar_compatible=expanded,prediction_separation_deg_s=abs(preds['add_without_transport']-preds['transport_old_rate'])*A.DEG,ordinary_full_error_deg_s=float(np.linalg.norm(no-w1))*A.DEG,transport_full_error_deg_s=float(np.linalg.norm(transport-w1))*A.DEG,direction_innovation_deg=A.ang(p,n1),kinematic_prediction_degrees=A.ang(n0,p))

def run_summary(key,rows):
    intervals=[];expanded=[];lambdas=[];n0=A.unit(rows[0]['n']);lam0=float(n0@A.v(rows[0]['w']));pred=lam0;errs=[]
    for i,r in enumerate(rows):
        n=A.unit(r['n']);w=A.v(r['w']);l,h=A.interval(n,w);b=uncertainty(r);intervals.append((l,h));expanded.append((l-b,h+b));lambdas.append(float(n@w))
        if i:
            p=rows[i-1];pred+=float(A.v(p['w'])@(n-A.unit(p['n'])))
        errs.append((pred-float(n@w))*A.DEG)
    lo=max(b[0] for b in intervals);hi=min(b[1] for b in intervals);loe=max(b[0] for b in expanded);hie=min(b[1] for b in expanded)
    return dict(replay=key[0],eid=key[1],uid=key[2],n=len(rows),tick0=rows[0]['tick'],tick1=rows[-1]['tick'],duration_s=(rows[-1]['tick']-rows[0]['tick'])/48,lambda_initial_deg_s=lambdas[0]*A.DEG,lambda_final_deg_s=lambdas[-1]*A.DEG,lambda_range_deg_s=(max(lambdas)-min(lambdas))*A.DEG,constant_lambda_compatible=lo<=hi,constant_lambda_expanded_compatible=loe<=hie,constant_interval_gap_deg_s=max(0,lo-hi)*A.DEG,constant_expanded_gap_deg_s=max(0,loe-hie)*A.DEG,no_transport_cumulative_error=A.metric(errs),first_sha=rows[0]['raw_sha'],last_sha=rows[-1]['raw_sha'])

def source_tests():
    root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'src'))
    from aim120_model.q_angular_observer import QCaptureAngularObserver
    rng=np.random.default_rng(912);maxerr=0
    for _ in range(100):
        p=A.unit(rng.normal(size=3));w=rng.normal(size=3)*.1;m=A.unit(p+rng.normal(size=3)*.03)
        obs=QCaptureAngularObserver();obs.direction=tuple(p);obs.rate=tuple(w);obs._correct(tuple(m),1/48)
        nn=A.unit(obs.direction);error=abs(float(nn@A.v(obs.rate))-float(nn@w));maxerr=max(maxerr,error)
        assert error<1e-12
    return dict(current_source_correction_contract_cases=100,max_axial_identity_difference_rad_s=maxerr)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--memory',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);z=ap.parse_args();z.out.mkdir(parents=True,exist_ok=False)
    nfiles=A.verify(z.memory);fresh=A.read(z.memory/'input/fresh.json.gz');adj=A.read(z.memory/'adjacency/adjacency.json.gz');ad={(r['replay'],r['index']):r for r in adj};groups=collections.defaultdict(list)
    for x in fresh:groups[x['replay'],x['eid'],x['uid']].append(x['record'])
    pairs=[];runs=[];excluded=collections.Counter()
    for key,rr in groups.items():
        segment=[]
        for r in rr:
            if r['flag']!=0:
                if len(segment)>=20:runs.append(run_summary(key,segment))
                segment=[];excluded['flag']+=1;continue
            prev=segment[-1] if segment else None
            adjacent=prev is not None and ad[key[0],r['index']]['previous_same_eid_index']==prev['index'] and 0<r['tick']-prev['tick']<=8
            if prev is not None and not adjacent:
                if len(segment)>=20:runs.append(run_summary(key,segment))
                segment=[];excluded['break']+=1
            if adjacent:
                assert ad[key[0],r['index']]['raw_sha']==r['raw_sha']
                pairs.append(dict(replay=key[0],eid=key[1],uid=key[2],**necessary_pair(prev,r)))
            segment.append(r)
        if len(segment)>=20:runs.append(run_summary(key,segment))
    cohorts={'all':pairs,'gap_at_most_two':[r for r in pairs if r['gap']<=2],'separation_above_005':[r for r in pairs if r['prediction_separation_deg_s']>.05]}
    stats={}
    for name,rr in cohorts.items():
        stats[name]=dict(n=len(rr),entities=len({(r['replay'],r['eid']) for r in rr}),axial_residuals={k:A.metric([r['axial_residuals_deg_s'][k] for r in rr]) for k in ('add_without_transport','transport_old_rate')},quantization_compatible={k:sum(r['scalar_compatible'][k] for r in rr) for k in ('add_without_transport','transport_old_rate')},expanded_compatible={k:sum(r['expanded_scalar_compatible'][k] for r in rr) for k in ('add_without_transport','transport_old_rate')},full_errors={k:A.metric([r[k] for r in rr]) for k in ('ordinary_full_error_deg_s','transport_full_error_deg_s')})
    out=dict(scope='Gain-free necessary identities using current n as condition, not autonomous observer. Expanded intervals bound uncertain normalized3xint16 directions but not every hidden timing or measurement source.',source_files_verified=nfiles,raw_fresh=len(fresh),raw_adjacency_identifiers=len(adj),cohorts=stats,long_runs=dict(n=len(runs),constant_lambda_compatible=sum(r['constant_lambda_compatible'] for r in runs),constant_lambda_expanded_compatible=sum(r['constant_lambda_expanded_compatible'] for r in runs)),anchor_runs=[r for r in runs if (r['replay'],r['eid']) in A.ANCHORS],worst_constancy=sorted(runs,key=lambda r:r['constant_expanded_gap_deg_s'],reverse=True)[:10],source_tests=source_tests(),excluded=dict(excluded))
    A.save(z.out/'summary.json',out);A.save(z.out/'fresh_pairs.json.gz',pairs);A.save(z.out/'fresh_runs.json.gz',runs)
    print('FRESH_RESULTS',json.dumps(A.clean(out)),flush=True)
if __name__=='__main__':main()
