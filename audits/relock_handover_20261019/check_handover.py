#!/usr/bin/env python3
"""Conditional replay-field comparison, no autonomous seeker or new coefficients."""
import argparse,collections,json,math,sys
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'capture_held_20261017'))
import audit as A
import gzip
RATE_GRID=15/32767
H=1/48
TGRID=10/32767

def read(p):
    with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return json.load(f)
def state(r):
    r=r.copy()
    for k in ('n','w','Q'):r[k]=np.array(r[k],float)
    return r

def cells(w):
    k=np.rint(w/RATE_GRID).astype(int)
    lo=k.astype(float);hi=k.astype(float)
    lo[k<=0]-=1;hi[k>=0]+=1
    return lo,hi

def compatible(M,a,b):
    lo,hi=cells(a);l1,u1=cells(b)
    r=linprog(np.zeros(3),A_ub=np.vstack([M,-M]),b_ub=np.r_[u1,-l1],bounds=list(zip(lo,hi)),method='highs')
    if r.status not in (0,2):raise RuntimeError(r.message)
    if r.success:
        assert np.max(M@r.x-u1)<1e-6 and np.max(l1-M@r.x)<1e-6
    return bool(r.success)

def calculate(meta,a,b):
    a,b=state(a),state(b);T=(b['tick']-a['tick'])*H
    n=b['n'];D=.8*np.eye(3)+.2*np.outer(n,n)
    matrices={'copy':np.eye(3),'Q':b['Q'],'D_Q':D@b['Q'],'D_noQ':D}
    predictions={k:M@a['w'] for k,M in matrices.items()}
    p=A.rotate(a['n'],a['w'],T);predictions['ordinary']=a['w']+A.correction(p,n)/(.078525+T/2)
    errors={k:norm(w-b['w'])*A.RAD for k,w in predictions.items()}
    feas={k:compatible(M,a['w'],b['w']) for k,M in matrices.items()}
    endage=a['timer']+T
    age='below_expiry' if endage<1-2*H-2*TGRID else ('expiry_band' if endage<=1+2*H+2*TGRID else 'above_expiry')
    return {k:meta.get(k) for k in ('replay','eid','uid','family')}|dict(tick0=a['tick'],tick1=b['tick'],index0=a['index'],index1=b['index'],code0=a['code'],code1=b['code'],timer0=a['timer'],timer1=b['timer'],flag0=a['flag'],flag1=b['flag'],fresh00=b['timer']==0 and a['flag']==b['flag']==0,age_group=age,predicted_age=endage,errors=errors,linear_cell_compatible=feas,raw_sha0=a['raw_sha'],raw_sha1=b['raw_sha'])

def norm(x):return float(np.linalg.norm(x))
def stats(rows):
    return dict(n=len(rows),entities=len({(r['replay'],r['eid']) for r in rows}),metrics={k:A.metric([r['errors'][k] for r in rows]) for k in ('copy','Q','D_Q','D_noQ','ordinary')},compatible={k:sum(r['linear_cell_compatible'][k] for r in rows) for k in ('copy','Q','D_Q','D_noQ')})

def main():
    p=argparse.ArgumentParser();p.add_argument('--prior',type=Path,required=True);p.add_argument('--flow',type=Path,required=True);p.add_argument('--out',type=Path,required=True);z=p.parse_args();z.out.mkdir(parents=True,exist_ok=False)
    stages=read(z.flow/'stage29.json.gz');rr=read(next(z.prior.rglob('fresh_return_events.json.gz')));cc=read(next(z.prior.rglob('capture_events.json.gz')))
    rows=[]
    for s in stages:
        if s['after'] is not None and s['after']['code']==31:
            r=calculate(s,s['last29'],s['after']);r['population']='via29';rows.append(r)
    assert sum(r['population']=='via29' for r in rows)==180
    for label,events in [('held31',rr),('from25',cc)]:
        for e in events:
            r=calculate(e,e['old'],e['new']);r['population']=label;rows.append(r)
    groups={}
    for family in ('all','R-77-1'):
        pop=[r for r in rows if family=='all' or r['family']==family]
        for origin in ('via29','held31','from25'):
            for flag in ('all','fresh00'):
                selected=[r for r in pop if r['population']==origin and (flag=='all' or r['fresh00'])]
                groups[family+'/'+origin+'/'+flag]=stats(selected)
                if origin=='held31':
                    for age in ('below_expiry','expiry_band','above_expiry'):groups[family+'/'+origin+'/'+flag+'/'+age]=stats([r for r in selected if r['age_group']==age])
    anchors=[r for r in rows if (r['replay'],r['eid']) in A.ANCHORS and (r['population']=='via29' or r['tick1'] in (11521,11617,14739))]
    summary=dict(scope='Fixed conditional rate reset versus ordinary endpoint relation; current n read as response, no fitting or independent measurement generation',rows=len(rows),groups=groups,anchors=anchors,parameters=dict(kappa=.8,tau=.078525,age_band_tick_padding=2,rate_grid=RATE_GRID),quantization='Closed signed-truncation cells; fixed decoded Q and n; LP only for the four linear candidates')
    A.save(z.out/'summary.json',summary);A.save(z.out/'events.json.gz',rows)
    for k,v in groups.items():
        if k.startswith('R-77-1/') and ('fresh00' in k or k=='R-77-1/via29/all'):print('HANDOVER_GROUP',k,json.dumps(A.serial(v)),flush=True)
    for r in anchors:print('ANCHOR_HANDOVER',json.dumps(A.serial(r)),flush=True)
if __name__=='__main__':main()
