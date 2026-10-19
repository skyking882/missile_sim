#!/usr/bin/env python3
"""Diagnostic of already-recorded game state sources, not a detector or guidance implementation."""
import argparse,json
from pathlib import Path
import numpy as np
import check_handover as C
A=C.A

def geom(P,V,p,v):
    d=P-p;r=float(d@d)
    if r<1e-6:raise ValueError('Degenerate geometric range')
    return np.cross(V-v,d)/r

def main():
    p=argparse.ArgumentParser();p.add_argument('--flow',type=Path,required=True);p.add_argument('--out',type=Path,required=True);z=p.parse_args();z.out.mkdir(parents=True,exist_ok=False)
    stages=C.read(z.flow/'stage29.json.gz');rows=[]
    for s in stages:
        b=s['after']
        if b is None or b['code']!=31:continue
        a=s['last29'];T=(b['tick']-a['tick'])/48
        P,V,p0,v0=[np.array(a[k],float) for k in ('Ps','Vs','p','vm')]
        p1,v1,n1,w1=[np.array(b[k],float) for k in ('p','vm','n','w')]
        w0=np.array(a['w']);Q=np.array(b['Q']);D=.8*np.eye(3)+.2*np.outer(n1,n1)
        g0=geom(P,V,p0,v0);g1=geom(P+V*T,V,p1,v1)
        seeds={'saved_rate':w0,'PV_at29':g0,'PV_propagated':g1,'saved_plus_geometric_increment':w0+g1-g0}
        errors={k:C.norm(D@Q@v-w1)*A.RAD for k,v in seeds.items()}
        rows.append({k:s[k] for k in ('replay','eid','uid','family')}|dict(tick0=a['tick'],tick1=b['tick'],flag0=a['flag'],flag1=b['flag'],new_timer=b['timer'],fresh00=a['flag']==b['flag']==0 and b['timer']==0,errors=errors,raw_sha0=a['raw_sha'],raw_sha1=b['raw_sha']))
    groups={}
    for fam in ('all','R-77-1'):
        for gate in ('all','fresh00'):
            rr=[r for r in rows if (fam=='all' or r['family']==fam) and (gate=='all' or r['fresh00'])]
            groups[fam+'/'+gate]=dict(n=len(rr),metrics={k:A.metric([r['errors'][k] for r in rr]) for k in ('saved_rate','PV_at29','PV_propagated','saved_plus_geometric_increment')})
    anchors=[r for r in rows if (r['replay'],r['eid']) in A.ANCHORS]
    result=dict(scope='Fixed input-derived seed diagnostics; no current targetPV, no coefficient fit; current direction remains an output-conditioned projection axis. No rate-cell compatibility claim for PV-derived seeds.',groups=groups,anchors=anchors,rows=rows)
    A.save(z.out/'SEED_RESULTS.json',result)
    print('SEED_GROUPS',json.dumps(A.serial(groups)),flush=True)
    print('SEED_ANCHORS',json.dumps(A.serial(anchors)),flush=True)
if __name__=='__main__':main()
