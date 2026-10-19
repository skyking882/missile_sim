#!/usr/bin/env python3
"""Independent arithmetic and selection review, not an independent AI agent."""
from __future__ import annotations
import argparse,collections,gzip,json,math
from pathlib import Path

def read(p):
    with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return json.load(f)
def diff(a,b):return math.sqrt(math.fsum((float(x)-float(y))**2 for x,y in zip(a,b)))
def dump(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False))

def main():
    p=argparse.ArgumentParser();p.add_argument('--prior',type=Path,required=True);p.add_argument('--native',type=Path,required=True);p.add_argument('--out',type=Path,required=True);z=p.parse_args();z.out.mkdir(parents=True,exist_ok=False)
    paths=read(z.prior/'native/paths.json.gz');stream=read(z.native/'transitions.json.gz');stages=read(z.native/'stage29.json.gz');episodes=read(z.native/'episodes.json.gz');summary=read(z.native/'summary.json')
    orig={};keys=lambda p,a,b:(p['replay'],p['eid'],a['index'],b['index'])
    for path in paths:
        for a,b in zip(path['states'],path['states'][1:]):
            if a['code']!=b['code'] and 0<b['tick']-a['tick']<=8:
                key=keys(path,a,b)
                if key in orig:assert orig[key]==(a,b)
                else:orig[key]=(a,b)
    idx={(r['replay'],r['eid'],r['index0'],r['index1']):r for r in stream};assert len(idx)==len(stream)
    maxerr=0.
    for k,(a,b) in orig.items():
        assert k in idx,k
        r=idx[k];assert r['raw_sha0']==a['raw_sha'] and r['raw_sha1']==b['raw_sha']
        assert r['w_equal']==(a['w']==b['w']);assert r['PV_equal']==(a['Ps']==b['Ps'] and a['Vs']==b['Vs'])
        for name,x,y in [('P_change_m',a['Ps'],b['Ps']),('V_change_m_s',a['Vs'],b['Vs'])]:maxerr=max(maxerr,abs(r[name]-diff(x,y)))
    assert maxerr<1e-9,maxerr
    for c,(aa,bb) in {25:(0,0),29:(1,0),31:(1,1)}.items():
        assert c==3*8+aa*4+bb*2+1;assert aa==(c>>2)&1 and bb==(c>>1)&1
    # Stage29 extraction must agree with individually reconstructed boundary rows.
    both31=[]
    for s in stages:
        a,b=s['before'],s['first29'];r=idx[keys(s,a,b)];assert r['code1']==29
        assert s['rows29']>=1
        if s['after'] is not None:
            x,y=s['last29'],s['after'];assert keys(s,x,y) in idx and y['code']!=29
        if s['before']['code']==31 and s['after'] is not None and s['after']['code']==31:
            both31.append(dict(replay=s['replay'],eid=s['eid'],uid=s['uid'],family=s['family'],tick29=b['tick'],tick_after=s['after']['tick'],timer29=b['timer'],PV_changes_before=b['Ps']!=a['Ps'] or b['Vs']!=a['Vs'],w_equal_before=b['w']==a['w'],PV_changes_after=b['Ps']!=s['after']['Ps'] or b['Vs']!=s['after']['Vs'],w_equal_after=b['w']==s['after']['w']))
    sourcecounts={}
    for group,pop in [('all',stream),('R77',[r for r in stream if r['family']=='R-77-1'])]:
        for dest in (29,25):
            rows=[r for r in pop if r['code0']==31 and r['code1']==dest]
            available=[r for r in rows if r.get('fixed_target_comparison',{}).get('packet') is not None]
            rr=[r['fixed_target_comparison']['packet'] for r in available]
            sourcecounts[group+'_'+str(dest)]=dict(n=len(rows),fixed_target_available=len(rr),P_within_5m=sum(v['refreshed_P_error_m']<=5 for v in rr),P_within_50m=sum(v['refreshed_P_error_m']<=50 for v in rr),P_median=sorted(v['refreshed_P_error_m'] for v in rr)[len(rr)//2] if rr else None,old_CV_median=sorted(v['last_fresh_CV_P_error_m'] for v in rr if v['last_fresh_CV_P_error_m'] is not None)[sum(v['last_fresh_CV_P_error_m'] is not None for v in rr)//2] if any(v['last_fresh_CV_P_error_m'] is not None for v in rr) else None)
    exception_rows=[r for r in stream if r['code0']==31 and r['code1']==29 and (not r['w_equal'] or r['PV_equal'])]
    review=dict(scope='Scalar check against independent archived paths plus original native extraction, not new independent trials',archived_transition_pairs=len(orig),native_transition_pairs=len(stream),maximum_scalar_difference=maxerr,code_is_constructed_flags=True,stage29_entries_checked=len(stages),stage29_return_to31_count=len(both31),source_comparison_summary=sourcecounts,exceptions_to_PV_refresh_rate_hold=[{k:r[k] for k in ('replay','eid','uid','family','tick0','tick1','timer0','timer1','flag0','flag1','w_equal','PV_equal','P_change_m','V_change_m_s','w_change_deg_s')} for r in exception_rows],branch_31_29_31=both31)
    dump(z.out/'verification.json',review)
    print('REVIEW',json.dumps({k:v for k,v in review.items() if k not in ('branch_31_29_31','exceptions_to_PV_refresh_rate_hold')}),flush=True)
    print('EXCEPTIONS',json.dumps(review['exceptions_to_PV_refresh_rate_hold']),flush=True)
    print('BRANCHES',json.dumps(both31[:12]),flush=True)
if __name__=='__main__':main()
