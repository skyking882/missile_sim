#!/usr/bin/env python3
"""Timing diagnostics for fixed serialized video-game events, not a seeker model."""
from __future__ import annotations
import argparse,gzip,hashlib,json,math
from pathlib import Path
H=1/48
Q=10/32767

def read(p):
    with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return json.load(f)
def save(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    with (gzip.open(p,'wt') if p.suffix=='.gz' else p.open('w')) as f:json.dump(data,f,ensure_ascii=False,separators=(',',':'),allow_nan=False)
def digest(p):
    s=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):s.update(chunk)
    return s.hexdigest()
def percentile(v,q):
    if not v:return None
    a=sorted(v);x=(len(a)-1)*q;i=int(x);f=x-i
    return a[i]*(1-f)+a[min(i+1,len(a)-1)]*f

def phase_in(a,b,phase=1,padding=0):
    lo,hi=int(a)-padding,int(b)+padding
    formula=(hi-phase)//48>(lo-phase)//48
    direct=any(k%48==phase for k in range(lo+1,hi+1))
    assert formula==direct
    return formula

def key(e):return(e['replay'],int(e['eid']),int(e['tick0']),int(e['tick1']))
def brief(r):
    return {k:r[k] for k in ('index','tick','packet_ms','code','timer','flag','flags','timer_int','key','raw_sha','n','w','Ps','Vs') if k in r}

def bound(a,b):
    dt=(b['tick']-a['tick'])*H;d=dt+a['timer']-b['timer']
    return dict(tick0=a['tick'],tick1=b['tick'],gap=b['tick']-a['tick'],timer0=a['timer'],timer1=b['timer'],flag0=a['flag'],flag1=b['flag'],code0=a['code'],code1=b['code'],packet_gap_s=(b['packet_ms']-a['packet_ms'])/1000,observation_gap_s=d,observation_gap_lower_s=max(0,d-2*Q),observation_gap_upper_s=d+2*Q,observation_gap_upper_with_2tick_slack_s=d+2*Q+2*H,phase1_in_transition=phase_in(a['tick'],b['tick']),phase1_with_2tick_slack=phase_in(a['tick'],b['tick'],padding=2),end_phase=b['tick']%48,raw_sha0=a['raw_sha'],raw_sha1=b['raw_sha'],index0=a['index'],index1=b['index'])

def stats(rows):
    vals=[r['observation_gap_s'] for r in rows]
    bins=[.1,.25,.5,.9,1.1,float('inf')];labels=['<=.1','.1-.25','.25-.5','.5-.9','.9-1.1','>1.1'];counts={x:0 for x in labels}
    for v in vals:
        for cap,label in zip(bins,labels):
            if v<=cap:counts[label]+=1;break
    histogram=[sum(r['end_phase']==p for r in rows) for p in range(48)]
    return dict(n=len(rows),entities=len({(r['replay'],r['eid']) for r in rows}),scenes=len({r['replay'] for r in rows}),below_one_with_timer_bound=sum(r['observation_gap_upper_s']<1 for r in rows),below_one_with_2tick_slack=sum(r['observation_gap_upper_with_2tick_slack_s']<1 for r in rows),phase1_compatible=sum(r['phase1_in_transition'] for r in rows),phase1_compatible_with_2tick_slack=sum(r['phase1_with_2tick_slack'] for r in rows),seconds=dict(min=min(vals) if vals else None,p10=percentile(vals,.1),median=percentile(vals,.5),p90=percentile(vals,.9),max=max(vals) if vals else None),duration_bins=counts,endpoint_phase_histogram=histogram)

def group(rows):
    sets={'all':rows,'R77':[r for r in rows if r['family']=='R-77-1'],'flag00':[r for r in rows if r['flag0']==r['flag1']==0],'R77_flag00':[r for r in rows if r['family']=='R-77-1' and r['flag0']==r['flag1']==0],'R77_flag11':[r for r in rows if r['family']=='R-77-1' and r['flag0']==r['flag1']==1],'R77_flagchange':[r for r in rows if r['family']=='R-77-1' and r['flag0']!=r['flag1']]}
    return {n:stats(rr) for n,rr in sets.items()}

def main():
    cli=argparse.ArgumentParser();cli.add_argument('--input',type=Path,required=True);cli.add_argument('--out',type=Path,required=True);a=cli.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    man=read(a.input/'MANIFEST.json')
    for r in man:
        p=a.input/r['path'];assert p.stat().st_size==r['bytes'] and digest(p)==r['sha256']
    returns=read(a.input/'fresh_return_events.json.gz');caps=read(a.input/'capture_events.json.gz');assert len(returns)==1035 and len(caps)==931
    rows=[]
    for e in returns:
        x,y=e['old'],e['new'];assert x['code']==y['code']==31 and x['timer']>0 and y['timer']==0
        rows.append({k:e[k] for k in ('replay','eid','uid','family')}|bound(x,y))
    cc={}
    for n,rr in [('first',[e for e in caps if e['original_capture']]),('subsequent',[e for e in caps if not e['original_capture']])]:
        cc[n]=dict(n=len(rr),phase1_compatible=sum(phase_in(e['tick0'],e['tick1']) for e in rr),phase1_padded=sum(phase_in(e['tick0'],e['tick1'],padding=2) for e in rr))
    output=dict(scope='Previous1035 saved held→fresh pairs; timer-based intervals only, no full-path validation yet',input_manifest_files=len(man),input_sha256={x:digest(a.input/x) for x in ('fresh_return_events.json.gz','capture_events.json.gz')},groups=group(rows),captures=cc,by_scene={s:stats([r for r in rows if r['family']=='R-77-1' and r['replay']==s]) for s in sorted({r['replay'] for r in rows})},examples=sorted([r for r in rows if r['family']=='R-77-1' and r['flag0']==r['flag1']==0],key=lambda r:(r['observation_gap_s'],key(r)))[:15])
    save(a.out/'summary.json',output);save(a.out/'returns.json.gz',rows)
    print('INTERVAL_GROUPS',json.dumps(output['groups']),flush=True)
    print('CAPTURE_PHASES',json.dumps(cc),flush=True)
    print('EXAMPLES',json.dumps(output['examples']),flush=True)
if __name__=='__main__':main()
