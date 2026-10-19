#!/usr/bin/env python3
"""Independent scalar/path checks and descriptive clock-origin contrasts.
No new fitted timing rules. Counterexamples are examples, not held-out trials.
"""
from __future__ import annotations
import argparse,math
from pathlib import Path
import timing_bounds as T

def nearest_phases(lo,hi,pad=0):
    return [x for x in range(lo-pad+1,hi+pad+1) if divmod(x,48)[1]==1]

def entry_intervals(p,code):
    ss=p['states'];j=next((i for i,s in enumerate(ss) if i>0 and s['code']==code and ss[i-1]['code']!=code),None)
    if j is None:return None
    x,y=ss[j-1],ss[j];a,b=ss[-2:]
    return dict(entry_before_tick=x['tick'],entry_first_tick=y['tick'],return_before_tick=a['tick'],return_first_tick=b['tick'],wait_lower_s=max(0,(a['tick']-y['tick'])/48),wait_upper_s=(b['tick']-x['tick'])/48,wait_upper_2tick_s=(b['tick']-x['tick']+2)/48+2*T.Q)

def main():
    cli=argparse.ArgumentParser();cli.add_argument('--native',type=Path,required=True);cli.add_argument('--interval',type=Path,required=True);cli.add_argument('--out',type=Path,required=True);a=cli.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    paths=T.read(a.native/'paths.json.gz');direct=T.read(a.native/'direct_returns.json.gz');endpoint=T.read(a.interval/'returns.json.gz');summary=T.read(a.native/'summary.json')
    by={T.key(r):r for r in direct};assert len(by)==1035
    for r in endpoint:assert r==by[T.key(r)]
    counts=dict(direct_endpoint_comparisons=len(endpoint),native_paths=len(paths),path_states_checked=0,phase_interval_checks=0)
    strict=[];complete=[];missorigins=[];maxerr=0.
    for p in paths:
        states=p['states'];codes=[]
        for s in states:
            if not codes or codes[-1]!=s['code']:codes.append(s['code'])
        assert codes==p['codes'];f=states[0];b=states[-1];old=states[-2]
        assert f['code']==b['code']==31 and f['timer']==b['timer']==0
        assert all(0<y['tick']-x['tick']<=8 for x,y in zip(states,states[1:]))
        assert len({s['index'] for s in states})==len(states)
        gap=(b['tick']-f['tick'])/48
        maxerr=max(maxerr,abs(gap-p['fresh_to_fresh_s']))
        assert p['phase1_in_transition']==bool(nearest_phases(old['tick'],b['tick']))
        assert p['phase1_padded']==bool(nearest_phases(old['tick'],b['tick'],2));counts['phase_interval_checks']+=2
        held=[s for s in states[1:-1] if s['code']==31 and s['timer']>0]
        constant_PV=all(s['Ps']==f['Ps'] and s['Vs']==f['Vs'] for s in held)
        constant_rate=all(s['w']==f['w'] for s in held)
        assert constant_PV==p['held_PV_constant'] and constant_rate==p['held_rate_constant']
        counts['path_states_checked']+=len(states)
        if p['coherent_sustained']:
            assert p['codes']==[31] and len(held)>=2 and max(s['timer'] for s in held)>=.1
            assert all(s['flag']==0 for s in states) and constant_PV and constant_rate
            for x,y in zip(states,states[1:-1]):assert abs((y['timer']-x['timer'])-(y['tick']-x['tick'])/48)<2*T.Q
            strict.append({k:v for k,v in p.items() if k!='states'}|{'output_PV_changed':b['Ps']!=f['Ps'] or b['Vs']!=f['Vs'],'output_rate_changed':b['w']!=f['w']})
        e25=entry_intervals(p,25);e29=entry_intervals(p,29)
        if e25:
            complete.append({k:p[k] for k in ('replay','eid','uid','family','return_tick','codes','phase1_in_transition')}|{'entry25':e25,'entry29':e29})
        if p['path_kind']=='same31':
            m=states[1]
            # A first missed output is only a bracket for physical loss.
            missorigins.append({k:p[k] for k in ('replay','eid','uid','family','fresh_to_fresh_s')}|{'loss_window_ticks':[f['tick'],m['tick']],'resumption_window_ticks':[old['tick'],b['tick']],'max_wait_from_possible_loss_s':(b['tick']-f['tick'])/48+2/48+2*T.Q})
    def strictstats(rr):
        return dict(n=len(rr),below_one=sum(p['waiting_since_loss_upper_s']<1 for p in rr),outside_phase1_even_2ticks=sum(not p['phase1_padded'] for p in rr),below_one_and_outside_phase=sum(p['waiting_since_loss_upper_s']<1 and not p['phase1_padded'] for p in rr),PV_changed_on_return=sum(p['output_PV_changed'] for p in rr),rate_changed_on_return=sum(p['output_rate_changed'] for p in rr),entities=len({(p['replay'],p['eid']) for p in rr}),scenes=len({p['replay'] for p in rr}))
    def entrystats(rr):
        vals=[p['entry25']['wait_upper_s'] for p in rr]
        return dict(n=len(rr),within_one_since_entry25_with_2ticks=sum(p['entry25']['wait_upper_2tick_s']<1 for p in rr),entry25_wait_upper_median=T.percentile(vals,.5),entry25_wait_upper_min=min(vals) if vals else None,entry25_wait_upper_max=max(vals) if vals else None,phase1_compatible=sum(p['phase1_in_transition'] for p in rr))
    sr=[p for p in strict if p['family']=='R-77-1'];cr=[p for p in complete if p['family']=='R-77-1']
    examples=summary['examples'];view=[]
    for p in examples:
        row={k:v for k,v in p.items() if k!='states'}
        row['timeline']=[{k:s[k] for k in ('tick','packet_ms','code','timer','flag','index','raw_sha')} for s in p['states']]
        view.append(row)
    anchors=[]
    for p in summary['anchor_paths']:
        st=p['states'];timeline=[st[0]]
        for i,s in enumerate(st[1:],1):
            if i==len(st)-1 or s['code']!=st[i-1]['code']:timeline.append(s)
        anchors.append({k:v for k,v in p.items() if k!='states'}|{'mode_timeline':[T.brief(s) for s in timeline],'entry25':entry_intervals(p,25),'entry29':entry_intervals(p,29)})
    nearsecond=[p for p in paths if p['family']=='R-77-1' and p['path_kind']=='same31' and .95<=p['fresh_to_fresh_s']<=1.1]
    out=dict(scope='Independent scalar checks and observed timing-origin intervals; no direct reflector-ID or internal retry-clock source',counts=counts,max_tick_duration_difference=maxerr,strict_coherent=dict(all=strictstats(strict),R77=strictstats(sr)),entry25_summary=dict(all=entrystats(complete),R77=entrystats(cr)),entry25_short_examples=sorted(cr,key=lambda p:p['entry25']['wait_upper_s'])[:8],sustained_examples=view,anchor_paths=anchors,near_one_second_R77=dict(n=len(nearsecond),all_flags0=sum(p['flags_all_zero'] for p in nearsecond),coherent=sum(p['coherent_sustained'] for p in nearsecond),phase1_padded=sum(p['phase1_padded'] for p in nearsecond),durations_ticks=sorted({round(48*p['fresh_to_fresh_s']) for p in nearsecond})))
    T.save(a.out/'verification.json',out);T.save(a.out/'strict_events.json.gz',strict);T.save(a.out/'entry25_intervals.json.gz',complete)
    for k in ('counts','strict_coherent','entry25_summary','near_one_second_R77','sustained_examples','anchor_paths','entry25_short_examples'):print('REVIEW',k,__import__('json').dumps(out[k]),flush=True)
if __name__=='__main__':main()
