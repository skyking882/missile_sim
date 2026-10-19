#!/usr/bin/env python3
"""Read-only timing/path validation on War Thunder native export records."""
from __future__ import annotations
import argparse,collections,gzip,json,sys
from pathlib import Path
import numpy as np
import timing_bounds as T
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'capture_held_20261017'))
import audit as A


def compact(r):return A.serial(T.brief(r))
def event_id(rep,eid,a,b):return(rep,int(eid),int(a['tick']),int(b['tick']))

def recovery_path(meta,path,end):
    states=path+[end];codes=[]
    for r in states:
        if not codes or codes[-1]!=r['code']:codes.append(r['code'])
    f=states[0];old=states[-2];held=[r for r in path[1:] if r['code']==31 and r['timer']>0]
    kind='via25' if 25 in codes else ('via29' if 29 in codes else ('same31' if codes==[31] else 'other'))
    continuous=True
    for x,y in zip(states,states[1:-1]):
        if x['code']!=31 or y['code']!=31 or y['timer']<=0 or abs(y['timer']-x['timer']-(y['tick']-x['tick'])*T.H)>2*T.Q:continuous=False
    # The zip above deliberately excludes the final accepted state.
    pvhold=all(np.array_equal(r['Ps'],f['Ps']) and np.array_equal(r['Vs'],f['Vs']) for r in held)
    whold=all(np.array_equal(r['w'],f['w']) for r in held)
    flag0=all(r['flag']==0 for r in states)
    inter=(end['tick']-f['tick'])*T.H;packet=(end['packet_ms']-f['packet_ms'])/1000
    out={k:meta[k] for k in ('replay','eid','uid','family')}
    out.update(path_kind=kind,codes=codes,last_fresh_tick=f['tick'],first_nonfresh_tick=states[1]['tick'],return_tick=end['tick'],preceding_tick=old['tick'],last_fresh_index=f['index'],return_index=end['index'],held_rows=len(held),total_rows=len(states),max_held_timer=max([r['timer'] for r in held],default=0.),flags_all_zero=flag0,stable_designation_key=len({r['key'] for r in states})==1,all_held_timer_continuous=continuous,held_PV_constant=pvhold,held_rate_constant=whold,fresh_to_fresh_s=inter,packet_fresh_to_fresh_s=packet,waiting_since_loss_upper_s=inter+2*T.H+2*T.Q,last_observation_estimated_gap_s=(end['tick']-old['tick'])*T.H+old['timer'],phase1_in_transition=T.phase_in(old['tick'],end['tick']),phase1_padded=T.phase_in(old['tick'],end['tick'],padding=2),return_phase=end['tick']%48,raw_sha_first=f['raw_sha'],raw_sha_last=end['raw_sha'],old_timer=old['timer'],return_key=end['key'],last_fresh_key=f['key'],end_flag=end['flag'],start_flag=f['flag'])
    out['sustained']=kind=='same31' and len(held)>=2 and out['max_held_timer']>=.1 and flag0 and continuous
    out['coherent_sustained']=out['sustained'] and pvhold and whold
    out['states']=[compact(r) for r in states]
    return out

def pathstats(rows):
    return dict(n=len(rows),entities=len({(r['replay'],r['eid']) for r in rows}),scenes=len({r['replay'] for r in rows}),below_one_tickbound=sum(r['waiting_since_loss_upper_s']<1 for r in rows),below_one_packetbound=sum(r['packet_fresh_to_fresh_s']+2*T.H+2*T.Q<1 for r in rows),phase1_compatible=sum(r['phase1_in_transition'] for r in rows),phase1_padded=sum(r['phase1_padded'] for r in rows),fresh_to_fresh_seconds={k:T.percentile([r['fresh_to_fresh_s'] for r in rows],q) for k,q in [('minimum',0),('median',.5),('p90',.9),('maximum',1)]},stable_designation=sum(r['stable_designation_key'] for r in rows),PV_constant=sum(r['held_PV_constant'] for r in rows),rate_constant=sum(r['held_rate_constant'] for r in rows))

def main():
    cli=argparse.ArgumentParser();cli.add_argument('--reference',type=Path,required=True);cli.add_argument('--derived',type=Path,required=True);cli.add_argument('--prior',type=Path,required=True);cli.add_argument('--out',type=Path,required=True);a=cli.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    caps=T.read(a.prior/'capture_events.json.gz');returns=T.read(a.prior/'fresh_return_events.json.gz')
    cohort={(e['replay'],e['eid']):{k:e[k] for k in ('replay','eid','uid','family')} for e in caps};assert len(cohort)==501
    capref={T.key(e):e for e in caps};retref={T.key(e):e for e in returns}
    mp=next(a.reference.rglob('missile_seeker_fieldmap.json'));maps=T.read(mp)['field_maps']
    streamfiles=sorted(a.derived.rglob('weapon_raw.jsonl.gz'));assert len(streamfiles)==5
    direct=[];mid=[];captures=[];paths=[];exits=[];seen_caps=set();seen_ret=set();checks=0;counts=collections.Counter();sourcehash={};open_end=[]
    for file in streamfiles:
        rep=file.parent.name;sourcehash[rep]=T.digest(file);wanted={eid:meta for (rr,eid),meta in cohort.items() if rr==rep};cache={eid:dict(prev=None,pending=None) for eid in wanted}
        with gzip.open(file,'rt') as f:
            for index,line in enumerate(f):
                raw=json.loads(line);counts['native_records']+=1;eid=int(raw['fields']['eid'])
                if eid not in wanted:continue
                counts['selected_records']+=1;st=cache[eid];b=A.parse(raw,maps,index)
                if b is None:
                    counts['unsupported_layout']+=1;st['prev']=None;st['pending']=None;continue
                x=st['prev'];meta=wanted[eid]
                if x is not None:
                    gap=b['tick']-x['tick']
                    if not 0<gap<=8:
                        counts['nonpositive_or_large_gap']+=1;st['pending']=None;x=None
                if x is not None:
                    key=event_id(rep,eid,x,b)
                    if x['code']==b['code']==31 and x['timer']>0 and b['timer']==0:
                        r={**meta,**T.bound(x,b)};direct.append(r)
                        assert key in retref,('unexpected direct',key)
                        old=retref[key]
                        assert x['index']==old['old']['index'] and b['index']==old['new']['index']
                        assert x['raw_sha']==old['old']['raw_sha'] and b['raw_sha']==old['new']['raw_sha'];seen_ret.add(key);checks+=2
                    if x['code']==25 and b['code']==31:
                        assert key in capref,('unexpected capture',key)
                        old=capref[key];assert x['raw_sha']==old['old']['raw_sha'] and b['raw_sha']==old['new']['raw_sha'];seen_caps.add(key);checks+=2
                        captures.append({**meta,**T.bound(x,b),'original':old['original_capture']})
                    if x['code']==29 and b['code']==31:mid.append({**meta,**T.bound(x,b)})
                    if x['code']==31 and b['code']!=31:
                        exits.append({**meta,**T.bound(x,b),'previous_timer_age_at_exit_s':x['timer']+gap*T.H})
                    # Build an observed nonfresh path from an actual fresh record.
                    if st['pending'] is None and x['code']==31 and x['timer']==0 and not(b['code']==31 and b['timer']==0):st['pending']=[x]
                    if st['pending'] is not None:
                        if b['code']==31 and b['timer']==0:
                            paths.append(recovery_path(meta,st['pending'],b));st['pending']=None
                        else:st['pending'].append(b)
                st['prev']=b
        for eid,st in cache.items():
            if st['pending']:
                p=st['pending'];open_end.append({**wanted[eid],'start_tick':p[0]['tick'],'last_tick':p[-1]['tick'],'records':len(p)})
        print('SCENE',rep,'direct',len(direct),'paths',len(paths),'native_records',counts['native_records'],flush=True)
    assert seen_caps==set(capref),(len(seen_caps),len(capref))
    assert seen_ret==set(retref),(len(seen_ret),len(retref))
    rr=[p for p in paths if p['family']=='R-77-1']
    selections={}
    for label,pop in [('all',paths),('R77',rr)]:
        for kind in ('same31','via29','via25','other'):
            selections[label+'_'+kind]=[p for p in pop if p['path_kind']==kind]
        selections[label+'_sustained_flag0']=[p for p in pop if p['sustained']]
        selections[label+'_coherent_sustained']=[p for p in pop if p['coherent_sustained']]
    examples=[]
    for scene in sorted({p['replay'] for p in rr}):
        cand=sorted([p for p in rr if p['replay']==scene and p['coherent_sustained'] and p['waiting_since_loss_upper_s']<1],key=lambda p:(p['fresh_to_fresh_s'],p['eid'],p['return_tick']))
        if cand:examples.append(cand[0])
    anchors=[p for p in paths if (p['replay'],p['eid']) in A.ANCHORS]
    # Regroup by known observed code, not presumed physical state names.
    exitgroups={str(code):dict(n=sum(e['code1']==code for e in exits),timer0_min=min([e['timer0'] for e in exits if e['code1']==code],default=None),timer0_median=T.percentile([e['timer0'] for e in exits if e['code1']==code],.5),timer0_max=max([e['timer0'] for e in exits if e['code1']==code],default=None)) for code in sorted({e['code1'] for e in exits})}
    summary=dict(scope='Original native-export reread with exact fixed endpoint matching; no new WRPL decode, fitting or target identity inference',counts=dict(counts),endpoint_sha_checks=checks,direct_return_count=len(direct),intermediate29_return_count=len(mid),captures_count=len(captures),recovered_paths=len(paths),open_or_censored_endpaths=len(open_end),source_sha256=sourcehash,fieldmap_sha256=T.digest(mp),direct_return_groups=T.group(direct),mid29_groups=T.group(mid),path_groups={k:pathstats(v) for k,v in selections.items()},exit_code_groups=exitgroups,by_scene_coherent_sustained={s:pathstats([p for p in rr if p['replay']==s and p['coherent_sustained']]) for s in sorted({p['replay'] for p in rr})},examples=examples,anchor_paths=anchors)
    A.save(a.out/'summary.json',summary);A.save(a.out/'paths.json.gz',paths);A.save(a.out/'direct_returns.json.gz',direct);A.save(a.out/'mid29_returns.json.gz',mid);A.save(a.out/'captures.json.gz',captures);A.save(a.out/'exits.json.gz',exits);A.save(a.out/'censored_endpaths.json',open_end)
    compact={k:v for k,v in summary.items() if k not in ('examples','anchor_paths')}
    compact['examples']=[{k:v for k,v in e.items() if k!='states'} for e in examples]
    compact['anchor_paths']=[{k:v for k,v in e.items() if k!='states'} for e in anchors]
    A.save(a.out/'compact.json',compact)
    print('NATIVE_SUMMARY',json.dumps(A.serial(compact)),flush=True)
if __name__=='__main__':main()
