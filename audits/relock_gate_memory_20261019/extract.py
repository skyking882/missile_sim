#!/usr/bin/env python3
"""Read-only native-export selection. No solver execution or learned thresholds."""
from __future__ import annotations
import argparse,collections,gzip,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'source_helper'))
import audit as A

def read(p):
    with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return json.load(f)
def brief(r):return A.serial(r)
def finish(ep,exit_row,why):
    ep['exit']=None if exit_row is None else brief(exit_row);ep['exit_reason']=why
    held=ep['held'];seed=ep['seed'];seq=[seed]+held
    ep['timer_continuous']=all(abs((b['timer']-a['timer'])-(b['tick']-a['tick'])/48)<=2*10/32767 for a,b in zip(seq,seq[1:]))
    ep['flags_zero']=all(r['flag']==0 for r in seq)
    ep['PV_rate_held']=all(all(r[k]==seed[k] for k in ('Ps','Vs','w')) for r in held)
    ep['key_stable']=all(r.get('key')==seed.get('key') for r in seq+([] if ep['exit'] is None else [ep['exit']]))
    ep['strict']=len(held)>=2 and max(r['timer'] for r in held)>=.1 and ep['flags_zero'] and ep['timer_continuous'] and ep['PV_rate_held']
    return ep

def main():
    p=argparse.ArgumentParser();p.add_argument('--derived',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--out',type=Path,required=True);z=p.parse_args();z.out.mkdir(parents=True,exist_ok=False)
    dp=next(z.reference.rglob('server_dataset.json.gz'));mp=next(z.reference.rglob('missile_seeker_fieldmap.json'));d=read(dp);infos=[r for r in d['infos'] if r['family']=='R-77-1'];maps=read(mp)['field_maps'];meta={(r['replay'],int(r['eid'])):r for r in infos}
    sources=[dict(path=str(dp),sha256=A.sha(dp)),dict(path=str(mp),sha256=A.sha(mp))]
    spells=[];fresh=[];anchors={};units={};rockets={};total=0;selected=0;breaks=[];layout_counts=collections.Counter();kept=0
    files=sorted(z.derived.rglob('weapon_raw.jsonl.gz'));assert len(files)==5
    for path in files:
        rep=path.parent.name;wanted={eid for rr,eid in meta if rr==rep};by=collections.defaultdict(list);sources.append(dict(path=str(path),sha256=A.sha(path)))
        with gzip.open(path,'rt') as f:
            for idx,line in enumerate(f):
                raw=json.loads(line);total+=1;eid=int(raw['fields']['eid'])
                if eid not in wanted:continue
                selected+=1;layout_counts[int(raw['fields']['skipped_bits'])]+=1;r=A.parse(raw,maps,idx)
                by[eid].append(r)
        up=path.with_name('unit_history.json.gz');ud=read(up);sources.append(dict(path=str(up),sha256=A.sha(up)))
        targetids={int(m['target_eid']) for (rr,eid),m in meta.items() if rr==rep and m.get('target_eid') is not None}
        units[rep]=[u for u in ud['units'] if int(u['eid']) in targetids]
        # Preserve object/lifetime metadata without pretending birth coordinates
        # are current chaff trajectories. Histories are retained when present.
        rockets[rep]=ud.get('rockets',[])
        for eid,rows in by.items():
            m=meta[rep,eid];previous=None;lastfresh=None;priorfresh=None;ep=None
            if (rep,eid) in A.ANCHORS:anchors[str(A.ANCHORS[rep,eid])]=[brief(r) for r in rows if r is not None]
            for r in rows:
                if r is None or (previous is not None and not 0<r['tick']-previous['tick']<=8):
                    if ep is not None:spells.append(finish(ep,None,'clock_or_layout_break'));ep=None
                    if r is not None and previous is not None:breaks.append(dict(replay=rep,eid=eid,index0=previous['index'],index1=r['index'],tick0=previous['tick'],tick1=r['tick']))
                    previous=lastfresh=priorfresh=None
                    if r is None:continue
                if ep is not None:
                    if r['code']==31 and r['timer']>0:
                        ep['held'].append(brief(r))
                    else:
                        spells.append(finish(ep,r,'fresh_return' if r['code']==31 and r['timer']==0 else 'code_exit'));ep=None
                elif previous is not None and previous['code']==31 and previous['timer']==0 and r['code']==31 and r['timer']>0:
                    ep=dict(replay=rep,eid=eid,uid=m['uid'],family=m['family'],epoch_tick=m['epoch_tick'],target_eid=m.get('target_eid'),seed=brief(previous),previous_fresh=None if priorfresh is None else brief(priorfresh),held=[brief(r)])
                if r['code']==31 and r['timer']==0:
                    fresh.append(dict(replay=rep,eid=eid,uid=m['uid'],target_eid=m.get('target_eid'),epoch_tick=m['epoch_tick'],record=brief(r)))
                    priorfresh=lastfresh;lastfresh=r
                elif r['code']!=31:
                    lastfresh=priorfresh=None
                previous=r
            if ep is not None:spells.append(finish(ep,None,'censored'))
        print('SCENE',rep,'native_read',total,'selected',selected,'spells',len(spells),flush=True)
    for i,e in enumerate(spells):e['episode_id']=i
    A.save(z.out/'episodes.json.gz',spells);A.save(z.out/'fresh.json.gz',fresh);A.save(z.out/'anchors.json.gz',anchors);A.save(z.out/'units.json.gz',units);A.save(z.out/'objects.json.gz',rockets);A.save(z.out/'infos.json',infos)
    receipt=dict(native_records_read=total,selected_R77_records=selected,entities=len(infos),held_spells=len(spells),strict_held_spells=sum(e['strict'] for e in spells),strict_exits=dict(collections.Counter(e['exit_reason'] if e['exit'] is None else str(e['exit']['code']) for e in spells if e['strict'])),fresh_records=len(fresh),layout_counts=dict(layout_counts),breaks=breaks,sources=sources)
    A.save(z.out/'extraction.json',receipt)
    for uid,rs in anchors.items():
        epoch=next(i['epoch_tick'] for i in infos if str(i['uid'])==uid and (i['replay'],i['eid']) in A.ANCHORS)
        if uid=='118':
            for r in rs:
                if r['tick'] in (11521,11522,11524,11526,11532,11571,11573,11575,11615,11617):print('ANCHOR',uid,json.dumps({k:r[k] for k in ('tick','code','timer','flag','R','d','closing','last','Ps','Vs','unknown','p','vm','n','w','angles')})[:5000],flush=True)
    print('EXTRACTED',json.dumps({k:v for k,v in receipt.items() if k!='sources'}),flush=True)
    for rep,obs in rockets.items():
        print('OBJECT_SCHEMA',rep,type(obs).__name__,len(obs),'first',json.dumps(obs[0] if isinstance(obs,list) and obs else list(obs.items())[:1] if isinstance(obs,dict) else None)[:2000],flush=True)
    manifest=[dict(path=p.relative_to(z.out).as_posix(),bytes=p.stat().st_size,sha256=A.sha(p)) for p in sorted(z.out.rglob('*')) if p.is_file()]
    A.save(z.out/'MANIFEST.json',manifest)
if __name__=='__main__':main()
