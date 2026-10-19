#!/usr/bin/env python3
"""Read-only War Thunder state-output audit; no model fitting or simulation."""
from __future__ import annotations
import argparse,bisect,collections,gzip,hashlib,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'capture_held_20261017'))
import audit as A
H=1/48
GRID=10/32767
ANCHORS={('2026.10.08_11.49.38',2439170):118,('2026.10.08_12.11.04',1612803):148}

def read(p):
    with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return json.load(f)
def save(p,x):A.save(p,x)
def equal(a,b,k):return bool(np.array_equal(a[k],b[k]))
def norm(x):return float(np.linalg.norm(x))
def state(r):return A.brief(r)|{k:r[k] for k in ('key','timer_int','flags')}
def ident(rep,eid,a,b):return(rep,int(eid),int(a['index']),int(b['index']))
def fresh(r):return r['code']==31 and r['timer']==0

def pair(meta,a,b):
    t=(b['tick']-a['tick'])*H
    pred=A.rotate(a['n'],a['w'],t)
    return dict(replay=meta['replay'],eid=meta['eid'],uid=meta['uid'],family=meta['family'],index0=a['index'],index1=b['index'],tick0=a['tick'],tick1=b['tick'],code0=a['code'],code1=b['code'],gap=b['tick']-a['tick'],timer0=a['timer'],timer1=b['timer'],flag0=a['flag'],flag1=b['flag'],w_equal=equal(a,b,'w'),PV_equal=equal(a,b,'Ps') and equal(a,b,'Vs'),P_change_m=norm(b['Ps']-a['Ps']),V_change_m_s=norm(b['Vs']-a['Vs']),w_change_deg_s=norm(b['w']-a['w'])*A.RAD,direction_continuation_error_deg=A.angle(pred,b['n'])*A.RAD,PV_CV_residual_m=norm(b['Ps']-a['Ps']-a['Vs']*t),source_closure_before_m=norm(a['p']+a['R']*a['n']-a['Ps']),source_closure_after_m=norm(b['p']+b['R']*b['n']-b['Ps']),timer_delta_residual_s=b['timer']-a['timer']-t,raw_sha0=a['raw_sha'],raw_sha1=b['raw_sha'])

def unit_series(units):
    out={}
    for u in units:
        d={};conflict=False
        for r in u.get('track',[]):
            t=int(r['t_ms']);xyz=r['xyz']
            if t in d and d[t]!=xyz:conflict=True
            d[t]=xyz
        if conflict or len(d)<2:continue
        ticks=sorted(d);out[int(u['eid'])]=(ticks,np.array([d[t] for t in ticks]),u.get('unit_name'))
    return out

def lookup(series,eid,time_ms):
    if eid not in series:return None
    ts,p,name=series[eid];j=bisect.bisect_right(ts,time_ms)-1
    if not 0<=j<len(ts)-1:return None
    dt=ts[j+1]-ts[j]
    if not 0<dt<=250:return None
    f=(time_ms-ts[j])/dt
    return dict(position=(1-f)*p[j]+f*p[j+1],velocity=(p[j+1]-p[j])/(dt/1000),bracket_ms=[ts[j],ts[j+1]],name=name)

def target_compare(meta,a,b,lastfresh,units):
    eid=meta.get('target_eid')
    if eid is None:return dict(status='no_frozen_target_identity')
    eid=int(eid);out=dict(status='fixed_prior_identity_not_proven_current_lock',target_eid=eid)
    for label,time in [('packet',b['packet_ms']),('state',b['tick']*1000/48),('previous_packet',a['packet_ms'])]:
        u=lookup(units,eid,time)
        if u is None:out[label]=None;continue
        lastp=lastfresh['Ps']+lastfresh['Vs']*((b['tick']-lastfresh['tick'])*H) if lastfresh else None
        out[label]=dict(target_name=u['name'],time_ms=time,bracket_ms=u['bracket_ms'],refreshed_P_error_m=norm(b['Ps']-u['position']),refreshed_V_error_m_s=norm(b['Vs']-u['velocity']),previous_saved_P_error_m=norm(a['Ps']-u['position']),last_fresh_CV_P_error_m=norm(lastp-u['position']) if lastp is not None else None,reference_position=u['position'],reference_velocity=u['velocity'])
    return out

def describe(rows):
    return dict(n=len(rows),entities=len({(r['replay'],r['eid']) for r in rows}),w_equal=sum(r['w_equal'] for r in rows),PV_changed=sum(not r['PV_equal'] for r in rows),PV_changed_w_equal=sum(not r['PV_equal'] and r['w_equal'] for r in rows),timers=A.metric([r['timer1'] for r in rows]),direction_continuation=A.metric([r['direction_continuation_error_deg'] for r in rows]),P_change=A.metric([r['P_change_m'] for r in rows]),closure_after=A.metric([r['source_closure_after_m'] for r in rows]))

def episode(start,rows,end,outcome,meta,seed_stable,continuous,flags0):
    return {k:meta[k] for k in ('replay','eid','uid','family')}|dict(start_tick=start['tick'],first_nonfresh_tick=rows[0]['tick'] if rows else end['tick'],end_tick=end['tick'],start_index=start['index'],end_index=end['index'],outcome=outcome,held_rows=len(rows),max_timer=max([r['timer'] for r in rows],default=0),elapsed_s=(end['tick']-start['tick'])*H,PV_and_rate_constant=seed_stable,timer_continuous=continuous,flags0=flags0,first_sha=start['raw_sha'],last_sha=end['raw_sha'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--derived',type=Path,required=True);p.add_argument('--prior',type=Path,required=True);p.add_argument('--out',type=Path,required=True);z=p.parse_args();z.out.mkdir(parents=True,exist_ok=False)
    man=read(z.prior/'MANIFEST.json')
    for r in man:
        f=z.prior/r['path'];assert f.stat().st_size==r['bytes'] and A.sha(f)==r['sha256'],str(f)
    archive=read(z.prior/'native/paths.json.gz');arch={}
    for path in archive:
        for r in path['states']:
            key=(path['replay'],path['eid'],r['index'])
            if key in arch:assert arch[key]['raw_sha']==r['raw_sha']
            else:arch[key]=r
    maps=read(next(z.reference.rglob('missile_seeker_fieldmap.json')))['field_maps'];dataset=read(next(z.reference.rglob('server_dataset.json.gz')))
    info={(i['replay'],int(i['eid'])):i for i in dataset['infos']};assert len(info)==501
    counts=collections.Counter();codes=collections.Counter();trans=collections.Counter();pairs=[];stages=[];episodes=[];anchors=[];sources={};matched=set()
    files=sorted(z.derived.rglob('weapon_raw.jsonl.gz'));assert len(files)==5
    for file in files:
        rep=file.parent.name;sources[rep]=A.sha(file);wanted={eid:m for (rr,eid),m in info.items() if rr==rep}
        uf=file.parent/'unit_history.json.gz';units=unit_series(read(uf)['units'])
        caches={eid:dict(prev=None,fresh=None,ep=None,stage=None) for eid in wanted}
        def finish_stage(st,current,reason):
            sg=st['stage']
            if sg is None:return
            sg['exit_reason']=reason;sg['after']=state(current) if current is not None else None
            sg['span_ticks']=sg['last29']['tick']-sg['first29']['tick'];stages.append(sg);st['stage']=None
        with gzip.open(file,'rt') as f:
            for idx,line in enumerate(f):
                raw=json.loads(line);counts['all_native']+=1;eid=int(raw['fields']['eid'])
                if eid not in wanted:continue
                counts['selected']+=1;meta=wanted[eid];st=caches[eid];b=A.parse(raw,maps,idx)
                if b is None:
                    counts['unsupported']+=1
                    if st['ep']:
                        e=st['ep'];episodes.append(episode(e['start'],e['held'],st['prev'],'censored_layout',meta,e['stable'],e['continuous'],e['flags0']));st['ep']=None
                    finish_stage(st,None,'censored_layout');st['prev']=None;st['fresh']=None;continue
                codes[b['code']]+=1;assert b['code']==24+4*((b['code']>>2)&1)+2*((b['code']>>1)&1)+(b['code']&1)
                key=(rep,eid,idx)
                if key in arch:
                    assert b['raw_sha']==arch[key]['raw_sha'];matched.add(key)
                    for k in ('Ps','Vs','n','w'):np.testing.assert_array_equal(b[k],arch[key][k])
                a=st['prev']
                if a is not None and not 0<b['tick']-a['tick']<=8:
                    counts['time_breaks']+=1
                    if st['ep']:
                        e=st['ep'];episodes.append(episode(e['start'],e['held'],a,'censored_time',meta,e['stable'],e['continuous'],e['flags0']));st['ep']=None
                    finish_stage(st,None,'censored_time');a=None;st['fresh']=None
                if a is not None:
                    if st['stage'] is not None:
                        if b['code']==29:st['stage']['rows29']+=1;st['stage']['last29']=state(b)
                        else:finish_stage(st,b,'code'+str(b['code']))
                    if a['code']!=b['code']:
                        trans[(a['code'],b['code'])]+=1;r=pair(meta,a,b);pairs.append(r)
                        if a['code']==31 and b['code'] in (29,25):
                            r['fixed_target_comparison']=target_compare(meta,a,b,st['fresh'],units)
                            if (rep,eid) in ANCHORS:anchors.append(dict(diagnostic=r,before=state(a),at=state(b),last_fresh=state(st['fresh']) if st['fresh'] else None))
                        if b['code']==29:
                            st['stage']=dict(replay=rep,eid=eid,uid=meta['uid'],family=meta['family'],before=state(a),first29=state(b),last29=state(b),rows29=1,entry=pair(meta,a,b))
                    if st['ep'] is None and fresh(a) and not fresh(b):
                        st['ep']=dict(start=a,held=[],stable=True,continuous=True,flags0=a['flag']==0)
                    e=st['ep']
                    if e is not None:
                        e['flags0']=e['flags0'] and b['flag']==0
                        if b['code']!=31:outcome='exit'+str(b['code'])
                        elif fresh(b):outcome='fresh_return'
                        elif b['timer']<a['timer']-2*GRID:outcome='positive_timer_reset'
                        else:outcome=None
                        if outcome:
                            episodes.append(episode(e['start'],e['held'],b,outcome,meta,e['stable'],e['continuous'],e['flags0']));st['ep']=None
                        else:
                            e['held'].append(b);e['stable']=e['stable'] and all(equal(b,e['start'],k) for k in ('Ps','Vs','w'))
                            e['continuous']=e['continuous'] and abs(b['timer']-a['timer']-(b['tick']-a['tick'])*H)<=2*GRID
                if fresh(b):st['fresh']=b
                st['prev']=b
        for eid,st in caches.items():
            if st['ep']:
                e=st['ep'];episodes.append(episode(e['start'],e['held'],st['prev'],'censored_end',wanted[eid],e['stable'],e['continuous'],e['flags0']))
            finish_stage(st,None,'censored_end')
        print('SCENE_COMPLETE',rep,'pairs',len(pairs),'episodes',len(episodes),flush=True)
    assert len(matched)==len(arch),(len(matched),len(arch))
    groups={}
    for fam,rr in [('all',pairs),('R77',[r for r in pairs if r['family']=='R-77-1'])]:
        for a,b in ((31,29),(29,25),(29,31),(31,25),(25,31)):
            groups[fam+f'_{a}_{b}']=describe([r for r in rr if r['code0']==a and r['code1']==b])
    stagegroups={}
    for fam,rr in [('all',stages),('R77',[r for r in stages if r['family']=='R-77-1'])]:
        stagegroups[fam]=dict(n=len(rr),single_record=sum(r['rows29']==1 for r in rr),rows_distribution=dict(collections.Counter(r['rows29'] for r in rr)),next_code_counts=dict(collections.Counter(r['exit_reason'] for r in rr)),entry_timer=A.metric([r['first29']['timer'] for r in rr]),exceptions_multirecord=[{k:r[k] for k in ('replay','eid','uid','rows29','span_ticks','exit_reason')} for r in rr if r['rows29']>1][:10])
    epgroups={}
    for fam,rr in [('all',episodes),('R77',[r for r in episodes if r['family']=='R-77-1'])]:
        ss=[r for r in rr if r['held_rows']>=2 and r['max_timer']>=.1 and r['flags0'] and r['timer_continuous'] and r['PV_and_rate_constant']]
        epgroups[fam]=dict(n=len(rr),outcomes=dict(collections.Counter(r['outcome'] for r in rr)),strict_held_n=len(ss),strict_outcomes=dict(collections.Counter(r['outcome'] for r in ss)),strict_late_returns=[r for r in ss if r['outcome']=='fresh_return' and r['elapsed_s']>=.95],strict_exit_elapsed={k:A.metric([r['elapsed_s'] for r in ss if r['outcome']==k]) for k in sorted({r['outcome'] for r in ss})})
    out=dict(scope='No-fit raw output-contract audit; status labels are constructed bit combinations, not recovered enum names',counts=dict(counts),cohort=len(info),archive_unique_states_checked=len(matched),prior_manifest_files=len(man),source_hashes=sources,code_counts=dict(codes),transition_counts={str(k):v for k,v in trans.items()},transition_groups=groups,stage29_groups=stagegroups,episode_groups=epgroups,anchors=anchors)
    save(z.out/'summary.json',out);save(z.out/'transitions.json.gz',pairs);save(z.out/'stage29.json.gz',stages);save(z.out/'episodes.json.gz',episodes)
    print('COUNTS',json.dumps({k:out[k] for k in ('counts','cohort','archive_unique_states_checked','code_counts','transition_counts')}),flush=True)
    print('TRANSITION_GROUPS',json.dumps(A.serial(groups)),flush=True)
    print('STAGE29_GROUPS',json.dumps(A.serial(stagegroups)),flush=True)
    print('EPISODES',json.dumps(A.serial({k:{kk:vv for kk,vv in v.items() if kk!='strict_late_returns'} for k,v in epgroups.items()})),flush=True)
    for x in anchors:
        r=x['diagnostic'];print('ANCHOR',json.dumps(A.serial(r)),flush=True)
if __name__=='__main__':main()
