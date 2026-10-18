#!/usr/bin/env python3
"""Read-only window audit on original native exports. No new fitted parameters."""
from __future__ import annotations
import argparse,collections,json,sys
from pathlib import Path
import numpy as np
import compare_states as B
A,C=B.A,B.C

def finish_window(item,maps,expected,classified):
    e=expected[item['identity']];meta=classified[item['identity']]
    states=[]
    for idx,raw in item['raw']:
        r=A.parse(raw,maps,idx)
        states.append(r)
    old=next(r for r in states if r is not None and r['index']==e['old']['index'])
    new=next(r for r in states if r is not None and r['index']==e['new']['index'])
    assert old['raw_sha']==e['old']['raw_sha'] and new['raw_sha']==e['new']['raw_sha'],item['identity']
    for tag,r in [('old',old),('new',new)]:
        for k in ('n','w','Ps','Vs','p','vm','Q'):
            np.testing.assert_allclose(r[k],np.array(e[tag][k]),atol=1e-12,rtol=0,err_msg=str((item['identity'],tag,k)))
    j=next(i for i,r in enumerate(states) if r is not None and r['index']==new['index'])
    output=[];pairs=[]
    for i,r in enumerate(states):
        if r is None:
            output.append({'unsupported':True,'index':item['raw'][i][0]});continue
        ray=A.unit(r['Ps']-r['p']);g=r['Q']@A.ray(r['angles'])
        row=A.brief(r)|dict(relative_record=i-j,target_offaxis_deg=B.ang(ray,r['Q'][:,0]),n_vs_Ps_deg=B.ang(r['n'],ray),n_vs_pointing_deg=B.ang(r['n'],g),geometry_error_deg_s=float(np.linalg.norm(r['w']-B.geometry(r)))*B.DEG)
        if i and states[i-1] is not None:
            a=states[i-1];T=(r['tick']-a['tick'])/48
            if 0<T<=8/48:
                c=A.correction(A.rotate(a['n'],a['w'],T),r['n']);nn0=A.unit(a['Ps']-a['p'])
                predP=a['Ps']+a['Vs']*T;dw=r['w']-a['w'];dgeo=B.geometry(r)-B.geometry(a)
                ps_time=(a['timer']+T);tmerr=abs(r['timer']-a['timer']-T)
                pair=dict(index0=a['index'],index1=r['index'],tick0=a['tick'],tick1=r['tick'],relative_end=i-j,code0=a['code'],code1=r['code'],gap=r['tick']-a['tick'],flag0=a['flag'],flag1=r['flag'],timer0=a['timer'],timer1=r['timer'],timer_continuous=tmerr<2*10/32767,
                          both_fresh00=a['timer']==r['timer']==0 and a['flag']==r['flag']==0,rate_same=bool(np.array_equal(a['w'],r['w'])),PV_same=bool(np.array_equal(a['Ps'],r['Ps']) and np.array_equal(a['Vs'],r['Vs'])),
                          direction_jump_deg=B.ang(a['n'],r['n']),Ps_ray_jump_deg=B.ang(nn0,ray),Ps_prediction_direction_error_deg=B.ang(A.unit(predP-r['p']),r['n']),Ps_CV_innovation_m=float(np.linalg.norm(r['Ps']-predP)),rate_increment_deg_s=float(np.linalg.norm(dw))*B.DEG,geometry_increment_deg_s=float(np.linalg.norm(dgeo))*B.DEG,increment_geometry_residual_deg_s=float(np.linalg.norm(dw-dgeo))*B.DEG,
                          ordinary_relation_residual_deg_s=float(np.linalg.norm(dw-c/(B.TAU+T/2)))*B.DEG)
                pairs.append(pair);row['preceding_pair']=pair
        output.append(row)
    nextfirst=next((p for p in pairs if p['relative_end']==1 and p['code0']==p['code1']==31 and p['both_fresh00']),None)
    after=[p for p in pairs if p['relative_end']>0 and p['code0']==p['code1']==31 and p['both_fresh00']]
    before=[p for p in pairs if p['relative_end']<0 and p['code0']==p['code1']==25]
    out=dict(identity=list(item['identity']),uid=e['uid'],family=e['family'],angle_group=meta['angle_group'],offaxis_deg=meta['offaxis_deg'],original=meta['original'],lag=meta['candidate_reacq_lag'],near=meta['near'],fresh00=meta['fresh00'],boundary_rate_error=meta['baseline_error_deg_s'],boundary_axial_error=meta['baseline_axial_deg_s'],boundary_transverse_error=meta['baseline_transverse_deg_s'],source_split_deg=meta['n0_vs_Ps_deg'],old_prior_vs_new_deg=meta['prior_vs_new_direction_deg'],new_vs_pointing_deg=meta['n1_vs_pointing_deg'],next_first=nextfirst,after_fresh_pairs=after,before_INS_pairs=before,states=output)
    return out

def describe(windows):
    out={}
    r77=[w for w in windows if w['family']=='R-77-1']
    for name,rr in [('R77_all',r77),('R77_first',[w for w in r77 if w['original']]),('R77_reacq_lag0',[w for w in r77 if w['fresh00'] and w['lag']==0]),('R77_reacq_lag1',[w for w in r77 if w['fresh00'] and w['lag']==1]),('all931',windows)]:
        out[name]={}
        for group in ('small','middle','large','all'):
            r=[w for w in rr if group=='all' or w['angle_group']==group];nxt=[w['next_first'] for w in r if w['next_first'] is not None]
            pp=[p for w in r for p in w['before_INS_pairs']];af=[p for w in r for p in w['after_fresh_pairs']]
            out[name][group]=dict(events=len(r),next_first_eligible=len(nxt),next_first_rate_increment=C.metric([p['rate_increment_deg_s'] for p in nxt]),next_first_ordinary_residual=C.metric([p['ordinary_relation_residual_deg_s'] for p in nxt]),before_INS_pairs=len(pp),before_INS_geometry_increment_residual=C.metric([p['increment_geometry_residual_deg_s'] for p in pp]),after_fresh_window_pairs=len(af),after_ordinary_residual=C.metric([p['ordinary_relation_residual_deg_s'] for p in af]))
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--main-results',type=Path,required=True);p.add_argument('--derived',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    source=a.source/'input/prior/evidence/data/main/evidence/capture_events.json.gz';ee=C.read(source);expected={B.ident(e):e for e in ee};rows=C.read(a.main_results/'events.json.gz');classified={B.ident(e):e for e in rows};assert len(expected)==len(classified)==931
    mappath=next(a.reference.rglob('missile_seeker_fieldmap.json'));maps=C.read(mappath)['field_maps']
    windows=[];total=0;sourcehash={};pending_left=0;files=sorted(a.derived.rglob('weapon_raw.jsonl.gz'));assert len(files)==5
    for path in files:
        rep=path.parent.name;sourcehash[rep]=C.digest(path);targets={e['new']['index']:B.ident(e) for e in ee if e['replay']==rep};wanted={e['eid'] for e in ee if e['replay']==rep};buffers=collections.defaultdict(lambda:collections.deque(maxlen=8));active=collections.defaultdict(list)
        import gzip
        with gzip.open(path,'rt') as f:
            for idx,line in enumerate(f):
                raw=json.loads(line);total+=1;eid=int(raw['fields']['eid'])
                if eid not in wanted:continue
                keep=[]
                for item in active[eid]:
                    item['raw'].append((idx,raw));item['remaining']-=1
                    if item['remaining']==0:windows.append(finish_window(item,maps,expected,classified))
                    else:keep.append(item)
                active[eid]=keep
                if idx in targets:
                    key=targets[idx];assert key[1]==eid
                    active[eid].append(dict(identity=key,raw=list(buffers[eid])+[(idx,raw)],remaining=8))
                buffers[eid].append((idx,raw))
        for items in active.values():
            for item in items:windows.append(finish_window(item,maps,expected,classified));pending_left+=1
        print('RAW_SCENE_COMPLETE',rep,'windows',len(windows),flush=True)
    assert len(windows)==931 and len({tuple(w['identity']) for w in windows})==931
    windows.sort(key=lambda w:tuple(w['identity']));groups=describe(windows)
    main=C.read(a.main_results/'summary.json');reps={B.ident(r) for r in main['representatives']};sel=[w for w in windows if tuple(w['identity']) in reps]
    # Adverse events are selected by response only for inspection, never excluded or fitted.
    adverse=sorted([w for w in windows if w['family']=='R-77-1' and w['angle_group']=='large' and w['fresh00'] and w['lag']==1],key=lambda w:w['boundary_rate_error'],reverse=True)[:4]
    summary=dict(scope='Native raw-export window validation, not new WRPL decode; same frozen events. Counts of overlapping multi-pair windows are not independent sample counts.',counts=dict(raw_records_read=total,windows=len(windows),endpoint_hash_checks=2*len(windows),partial_tail_windows=pending_left),raw_export_sha256=sourcehash,fieldmap_sha256=C.digest(mappath),groups=groups,representatives=sel,adverse_late_large=adverse)
    C.save(a.out/'summary.json',summary);C.save(a.out/'windows.json.gz',windows)
    for k,v in groups.items():
        if k!='all931':print('WINDOW_GROUP',k,json.dumps(C.serial(v)),flush=True)
    for label,ss in [('REPRESENTATIVE',sel),('ADVERSE_LATE',adverse)]:
        for w in ss:
            mini={k:v for k,v in w.items() if k not in ('states','before_INS_pairs','after_fresh_pairs')}
            mini['window']=[{k:r.get(k) for k in ('tick','code','relative_record','timer','flag','target_offaxis_deg','n_vs_Ps_deg','n_vs_pointing_deg','geometry_error_deg_s','preceding_pair')} for r in w['states']]
            print(label,json.dumps(C.serial(mini)),flush=True)
    print('RAW_VALIDATION',json.dumps(summary['counts']),flush=True)
if __name__=='__main__':main()
