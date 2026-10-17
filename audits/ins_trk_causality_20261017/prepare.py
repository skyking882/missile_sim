#!/usr/bin/env python3
from pathlib import Path
import argparse,collections,gzip,hashlib,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'capture_held_20261017'))
import audit as A

def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--derived',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((args.evidence/'MANIFEST.json').read_text())
    for row in manifest:
        f=args.evidence/row['path'];assert f.stat().st_size==row['bytes'] and A.sha(f)==row['sha256'],str(f)
    dp=next(args.reference.rglob('server_dataset.json.gz'));mp=next(args.reference.rglob('missile_seeker_fieldmap.json'))
    ds=json.load(gzip.open(dp,'rt'));maps=json.loads(mp.read_text())['field_maps']
    cases={};refs={};duplicates=0
    for c,l in zip(ds['cases'],ds['labels']):
        info=c['info'];key=(info['replay'],int(info['eid']));cases[key]=dict(info=info,lo=min(c['ticks']),hi=max(c['ticks']))
        for i,t in enumerate(c['ticks']):
            k=(*key,int(t));duplicates+=int(k in refs)
            refs[k]=dict(Q=np.array(c['Q'][i]),n=np.array(l['n'][i-1]) if i else None,w=np.array(l['w'][i-1]) if i else np.array(c['seed'])/A.RAD)
    assert len(cases)==501
    files=sorted(args.derived.rglob('weapon_raw.jsonl.gz'));assert len(files)==5
    inv=[];hashes={str(dp):A.sha(dp),str(mp):A.sha(mp)};groups=collections.defaultdict(list);maxerr=collections.defaultdict(float);seen=set();rawcount=0
    for f in files:
        rep=f.parent.name;want={eid:z for (s,eid),z in cases.items() if s==rep};cnt=collections.Counter();hashes[str(f)]=A.sha(f)
        with gzip.open(f,'rt') as stream:
            for i,line in enumerate(stream):
                raw=json.loads(line);rawcount+=1;cnt['raw']+=1;fields=raw['fields'];eid=int(fields['eid']);tick=int(fields['b5'])
                if eid not in want or not(want[eid]['lo']<=tick<=want[eid]['hi']):continue
                r=A.parse(raw,maps,i)
                if r is None:cnt['unsupported']+=1;groups[(rep,eid)].append(None);continue
                info=want[eid]['info'];r.update(replay=rep,family=info['family'],uid=info['uid'],epoch_tick=info['epoch_tick'])
                groups[(rep,eid)].append(r);cnt['selected']+=1
                key=(rep,eid,tick)
                if key in refs:
                    seen.add(key)
                    for k,v in refs[key].items():
                        if v is not None:maxerr[k]=max(maxerr[k],float(np.max(np.abs(v-r[k]))))
        inv.append(dict(session=rep,**cnt));print('PREPARED',rep,dict(cnt),flush=True)
    assert rawcount==1633750,rawcount
    assert len(seen)==len(refs),(len(seen),len(refs))
    assert max(maxerr.values())<1e-12,dict(maxerr)
    series=[]
    for (rep,eid),rows in sorted(groups.items()):series.append(dict(replay=rep,eid=eid,info=cases[(rep,eid)]['info'],rows=rows))
    A.save(args.out/'cohort.json.gz',series)
    verification=dict(native_records_scanned=rawcount,series=len(series),selected_rows=sum(r is not None for s in series for r in s['rows']),reference_unique_states=len(seen),reference_duplicate_keys=duplicates,reference_max_abs_difference=dict(maxerr),core_manifest_checked=len(manifest),source_hashes=hashes,inventory=inv)
    A.save(args.out/'source_verification.json',verification);print('PREPARE_VERIFIED',json.dumps(verification,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
