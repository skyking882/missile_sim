#!/usr/bin/env python3
"""Scan raw row identifiers only; do not infer adjacency from tick separation."""
import argparse,gzip,hashlib,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--derived',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    fresh=json.load(gzip.open(a.input/'fresh.json.gz','rt'));wanted={(x['replay'],x['record']['index']):x for x in fresh};entities={x['eid'] for x in fresh};out=[];count=0;sources=[]
    for f in sorted(a.derived.rglob('weapon_raw.jsonl.gz')):
        rep=f.parent.name;last={};sources.append(dict(file=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
        with gzip.open(f,'rt') as fp:
            for idx,line in enumerate(fp):
                count+=1;r=json.loads(line);eid=int(r['fields']['eid'])
                if eid not in entities:continue
                if (rep,idx) in wanted:
                    x=wanted[rep,idx];assert x['eid']==eid
                    out.append(dict(replay=rep,eid=eid,index=idx,previous_same_eid_index=last.get(eid),tick=x['record']['tick'],raw_sha=x['record']['raw_sha']))
                last[eid]=idx
    assert len(out)==len(fresh)
    with gzip.open(a.out/'adjacency.json.gz','wt') as fp:json.dump(out,fp,separators=(',',':'))
    (a.out/'receipt.json').write_text(json.dumps(dict(native_rows_read=count,fresh_identifiers_checked=len(out),sources=sources),indent=2))
    print('ADJACENCY',json.dumps(dict(native_rows_read=count,fresh_identifiers_checked=len(out))))
if __name__=='__main__':main()
