#!/usr/bin/env python3
"""Recompute delivered timing diagnostics; full native-stream reread is separate."""
import gzip,hashlib,json,os,subprocess,sys
from pathlib import Path

def read(p):
    with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return json.load(f)
def main():
    if len(sys.argv)!=2:raise SystemExit('Usage: python3 reproduce.py /absolute/path/to/new-output')
    root=Path(__file__).resolve().parent;out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=False)
    manifest=read(root/'MANIFEST.json')
    for r in manifest:
        p=root/r['path'];assert p.is_file() and p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],str(p)
    code=root/'audits/relock_timing_20261019';env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1')
    runs=[('tests',[str(code/'test_timing.py')]),('interval',[str(code/'timing_bounds.py'),'--input',str(root/'prior'),'--out',str(out/'interval')]),('review',[str(code/'review.py'),'--native',str(root/'native'),'--interval',str(out/'interval'),'--out',str(out/'review')])]
    for name,args in runs:
        with (out/(name+'.log')).open('w') as f:subprocess.run([sys.executable,*args],stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
        print('COMPLETED',name,flush=True)
    pairs=[('interval/summary.json','interval/summary.json'),('interval/returns.json.gz','interval/returns.json.gz'),('review/verification.json','review/verification.json'),('review/strict_events.json.gz','review/strict_events.json.gz'),('review/entry25_intervals.json.gz','review/entry25_intervals.json.gz')]
    checks=[]
    for a,b in pairs:
        ok=read(root/a)==read(out/b);checks.append(dict(file=a,exact_json=ok))
    receipt=dict(scope='Same-data scalar reproduction and path review, no new native decode or autonomous flight',manifest_files=len(manifest),comparisons=checks,all_exact=all(x['exact_json'] for x in checks))
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
    if not receipt['all_exact']:raise AssertionError('A reproduced JSON output differs')
if __name__=='__main__':main()
