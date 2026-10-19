#!/usr/bin/env python3
"""Reproduce the delivered evidence review, not a new raw-stream decode."""
from __future__ import annotations
import hashlib,json,os,subprocess,sys
from pathlib import Path

def main():
    if len(sys.argv)!=2:raise SystemExit('Usage: python3 reproduce.py /absolute/path/to/new-output')
    root=Path(__file__).resolve().parent;out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((root/'MANIFEST.json').read_text())
    for r in manifest:
        p=root/r['path'];assert p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],str(p)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    code=root/'audits/relock_output_20261019'
    for label,args in [('contracts',[str(code/'test_contract.py')]),('review',[str(code/'review_flow.py'),'--prior',str(root/'prior'),'--native',str(root/'results'),'--out',str(out/'review')])]:
        with (out/(label+'.log')).open('w') as f:subprocess.run([sys.executable,*args],stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
        print('COMPLETED',label,flush=True)
    ref=json.loads((root/'review/verification.json').read_text());got=json.loads((out/'review/verification.json').read_text())
    assert got==ref,'Reviewed JSON differs from completed reference'
    result=dict(scope='Same-data contract tests and independent arithmetic review of archived and newly extracted state transitions; not a new1.63M-row scan, WRPL decoding, or production simulation',files_verified=len(manifest),contract_tests=9,review_json_exact=True,transition_checks=got['archived_transition_pairs'],stage29_checks=got['stage29_entries_checked'])
    (out/'receipt.json').write_text(json.dumps(result,indent=2));print('REPRODUCTION',json.dumps(result),flush=True)
if __name__=='__main__':main()
