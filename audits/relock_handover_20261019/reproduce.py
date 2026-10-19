#!/usr/bin/env python3
"""Offline reproduction of the output review and fixed conditional comparisons."""
import gzip,hashlib,json,os,subprocess,sys
from pathlib import Path

def read(p):
    with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return json.load(f)
def main():
    if len(sys.argv)!=2:raise SystemExit('Usage: python3 reproduce.py /absolute/path/to/new-output')
    root=Path(__file__).resolve().parent;out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=False)
    manifest=read(root/'MANIFEST.json')
    for r in manifest:
        p=root/r['path'];assert p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],str(p)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    flow=root/'audits/relock_output_20261019';hand=root/'audits/relock_handover_20261019'
    commands=[('contracts',[flow/'test_contract.py']),('review',[flow/'review_flow.py','--prior',root/'prior','--native',root/'results','--out',out/'review']),('handover',[hand/'check_handover.py','--prior',root/'prior','--flow',root/'results','--out',out/'handover']),('seed',[hand/'check_seed.py','--flow',root/'results','--out',out/'seed'])]
    for name,args in commands:
        with (out/(name+'.log')).open('w') as f:subprocess.run([sys.executable,*map(str,args)],stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
        print('COMPLETED',name,flush=True)
    checks=[]
    for name in ('review/verification.json','handover/summary.json','handover/events.json.gz','seed/SEED_RESULTS.json'):
        same=read(root/name)==read(out/name);checks.append(dict(file=name,exact_json=same));assert same,name
    receipt=dict(scope='Same-data contract tests and offline conditional state comparisons; not a new native-stream reread or autonomous flight',manifest_files=len(manifest),contract_tests=9,checks=checks,all_exact=all(r['exact_json'] for r in checks))
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2));print('REPRODUCED',json.dumps(receipt),flush=True)
if __name__=='__main__':main()
