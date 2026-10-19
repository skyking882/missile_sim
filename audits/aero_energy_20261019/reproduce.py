#!/usr/bin/env python3
"""Offline replay-conditioned game-plant checks; refuses overwriting results."""
import gzip,hashlib,json,math,os,subprocess,sys
from pathlib import Path

def load(p):
    with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return json.load(f)

def compare(a,b):
    result={'exact':True,'maximum_numeric_difference':0.,'errors':[]}
    def visit(x,y,path):
        if isinstance(x,(int,float)) and not isinstance(x,bool) and isinstance(y,(int,float)) and not isinstance(y,bool):
            if isinstance(x,int) and isinstance(y,int):
                if x!=y:result['errors'].append(path+': integer mismatch')
            else:
                d=abs(x-y);result['maximum_numeric_difference']=max(result['maximum_numeric_difference'],d)
                if d:result['exact']=False
                if not math.isfinite(d) or d>1e-8+1e-11*max(abs(x),abs(y)):result['errors'].append(path+': numeric mismatch')
        elif type(x)!=type(y):result['errors'].append(path+': type mismatch')
        elif isinstance(x,dict):
            if x.keys()!=y.keys():result['errors'].append(path+': keys mismatch')
            else:
                for k in x:visit(x[k],y[k],path+'/'+k)
        elif isinstance(x,list):
            if len(x)!=len(y):result['errors'].append(path+': length mismatch')
            else:
                for i,(u,v) in enumerate(zip(x,y)):visit(u,v,path+'/'+str(i))
        elif x!=y:result['errors'].append(path+': value mismatch')
    visit(a,b,'');result['passed']=not result['errors'];result['exact']=result['exact'] and result['passed'];return result

def main():
    if len(sys.argv)!=2:raise SystemExit('python3 reproduce.py /absolute/path/to/new-output')
    root=Path(__file__).resolve().parent;out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=False)
    manifest=load(root/'MANIFEST.json')
    for r in manifest:
        p=root/r['path'];assert p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],str(p)
    code=root/'audits/aero_energy_20261019';env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    commands=[('energy',[code/'check_energy.py','--input',root/'input','--out',out/'energy']),('attitude',[code/'attitude_isolation.py','--input',root/'input','--main',out/'energy','--out',out/'attitude'])]
    for label,args in commands:
        with (out/(label+'.log')).open('w') as f:subprocess.run([sys.executable,*map(str,args)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        print('COMPLETED',label,flush=True)
    results=[]
    for rel in ['energy/summary.json','energy/samples_118.json.gz','energy/samples_148.json.gz','attitude/summary.json']:
        results.append({'file':rel,**compare(load(root/'results'/rel),load(out/rel))})
    receipt={'scope':'Same-data force/work and prescribed-Q integration reproduction, not autonomous guidance or physical-game source proof','verified_files':len(manifest),'result_sets':len(results),'all_passed':all(r['passed'] for r in results),'all_exact':all(r['exact'] for r in results),'maximum_numeric_difference':max(r['maximum_numeric_difference'] for r in results),'comparisons':results}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2));print('REPRODUCED',json.dumps(receipt),flush=True)
    if not receipt['all_passed']:raise AssertionError('Result mismatch: inspect receipt')
if __name__=='__main__':main()
