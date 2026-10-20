#!/usr/bin/env python3
"""Offline reproduction for the delivered game-state PN comparison."""
from __future__ import annotations
import gzip,hashlib,json,math,os,subprocess,sys
from pathlib import Path

def read(p):
    with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return json.load(f)

def compare(a,b):
    result=dict(exact=True,numeric_differences=0,maximum_absolute_difference=0.,errors=[])
    def walk(x,y,path):
        if isinstance(x,(float,int)) and not isinstance(x,bool) and isinstance(y,(float,int)) and not isinstance(y,bool):
            if isinstance(x,int) and isinstance(y,int):
                if x!=y:result['errors'].append(path+': integer differs')
                return
            d=abs(float(x)-float(y))
            if d:
                result['exact']=False;result['numeric_differences']+=1;result['maximum_absolute_difference']=max(result['maximum_absolute_difference'],d)
                if not math.isfinite(d) or d>1e-8+1e-11*max(abs(x),abs(y)):result['errors'].append(path+': float outside tolerance')
            return
        if type(x)!=type(y):result['errors'].append(path+': type differs');return
        if isinstance(x,dict):
            if set(x)!=set(y):result['errors'].append(path+': keys differ');return
            for k in x:walk(x[k],y[k],path+'/'+str(k))
        elif isinstance(x,list):
            if len(x)!=len(y):result['errors'].append(path+': length differs');return
            for i,(v,w) in enumerate(zip(x,y)):walk(v,w,path+'/'+str(i))
        elif x!=y:result['errors'].append(path+': value differs')
    walk(a,b,'');result['passed']=not result['errors']
    if result['errors']:result['exact']=False
    return result

def main():
    if len(sys.argv)!=2:raise SystemExit('Usage: python3 reproduce.py /absolute/path/to/new-output')
    root=Path(__file__).resolve().parent
    if not (root/'MANIFEST.json').is_file():root=root.parent
    out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=False)
    manifest=read(root/'MANIFEST.json')
    for r in manifest:
        p=root/r['path'];assert p.is_file() and p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],str(p)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    code=root/'audits/paper_pn_20261020'
    jobs=[('commands',[str(code/'compare.py'),'--input',str(root/'input'),'--out',str(out/'commands')]),('flights',[str(code/'flights.py'),'--input',str(root/'input'),'--out',str(out/'flights')]),('review',[str(root/'tools/review_paper_pn_20261020.py'),'--input',str(root/'input'),'--commands',str(out/'commands'),'--flights',str(out/'flights'),'--out',str(out/'review')])]
    for label,args in jobs:
        with (out/(label+'.log')).open('w') as f:subprocess.run([sys.executable,*args],stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
        print('COMPLETED',label,flush=True)
    results=[]
    for group in ('commands','flights','review'):
        for p in sorted((root/group).iterdir()):
            if p.name.endswith(('.json','.json.gz')):
                result=compare(read(p),read(out/group/p.name));results.append(dict(path=f'{group}/{p.name}',**result))
    receipt=dict(scope='Same-data numerical reproduction, not a new game replay, independent decoder, autonomous seeker or real-weapon test.',manifest_files=len(manifest),result_files=len(results),passed=sum(r['passed'] for r in results),exact=sum(r['exact'] for r in results),maximum_numeric_difference=max((r['maximum_absolute_difference'] for r in results),default=0.),float_tolerance='absolute1e-8+relative1e-11; integer/text/structure exact',details=results)
    (out/'REPRODUCTION.json').write_text(json.dumps(receipt,indent=2))
    print('REPRODUCTION',json.dumps({k:v for k,v in receipt.items() if k!='details'}),flush=True)
    if receipt['passed']!=receipt['result_files']:raise AssertionError('Reproduction mismatch; inspect receipt')
if __name__=='__main__':main()
