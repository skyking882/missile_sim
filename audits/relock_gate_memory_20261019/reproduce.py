#!/usr/bin/env python3
"""Offline rerun. Native full-stream extraction is separately documented."""
from __future__ import annotations
import gzip,hashlib,json,math,os,subprocess,sys
from pathlib import Path

def read(p):
    with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return json.load(f)
def compare(a,b):
    result=dict(exact=True,max_numeric_difference=0.,numeric_differences=0,errors=[])
    def visit(x,y,path):
        if isinstance(x,(float,int)) and not isinstance(x,bool) and isinstance(y,(float,int)) and not isinstance(y,bool):
            if isinstance(x,int) and isinstance(y,int):
                if x!=y:result['errors'].append(path+': integer mismatch')
                return
            delta=abs(float(x)-float(y))
            if delta:
                result['exact']=False;result['numeric_differences']+=1;result['max_numeric_difference']=max(result['max_numeric_difference'],delta)
                if not math.isfinite(delta) or delta>1e-8+1e-11*max(abs(x),abs(y)):result['errors'].append(path+': float tolerance')
            return
        if type(x)!=type(y):result['errors'].append(path+': different types');return
        if isinstance(x,dict):
            if set(x)!=set(y):result['errors'].append(path+': different keys');return
            for k in x:visit(x[k],y[k],path+'/'+str(k))
        elif isinstance(x,list):
            if len(x)!=len(y):result['errors'].append(path+': different lengths');return
            for i,(u,v) in enumerate(zip(x,y)):visit(u,v,path+'/'+str(i))
        elif x!=y:result['errors'].append(path+': value mismatch')
    visit(a,b,'');result['passed']=not result['errors']
    if result['errors']:result['exact']=False
    return result

def main():
    if len(sys.argv)!=2:raise SystemExit('Usage: python3 reproduce.py /absolute/path/to/new-output')
    root=Path(__file__).resolve().parent;out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=False)
    manifest=read(root/'MANIFEST.json')
    for x in manifest:
        p=root/x['path'];assert p.is_file() and p.stat().st_size==x['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256'],str(p)
    code=root/'audits/relock_gate_memory_20261019';env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    cmds=[('general',['analyze.py','--input',str(root/'input'),'--out',str(out/'general')]),('scalar',['scalar_memory.py','--input',str(root/'input'),'--main',str(out/'general'),'--out',str(out/'scalar')]),('review',['review.py','--input',str(root/'input'),'--main',str(out/'general'),'--scalar',str(out/'scalar'),'--out',str(out/'review')]),('final',['final_checks.py','--input',str(root/'input'),'--main',str(out/'general'),'--review',str(out/'review'),'--adjacency',str(root/'adjacency'),'--out',str(out/'final')])]
    for label,args in cmds:
        args[0]=str(code/args[0])
        with (out/(label+'.log')).open('w') as f:subprocess.run([sys.executable,*args],stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
        print('COMPLETED',label,flush=True)
    results=[]
    for group in ('general','scalar','review','final'):
        for p in sorted((root/'results'/group).iterdir()):
            if p.name.endswith(('.json','.json.gz')):
                rel=Path(group)/p.name;check=compare(read(p),read(out/rel));results.append(dict(file=rel.as_posix(),**check));print('COMPARED',rel,check['passed'],check['exact'],flush=True)
    receipt=dict(scope='Same-native-data reconstruction and diagnostic-gate replication, not a new blind dataset or autonomous flight; full native scan not repeated by this entrypoint.',verified_files=len(manifest),compared_result_files=len(results),passed=sum(r['passed'] for r in results),exact=sum(r['exact'] for r in results),max_numeric_difference=max((r['max_numeric_difference'] for r in results),default=0),comparison_tolerance='float absolute1e-8+relative1e-11; integer/text/structure exact; not a physical acceptance tolerance',comparisons=results)
    (out/'REPRODUCTION.json').write_text(json.dumps(receipt,indent=2));print('REPRODUCTION',json.dumps({k:v for k,v in receipt.items() if k!='comparisons'}),flush=True)
    if receipt['passed']!=len(results):raise AssertionError('Result mismatch; inspect REPRODUCTION.json')
if __name__=='__main__':main()
