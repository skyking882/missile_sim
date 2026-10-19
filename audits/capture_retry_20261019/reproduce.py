#!/usr/bin/env python3
"""Run the delivered clock and replay-component tests without network calls."""
from __future__ import annotations
import gzip,hashlib,json,math,os,subprocess,sys
from pathlib import Path

def read(p):
    with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return json.load(f)

def compare(x,y):
    r=dict(exact=True,numeric_differences=0,max_absolute_difference=0.,errors=[])
    def visit(a,b,path):
        if isinstance(a,(float,int)) and not isinstance(a,bool) and isinstance(b,(float,int)) and not isinstance(b,bool):
            if isinstance(a,int) and isinstance(b,int):
                if a!=b:r['errors'].append(path+': integer differs')
                return
            d=abs(float(a)-float(b))
            if d:
                r['exact']=False;r['numeric_differences']+=1;r['max_absolute_difference']=max(r['max_absolute_difference'],d)
                if not math.isfinite(d) or d>1e-8+1e-11*max(abs(a),abs(b)):r['errors'].append(path+': float outside tolerance')
            return
        if type(a)!=type(b):r['errors'].append(path+': type differs');return
        if isinstance(a,dict):
            if set(a)!=set(b):r['errors'].append(path+': keys differ');return
            for k in a:visit(a[k],b[k],path+'/'+str(k))
        elif isinstance(a,list):
            if len(a)!=len(b):r['errors'].append(path+': length differs');return
            for i,(u,v) in enumerate(zip(a,b)):visit(u,v,path+'/'+str(i))
        elif a!=b:r['errors'].append(path+': value differs')
    visit(x,y,'');r['passed']=not r['errors']
    if r['errors']:r['exact']=False
    return r

def main():
    if len(sys.argv)!=2:raise SystemExit('Usage: python3 reproduce.py /absolute/path/to/new-output')
    root=Path(__file__).resolve().parent;out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=False)
    manifest=read(root/'MANIFEST.json')
    for row in manifest:
        p=root/row['path'];assert p.is_file() and p.stat().st_size==row['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],str(p)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    code=root/'audits/capture_retry_20261019'
    commands=[('clock_tests',[str(code/'test_retry_clock.py')]),('replay',[str(code/'replay_check.py'),'--input',str(root/'input'),'--out',str(out/'replay')]),('past_pose',[str(code/'past_pose_check.py'),'--input',str(root/'input'),'--baseline',str(out/'replay'),'--out',str(out/'past_pose')])]
    for label,args in commands:
        with (out/(label+'.log')).open('w') as f:subprocess.run([sys.executable,*args],stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
        print('COMPLETED',label,flush=True)
    results=[]
    for group in ('replay','past_pose'):
        for p in sorted((root/'results'/group).iterdir()):
            if p.name.endswith(('.json','.json.gz')):
                rel=Path(group)/p.name;r=compare(read(p),read(out/rel));results.append(dict(file=rel.as_posix(),**r))
    receipt=dict(scope='Same-data deterministic reproduction; not new blind validation, autonomous flight or new native WRPL decode',verified_manifest_files=len(manifest),result_files=len(results),passed=sum(r['passed'] for r in results),exact=sum(r['exact'] for r in results),maximum_numeric_difference=max([r['max_absolute_difference'] for r in results],default=0),tolerance='floating absolute1e-8+relative1e-11; integer/text/structure exact',details=results)
    (out/'reproduction.json').write_text(json.dumps(receipt,indent=2))
    print('REPRODUCTION',json.dumps({k:v for k,v in receipt.items() if k!='details'}),flush=True)
    if receipt['passed']!=receipt['result_files']:raise AssertionError('Result mismatch; read reproduction.json')
if __name__=='__main__':main()
