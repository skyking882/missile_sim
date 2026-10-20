#!/usr/bin/env python3
"""Offline reproduction of coordinate tests, not a new flight simulation."""
import gzip,hashlib,json,math,os,subprocess,sys
from pathlib import Path

def read(p):
    with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return json.load(f)

def compare(a,b):
    out={'exact':True,'max_difference':0.,'errors':[]}
    def walk(x,y,path):
        if isinstance(x,(float,int)) and not isinstance(x,bool) and isinstance(y,(float,int)) and not isinstance(y,bool):
            if isinstance(x,int) and isinstance(y,int):
                if x!=y:out['errors'].append(path+': integer')
            else:
                d=abs(float(x)-float(y));out['max_difference']=max(out['max_difference'],d)
                if d:out['exact']=False
                if not math.isfinite(d) or d>1e-8+1e-11*max(abs(x),abs(y)):out['errors'].append(path+': numeric')
            return
        if type(x)!=type(y):out['errors'].append(path+': type');return
        if isinstance(x,dict):
            if x.keys()!=y.keys():out['errors'].append(path+': keys');return
            for k in x:walk(x[k],y[k],path+'/'+str(k))
        elif isinstance(x,list):
            if len(x)!=len(y):out['errors'].append(path+': length');return
            for i,(u,v) in enumerate(zip(x,y)):walk(u,v,path+'/'+str(i))
        elif x!=y:out['errors'].append(path+': value')
    walk(a,b,'');out['passed']=not out['errors']
    if out['errors']:out['exact']=False
    return out

def main():
    if len(sys.argv)!=2:raise SystemExit('Usage: python3 reproduce.py /absolute/new/output')
    root=Path(__file__).resolve().parent;out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=False)
    manifest=read(root/'MANIFEST.json')
    for r in manifest:
        p=root/r['path'];assert p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],str(p)
    code=root/'audits/rate_frame_contract_20261020';env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    commands=[(m,[str(code/'check_frames.py'),'--input',str(root/'input'),'--out',str(out/m),'--mode',m]) for m in ('contract','propagation')]
    commands.append(('review',[str(code/'review.py'),'--input',str(root/'input'),'--contract',str(out/'contract'),'--propagation',str(out/'propagation'),'--out',str(out/'review')]))
    for name,args in commands:
        with (out/(name+'.log')).open('w') as f:subprocess.run([sys.executable,*args],stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
        print('COMPLETED',name,flush=True)
    checks=[]
    for group in ('contract','propagation','review'):
        for p in sorted((root/group).glob('*.json*')):
            c=compare(read(p),read(out/group/p.name));checks.append(dict(file=f'{group}/{p.name}',**c))
    receipt=dict(scope='Same fixed game records, no new blinded data or autonomous flight',manifest_files=len(manifest),result_files=len(checks),passed=sum(c['passed'] for c in checks),exact=sum(c['exact'] for c in checks),max_numeric_difference=max([c['max_difference'] for c in checks],default=0.),details=checks)
    (out/'REPRODUCTION.json').write_text(json.dumps(receipt,indent=2))
    print('REPRODUCTION',json.dumps({k:v for k,v in receipt.items() if k!='details'}),flush=True)
    assert receipt['passed']==receipt['result_files'],receipt
if __name__=='__main__':main()
