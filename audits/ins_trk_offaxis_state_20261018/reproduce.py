#!/usr/bin/env python3
"""Reproduce the included fixed comparisons; original WRPL scanning is separate."""
from __future__ import annotations
import gzip,hashlib,json,math,os,subprocess,sys
from pathlib import Path

def load(p):
    with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return json.load(f)

def compare(a,b):
    stats={'exact':True,'numeric_differences':0,'maximum_absolute_difference':0.,'outside_tolerance':[],'structure_errors':[]}
    def walk(x,y,path):
        if isinstance(x,(float,int)) and not isinstance(x,bool) and isinstance(y,(float,int)) and not isinstance(y,bool):
            if isinstance(x,int) and isinstance(y,int):
                if x!=y:stats['structure_errors'].append(path+': different integers')
                return
            d=abs(float(x)-float(y))
            if d:
                stats['exact']=False;stats['numeric_differences']+=1;stats['maximum_absolute_difference']=max(stats['maximum_absolute_difference'],d)
                if not math.isfinite(d) or d>1e-9+1e-11*max(abs(x),abs(y)):stats['outside_tolerance'].append({'path':path,'reference':x,'new':y})
            return
        if type(x)!=type(y):stats['structure_errors'].append(path+': different types');return
        if isinstance(x,dict):
            if x.keys()!=y.keys():stats['structure_errors'].append(path+': different keys');return
            for k in x:walk(x[k],y[k],path+'/'+str(k))
        elif isinstance(x,list):
            if len(x)!=len(y):stats['structure_errors'].append(path+': different lengths');return
            for i,(u,v) in enumerate(zip(x,y)):walk(u,v,path+'/'+str(i))
        elif x!=y:stats['structure_errors'].append(path+': different values')
    walk(a,b,'')
    if stats['structure_errors']:stats['exact']=False
    stats['passed']=not(stats['structure_errors'] or stats['outside_tolerance'])
    return stats

def main():
    if len(sys.argv)!=2:raise SystemExit('Usage: python3 reproduce.py /absolute/path/to/new-output')
    root=Path(__file__).resolve().parent;out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=False)
    manifest=load(root/'MANIFEST.json')
    for r in manifest:
        p=root/r['path'];assert p.is_file() and p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],str(p)
    code=root/'audits/ins_trk_offaxis_state_20261018'
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    commands=[('main',[str(code/'compare_states.py'),'--input',str(root/'input'),'--out',str(out/'main')]),('review',[str(code/'review.py'),'--input',str(root/'input'),'--results',str(out/'main'),'--windows',str(root/'windows'),'--out',str(out/'review')])]
    for name,args in commands:
        with (out/(name+'.log')).open('w') as f:subprocess.run([sys.executable,*args],stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
        print('COMPLETED',name,flush=True)
    pairs=[('main/summary.json','main/summary.json'),('main/events.json.gz','main/events.json.gz'),('main/anchor_windows.json','main/anchor_windows.json'),('main/compact.json','main/compact.json'),('review/verification.json','review/verification.json')]
    results=[]
    for saved,new in pairs:
        r=compare(load(root/saved),load(out/new));results.append({'source':saved,**r});print('COMPARISON',saved,r['passed'],r['exact'],r['maximum_absolute_difference'],flush=True)
    receipt={'scope':'Same-event reproducibility and independent arithmetic checks; not new blinded data, new WRPL decoding, or production regression','manifest_files_verified':len(manifest),'result_sets':len(results),'passed':sum(r['passed'] for r in results),'exact':sum(r['exact'] for r in results),'float_tolerance':'absolute1e-9 + relative1e-11, integer/text/structure exact; not a physical tolerance','comparisons':results}
    (out/'reproduction_verification.json').write_text(json.dumps(receipt,indent=2))
    if not all(r['passed'] for r in results):raise AssertionError('Numerical reproduction did not pass; inspect receipt and logs')
    print('REPRODUCTION',json.dumps({k:v for k,v in receipt.items() if k!='comparisons'}),flush=True)
if __name__=='__main__':main()
