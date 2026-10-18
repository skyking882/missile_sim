#!/usr/bin/env python3
"""Run from the extracted handover root. No network calls; no input overwrites."""
from __future__ import annotations
import gzip,hashlib,json,math,os,subprocess,sys
from pathlib import Path

def load(p):
    with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return json.load(f)
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def compare(x,y):
    r=dict(structural_errors=[],float_differences=0,max_abs_float_difference=0.,max_tolerance_ratio=0.,examples=[])
    def walk(a,b,path):
        if isinstance(a,(int,float)) and not isinstance(a,bool) and isinstance(b,(int,float)) and not isinstance(b,bool):
            if isinstance(a,int) and isinstance(b,int):
                if a!=b:r['structural_errors'].append(path+': integer differs')
                return
            delta=abs(float(a)-float(b))
            if not math.isfinite(delta):r['structural_errors'].append(path+': nonfinite');return
            if delta:
                tol=1e-8+1e-11*max(abs(float(a)),abs(float(b)))
                r['float_differences']+=1;r['max_abs_float_difference']=max(r['max_abs_float_difference'],delta);r['max_tolerance_ratio']=max(r['max_tolerance_ratio'],delta/tol)
                if len(r['examples'])<10:r['examples'].append(dict(path=path,reference=a,reproduced=b,difference=delta))
            return
        if type(a)!=type(b):r['structural_errors'].append(path+': type differs');return
        if isinstance(a,dict):
            if set(a)!=set(b):r['structural_errors'].append(path+': keys differ');return
            for k in a:walk(a[k],b[k],path+'/'+k)
        elif isinstance(a,list):
            if len(a)!=len(b):r['structural_errors'].append(path+': list length differs');return
            for i,(v,w) in enumerate(zip(a,b)):walk(v,w,path+'/'+str(i))
        elif a!=b:r['structural_errors'].append(path+': value differs')
    walk(x,y,'')
    r['exact_json']=not r['structural_errors'] and r['float_differences']==0;r['passed']=not r['structural_errors'] and r['max_tolerance_ratio']<=1
    return r

def main():
    if len(sys.argv)!=2:raise SystemExit('Usage: python3 reproduce.py /absolute/path/to/new-output')
    root=Path(__file__).resolve().parent;out=Path(sys.argv[1]).resolve()
    if out.exists():raise SystemExit('Output exists; choose a new path')
    manifest=root/'MANIFEST.json'
    if not manifest.exists():raise SystemExit('Run this from the complete extracted handover; root MANIFEST.json missing')
    checked=0
    for r in load(manifest):
        p=root/r['path']
        if p.stat().st_size!=r['bytes'] or digest(p)!=r['sha256']:raise RuntimeError('Delivery hash mismatch: '+str(p))
        checked+=1
    out.mkdir(parents=True);env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    script=root/'audits/ins_trk_handover_20261018/final_round.py';review=root/'audits/ins_trk_handover_20261018/review_handover.py'
    commands=[('tests',[sys.executable,str(script),'tests']),('round',[sys.executable,str(script),'run','--input',str(root/'prior'),'--out',str(out/'final-results')]),('review',[sys.executable,str(review),'--results',str(out/'final-results'),'--out',str(out/'review')])]
    for name,cmd in commands:
        with (out/(name+'.log')).open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
        print('COMPLETED',name,flush=True)
    paths=['final-results/joint_summary.json','final-results/joint_rows.json.gz','final-results/seed_invariant_summary.json','final-results/seed_rows.json.gz','final-results/six_exceptions.json','review/independent_review.json']
    checks=[]
    for path in paths:
        r=compare(load(root/path),load(out/path));checks.append(dict(path=path,**r));print('COMPARISON',path,r['passed'],r['exact_json'],r['max_abs_float_difference'],flush=True)
    report=dict(scope='Reexecution of same frozen data; not new replay observations or production model regression',delivery_files_hash_checked=checked,result_sets=len(checks),passed_sets=sum(r['passed'] for r in checks),exact_sets=sum(r['exact_json'] for r in checks),maximum_absolute_float_difference=max(r['max_abs_float_difference'] for r in checks),tolerance='float abs1e-8 + relative1e-11; integers, booleans, strings, key sets and list lengths exact',results=checks)
    (out/'reproduction_verification.json').write_text(json.dumps(report,indent=2))
    print('REPRODUCTION',json.dumps({k:v for k,v in report.items() if k!='results'}),flush=True)
    if not all(r['passed'] for r in checks):raise RuntimeError('Reproduction mismatch; diagnostics preserved')
if __name__=='__main__':main()
