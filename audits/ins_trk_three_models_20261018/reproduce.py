#!/usr/bin/env python3
"""Reexecute the supplied game-replay diagnostics without network access.
The original input, recorded predictions and coefficients remain read-only.
"""
from __future__ import annotations
import gzip,hashlib,json,math,os,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def load(p):
    with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return json.load(f)

def compare(a,b):
    result=dict(exact=True,passed=True,numeric_differences=0,max_abs=0.,largest=[])
    def walk(x,y,path):
        if isinstance(x,(float,int)) and not isinstance(x,bool) and isinstance(y,(float,int)) and not isinstance(y,bool):
            if isinstance(x,int) and isinstance(y,int):
                if x!=y:result['exact']=result['passed']=False;result['largest'].append({'path':path,'kind':'integer_difference'})
                return
            d=abs(float(x)-float(y))
            if d:
                result['exact']=False;result['numeric_differences']+=1;result['max_abs']=max(result['max_abs'],d)
                tol=1e-9+1e-11*max(abs(float(x)),abs(float(y)))
                if d>tol:result['passed']=False
                result['largest'].append(dict(path=path,reference=x,reproduced=y,abs_difference=d,tolerance=tol))
            return
        if type(x)!=type(y):result['passed']=result['exact']=False;result['largest'].append(dict(path=path,kind='type_difference'));return
        if isinstance(x,dict):
            if set(x)!=set(y):result['passed']=result['exact']=False;result['largest'].append(dict(path=path,kind='key_difference'));return
            for k in x:walk(x[k],y[k],path+'/'+str(k))
        elif isinstance(x,list):
            if len(x)!=len(y):result['passed']=result['exact']=False;result['largest'].append(dict(path=path,kind='length_difference'));return
            for i,(v,w) in enumerate(zip(x,y)):walk(v,w,path+'/'+str(i))
        elif x!=y:result['passed']=result['exact']=False;result['largest'].append(dict(path=path,kind='value_difference'))
    walk(a,b,'')
    result['largest']=sorted(result['largest'],key=lambda r:r.get('abs_difference',math.inf),reverse=True)[:10]
    return result

def main():
    if len(sys.argv)!=2:raise SystemExit('Usage: python3 reproduce.py /absolute/path/to/new-output')
    out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=False)
    manifest=load(ROOT/'MANIFEST.json')
    for r in manifest:
        p=ROOT/r['path'];assert p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],str(p)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    code=ROOT/'code';jobs=[('test_range',[str(code/'test_range_check.py')]),('test_other',[str(code/'other_checks.py'),'tests'])]
    jobs.append(('range',[str(code/'range_check.py'),'--input',str(ROOT/'input'),'--out',str(out/'studies/range/results')]))
    for fam in ('mixture','representation'):jobs.append((fam,[str(code/'other_checks.py'),'--family',fam,'--input',str(ROOT/'input'),'--out',str(out/f'studies/{fam}/results')]))
    jobs.append(('review',[str(code/'review_results.py'),'--input',str(ROOT/'input'),'--saved',str(out/'studies'),'--out',str(out/'review')]))
    for label,args in jobs:
        # Result directories are made by their own scripts; create parents only.
        if label in ('range','mixture','representation'):(out/f'studies/{label}').mkdir(parents=True,exist_ok=True)
        with (out/(label+'.log')).open('w') as f:subprocess.run([sys.executable,*args],check=True,stdout=f,stderr=subprocess.STDOUT,env=env)
        print('COMPLETED',label,flush=True)
    pairs=[]
    for fam in ('range','mixture','representation'):
        pairs.extend([(f'studies/{fam}/results/summary.json',True),(f'studies/{fam}/results/rows.json.gz',False)])
    pairs.extend([('studies/range/results/scaling_diagnostics.json.gz',False),('review/review.json',False),('review/two_component_diagnostics.json.gz',False)])
    reports=[]
    for path,strip in pairs:
        a,b=load(ROOT/path),load(out/path)
        provenance=None
        if strip:
            pa=a.pop('provenance');pb=b.pop('provenance')
            assert pa['input_capture_sha256' if 'input_capture_sha256' in pa else 'capture_sha256']==pb['input_capture_sha256' if 'input_capture_sha256' in pb else 'capture_sha256']
            # Source hashes are stable; environment/path metadata is reported separately.
            for k in pa:
                if 'sha256' in k:assert pa[k]==pb[k],(path,k)
            provenance=dict(reference=pa,reproduced=pb)
        r=compare(a,b);reports.append(dict(path=path,**r,provenance=provenance))
        print('COMPARISON',path,r['passed'],r['exact'],r['max_abs'],flush=True)
    receipt=dict(scope='Same frozen observations rerun; not new replay observations or production regression',files_hash_checked=len(manifest),result_sets=len(reports),passed_sets=sum(r['passed'] for r in reports),exact_sets=sum(r['exact'] for r in reports),max_abs_difference=max(r['max_abs'] for r in reports),float_tolerance='absolute1e-9 plus relative1e-11, structures/integers/text exact',comparisons=reports)
    (out/'reproduction_verification.json').write_text(json.dumps(receipt,indent=2))
    print('REPRODUCTION',json.dumps({k:v for k,v in receipt.items() if k!='comparisons'}),flush=True)
    if not all(r['passed'] for r in reports):raise AssertionError('Reproduction differs; inspect reproduction_verification.json')

if __name__=='__main__':main()
