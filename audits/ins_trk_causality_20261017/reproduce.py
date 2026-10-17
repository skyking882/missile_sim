#!/usr/bin/env python3
"""Reproduce included cohort-level experiments without network access.
Use: python3 reproduce.py /absolute/path/to/new-output-directory
Requires NumPy and SciPy. Original WRPL decoding is a separate larger workflow.
"""
from __future__ import annotations
import gzip,json,math,os,subprocess,sys
from pathlib import Path

ABS=1e-7
REL=1e-10

def load(p):
    with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open(encoding='utf8')) as f:return json.load(f)

def compare(a,b):
    r=dict(exact=True,float_differences=0,max_abs_difference=0.,max_tolerance_ratio=0.,structural_errors=[],largest_differences=[])
    def walk(x,y,path):
        if isinstance(x,(float,int)) and not isinstance(x,bool) and isinstance(y,(float,int)) and not isinstance(y,bool):
            if isinstance(x,int) and isinstance(y,int):
                if x!=y:r['structural_errors'].append(path+': integer mismatch')
                return
            dx=abs(float(x)-float(y))
            if not math.isfinite(dx):r['structural_errors'].append(path+': nonfinite difference');return
            if dx:
                tol=ABS+REL*max(abs(x),abs(y));r['exact']=False;r['float_differences']+=1;r['max_abs_difference']=max(r['max_abs_difference'],dx);r['max_tolerance_ratio']=max(r['max_tolerance_ratio'],dx/tol)
                if len(r['largest_differences'])<12 or dx>min(z['difference'] for z in r['largest_differences']):
                    r['largest_differences'].append(dict(path=path,reference=x,reproduced=y,difference=dx,tolerance=tol));r['largest_differences']=sorted(r['largest_differences'],key=lambda z:z['difference'],reverse=True)[:12]
            return
        if type(x)!=type(y):r['structural_errors'].append(path+': type mismatch');return
        if isinstance(x,dict):
            if set(x)!=set(y):r['structural_errors'].append(path+': keys mismatch');return
            for k in x:walk(x[k],y[k],path+'/'+str(k))
        elif isinstance(x,list):
            if len(x)!=len(y):r['structural_errors'].append(path+': length mismatch');return
            for i,(u,v) in enumerate(zip(x,y)):walk(u,v,path+'/'+str(i))
        elif x!=y:r['structural_errors'].append(path+': value mismatch')
    walk(a,b,'');r['passed']=not r['structural_errors'] and r['max_tolerance_ratio']<=1
    if r['structural_errors']:r['exact']=False
    return r

def main():
    if len(sys.argv)!=2:raise SystemExit('Usage: python3 reproduce.py /path/to/new-output')
    root=Path(__file__).resolve().parent;out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=False)
    src=root/'audits/ins_trk_causality_20261017';main=root/'data/main';reset=root/'data/reset';implicit=root/'data/implicit';legacy=root/'data/legacy'
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    for task in ('frame','capture','causality','verification'):
        with (out/(task+'.log')).open('w') as log:subprocess.run([sys.executable,str(src/'explore.py'),task,'--evidence',str(legacy),'--cohort',str(main/'input'),'--out',str(out/task)],stdout=log,stderr=subprocess.STDOUT,check=True,env=env)
    for task in ('reset','implicit'):
        with (out/(task+'.log')).open('w') as log:subprocess.run([sys.executable,str(src/'followup.py'),task,'--input',str(main),'--out',str(out/task)],stdout=log,stderr=subprocess.STDOUT,check=True,env=env)
    with (out/'review.log').open('w') as log:subprocess.run([sys.executable,str(src/'review_checks.py'),'--main',str(main),'--reset',str(reset),'--implicit',str(implicit),'--out',str(out/'independent_review')],stdout=log,stderr=subprocess.STDOUT,check=True,env=env)
    tasks={'frame':['frame_summary.json','frame_rows.json.gz','shrink_rows.json.gz'],'capture':['capture_timing_summary.json','capture_timing_events.json.gz','capture_phase_exclusions.json'],'causality':['causality_summary.json','causality_oof_rows.json.gz'],'verification':['verification.json'],'reset':['reset_summary.json','reset_events.json.gz','reset_fitted_oof.json.gz','contracted_multistep_events.json.gz'],'implicit':['implicit_summary.json','implicit_rows.json.gz']}
    rows=[]
    for task,names in tasks.items():
        reference_root=main/task if task in ('frame','capture','causality','verification') else (reset if task=='reset' else implicit)
        for name in names:
            reference=next(reference_root.rglob(name));reproduced=out/task/name;r=compare(load(reference),load(reproduced));rows.append(dict(task=task,file=name,**r));print('REPRODUCED',task,name,r['passed'],r['exact'],r['max_abs_difference'],flush=True)
    report=dict(floating_tolerance=dict(absolute=ABS,relative=REL),integer_text_keys_and_list_order='exact',results=rows,all_passed=all(r['passed'] for r in rows))
    (out/'reproduction_verification.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    if not report['all_passed']:raise AssertionError('Reproduction differs beyond declared numerical tolerance; original outputs were not overwritten')
    print('ALL_REPRODUCED',len(rows),flush=True)
if __name__=='__main__':main()
