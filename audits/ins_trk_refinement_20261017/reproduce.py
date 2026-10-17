#!/usr/bin/env python3
"""Run the packaged offline replay diagnostics without altering their reference results."""
from __future__ import annotations
import gzip,hashlib,json,math,os,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
if not (ROOT/'evidence').is_dir():
    raise SystemExit('Run the copy of reproduce.py at the extracted delivery root.')

def load(path):
    with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as f:return json.load(f)

def compare(x,y):
    r=dict(exact=True,structural_errors=[],float_differences=0,max_abs_float_difference=0.,max_tolerance_ratio=0.,largest_differences=[])
    def walk(a,b,path):
        if isinstance(a,(int,float)) and not isinstance(a,bool) and isinstance(b,(int,float)) and not isinstance(b,bool):
            if isinstance(a,int) and isinstance(b,int):
                if a!=b:r['structural_errors'].append(path+': integer differs');r['exact']=False
                return
            if not(math.isfinite(float(a)) and math.isfinite(float(b))):raise ValueError('Nonfinite saved metric: '+path)
            diff=abs(a-b)
            if diff:
                r['exact']=False;r['float_differences']+=1;tol=1e-5+1e-9*max(abs(a),abs(b));ratio=diff/tol
                r['max_abs_float_difference']=max(r['max_abs_float_difference'],diff);r['max_tolerance_ratio']=max(r['max_tolerance_ratio'],ratio)
                if len(r['largest_differences'])<12 or diff>r['largest_differences'][-1]['difference']:
                    r['largest_differences'].append(dict(path=path,reference=a,reproduced=b,difference=diff,tolerance=tol));r['largest_differences'].sort(key=lambda z:z['difference'],reverse=True);r['largest_differences']=r['largest_differences'][:12]
            return
        if type(a)!=type(b):r['structural_errors'].append(path+': type differs');r['exact']=False;return
        if isinstance(a,dict):
            if set(a)!=set(b):r['structural_errors'].append(path+': keys differ');r['exact']=False;return
            for key in a:walk(a[key],b[key],path+'/'+str(key))
        elif isinstance(a,list):
            if len(a)!=len(b):r['structural_errors'].append(path+': length differs');r['exact']=False;return
            for i,(u,v) in enumerate(zip(a,b)):walk(u,v,path+'/'+str(i))
        elif a!=b:r['structural_errors'].append(path+': value differs');r['exact']=False
    walk(x,y,'');r['passed']=not r['structural_errors'] and r['max_tolerance_ratio']<=1
    return r

def main():
    if len(sys.argv)!=2:raise SystemExit('Usage: python3 reproduce.py /absolute/path/to/new-output')
    out=Path(sys.argv[1]).expanduser().resolve();out.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    code=ROOT/'audits/ins_trk_refinement_20261017';evidence=ROOT/'evidence'
    sources=load(ROOT/'tracks/quant/sources.json')
    for rel,h in sources.items():
        assert hashlib.sha256((evidence/rel).read_bytes()).hexdigest()==h,rel
    def run(label,args):
        with (out/(label+'.log')).open('w') as f:
            result=subprocess.run([sys.executable,'-u',*map(str,args)],env=env,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('Execution failed; inspect '+str(out/(label+'.log')))
        print('COMPLETED',label,flush=True)
    run('tests',[code/'refine.py','tests'])
    for task in ('quant','phase','temporal'):run(task,[code/'refine.py',task,'--input',evidence,'--out',out/'tracks'/task])
    run('followup',[code/'followup_review.py','--input',evidence,'--results',out/'tracks','--out',out/'followup'])
    run('bridge',[code/'bridge.py','--input',evidence,'--lag',out/'followup/lag_summary.json','--out',out/'bridge'])
    files=['tracks/quant/quantization_summary.json','tracks/quant/quantization_rows.json.gz','tracks/phase/phase_summary.json','tracks/phase/phase_rows.json.gz','tracks/temporal/temporal_summary.json','tracks/temporal/temporal_rows.json.gz','followup/independent_review.json','followup/anchor_loss_boundaries_corrected.json','followup/first_capture_summary.json','followup/first_capture_rows.json.gz','followup/lag_summary.json','followup/lag_rows.json.gz','bridge/bridge_summary.json','bridge/bridge_rows.json.gz']
    comparisons=[]
    for rel in files:
        result=compare(load(ROOT/rel),load(out/rel));comparisons.append(dict(file=rel,**result))
        print('COMPARISON',rel,result['passed'],result['exact'],result['max_abs_float_difference'],flush=True)
    report=dict(scope='Full numeric refitting and diagnostic rerun; not new experimental observations.',floating_tolerance='absolute1e-5 + relative1e-9; structures, integers, strings, booleans exact',input_files_hash_checked=len(sources),output_sets=len(comparisons),passed_sets=sum(r['passed'] for r in comparisons),exact_sets=sum(r['exact'] for r in comparisons),all_passed=all(r['passed'] for r in comparisons),maximum_absolute_float_difference=max(r['max_abs_float_difference'] for r in comparisons),results=comparisons)
    (out/'reproduction_verification.json').write_text(json.dumps(report,indent=2))
    print('REPRODUCTION_RESULT',json.dumps({k:v for k,v in report.items() if k!='results'}),flush=True)
    if not report['all_passed']:raise AssertionError('Comparison failed; original results untouched, inspect reproduction_verification.json')

if __name__=='__main__':main()
