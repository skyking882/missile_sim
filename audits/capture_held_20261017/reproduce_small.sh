#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")" && pwd)
py=${PYTHON3:-python3}
out=${1:-"$root/rerun"}
test ! -e "$out" || { echo 'Choose a new output directory' >&2; exit 2; }
mkdir -p "$out"
out=$(cd -- "$out" && pwd)
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$py" "$root/core/test_contract.py" 2>&1 | tee "$out/tests.log"
"$py" "$root/followup/clock_followup.py" --results "$root/core" --out "$out/followup" > "$out/followup.log"
work="$out/work"
mkdir -p "$work/completed" "$work/audits/capture_held_20261017" "$work/phase-results" "$work/prior-results"
ln -s "$root/core" "$work/completed/audit-results"
cp "$root/core/audit.py" "$work/audits/capture_held_20261017/audit.py"
(cd "$work" && "$py" "$root/phase/phase_diagnostic.py") > "$out/phase.log"
(cd "$work" && "$py" "$root/censored_prior/censored_capture.py") > "$out/censored_prior.log"
"$py" - "$root" "$out" <<'PY'
import gzip,json,math,sys
from pathlib import Path
root,out=map(Path,sys.argv[1:])
pairs=[('followup/followup.json','followup/followup.json'),('followup/fresh_return_predictions.json.gz','followup/fresh_return_predictions.json.gz'),('followup/anchor_time_diagnostics.json.gz','followup/anchor_time_diagnostics.json.gz'),('phase/phase_results.json','work/phase-results/phase_results.json'),('censored_prior/censored_capture_results.json','work/prior-results/censored_capture_results.json'),('censored_prior/censored_capture_events.json.gz','work/prior-results/censored_capture_events.json.gz')]
def load(p):
    with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return json.load(f)
def compare(x,y):
    report=dict(json_exact_match=True,structural_errors=[],numeric_differences=0,max_abs_difference=0.,max_tolerance_ratio=0.,largest_differences=[])
    def walk(a,b,path):
        if isinstance(a,(float,int)) and not isinstance(a,bool) and isinstance(b,(float,int)) and not isinstance(b,bool):
            if isinstance(a,int) and isinstance(b,int):
                if a!=b:report['structural_errors'].append(path+': integer differs')
                return
            delta=abs(float(a)-float(b))
            if delta:
                tol=1e-9+1e-12*max(abs(float(a)),abs(float(b)))
                report['json_exact_match']=False;report['numeric_differences']+=1
                report['max_abs_difference']=max(report['max_abs_difference'],delta)
                report['max_tolerance_ratio']=max(report['max_tolerance_ratio'],delta/tol)
                report['largest_differences'].append(dict(path=path,reference=a,reproduced=b,abs_difference=delta,tolerance=tol))
            return
        if type(a)!=type(b):report['structural_errors'].append(path+': type differs');return
        if isinstance(a,dict):
            if set(a)!=set(b):report['structural_errors'].append(path+': keys differ');return
            for k in a:walk(a[k],b[k],path+'/'+str(k))
        elif isinstance(a,list):
            if len(a)!=len(b):report['structural_errors'].append(path+': length differs');return
            for i,(u,v) in enumerate(zip(a,b)):walk(u,v,path+'/'+str(i))
        elif a!=b:report['structural_errors'].append(path+': value differs')
    walk(x,y,'')
    report['largest_differences']=sorted(report['largest_differences'],key=lambda d:d['abs_difference'],reverse=True)[:15]
    report['within_declared_numeric_tolerance']=not report['structural_errors'] and report['max_tolerance_ratio']<=1
    if report['structural_errors']:report['json_exact_match']=False
    return report
rows=[]
for a,b in pairs:
    r=compare(load(root/a),load(out/b));rows.append(dict(reference=a,reproduced=b,**r))
(out/'reproduction-verification.json').write_text(json.dumps(dict(tolerance='absolute 1e-9 + relative 1e-12 for floating values only; integer/text/key/order checks exact',results=rows),indent=2))
for r in rows:print('REPRODUCTION_COMPARISON',json.dumps(r),flush=True)
if not all(r['within_declared_numeric_tolerance'] for r in rows):raise AssertionError('Packaged reproduction differs beyond declared tolerance; see reproduction-verification.json')
print(json.dumps({'checked_result_sets':len(rows),'exact_json_sets':sum(r['json_exact_match'] for r in rows),'all_within_numeric_tolerance':True},indent=2))
PY
