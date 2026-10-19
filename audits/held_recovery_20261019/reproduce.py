#!/usr/bin/env python3
"""Offline reproduction of the already completed diagnosis only.
The blocked FOLLOWUP_PROTOCOL experiments are not implemented here.
"""
from __future__ import annotations
import gzip,hashlib,importlib.util,json,math,os,subprocess,sys
from pathlib import Path

def load(p):
    with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return json.load(f)
def dot(a,b):return math.fsum(float(x)*float(y) for x,y in zip(a,b))
def norm(a):return math.sqrt(dot(a,a))
def cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
def angle(a,b):return math.atan2(norm(cross(a,b)),dot(a,b))*180/math.pi

def verify_arithmetic(prior,summary):
    histories=load(prior/'input/anchor_full_histories.json.gz');direction_checks=axis_checks=power_checks=0;mx={'direction_deg':0.,'pointing_deg':0.,'signal_db':0.}
    for uid,case in summary['cases'].items():
        by={r['index']:r for r in histories[uid]}
        for episode in case['episodes']:
            seed=episode['seed'];n=seed['n'];w=seed['w'];wn=norm(w);axis=[v/wn for v in w];nxw=cross(n,axis);nw=dot(n,axis)
            for r in episode['rows']:
                actual=by[r['index']];assert actual['raw_sha']==r['raw_sha'];dt=(r['tick']-seed['tick'])/48;c=math.cos(wn*dt);s=math.sin(wn*dt)
                predicted=[n[i]*c+nxw[i]*s+axis[i]*nw*(1-c) for i in range(3)]
                err=angle(predicted,actual['n']);mx['direction_deg']=max(mx['direction_deg'],abs(err-r['direction_prediction_error_deg']));direction_checks+=1
                q=actual['Q'];local=[dot([q[j][i] for j in range(3)],predicted) for i in range(3)];scale=norm(local);local=[v/scale for v in local]
                yaw=max(-55*math.pi/180,min(55*math.pi/180,math.atan2(-local[2],local[0])));pitch=max(-55*math.pi/180,min(55*math.pi/180,math.asin(max(-1,min(1,local[1])))))
                v=[math.cos(pitch)*math.cos(yaw),math.sin(pitch),-math.cos(pitch)*math.sin(yaw)];predaxis=[dot(row,v) for row in q]
                y,p=actual['angles'];v=[math.cos(p)*math.cos(y),math.sin(p),-math.cos(p)*math.sin(y)];trueaxis=[dot(row,v) for row in q]
                mx['pointing_deg']=max(mx['pointing_deg'],abs(angle(predaxis,trueaxis)-r['candidates']['angular_const_rate']['axis_error_deg']));axis_checks+=1
                for candidate in r['candidates'].values():
                    for key,value in candidate['beam'].items():
                        width,rcs=map(float,key.split('_'));lg=max(-math.log(2)*(value['offset_deg']/width)**2,math.log(.001));margin=10*math.log10(rcs/2)+40*math.log10(16000/value['range_m'])+20/math.log(10)*lg
                        mx['signal_db']=max(mx['signal_db'],abs(margin-value['margin_db']));assert (margin>=0)==value['pass_signal'];power_checks+=1
    assert max(mx.values())<1e-9,mx
    return dict(scope='Alternate scalar Rodrigues/matrix arithmetic and logarithmic evaluation of the already completed formulas; no new model or samples',direction_checks=direction_checks,pointing_checks=axis_checks,power_checks=power_checks,maximum_absolute_differences=mx)

def main():
    if len(sys.argv)!=2:raise SystemExit('Usage: python3 reproduce.py /absolute/path/to/new-output')
    root=Path(__file__).resolve().parent;out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=False);manifest=load(root/'MANIFEST.json')
    for r in manifest:
        p=root/r['path'];assert p.is_file() and p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],str(p)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1');code=root/'audits/held_recovery_20261019/diagnose.py'
    with (out/'diagnosis.log').open('w') as f:subprocess.run([sys.executable,str(code),'--input',str(root/'prior'),'--out',str(out/'results')],stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
    spec=importlib.util.spec_from_file_location('prior_comparator',root/'prior/reproduce.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    comparisons=[]
    for name in ('summary.json','all_gates_118.json.gz','all_gates_148.json.gz'):
        r=mod.compare(load(root/'results'/name),load(out/'results'/name));comparisons.append(dict(file=name,**r))
    arithmetic=verify_arithmetic(root/'prior',load(out/'results/summary.json'));receipt=dict(scope='Same-input reproduction, not blind validation or a production flight',manifest_files=len(manifest),comparison_count=len(comparisons),passed=sum(c['passed'] for c in comparisons),exact=sum(c['exact'] for c in comparisons),comparisons=comparisons,independent_arithmetic=arithmetic)
    (out/'verification.json').write_text(json.dumps(receipt,indent=2));print('VERIFICATION',json.dumps(receipt),flush=True)
    assert receipt['passed']==len(comparisons)
if __name__=='__main__':main()
