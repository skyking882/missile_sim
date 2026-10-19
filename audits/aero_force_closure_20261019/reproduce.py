#!/usr/bin/env python3
"""Reproduce all completed fixed game-force experiments, including failures.
No network access, coefficient optimization or per-record trajectory correction.
"""
from __future__ import annotations
import gzip,hashlib,json,math,os,subprocess,sys
from pathlib import Path

def read(p):
    with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return json.load(f)

def compare(a,b):
    out=dict(exact=True,max_numeric_difference=0.,errors=[])
    def walk(x,y,path):
        if isinstance(x,(int,float)) and not isinstance(x,bool) and isinstance(y,(int,float)) and not isinstance(y,bool):
            if isinstance(x,int) and isinstance(y,int):
                if x!=y:out['errors'].append(path+': integer')
                return
            d=abs(float(x)-float(y))
            if d:
                out['exact']=False;out['max_numeric_difference']=max(out['max_numeric_difference'],d)
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
    walk(a,b,'');out['passed']=not out['errors'];out['exact']=out['exact'] and out['passed'];return out

def main():
    if len(sys.argv)!=2:raise SystemExit('Usage: python3 reproduce.py /absolute/path/to/new-output')
    root=Path(__file__).resolve().parent;out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=False)
    manifest=read(root/'MANIFEST.json')
    for r in manifest:
        p=root/r['path'];assert p.is_file() and p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],str(p)
    code=root/'audits/aero_force_closure_20261019';sys.path.insert(0,str(code))
    import compare_candidates as B
    E=B.E;prior=root/'prior';histories=E.read(prior/'input/anchor_full_histories.json.gz');infos=E.read(prior/'input/infos.json');profile=E.read(root/'data/profile_sets/2.59.0.27/missiles/su_r_77_1.json');defaults=E.read(root/'config/effective_surface_defaults.json')
    objects={}
    for uid,(rep,eid) in E.ANCHORS.items():
        info=next(i for i in infos if i['replay']==rep and i['eid']==eid);s=E.Series(histories[uid],info['epoch_tick']);objects[uid]=(E.make_runtime(profile,defaults,histories[uid][0]),B.Forcing(s))
    family={c['name']:c for c in B.candidates()};first=out/'first';first.mkdir();checks=[]
    # Replay exact stored candidate choices, not a new optimization/selection.
    for p in sorted((root/'results/first').glob('*.json.gz')):
        expected=E.read(p);c=expected['candidate'];uid='118' if p.name.startswith('train118_') else p.name.split('_')[1]
        assert family[c['name']]==c
        sub=expected['substeps'];got=B.run(*objects[uid],12.5 if uid=='118' else 15.,c,sub);E.save(first/p.name,got)
        checks.append(dict(file='first/'+p.name,**compare(expected,E.clean(got))))
    print('FIRST_PASS_REPRODUCED',len(checks),flush=True)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    with (out/'operators.log').open('w') as f:subprocess.run([sys.executable,str(code/'verify_and_extend.py'),'--source',str(prior),'--first',str(first),'--out',str(out/'operators')],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    for p in sorted((root/'results/operators').iterdir()):
        if p.name.endswith(('.json','.json.gz')):checks.append(dict(file='operators/'+p.name,**compare(E.read(p),E.read(out/'operators'/p.name))))
    receipt=dict(scope='Same-data reconstruction of32 fixed hypotheses. Reproduction success does not mean5m/s or10m physical targets pass. No autonomous flight or new independent data.',manifest_files=len(manifest),result_files=len(checks),passed=sum(r['passed'] for r in checks),exact=sum(r['exact'] for r in checks),maximum_numeric_difference=max(r['max_numeric_difference'] for r in checks),checks=checks)
    (out/'reproduction.json').write_text(json.dumps(receipt,indent=2));print('REPRODUCED',json.dumps({k:v for k,v in receipt.items() if k!='checks'}),flush=True)
    if receipt['passed']!=receipt['result_files']:raise AssertionError('Numerical reproduction failed; see reproduction.json')
if __name__=='__main__':main()
