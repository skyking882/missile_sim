#!/usr/bin/env python3
"""Bounded War Thunder serialized-state audit; not a production seeker model."""
from __future__ import annotations
import argparse,collections,copy,csv,hashlib,itertools,json,math,platform,sys,unittest
from pathlib import Path
import numpy as np
from scipy.optimize import linprog,minimize
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ins_trk_refinement_20261017'))
import refine as F
E,A=F.E,F.A
H=1/48;TAU=.078525;K=.8;QSTEP=15/32767

def vector(x):return np.asarray(x,dtype=float)
def hashfile(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def verify_manifest(root):
    rows=json.loads((root/'MANIFEST.json').read_text())
    for r in rows:
        p=root/r['path']
        if p.stat().st_size!=r['bytes'] or hashfile(p)!=r['sha256']:raise RuntimeError('Input checksum: '+str(p))
    return len(rows)

def feedback(m,s,alpha):
    p=A.rotate(m,s,H);n=A.blend(p,m,alpha)
    return n,s+A.correction(p,n)/(TAU+H/2)

def inverse_direction(n,s,alpha):
    """Eliminate unknown reset ray using n only. Never receives response rate/PV."""
    m=vector(n).copy()
    for _ in range(20):
        out,_=feedback(m,s,alpha)
        m=A.unit(Rotation.from_rotvec(-A.correction(out,n)).apply(m))
    out,w=feedback(m,s,alpha)
    return m,out,w,A.angle(out,n)

def box_norm_range(lo,hi,B=None):
    """Exact convex-quadratic box minimum by all active sets; max at vertices."""
    lo,hi=vector(lo),vector(hi);B=np.eye(3) if B is None else vector(B)
    C=B.T@B;best=math.inf;arg=None
    for state in itertools.product((-1,0,1),repeat=3):
        x=np.zeros(3);fixed=[i for i,v in enumerate(state) if v];free=[i for i,v in enumerate(state) if not v]
        for i in fixed:x[i]=lo[i] if state[i]<0 else hi[i]
        if free:
            x[free]=np.linalg.solve(C[np.ix_(free,free)],-C[np.ix_(free,fixed)]@x[fixed])
        if np.any(x<lo-1e-9) or np.any(x>hi+1e-9):continue
        z=float(np.linalg.norm(B@x))
        if z<best:best=z;arg=x.copy()
    corners=np.array(list(itertools.product(*zip(lo,hi))));magnitudes=np.linalg.norm(corners@B.T,axis=1)
    if arg is None:raise RuntimeError('No box active-set solution')
    return best,float(magnitudes.max()),arg

def affine_slack(M,bias,k0,k1):
    lo,hi=F.cell(k0,'truncate');l1,u1=F.cell(k1,'truncate');bias=vector(bias)
    mat=np.vstack([np.column_stack((M,-np.ones(3))),np.column_stack((-M,-np.ones(3)))])
    rhs=np.r_[u1-bias,-l1+bias]
    fit=linprog([0,0,0,1],A_ub=mat,b_ub=rhs,bounds=list(zip(lo,hi))+[(0,None)],method='highs')
    if not fit.success:raise RuntimeError(fit.message)
    check=M@fit.x[:3]+bias
    if max(np.max(l1-check),np.max(check-u1))>fit.x[3]+1e-6:raise RuntimeError('Invalid LP solution')
    return float(max(0,fit.x[3])),fit.x[:3]

def geo(P,Vs,p,vm):
    r=P-p
    return np.cross(Vs-vm,r)/float(r@r)

def seed_candidates(old,own_p_new,own_v_new,T):
    """Current target state and response direction/rate cannot enter here."""
    x=old;g0=geo(x['Ps'],x['Vs'],x['p'],x['vm']);out={'raw':x['w'].copy(),'geometry_old':g0}
    for label,t in [('half',T/2),('before_last_tick',max(0,T-H)),('full',T)]:
        f=t/T;p=x['p']+f*(own_p_new-x['p']);vm=x['vm']+f*(own_v_new-x['vm'])
        gt=geo(x['Ps']+x['Vs']*t,x['Vs'],p,vm)
        out['delta_geometry_'+label]=x['w']+gt-g0
        if label=='full':out['geometry_full']=gt
    return out

def summary_models(rows,key='scores'):
    if not rows:return {'n':0}
    names=list(rows[0][key]);out={}
    for name in names:
        rr=[r[key][name] for r in rows]
        out[name]={k:A.metric([v[k] for v in rr]) for k in ('rate_deg_s','axial_deg_s','transverse_deg_s') if k in rr[0]}
        if 'lp_slack_cells' in rr[0]:out[name]['rate_cell_feasible']=sum(v['lp_slack_cells']<=1e-7 for v in rr);out[name]['lp_slack_cells']=A.metric([v['lp_slack_cells'] for v in rr])
    return out

def err(w,b):
    e=w-b['w'];para=float(e@b['n'])
    return dict(rate_deg_s=float(np.linalg.norm(e))*A.RAD,axial_deg_s=para*A.RAD,transverse_deg_s=float(np.linalg.norm(e-para*b['n']))*A.RAD)

def joint_task(events,alphas,out):
    rows=[];guards=0
    for e in events:
        a,b=E.arr(e['old']),E.arr(e['new']);alpha=alphas[e['replay']];s=b['Q']@a['w'];n=b['n']
        npre,wpre=feedback(n,s,alpha);m,npost,wpost,res=inverse_direction(n,s,alpha)
        if res>1e-10:raise RuntimeError('Inverse direction residual '+str(res))
        # Synthetic independent starting perturbations converge to the same root.
        mm=m.copy()
        for _ in range(10):
            check,_=feedback(mm,s,alpha);mm=A.unit(Rotation.from_rotvec(-A.correction(check,n)).apply(mm))
        if A.angle(mm,m)>1e-10:raise RuntimeError('Inverse root not stable')
        inputs=(n.copy(),s.copy(),alpha)
        poison=copy.deepcopy(b)
        for k in ('Ps','Vs','w'):poison[k]=np.full(3,np.nan)
        check=inverse_direction(poison['n'],s,alpha)
        np.testing.assert_array_equal(m,check[0]);np.testing.assert_array_equal(wpost,check[2]);guards+=1
        past=A.unit(a['Ps']+a['Vs']*((b['tick']-a['tick'])/48)-b['p'])
        rows.append(F.ident(e)|dict(alpha_h=alpha,scores={'explicit_k08':err(F.projector(n,K)@s,b),'n_is_before_feedback':err(wpre,b),'n_is_after_feedback':err(wpost,b)},inverse_residual_rad=res,inferred_measurement=m,stored_n=n,pre_to_post_n_deg=A.angle(n,npre)*A.RAD,inferred_m_to_saved_n_deg=A.angle(m,n)*A.RAD,rate_difference_between_stages_deg_s=float(np.linalg.norm(wpre-wpost))*A.RAD,inferred_m_to_pastPV_deg=A.angle(m,past)*A.RAD,saved_n_to_pastPV_deg=A.angle(n,past)*A.RAD,rate_pred_pre=wpre,rate_pred_post=wpost))
    groups={}
    keys=('inverse_residual_rad','pre_to_post_n_deg','inferred_m_to_saved_n_deg','rate_difference_between_stages_deg_s','inferred_m_to_pastPV_deg','saved_n_to_pastPV_deg')
    for name,rr in [('all244',rows),('R77_79',[r for r in rows if r['family']=='R-77-1'])]:groups[name]=dict(n=len(rr),models=summary_models(rr),metrics={k:A.metric([r[k] for r in rr]) for k in keys})
    result=dict(scope='Conditional inversion F_s(m)=saved_n, not observed raw measurement or autonomous prediction. Both stage hypotheses retained.',groups=groups,current_rate_PV_poison_guards=guards,anchors=[r for r in rows if (r['replay'],r['eid']) in A.ANCHORS])
    A.save(out/'joint_summary.json',result);A.save(out/'joint_rows.json.gz',rows)
    print('JOINT',json.dumps(A.serial({k:v for k,v in result.items() if k!='anchors'})),flush=True)
    for r in result['anchors']:print('JOINT_ANCHOR',json.dumps(A.serial(r)),flush=True)
    return rows

def seed_invariant_task(events,out):
    rows=[];guards=0
    for e in events:
        a,b=E.arr(e['old']),E.arr(e['new']);T=(b['tick']-a['tick'])/48;n=b['n'];P=F.projector(n,K);Q=b['Q'];k0,k1=F.rateints(a['w']),F.rateints(b['w'])
        seeds=seed_candidates(a,b['p'],b['vm'],T);scores={}
        for label,w in seeds.items():
            pred=P@Q@w;ex=err(pred,b)
            if label=='raw' or label.startswith('delta_'):
                bias=P@Q@(w-a['w'])/QSTEP;sl,witness=affine_slack(P@Q,bias,k0,k1);ex['lp_slack_cells']=sl
            scores[label]=ex
        # Only input-centered geometric models above; Q-source sensitivity kept separate.
        qmid=Rotation.from_rotvec(.5*Rotation.from_matrix(b['Q']@a['Q'].T).as_rotvec()).as_matrix()@a['Q']
        qs={name:err(P@q@a['w'],b) for name,q in [('old',a['Q']),('midpoint',qmid),('new',Q)]}
        l0,u0=F.cell(k0,'truncate');l1,u1=F.cell(k1,'truncate')
        inmin,inmax,_=box_norm_range(l0,u0)
        Pi=np.eye(3)/K+(1-1/K)*np.outer(n,n)
        outmin,outmax,_=box_norm_range(l1,u1,Pi)
        _,maxw1,_=box_norm_range(l1,u1)
        epsn=.0031/A.RAD;nb=(1/K-1)*math.sin(epsn)*maxw1
        gap=max(0.,outmin-inmax,inmin-outmax)
        gap_n=max(0.,outmin-nb-inmax,inmin-outmax-nb)
        base_sl=scores['raw']['lp_slack_cells']
        sens={}
        for epsq_deg in (0.,.01,.1):
            bound=inmax*((1-K)*math.sin(epsn)+2*math.sin(epsq_deg/A.RAD/2))
            sens[str(epsq_deg)]=dict(max_possible_change_cells=bound,lp_gap_exceeds_bound=base_sl>bound+1e-7)
        # Inverting observed output is diagnostic: does not feed the seed predictions.
        implied=Q.T@Pi@b['w']
        poison=copy.deepcopy(b)
        for k in ('Ps','Vs','n','w'):poison[k]=np.full(3,np.nan)
        pp=seed_candidates(a,poison['p'],poison['vm'],T)
        for name,w in seeds.items():np.testing.assert_array_equal(w,pp[name])
        guards+=1
        r=F.ident(e)|dict(scores=scores,Q_scores=qs,invariant=dict(old_norm_interval_cells=[inmin,inmax],inverse_output_norm_interval_cells=[outmin,outmax],interval_gap_cells=gap,gap_allowing_direction_uncertainty_cells=gap_n,arbitrary_Q_necessary_feasible=gap<=1e-7,arbitrary_Q_necessary_feasible_with_n=gap_n<=1e-7),orientation_sensitivity=sens,geometric_seed_candidates=seeds,observed_inverse_seed_DIAGNOSTIC=implied,old_state=a,new_state=b)
        rows.append(r)
    groups={}
    for name,rr in [('all244',rows),('R77_79',[r for r in rows if r['family']=='R-77-1']),('R77_first',[r for r in rows if r['family']=='R-77-1' and r['original_capture']]),('R77_reacquisition',[r for r in rows if r['family']=='R-77-1' and not r['original_capture']])]:
        groups[name]=dict(n=len(rr),models=summary_models(rr),Q=summary_models(rr,'Q_scores'),invariant_impossible_any_Q=sum(not r['invariant']['arbitrary_Q_necessary_feasible'] for r in rr),invariant_impossible_any_Q_allow_n=sum(not r['invariant']['arbitrary_Q_necessary_feasible_with_n'] for r in rr))
    six=[r for r in rows if r['family']=='R-77-1' and r['scores']['raw']['lp_slack_cells']>1e-7]
    if len(six)!=6:raise AssertionError('Previously fixed six exceptions did not reproduce: '+str(len(six)))
    result=dict(scope='All seed models fixed before scoring. LP geometric correction is deterministic but inherits old PV uncertainty; no new target truth. Arbitrary-Q invariant is necessary, not sufficient. Q stress-test angles are not a sourced precision bound.',groups=groups,current_target_label_poison_guards=guards,six_exceptions=six)
    A.save(out/'seed_invariant_summary.json',result);A.save(out/'seed_rows.json.gz',rows);A.save(out/'six_exceptions.json',six)
    fields=['replay','eid','uid','tick0','tick1','original_capture','limited','raw_rate_error_deg_s','raw_slack_cells','delta_full_rate_error_deg_s','delta_full_slack_cells','delta_half_slack_cells','arbitrary_Q_impossible','arbitrary_Q_impossible_allow_n']
    with (out/'six_exceptions.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in six:w.writerow({k:r[k] for k in fields[:7]}|dict(raw_rate_error_deg_s=r['scores']['raw']['rate_deg_s'],raw_slack_cells=r['scores']['raw']['lp_slack_cells'],delta_full_rate_error_deg_s=r['scores']['delta_geometry_full']['rate_deg_s'],delta_full_slack_cells=r['scores']['delta_geometry_full']['lp_slack_cells'],delta_half_slack_cells=r['scores']['delta_geometry_half']['lp_slack_cells'],arbitrary_Q_impossible=not r['invariant']['arbitrary_Q_necessary_feasible'],arbitrary_Q_impossible_allow_n=not r['invariant']['arbitrary_Q_necessary_feasible_with_n']))
    print('SEED_GROUPS',json.dumps(A.serial(groups)),flush=True)
    for r in six:print('SIX',json.dumps(A.serial({k:r[k] for k in ('replay','uid','tick0','tick1','scores','invariant','orientation_sensitivity')})),flush=True)
    return rows

class Tests(unittest.TestCase):
    def test_forward_inverse(self):
        rng=np.random.default_rng(2018)
        for _ in range(100):
            m=A.unit(rng.normal(size=3));s=rng.normal(size=3)*.4;a=rng.uniform(.7,.99)
            n,w=feedback(m,s,a);mi,ni,wi,r=inverse_direction(n,s,a)
            np.testing.assert_allclose(mi,m,atol=1e-11,rtol=0);np.testing.assert_allclose(wi,w,atol=1e-10,rtol=0);self.assertLess(r,1e-11)
    def test_rotation_invariant(self):
        rng=np.random.default_rng(108)
        for _ in range(200):
            n=A.unit(rng.normal(size=3));Q=Rotation.random(random_state=rng).as_matrix();w=rng.normal(size=3)
            y=F.projector(n,K)@Q@w;P=np.eye(3)/K+(1-1/K)*np.outer(n,n)
            self.assertAlmostEqual(np.linalg.norm(P@y),np.linalg.norm(w),places=12)
    def test_box_extrema(self):
        rng=np.random.default_rng(102)
        for _ in range(60):
            k=rng.integers(-30,30,size=3);lo,hi=F.cell(k,'truncate');n=A.unit(rng.normal(size=3));P=np.eye(3)/K+(1-1/K)*np.outer(n,n)
            mn,mx,z=box_norm_range(lo,hi,P);C=P.T@P
            fit=minimize(lambda x:float(x@C@x),(lo+hi)/2,jac=lambda x:2*C@x,bounds=list(zip(lo,hi)),method='L-BFGS-B',options={'ftol':1e-14,'gtol':1e-10})
            self.assertAlmostEqual(mn*mn,float(fit.fun),places=7)
            pts=rng.uniform(lo,hi,size=(100,3));vals=np.linalg.norm(pts@P.T,axis=1)
            self.assertGreaterEqual(float(vals.min()),mn-1e-8);self.assertLessEqual(float(vals.max()),mx+1e-8)
    def test_affine_lp_reduces_to_old(self):
        rng=np.random.default_rng(11)
        for _ in range(50):
            n=A.unit(rng.normal(size=3));Q=Rotation.random(random_state=rng).as_matrix();M=F.projector(n,K)@Q;k0=rng.integers(-100,100,size=3);k1=np.trunc(M@(k0+.5*np.sign(k0)))
            a,_=affine_slack(M,np.zeros(3),k0,k1);b,_=F.slack_lp(M,k0,k1,'truncate');self.assertAlmostEqual(a,b,places=7)
    def test_seed_zero_increment(self):
        x=dict(Ps=vector([1000.,0,0]),Vs=vector([50.,0,0]),p=np.zeros(3),vm=vector([100.,0,0]),w=vector([.1,0,0]))
        out=seed_candidates(x,vector([100*2/48,0,0]),x['vm'],2/48)
        np.testing.assert_array_equal(out['raw'],out['delta_geometry_full'])
    def test_inverse_contract(self):
        n=A.unit([1,.2,.1]);s=vector([.1,-.2,.3]);out=inverse_direction(n,s,.85)
        self.assertLess(out[3],1e-12)
        self.assertTrue(all(np.isfinite(v).all() for v in out[:3]))

def main():
    p=argparse.ArgumentParser();p.add_argument('task',choices=['tests','run']);p.add_argument('--input',type=Path);p.add_argument('--out',type=Path);a=p.parse_args()
    if a.task=='tests':
        rr=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests));sys.exit(not rr.wasSuccessful())
    if a.input is None or a.out is None:raise ValueError('--input and --out required')
    checked=verify_manifest(a.input);a.out.mkdir(parents=True,exist_ok=False)
    cp=a.input/'evidence/data/main/evidence/capture_events.json.gz';events=E.load(cp)
    selected=[e for e in events if e['new']['tick']==F.phase_after(e['old']['tick']) and e['new']['timer']==0 and e['old']['flag']==e['new']['flag']==0]
    if len(selected)!=244 or sum(e['family']=='R-77-1' for e in selected)!=79:raise RuntimeError('Population changed')
    bridge=E.load(a.input/'bridge/bridge_summary.json');alpha={r['held_scene']:r['alpha_h']['fractional_fixed_lag'] for r in bridge['constants']}
    j=joint_task(selected,alpha,a.out);s=seed_invariant_task(selected,a.out)
    v=dict(input_manifest_checked=checked,total_captures=len(events),selected=len(selected),R77=sum(e['family']=='R-77-1' for e in selected),input_capture_sha256=hashfile(cp),input_manifest_sha256=hashfile(a.input/'MANIFEST.json'),alpha_source_sha256=hashfile(a.input/'bridge/bridge_summary.json'),alpha_h_by_scene=alpha,task_source_sha256=hashfile(Path(__file__)),environment=dict(python=sys.version,numpy=np.__version__,platform=platform.platform()),scope='No new raw replay extraction. Same fixed derived data. Conditional projection/inversion uses current n but not current rate as input.')
    A.save(a.out/'verification.json',v);print('VERIFICATION',json.dumps(v),flush=True)
if __name__=='__main__':main()
