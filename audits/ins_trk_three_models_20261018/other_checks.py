#!/usr/bin/env python3
"""Additional prespecified serialized-state comparisons. No simulation edits.
Builders intentionally receive no current response vector or current target PV.
"""
import argparse,json,math,unittest,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import range_check as C

def geometry(P,V,p,v):
    r=C.vec(P)-C.vec(p);d=float(r@r)
    if d<1e-12 or not math.isfinite(d):return None
    return np.cross(C.vec(V)-C.vec(v),r)/d

def mixtures(old,new,T):
    n=C.vec(new['n']);Q=C.vec(new['Q']);P=C.D(n);base=P@Q;rates={}
    rates['old_geometry']=geometry(old['Ps'],old['Vs'],old['p'],old['vm'])
    for label,f in [('half_geometry',.5),('full_geometry',1.)]:
        t=T*f;p=(1-f)*C.vec(old['p'])+f*C.vec(new['p']);vm=(1-f)*C.vec(old['vm'])+f*C.vec(new['vm'])
        rates[label]=geometry(C.vec(old['Ps'])+C.vec(old['Vs'])*t,old['Vs'],p,vm)
    R=float(new['R']);rates['conditional_ray_old_V']=np.cross(C.vec(old['Vs'])-C.vec(new['vm']),n)/R if R>1e-6 else None
    out={};bad={}
    for label,u in rates.items():
        valid=u is not None and np.all(np.isfinite(u))
        for order in ('before_Q','after_Q'):
            key=label+'/'+order
            out[key]=(np.zeros((3,3)),(P@Q@u if order=='before_Q' else P@u)) if valid else (base,np.zeros(3))
            bad[key]=None if valid else 'degenerate_geometry_baseline_fallback'
    return out,bad,{k:v for k,v in rates.items()}

def jacobian(Q):
    """Columns: native bank, heading, pitch derivatives; Q=Ry(h)Rz(p)Rx(b)."""
    Q=C.vec(Q);pitch=math.asin(float(np.clip(Q[1,0],-1,1)));bank=math.atan2(-Q[1,2],Q[1,1])
    cp,sp=math.cos(pitch),math.sin(pitch);cb,sb=math.cos(bank),math.sin(bank)
    return np.array([[1,sp,0],[0,cp*cb,sb],[0,-cp*sb,cb]])

def representations(old,new,T):
    n=C.vec(new['n']);Q=C.vec(new['Q']);P=C.D(n);base=P@Q;out={};bad={};extra={}
    for source,q in [('new',Q),('old',C.vec(old['Q']))]:
        J=jacobian(q);cond=float(np.linalg.cond(J));extra[source+'_J_condition']=cond
        for operation in ('J','inverse_J'):
            key=source+'/'+operation;ok=np.isfinite(cond) and (operation!='inverse_J' or cond<=1000)
            out[key]=(P@Q@(J if operation=='J' else np.linalg.inv(J)),np.zeros(3)) if ok else (base,np.zeros(3))
            bad[key]=None if ok else 'inverse_condition_gt1000_baseline_fallback'
    body=Q.T@n;ce=float(np.hypot(body[0],body[2]));yaw=math.atan2(-body[2],body[0]);tan=np.array([-math.sin(yaw),0.,-math.cos(yaw)])
    axis=Q@np.cross(tan,body);axis/=np.linalg.norm(axis);extra['polar_cos_elevation']=ce
    for operation,factor in [('metric',ce),('inverse_metric',1/ce if ce>=1e-3 else 1.)]:
        key='polar/'+operation;ok=ce>=1e-3
        G=np.eye(3)+(factor-1)*np.outer(axis,axis)
        out[key]=(P@G@Q,np.zeros(3)) if ok else (base,np.zeros(3));bad[key]=None if ok else 'polar_cos_lt1e-3_baseline_fallback'
    return out,bad,extra

def native_matrix(a):
    b,h,p=a
    return Rotation.from_rotvec([0,h,0]).as_matrix()@Rotation.from_rotvec([0,0,p]).as_matrix()@Rotation.from_rotvec([b,0,0]).as_matrix()

class Tests(unittest.TestCase):
    def test_jacobian_independent_finite_difference(self):
        rng=np.random.default_rng(903)
        for _ in range(150):
            a=rng.uniform(-1.2,1.2,3);v=rng.normal(size=3);Q=native_matrix(a);dt=1e-6
            derivative=(native_matrix(a+dt*v)-native_matrix(a-dt*v))/(2*dt);B=Q.T@derivative
            wb=np.array([B[2,1]-B[1,2],B[0,2]-B[2,0],B[1,0]-B[0,1]])/2
            np.testing.assert_allclose(jacobian(Q)@v,wb,atol=2e-9,rtol=1e-9)
    def test_jacobian_inverse(self):
        for a in ([.1,.2,.3],[-.5,.4,.8],[.7,.1,-.2]):
            J=jacobian(native_matrix(a));np.testing.assert_allclose(J@np.linalg.inv(J),np.eye(3),atol=1e-14)
    def test_identity_representation_at_zero(self):
        n=np.array([1.,0,0]);x={'Q':np.eye(3)};y={'Q':np.eye(3),'n':n}
        mods,bad,_=representations(x,y,.02)
        self.assertEqual(len(mods),6)
        for M,b in mods.values():np.testing.assert_allclose(M,C.D(n),atol=1e-14)
    def test_polar_preserves_axial_and_other_tangent(self):
        p=.9;n=np.array([math.cos(p),math.sin(p),0.]);Q=np.eye(3);mods,_,_=representations({'Q':Q},{'Q':Q,'n':n},.02)
        M=mods['polar/metric'][0];Mi=mods['polar/inverse_metric'][0]
        np.testing.assert_allclose(M@n,n,atol=1e-14)
        y=np.cross([0,0,-1],n);np.testing.assert_allclose(M@y,.8*math.cos(p)*y,atol=1e-14)
        np.testing.assert_allclose(Mi@y,.8/math.cos(p)*y,atol=1e-14)
    def test_geometry_sign(self):
        w=geometry([10,0,0],[0,2,0],[0,0,0],[0,0,0]);np.testing.assert_allclose(w,[0,0,-.2])
        np.testing.assert_allclose(np.cross([1,0,0],w),[0,.2,0])
    def test_eight_mixtures_and_input_isolation(self):
        x=dict(Ps=[100.,3,7],Vs=[10.,2,1],p=[0,0,0],vm=[20,0,0],Q=native_matrix([.1,.2,.3]));y=dict(n=np.array([1.,0,0]),Q=native_matrix([.2,.3,.4]),p=[1,0,0],vm=[21,0,0],R=99.)
        for builder in (mixtures,representations):
            a,_,_=builder(x,y,.04);b,_,_=builder(x,dict(y,Ps=np.full(3,np.nan),Vs=np.full(3,np.nan),w=np.full(3,np.nan)),.04)
            self.assertEqual(len(a),8 if builder is mixtures else 6)
            for key in a:
                for u,v in zip(a[key],b[key]):np.testing.assert_array_equal(u,v)

def run(root,out,family):
    out.mkdir(parents=True,exist_ok=False);checked=C.verify(root);source=root/'prior/evidence/data/main/evidence/capture_events.json.gz';events=C.read(source)
    builder,weight=(mixtures,.2) if family=='mixture' else (representations,1.)
    result,rows,_=C.evaluate_family(events,builder,family,weight)
    result['provenance']=dict(input_manifest_verified=checked,capture_sha256=C.digest(source),code_sha256=C.digest(Path(__file__)),shared_code_sha256=C.digest(Path(C.__file__)),python=sys.version,numpy=np.__version__)
    C.save(out/'summary.json',result);C.save(out/'rows.json.gz',rows)
    compact=dict(family=family,counts=result['counts'],domain_fallbacks=result['domain_fallback_counts'],selected_scores={g:{k:v for k,v in ss.items() if k in ('baseline','selected_LOSO')} for g,ss in result['groups'].items()},six_selected=[{k:r[k] for k in ('replay','uid','tick0','tick1','selected','selected_weight','auxiliary_inputs')}|{'scores':{k:r['scores'][k] for k in ('baseline','selected_LOSO')}} for r in result['six_cases']],R77_folds=[r for r in result['folds'] if r['target_family']=='R-77-1'],all_models_near_R77=result['groups']['near_R77_79'])
    C.save(out/'compact.json',compact)
    for key,v in compact.items():print('OTHER_RESULT',key,json.dumps(C.serial(v),ensure_ascii=False),flush=True)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='tests':unittest.main(argv=[sys.argv[0]],verbosity=2)
    else:
        p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--family',choices=['mixture','representation'],required=True);a=p.parse_args();run(a.input,a.out,a.family)
