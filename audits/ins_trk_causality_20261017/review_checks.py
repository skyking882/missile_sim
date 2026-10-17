#!/usr/bin/env python3
"""Independent algebra/interface checks and bounded review of saved results."""
import argparse,copy,json,math,unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import explore as E
import followup as F
A=E.A

class GuardedCurrent(dict):
    def __getitem__(self,key):
        if key in ('Ps','Vs','n','w'):raise AssertionError('Current response accessed: '+key)
        return super().__getitem__(key)

class Tests(unittest.TestCase):
    def test_passive_rotation_preserves_axial_component(self):
        rng=np.random.default_rng(741)
        for _ in range(200):
            n=A.unit(rng.normal(size=3));w=rng.normal(size=3);q=Rotation.random(random_state=rng).as_matrix()
            self.assertAlmostEqual(float(n@w),float((q@n)@(q@w)),places=12)
    def test_rotating_rate_alone_is_not_a_passive_coordinate_change(self):
        n=np.array([1.,0,0]);w=np.array([0.,1,0]);q=Rotation.from_rotvec([0,0,-math.pi/2]).as_matrix()
        self.assertAlmostEqual(float(n@w),0.);self.assertAlmostEqual(float(n@(q@w)),1.)
    def test_contraction_matrix(self):
        rng=np.random.default_rng(713)
        for _ in range(200):
            n=A.unit(rng.normal(size=3));s=rng.normal(size=3);k=.8
            w=(k*np.eye(3)+(1-k)*np.outer(n,n))@s
            self.assertAlmostEqual(float(w@n),float(s@n),places=12)
            np.testing.assert_allclose(w-(w@n)*n,k*(s-(s@n)*n),atol=1e-14,rtol=0)
    def test_effective_alpha_linear_implicit_identity(self):
        for alpha in (.5,.8,.876,.926):
            for x in (.1,.3,.42):
                e=np.array([.001,-.003]);delta=np.zeros(2)
                for _ in range(120):delta=alpha*e+(1-alpha)*x*delta
                np.testing.assert_allclose(delta,alpha/(1-(1-alpha)*x)*e,atol=1e-14,rtol=0)
    def test_capture_past_only_rejects_current_label_access(self):
        a=dict(tick=47,Ps=np.array([10000.,3000.,2000.]),Vs=np.array([100.,20.,0.]),p=np.zeros(3),vm=np.array([400.,0.,0.]),n=A.unit([1,.3,.2]),w=np.array([.01,-.03,.05]),Q=A.matrix(.1,.2,.3))
        b=GuardedCurrent(tick=51,p=np.array([30.,1.,0.]),vm=np.array([420.,0.,0.]),Q=A.matrix(.15,.22,.31),angles=np.array([-.2,.1]))
        for q in ('Qold','Qphase'):
            for init in ('reset','self_h'):E.capture_predict(a,b,49,'past_PV',q,init,.926)
            F.contracted_capture(a,b,49,'past_PV',q,.8)
        with self.assertRaises(AssertionError):E.capture_predict(a,b,49,'endpoint_PV_DIAGNOSTIC','Qold','reset',.926)
    def test_pointing_and_past_projection_do_not_access_current_labels(self):
        a=dict(tick=47,Ps=np.array([10000.,3000.,2000.]),Vs=np.array([100.,20.,0.]),p=np.zeros(3),vm=np.array([400.,0.,0.]),n=A.unit([1,.3,.2]),w=np.array([.01,-.03,.05]),Q=A.matrix(.1,.2,.3))
        b=GuardedCurrent(tick=49,p=np.array([15.,1.,0.]),vm=np.array([410.,0.,0.]),Q=A.matrix(.11,.21,.3),angles=np.array([-.2,.1]))
        for src in ('stored','old_geometry','geometry_at_endpoint'):
            for q in ('Qold','Qnew'):
                for ns in ('past_PV','pointing_CONDITIONAL'):F.reset_predict(a,b,.8,src,q,ns)
        with self.assertRaises(AssertionError):F.reset_predict(a,b,.8,'stored','Qnew','current_n_DIAGNOSTIC')
    def test_implicit_solver_with_only_declared_inputs(self):
        n=np.array([[1.,0,0]]);z=np.zeros((1,3));inp=dict(P0=np.array([[1000.,0,0]]),n0=n,w0=z,p=z,vm=z,R=np.array([1000.]),d=np.array([0.]),T=np.array([1/48]),m=n)
        P,nn,w,V,res=F.solve(inp,np.array([.876]))
        np.testing.assert_array_equal(P,inp['P0']);np.testing.assert_array_equal(nn,n);np.testing.assert_array_equal(w,z);np.testing.assert_array_equal(V,z);self.assertEqual(float(res[0]),0.)

def main():
    p=argparse.ArgumentParser();p.add_argument('--main',type=Path,required=True);p.add_argument('--reset',type=Path,required=True);p.add_argument('--implicit',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests);r=unittest.TextTestRunner(verbosity=2).run(suite)
    if not r.wasSuccessful():raise AssertionError('Algebra/interface tests failed')
    reset=E.load(next(args.reset.rglob('reset_events.json.gz')));caps=E.load(args.main/'evidence/capture_events.json.gz');idx={(r['replay'],r['eid'],r['age1']):r for r in caps}
    mx=0.;checked=0
    for row in reset:
        e=idx[(row['replay'],row['eid'],row['age1'])];a,b=E.arr(e['old']),E.arr(e['new'])
        for qlabel,q in [('Qold',a['Q']),('Qnew',b['Q'])]:
            n=b['n'];v=(.8*np.eye(3)+.2*np.outer(n,n))@q@a['w'];err=np.linalg.norm(v-b['w'])*A.RAD
            recorded=row['scores'][f'stored/{qlabel}/current_n_DIAGNOSTIC/retention_0p8']['w_deg_s'];mx=max(mx,abs(err-recorded));checked+=1
    assert mx<1e-11,mx
    orig=E.load(next((args.main/'causality').rglob('causality_oof_rows.json.gz')));imp=E.load(next(args.implicit.rglob('implicit_rows.json.gz')))
    key=lambda r:(r['group'],r['replay'],r['eid'],r['tick0'],r['tick1'])
    assert [key(r) for r in orig]==[key(r) for r in imp]
    summary=E.load(next(args.reset.rglob('reset_summary.json')));isum=E.load(next(args.implicit.rglob('implicit_summary.json')));source=E.load(args.main/'input/source_verification.json')
    compact=dict(tests_run=r.testsRun,tests_passed=True,reset_matrix_cases=checked,reset_matrix_max_error_difference=mx,identical_ordered_ordinary_sample_membership=len(orig),source={k:source[k] for k in ('native_records_scanned','series','selected_rows','reference_unique_states','reference_duplicate_keys','reference_max_abs_difference')},zero_phase_events=summary['zero_phase_events'],projection_input_poison_checks=summary['past_or_pointing_label_isolation_checks'],reset_groups={},implicit=isum['groups'])
    for group,g in summary['groups'].items():
        models={model:v for model,v in g['models'].items() if model.startswith('stored/Qnew/') and model.endswith('retention_0p8')}
        compact['reset_groups'][group]=dict(n=g['n'],models=models,shared_kappa_fit=g['fitted']['stored/Qnew/current_n_DIAGNOSTIC'])
    csum=E.load(next((args.main/'capture').rglob('capture_timing_summary.json')));compact['capture_same_rule_anchors']=[]
    for uid,limit in [(118,2),(148,1)]:
        rr=[r for r in csum['anchors'] if r['uid']==uid][:limit]
        for row in rr:
            rr2=next(r for r in summary['anchors_multi'] if (r['replay'],r['eid'],r['age1'])==(row['replay'],row['eid'],row['age1']))
            compact['capture_same_rule_anchors'].append({k:row[k] for k in ('uid','age0','age1','phase','post_ticks')}|dict(initialization_then_feedback=row['scores']['past_PV/Qold/self_h/a0.926'],direct_contraction_Qphase=rr2['scores']['past_PV/Qphase/retention_0p8']))
    allimplicit=E.load(next(args.implicit.rglob('implicit_rows.json.gz')))
    compact['implicit_unconverged_retained']=[r for r in allimplicit if r['fixed_point_residual_rad']>1e-10]
    T=2/48;g=1/(TAU+T/2) if False else 1/(E.TAU+T/2)
    pars=list(isum['groups']['R77_54p9']['alpha_by_test_scene'].values())
    compact['effective_alpha_approximation']=dict(assumptions='linearized common tangent plane, same base predictor, only angular part of velocity correction; not an exact nonlinear identity',T=T,g=g,alpha_implicit=pars,alpha_effective=[a/(1-(1-a)*T*g) for a in pars])
    A.save(args.out/'independent_review.json',compact)
    for k,v in compact.items():print('REVIEW_FINAL',k,json.dumps(A.serial(v)),flush=True)
if __name__=='__main__':main()
