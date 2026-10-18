import unittest
import numpy as np
from scipy.spatial.transform import Rotation
import range_check as R

class Checks(unittest.TestCase):
    def test_positive_scale_recovers_known_change(self):
        for M in (np.eye(3),np.diag([.8,1,.8])):
            k=np.array([100.,-70.,40.]);y=np.trunc(1.4*M@k)
            d=R.scaling_diagnostic(M,k,y)
            self.assertTrue(d['feasible']);self.assertLessEqual(d['rho_min'],1.4);self.assertGreaterEqual(d['rho_max'],1.4)
            self.assertGreater(R.slack(M,np.zeros(3),k,y),1.)
    def test_length_cannot_fix_opposite_direction(self):
        self.assertFalse(R.scaling_diagnostic(np.eye(3),[20,30,40],[-20,-30,-40])['feasible'])
    def test_baseline_implies_scaling_feasibility(self):
        rng=np.random.default_rng(1806)
        for _ in range(40):
            n=rng.normal(size=3);n/=np.linalg.norm(n);M=R.D(n)@Rotation.random(random_state=rng).as_matrix()
            x=rng.uniform(-100,100,3);k0=np.trunc(x);k1=np.trunc(M@x)
            self.assertLess(R.slack(M,np.zeros(3),k0,k1),R.TOL)
            d=R.scaling_diagnostic(M,k0,k1)
            self.assertTrue(d['feasible']);self.assertLessEqual(d['rho_min'],1+1e-8)
            if d['rho_max'] is not None:self.assertGreaterEqual(d['rho_max'],1-1e-8)
    def test_ratio_identity_and_transverse_metric(self):
        n=np.array([1.,0,0]);old=dict(R=100.,Ps=[100.,0,0],p=[0.,0,0],Vs=[0.,0,0]);new=dict(R=100.,n=n,Q=np.eye(3),p=[0.,0,0])
        c,_,_=R.range_candidates(old,new,1/48)
        self.assertEqual(len(c),16)
        for M,b in c.values():np.testing.assert_allclose(M,R.D(n),atol=1e-15);np.testing.assert_array_equal(b,np.zeros(3))
        new['R']=50.;c,_,_=R.range_candidates(old,new,1/48)
        np.testing.assert_allclose(c['cache_new/ratio/transverse'][0]@n,n)
        np.testing.assert_allclose(c['cache_new/ratio/whole'][0]@n,2*n)
    def test_domain_fallback_is_explicit(self):
        old=dict(R=0.,Ps=[0.,0,0],p=[0.,0,0],Vs=[0.,0,0]);new=dict(R=0.,n=np.array([1.,0,0]),Q=np.eye(3),p=[0.,0,0])
        candidates,bad,_=R.range_candidates(old,new,1/48)
        self.assertTrue(all(bad.values()))
        for M,b in candidates.values():np.testing.assert_array_equal(M,R.D(new['n']))
    def test_builder_does_not_require_current_target_answers(self):
        old=dict(R=100.,Ps=[100.,2,3],p=[0.,0,0],Vs=[1.,2,3]);new=dict(R=98.,n=np.array([1.,0,0]),Q=np.eye(3),p=[1.,0,0])
        c,_,_=R.range_candidates(old,new,2/48)
        poison=dict(new,Ps=np.full(3,np.nan),Vs=np.full(3,np.nan),w=np.full(3,np.nan))
        d,_,_=R.range_candidates(old,poison,2/48)
        for k in c:
            for a,b in zip(c[k],d[k]):np.testing.assert_array_equal(a,b)

if __name__=='__main__':unittest.main(verbosity=2)
