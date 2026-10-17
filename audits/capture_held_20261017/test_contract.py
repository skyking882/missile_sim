"""Mathematical/interface checks, not replay fitting or independent-agent review."""
import copy,math,unittest
import numpy as np
from scipy.spatial.transform import Rotation
import audit as A

class ContractTests(unittest.TestCase):
    def test_native_bank_heading_pitch(self):
        rng=np.random.default_rng(731)
        for bank,heading,pitch in rng.uniform(-2,2,(200,3)):
            q=Rotation.from_rotvec([0,heading,0]).as_matrix()@Rotation.from_rotvec([0,0,pitch]).as_matrix()@Rotation.from_rotvec([bank,0,0]).as_matrix()
            np.testing.assert_allclose(A.matrix(bank,heading,pitch),q,rtol=0,atol=1e-14)
    def test_rotation_derivative_and_semigroup(self):
        n=A.unit([.8,.3,-.4]);w=np.array([.04,-.08,.13]);dt=1e-6
        np.testing.assert_allclose((A.rotate(n,w,dt)-A.rotate(n,w,-dt))/(2*dt),np.cross(n,w),atol=1e-9,rtol=0)
        np.testing.assert_allclose(A.rotate(A.rotate(n,w,.12),w,.37),A.rotate(n,w,.49),atol=1e-14,rtol=0)
    def test_spherical_correction_sign(self):
        p=A.unit([1,.2,.3]);m=A.unit([.6,.5,.7])
        np.testing.assert_allclose(A.blend(p,m,1),m,atol=1e-14,rtol=0)
        np.testing.assert_allclose(A.blend(p,m,0),p,atol=1e-14,rtol=0)
        self.assertAlmostEqual(A.angle(p,A.blend(p,m,.37)),.37*A.angle(p,m),places=13)
    def test_ray_inverse(self):
        for yaw,pitch in [(.2,.3),(-.8,.7),(2.9,-.1),(-2.9,-.6)]:
            np.testing.assert_allclose(A.angles(A.ray([yaw,pitch])),[yaw,pitch],atol=1e-14,rtol=0)
    def test_current_answer_fields_not_read_by_capture_prediction(self):
        a=dict(Ps=np.array([1500.,800.,300.]),Vs=np.array([100.,-10.,20.]),p=np.zeros(3),vm=np.array([350.,10.,3.]),w=np.array([.04,-.02,.06]),Q=A.matrix(.2,.3,.1),n=A.unit([1,.3,.2]),tick=10)
        b=dict(p=np.array([10.,1.,.2]),Q=A.matrix(.23,.31,.09),m=A.unit([1,.4,.3]),tick=12,n=A.unit([1,.3,.3]),w=np.ones(3),Ps=np.ones(3),Vs=np.ones(3),R=1600.)
        expected=A.capture_models(a,b,2/48)
        poisoned=copy.deepcopy(b)
        for key in ('n','w','Ps','Vs'):poisoned[key]=np.full(3,np.nan)
        actual=A.capture_models(a,poisoned,2/48)
        self.assertEqual(len(expected),36)
        for key in expected:
            for v0,v1 in zip(expected[key],actual[key]):np.testing.assert_array_equal(v0,v1)
    def test_axial_rate_does_not_move_parallel_direction(self):
        n=A.unit([.2,.3,.7]);w=.4*n
        for T in (1/48,2/48,.7):
            nn,ww=A.step(n,w,n,T)
            np.testing.assert_allclose(nn,n,rtol=0,atol=1e-14)
            np.testing.assert_allclose(ww,w,rtol=0,atol=1e-14)

if __name__=='__main__':unittest.main(verbosity=2)
