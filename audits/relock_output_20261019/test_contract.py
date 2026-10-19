import unittest
import numpy as np
import inspect_flow as F

META=dict(replay='synthetic',eid=1,uid=1,family='R-77-1',target_eid=10)
def row(t=0,code=31,timer=0):
    return dict(tick=t,index=t,packet_ms=round(t*1000/48),code=code,timer=timer,flag=0,raw_sha=str(t),n=np.array([1.,0.,0.]),w=np.zeros(3),Ps=np.array([1000.,0.,0.]),Vs=np.zeros(3),p=np.zeros(3),R=1000.)

class ContractTests(unittest.TestCase):
    def test_constructed_code_bits(self):
        for code,(a,b) in {25:(0,0),29:(1,0),31:(1,1)}.items():
            self.assertEqual(code,24+4*a+2*b+1)
            self.assertEqual((code>>2)&1,a);self.assertEqual((code>>1)&1,b)
    def test_held_is_not_a_header_change(self):
        a=row(1,31,0);b=row(2,31,1/48)
        self.assertTrue(F.fresh(a));self.assertFalse(F.fresh(b));self.assertEqual(a['code'],b['code'])
    def test_timer_zero_code29_is_not_fresh_by_definition(self):
        self.assertFalse(F.fresh(row(0,29,0)))
    def test_PV_change_can_coexist_with_rate_copy(self):
        a=row(0,31,0.99);b=row(2,29,1.03);b['Ps']=np.array([1001.,2.,0.]);b['Vs']=np.array([1.,0.,0.])
        r=F.pair(META,a,b);self.assertTrue(r['w_equal']);self.assertFalse(r['PV_equal']);self.assertAlmostEqual(r['P_change_m'],5**.5)
        self.assertEqual(r['direction_continuation_error_deg'],0)
    def test_distinct_timer_flag_not_same_as_header(self):
        b=row(1,29,1.);b['flag']=1
        self.assertEqual((b['code']>>2)&1,1);self.assertEqual((b['code']>>1)&1,0)
        self.assertFalse(F.fresh(b))
    def test_interpolation_never_uses_unbracketed_time(self):
        units=[dict(eid=10,unit_name='test',track=[dict(t_ms=1000,xyz=[1,2,3]),dict(t_ms=1100,xyz=[2,2,3])])]
        u=F.unit_series(units);v=F.lookup(u,10,1050)
        np.testing.assert_array_equal(v['position'],[1.5,2,3]);np.testing.assert_array_equal(v['velocity'],[10,0,0])
        self.assertIsNone(F.lookup(u,10,999));self.assertIsNone(F.lookup(u,10,1101))
    def test_conflicting_same_timestamp_is_not_silently_joined(self):
        units=[dict(eid=10,track=[dict(t_ms=0,xyz=[0,0,0]),dict(t_ms=0,xyz=[1,0,0]),dict(t_ms=10,xyz=[2,0,0])])]
        self.assertNotIn(10,F.unit_series(units))
    def test_missing_identity_does_not_select_nearest_target(self):
        m={k:v for k,v in META.items() if k!='target_eid'}
        self.assertEqual(F.target_compare(m,row(),row(2),row(),{}),dict(status='no_frozen_target_identity'))
    def test_end_censor_is_not_counted_as_recovery(self):
        a=row();b=row(24,31,.5)
        e=F.episode(a,[b],b,'censored_end',META,True,True,True)
        self.assertEqual(e['outcome'],'censored_end');self.assertEqual(e['elapsed_s'],.5)
if __name__=='__main__':unittest.main(verbosity=2)
