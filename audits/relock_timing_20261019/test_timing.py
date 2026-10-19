import sys,unittest
from pathlib import Path
import numpy as np
import timing_bounds as T
from native_paths import recovery_path

META=dict(replay='synthetic_contract',eid=1,uid=1,family='R-77-1')
def row(t,code=31,timer=0.,flag=0):
    return dict(tick=t,index=t,packet_ms=round(1000*t/48),code=code,timer=timer,flag=flag,key=7,raw_sha=str(t),Ps=np.array([1.,2.,3.]),Vs=np.array([3.,4.,5.]),w=np.array([.1,.2,.3]),n=np.array([1.,0.,0.]))

class TimingTests(unittest.TestCase):
    def test_phase_all_offsets(self):
        for start in range(-5,100):
            for gap in (1,2,3,8):
                for phase in range(48):
                    self.assertEqual(T.phase_in(start,start+gap,phase),any(k%48==phase for k in range(start+1,start+gap+1)))
    def test_interval_not_endpoint_phase(self):
        self.assertTrue(T.phase_in(48,51));self.assertNotEqual(51%48,1)
        self.assertFalse(T.phase_in(55,57));self.assertFalse(T.phase_in(55,57,padding=2))
    def test_positive_timer_interval(self):
        x,y=row(100,timer=.2),row(102)
        b=T.bound(x,y);self.assertAlmostEqual(b['observation_gap_s'],.2+2/48)
        self.assertGreater(b['observation_gap_upper_with_2tick_slack_s'],b['observation_gap_upper_s'])
    def test_whole_held_episode(self):
        a=row(100);held=[row(t,timer=(t-100)/48) for t in (102,104,106)]
        p=recovery_path(META,[a]+held,row(108))
        self.assertTrue(p['coherent_sustained']);self.assertTrue(p['all_held_timer_continuous']);self.assertEqual(p['held_rows'],3)
        self.assertLess(p['waiting_since_loss_upper_s'],1)
    def test_hidden_age_reset_not_continuous(self):
        p=recovery_path(META,[row(100),row(104,timer=4/48),row(108,timer=1/48)],row(110))
        self.assertFalse(p['all_held_timer_continuous']);self.assertFalse(p['sustained'])
    def test_intermediate_code_not_named_held(self):
        p=recovery_path(META,[row(100),row(104,timer=4/48),row(106,29,timer=6/48)],row(108))
        self.assertEqual(p['path_kind'],'via29');self.assertFalse(p['sustained'])
    def test_ins_path_separated(self):
        p=recovery_path(META,[row(100),row(148,29,1),row(150,25,1.04)],row(193))
        self.assertEqual(p['path_kind'],'via25')
        self.assertTrue(p['phase1_in_transition'])
    def test_timer_boundary_uncertain(self):
        b=T.bound(row(100,timer=1-2/48),row(102))
        self.assertLess(b['observation_gap_lower_s'],1);self.assertGreater(b['observation_gap_upper_s'],1)
    def test_flags_do_not_define_target_identity(self):
        p=recovery_path(META,[row(100),row(106,timer=6/48,flag=1)],row(108,flag=1))
        self.assertFalse(p['flags_all_zero']);self.assertTrue(p['stable_designation_key']);self.assertNotIn('same_target_proven',p)
if __name__=='__main__':unittest.main(verbosity=2)
