import unittest
from retry_clock import RetryClock

class RetryClockTests(unittest.TestCase):
    def test_immediate_first_success_at_every_phase(self):
        for phase in range(48):
            c=RetryClock();r=c.step(480+phase,enabled=True,search_ready=True,evaluate=lambda searching:True)
            self.assertTrue(r.accepted);self.assertEqual(r.reason,'initial_attempt_accepted')

    def test_failure_then_exact_global_phase(self):
        c=RetryClock();calls=[]
        def f(s):calls.append(s);return len(calls)>1
        self.assertFalse(c.step(10,enabled=True,search_ready=True,evaluate=f).accepted)
        for t in range(11,49):self.assertFalse(c.step(t,enabled=True,search_ready=True,evaluate=f).attempted)
        self.assertTrue(c.step(49,enabled=True,search_ready=True,evaluate=f).accepted)
        self.assertEqual(len(calls),2)

    def test_failed_attempt_on_phase_does_not_repeat_same_tick(self):
        c=RetryClock();calls=[]
        def f(s):calls.append(s);return False
        a=c.step(49,enabled=True,search_ready=True,evaluate=f)
        b=c.step(49,enabled=True,search_ready=True,evaluate=lambda s:True)
        self.assertEqual(a,b);self.assertEqual(c.next_search_tick,97);self.assertEqual(len(calls),1)

    def test_no_early_visibility_cache(self):
        c=RetryClock();seen=[]
        for t in range(10,50):
            def f(s,t=t):seen.append(t);return 20<=t<=30
            c.step(t,enabled=True,search_ready=True,evaluate=f)
        self.assertEqual(seen,[10,49]);self.assertFalse(c.ever_accepted)

    def test_hold_recovery_is_not_periodic(self):
        c=RetryClock();c.step(10,enabled=True,search_ready=True,evaluate=lambda s:True)
        a=c.step(11,enabled=True,search_ready=False,evaluate=lambda s:False)
        b=c.step(12,enabled=True,search_ready=False,evaluate=lambda s:not s)
        self.assertEqual(a.state,'HOLD');self.assertTrue(b.accepted);self.assertFalse(b.searching)

    def test_one_second_boundary_then_next_global_retry(self):
        c=RetryClock();c.step(11524,enabled=True,search_ready=True,evaluate=lambda s:True)
        for t in range(11525,11574):
            r=c.step(t,enabled=True,search_ready=True,evaluate=lambda s:False)
        self.assertEqual(r.state,'RETRY');self.assertEqual(r.next_search_tick,11617)
        self.assertTrue(c.step(11617,enabled=True,search_ready=True,evaluate=lambda s:True).accepted)

    def test_range_exit_does_not_create_another_immediate_attempt(self):
        c=RetryClock();c.step(10,enabled=True,search_ready=True,evaluate=lambda s:False)
        c.step(11,enabled=True,search_ready=False,evaluate=lambda s:True)
        r=c.step(12,enabled=True,search_ready=True,evaluate=lambda s:True)
        self.assertFalse(r.attempted);self.assertEqual(r.next_search_tick,49)

    def test_not_ready_does_not_consume_first_attempt(self):
        c=RetryClock();r=c.step(10,enabled=True,search_ready=False,evaluate=lambda s:True)
        self.assertFalse(r.attempted);self.assertFalse(c.first_attempted)
        self.assertTrue(c.step(11,enabled=True,search_ready=True,evaluate=lambda s:True).accepted)

    def test_missing_scheduled_tick_is_not_replayed(self):
        c=RetryClock();c.step(10,enabled=True,search_ready=True,evaluate=lambda s:False)
        r=c.step(50,enabled=True,search_ready=True,evaluate=lambda s:True)
        self.assertFalse(r.attempted);self.assertEqual(r.next_search_tick,97)

    def test_backward_time_and_noninteger_rejected(self):
        c=RetryClock();c.step(10,enabled=True,search_ready=True,evaluate=lambda s:False)
        with self.assertRaises(ValueError):c.step(9,enabled=True,search_ready=True,evaluate=lambda s:True)
        with self.assertRaises(TypeError):c.step(10.5,enabled=True,search_ready=True,evaluate=lambda s:True)

    def test_fixed_global_epoch_not_local_age(self):
        a,b=RetryClock(),RetryClock()
        a.step(100,enabled=True,search_ready=True,evaluate=lambda s:False)
        b.step(110,enabled=True,search_ready=True,evaluate=lambda s:False)
        self.assertEqual(a.next_search_tick,b.next_search_tick)
        self.assertNotEqual(a.next_search_tick-100,b.next_search_tick-110)

    def test_disabled_then_enabled_preserves_failure_history(self):
        c=RetryClock();c.step(10,enabled=True,search_ready=True,evaluate=lambda s:False)
        c.step(11,enabled=False,search_ready=True,evaluate=lambda s:True)
        self.assertFalse(c.step(12,enabled=True,search_ready=True,evaluate=lambda s:True).attempted)

    def test_failed_retries_continue_without_fixed_dwell(self):
        c=RetryClock();attempts=[]
        for t in range(10,100):
            def f(s,t=t):attempts.append(t);return False
            c.step(t,enabled=True,search_ready=True,evaluate=f)
        self.assertEqual(attempts,[10,49,97])

    def test_independent_enumeration_of_next_phase(self):
        c=RetryClock()
        for t in range(-100,200):
            got=c.next_phase(t)
            expect=next(k for k in range(t+1,t+49) if k%48==1)
            self.assertEqual(got,expect)

if __name__=='__main__':unittest.main(verbosity=2)
