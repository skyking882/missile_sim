"""Focused pure-Python checks of the explicit PD illuminator contract."""
import math
import unittest

from aim120_model.reconstructed.sarh import SemiActiveSeeker
from aim120_model.reconstructed.types import BodyState, IlluminatorState, SceneSnapshot


def rocket(distance=False):
    # Small explicit scene/resource fixture, not defaults borrowed from a missile.
    return {'guidance': {'radarSeeker': {
        'filterAlpha': .85, 'filterBetta': .2, 'trackResetTimeMax': .2,
        'angleMax': 60., 'lockAngleMax': 35., 'rateMax': 20., 'angleGateRate': 30.,
        'prolongationTimeMax': 1.,
        'transmitter': {'power': 200., 'antenna': {'angleHalfSens': 10., 'sideLobesSensitivity': -30.}},
        'receiver': {'rcs': 2., 'range': 40000., 'rangeMax': 70000.,
                     'antenna': {'angleHalfSens': 13., 'sideLobesSensitivity': -17.}},
        'distance': {'presents': distance, 'minValue': 0., 'maxValue': 100000.,
                     'width': 1000., 'signalWidthMin': 10.},
        'dopplerSpeed': {'presents': True, 'minValue': -3000., 'maxValue': 3000.,
                         'width': 100., 'signalWidthMin': 5.},
        'dopplerSpeedGate': {'filterAlpha': .25, 'filterBetta': .03,
                            'rateLimit': [100., 250.], 'limitTimeOut': 1.}
    }}}


def scene(t=0., *, illuminating=True, target=None, tx=(0., 0., 10000.),
          tv=(100., 0., 0.), identity='target', waveform='pd', stamp=None):
    target = target or BodyState((10000., 0., 0.), (-200., 0., 0.))
    direction = tuple(x-y for x, y in zip(target.p, tx))
    return SceneSnapshot(target, IlluminatorState(tx, tv, direction, illuminating,
                         identity, waveform, t if stamp is None else stamp), target_rcs_m2=2.)


MISSILE = BodyState((0., 0., 0.), (800., 0., 0.), identity='missile')


class SemiActiveTests(unittest.TestCase):
    def test_real_bistatic_geometry_and_unobservable_receiver_closure(self):
        m = SemiActiveSeeker(rocket()).update(0., .1, MISSILE, scene())
        self.assertTrue(m.fresh)
        self.assertAlmostEqual(m.metadata['tx_range_m'], math.sqrt(2)*10000)
        expected = .5*(300/math.sqrt(2)+1000)
        self.assertAlmostEqual(m.metadata['bistatic_closing_mps'], expected)
        self.assertNotEqual(m.metadata['bistatic_closing_mps'], 1000.)
        self.assertTrue(m.metadata['bistatic_doppler_valid'])
        self.assertFalse(m.range_valid)
        self.assertFalse(m.doppler_valid)
        self.assertIsNone(m.range_m)
        self.assertIsNone(m.closing_mps)

    def test_range_and_receiver_closure_when_observable(self):
        m = SemiActiveSeeker(rocket(True)).update(0., .1, MISSILE, scene())
        self.assertTrue(m.range_valid)
        self.assertTrue(m.doppler_valid)
        self.assertAlmostEqual(m.range_m, 10000.)
        self.assertAlmostEqual(m.closing_mps, 1000.)

    def test_hold_does_not_follow_truth_and_expires_then_reacquires(self):
        seeker = SemiActiveSeeker(rocket())
        first = seeker.update(0., .1, MISSILE, scene())
        hidden = BodyState((0., 30000., 30000.), (1900., -1800., 400.))
        held = seeker.update(.1, .1, MISSILE, scene(.1, illuminating=False, target=hidden))
        self.assertTrue(held.valid and held.held)
        self.assertFalse(held.fresh)
        self.assertEqual(held.direction, first.direction)
        self.assertEqual(held.angular_rate, first.angular_rate)
        lost = seeker.update(1., .9, MISSILE, scene(1., illuminating=False, target=hidden))
        self.assertFalse(lost.valid)
        self.assertEqual(lost.mode, 'search')
        reacquired = seeker.update(1.1, .1, MISSILE, scene(1.1))
        self.assertTrue(reacquired.valid and reacquired.fresh)
        self.assertFalse(reacquired.held)

    def test_absent_stale_and_wrong_association_do_not_acquire(self):
        target = scene().target
        missing = SceneSnapshot(target, target_rcs_m2=2.)
        cases = [missing, scene(1., stamp=0.), scene(1., identity='other')]
        for s in cases:
            m = SemiActiveSeeker(rocket()).update(1., .1, MISSILE, s)
            self.assertFalse(m.valid)
            self.assertFalse(m.fresh)

    def test_angular_rate_comes_from_measurements_not_target_velocity(self):
        a, b = SemiActiveSeeker(rocket()), SemiActiveSeeker(rocket())
        for seeker in (a, b): seeker.update(0., .1, MISSILE, scene())
        # Same measured direction and radial Doppler; different transverse V.
        # Tx is co-located, so transverse target speed is Doppler-unobservable.
        target_a = BodyState((10000., 10., 0.), (-200., 0., 0.))
        target_b = BodyState((10000., 10., 0.), (-200.5, 500., 0.))
        ma = a.update(.1, .1, MISSILE, scene(.1, target=target_a, tx=(0., 0., 0.)))
        mb = b.update(.1, .1, MISSILE, scene(.1, target=target_b, tx=(0., 0., 0.)))
        self.assertTrue(ma.fresh and mb.fresh)
        self.assertEqual(ma.direction, mb.direction)
        self.assertEqual(ma.angular_rate, mb.angular_rate)
        self.assertLess(ma.angular_rate[2], 0.)

    def test_raw_power_receiver_fov_and_doppler_bounds_gate(self):
        weak = rocket()
        weak['guidance']['radarSeeker']['transmitter']['power'] = 0.
        self.assertFalse(SemiActiveSeeker(weak).update(0., .1, MISSILE, scene()).valid)
        outside = BodyState((100., 20000., 0.), (-200., 0., 0.))
        self.assertEqual(SemiActiveSeeker(rocket()).update(0., .1, MISSILE,
                         scene(target=outside)).metadata['reason'], 'receiver_fov')
        too_fast = BodyState((10000., 0., 0.), (-10000., 0., 0.))
        self.assertFalse(SemiActiveSeeker(rocket()).update(0., .1, MISSILE,
                         scene(target=too_fast)).valid)

    def test_active_false_or_null_and_missing_semiactive_use_provider(self):
        for active in (False, None):
            r = rocket()
            r['guidance']['radarSeeker']['active'] = active
            m = SemiActiveSeeker(r).update(0., .1, MISSILE, scene())
            self.assertTrue(m.valid)
            self.assertEqual(m.metadata['branch'], 'semiActive_pd')

    def test_unsupported_modes_and_missing_defaults_fail_clearly(self):
        for key, value in [('active', True), ('semiActive', False), ('useTargetId', True),
                           ('absDopplerSpeed', True), ('zeroDopplerNotchWidth', 50.)]:
            r = rocket()
            r['guidance']['radarSeeker'][key] = value
            with self.assertRaises((ValueError, NotImplementedError)):
                SemiActiveSeeker(r)
        with self.assertRaises(NotImplementedError):
            SemiActiveSeeker(rocket()).update(0., .1, MISSILE, scene(waveform='cw'))
        r = rocket()
        del r['guidance']['radarSeeker']['filterAlpha']
        with self.assertRaisesRegex(ValueError, 'filterAlpha'):
            SemiActiveSeeker(r)
        self.assertTrue(SemiActiveSeeker(r, defaults={'filterAlpha': .85}).update(0., .1, MISSILE, scene()).fresh)


if __name__ == '__main__':
    unittest.main()
