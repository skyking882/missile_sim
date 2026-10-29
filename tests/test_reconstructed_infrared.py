"""Causal IR branch checks with explicitly declared synthetic source fixtures."""
import math

import unittest

from aim120_model.reconstructed.infrared import InfraredSeeker, emission_profile
from aim120_model.reconstructed.types import BodyState, SceneSnapshot, SignalSource


MISSILE = BodyState((0., 0., 0.), (300., 0., 0.))


def seeker(**values):
    # Wide synthetic optics make these state transitions inspectable. Native
    # defaults are exercised separately; no real weapon performance is fitted.
    optical = dict(fov=20., angleMax=40., lockAngleMax=20., rateMax=100.,
                   rangeBand0=1000., rangeBand1=1000., rangeBand2=1000.,
                   rangeMax=10000., prolongationTimeMax=.2,
                   filterAlpha=1., filterBetta=1.)
    optical.update(values)
    return InfraredSeeker({'guidance': {'opticalSeeker': optical}})


def scene(angle=0., emission=None, flares=(), present=True, velocity=(0., 0., 0.)):
    a = math.radians(angle)
    target = BodyState((1000*math.cos(a), 0., 1000*math.sin(a)), velocity,
                       present=present)
    return SceneSnapshot(target, target_ir={0: 10.} if emission is None else emission,
                         flares=tuple(flares))


def flare(angle=1., strength=100., band=2):
    a = math.radians(angle)
    return SignalSource('flare', (1000*math.cos(a), 0., 1000*math.sin(a)),
                        (99999., 99999., 99999.), {band: strength})


class TestReconstructedInfrared(unittest.TestCase):
    def test_measured_angular_rate_has_no_truth_velocity_or_range_channel(self):
        a, b = seeker(), seeker()
        for t, angle in [(0., 0.), (.01, .1), (.02, .2)]:
            first = a.update(t, .01, MISSILE, scene(angle))
            second = b.update(t, .01, MISSILE, scene(angle, velocity=(1e9, -1e9, 1e9)))
            assert first.direction == second.direction
            assert first.angular_rate == second.angular_rate
        assert first.valid and first.fresh
        assert first.angular_rate[1] > 0  # native vrel x r sign, observed increasing Z
        assert first.range_m is None and not first.range_valid
        assert first.closing_mps is None and not first.doppler_valid


    def test_absent_or_dark_source_cannot_lock_from_geometry(self):
        for observed in (scene(emission={}), scene(present=False)):
            result = seeker().update(0., .01, MISSILE, observed)
            assert not result.valid and result.mode == 'search'
            assert not result.fresh


    def test_max_band_not_sum_and_centroid_not_strongest_entity(self):
        ir = seeker()
        weak = ir.update(0., .01, MISSILE, scene(emission={0: .6, 1: .6}))
        self.assertAlmostEqual(not weak.valid and weak.metadata['gate_signal'], .6)
        result = ir.update(.01, .01, MISSILE, scene(emission={0: 4., 1: 3.},
                                                  flares=[flare(4., 2.)]))
        assert result.valid
        angle = math.degrees(math.atan2(result.direction[2], result.direction[0]))
        assert 0 < angle < 4
        self.assertAlmostEqual(result.metadata['signal'], 6.)
        assert result.metadata['selected_bands'] == {'target': 0, 'flare': 2}


    def test_rejection_uses_selected_band_instead_of_flare_entity_type(self):
        ir = seeker(bandMaskToReject=4, rejectionReactionTime=0.)
        # Band 0 wins on this flare even though it also emits band 2.
        source = SignalSource('mixed_flare', (1000., 0., 0.), (0., 0., 0.),
                              {0: 20., 2: 10.})
        measured = ir.update(0., .01, MISSILE, scene(flares=[source]))
        assert measured.valid and measured.fresh
        assert measured.metadata['rejected_signal'] == 0.
        assert measured.metadata['selected_bands']['mixed_flare'] == 0


    def test_irccm_strict_reaction_timer_predicts_then_recovers_immediately(self):
        ir = seeker(bandMaskToReject=4, signalRelRejectedTreshold=.1,
                    rejectionReactionTime=.02)
        ir.update(0., .01, MISSILE, scene())
        moving = ir.update(.01, .01, MISSILE, scene(.1))
        assert moving.angular_rate[1] > 0
        flare_scene = scene(.2, flares=[flare(.3)])
        one = ir.update(.02, .01, MISSILE, flare_scene)
        two = ir.update(.03, .01, MISSILE, flare_scene)
        assert one.fresh and two.fresh  # equality is accepted by native comparison
        suspended = ir.update(.04, .01, MISSILE, flare_scene)
        assert suspended.valid and suspended.held and not suspended.fresh
        assert suspended.mode == 'suspended'
        old_direction = suspended.direction
        next_hold = ir.update(.05, .01, MISSILE, flare_scene)
        assert next_hold.direction != old_direction
        assert next_hold.angular_rate == suspended.angular_rate
        recovered = ir.update(.06, .01, MISSILE, scene(.6))
        assert recovered.fresh and not recovered.held
        assert recovered.metadata['rejection_age_s'] == 0.


    def test_gate_outside_flare_is_retained_but_only_selected_set_rejects(self):
        ir = seeker(gateWidth=.5, bandMaskToReject=4,
                    signalRelRejectedTreshold=.1, rejectionReactionTime=0.)
        ir.update(0., .01, MISSILE, scene())
        clean = ir.update(.01, .01, MISSILE, scene(flares=[flare()]))
        assert clean.fresh
        self.assertAlmostEqual(clean.metadata['outside_signal'], 100.)
        assert clean.metadata['selected_set'] == 'gate'
        assert clean.metadata['rejected_signal'] == 0.
        rejected = ir.update(.02, .01, MISSILE,
                             scene(emission={0: .5}, flares=[flare()]))
        assert rejected.mode == 'suspended'
        assert rejected.metadata['selected_set'] == 'outside'
        self.assertAlmostEqual(rejected.metadata['rejected_signal'], 100.)
        assert rejected.metadata['gate_sine'] == clean.metadata['gate_sine']


    def test_nonzero_gate_transparency_can_cause_irccm_suspension(self):
        ir = seeker(gateWidth=.5, outGateTransparency=.05,
                    bandMaskToReject=4, signalRelRejectedTreshold=.1,
                    rejectionReactionTime=0.)
        ir.update(0., .01, MISSILE, scene())
        measured = ir.update(.01, .01, MISSILE, scene(flares=[flare()]))
        assert measured.mode == 'suspended'
        self.assertAlmostEqual(measured.metadata['gate_signal'], 15.)
        self.assertAlmostEqual(measured.metadata['rejected_signal'], 5.)


    def test_fov_loss_hold_expiry_and_reacquisition(self):
        ir = seeker(gateWidth=.5)
        ir.update(0., .01, MISSILE, scene())
        held = ir.update(.1, .1, MISSILE, scene(30.))
        assert held.held and held.valid and not held.fresh
        lost = ir.update(.21, .11, MISSILE, scene(30.))
        assert not lost.valid and lost.mode == 'lost'
        self.assertAlmostEqual(lost.metadata['gate_sine'], math.sin(math.radians(10)))
        reacquired = ir.update(.22, .01, MISSILE, scene())
        assert reacquired.valid and reacquired.fresh and reacquired.mode == 'track'


    def test_body_orientation_controls_search_boresight_and_gimbal(self):
        # XYZW rotation about +Y points body X toward world -Z.
        q = (0., math.sin(math.pi/4), 0., math.cos(math.pi/4))
        missile = BodyState(MISSILE.position, MISSILE.velocity, q)
        ir = seeker()
        blocked = ir.update(0., .01, missile, scene())
        assert not blocked.valid
        found = ir.update(.01, .01, missile, scene(-90.))
        self.assertAlmostEqual(found.valid and found.direction[2], -1.)


    def test_rate_limit_and_angular_gate_reject_without_truth_fallback(self):
        ir = seeker(rateMax=1., angleGateRate=.5)
        ir.update(0., .01, MISSILE, scene())
        result = ir.update(.01, .01, MISSILE, scene(1.))
        assert result.held and not result.fresh
        assert result.metadata['reason'] == 'angular_gate'
        assert result.angular_rate == (0., 0., 0.)
        bounded = seeker(rateMax=1.)
        bounded.update(0., .01, MISSILE, scene())
        limited = bounded.update(.01, .01, MISSILE, scene(1.))
        assert 0 < math.degrees(math.atan2(limited.direction[2], limited.direction[0])) <= .010001


    def test_native_defaults_and_explicit_unsupported_branches(self):
        ir = InfraredSeeker({'guidance': {'opticalSeeker': {}}})
        assert ir.p['gateWidth'] == 90.
        assert ir.p['signalRelRejectedTreshold'] == .001
        assert ir.p['rejectionReactionTime'] == 0.
        self.assertAlmostEqual(ir.p['filterAlpha'], .85)
        self.assertAlmostEqual(ir.p['filterBetta'], .2)
        assert ir.p['trackResetTimeMax'] == 1000.
        for unsupported in ({'targetsResolution': True}, {'scanHalfSector': 30.},
                            {'mandatoryDesignationSources': True},
                            {'targetSignatureType': 'visual'}):
            with self.assertRaises(NotImplementedError): seeker(**unsupported)

    def test_loss_restores_acquisition_cone_and_threshold_equality_clears_timer(self):
        ir = seeker(fov=40., lockAngleMax=5., prolongationTimeMax=.01,
                    bandMaskToReject=4, signalRelRejectedTreshold=.5,
                    rejectionReactionTime=0.)
        measured = ir.update(0., .01, MISSILE,
                             scene(emission={0: 10.}, flares=[flare(0., 10.)]))
        assert measured.fresh  # R == threshold*S does not accumulate rejection.
        assert measured.metadata['rejection_age_s'] == 0.
        lost = ir.update(.02, .02, MISSILE, scene(emission={}))
        assert not lost.valid
        out_of_acquire = ir.update(.03, .01, MISSILE, scene(10.))
        assert not out_of_acquire.valid  # within track gimbal, outside lock cone


    def test_recovered_flare_envelope_rise_plateau_fall_and_boundaries(self):
        times = (0., .4, 4., 4.4)
        self.assertAlmostEqual(emission_profile(.2, times), .5)
        assert emission_profile(2., times) == 1.
        self.assertAlmostEqual(emission_profile(4.2, times), .5)
        assert emission_profile(4.4, times) == 0.
        assert emission_profile(4.5, times) == 0.


    def test_equal_time_is_idempotent_and_time_cannot_go_backwards(self):
        ir = seeker()
        first = ir.update(1., .01, MISSILE, scene())
        assert ir.update(1., .01, MISSILE, scene(emission={})) is first
        with self.assertRaises(ValueError): ir.update(.9, .01, MISSILE, scene())


if __name__ == "__main__":
    unittest.main()
