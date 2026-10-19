"""Opt-in empirical complete-rate observer for WT's active-radar game model.

Q at acquisition and the shared interval-dependent angular correction are
approximations identified from replay relationships, not recovered WT code.
All inputs here are received observations or the missile's own state.
"""
from __future__ import annotations

import math
from dataclasses import replace

from .aerodynamics import body_axes_for_state
from .angular_guidance import AngularObservation, _rotate
from .math3d import add, cross, dot, norm, scale, sub
from .tracking import TrackMode, INTENDED_TARGET, reflector_identity


def validate_angular_observer(mode):
    if mode not in (None, 'q_capture'):
        raise ValueError("angular_observer must be None or 'q_capture'")
    return mode


class QCaptureAngularObserver:
    capture_step_s = 1 / 48
    correction_tau_s = .078525
    direction_lag_s = .0031
    provenance = dict(mode='q_capture', kind='assumed', grade='D',
        capture_step_s=capture_step_s,
        rate_correction_tau_s=correction_tau_s,
        direction_lag_s=direction_lag_s,
        parameter_source='shared representative L=.0031s from ordinary-record leave-scene fits; tau=.078525s research fit',
        initialization='available INS/DL full rate (geometric if absent), current full body Q once; accepted LOS; one ordinary tick',
        update='alpha(T)=1-L/T; g(T)=1/(tau+T/2); n_dot=n x rate; no rate transport or additional .8 multiplier',
        missing_prior='zero rate', reacquisition='same-object coast keeps memory; expired capture uses Q once; not independently verified',
        radial_source='instantaneous accepted sensor relative radial speed; not decoded replay scalar d')

    def __init__(self):
        self.direction = None
        self.rate = None
        self.range_m = None
        self.closing_mps = None
        self.time_s = None
        self.sample_time_s = None
        self.identity = None
        self.memory_identity = INTENDED_TARGET
        self.prior_identity = None
        self.prior_rate = None
        self.correction_count = 0
        self.capture_count = 0
        self.capture_rate_seed = None
        self.last_correction_interval_s = None
        self.measurement_cache = None
        self.prior_solution = None
        self.seed_source = None
        self.handover_reason = None
        self.stage_callback = None

    def _clear(self):
        self.direction = self.rate = self.range_m = self.closing_mps = None
        self.time_s = self.sample_time_s = self.identity = None

    def _correct(self, measured, interval_s):
        """One spherical ordinary update, using the accepted observation interval."""
        if interval_s <= self.direction_lag_s:
            raise ValueError('angular observation interval must exceed the fitted direction lag')
        axis = cross(measured, self.direction)
        sine = norm(axis)
        cosine = max(-1., min(1., dot(measured, self.direction)))
        if sine <= 1e-12 and cosine < 0:
            raise ValueError('antipodal angular measurement has no unique innovation axis')
        alpha = 1 - self.direction_lag_s / interval_s
        correction = (scale(axis, alpha * math.atan2(sine, cosine) / sine)
                      if sine > 1e-12 else axis)
        self.direction = _rotate(self.direction, correction, -1.)
        self.rate = add(self.rate, scale(correction, 1 / (self.correction_tau_s + interval_s / 2)))
        self.last_correction_interval_s = interval_s
        self.correction_count += 1

    def _stage(self, stage, solution, reason=""):
        if self.stage_callback is not None:
            self.stage_callback(stage, solution, reason)

    def _predict(self, time_s):
        if self.time_s is None:
            return
        elapsed = time_s - self.time_s
        if elapsed < -1e-12:
            raise ValueError('angular observer time must not move backwards')
        self.direction = _rotate(self.direction, self.rate, -max(elapsed, 0.))
        self.range_m -= self.closing_mps * max(elapsed, 0.)
        self.time_s = time_s

    def refresh_prior(self, solution, missile):
        """Refresh an available INS estimate without modifying active TRK state."""
        self.prior_rate = None
        self.prior_solution = None
        if solution.valid:
            relative = sub(solution.position, missile.position)
            distance = norm(relative)
            if distance > 1e-12:
                n = scale(relative, 1 / distance)
                self.prior_rate = (solution.full_rate_rad_s if solution.full_rate_rad_s is not None
                                   else scale(cross(sub(solution.velocity, missile.velocity), n), 1 / distance))
                self.prior_identity = (solution.identity or
                    (INTENDED_TARGET if solution.mode is TrackMode.DATALINK else self.memory_identity))
                self.prior_solution = solution
        self._stage('prior_refresh', solution)

    def _refresh_prior(self, solution, missile):
        self._clear()
        self.refresh_prior(solution, missile)

    def _capture(self, measured, identity, solution, missile, changed_identity):
        """Experimental seed choice and one ordinary update are distinct stages."""
        self.rate = (0., 0., 0.)
        self.handover_reason = 'reflector_changed' if changed_identity else 'capture_or_expired_reacquisition'
        self.seed_source = 'experimental_zero_without_same_object_prior'
        if self.prior_rate is not None and self.prior_identity == identity:
            axes = body_axes_for_state(missile)
            # Runtime Q = C Q_native C. Axial parity factors cancel in Q*w.
            self.rate = add(add(scale(axes.forward, self.prior_rate[0]),
                                scale(axes.up, self.prior_rate[1])),
                            scale(axes.right, self.prior_rate[2]))
            self.seed_source = ('available_full_prior' if self.prior_solution.full_rate_rad_s is not None
                                else 'geometric_available_PV_prior')
        self.capture_rate_seed = self.rate
        self.direction = measured
        self.identity = identity
        self.time_s = solution.solution_time_s
        self._stage('capture_handover', solution, self.handover_reason)
        # A hypothesized internal ordinary tick, not another state-time advance.
        # It supplies the contraction itself; never apply a separate .8 factor.
        self.direction = _rotate(self.direction, self.rate, -self.capture_step_s)
        self._correct(measured, self.capture_step_s)
        self.capture_count += 1

    def update(self, solution, missile, *, received, identity, closing_mps):
        """Advance state once. Only an actually accepted, newer echo corrects it.

        Original sample time, state time and publication time remain separate.
        Special replay flag=1 timing is not routed through this live candidate.
        """
        time_s = solution.solution_time_s
        if solution.sample_time_s > time_s + 1e-12:
            raise ValueError('angular measurement timestamp must not be in the future')
        self._predict(time_s)
        active_mode = solution.valid and solution.mode in (TrackMode.RADAR_TRACK, TrackMode.RADAR_COAST)
        if not active_mode:
            self._refresh_prior(solution, missile)
            return solution

        identity = solution.identity or reflector_identity(identity)
        fresh = received and (self.sample_time_s is None or solution.sample_time_s > self.sample_time_s)
        changed_identity = ((self.direction is not None and identity != self.identity) or
            (self.direction is None and self.prior_rate is not None and self.prior_identity != identity) or
            (self.direction is None and self.capture_count > 0 and self.memory_identity != identity))
        if received and changed_identity:
            self._clear()
            self.prior_rate = None
            self.prior_solution = None
            fresh = True
        if fresh:
            relative = sub(solution.position, missile.position)
            distance = norm(relative)
            if distance <= 1e-12:
                self._clear()
                self.prior_rate = None
                return solution
            measured = scale(relative, 1 / distance)
            if self.direction is None:
                self._capture(measured, identity, solution, missile, changed_identity)
            else:
                self._correct(measured, solution.sample_time_s - self.sample_time_s)
            self.range_m = distance
            self.closing_mps = float(closing_mps)
            self.time_s = time_s
            self.sample_time_s = solution.sample_time_s
            self.identity = self.memory_identity = identity
            self.measurement_cache = solution
            self.prior_rate = None
            self._stage('accepted_measurement', solution)
        elif self.direction is not None:
            self._stage('held_propagation', solution, 'no_new_accepted_measurement')

        if self.direction is None or self.range_m <= 1e-12:
            self._clear()
            return solution
        angular = AngularObservation(self.direction, self.rate, self.range_m, self.closing_mps,
                                     missile.position, missile.velocity)
        track = angular.track_for(time_s, missile.position, missile.velocity,
            sample_time_s=self.sample_time_s, mode=solution.mode, source='q_capture_angular')
        return replace(track, identity=self.identity,
            estimate_source=(self.measurement_cache.estimate_source or self.measurement_cache.source),
            publication_time_s=time_s)
