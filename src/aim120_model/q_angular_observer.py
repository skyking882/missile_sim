"""Opt-in empirical complete-rate observer for WT's active-radar game model.

Q at acquisition and the shared 11.23 /s correction gain are approximations
identified from replay state relationships, not recovered WT implementation.
All inputs here are received observations or the missile's own state.
"""
from __future__ import annotations

import math

from .aerodynamics import body_axes_for_state
from .angular_guidance import AngularObservation, _rotate
from .math3d import add, cross, dot, norm, scale, sub
from .tracking import TrackMode


def validate_angular_observer(mode):
    if mode not in (None, 'q_capture'):
        raise ValueError("angular_observer must be None or 'q_capture'")
    return mode


class QCaptureAngularObserver:
    gain_per_s = 11.23
    provenance = dict(mode='q_capture', kind='assumed', grade='D',
        rate_correction_gain_per_s=gain_per_s,
        initialization='previous available INS/DL geometric rate, previous full body posture Q once; measured direction',
        update='received LOS innovation; n_dot = n x rate; no rate transport at correction',
        missing_prior='zero rate', reacquisition='same Q reset rule; not independently verified')

    def __init__(self):
        self.direction = None
        self.rate = None
        self.range_m = None
        self.closing_mps = None
        self.time_s = None
        self.sample_time_s = None
        self.identity = None
        self.memory_identity = ('target',)
        self.prior_identity = None
        self.prior_rate = None
        self.prior_axes = None
        self.correction_count = 0
        self.capture_count = 0

    def _clear(self):
        self.direction = self.rate = self.range_m = self.closing_mps = None
        self.time_s = self.sample_time_s = self.identity = None

    def update(self, solution, missile, *, received, identity, closing_mps):
        """Advance once; ``received`` means an actually accepted radar echo."""
        time_s = solution.solution_time_s
        if self.time_s is not None:
            elapsed = time_s - self.time_s
            if elapsed < -1e-12:
                raise ValueError('angular observer time must not move backwards')
            self.direction = _rotate(self.direction, self.rate, -max(elapsed, 0.))
            self.range_m -= self.closing_mps * max(elapsed, 0.)
            self.time_s = time_s

        active_mode = solution.valid and solution.mode in (TrackMode.RADAR_TRACK, TrackMode.RADAR_COAST)
        if not active_mode:
            self._clear()
            # Only an already available provider solution may initialize Q.
            # A loss without an estimate must not retain a previous target.
            self.prior_rate = None
            self.prior_axes = None
            if solution.valid:
                relative = sub(solution.position, missile.position)
                distance = norm(relative)
                if distance > 1e-12:
                    n = scale(relative, 1 / distance)
                    self.prior_rate = scale(cross(sub(solution.velocity, missile.velocity), n), 1 / distance)
                    self.prior_axes = body_axes_for_state(missile)
                    self.prior_identity = (('target',) if solution.mode is TrackMode.DATALINK
                                           else self.memory_identity)
            return solution

        fresh = received and (self.sample_time_s is None or solution.sample_time_s > self.sample_time_s)
        changed_identity = self.direction is not None and identity != self.identity
        if received and changed_identity:
            # The old target's rate is not a prior for a new reflector.
            self._clear()
            self.prior_rate = self.prior_axes = None
            fresh = True
        if fresh:
            relative = sub(solution.position, missile.position)
            distance = norm(relative)
            if distance <= 1e-12:
                self._clear()
                self.prior_rate = self.prior_axes = None
                return solution
            measured = scale(relative, 1 / distance)
            if self.direction is None:
                self.rate = (0., 0., 0.)
                if self.prior_rate is not None and self.prior_identity == identity:
                    axes = self.prior_axes
                    # Runtime Q = C Q_native C. Both axial parity factors cancel
                    # in Q*w, so the runtime's full body basis applies directly.
                    self.rate = add(add(scale(axes.forward, self.prior_rate[0]),
                                        scale(axes.up, self.prior_rate[1])),
                                    scale(axes.right, self.prior_rate[2]))
                self.capture_count += 1
            else:
                axis = cross(measured, self.direction)
                sine = norm(axis)
                cosine = max(-1., min(1., dot(measured, self.direction)))
                if sine <= 1e-12 and cosine < 0:
                    raise ValueError('antipodal angular measurement has no unique innovation axis')
                correction = scale(axis, math.atan2(sine, cosine) / sine) if sine > 1e-12 else axis
                self.rate = add(self.rate, scale(correction, self.gain_per_s))
                self.correction_count += 1
            self.direction = measured
            self.range_m = distance
            self.closing_mps = float(closing_mps)
            self.time_s = time_s
            self.sample_time_s = solution.sample_time_s
            self.identity = identity
            self.memory_identity = identity
            self.prior_rate = self.prior_axes = None

        if self.direction is None or self.range_m <= 1e-12:
            self._clear()
            return solution
        angular = AngularObservation(self.direction, self.rate, self.range_m, self.closing_mps,
                                     missile.position, missile.velocity)
        return angular.track_for(time_s, missile.position, missile.velocity,
            sample_time_s=self.sample_time_s, mode=solution.mode, source='q_capture_angular')
