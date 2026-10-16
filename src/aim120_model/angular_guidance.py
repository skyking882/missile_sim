"""Complete seeker angular inputs for the shared surface runtime.

World axes are the runtime's x/y-up/z. The stored rate uses n_dot = n x w,
not the right-hand angular-velocity sign. This is an explicit game-equivalent
input contract, not a model for generating the unknown axial seeker state.
"""
from __future__ import annotations

import bisect
import math
import struct
from dataclasses import dataclass, replace
from typing import Callable, Sequence

from .math3d import Vector, add, as_vector, cross, dot, lerp, norm, scale, sub
from .tracking import TrackMode, TrackSolution


def _vector(value) -> Vector:
    value = as_vector(value)
    if not all(math.isfinite(x) for x in value):
        raise ValueError("Angular input vectors must be finite")
    return value


def _unit(value) -> Vector:
    value = _vector(value)
    size = norm(value)
    if size <= 1e-12:
        raise ValueError("Angular input direction must be nonzero")
    return scale(value, 1 / size)


@dataclass(frozen=True)
class AngularObservation:
    """One complete angular state and the observing missile's world state.

    Range is metres, closing speed is positive on approach, rate is rad/s.
    The observer state must be at the same time as these angular quantities.
    ``from_native`` converts the replay native axes, including the axial-vector
    parity of rate. Convert positions to a common origin before calling it.
    """
    direction: Vector
    full_rate_rad_s: Vector
    range_m: float
    closing_mps: float
    observer_position_m: Vector
    observer_velocity_mps: Vector

    def __post_init__(self):
        object.__setattr__(self, "direction", _unit(self.direction))
        for name in ("full_rate_rad_s", "observer_position_m", "observer_velocity_mps"):
            object.__setattr__(self, name, _vector(getattr(self, name)))
        for name in ("range_m", "closing_mps"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.range_m <= 0:
            raise ValueError("Angular observation range must be positive")

    @classmethod
    def from_native(cls, *, direction, full_rate_rad_s, range_m, closing_mps,
                    observer_position_m, observer_velocity_mps):
        """Native→runtime: polar (x,y,-z), axial rate (-x,-y,z)."""
        def polar(v):
            x, y, z = _vector(v)
            return x, y, -z
        return cls(polar(direction), scale(polar(full_rate_rad_s), -1.),
                   range_m, closing_mps, polar(observer_position_m), polar(observer_velocity_mps))

    @property
    def axial_rate_rad_s(self):
        return dot(self.direction, self.full_rate_rad_s)

    def target_motion(self):
        n = self.direction
        relative_velocity = sub(scale(cross(n, self.full_rate_rad_s), self.range_m),
                                scale(n, self.closing_mps))
        return (add(self.observer_position_m, scale(n, self.range_m)),
                add(self.observer_velocity_mps, relative_velocity))

    def at_observer(self, position_m, velocity_mps):
        """Keep absolute target motion and axial state; change own motion only."""
        position_m, velocity_mps = _vector(position_m), _vector(velocity_mps)
        p, v = self.target_motion()
        relative, rv = sub(p, position_m), sub(v, velocity_mps)
        distance = norm(relative)
        n = _unit(relative)
        w = add(scale(cross(rv, n), 1 / distance), scale(n, self.axial_rate_rad_s))
        return AngularObservation(n, w, distance, -dot(rv, n), position_m, velocity_mps)

    def track_for(self, time_s, position_m, velocity_mps, *, sample_time_s=None,
                  mode=TrackMode.RADAR_TRACK, source="angular_observation"):
        p, v = self.target_motion()
        # At contact guidance has no defined LOS; the runtime returns zero there.
        rate = None if norm(sub(p, position_m)) <= 1e-12 else self.at_observer(position_m, velocity_mps).full_rate_rad_s
        return TrackSolution(p, v, time_s if sample_time_s is None else sample_time_s,
                             time_s, mode, True, source, rate)


class GuidanceInputProvider:
    """Runtime adapter for source(t, own_position, own_velocity) -> TrackSolution.

    Truth remains available to events/telemetry, but is not passed to this
    source. A live observer may implement the same contract as replay history.
    """
    provider_name = "external_guidance_input"

    def __init__(self, source):
        if not callable(source):
            raise ValueError("guidance_input must be callable")
        self.source = source
        self.radar_has_tracked_once = False

    def update(self, time_s, dt_s, missile, target_truth):
        solution = self.source(time_s, missile.position, missile.velocity)
        if not isinstance(solution, TrackSolution):
            raise ValueError("guidance_input must return TrackSolution")
        if abs(solution.solution_time_s - time_s) > 1e-10:
            raise ValueError("guidance_input must return a solution at the requested time")
        if solution.valid and solution.mode == TrackMode.RADAR_TRACK:
            self.radar_has_tracked_once = True
        return solution


def native_tick_time(time_s: float, physics_dt_s: float = 1 / 48) -> float:
    """Restore a float32 physics timestamp, not an assumed radar update rate.

    Reject arbitrary timestamps instead of rounding a real event to a tick.
    Exact native tick coordinates are accepted as well as their float32 export.
    """
    if not math.isfinite(physics_dt_s) or physics_dt_s <= 0 or not math.isfinite(time_s):
        raise ValueError("Native clock needs finite time and positive physics_dt_s")
    nominal = round(time_s / physics_dt_s) / (1 / physics_dt_s)
    if time_s != nominal and time_s != struct.unpack("f", struct.pack("f", nominal))[0]:
        raise ValueError("Recorded time is not a float32 native physics tick")
    return nominal


@dataclass(frozen=True)
class RecordedSeekerSample:
    """Saved INS/DL Ps/Vs or TRK angular state in runtime coordinates.

    Ps/Vs are used without another alpha-beta pass. For angular TRK records,
    position/velocity are retained as recorded metadata; guidance reconstructs
    target motion from ``angular`` instead. Times are seconds since launch.
    Optional ``last_measurement_time_s`` identifies a cached angular range.
    Direction/rate/closing still describe the saved state time. Without this
    metadata the existing interpolation contract is unchanged.
    """
    time_s: float
    mode: TrackMode
    position_m: Vector
    velocity_mps: Vector
    angular: AngularObservation | None = None
    valid: bool = True
    last_measurement_time_s: float | None = None

    def __post_init__(self):
        if not math.isfinite(self.time_s):
            raise ValueError("Seeker sample time must be finite")
        object.__setattr__(self, "mode", TrackMode(self.mode))
        object.__setattr__(self, "position_m", _vector(self.position_m))
        object.__setattr__(self, "velocity_mps", _vector(self.velocity_mps))
        if self.angular is not None and self.mode not in (TrackMode.RADAR_TRACK, TrackMode.RADAR_COAST):
            raise ValueError("Angular state requires radar track/coast mode")
        if self.last_measurement_time_s is not None:
            if (not math.isfinite(self.last_measurement_time_s)
                    or self.last_measurement_time_s > self.time_s + 1e-10):
                raise ValueError("Last measurement must be finite and not after its saved state")
            if self.angular is None:
                raise ValueError("Measurement timing requires angular state")


def _hermite(p0, v0, p1, v1, dt, u):
    p = add(add(scale(p0, 2*u**3-3*u*u+1), scale(v0, (u**3-2*u*u+u)*dt)),
            add(scale(p1, -2*u**3+3*u*u), scale(v1, (u**3-u*u)*dt)))
    v = add(add(scale(p0, (6*u*u-6*u)/dt), scale(v0, 3*u*u-4*u+1)),
            add(scale(p1, (-6*u*u+6*u)/dt), scale(v1, 3*u*u-2*u)))
    return p, v


def _rotate(value, rate, elapsed):
    """Rodrigues rotation; positive elapsed undoes n_dot = n x rate."""
    speed = norm(rate)
    if speed <= 1e-12:
        return value
    axis, angle = scale(rate, 1/speed), speed*elapsed
    return add(add(scale(value, math.cos(angle)), scale(cross(axis, value), math.sin(angle))),
               scale(axis, dot(axis, value)*(1-math.cos(angle))))


class RecordedSeekerHistory:
    """Offline replay input with explicit mode boundaries and full rate.

    Same-mode records interpolate. Across mode changes the old state predicts
    forward until the new timestamp; a later TRK never affects pre-capture INS.
    Before the first sample is an error; after the last, coast the last target
    motion. No capture time is inferred from saved-record spacing.

    Optional ``capture_times_s`` maps first saved TRK time -> supplied acquisition
    time in its preceding INS gap. This enables the prior-constrained offline
    bridge only after that acquisition. ``reference_motion(t) -> (p,v)`` is then
    required for the recorded missile, in runtime coordinates. The bridge uses
    the prior INS velocity and first TRK target velocity, never two future TRK
    records extrapolated backwards. Its new axial state is the first saved TRK
    value; this is a reconstruction assumption, not a decoded initialization.
    """

    def __init__(self, samples: Sequence[RecordedSeekerSample], *,
                 physics_dt_s: float = 1 / 48, restore_native_clock: bool = True,
                 capture_times_s: dict[float, float] | None = None,
                 reference_motion: Callable[[float], tuple[Vector, Vector]] | None = None):
        if not math.isfinite(physics_dt_s) or physics_dt_s <= 0:
            raise ValueError("physics_dt_s must be positive and finite")
        self.physics_dt_s = physics_dt_s
        restore = lambda t: native_tick_time(t, physics_dt_s) if restore_native_clock else float(t)
        self.samples = tuple(replace(s, time_s=restore(s.time_s)) for s in samples)
        self.times = tuple(s.time_s for s in self.samples)
        if not self.times or any(b <= a for a, b in zip(self.times, self.times[1:])):
            raise ValueError("Seeker history needs strictly increasing samples")
        self.reference_motion = reference_motion
        # Optional replay freshness contract. Direction/rate are state-time
        # quantities, while a held range still belongs to the last measurement.
        # Closing and own velocity remain state-time quantities: do not move
        # every field backwards with the cached range.
        self.measurement_motion = {}
        previous_stamp, previous_motion = None, None
        for i, sample in enumerate(self.samples):
            stamp = sample.last_measurement_time_s
            if stamp is None or not sample.valid:
                previous_stamp, previous_motion = None, None
                continue
            if previous_stamp == stamp:
                motion = previous_motion
            elif stamp == sample.time_s:
                motion = sample.angular.target_motion()
            else:
                if not callable(reference_motion):
                    raise ValueError("Held angular timing requires recorded reference_motion")
                position, _ = reference_motion(stamp)
                angular = sample.angular
                direction = _rotate(angular.direction, angular.full_rate_rad_s, sample.time_s-stamp)
                _, velocity = angular.target_motion()
                motion = (add(position, scale(direction, angular.range_m)), velocity)
            self.measurement_motion[i] = motion
            previous_stamp, previous_motion = stamp, motion
        self.captures = {}
        for end, start in (capture_times_s or {}).items():
            end, start = restore(end), float(start)
            i = bisect.bisect_left(self.times, end)
            if (not math.isfinite(start) or i == 0 or i == len(self.times) or self.times[i] != end
                    or not self.times[i-1] < start <= end):
                raise ValueError("Capture must lie after the preceding INS and at/before saved TRK")
            a, b = self.samples[i-1:i+1]
            if (a.mode not in (TrackMode.INERTIAL, TrackMode.INS_SEARCH, TrackMode.DATALINK)
                    or a.angular is not None or not a.valid or not b.valid
                    or b.mode != TrackMode.RADAR_TRACK or b.angular is None):
                raise ValueError("Capture bridge requires INS/DL followed by angular TRK")
            if not callable(reference_motion):
                raise ValueError("Capture bridge requires recorded reference_motion")
            self.captures[i] = start

    def _bridge(self, a, b, start, time_s):
        dt = b.time_s - start
        u = (time_s - start) / dt
        own0, vel0 = self.reference_motion(start)
        own, vel = self.reference_motion(time_s)
        p0 = add(a.position_m, scale(a.velocity_mps, start-a.time_s))
        _, v1 = b.angular.target_motion()
        r0, rv0 = sub(p0, own0), sub(a.velocity_mps, vel0)
        r1 = scale(b.angular.direction, b.angular.range_m)
        rv1 = sub(v1, b.angular.observer_velocity_mps)
        r, _ = _hermite(r0, rv0, r1, rv1, dt, u)
        n, distance = _unit(r), norm(r)
        # Velocity is an estimator state, not the derivative of the positional
        # correction. That derivative created a false first-gap turn impulse.
        rv = sub(lerp(a.velocity_mps, v1, u), vel)
        rate = add(scale(cross(rv, n), 1/distance), scale(n, b.angular.axial_rate_rad_s))
        return AngularObservation(n, rate, distance, -dot(rv, n), own, vel)

    def __call__(self, time_s, own_position_m, own_velocity_mps):
        if not math.isfinite(time_s):
            raise ValueError("Seeker query time must be finite")
        nominal = round(time_s / self.physics_dt_s) / (1 / self.physics_dt_s)
        t = nominal if abs(time_s-nominal) < 1e-10 else time_s
        i = bisect.bisect_right(self.times, t) - 1
        if i < 0:
            raise ValueError("Seeker history does not cover launch/query time")
        a = self.samples[i]
        b = self.samples[i+1] if i+1 < len(self.samples) else None
        start = self.captures.get(i+1)
        if b is not None and start is not None and start <= t < b.time_s:
            return self._bridge(a, b, start, t).track_for(
                time_s, own_position_m, own_velocity_mps, sample_time_s=start,
                source="replay_acquisition_bridge")
        same_mode = (b is not None and a.valid and b.valid and a.mode == b.mode
                     and (a.angular is None) == (b.angular is None))
        u = (t-a.time_s)/(b.time_s-a.time_s) if same_mode else 0.
        if a.angular is not None and a.valid:
            held = a.last_measurement_time_s is not None and a.last_measurement_time_s < a.time_s
            next_held = (same_mode and b.last_measurement_time_s is not None
                         and b.last_measurement_time_s < b.time_s)
            if a.last_measurement_time_s is not None and (held or next_held or (not same_mode and t != a.time_s)):
                # Keep the absolute last measurement anchored; own movement
                # cannot translate it by pairing old range with current own p.
                # A later fresh record becomes available only at its timestamp.
                p, v = self.measurement_motion[i]
                p = add(p, scale(v, t-a.last_measurement_time_s))
                relative, rv = sub(p, own_position_m), sub(v, own_velocity_mps)
                distance = norm(relative)
                rate = None
                if distance > 1e-12:
                    n = scale(relative, 1/distance)
                    rate = add(scale(cross(rv, n), 1/distance),
                               scale(n, a.angular.axial_rate_rad_s))
                return TrackSolution(p, v, a.last_measurement_time_s, time_s, a.mode, True,
                                     "replay_timed_angular_hold", rate)
            ang = a.angular
            if same_mode:
                other = b.angular
                p, v = (self.reference_motion(t) if self.reference_motion is not None else
                        _hermite(ang.observer_position_m, ang.observer_velocity_mps,
                                 other.observer_position_m, other.observer_velocity_mps,
                                 b.time_s-a.time_s, u))
                ang = AngularObservation(lerp(ang.direction, other.direction, u),
                    lerp(ang.full_rate_rad_s, other.full_rate_rad_s, u),
                    ang.range_m+u*(other.range_m-ang.range_m),
                    ang.closing_mps+u*(other.closing_mps-ang.closing_mps), p, v)
            elif t != a.time_s:
                p, v = ang.target_motion()
                p = add(p, scale(v, t-a.time_s))
                relative = sub(p, own_position_m)
                distance, n = norm(relative), _unit(relative)
                rv = sub(v, own_velocity_mps)
                rate = add(scale(cross(rv, n), 1/distance), scale(n, ang.axial_rate_rad_s))
                ang = AngularObservation(n, rate, distance, -dot(rv, n), own_position_m, own_velocity_mps)
            return ang.track_for(time_s, own_position_m, own_velocity_mps,
                                 sample_time_s=a.time_s, mode=a.mode, source="replay_full_rate")
        if same_mode:
            p, v = lerp(a.position_m, b.position_m, u), lerp(a.velocity_mps, b.velocity_mps, u)
        else:
            p, v = add(a.position_m, scale(a.velocity_mps, t-a.time_s)), a.velocity_mps
        return TrackSolution(p, v, a.time_s, time_s, a.mode, a.valid, "replay_PsVs")
