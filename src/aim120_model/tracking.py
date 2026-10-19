"""Small, deterministic tracking objects used by the guidance boundary.

The tracking layer deliberately contains no probability model or random noise.
It copies numerical observations, applies the scalar alpha-beta gates from
PLAN8 or an explicit 3D observer, and labels the solution consumed by PN.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from .math3d import Vector, add, as_vector, scale, sub


class TrackMode(str, Enum):
    IDEAL_TRUTH = "ideal_truth"
    RADAR_SEARCH = "radar_search"
    RADAR_TRACK = "radar_track"
    RADAR_COAST = "radar_coast"
    INS_SEARCH = "ins_search"
    PROFILE_KINEMATIC = "profile_kinematic_track"
    INFRARED_TRACK = "infrared_track"
    DATALINK = "datalink"
    INERTIAL = "inertial"
    SNS_FIXED_POINT = "sns_fixed_point"
    LOST = "lost"


@dataclass(frozen=True)
class TrackIdentity:
    """Object identity, independent of the sensor currently reporting it.

    Unknown is a real absence of classification, not an inferred aircraft.
    Object IDs must be immutable so snapshots cannot change retroactively.
    """
    object_type: str = "unknown"
    object_id: str | int | tuple | None = None

    def __post_init__(self):
        if self.object_type not in ("aircraft", "missile", "chaff", "unknown"):
            raise ValueError("unsupported track object_type")
        def immutable(value):
            if isinstance(value, (tuple, list)):
                return tuple(immutable(x) for x in value)
            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            raise ValueError("track object_id must be an immutable scalar or tuple")
        object.__setattr__(self, "object_id", immutable(self.object_id))


INTENDED_TARGET = TrackIdentity("unknown", "intended_target")


def reflector_identity(identity, intended=INTENDED_TARGET, kind="unknown"):
    """Translate existing association keys without guessing an object's type."""
    if isinstance(identity, TrackIdentity):
        return identity
    if identity == ("target",):
        return intended
    if isinstance(identity, tuple) and len(identity) == 2 and identity[0] == "decoy":
        return TrackIdentity(kind if kind in ("aircraft", "missile", "chaff") else "unknown", identity)
    return None


@dataclass(frozen=True)
class TrackSolution:
    """A copied target solution presented to guidance.

    ``position`` and ``velocity`` are normalized to fresh tuples in
    ``__post_init__``.  This is intentional: a solution must never retain a
    ``TargetState`` or another mutable observation object by reference.
    """

    position: Vector
    velocity: Vector
    sample_time_s: float
    solution_time_s: float
    mode: TrackMode
    valid: bool
    source: str
    # Optional complete angular estimator state in runtime world coordinates.
    # Convention: n_dot = n x full_rate_rad_s (opposite to right-hand omega).
    # Includes the independent component along LOS; position/velocity alone
    # cannot reconstruct it. Relative to the missile consuming this solution.
    full_rate_rad_s: Vector | None = None
    identity: TrackIdentity | None = None
    # Origin of the estimated P/V survives changes of transport/mode/source.
    estimate_source: str | None = None
    # Publication is distinct from the state and original measurement times.
    publication_time_s: float | None = None

    def __post_init__(self) -> None:
        position = as_vector(tuple(self.position))
        velocity = as_vector(tuple(self.velocity))
        if not all(math.isfinite(value) for value in position + velocity):
            raise ValueError("TrackSolution vectors must contain finite values")
        sample_time_s = float(self.sample_time_s)
        solution_time_s = float(self.solution_time_s)
        if not math.isfinite(sample_time_s) or not math.isfinite(solution_time_s):
            raise ValueError("TrackSolution times must be finite")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "velocity", velocity)
        object.__setattr__(self, "sample_time_s", sample_time_s)
        object.__setattr__(self, "solution_time_s", solution_time_s)
        object.__setattr__(self, "mode", TrackMode(self.mode))
        object.__setattr__(self, "valid", bool(self.valid))
        object.__setattr__(self, "source", str(self.source))
        if self.identity is not None and not isinstance(self.identity, TrackIdentity):
            raise TypeError("TrackSolution identity must be TrackIdentity or None")
        if self.estimate_source is not None:
            object.__setattr__(self, "estimate_source", str(self.estimate_source))
        if self.publication_time_s is not None:
            stamp = float(self.publication_time_s)
            if not math.isfinite(stamp) or stamp < solution_time_s - 1e-12:
                raise ValueError("publication must be finite and not precede state time")
            object.__setattr__(self, "publication_time_s", stamp)
        if self.full_rate_rad_s is not None:
            rate = as_vector(tuple(self.full_rate_rad_s))
            if not all(math.isfinite(value) for value in rate):
                raise ValueError("TrackSolution full_rate_rad_s must be finite")
            object.__setattr__(self, "full_rate_rad_s", rate)

    @property
    def age_s(self) -> float:
        return max(0.0, self.solution_time_s - self.sample_time_s)


@dataclass(frozen=True)
class ObserverStageSnapshot:
    """Copied state at one executed stage; update_index is not a physics tick.

    The provider retains only this call's bounded tuple of snapshots. There is
    no inferred capture sub-tick or implied replay serialization stage.
    """
    update_index: int
    time_s: float
    stage: str
    intended_identity: TrackIdentity
    reflector_identity: TrackIdentity | None
    estimator_identity: TrackIdentity | None
    estimate: TrackSolution | None
    measurement_cache: TrackSolution | None
    previous_accepted_time_s: float | None
    accepted_measurement_time_s: float | None
    angular_state_time_s: float | None
    publication_time_s: float | None
    antenna_axis_world: Vector | None
    target_los_estimate_world: Vector | None
    track_direction_world: Vector | None
    full_rate_world_rad_s: Vector | None
    range_m: float | None
    closing_mps: float | None
    radial_source: str | None
    reason: str = ""
    seed_source: str | None = None
    seed_rate_world_rad_s: Vector | None = None
    measurement_range_m: float | None = None
    measurement_relative_radial_mps: float | None = None
    measurement_source: str | None = None


@dataclass
class AlphaBetaGate:
    """One scalar deterministic alpha-beta measurement gate."""

    alpha: float
    beta: float
    search_range: float
    value: float = 0.0
    rate: float = 0.0
    initialized: bool = False

    def __post_init__(self) -> None:
        self.alpha = float(self.alpha)
        self.beta = float(self.beta)
        self.search_range = float(self.search_range)
        self.value = float(self.value)
        self.rate = float(self.rate)
        if not all(math.isfinite(value) for value in (self.alpha, self.beta, self.search_range, self.value, self.rate)):
            raise ValueError("AlphaBetaGate parameters must be finite")
        if self.search_range < 0.0:
            raise ValueError("AlphaBetaGate search_range must be non-negative")

    @staticmethod
    def _require_dt(dt_s: float) -> float:
        dt = float(dt_s)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("AlphaBetaGate requires a positive finite dt_s")
        return dt

    def predict(self, dt_s: float) -> float:
        dt = self._require_dt(dt_s)
        if not self.initialized:
            return self.value
        return self.value + self.rate * dt

    def accepts(self, measurement: float, dt_s: float) -> bool:
        measurement = float(measurement)
        if not math.isfinite(measurement):
            return False
        if not self.initialized:
            return True
        predicted = self.predict(dt_s)
        return abs(measurement - predicted) <= self.search_range

    def update(self, measurement: float, dt_s: float) -> None:
        measurement = float(measurement)
        if not math.isfinite(measurement):
            raise ValueError("AlphaBetaGate measurements must be finite")
        if not self.initialized:
            self.value = measurement
            self.rate = 0.0
            self.initialized = True
            return
        dt = self._require_dt(dt_s)
        predicted = self.value + self.rate * dt
        residual = measurement - predicted
        self.value = predicted + self.alpha * residual
        self.rate = self.rate + self.beta / dt * residual


class AlphaBetaStateObserver:
    """Opt-in 3D constant-velocity observer, corrected by position only.

    Velocity is seeded once from an available track. Subsequent measurement
    velocities are ignored. Predictions do not alter the last correction time,
    so an observation after a gap uses the full elapsed interval.
    """

    def __init__(self, alpha: float, beta: float):
        self.alpha, self.beta = float(alpha), float(beta)
        if not math.isfinite(self.alpha) or not math.isfinite(self.beta):
            raise ValueError("AlphaBetaStateObserver coefficients must be finite")
        self.initial_solution: TrackSolution | None = None
        self.solution: TrackSolution | None = None
        self.update_count = 0
        self.last_dt_s: float | None = None
        self.last_prediction: Vector | None = None
        self.last_innovation: Vector | None = None

    def initialize(self, solution: TrackSolution) -> None:
        if self.solution is not None:
            raise RuntimeError("AlphaBetaStateObserver velocity can only be initialized once")
        if not solution.valid:
            raise ValueError("AlphaBetaStateObserver requires a valid initial track")
        self.initial_solution = self.solution = solution

    def predict(self, time_s: float) -> Vector:
        if self.solution is None:
            raise RuntimeError("AlphaBetaStateObserver has not been initialized")
        dt = float(time_s) - self.solution.solution_time_s
        if not math.isfinite(dt) or dt < 0.0:
            raise ValueError("AlphaBetaStateObserver prediction time must not go backwards")
        return add(self.solution.position, scale(self.solution.velocity, dt))

    def update(self, measurement: TrackSolution, *, source: str = "radar_alpha_beta",
               allow_same_time: bool = False) -> TrackSolution:
        if self.solution is None:
            raise RuntimeError("AlphaBetaStateObserver has not been initialized")
        if not measurement.valid:
            raise ValueError("AlphaBetaStateObserver requires a valid position measurement")
        dt = measurement.solution_time_s - self.solution.solution_time_s
        if not math.isfinite(dt) or dt < 0.0 or (dt == 0.0 and not allow_same_time):
            raise ValueError("AlphaBetaStateObserver correction requires increasing times")
        predicted = self.predict(measurement.solution_time_s)
        innovation = sub(measurement.position, predicted)
        self.solution = TrackSolution(
            position=add(predicted, scale(innovation, self.alpha)),
            # Simultaneous independent position observations cannot identify a
            # new velocity. Only the unified mode explicitly permits this case.
            velocity=add(self.solution.velocity, scale(innovation, self.beta / dt if dt > 0.0 else 0.0)),
            sample_time_s=measurement.sample_time_s,
            solution_time_s=measurement.solution_time_s,
            mode=measurement.mode,
            valid=True,
            source=source,
            identity=measurement.identity,
            estimate_source=measurement.estimate_source or measurement.source,
        )
        self.update_count += 1
        self.last_dt_s, self.last_prediction, self.last_innovation = dt, predicted, innovation
        return self.solution


@dataclass(frozen=True)
class DatalinkObservation:
    """An external estimated world state at the original packet timestamp.

    Return (reason, observation) from launcher_support. Position is at
    sample_time_s (seconds since launch), not its per-tick extrapolated value.
    Velocity is a legitimate external prior, used only when initializing the
    unified observer. Repeating this packet never creates another correction.
    """

    position: Vector
    velocity: Vector
    sample_time_s: float
    source: str = "external_datalink"
    identity: TrackIdentity | None = None
    full_rate_rad_s: Vector | None = None

    def __post_init__(self) -> None:
        copied = TrackSolution(self.position, self.velocity, self.sample_time_s,
                               self.sample_time_s, TrackMode.DATALINK, True, self.source,
                               self.full_rate_rad_s, self.identity, self.source)
        object.__setattr__(self, "position", copied.position)
        object.__setattr__(self, "velocity", copied.velocity)
        object.__setattr__(self, "sample_time_s", copied.sample_time_s)
        object.__setattr__(self, "source", copied.source)
        object.__setattr__(self, "full_rate_rad_s", copied.full_rate_rad_s)


__all__ = ["AlphaBetaGate", "AlphaBetaStateObserver", "DatalinkObservation", "TrackMode", "TrackSolution", "TrackIdentity", "ObserverStageSnapshot"]
