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
        if self.full_rate_rad_s is not None:
            rate = as_vector(tuple(self.full_rate_rad_s))
            if not all(math.isfinite(value) for value in rate):
                raise ValueError("TrackSolution full_rate_rad_s must be finite")
            object.__setattr__(self, "full_rate_rad_s", rate)

    @property
    def age_s(self) -> float:
        return max(0.0, self.solution_time_s - self.sample_time_s)


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

    def __post_init__(self) -> None:
        copied = TrackSolution(self.position, self.velocity, self.sample_time_s,
                               self.sample_time_s, TrackMode.DATALINK, True, self.source)
        object.__setattr__(self, "position", copied.position)
        object.__setattr__(self, "velocity", copied.velocity)
        object.__setattr__(self, "sample_time_s", copied.sample_time_s)
        object.__setattr__(self, "source", copied.source)


__all__ = ["AlphaBetaGate", "AlphaBetaStateObserver", "DatalinkObservation", "TrackMode", "TrackSolution"]
