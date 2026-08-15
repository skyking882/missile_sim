"""Small, deterministic tracking objects used by the guidance boundary.

The tracking layer deliberately contains no probability model or random noise.
It only copies numerical observations, applies the scalar alpha-beta gates from
PLAN8, and labels the source of the solution consumed by PN.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from .math3d import Vector, as_vector


class TrackMode(str, Enum):
    IDEAL_TRUTH = "ideal_truth"
    RADAR_SEARCH = "radar_search"
    RADAR_TRACK = "radar_track"
    RADAR_COAST = "radar_coast"
    INS_SEARCH = "ins_search"
    PROFILE_KINEMATIC = "profile_kinematic_track"
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


__all__ = ["AlphaBetaGate", "TrackMode", "TrackSolution"]
