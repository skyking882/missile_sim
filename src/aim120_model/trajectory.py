"""Time-tabulated target trajectories with deterministic interpolation."""

from __future__ import annotations

import bisect
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .math3d import Vector, add, as_vector, scale, sub
from .target import TabulatedTargetModel, TargetState


def _finite_vector(values: Iterable[float], label: str) -> Vector:
    vector = as_vector(tuple(values))
    if not all(math.isfinite(value) for value in vector):
        raise ValueError(f"{label} must contain finite values")
    return vector


@dataclass(frozen=True)
class TrajectoryPoint:
    time_s: float
    position: Vector
    velocity: Vector | None = None

    def __post_init__(self) -> None:
        time_s = float(self.time_s)
        if not math.isfinite(time_s):
            raise ValueError("trajectory time_s must be finite")
        position = _finite_vector(self.position, "trajectory position")
        velocity = None if self.velocity is None else _finite_vector(self.velocity, "trajectory velocity")
        object.__setattr__(self, "time_s", time_s)
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "velocity", velocity)


class TabulatedTrajectory:
    """A finite sequence of target points with no extrapolation."""

    REQUIRED_COLUMNS = ("time_s", "x_m", "y_m", "z_m")
    VELOCITY_COLUMNS = ("vx_mps", "vy_mps", "vz_mps")

    def __init__(self, points: Iterable[TrajectoryPoint]):
        materialized = tuple(points)
        if len(materialized) < 2:
            raise ValueError("tabulated trajectory requires at least two points")
        for previous, current in zip(materialized, materialized[1:]):
            if current.time_s <= previous.time_s:
                raise ValueError("trajectory time_s must be strictly increasing")
        self._points = materialized
        self._times = tuple(point.time_s for point in materialized)
        self._has_velocity = all(point.velocity is not None for point in materialized)
        if any((point.velocity is None) != (not self._has_velocity) for point in materialized):
            raise ValueError("velocity columns must be complete for every trajectory point")

    @classmethod
    def from_csv(cls, path: str | Path) -> "TabulatedTrajectory":
        file_path = Path(path)
        with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            missing = [column for column in cls.REQUIRED_COLUMNS if column not in fieldnames]
            if missing:
                raise ValueError(f"trajectory CSV missing required columns: {', '.join(missing)}")
            velocity_present = [column in fieldnames for column in cls.VELOCITY_COLUMNS]
            if any(velocity_present) and not all(velocity_present):
                raise ValueError("trajectory velocity columns must be supplied as a complete group")
            points: list[TrajectoryPoint] = []
            for row_number, row in enumerate(reader, start=2):
                if not row or all(value in (None, "") for value in row.values()):
                    continue
                try:
                    time_s = float(row["time_s"])
                    position = (
                        float(row["x_m"]),
                        float(row["y_m"]),
                        float(row["z_m"]),
                    )
                    velocity = None
                    if all(velocity_present):
                        velocity = (
                            float(row["vx_mps"]),
                            float(row["vy_mps"]),
                            float(row["vz_mps"]),
                        )
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"invalid trajectory CSV row {row_number}") from exc
                points.append(TrajectoryPoint(time_s, position, velocity))
        return cls(points)

    @property
    def points(self) -> tuple[TrajectoryPoint, ...]:
        return self._points

    @property
    def start_time_s(self) -> float:
        return self._times[0]

    @property
    def end_time_s(self) -> float:
        return self._times[-1]

    def _segment(self, time_s: float) -> tuple[int, float, float]:
        if not math.isfinite(time_s):
            raise ValueError("trajectory query time_s must be finite")
        if time_s < self.start_time_s or time_s > self.end_time_s:
            raise ValueError(
                f"trajectory query {time_s:g}s is outside [{self.start_time_s:g}, {self.end_time_s:g}]s"
            )
        if time_s == self.start_time_s:
            return 0, 0.0, self._times[1] - self._times[0]
        if time_s == self.end_time_s:
            index = len(self._points) - 2
            return index, 1.0, self._times[-1] - self._times[-2]
        index = max(0, min(len(self._points) - 2, bisect.bisect_right(self._times, time_s) - 1))
        h = self._times[index + 1] - self._times[index]
        fraction = (time_s - self._times[index]) / h
        return index, fraction, h

    @staticmethod
    def _linear_state(point0: TrajectoryPoint, point1: TrajectoryPoint, fraction: float, h: float) -> TargetState:
        position = add(point0.position, scale(sub(point1.position, point0.position), fraction))
        velocity = scale(sub(point1.position, point0.position), 1.0 / h)
        return TargetState(position, velocity)

    @staticmethod
    def _hermite_state(point0: TrajectoryPoint, point1: TrajectoryPoint, fraction: float, h: float) -> TargetState:
        assert point0.velocity is not None and point1.velocity is not None
        u = fraction
        h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
        h10 = u**3 - 2.0 * u**2 + u
        h01 = -2.0 * u**3 + 3.0 * u**2
        h11 = u**3 - u**2
        position = (
            h00 * point0.position[0] + h10 * h * point0.velocity[0] + h01 * point1.position[0] + h11 * h * point1.velocity[0],
            h00 * point0.position[1] + h10 * h * point0.velocity[1] + h01 * point1.position[1] + h11 * h * point1.velocity[1],
            h00 * point0.position[2] + h10 * h * point0.velocity[2] + h01 * point1.position[2] + h11 * h * point1.velocity[2],
        )
        dh00 = 6.0 * u**2 - 6.0 * u
        dh10 = 3.0 * u**2 - 4.0 * u + 1.0
        dh01 = -6.0 * u**2 + 6.0 * u
        dh11 = 3.0 * u**2 - 2.0 * u
        velocity = (
            (dh00 * point0.position[0] + dh10 * h * point0.velocity[0] + dh01 * point1.position[0] + dh11 * h * point1.velocity[0]) / h,
            (dh00 * point0.position[1] + dh10 * h * point0.velocity[1] + dh01 * point1.position[1] + dh11 * h * point1.velocity[1]) / h,
            (dh00 * point0.position[2] + dh10 * h * point0.velocity[2] + dh01 * point1.position[2] + dh11 * h * point1.velocity[2]) / h,
        )
        return TargetState(position, velocity)

    def state_at(self, time_s: float) -> TargetState:
        index, fraction, h = self._segment(float(time_s))
        point0 = self._points[index]
        point1 = self._points[index + 1]
        if fraction == 0.0:
            if point0.velocity is not None:
                return TargetState(point0.position, point0.velocity)
            return self._linear_state(point0, point1, 0.0, h)
        if fraction == 1.0:
            if point1.velocity is not None:
                return TargetState(point1.position, point1.velocity)
            return self._linear_state(point0, point1, 1.0, h)
        if self._has_velocity:
            return self._hermite_state(point0, point1, fraction, h)
        return self._linear_state(point0, point1, fraction, h)


__all__ = ["TabulatedTargetModel", "TabulatedTrajectory", "TrajectoryPoint"]
