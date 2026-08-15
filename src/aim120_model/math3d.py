"""Small dependency-free 3D vector helpers."""

from __future__ import annotations

import math
from typing import Iterable, Tuple

Vector = Tuple[float, float, float]


def as_vector(values: Iterable[float]) -> Vector:
    x, y, z = values
    return (float(x), float(y), float(z))


def add(a: Vector, b: Vector) -> Vector:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Vector, b: Vector) -> Vector:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def scale(a: Vector, factor: float) -> Vector:
    return (a[0] * factor, a[1] * factor, a[2] * factor)


def dot(a: Vector, b: Vector) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vector, b: Vector) -> Vector:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm(a: Vector) -> float:
    return math.sqrt(dot(a, a))


def normalize(a: Vector, epsilon: float = 1e-12, fallback: Vector = (1.0, 0.0, 0.0)) -> Vector:
    length = norm(a)
    if length <= epsilon:
        return fallback
    return scale(a, 1.0 / length)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def clamp_norm(a: Vector, maximum: float, epsilon: float = 1e-12) -> Vector:
    length = norm(a)
    if length <= maximum or length <= epsilon:
        return a
    return scale(a, maximum / length)


def lerp(a: Vector, b: Vector, fraction: float) -> Vector:
    return add(a, scale(sub(b, a), fraction))


def is_finite_vector(a: Vector) -> bool:
    return all(math.isfinite(value) for value in a)
