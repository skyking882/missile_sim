"""Explicit event crossing detection and interpolation helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .math3d import dot, norm, sub
from .target import TargetState


@dataclass(frozen=True)
class EventCandidate:
    event_type: str
    fraction: float


def _crossing_fraction(value0: float, value1: float, threshold: float) -> float | None:
    if value0 <= threshold:
        return 0.0
    if value1 > threshold:
        return None
    denominator = value1 - value0
    if abs(denominator) <= 1e-15:
        return 1.0
    return max(0.0, min(1.0, (threshold - value0) / denominator))


def _upward_crossing_fraction(value0: float, value1: float, threshold: float) -> float | None:
    """Detect a crossing from below to above, used for maximum range."""

    if value0 >= threshold:
        return 0.0
    if value1 < threshold:
        return None
    denominator = value1 - value0
    if abs(denominator) <= 1e-15:
        return 1.0
    return max(0.0, min(1.0, (threshold - value0) / denominator))


def _segment_sphere_entry_fraction(
    relative0: tuple[float, float, float],
    relative1: tuple[float, float, float],
    radius: float,
) -> float | None:
    """Return the first within-step intersection with a proximity-fuse sphere.

    Missile and target positions are linearly interpolated over one integration
    step, so their relative motion is also a line segment.  Solving the segment
    against the fuse sphere catches high-speed passes whose two sampled endpoint
    distances are both outside the radius.
    """

    radius_sq = float(radius) * float(radius)
    if dot(relative0, relative0) <= radius_sq:
        return 0.0

    delta = sub(relative1, relative0)
    a = dot(delta, delta)
    if a <= 1e-30:
        return None

    b = dot(relative0, delta)
    c = dot(relative0, relative0) - radius_sq
    discriminant = b * b - a * c
    tolerance = 1e-12 * max(b * b, abs(a * c), 1.0)
    if discriminant < -tolerance:
        return None

    root = math.sqrt(max(0.0, discriminant))
    entry = (-b - root) / a
    exit_ = (-b + root) / a
    if exit_ < 0.0 or entry > 1.0:
        return None
    return max(0.0, min(1.0, entry))


def event_candidates(
    state0: Any,
    state1: Any,
    target0: TargetState,
    target1: TargetState,
    time0_s: float,
    time1_s: float,
    config: dict[str, Any],
    launch_position: tuple[float, float, float],
) -> list[EventCandidate]:
    candidates: list[EventCandidate] = []
    guidance_cfg = config["guidance"]
    if guidance_cfg.get("proximity_fuse_enabled", True):
        relative0 = sub(target0.position, state0.position)
        relative1 = sub(target1.position, state1.position)
        fraction = _segment_sphere_entry_fraction(
            relative0,
            relative1,
            guidance_cfg["proximity_radius_m"],
        )
        if fraction is not None:
            candidates.append(EventCandidate("fuse", fraction))
    ground_fraction = _crossing_fraction(state0.position[1], state1.position[1], 0.0)
    if ground_fraction is not None:
        candidates.append(EventCandidate("ground", ground_fraction))
    performance = config["performance"]
    range0 = norm(sub(state0.position, launch_position))
    range1 = norm(sub(state1.position, launch_position))
    range_fraction = _upward_crossing_fraction(range0, range1, performance["maximum_distance_m"])
    if range_fraction is not None:
        candidates.append(EventCandidate("max_distance", range_fraction))
    if time1_s >= performance["lifetime_s"] and time0_s < performance["lifetime_s"]:
        fraction = (performance["lifetime_s"] - time0_s) / max(time1_s - time0_s, 1e-15)
        candidates.append(EventCandidate("lifetime", max(0.0, min(1.0, fraction))))
    priority = {"fuse": 0, "ground": 1, "max_distance": 2, "lifetime": 3}
    return sorted(candidates, key=lambda item: (item.fraction, priority[item.event_type]))
