"""Closed-loop evasive target for escape-window sweeps.

Before ``start_s`` the target is the unchanged scenario target. Afterwards it
turns its horizontal velocity toward a beam or drag heading relative to the
missile's last observed position, at constant speed and with the horizontal
turn rate limited by ``max_turn_g``. Vertical velocity is kept.

This is a kinematic geometry probe, not an aircraft flight model: it has no
energy loss, roll-in delay, altitude change or chaff. Within a 48 Hz control
tick the missile position is held from the tick start.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Any, Callable

from .target import TargetState

KINDS = ("beam", "drag")
SUBSTEP_S = 1 / 192


@dataclass(frozen=True)
class EvasionPlan:
    kind: str
    start_s: float
    max_turn_g: float = 9.0

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"evasion kind must be one of {KINDS}")
        if not math.isfinite(self.start_s) or self.start_s < 0:
            raise ValueError("evasion start_s must be finite and nonnegative")
        if not math.isfinite(self.max_turn_g) or self.max_turn_g <= 0:
            raise ValueError("evasion max_turn_g must be finite and positive")


class EvasiveTarget:
    def __init__(self, base: Any, plan: EvasionPlan, gravity_mps2: float):
        self.base = base
        self.plan = plan
        self.gravity_mps2 = gravity_mps2
        self.initial_state = base.state_at(0.0)
        self._missile: tuple[float, float, float] | None = None
        self._beam_sign: float | None = None
        self._times = [plan.start_s]
        self._states = [base.state_at(plan.start_s)]

    def observe_missile(self, time_s: float, position, velocity) -> None:
        self._missile = tuple(position)

    def describe(self) -> dict[str, Any]:
        return {
            "kind": self.plan.kind,
            "start_s": self.plan.start_s,
            "max_turn_g": self.plan.max_turn_g,
            "model": "kinematic_constant_speed_horizontal_turn",
        }

    def state_at(self, time_s: float) -> TargetState:
        if time_s <= self.plan.start_s:
            return self.base.state_at(time_s)
        while self._times[-1] < time_s - 1e-12:
            self._advance()
        index = bisect.bisect_left(self._times, time_s - 1e-12)
        if abs(self._times[index] - time_s) <= 1e-12:
            return self._states[index]
        t0, t1 = self._times[index - 1], self._times[index]
        s0, s1 = self._states[index - 1], self._states[index]
        w = (time_s - t0) / (t1 - t0)
        return TargetState(
            tuple(a + w * (b - a) for a, b in zip(s0.position, s1.position)),
            tuple(a + w * (b - a) for a, b in zip(s0.velocity, s1.velocity)),
        )

    def _desired_heading(self, position, velocity) -> float | None:
        if self._missile is None:
            return None
        los = (position[0] - self._missile[0], position[2] - self._missile[2])
        if math.hypot(*los) <= 1e-9:
            return None
        away = math.atan2(los[1], los[0])
        if self.plan.kind == "drag":
            return away
        if self._beam_sign is None:
            # Lock the beam side nearest the current heading at the first step.
            cross = los[0] * velocity[2] - los[1] * velocity[0]
            self._beam_sign = 1.0 if cross >= 0 else -1.0
        return away + self._beam_sign * math.pi / 2

    def _advance(self) -> None:
        h = SUBSTEP_S
        state = self._states[-1]
        p, v = state.position, state.velocity
        speed = math.hypot(v[0], v[2])
        desired = self._desired_heading(p, v)
        new_v = v
        if desired is not None and speed > 1e-9:
            error = desired - math.atan2(v[2], v[0])
            error = (error + math.pi) % (2 * math.pi) - math.pi
            limit = self.plan.max_turn_g * self.gravity_mps2 / speed * h
            turn = max(-limit, min(limit, error))
            c, s = math.cos(turn), math.sin(turn)
            new_v = (c * v[0] - s * v[2], v[1], s * v[0] + c * v[2])
        mid = tuple((a + b) / 2 for a, b in zip(v, new_v))
        self._times.append(self._times[-1] + h)
        self._states.append(TargetState(tuple(a + b * h for a, b in zip(p, mid)), new_v))


def evasive_target_factory(plan: EvasionPlan, gravity_mps2: float = 9.80665) -> Callable[[Any], EvasiveTarget]:
    return lambda base: EvasiveTarget(base, plan, gravity_mps2)


__all__ = ["EvasionPlan", "EvasiveTarget", "KINDS", "evasive_target_factory"]
