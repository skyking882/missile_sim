"""Opt-in chaff decoys for the radar seeker (evidence level D unless noted).

Datamine (countermeasure_split_launcher_jet_bol.blkx, 2.59.0.27): timeLife 15 s,
startSpeed 70 m/s, CxK 100 on a 0.01 kg bundle, signature.radar.profile
[0.05, 0.1, 12, 15], multiplier0 0.05, isCloud. Assumed meanings (D):

* profile = RCS envelope times: rise 0.05->0.1 s, plateau to 12 s, fall to 0
  at 15 s (the fall end equals timeLife);
* CxK 100 stops the bundle relative to still air within ~0.1 s; it is
  modelled as exponential decay of the ejection-plus-carrier velocity;
* peak RCS relative to the aircraft is NOT recoverable from multiplier0
  without the aircraft's own signature, so ``rcs_ratio`` is a required input.

The seeker treats every bundle as an extra reflector; see
``RadarSeekerObserver.update(decoys=...)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .math3d import Vector, add, scale


@dataclass(frozen=True)
class Reflector:
    kind: str
    position: Vector
    velocity: Vector
    rcs_m2: float
    identity: object | None = None
    release_identity: object | None = None
    release_time_s: float | None = None
    release_position: Vector | None = None
    release_velocity: Vector | None = None


@dataclass(frozen=True)
class ChaffSpec:
    rcs_ratio: float                       # Peak bundle RCS / aircraft RCS (required assumption).
    lifetime_s: float = 15.0               # timeLife (A: datamine value; meaning assumed).
    eject_speed_mps: float = 70.0          # startSpeed.
    profile_s: tuple[float, float, float, float] = (0.05, 0.1, 12.0, 15.0)
    stop_time_constant_s: float = 0.1      # From CxK 100 (D).

    def __post_init__(self) -> None:
        values = (self.rcs_ratio, self.lifetime_s, self.eject_speed_mps, self.stop_time_constant_s, *self.profile_s)
        if not all(math.isfinite(v) for v in values) or self.rcs_ratio <= 0 or self.lifetime_s <= 0 \
                or self.stop_time_constant_s <= 0 or self.eject_speed_mps < 0:
            raise ValueError("invalid chaff spec")
        if list(self.profile_s) != sorted(self.profile_s):
            raise ValueError("chaff profile times must be nondecreasing")

    def envelope(self, age_s: float) -> float:
        a, b, c, d = self.profile_s
        if age_s < a or age_s >= min(d, self.lifetime_s):
            return 0.0
        if age_s < b:
            return (age_s - a) / (b - a) if b > a else 1.0
        if age_s <= c:
            return 1.0
        return (d - age_s) / (d - c) if d > c else 0.0


@dataclass(frozen=True)
class ChaffProgram:
    start_s: float
    interval_s: float = 1.0
    per_salvo: int = 1
    total: int = 30

    def __post_init__(self) -> None:
        if not (math.isfinite(self.start_s) and self.start_s >= 0 and math.isfinite(self.interval_s)
                and self.interval_s > 0 and self.per_salvo >= 1 and self.total >= 0):
            raise ValueError("invalid chaff program")

    def release_times(self) -> list[float]:
        times, k = [], 0
        while len(times) < self.total:
            times.extend([self.start_s + k * self.interval_s] * min(self.per_salvo, self.total - len(times)))
            k += 1
        return times


class ChaffingTarget:
    """Wrap any external target; releases bundles from its truth trajectory.

    Bundles are ejected straight down (missile_sim y is up) on top of the
    carrier velocity; the target itself is unchanged. ``rcs_m2`` of the
    aircraft is 1 so bundle RCS equals ``spec.rcs_ratio`` at the plateau.
    """

    rcs_m2 = 1.0

    def __init__(self, inner: Any, spec: ChaffSpec, program: ChaffProgram):
        self.inner, self.spec, self.program = inner, spec, program
        self.initial_state = getattr(inner, "initial_state", None)
        self._pending = program.release_times()
        self._bundles: list[tuple[float, Vector, Vector]] = []
        self.released = 0
        self.capture_mode = False

    def state_at(self, time_s: float):
        return self.inner.state_at(time_s)

    def observe_missile(self, time_s, position, velocity) -> None:
        observe = getattr(self.inner, "observe_missile", None)
        if observe is not None:
            observe(time_s, position, velocity)

    def describe(self) -> dict[str, Any]:
        inner = getattr(self.inner, "describe", None)
        base = inner() if inner is not None else {"model": type(self.inner).__name__}
        p, s = self.program, self.spec
        return {**base, "chaff": dict(start_s=p.start_s, interval_s=p.interval_s, per_salvo=p.per_salvo,
                                      total=p.total, released=self.released, rcs_ratio=s.rcs_ratio)}

    def decoys_at(self, time_s: float) -> list[Reflector]:
        while self._pending and self._pending[0] <= time_s + 1e-12:
            t = self._pending.pop(0)
            carrier = self.inner.state_at(t)
            velocity = add(tuple(carrier.velocity), (0.0, -self.spec.eject_speed_mps, 0.0))
            self._bundles.append((t, tuple(carrier.position), velocity))
            self.released += 1
        out = []
        tau = self.spec.stop_time_constant_s
        for index, (t0, p0, v0) in enumerate(self._bundles):
            age = time_s - t0
            rcs = self.spec.rcs_ratio * self.spec.envelope(age)
            if age < 0 or (rcs <= 0 and not (self.capture_mode and age < self.spec.lifetime_s)):
                continue
            decay = math.exp(-age / tau)
            out.append(Reflector("chaff", add(p0, scale(v0, tau * (1 - decay))), scale(v0, decay), rcs,
                                 identity=("chaff", index), release_identity=("chaff", t0),
                                 release_time_s=t0, release_position=p0,
                                 release_velocity=add(v0, (0.0, self.spec.eject_speed_mps, 0.0))))
        return out


def chaffing_factory(inner_factory, spec: ChaffSpec, program: ChaffProgram):
    def make(base):
        inner = inner_factory(base) if inner_factory is not None else base
        return ChaffingTarget(inner, spec, program)
    return make


__all__ = ["ChaffProgram", "ChaffSpec", "ChaffingTarget", "Reflector", "chaffing_factory"]
