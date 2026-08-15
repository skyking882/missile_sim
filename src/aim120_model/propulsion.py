"""Piecewise-constant propulsion and mass-flow model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PropulsionStage:
    name: str
    duration_s: float
    thrust_n: float
    mass_lost_kg: float
    isp_s: float

    @property
    def mass_flow_kg_s(self) -> float:
        if self.duration_s <= 0.0:
            return 0.0
        return -self.mass_lost_kg / self.duration_s


@dataclass(frozen=True)
class PropulsionSample:
    thrust_n: float
    mass_flow_kg_s: float
    mass_kg: float
    stage_name: str | None


class PiecewisePropulsion:
    def __init__(self, initial_mass_kg: float, stages: list[PropulsionStage]):
        if initial_mass_kg <= 0.0:
            raise ValueError("initial mass must be positive")
        if not stages:
            raise ValueError("at least one propulsion stage is required")
        self.initial_mass_kg = float(initial_mass_kg)
        self.stages = tuple(stages)
        self._ends: list[float] = []
        total = 0.0
        for stage in self.stages:
            if stage.duration_s <= 0.0:
                raise ValueError(f"stage {stage.name} has non-positive duration")
            total += stage.duration_s
            self._ends.append(total)
        self.burn_time_s = total

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "PiecewisePropulsion":
        geometry = config["geometry"]
        stages = [PropulsionStage(**stage) for stage in config["propulsion"]["stages"]]
        return cls(geometry["initial_mass_kg"], stages)

    def _mass_lost_before(self, index: int) -> float:
        return sum(stage.mass_lost_kg for stage in self.stages[:index])

    def mass_at(self, time_s: float, powered: bool = True) -> float:
        if not powered:
            return self.initial_mass_kg
        t = max(0.0, float(time_s))
        previous_end = 0.0
        lost = 0.0
        for stage, end in zip(self.stages, self._ends):
            if t < end:
                lost += stage.mass_lost_kg * max(0.0, t - previous_end) / stage.duration_s
                return self.initial_mass_kg - lost
            lost += stage.mass_lost_kg
            previous_end = end
        return self.initial_mass_kg - lost

    def sample(self, time_s: float, powered: bool = True) -> PropulsionSample:
        if not powered:
            return PropulsionSample(0.0, 0.0, self.initial_mass_kg, None)
        t = max(0.0, float(time_s))
        previous_end = 0.0
        for stage, end in zip(self.stages, self._ends):
            if t < end:
                return PropulsionSample(
                    stage.thrust_n,
                    stage.mass_flow_kg_s,
                    self.mass_at(t, powered=True),
                    stage.name,
                )
            previous_end = end
        return PropulsionSample(0.0, 0.0, self.mass_at(t, powered=True), None)

    def next_boundary_after(self, time_s: float) -> float | None:
        t = float(time_s)
        for end in self._ends:
            if end > t + 1e-12:
                return end
        return None

