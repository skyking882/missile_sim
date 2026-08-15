"""Positive log-knot effective drag-area model for H4 glide analysis."""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class LogCdaEnvelope:
    """Piecewise-linear interpolation in log(CdA), with explicit boundaries."""

    mach_knots: tuple[float, ...]
    log_cda_knots: tuple[float, ...]
    interpolation: str = "log_linear"
    outside_support_policy: str = "endpoint_hold_labeled_extrapolation"

    def __post_init__(self) -> None:
        if self.interpolation != "log_linear":
            raise ValueError("H4 first-pass interpolation must be log_linear")
        if len(self.mach_knots) != len(self.log_cda_knots) or len(self.mach_knots) < 2:
            raise ValueError("mach and log(CdA) knots must have equal length >= 2")
        previous = None
        for mach, log_cda in zip(self.mach_knots, self.log_cda_knots):
            if not math.isfinite(float(mach)) or not math.isfinite(float(log_cda)):
                raise ValueError("all knots must be finite")
            if previous is not None and float(mach) <= previous:
                raise ValueError("Mach knots must be strictly increasing")
            previous = float(mach)

    @classmethod
    def from_cda_knots(
        cls,
        mach_knots: Sequence[float],
        cda_knots_m2: Sequence[float],
        interpolation: str = "log_linear",
    ) -> "LogCdaEnvelope":
        logs: list[float] = []
        for cda in cda_knots_m2:
            value = float(cda)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("CdA knot values must be finite and positive")
            logs.append(math.log(value))
        return cls(tuple(float(value) for value in mach_knots), tuple(logs), interpolation)

    def support_range(self) -> tuple[float, float]:
        return self.mach_knots[0], self.mach_knots[-1]

    def _log_cda_at(self, mach: float) -> tuple[float, str]:
        value = float(mach)
        if not math.isfinite(value):
            raise ValueError("Mach must be finite")
        lower, upper = self.support_range()
        if value < lower:
            return self.log_cda_knots[0], "extrapolation"
        if value > upper:
            return self.log_cda_knots[-1], "extrapolation"
        index = bisect.bisect_right(self.mach_knots, value) - 1
        if index >= len(self.mach_knots) - 1:
            return self.log_cda_knots[-1], "direct_support"
        if abs(value - self.mach_knots[index]) <= 1.0e-12:
            return self.log_cda_knots[index], "direct_support"
        left_mach = self.mach_knots[index]
        right_mach = self.mach_knots[index + 1]
        fraction = (value - left_mach) / (right_mach - left_mach)
        log_cda = self.log_cda_knots[index] + fraction * (
            self.log_cda_knots[index + 1] - self.log_cda_knots[index]
        )
        return log_cda, "interpolation"

    def evaluate(self, mach: float) -> dict[str, float | str]:
        log_cda, model_label = self._log_cda_at(mach)
        cda = math.exp(log_cda)
        if not math.isfinite(cda) or cda <= 0.0:
            raise ValueError("evaluated CdA must be finite and positive")
        return {
            "mach": float(mach),
            "log_cda": float(log_cda),
            "cda_m2": float(cda),
            "model_support_label": model_label,
        }

    def cda_m2(self, mach: float) -> float:
        return float(self.evaluate(mach)["cda_m2"])

    def evaluate_grid(self, mach_values: Iterable[float]) -> list[dict[str, float | str]]:
        return [self.evaluate(value) for value in mach_values]


def merge_intervals(intervals: Iterable[tuple[float, float]], tolerance: float = 1.0e-12) -> list[tuple[float, float]]:
    """Merge overlapping Mach support intervals without inventing coverage."""

    ordered = sorted((min(float(a), float(b)), max(float(a), float(b))) for a, b in intervals)
    merged: list[list[float]] = []
    for lower, upper in ordered:
        if not merged or lower > merged[-1][1] + tolerance:
            merged.append([lower, upper])
        else:
            merged[-1][1] = max(merged[-1][1], upper)
    return [(lower, upper) for lower, upper in merged]


def classify_mach_support(
    mach: float,
    direct_intervals: Sequence[tuple[float, float]],
    target_range: tuple[float, float] = (0.2, 4.5),
    direct_margin: float = 0.0,
) -> str:
    """Classify a Mach value using measured direct-support intervals."""

    value = float(mach)
    target_min, target_max = target_range
    if value < target_min or value > target_max:
        return "extrapolation"
    for lower, upper in direct_intervals:
        if lower - direct_margin <= value <= upper + direct_margin:
            return "direct_support"
    merged = merge_intervals(direct_intervals)
    if not merged:
        return "extrapolation"
    if merged[0][0] <= value <= merged[-1][1]:
        return "interpolation"
    return "extrapolation"


def cda_physical_checks(envelope: LogCdaEnvelope, mach_min: float = 0.2, mach_max: float = 4.5, step: float = 0.05) -> dict[str, object]:
    """Check positivity and finite values over a requested grid."""

    values: list[dict[str, float | str]] = []
    count = int(round((mach_max - mach_min) / step)) + 1
    for index in range(count):
        values.append(envelope.evaluate(mach_min + index * step))
    violations = [item for item in values if float(item["cda_m2"]) <= 0.0 or not math.isfinite(float(item["cda_m2"]))]
    return {
        "mach_min": mach_min,
        "mach_max": mach_max,
        "grid_step": step,
        "grid_count": len(values),
        "positive_and_finite": not violations,
        "violation_count": len(violations),
        "min_cda_m2": min(float(item["cda_m2"]) for item in values),
        "max_cda_m2": max(float(item["cda_m2"]) for item in values),
    }


def balanced_trajectory_weights(rows: Sequence[Mapping[str, object]]) -> dict[str, float]:
    """Return equal mean weight per trajectory for future fit objectives."""

    counts: dict[str, int] = {}
    for row in rows:
        trajectory_id = str(row.get("trajectory_id", row.get("case_id", "unknown")))
        counts[trajectory_id] = counts.get(trajectory_id, 0) + 1
    return {trajectory_id: 1.0 / max(count, 1) for trajectory_id, count in counts.items()}


__all__ = [
    "LogCdaEnvelope",
    "balanced_trajectory_weights",
    "cda_physical_checks",
    "classify_mach_support",
    "merge_intervals",
]
