"""Inverse effective CdA diagnostics for unpowered, low-g trajectories."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import median
from typing import Any, Mapping, Sequence

from .atmosphere import StandardAtmosphere


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _local_slope(rows: Sequence[Mapping[str, Any]], index: int, window_points: int) -> tuple[float | None, int]:
    half = max(1, int(window_points) // 2)
    start = max(0, index - half)
    stop = min(len(rows), index + half + 1)
    window = rows[start:stop]
    if len(window) < 3:
        return None, len(window)
    times = [float(row["time_s"]) for row in window]
    speeds = [float(row["speed_mps"]) for row in window]
    if any(not math.isfinite(value) for value in times + speeds):
        return None, len(window)
    if any(right <= left for left, right in zip(times, times[1:])):
        return None, len(window)
    if any(right - left > 4.5 for left, right in zip(times, times[1:])):
        return None, len(window)
    time_mean = sum(times) / len(times)
    speed_mean = sum(speeds) / len(speeds)
    denominator = sum((time - time_mean) ** 2 for time in times)
    if denominator <= 1.0e-12:
        return None, len(window)
    slope = sum((time - time_mean) * (speed - speed_mean) for time, speed in zip(times, speeds)) / denominator
    return slope, len(window)


def estimate_inverse_cda(
    rows: Sequence[Mapping[str, Any]],
    atmosphere: StandardAtmosphere | None = None,
    gravity_mps2: float = 9.80665,
    window_points: int = 5,
) -> list[dict[str, Any]]:
    """Estimate CdA from m*dV/dt = -q*CdA - m*g*sin(gamma).

    The visible StatShark speed is quantized, so a local least-squares slope is
    used instead of a single two-point difference.  This is an inverse
    diagnostic, not proof of the site solver's internal equations.
    """

    atmosphere_model = atmosphere or StandardAtmosphere()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        trajectory_id = str(row.get("trajectory_id", row.get("case_id", "unknown")))
        grouped[trajectory_id].append(dict(row))
    result: list[dict[str, Any]] = []
    for trajectory_id, trajectory_rows in grouped.items():
        ordered = sorted(trajectory_rows, key=lambda item: float(item.get("time_s", float("nan"))))
        for index, row in enumerate(ordered):
            item = dict(row)
            item["trajectory_id"] = trajectory_id
            slope, window_count = _local_slope(ordered, index, window_points)
            item["inverse_derivative_window_samples"] = window_count
            item["dv_dt_mps2"] = slope
            item["inverse_cda_valid"] = False
            item["inverse_cda_reason"] = None
            if slope is None:
                item["inverse_cda_reason"] = "insufficient_or_nonuniform_time_window"
                result.append(item)
                continue
            speed = float(item.get("speed_mps", float("nan")))
            altitude = float(item.get("altitude_m", float("nan")))
            mass = float(item.get("mass_kg", float("nan")))
            gamma = float(item.get("flight_path_angle_rad", float("nan")))
            if not all(math.isfinite(value) for value in (speed, altitude, mass, gamma)) or speed <= 0.0 or mass <= 0.0:
                item["inverse_cda_reason"] = "non_finite_or_non_positive_state"
                result.append(item)
                continue
            atmosphere_sample = atmosphere_model.sample(altitude)
            dynamic_pressure = 0.5 * atmosphere_sample.density_kg_m3 * speed * speed
            gravity_axial = float(gravity_mps2) * math.sin(gamma)
            axial_drag_accel = -(slope + gravity_axial)
            drag_force = mass * axial_drag_accel
            inverse_cda = drag_force / dynamic_pressure if dynamic_pressure > 0.0 else float("nan")
            item.update({
                "dynamic_pressure_pa": dynamic_pressure,
                "gravity_axial_mps2": gravity_axial,
                "axial_drag_accel_mps2": axial_drag_accel,
                "drag_force_n": drag_force,
                "inverse_cda_m2": inverse_cda,
            })
            if not math.isfinite(dynamic_pressure) or dynamic_pressure <= 1000.0:
                item["inverse_cda_reason"] = "dynamic_pressure_below_inverse_floor"
            elif not math.isfinite(inverse_cda) or inverse_cda <= 0.0:
                item["inverse_cda_reason"] = "non_positive_inverse_cda"
            else:
                item["inverse_cda_valid"] = True
            result.append(item)
    return sorted(result, key=lambda item: (str(item.get("trajectory_id")), float(item.get("time_s", 0.0))))


def summarize_inverse_cda(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("trajectory_id", row.get("case_id", "unknown")))].append(row)
    summary: dict[str, Any] = {}
    for trajectory_id, values in sorted(grouped.items()):
        valid = [float(row["inverse_cda_m2"]) for row in values if row.get("inverse_cda_valid") and _finite(row.get("inverse_cda_m2"))]
        mach = [float(row["mach"]) for row in values if row.get("inverse_cda_valid") and _finite(row.get("mach"))]
        summary[trajectory_id] = {
            "sample_count": len(values),
            "valid_inverse_sample_count": len(valid),
            "mach_min": min(mach) if mach else None,
            "mach_max": max(mach) if mach else None,
            "inverse_cda_median_m2": median(valid) if valid else None,
            "inverse_cda_min_m2": min(valid) if valid else None,
            "inverse_cda_max_m2": max(valid) if valid else None,
        }
    return summary


def fit_log_cda_knots(
    rows: Sequence[Mapping[str, Any]],
    mach_knots: Sequence[float],
    minimum_samples_per_node: int = 3,
) -> dict[str, Any]:
    """Fit robust positive nodes using per-trajectory medians in Mach cells."""

    knots = sorted(float(value) for value in mach_knots)
    if len(knots) < 2 or any(right <= left for left, right in zip(knots, knots[1:])):
        raise ValueError("mach_knots must be strictly increasing and contain at least two values")
    valid = [
        row for row in rows
        if row.get("inverse_cda_valid") and _finite(row.get("mach")) and _finite(row.get("inverse_cda_m2")) and float(row["inverse_cda_m2"]) > 0.0
    ]
    node_records: list[dict[str, Any]] = []
    direct_knots: list[float] = []
    cda_values: list[float] = []
    for index, knot in enumerate(knots):
        lower = -float("inf") if index == 0 else (knots[index - 1] + knot) / 2.0
        upper = float("inf") if index == len(knots) - 1 else (knot + knots[index + 1]) / 2.0
        cell = [row for row in valid if lower <= float(row["mach"]) < upper]
        by_trajectory: dict[str, list[float]] = defaultdict(list)
        for row in cell:
            by_trajectory[str(row.get("trajectory_id", row.get("case_id", "unknown")))].append(float(row["inverse_cda_m2"]))
        trajectory_medians = {key: median(values) for key, values in sorted(by_trajectory.items())}
        pooled = [value for values in by_trajectory.values() for value in values]
        direct = len(pooled) >= int(minimum_samples_per_node)
        node_record = {
            "mach": knot,
            "cell_mach_min": None if not cell else min(float(row["mach"]) for row in cell),
            "cell_mach_max": None if not cell else max(float(row["mach"]) for row in cell),
            "sample_count": len(pooled),
            "trajectory_count": len(trajectory_medians),
            "trajectory_medians_m2": trajectory_medians,
            "cda_median_m2": median(list(trajectory_medians.values())) if trajectory_medians else None,
            "direct_node": direct,
        }
        node_records.append(node_record)
        if direct and node_record["cda_median_m2"] is not None:
            direct_knots.append(knot)
            cda_values.append(float(node_record["cda_median_m2"]))
    if len(direct_knots) < 2:
        raise ValueError("fewer than two stable inverse CdA nodes")
    return {
        "mach_knots": direct_knots,
        "cda_knots_m2": cda_values,
        "node_records": node_records,
        "valid_inverse_sample_count": len(valid),
        "valid_inverse_trajectory_count": len({str(row.get("trajectory_id", row.get("case_id", "unknown"))) for row in valid}),
    }


__all__ = ["estimate_inverse_cda", "fit_log_cda_knots", "summarize_inverse_cda"]
