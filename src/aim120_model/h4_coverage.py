"""Coverage, overlap, and low-speed cancellation audits for H4."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from .glide_drag_envelope import classify_mach_support, merge_intervals


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def trajectory_ranges(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    times: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        trajectory_id = str(row.get("trajectory_id", row.get("case_id", "unknown")))
        if _finite(row.get("mach")):
            grouped[trajectory_id].append(float(row["mach"]))
        if _finite(row.get("time_s")):
            times[trajectory_id].append(float(row["time_s"]))
    result: dict[str, dict[str, float | int | None]] = {}
    for trajectory_id in sorted(set(grouped) | set(times)):
        mach_values = grouped.get(trajectory_id, [])
        time_values = times.get(trajectory_id, [])
        result[trajectory_id] = {
            "mach_min": min(mach_values) if mach_values else None,
            "mach_max": max(mach_values) if mach_values else None,
            "sample_count": len(mach_values),
            "time_min_s": min(time_values) if time_values else None,
            "time_max_s": max(time_values) if time_values else None,
        }
    return result


def direct_intervals_from_trajectories(rows: Iterable[Mapping[str, Any]]) -> list[tuple[float, float]]:
    return [
        (float(item["mach_min"]), float(item["mach_max"]))
        for item in trajectory_ranges(rows).values()
        if item["mach_min"] is not None and item["mach_max"] is not None
    ]


def adjacent_overlap_report(rows: Iterable[Mapping[str, Any]], minimum_width_mach: float = 0.3) -> list[dict[str, Any]]:
    ranges = trajectory_ranges(rows)
    ordered = sorted(
        (
            trajectory_id,
            float(item["mach_min"]),
            float(item["mach_max"]),
        )
        for trajectory_id, item in ranges.items()
        if item["mach_min"] is not None and item["mach_max"] is not None
    )
    comparisons: list[dict[str, Any]] = []
    for left, right in zip(ordered, ordered[1:]):
        overlap_min = max(left[1], right[1])
        overlap_max = min(left[2], right[2])
        width = max(0.0, overlap_max - overlap_min)
        comparisons.append({
            "left_trajectory_id": left[0],
            "right_trajectory_id": right[0],
            "overlap_min_mach": overlap_min if width > 0.0 else None,
            "overlap_max_mach": overlap_max if width > 0.0 else None,
            "overlap_width_mach": width,
            "meets_target_width": width >= minimum_width_mach,
        })
    return comparisons


def missing_target_ranges(
    direct_intervals: Sequence[tuple[float, float]],
    target_range: tuple[float, float] = (0.2, 4.5),
) -> list[tuple[float, float]]:
    target_min, target_max = target_range
    clipped = [
        (max(target_min, lower), min(target_max, upper))
        for lower, upper in direct_intervals
        if upper >= target_min and lower <= target_max
    ]
    merged = merge_intervals(clipped)
    missing: list[tuple[float, float]] = []
    cursor = target_min
    for lower, upper in merged:
        if lower > cursor:
            missing.append((cursor, lower))
        cursor = max(cursor, upper)
    if cursor < target_max:
        missing.append((cursor, target_max))
    return missing


def support_labels_for_rows(
    rows: Sequence[Mapping[str, Any]],
    target_range: tuple[float, float] = (0.2, 4.5),
    direct_margin: float = 0.0,
) -> list[dict[str, Any]]:
    intervals = direct_intervals_from_trajectories(rows)
    result: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["support_label"] = (
            classify_mach_support(float(row["mach"]), intervals, target_range, direct_margin)
            if _finite(row.get("mach"))
            else "invalid"
        )
        result.append(item)
    return result


def gravity_cancellation_audit(rows: Iterable[Mapping[str, Any]], gravity_mps2: float = 9.80665) -> dict[str, Any]:
    ratios: list[float] = []
    by_trajectory: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        gamma = row.get("flight_path_angle_rad")
        if gamma is None and _finite(row.get("flight_path_angle_deg")):
            gamma = math.radians(float(row["flight_path_angle_deg"]))
        drag_accel = row.get("axial_drag_accel_mps2")
        if not _finite(gamma) or not _finite(drag_accel):
            continue
        gravity_axial = float(gravity_mps2) * math.sin(float(gamma))
        ratio = abs(gravity_axial) / max(abs(float(drag_accel)), 1.0e-9)
        ratios.append(ratio)
        trajectory_id = str(row.get("trajectory_id", row.get("case_id", "unknown")))
        by_trajectory[trajectory_id].append(ratio)
    thresholds = (1.0, 3.0, 5.0)
    return {
        "sample_count": len(ratios),
        "mean_ratio": mean(ratios) if ratios else None,
        "max_ratio": max(ratios) if ratios else None,
        "fraction_above_threshold": {
            str(threshold): (
                sum(1 for value in ratios if value > threshold) / len(ratios) if ratios else None
            )
            for threshold in thresholds
        },
        "by_trajectory": {
            trajectory_id: {
                "sample_count": len(values),
                "mean_ratio": mean(values),
                "max_ratio": max(values),
            }
            for trajectory_id, values in by_trajectory.items()
        },
    }


def coverage_report(
    rows: Sequence[Mapping[str, Any]],
    target_range: tuple[float, float] = (0.2, 4.5),
    minimum_overlap_width_mach: float = 0.3,
) -> dict[str, Any]:
    ranges = trajectory_ranges(rows)
    intervals = direct_intervals_from_trajectories(rows)
    merged = merge_intervals(intervals)
    return {
        "target_range_mach": {"min": target_range[0], "max": target_range[1]},
        "trajectory_count": len(ranges),
        "trajectory_ranges": ranges,
        "direct_intervals": intervals,
        "merged_direct_intervals": merged,
        "missing_target_ranges": missing_target_ranges(intervals, target_range),
        "actual_direct_support_range": {
            "min": min((item[0] for item in merged), default=None),
            "max": max((item[1] for item in merged), default=None),
        },
        "adjacent_overlap": adjacent_overlap_report(rows, minimum_overlap_width_mach),
        "support_status": "reference_data_present" if rows else "blocked_missing_statshark_reference_time_series",
    }


__all__ = [
    "adjacent_overlap_report",
    "coverage_report",
    "direct_intervals_from_trajectories",
    "gravity_cancellation_audit",
    "missing_target_ranges",
    "support_labels_for_rows",
    "trajectory_ranges",
]
