"""Axial speed differentiation and effective drag-area inversion for H3."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .sample_filters import apply_filter


def _solve_linear(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Small Gaussian-elimination solver with partial pivoting."""

    size = len(vector)
    augmented = [list(row) + [vector[index]] for index, row in enumerate(matrix)]
    for pivot_index in range(size):
        pivot_row = max(
            range(pivot_index, size), key=lambda row: abs(augmented[row][pivot_index])
        )
        if abs(augmented[pivot_row][pivot_index]) <= 1.0e-14:
            raise ValueError("singular local polynomial system")
        if pivot_row != pivot_index:
            augmented[pivot_index], augmented[pivot_row] = augmented[pivot_row], augmented[pivot_index]
        pivot = augmented[pivot_index][pivot_index]
        for column in range(pivot_index, size + 1):
            augmented[pivot_index][column] /= pivot
        for row in range(size):
            if row == pivot_index:
                continue
            factor = augmented[row][pivot_index]
            if abs(factor) <= 1.0e-18:
                continue
            for column in range(pivot_index, size + 1):
                augmented[row][column] -= factor * augmented[pivot_index][column]
    return [augmented[index][size] for index in range(size)]


def _local_polynomial_fit(
    points: Sequence[tuple[float, float]],
    center_time_s: float,
    window_s: float,
    polynomial_order: int = 2,
) -> tuple[float, float]:
    selected = [
        (time_s, speed_mps)
        for time_s, speed_mps in points
        if abs(time_s - center_time_s) <= window_s + 1.0e-12
    ]
    if len(selected) < 2:
        raise ValueError("not enough points for local derivative")
    degree = min(int(polynomial_order), len(selected) - 1)
    # Centering time makes the normal equations much better conditioned near
    # the high-Mach end while keeping the first coefficient as the derivative.
    normal = [[0.0 for _ in range(degree + 1)] for _ in range(degree + 1)]
    rhs = [0.0 for _ in range(degree + 1)]
    safe_window = max(float(window_s), 1.0e-9)
    for time_s, speed_mps in selected:
        x = (time_s - center_time_s) / safe_window
        weight = 1.0 / (1.0 + x * x)
        powers = [1.0]
        for _power in range(2 * degree):
            powers.append(powers[-1] * x)
        for row in range(degree + 1):
            rhs[row] += weight * speed_mps * powers[row]
            for column in range(degree + 1):
                normal[row][column] += weight * powers[row + column]
    coefficients = _solve_linear(normal, rhs)
    return coefficients[0], coefficients[1] / safe_window if degree >= 1 else 0.0


def _local_polynomial_derivative(
    points: Sequence[tuple[float, float]],
    center_time_s: float,
    window_s: float,
    polynomial_order: int = 2,
) -> float:
    return _local_polynomial_fit(points, center_time_s, window_s, polynomial_order)[1]


def _local_polynomial_value(
    points: Sequence[tuple[float, float]],
    center_time_s: float,
    window_s: float,
    polynomial_order: int = 2,
) -> float:
    return _local_polynomial_fit(points, center_time_s, window_s, polynomial_order)[0]


def _segment_key(row: Mapping[str, Any]) -> tuple[bool, int]:
    return bool(row.get("powered", False)), int(row.get("engine_stage", 0))


def estimate_speed_derivatives(
    rows: Sequence[Mapping[str, Any]],
    window_s: float,
    polynomial_order: int = 2,
) -> list[float]:
    """Estimate dV/dt without smoothing across engine stage boundaries.

    Rows are grouped by powered/coast state and engine stage.  A window can
    therefore become one-sided near a segment edge, but it never blends the
    1.7 s or 7.0 s transitions into the derivative.
    """

    groups: dict[tuple[bool, int], list[tuple[float, float, int]]] = {}
    for index, row in enumerate(rows):
        time_s = float(row.get("time_s", float("nan")))
        speed_mps = float(row.get("speed_mps", float("nan")))
        if not (math.isfinite(time_s) and math.isfinite(speed_mps)):
            continue
        groups.setdefault(_segment_key(row), []).append((time_s, speed_mps, index))

    derivatives = [float("nan") for _ in rows]
    for group_rows in groups.values():
        group_rows.sort(key=lambda item: item[0])
        points = [(time_s, speed_mps) for time_s, speed_mps, _index in group_rows]
        for time_s, _speed_mps, index in group_rows:
            try:
                derivatives[index] = _local_polynomial_derivative(
                    points,
                    time_s,
                    window_s,
                    polynomial_order,
                )
            except ValueError:
                # A two-point one-sided finite difference is preferable to
                # silently dropping an otherwise useful low-g sample.
                position = next(
                    position
                    for position, item in enumerate(group_rows)
                    if item[2] == index
                )
                if len(group_rows) >= 2:
                    if position == 0:
                        left, right = group_rows[0], group_rows[1]
                    else:
                        left, right = group_rows[position - 1], group_rows[position]
                    delta_t = right[0] - left[0]
                    derivatives[index] = (
                        (right[1] - left[1]) / delta_t if abs(delta_t) > 1.0e-15 else float("nan")
                    )
    return derivatives


def estimate_smoothed_speeds(
    rows: Sequence[Mapping[str, Any]],
    window_s: float,
    polynomial_order: int = 2,
) -> list[float]:
    """Return local-polynomial speed estimates using the same boundaries as dV/dt."""

    groups: dict[tuple[bool, int], list[tuple[float, float, int]]] = {}
    for index, row in enumerate(rows):
        time_s = float(row.get("time_s", float("nan")))
        speed_mps = float(row.get("speed_mps", float("nan")))
        if not (math.isfinite(time_s) and math.isfinite(speed_mps)):
            continue
        groups.setdefault(_segment_key(row), []).append((time_s, speed_mps, index))

    smoothed = [float("nan") for _ in rows]
    for group_rows in groups.values():
        group_rows.sort(key=lambda item: item[0])
        points = [(time_s, speed_mps) for time_s, speed_mps, _index in group_rows]
        for time_s, speed_mps, index in group_rows:
            try:
                smoothed[index] = _local_polynomial_value(
                    points,
                    time_s,
                    window_s,
                    polynomial_order,
                )
            except ValueError:
                smoothed[index] = speed_mps
    return smoothed


def inverse_drag_sample(
    sample: Mapping[str, Any],
    speed_derivative_mps2: float,
    gravity_mps2: float = 9.80665,
    settings: Any | None = None,
) -> dict[str, Any]:
    """Invert the axial equation into observed drag and effective CdA."""

    result = dict(sample)
    result["speed_derivative_mps2"] = float(speed_derivative_mps2)
    if settings is not None:
        result = apply_filter(result, settings)
    else:
        result.setdefault("accepted", True)
        result.setdefault("rejection_reasons", [])

    reasons = list(result.get("rejection_reasons", []))
    mass = float(result.get("mass_kg", float("nan")))
    q_pa = float(result.get("dynamic_pressure_pa", float("nan")))
    gamma_rad = float(result.get("flight_path_angle_rad", float("nan")))
    if not math.isfinite(speed_derivative_mps2):
        reasons.append("speed_derivative_non_finite")
    if not math.isfinite(mass) or mass <= 0.0:
        reasons.append("mass_non_positive")
    if not math.isfinite(q_pa) or q_pa <= 0.0:
        reasons.append("dynamic_pressure_non_positive")
    if not math.isfinite(gamma_rad):
        reasons.append("flight_path_angle_non_finite")

    powered = bool(result.get("powered", False))
    thrust_n = float(result.get("thrust_n", 0.0)) if powered else 0.0
    observed_drag_n = float("nan")
    observed_cda_m2 = float("nan")
    if not reasons:
        observed_drag_n = thrust_n - mass * (
            float(speed_derivative_mps2) + float(gravity_mps2) * math.sin(gamma_rad)
        )
        if not math.isfinite(observed_drag_n):
            reasons.append("observed_drag_non_finite")
        elif observed_drag_n <= 0.0:
            reasons.append("observed_drag_non_positive")
        else:
            observed_cda_m2 = observed_drag_n / q_pa
            if not math.isfinite(observed_cda_m2) or observed_cda_m2 <= 0.0:
                reasons.append("observed_cda_non_positive")

    result["observed_drag_n"] = observed_drag_n
    result["observed_cda_m2"] = observed_cda_m2
    result["accepted"] = not reasons
    result["rejection_reasons"] = sorted(set(str(reason) for reason in reasons))
    return result


def invert_rows(
    rows: Sequence[Mapping[str, Any]],
    derivatives_by_window: Mapping[str, Sequence[float]],
    primary_window_s: float,
    gravity_mps2: float = 9.80665,
    settings: Any | None = None,
) -> list[dict[str, Any]]:
    """Build inverse rows and retain derivative estimates for every window."""

    key = str(primary_window_s)
    if key not in derivatives_by_window:
        key = next(iter(derivatives_by_window))
        primary_window_s = float(key)
    primary = derivatives_by_window[key]
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        inverse = inverse_drag_sample(
            row,
            float(primary[index]),
            gravity_mps2=gravity_mps2,
            settings=settings,
        )
        inverse["speed_derivative_mps2_by_window"] = {
            str(window): float(values[index])
            for window, values in derivatives_by_window.items()
        }
        inverse["observed_cda_m2_by_window"] = {}
        for window, values in derivatives_by_window.items():
            candidate = inverse_drag_sample(
                row,
                float(values[index]),
                gravity_mps2=gravity_mps2,
                settings=settings,
            )
            inverse["observed_cda_m2_by_window"][str(window)] = (
                candidate["observed_cda_m2"]
                if candidate.get("accepted")
                else None
            )
        result.append(inverse)
    return result


__all__ = [
    "estimate_speed_derivatives",
    "estimate_smoothed_speeds",
    "inverse_drag_sample",
    "invert_rows",
]
