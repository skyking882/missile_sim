"""Terminal summaries and comparison against existing approximate references."""

from __future__ import annotations

import math
from typing import Any

from .math3d import dot, lerp, norm, sub
from .units import mps_to_kmh


def terminal_summary(result: dict[str, Any]) -> dict[str, Any]:
    terminal = result["samples"][-1]
    velocity = tuple(terminal["velocity_mps"])
    return {
        "event_type": result["event_type"],
        "terminal_time_s": terminal["time_s"],
        "terminal_speed_kmh": mps_to_kmh(norm(velocity)),
        "terminal_altitude_m": terminal["position_m"][1],
        "terminal_distance_to_target_m": terminal["distance_to_target_m"],
    }


def continuous_closest_approach(samples: list[dict[str, Any]]) -> dict[str, float | None]:
    """Closest point on piecewise-linear relative-position segments.

    Closing speed is positive when range is decreasing.
    """

    if not samples:
        return {
            "continuous_minimum_distance_m": None,
            "time_at_minimum_distance_s": None,
            "closing_speed_at_minimum_distance_mps": None,
        }
    best: tuple[float, float, float] | None = None
    for first, second in zip(samples, samples[1:]):
        r0 = sub(tuple(first["target_position_m"]), tuple(first["position_m"]))
        r1 = sub(tuple(second["target_position_m"]), tuple(second["position_m"]))
        delta = sub(r1, r0)
        denominator = dot(delta, delta)
        fraction = 0.0 if denominator <= 1e-18 else max(0.0, min(1.0, -dot(r0, delta) / denominator))
        relative = lerp(r0, r1, fraction)
        distance = norm(relative)
        time_s = float(first["time_s"]) + fraction * (
            float(second["time_s"]) - float(first["time_s"])
        )
        missile_velocity = lerp(tuple(first["velocity_mps"]), tuple(second["velocity_mps"]), fraction)
        target_velocity = lerp(tuple(first["target_velocity_mps"]), tuple(second["target_velocity_mps"]), fraction)
        relative_velocity = sub(target_velocity, missile_velocity)
        closing = 0.0 if distance <= 1e-12 else -dot(relative, relative_velocity) / distance
        candidate = (distance, time_s, closing)
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        sample = samples[0]
        return {
            "continuous_minimum_distance_m": float(sample["distance_to_target_m"]),
            "time_at_minimum_distance_s": float(sample["time_s"]),
            "closing_speed_at_minimum_distance_mps": float(sample.get("closing_speed_mps", 0.0)),
        }
    return {
        "continuous_minimum_distance_m": best[0],
        "time_at_minimum_distance_s": best[1],
        "closing_speed_at_minimum_distance_mps": best[2],
    }


def _reference_value(reference: dict[str, Any], key: str) -> float | None:
    value = reference.get(key)
    if not isinstance(value, dict) or "value" not in value:
        return None
    return float(value["value"])


def compare_result_to_reference(result: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    local = terminal_summary(result)
    reference_event_type = reference.get("event_type")
    reference_event_known = isinstance(reference_event_type, str) and bool(reference_event_type)
    comparison: dict[str, Any] = {
        "event_match": local["event_type"] == reference_event_type if reference_event_known else None,
        "reference_event_known": reference_event_known,
        "reference_event_label": reference.get("event_label"),
        "local": local,
        "reference": {
            "event_type": reference.get("event_type"),
            "terminal_time_s": reference.get("terminal_time_s"),
            "terminal_speed_kmh": reference.get("terminal_speed_kmh"),
            "terminal_altitude_m": reference.get("terminal_altitude_m"),
            "terminal_distance_to_target_m": reference.get("terminal_distance_to_target_m"),
        },
        "absolute_error": {},
        "reference_is_approximate": any(
            isinstance(reference.get(key), dict) and reference[key].get("approximate", False)
            for key in (
                "terminal_time_s",
                "terminal_speed_kmh",
                "terminal_altitude_m",
                "terminal_distance_to_target_m",
            )
        ),
    }
    for local_key, reference_key in (
        ("terminal_time_s", "terminal_time_s"),
        ("terminal_speed_kmh", "terminal_speed_kmh"),
        ("terminal_altitude_m", "terminal_altitude_m"),
        ("terminal_distance_to_target_m", "terminal_distance_to_target_m"),
    ):
        ref_value = _reference_value(reference, reference_key)
        if ref_value is not None:
            comparison["absolute_error"][local_key] = abs(local[local_key] - ref_value)
    return comparison
