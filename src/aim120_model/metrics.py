"""Terminal summaries and comparison against existing approximate references."""

from __future__ import annotations

import math
from typing import Any

from .math3d import norm, sub
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
