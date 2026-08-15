"""Normalization and low-g sample selection for H3."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

from .atmosphere import StandardAtmosphere


@dataclass(frozen=True)
class LowGFilterSettings:
    lateral_load_threshold_g: float = 2.0
    alpha_threshold_deg: float = 5.0
    flight_path_threshold_deg: float = 5.0
    q_min_pa: float = 1000.0
    burn_stage_1_end_s: float = 1.7
    burn_end_s: float = 7.0
    stage_1_exclusion_window_s: float = 0.15
    burn_end_exclusion_window_s: float = 0.20

    def to_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


def vector_norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _first(raw: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in raw and raw[name] is not None:
            return raw[name]
    return default


def classify_engine_stage(time_s: float, powered: bool, settings: LowGFilterSettings) -> int:
    if not powered:
        return 0
    if float(time_s) < settings.burn_stage_1_end_s:
        return 1
    if float(time_s) < settings.burn_end_s:
        return 2
    return 0


def near_engine_boundary(time_s: float, settings: LowGFilterSettings) -> str | None:
    if abs(float(time_s) - settings.burn_stage_1_end_s) < settings.stage_1_exclusion_window_s:
        return "near_stage_1_boundary"
    if abs(float(time_s) - settings.burn_end_s) < settings.burn_end_exclusion_window_s:
        return "near_burn_end_boundary"
    return None


def normalize_sample(
    raw: Mapping[str, Any],
    settings: LowGFilterSettings | None = None,
    atmosphere: StandardAtmosphere | None = None,
    default_source_case: str = "unknown",
    default_source_kind: str = "unknown",
) -> dict[str, Any]:
    """Convert H2-like or plan3-like fields to one auditable sample schema."""

    cfg = settings or LowGFilterSettings()
    atm_model = atmosphere or StandardAtmosphere()
    time_s = float(_first(raw, "time_s", default=0.0))
    velocity_raw = _first(raw, "velocity_mps", "velocity_vector_mps")
    velocity: tuple[float, float, float] | None = None
    if isinstance(velocity_raw, (list, tuple)) and len(velocity_raw) == 3:
        velocity = tuple(float(value) for value in velocity_raw)
        speed_mps = vector_norm(velocity)
    else:
        speed_mps = float(_first(raw, "speed_mps", default=float("nan")))
    if not _finite(speed_mps):
        speed_mps = float("nan")

    position_raw = _first(raw, "position_m", "position_vector_m")
    position: tuple[float, float, float] | None = None
    if isinstance(position_raw, (list, tuple)) and len(position_raw) == 3:
        position = tuple(float(value) for value in position_raw)
    altitude_m = float(
        _first(
            raw,
            "altitude_m",
            default=position[1] if position is not None else float("nan"),
        )
    )

    powered_value = _first(raw, "powered")
    if powered_value is None:
        powered = float(_first(raw, "thrust_n", default=0.0)) > 1.0e-6
    else:
        powered = bool(powered_value)
    stage_raw = _first(raw, "engine_stage", "stage")
    stage = int(stage_raw) if stage_raw is not None else classify_engine_stage(time_s, powered, cfg)

    if not _finite(altitude_m):
        atmosphere_sample = None
        q_pa = float("nan")
        mach = float(_first(raw, "mach", default=float("nan")))
    else:
        atmosphere_sample = atm_model.sample(altitude_m)
        q_pa = float(_first(raw, "dynamic_pressure_pa", default=float("nan")))
        if not _finite(q_pa) and _finite(speed_mps):
            q_pa = 0.5 * atmosphere_sample.density_kg_m3 * speed_mps * speed_mps
        mach = float(_first(raw, "mach", default=float("nan")))
        if not _finite(mach) and atmosphere_sample.speed_of_sound_mps > 0.0:
            mach = speed_mps / atmosphere_sample.speed_of_sound_mps

    if velocity is not None and speed_mps > 1.0e-12:
        gamma_rad = math.asin(max(-1.0, min(1.0, velocity[1] / speed_mps)))
    else:
        gamma_raw = _first(raw, "flight_path_angle_rad")
        if gamma_raw is None:
            gamma_deg = _first(raw, "flight_path_angle_deg", default=float("nan"))
            gamma_rad = math.radians(float(gamma_deg)) if _finite(gamma_deg) else float("nan")
        else:
            gamma_rad = float(gamma_raw)

    alpha_raw = _first(raw, "angle_of_attack_rad", "alpha_rad")
    if alpha_raw is None:
        pitch_alpha = float(_first(raw, "pitch_alpha_rad", default=0.0))
        yaw_alpha = float(_first(raw, "yaw_alpha_rad", default=0.0))
        alpha_rad = math.hypot(pitch_alpha, yaw_alpha)
    else:
        alpha_rad = abs(float(alpha_raw))

    lateral_load_g = float(
        _first(raw, "lateral_load_g", "actual_overload_g", default=float("nan"))
    )
    mass_kg = float(_first(raw, "mass_kg", default=float("nan")))
    thrust_n = float(_first(raw, "thrust_n", default=0.0))
    horizontal_distance_m = float(_first(raw, "horizontal_distance_m", default=float("nan")))
    if not _finite(horizontal_distance_m) and position is not None:
        horizontal_distance_m = math.hypot(position[0], position[2])

    source_case = str(_first(raw, "source_case", default=default_source_case))
    source_kind = str(_first(raw, "source_kind", default=default_source_kind))
    local_model_cda = _first(raw, "total_cda_m2", "local_model_total_cda_m2")
    local_model_cda0 = _first(raw, "cda0_m2", "local_model_cda0_m2")

    normalized: dict[str, Any] = {
        "source_case": source_case,
        "source_kind": source_kind,
        "time_s": time_s,
        "powered": powered,
        "engine_stage": stage,
        "mass_kg": mass_kg,
        "thrust_n": thrust_n,
        "speed_mps": speed_mps,
        "velocity_mps": list(velocity) if velocity is not None else None,
        "mach": mach,
        "altitude_m": altitude_m,
        "horizontal_distance_m": horizontal_distance_m,
        "flight_path_angle_rad": gamma_rad,
        "flight_path_angle_deg": math.degrees(gamma_rad) if _finite(gamma_rad) else float("nan"),
        "alpha_rad": alpha_rad,
        "alpha_total_deg": math.degrees(alpha_rad) if _finite(alpha_rad) else float("nan"),
        "lateral_load_g": lateral_load_g,
        "dynamic_pressure_pa": q_pa,
        "local_model_total_cda_m2": float(local_model_cda) if _finite(local_model_cda) else None,
        "local_model_cda0_m2": float(local_model_cda0) if _finite(local_model_cda0) else None,
        "source_time_index": _first(raw, "source_time_index", default=None),
    }
    return normalized


def filter_reasons(sample: Mapping[str, Any], settings: LowGFilterSettings) -> list[str]:
    """Return every reason a row is not a default low-g inverse sample."""

    reasons: list[str] = []
    required = (
        "time_s",
        "mass_kg",
        "speed_mps",
        "mach",
        "altitude_m",
        "flight_path_angle_deg",
        "alpha_total_deg",
        "lateral_load_g",
        "dynamic_pressure_pa",
    )
    for name in required:
        if not _finite(sample.get(name)):
            reasons.append("non_finite_" + name)
    if _finite(sample.get("speed_mps")) and float(sample["speed_mps"]) <= 1.0e-9:
        reasons.append("speed_too_low")
    if _finite(sample.get("mass_kg")) and float(sample["mass_kg"]) <= 0.0:
        reasons.append("mass_non_positive")
    if _finite(sample.get("dynamic_pressure_pa")) and float(sample["dynamic_pressure_pa"]) <= settings.q_min_pa:
        reasons.append("dynamic_pressure_below_q_min")
    if _finite(sample.get("lateral_load_g")) and abs(float(sample["lateral_load_g"])) > settings.lateral_load_threshold_g:
        reasons.append("lateral_load_above_threshold")
    if _finite(sample.get("alpha_total_deg")) and abs(float(sample["alpha_total_deg"])) > settings.alpha_threshold_deg:
        reasons.append("alpha_above_threshold")
    if _finite(sample.get("flight_path_angle_deg")) and abs(float(sample["flight_path_angle_deg"])) > settings.flight_path_threshold_deg:
        reasons.append("flight_path_above_threshold")
    boundary_reason = near_engine_boundary(float(sample.get("time_s", float("nan"))), settings)
    if boundary_reason:
        reasons.append(boundary_reason)
    return reasons


def apply_filter(sample: Mapping[str, Any], settings: LowGFilterSettings) -> dict[str, Any]:
    result = dict(sample)
    reasons = filter_reasons(result, settings)
    result["accepted"] = not reasons
    result["rejection_reasons"] = reasons
    return result


def summarize_filter_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows_list = list(rows)
    reason_counts: Counter[str] = Counter()
    powered_counts: Counter[str] = Counter()
    stage_counts: Counter[str] = Counter()
    for row in rows_list:
        key = "powered" if bool(row.get("powered", False)) else "coast"
        powered_counts[key] += 1
        stage_counts[str(row.get("engine_stage", 0))] += 1
        for reason in row.get("rejection_reasons", []):
            reason_counts[str(reason)] += 1
    return {
        "total_rows": len(rows_list),
        "accepted_rows": sum(1 for row in rows_list if row.get("accepted")),
        "rejected_rows": sum(1 for row in rows_list if not row.get("accepted")),
        "by_power_state": dict(powered_counts),
        "by_engine_stage": dict(stage_counts),
        "rejection_reason_counts": dict(reason_counts),
    }


__all__ = [
    "LowGFilterSettings",
    "apply_filter",
    "classify_engine_stage",
    "filter_reasons",
    "near_engine_boundary",
    "normalize_sample",
    "summarize_filter_rows",
]
