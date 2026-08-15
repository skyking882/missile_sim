"""Inverse kinematics and effective fin-force extraction for H6.

The module intentionally works on effective force/normal acceleration.  It
does not invent an unobserved fin-deflection history and it keeps raw backend
arrays outside the derived sample table.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .aerodynamics import body_axes, reference_area
from .atmosphere import StandardAtmosphere
from .h6_utils import derivative, finite, mapping_get, unwrap_angles, wrap_angle
from .math3d import dot, norm, normalize


DEFAULT_MASS_KG = 147.87
DEFAULT_CALIBER_M = 0.1778
DEFAULT_WING_AREA_MULT = 1.275
DEFAULT_CY_K = 0.0
DEFAULT_CY_MAX = 1.0


FIELD_ALIASES: Dict[str, Tuple[str, ...]] = {
    "times": ("times", "time", "time_s"),
    "x_m": ("missileX", "missile_x", "x_m", "x"),
    "y_m": ("missileY", "missile_y", "y_m", "y"),
    "z_m": ("missileZ", "missile_z", "z_m", "z"),
    "target_x_m": ("targetX", "target_x", "target_x_m"),
    "target_y_m": ("targetY", "target_y", "target_y_m"),
    "target_z_m": ("targetZ", "target_z", "target_z_m"),
    "speed_mps": ("missileSpeedMs", "missile_speed_ms", "speed_mps", "speed"),
    "speed_kmh": ("missileSpeedKmh", "missile_speed_kmh", "speed_kmh"),
    "mach": ("machNumber", "mach", "mach_number"),
    "mass_kg": ("currentMass", "mass_kg", "mass"),
    "thrust_n": ("currentThrust", "thrust_n", "thrust"),
    "drag_n": ("drag", "drag_n", "reported_drag_n"),
    "cd_reported": ("Cd", "cd", "cd_reported"),
    "pitch_rad": ("pitch_rad", "pitch"),
    "angle": ("angle", "angle_rad", "flight_angle"),
    "yaw_rad": ("yaw_rad", "yaw"),
    "aoa_rad": ("aoa_rad", "aoa"),
    "current_g_reported": ("currentG", "current_g", "current_g_reported"),
    "available_g_reported": ("gLoad", "availableG", "available_g", "available_g_reported"),
    "a_cmd_pitch_g": ("aCmd", "a_cmd_pitch_g", "a_cmd_pitch"),
    "a_cmd_yaw_g": ("aCmdYaw", "a_cmd_yaw_g", "a_cmd_yaw"),
}


def _mapping_layers(value: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    """Yield likely response containers without recursively guessing fields."""

    yield value
    for key in ("data", "result", "trajectory", "simulation", "payload"):
        child = value.get(key)
        if isinstance(child, Mapping):
            yield child


def _series_from_mapping(mapping: Mapping[str, Any], aliases: Sequence[str]) -> Optional[List[Any]]:
    for layer in _mapping_layers(mapping):
        for key in aliases:
            value = layer.get(key)
            if isinstance(value, (list, tuple)):
                return list(value)
    return None


def _result_for_model(response: Any, model_id: Optional[str] = None, result_index: int = 0) -> Mapping[str, Any]:
    """Select one model result from common multi-missile response shapes."""

    if not isinstance(response, Mapping):
        raise ValueError("backend response must be an object")
    results = response.get("results")
    missile_ids = response.get("missileIds", response.get("missile_ids"))
    if isinstance(results, Mapping):
        if model_id in results and isinstance(results[model_id], Mapping):
            return results[model_id]
        if len(results) == 1:
            only = next(iter(results.values()))
            if isinstance(only, Mapping):
                return only
    if isinstance(results, list):
        if isinstance(missile_ids, list) and model_id in missile_ids:
            index = missile_ids.index(model_id)
        else:
            index = result_index
        if 0 <= index < len(results) and isinstance(results[index], Mapping):
            return results[index]
    return response


def _angle_to_rad(value: Any, unit: str) -> Optional[float]:
    if not finite(value):
        return None
    value_float = float(value)
    if unit.lower() in ("rad", "radian", "radians"):
        return value_float
    if unit.lower() in ("deg", "degree", "degrees"):
        return math.radians(value_float)
    raise ValueError("angle_unit must be rad or deg")


def _value_at(series: Optional[Sequence[Any]], index: int, default: Any = None) -> Any:
    if series is None or index >= len(series):
        return default
    return series[index]


def validate_response_arrays(response: Mapping[str, Any], model_id: str = "unknown") -> Dict[str, Any]:
    """Validate core arrays and return a report without deleting bad data."""

    arrays: Dict[str, Optional[List[Any]]] = {
        name: _series_from_mapping(response, aliases)
        for name, aliases in FIELD_ALIASES.items()
    }
    issues: List[str] = []
    required = ("times", "x_m", "y_m", "z_m", "yaw_rad", "angle")
    for name in required:
        if not arrays[name]:
            issues.append("missing_or_empty:{}".format(name))
    lengths = {name: len(values) for name, values in arrays.items() if values is not None}
    time_length = lengths.get("times", 0)
    if time_length:
        for name in required:
            if lengths.get(name) != time_length:
                issues.append("length_mismatch:{}".format(name))
        times = [float(value) for value in arrays["times"] or []]
        if any(not finite(value) for value in times):
            issues.append("non_finite_times")
        if any(right <= left for left, right in zip(times, times[1:])):
            issues.append("time_not_strictly_increasing")
    return {
        "model_id": model_id,
        "status": "pass" if not issues else "invalid",
        "issues": issues,
        "array_lengths": lengths,
        "required_arrays": list(required),
    }


def normalize_backend_result(
    response: Mapping[str, Any],
    model_id: str,
    case_id: str,
    source_kind: str = "statshark_backend_timeseries",
    angle_unit: str = "deg",
    result_index: int = 0,
    body: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Convert one raw backend result to the standard H6 row schema."""

    selected = _result_for_model(response, model_id=model_id, result_index=result_index)
    validation = validate_response_arrays(selected, model_id=model_id)
    if validation["status"] != "pass":
        return {
            "case_id": case_id,
            "model_id": model_id,
            "source_kind": source_kind,
            "status": "invalid",
            "validation": validation,
            "rows": [],
        }

    arrays: Dict[str, Optional[List[Any]]] = {
        name: _series_from_mapping(selected, aliases)
        for name, aliases in FIELD_ALIASES.items()
    }
    times = [float(value) for value in arrays["times"] or []]
    count = len(times)
    positions = [
        (
            float(_value_at(arrays["x_m"], index)),
            float(_value_at(arrays["y_m"], index)),
            float(_value_at(arrays["z_m"], index)),
        )
        for index in range(count)
    ]
    velocities: List[Tuple[float, float, float]] = []
    for axis in range(3):
        values = [position[axis] for position in positions]
        velocities.append(tuple(derivative(values, times)))  # type: ignore[arg-type]
    velocity_vectors = [tuple(velocities[axis][index] for axis in range(3)) for index in range(count)]
    speed_values: List[float] = []
    for index, velocity in enumerate(velocity_vectors):
        raw_speed = _value_at(arrays["speed_mps"], index)
        if finite(raw_speed):
            speed_values.append(float(raw_speed))
        else:
            raw_kmh = _value_at(arrays["speed_kmh"], index)
            speed_values.append(float(raw_kmh) / 3.6 if finite(raw_kmh) else norm(velocity))

    yaw_values = [_angle_to_rad(_value_at(arrays["yaw_rad"], index), angle_unit) for index in range(count)]
    pitch_values = [
        _angle_to_rad(_value_at(arrays["pitch_rad"], index, _value_at(arrays["angle"], index)), angle_unit)
        for index in range(count)
    ]
    if any(value is None for value in yaw_values + pitch_values):
        raise ValueError("backend angle arrays contain non-finite values")
    yaw_unwrapped = unwrap_angles([float(value) for value in yaw_values])
    pitch_unwrapped = unwrap_angles([float(value) for value in pitch_values])
    flight_path_yaw = unwrap_angles([
        math.atan2(velocity[2], velocity[0]) for velocity in velocity_vectors
    ])
    flight_path_pitch = [
        math.atan2(velocity[1], math.hypot(velocity[0], velocity[2]))
        for velocity in velocity_vectors
    ]
    yaw_rate = derivative(yaw_unwrapped, times)
    pitch_rate = derivative(pitch_unwrapped, times)
    flight_path_yaw_rate = derivative(flight_path_yaw, times)
    normal_yaw = [speed * rate for speed, rate in zip(speed_values, flight_path_yaw_rate)]
    body_cfg = dict(body or {})
    cy_k = float(body_cfg.get("cy_k", DEFAULT_CY_K))
    cy_max = float(body_cfg.get("cy_max_aoa", DEFAULT_CY_MAX))
    cy_mach_mult = float(body_cfg.get("cy_mach_mult", 1.0))
    mass_default = float(body_cfg.get("mass_kg", DEFAULT_MASS_KG))
    area = float(body_cfg.get("reference_area_m2", math.pi * DEFAULT_CALIBER_M ** 2 / 4.0 * DEFAULT_WING_AREA_MULT))
    atmosphere = StandardAtmosphere()
    rows: List[Dict[str, Any]] = []
    for index, time_s in enumerate(times):
        position = positions[index]
        velocity = velocity_vectors[index]
        speed = max(float(speed_values[index]), 0.0)
        altitude = position[1]
        atmospheric = atmosphere.sample(altitude)
        q = 0.5 * atmospheric.density_kg_m3 * speed * speed
        mach = _value_at(arrays["mach"], index)
        if not finite(mach):
            mach = speed / atmospheric.speed_of_sound_mps if atmospheric.speed_of_sound_mps else 0.0
        mass_value = _value_at(arrays["mass_kg"], index)
        mass = float(mass_value) if finite(mass_value) and float(mass_value) > 0.0 else mass_default
        beta = wrap_angle(yaw_unwrapped[index] - flight_path_yaw[index])
        alpha = float(pitch_unwrapped[index]) - float(flight_path_pitch[index])
        cy = max(-cy_max, min(cy_max, cy_k * cy_mach_mult * beta))
        body_accel = q * area * cy / mass if mass > 0.0 else 0.0
        fin_accel = normal_yaw[index] - body_accel
        raw_speed = _value_at(arrays["speed_mps"], index)
        raw_kmh = _value_at(arrays["speed_kmh"], index)
        row: Dict[str, Any] = {
            "case_id": case_id,
            "model_id": model_id,
            "source_kind": source_kind,
            "time_s": time_s,
            "x_m": position[0],
            "y_m": position[1],
            "z_m": position[2],
            "target_x_m": _value_at(arrays["target_x_m"], index),
            "target_y_m": _value_at(arrays["target_y_m"], index),
            "target_z_m": _value_at(arrays["target_z_m"], index),
            "speed_mps": speed,
            "speed_raw_mps": float(raw_speed) if finite(raw_speed) else None,
            "speed_raw_kmh": float(raw_kmh) if finite(raw_kmh) else None,
            "mach": float(mach),
            "mass_kg": mass,
            "thrust_n": _value_at(arrays["thrust_n"], index),
            "drag_n": _value_at(arrays["drag_n"], index),
            "cd_reported": _value_at(arrays["cd_reported"], index),
            "pitch_rad": float(pitch_unwrapped[index]),
            "yaw_rad": float(yaw_unwrapped[index]),
            "aoa_rad": _angle_to_rad(_value_at(arrays["aoa_rad"], index), angle_unit),
            "current_g_reported": _value_at(arrays["current_g_reported"], index),
            "available_g_reported": _value_at(arrays["available_g_reported"], index),
            "a_cmd_pitch_g": _value_at(arrays["a_cmd_pitch_g"], index),
            "a_cmd_yaw_g": _value_at(arrays["a_cmd_yaw_g"], index),
            "raw_index": index,
            "vx": velocity[0],
            "vy": velocity[1],
            "vz": velocity[2],
            "flight_path_pitch_rad": float(flight_path_pitch[index]),
            "flight_path_yaw_rad": float(flight_path_yaw[index]),
            "alpha_pitch_rad": alpha,
            "beta_yaw_rad": beta,
            "pitch_rate_rad_s": pitch_rate[index],
            "yaw_rate_rad_s": yaw_rate[index],
            "flight_path_turn_rate_rad_s": flight_path_yaw_rate[index],
            "normal_accel_pitch_mps2": speed * derivative(flight_path_pitch, times)[index],
            "normal_accel_yaw_mps2": normal_yaw[index],
            "body_normal_accel_mps2": body_accel,
            "fin_normal_accel_mps2": fin_accel,
            "fin_force_n": mass * fin_accel,
            "dynamic_pressure_pa": q,
            "extra_drag_residual_n": None,
            "validity_flags": [
                "angles_unwrapped_before_derivative",
                "fin_force_is_effective_curvature_minus_body",
            ],
        }
        if not finite(raw_speed) and not finite(raw_kmh):
            row["validity_flags"].append("speed_derived_from_position")
        rows.append(row)
    return {
        "case_id": case_id,
        "model_id": model_id,
        "source_kind": source_kind,
        "status": "normalized",
        "validation": validation,
        "angle_unit_input": angle_unit,
        "body_assumptions": {
            "mass_default_kg": mass_default,
            "reference_area_m2": area,
            "cy_k": cy_k,
            "cy_max_aoa": cy_max,
            "cy_mach_mult": cy_mach_mult,
        },
        "rows": rows,
    }


def normalize_capture_bundle(bundle: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize a capture bundle while retaining failed/empty records."""

    records = bundle.get("captures", bundle.get("results", bundle.get("cases", [])))
    if isinstance(records, Mapping):
        records = [records]
    if not isinstance(records, list):
        raise ValueError("bundle requires a captures/results/cases list")
    normalized: List[Dict[str, Any]] = []
    for record_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            normalized.append({"record_index": record_index, "status": "invalid_record"})
            continue
        case_id = str(record.get("case_id", "case-{}".format(record_index)))
        model_id = str(record.get("model_id", record.get("model_name", "model-{}".format(record_index))))
        status = str(record.get("status", "success"))
        response = record.get("response", record.get("raw_response", record.get("result")))
        if status not in ("success", "captured", "partial") or not isinstance(response, Mapping):
            normalized.append({
                "record_index": record_index,
                "case_id": case_id,
                "model_id": model_id,
                "source_kind": str(record.get("source_kind", "statshark_backend_timeseries")),
                "status": "preserved_non_success",
                "capture_status": status,
                "failure": record.get("failure", record.get("error")),
                "rows": [],
            })
            continue
        normalized_item = normalize_backend_result(
            response=response,
            model_id=model_id,
            case_id=case_id,
            source_kind=str(record.get("source_kind", "statshark_backend_timeseries")),
            angle_unit=str(record.get("angle_unit", bundle.get("angle_unit", "deg"))),
            result_index=int(record.get("result_index", record_index)),
            body=record.get("body_assumptions", bundle.get("body_assumptions")),
        )
        normalized_item["record_index"] = record_index
        normalized_item["capture_status"] = status
        normalized.append(normalized_item)
    return {
        "schema_version": 1,
        "source_kind": "statshark_backend_timeseries",
        "raw_bundle_schema": bundle.get("schema_version"),
        "record_count": len(normalized),
        "normalized_records": normalized,
        "normalized_rows": [row for item in normalized for row in item.get("rows", [])],
        "status": "complete" if normalized else "empty_bundle",
    }


__all__ = [
    "DEFAULT_MASS_KG",
    "FIELD_ALIASES",
    "normalize_backend_result",
    "normalize_capture_bundle",
    "validate_response_arrays",
]
