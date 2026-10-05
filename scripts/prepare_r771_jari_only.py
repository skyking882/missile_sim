#!/usr/bin/env python3
"""Prepare existing public JARI exports as the sole observation dataset.

This is an offline importer for the commercial-video-game simulator project.
It does not import any forward model, load game telemetry, fit coefficients,
make network requests, or run flight predictions. Creation inputs have their
own accessor so no future source samples are needed by a free forward run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/jari_only/dataset"
G0 = 9.80665
VERSION = "2.59.0.28"
RAW_PROFILE = ROOT / "data/datamine_2.59.0.27/aces.vromfs.bin_u/gamedata/weapons/rocketguns/su_r_77_1.blkx"
FIT_CASES = ("angle_p0", "angle_p30", "angle_p60")
VALIDATION_CASES = ("angle_p5", "angle_p10", "angle_p15", "angle_p45", "angle_p75", "angle_m30", "collision")
SAMPLE_FIELDS = {
    "t", "position", "velocity", "quaternion", "omega", "target", "speed",
    "asl", "mass", "aoa", "range", "acc_drag", "acc_lift", "acc_thrust",
    "thrust", "controls", "loft", "guidance_label", "cd_base", "cd_aoa",
    "drag_base_n", "drag_aoa_n",
}

# Paths are public JARI captures. The registry is fixed before any new fit.
SOURCE_SPECS = {
    **{f"angle_p{a}": (f"outputs/jari_offaxis/reference/jari_259_{a}_response.json",
                       f"outputs/jari_offaxis/reference/old_{a}_request.json", None)
       for a in (0, 30, 60)},
    **{f"angle_{'m' if a < 0 else 'p'}{abs(a)}": (
        f"outputs/jari_offaxis/angle_grid/angle_{'m' if a < 0 else 'p'}{abs(a)}_response.json",
        f"outputs/jari_offaxis/angle_grid/angle_{'m' if a < 0 else 'p'}{abs(a)}_request.json",
        f"outputs/jari_offaxis/angle_grid/angle_{'m' if a < 0 else 'p'}{abs(a)}_frontend_download.json")
       for a in (5, 10, 15, 45, 75, -30)},
    "collision": ("outputs/jari_offaxis/collision_pn_probe/collision_config_response.json",
                  "outputs/jari_offaxis/collision_pn_probe/collision_config_request.json", None),
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def file_receipt(path: Path, kind: str) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "kind": kind, "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def protected_receipt() -> dict[str, tuple[int, int]]:
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns)
            for folder in ("src", "config", "configs", "schemas", "docs", "data", "outputs/jari_offaxis")
            for p in (ROOT / folder).rglob("*") if p.is_file()
            and "__pycache__" not in p.parts and ".git" not in p.parts}


def finite_vector(value: Any, size: int, label: str) -> np.ndarray:
    a = np.asarray(value, dtype=float)
    if a.shape != (size,) or not np.isfinite(a).all():
        raise ValueError(f"{label}: expected {size} finite components")
    return a


def derivative(values: np.ndarray, t: np.ndarray, width: int) -> np.ndarray:
    """Offline local differences on original timestamps; uses future rows."""
    result = np.empty_like(values, dtype=float)
    for i in range(len(t)):
        start = min(max(i - width // 2, 0), len(t) - width)
        ids = np.arange(start, start + width)
        delta = t[ids] - t[i]
        matrix = np.stack([delta ** j for j in range(width)])
        rhs = np.zeros(width)
        rhs[1] = 1.0
        result[i] = np.linalg.solve(matrix, rhs) @ values[ids]
    return result


def derivative_valid(t: np.ndarray, width: int) -> np.ndarray:
    valid = np.zeros(len(t), dtype=bool)
    half = width // 2
    nominal = float(np.median(np.diff(t)))
    for i in range(half, len(t) - half):
        valid[i] = np.all(np.abs(np.diff(t[i-half:i+half+1]) - nominal) < 1e-9)
    return valid


def normalized_quaternions(raw: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(raw, axis=1)
    if np.any(length <= 1e-12):
        raise ValueError("zero quaternion cannot be interpreted")
    q = raw / length[:, None]
    for i in range(1, len(q)):
        if q[i-1] @ q[i] < 0.0:
            q[i] *= -1.0
    return q


def body_axes_xyzw(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, z, w = q.T
    forward = np.column_stack((1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y)))
    up = np.column_stack((2*(x*y-w*z), 1-2*(x*x+z*z), 2*(y*z+w*x)))
    right = np.column_stack((2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)))
    return forward, up, right


def quaternion_body_rates(q: np.ndarray, t: np.ndarray, width: int) -> np.ndarray:
    qdot = derivative(q, t, width)
    vector, scalar = q[:, :3], q[:, 3]
    vector_dot, scalar_dot = qdot[:, :3], qdot[:, 3]
    # 2 * vector(conjugate(q) * qdot), in the chosen quaternion convention.
    return 2.0 * (scalar[:, None] * vector_dot - scalar_dot[:, None] * vector
                  - np.cross(vector, vector_dot))


def numeric_samples(samples: list[dict[str, Any]], field: str, width: int | None = None) -> np.ndarray:
    if width is None:
        a = np.asarray([float(s[field]) for s in samples])
    else:
        a = np.stack([finite_vector(s[field], width, field) for s in samples])
    if not np.isfinite(a).all():
        raise ValueError(f"nonfinite exported {field}")
    return a


def column_inventory() -> dict[str, dict[str, str]]:
    inventory: dict[str, dict[str, str]] = {}
    def add(name: str, kind: str, unit: str, source: str, caveat: str = "") -> None:
        inventory[name] = {"kind": kind, "units": unit, "source_or_formula": source, "caveat": caveat}
    for name, field, unit in (
        ("t", "t", "s"), ("position_xyz_m", "position", "m"),
        ("velocity_xyz_mps", "velocity", "m/s"), ("target_xyz_m", "target", "m"),
        ("speed_source_mps", "speed", "m/s"), ("asl_source_m", "asl", "m"),
        ("mass_source_kg", "mass", "kg"), ("aoa_source_unsigned_deg", "aoa", "deg"),
        ("range_source_m", "range", "m"), ("acc_drag_source_mps2", "acc_drag", "m/s^2"),
        ("acc_lift_source_mps2", "acc_lift", "m/s^2"), ("acc_thrust_source_mps2", "acc_thrust", "m/s^2"),
        ("thrust_source_n", "thrust", "N"), ("cd_base_source", "cd_base", "dimensionless"),
        ("cd_aoa_source", "cd_aoa", "dimensionless"), ("drag_base_source_n", "drag_base_n", "N"),
        ("drag_aoa_source_n", "drag_aoa_n", "N")):
        caveat = "Export scalar display diagnostic; direction/projection and update phase unverified" if field.startswith("acc_") else ""
        if field == "aoa": caveat = "Unsigned total incidence; must not be relabelled signed pitch AoA"
        add(name, "JARI_exported", unit, f"samples.{field}", caveat)
    add("quaternion_xyzw_raw", "JARI_exported", "dimensionless", "samples.quaternion", "Terminal correspondence to named qx/qy/qz/qw proves order; raw values retained")
    add("omega_export_raw", "JARI_exported", "rad/s", "samples.omega", "Terminal correspondence to named omega_x/y/z_rad_s proves units and component order; frame and sample phase remain unverified")
    add("controls_export_raw", "JARI_exported", "undeclared", "samples.controls", "Terminal correspondence proves [control_horizontal,control_vertical]; scale/filter/held-output phase unverified; not a fin-angle observation")
    add("loft_export_is_nonnull", "JARI_exported_null_mask", "boolean", "samples.loft is not null", "All current rows null; cannot establish internal loft activation")
    add("guidance_label_export", "JARI_exported", "category", "samples.guidance_label", "All current rows API; no guidance request decomposition")
    derived = {
        "speed_mps": ("m/s", "norm(velocity)"),
        "quaternion_xyzw_normalized": ("dimensionless", "normalize raw XYZW and choose continuous antipodal signs"),
        "body_forward": ("dimensionless", "rotate +X by normalized quaternion"),
        "body_up": ("dimensionless", "rotate +Y by normalized quaternion"),
        "body_right": ("dimensionless", "rotate +Z by normalized quaternion"),
        "velocity_tangent": ("dimensionless", "velocity/norm(velocity)"),
        "geometric_incidence_unsigned_deg": ("deg", "atan2(norm(tangent cross body_forward), tangent dot body_forward)"),
        "geometric_pitch_incidence_rad": ("rad", "atan2(-tangent dot body_up, tangent dot body_forward)"),
        "geometric_yaw_incidence_rad": ("rad", "atan2(-tangent dot body_right, tangent dot body_forward)"),
        "body_pitch_deg": ("deg", "asin(body_forward.y)"),
        "body_yaw_deg": ("deg", "unwrap(atan2(body_forward.z,body_forward.x))"),
        "body_bank_deg": ("deg", "atan2(body_up dot nominal_right,body_up dot nominal_up)"),
        "relative_position_xyz_m": ("m", "export target-export position"),
        "relative_velocity_xyz_mps": ("m/s", "scenario target_velocity-export velocity"),
        "los_rate_xyz_rad_s": ("rad/s", "relative_position cross relative_velocity / range_squared"),
        "closing_speed_mps": ("m/s", "-relative_position dot relative_velocity/range"),
        "drag_total_scalar_source_derived_n": ("N", "export drag_base_n+drag_aoa_n"),
        "early_mask": ("boolean", "0<=original t<=3.5"),
        "full_mask": ("boolean", "every original export row, including terminal interpolated event"),
    }
    for name, (unit, formula) in derived.items():
        add(name, "offline_derived", unit, formula, "No model coefficients; quaternion conventions apply to quaternion-derived channels")
    for width in (3, 5):
        for name, unit, formula in (
            (f"acceleration_{width}point_xyz_mps2", "m/s^2", "finite difference of exported velocity"),
            (f"specific_acceleration_{width}point_xyz_mps2", "m/s^2", "acceleration+(0,9.80665,0)"),
            (f"normal_specific_load_{width}point_g", "G", "norm(specific_acceleration minus tangent projection)/9.80665"),
            (f"total_nongrav_force_{width}point_xyz_n", "N", "source mass*specific acceleration; kinematic estimate, not exported force"),
            (f"quaternion_body_rate_{width}point_xyz_rad_s", "rad/s", "2*vector(conjugate(normalized_q)*dq/dt)"),
            (f"body_pitch_rate_{width}point_deg_s", "deg/s", "degrees(quaternion_body_rate.z)"),
            (f"body_yaw_rate_{width}point_deg_s", "deg/s", "degrees(-quaternion_body_rate.y)"),
            (f"derivative_{width}point_valid_mask", "boolean", "interior symmetric stencil with regular original sample spacing")):
            add(name, "offline_derived_uses_future_samples", unit, formula, "Display-cadence derivative; never a runtime forcing input; use valid mask to avoid endpoint/terminal stencils")
    return inventory


def convert_run(run: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if run["schema"] != 3 or run["missile"] != "su_r_77_1.blk":
        raise ValueError("Expected original R-77-1 schema-3 JARI export")
    if run["backend"] != {"name": "compiled-c", "version": VERSION, "fallback_reason": None}:
        raise ValueError("JARI backend/version differs from fixed dataset boundary")
    scenario = run["scenario"]
    if scenario["guidance"] != "perfect-track" or scenario["loft_state"] != "config":
        raise ValueError("Only existing standard configured-loft perfect-track cases are admitted")
    if "ASL" not in run["axes"]["y"] or "deg" not in run["axes"]["units"]:
        raise ValueError("World Y-up/ASL and degree export conventions absent")
    samples = run["samples"]
    if len(samples) < 5 or any(set(s) != SAMPLE_FIELDS for s in samples):
        raise ValueError("Unrecognized sample inventory; do not silently discard added/missing channels")
    t = numeric_samples(samples, "t")
    if t[0] != 0.0 or np.any(np.diff(t) <= 0.0):
        raise ValueError("Original creation timestamp and strict time order required")
    terminal = run["proximity_event"]["state"]
    terminal_mapping = {
        "quaternion": [terminal[k] for k in ("qx", "qy", "qz", "qw")],
        "omega": [terminal[f"omega_{axis}_rad_s"] for axis in "xyz"],
        "controls": [terminal[f"control_{axis}"] for axis in ("horizontal", "vertical")],
    }
    terminal_differences = {key: float(np.max(np.abs(np.asarray(samples[-1][key])-value)))
                            for key, value in terminal_mapping.items()}
    if terminal["time_s"] != t[-1] or any(terminal_differences.values()):
        raise ValueError("Terminal named columns do not match sample export; cannot claim order/units")
    a = {"t": t, "position_xyz_m": numeric_samples(samples, "position", 3),
         "velocity_xyz_mps": numeric_samples(samples, "velocity", 3),
         "target_xyz_m": numeric_samples(samples, "target", 3),
         "quaternion_xyzw_raw": numeric_samples(samples, "quaternion", 4),
         "omega_export_raw": numeric_samples(samples, "omega", 3),
         "controls_export_raw": numeric_samples(samples, "controls", 2)}
    mapping = {"speed": "speed_source_mps", "asl": "asl_source_m", "mass": "mass_source_kg",
               "aoa": "aoa_source_unsigned_deg", "range": "range_source_m", "thrust": "thrust_source_n",
               **{f"acc_{name}": f"acc_{name}_source_mps2" for name in ("drag", "lift", "thrust")},
               **{f"cd_{name}": f"cd_{name}_source" for name in ("base", "aoa")},
               **{f"drag_{name}_n": f"drag_{name}_source_n" for name in ("base", "aoa")}}
    a.update({name: numeric_samples(samples, field) for field, name in mapping.items()})
    a["loft_export_is_nonnull"] = np.asarray([s["loft"] is not None for s in samples])
    if a["loft_export_is_nonnull"].any():
        raise ValueError("New non-null loft schema requires explicit handling")
    a["guidance_label_export"] = np.asarray([s["guidance_label"] for s in samples], dtype=str)
    q = normalized_quaternions(a["quaternion_xyzw_raw"])
    forward, up, right = body_axes_xyzw(q)
    speed = np.linalg.norm(a["velocity_xyz_mps"], axis=1)
    if np.any(speed <= 0.0) or np.any(a["mass_source_kg"] <= 0.0) or np.any(a["aoa_source_unsigned_deg"] < 0.0):
        raise ValueError("Positive speed/mass and unsigned exported incidence required")
    tangent = a["velocity_xyz_mps"] / speed[:, None]
    dotf = np.sum(tangent * forward, axis=1)
    pitch = np.arcsin(np.clip(forward[:, 1], -1.0, 1.0))
    yaw = np.unwrap(np.arctan2(forward[:, 2], forward[:, 0]))
    nominal_up = np.column_stack((-np.sin(pitch)*np.cos(yaw), np.cos(pitch), -np.sin(pitch)*np.sin(yaw)))
    nominal_right = np.column_stack((-np.sin(yaw), np.zeros(len(t)), np.cos(yaw)))
    relative = a["target_xyz_m"] - a["position_xyz_m"]
    relative_v = np.asarray(scenario["target_velocity"])[None, :] - a["velocity_xyz_mps"]
    geometric_range = np.linalg.norm(relative, axis=1)
    if np.any(geometric_range <= 0.0):
        raise ValueError("Zero-range state has undefined LOS rate")
    a.update(speed_mps=speed, quaternion_xyzw_normalized=q, body_forward=forward,
             body_up=up, body_right=right, velocity_tangent=tangent,
             geometric_incidence_unsigned_deg=np.degrees(np.arctan2(np.linalg.norm(np.cross(tangent, forward), axis=1), dotf)),
             geometric_pitch_incidence_rad=np.arctan2(-np.sum(tangent*up, axis=1), dotf),
             geometric_yaw_incidence_rad=np.arctan2(-np.sum(tangent*right, axis=1), dotf),
             body_pitch_deg=np.degrees(pitch), body_yaw_deg=np.degrees(yaw),
             body_bank_deg=np.degrees(np.arctan2(np.sum(up*nominal_right, axis=1), np.sum(up*nominal_up, axis=1))),
             relative_position_xyz_m=relative, relative_velocity_xyz_mps=relative_v,
             los_rate_xyz_rad_s=np.cross(relative, relative_v)/geometric_range[:, None]**2,
             closing_speed_mps=-np.sum(relative*relative_v, axis=1)/geometric_range,
             drag_total_scalar_source_derived_n=a["drag_base_source_n"]+a["drag_aoa_source_n"],
             early_mask=(t >= 0.0) & (t <= 3.5), full_mask=np.ones(len(t), dtype=bool))
    target_expected = np.asarray(scenario["target_position"])+t[:, None]*np.asarray(scenario["target_velocity"])
    diagnostic = {
        "rows": len(t), "early_rows": int(a["early_mask"].sum()), "full_support_s": [float(t[0]), float(t[-1])],
        "time_step_min_median_max_s": np.quantile(np.diff(t), [0.0, 0.5, 1.0]),
        "source_speed_minus_norm_velocity_max_mps": float(np.max(np.abs(a["speed_source_mps"]-speed))),
        "source_asl_minus_position_y_max_m": float(np.max(np.abs(a["asl_source_m"]-a["position_xyz_m"][:, 1]))),
        "source_range_minus_geometric_range_max_m": float(np.max(np.abs(a["range_source_m"]-geometric_range))),
        "target_linear_motion_max_m": float(np.max(np.linalg.norm(a["target_xyz_m"]-target_expected, axis=1))),
        "raw_quaternion_norm_minus_one_max": float(np.max(np.abs(np.linalg.norm(a["quaternion_xyzw_raw"], axis=1)-1.0))),
        "source_unsigned_aoa_minus_geometric_incidence_max_deg": float(np.max(np.abs(a["aoa_source_unsigned_deg"]-a["geometric_incidence_unsigned_deg"]))),
        "drag_scalar_closure_max_n": float(np.max(np.abs(a["drag_total_scalar_source_derived_n"]-a["mass_source_kg"]*a["acc_drag_source_mps2"]))),
        "thrust_scalar_closure_max_n": float(np.max(np.abs(a["thrust_source_n"]-a["mass_source_kg"]*a["acc_thrust_source_mps2"]))),
        "controls_min_max_by_component": np.column_stack((a["controls_export_raw"].min(axis=0), a["controls_export_raw"].max(axis=0))),
        "non_null_loft_rows": int(a["loft_export_is_nonnull"].sum()),
        "terminal_named_field_correspondence_max_abs": terminal_differences,
        "terminal_native_field_inventory": sorted(terminal),
        "terminal_guidance_velocity_source": terminal["guidance_velocity_source"],
    }
    for width in (3, 5):
        accel = derivative(a["velocity_xyz_mps"], t, width)
        specific = accel+np.asarray([0.0, G0, 0.0])
        rates = quaternion_body_rates(q, t, width)
        mask = derivative_valid(t, width)
        a.update({f"acceleration_{width}point_xyz_mps2": accel,
                  f"specific_acceleration_{width}point_xyz_mps2": specific,
                  f"normal_specific_load_{width}point_g": np.linalg.norm(specific-np.sum(specific*tangent, axis=1)[:, None]*tangent, axis=1)/G0,
                  f"total_nongrav_force_{width}point_xyz_n": a["mass_source_kg"][:, None]*specific,
                  f"quaternion_body_rate_{width}point_xyz_rad_s": rates,
                  f"body_pitch_rate_{width}point_deg_s": np.degrees(rates[:, 2]),
                  f"body_yaw_rate_{width}point_deg_s": np.degrees(-rates[:, 1]),
                  f"derivative_{width}point_valid_mask": mask})
        position_rate_error = derivative(a["position_xyz_m"], t, width)-a["velocity_xyz_mps"]
        diagnostic[f"position_derivative_{width}point_minus_velocity"] = {
            "valid_rows": int(mask.sum()), "units": "m/s", "rmse_vector_norm": float(np.sqrt(np.mean(np.sum(position_rate_error[mask]**2, axis=1)))),
            "max_vector_norm": float(np.max(np.linalg.norm(position_rate_error[mask], axis=1))),
            "interpretation": "Finite-cadence/integration-phase residual; not expected to close exactly"}
    for key, value in a.items():
        if value.dtype.kind in "fiu" and not np.isfinite(value).all():
            raise ValueError(f"Nonfinite conversion result {key}")
    if set(a) != set(column_inventory()):
        raise AssertionError("Column metadata and array inventory differ")
    return a, diagnostic


def load_case(case_id: str, directory: Path = OUT) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Offline fitting/diagnostic accessor; source derivatives use future rows."""
    manifest = read_json(directory / "manifest.json")
    metadata = manifest["cases"][case_id]
    with np.load(directory / "observations.npz", allow_pickle=False) as archive:
        prefix = case_id+"__"
        arrays = {k[len(prefix):]: archive[k].copy() for k in archive.files if k.startswith(prefix)}
    return metadata, arrays


def creation_input(case_id: str, directory: Path = OUT) -> dict[str, Any]:
    """Free-forward accessor: only launch inputs, no observation/terminal file."""
    creation = read_json(directory / "creation_inputs.json")
    return {"case_id": case_id, **creation["cases"][case_id]}


def prepare() -> dict[str, Any]:
    before = protected_receipt()
    arrays = {}
    cases = {}
    input_receipts = []
    provenance = None
    backend = None
    for case_id in FIT_CASES+VALIDATION_CASES:
        source_relative, request_relative, wrapper_relative = SOURCE_SPECS[case_id]
        source_path, request_path = ROOT/source_relative, ROOT/request_relative
        run, request = read_json(source_path), read_json(request_path)
        input_receipts += [file_receipt(source_path, "JARI_export"), file_receipt(request_path, "JARI_request")]
        # The -30 degree frontend rounded -3999.9999999999995 to -4000.
        # Preserve both original inputs; admit only the recorded sub-nanometre
        # target-coordinate serialization difference, never replace creation.
        coordinate_difference = np.asarray(run["scenario"]["target_position"])-np.asarray(request["target_position"])
        if (np.max(np.abs(coordinate_difference)) > 1e-9 or
                any(run["scenario"].get(k) != value for k, value in request.items() if k != "target_position")):
            raise ValueError(f"{case_id}: request and exported scenario differ")
        if provenance is None:
            provenance, backend = run["profile_source"], run["backend"]
        if run["profile_source"] != provenance or run["provenance"] != provenance or run["backend"] != backend:
            raise ValueError("Mixed resource/profile/backend provenance")
        if wrapper_relative is not None:
            wrapper_path = ROOT/wrapper_relative
            wrapper = read_json(wrapper_path)
            if wrapper["kind"] != "missile_comparison" or wrapper["selected_missiles"] != ["su_r_77_1.blk"] or wrapper["errors"] or wrapper["runs"] != [run]:
                raise ValueError("Saved original frontend wrapper and extracted run differ")
            input_receipts.append(file_receipt(wrapper_path, "JARI_original_frontend_wrapper"))
        a, diagnostic = convert_run(run)
        arrays.update({case_id+"__"+key: value for key, value in a.items()})
        first = run["samples"][0]
        cases[case_id] = {
            "split": "fit" if case_id in FIT_CASES else "site_validation",
            "historically_examined": True, "blind_test": False,
            "source_path": str(source_path), "request_path": str(request_path),
            "original_frontend_wrapper_path": str(ROOT/wrapper_relative) if wrapper_relative else None,
            "scenario_exact": run["scenario"], "request_exact": request, "axes_exact": run["axes"],
            "request_to_export_target_coordinate_difference_m": coordinate_difference,
            "api_summary_exact": run["api_summary"], "summary_exact": run["summary"],
            "proximity_event_exact": run["proximity_event"], "assumptions_exact": run["assumptions"],
            "field_inventory": sorted(SAMPLE_FIELDS), "diagnostics": diagnostic,
            "creation_input": {"scenario_exact": run["scenario"],
                "position_xyz_m": first["position"], "velocity_xyz_mps": first["velocity"],
                "quaternion_xyzw_raw": first["quaternion"], "omega_export_raw": first["omega"],
                "mass_source_kg": first["mass"], "target_position_xyz_m": first["target"],
                "target_velocity_xyz_mps": run["scenario"]["target_velocity"],
                "time_s": first["t"], "controls_export_raw": first["controls"],
                "precision_note": "Actual exported creation values, including float32 boundaries; no idealized replacement"},
        }
    p, m = "angle_p30__", "angle_m30__"
    mirrored_errors = {}
    for key, signs in (("position_xyz_m", [1, 1, -1]), ("velocity_xyz_mps", [1, 1, -1]),
                       ("target_xyz_m", [1, 1, -1]), ("quaternion_xyzw_raw", [-1, -1, 1, 1]),
                       ("omega_export_raw", [-1, -1, 1]), ("controls_export_raw", [-1, 1])):
        if arrays[p+key].shape != arrays[m+key].shape:
            raise AssertionError("Mirror pair supports differ")
        mirrored_errors[key] = float(np.max(np.abs(arrays[m+key]-arrays[p+key]*np.asarray(signs))))
    for key in ("t", "speed_source_mps", "aoa_source_unsigned_deg", "mass_source_kg", "range_source_m"):
        mirrored_errors[key] = float(np.max(np.abs(arrays[m+key]-arrays[p+key])))
    raw = read_json(RAW_PROFILE)
    input_receipts.append(file_receipt(RAW_PROFILE, "local_raw_parameter_metadata_only"))
    frozen_path = ROOT/"config/r771_simple_joint_frozen.json"
    frozen = read_json(frozen_path)
    input_receipts.append(file_receipt(frozen_path, "previous_frozen_config_provenance_audit_only"))
    frozen_audit = {key: frozen[key] for key in ("model_family", "status", "source_results", "source_candidate", "source_version", "source_strict_fit_acceptance")}
    manifest = {
        "schema": 1, "status": "prepared_jari_only_inputs", "observation_source": "existing_public_JARI_exports_only",
        "site_url": "https://jari-aerospace.top", "reference_profile_source_exact": provenance,
        "reference_backend_exact": backend, "arrays_path": str(OUT/"observations.npz"),
        "creation_inputs_path": str(OUT/"creation_inputs.json"),
        "split_predeclared_before_new_fit": {"fit": list(FIT_CASES), "site_validation": list(VALIDATION_CASES),
            "historically_examined_all": True, "blind": False,
            "game_telemetry": "reserved future independent test set; no telemetry loaded or selected here"},
        "windows_predeclared_before_new_fit": {"early": [0.0, 3.5], "full": "each complete original source support, including terminal event",
            "comparison_rule": "Use original source timestamps, interpolate prediction within its support only; report every uncovered source row, no extrapolation/time fit",
            "derivative_rule": "Separate 3/5-point diagnostics; regular symmetric-stencil masks exclude endpoint and terminal contamination"},
        "columns": column_inventory(), "cases": cases,
        "interface_identifiability": {
            "actual_controls": "Terminal named-state correspondence establishes export order [horizontal,vertical]; units/scale/hold phase not independently identified",
            "omega": "Terminal named-state correspondence establishes [x,y,z] rad/s; frame and sample phase not independently identified",
            "quaternion": "Terminal named-state correspondence establishes [x,y,z,w] order",
            "guidance_request": "No time-series PN/loft/request vector; terminal state has requested_g scalar and guidance velocity fields; API peak_requested_g_before_native_limit is only a summary",
            "pid_internal_terms": "not exported",
            "forces": "Time series has scalar drag_base_n/drag_aoa_n/thrust and scalar display acceleration components; terminal named state additionally gives acceleration/feedback acceleration/specific_force_g/trajectory_normal_g, but no force-vector time series",
            "moments": "no torque vector or inertia tensor exported; cannot independently close rotational equation",
            "loft": "all samples.loft=null; configured loft input does not reveal internal bias history",
            "coupled_terms": "unknown control input scale and plant control gain remain confounded; request and PID memory not separately observed",
        },
        "local_raw_parameter_metadata": {"receipt": file_receipt(RAW_PROFILE, "local_raw_parameter_metadata_only"),
            "version": "2.59.0.27", "rocket_exact": raw["rocket"],
            "same_as_jari_28_resource_independently_verified": False,
            "use_boundary": "Raw fields are metadata/design inputs, not observed .28 control semantics; no replay-parent version bridge loaded"},
        "previous_frozen_config_audit": {**frozen_audit, "coefficients_carried_into_dataset": False,
            "use_boundary": "Historical provenance only; prior replay-calibrated coefficients are neither identified facts nor constraints for JARI-only fitting"},
        "independent_conversion_checks": {"mirror_plus_minus_30": mirrored_errors,
            "all_numeric_arrays_finite": True, "quaternions_nonzero_and_normalized_for_derivation": True,
            "source_times_unchanged": True, "future_derivatives_runtime_forcing": False},
        "dataflow_audit": {"loaded_inputs": input_receipts,
            "imports": ["stdlib", "numpy"], "model_or_old_harness_imports": [],
            "game_telemetry_loaded": False, "old_runtime_profile_snapshot_loaded": False,
            "replay_parent_version_bridge_loaded": False, "old_candidate_forecast_loaded": False,
            "new_source_requests": 0, "fits": 0, "free_forwards": 0,
            "creation_runtime_accessor": "creation_input() reads creation_inputs.json only; no observations.npz or manifest/terminal metadata"},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT/"observations.npz", **arrays)
    creation = {"schema": 1, "source": "actual JARI creation states and scenario inputs only",
                "future_samples_present": False,
                "cases": {case_id: case["creation_input"] for case_id, case in cases.items()}}
    (OUT/"creation_inputs.json").write_text(json.dumps(jsonable(creation), ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    manifest["independent_conversion_checks"]["protected_size_mtime_unchanged"] = before == protected_receipt()
    manifest["independent_conversion_checks"]["protected_file_count"] = len(before)
    if not manifest["independent_conversion_checks"]["protected_size_mtime_unchanged"]:
        raise AssertionError("Protected source/history files changed")
    (OUT/"manifest.json").write_text(json.dumps(jsonable(manifest), ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    # Focused round-trip checks include original controls and timestamps.
    for case_id in FIT_CASES+VALIDATION_CASES:
        metadata, actual = load_case(case_id)
        if set(actual) != set(column_inventory()):
            raise AssertionError("Saved input column inventory differs")
        for key, value in actual.items():
            if not np.array_equal(value, arrays[case_id+"__"+key]):
                raise AssertionError(f"NPZ round trip changed {case_id}/{key}")
        if creation_input(case_id)["velocity_xyz_mps"] != metadata["creation_input"]["velocity_xyz_mps"]:
            raise AssertionError("Creation-only accessor changed launch state")
    return {"cases": len(cases), "fit": list(FIT_CASES), "site_validation": list(VALIDATION_CASES),
            "columns_per_case": len(column_inventory()), "source_rows": sum(c["diagnostics"]["rows"] for c in cases.values()),
            "protected_files": len(before), "protected_unchanged": True,
            "new_source_requests": 0, "fits": 0, "free_forwards": 0,
            "manifest": str(OUT/"manifest.json"), "arrays": str(OUT/"observations.npz"),
            "creation_inputs": str(OUT/"creation_inputs.json")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
