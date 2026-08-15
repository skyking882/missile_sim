"""Force-direction and energy bookkeeping for H6."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .aerodynamics import body_axes
from .h6_utils import derivative, finite, normal_equations, rms, solve_linear_system
from .math3d import dot, normalize, scale, sub


Vector = Tuple[float, float, float]


def _project_perpendicular(vector: Vector, axis_hat: Vector) -> Vector:
    return sub(vector, scale(axis_hat, dot(vector, axis_hat)))


def fin_force_vector(
    force_n: float,
    mode: str,
    velocity_hat: Vector,
    body_forward: Vector,
    body_lateral: Vector,
) -> Vector:
    """Construct flow-normal or body-normal effective force direction."""

    if mode not in ("flow_normal", "body_normal"):
        raise ValueError("mode must be flow_normal or body_normal")
    if mode == "flow_normal":
        direction = normalize(_project_perpendicular(body_lateral, velocity_hat), fallback=body_lateral)
    else:
        direction = normalize(_project_perpendicular(body_lateral, body_forward), fallback=body_lateral)
    return scale(direction, float(force_n))


def augment_force_direction(rows: Sequence[Mapping[str, Any]], mode: str) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        velocity = (
            float(row.get("vx", 0.0)),
            float(row.get("vy", 0.0)),
            float(row.get("vz", 0.0)),
        )
        speed = math.sqrt(dot(velocity, velocity))
        velocity_hat = normalize(velocity, fallback=(1.0, 0.0, 0.0))
        axes = body_axes(float(row.get("pitch_rad", 0.0)), float(row.get("yaw_rad", 0.0)))
        force = fin_force_vector(
            float(row.get("fin_force_n", 0.0)),
            mode,
            velocity_hat,
            axes.forward,
            axes.right,
        )
        power = dot(force, velocity)
        row["force_direction_model"] = mode
        row["fin_force_x_n"] = force[0]
        row["fin_force_y_n"] = force[1]
        row["fin_force_z_n"] = force[2]
        row["fin_force_work_w"] = power
        row["fin_axial_projection_n"] = -dot(force, velocity_hat)
        row["fin_force_perpendicular_error_n"] = abs(dot(force, velocity_hat)) if mode == "flow_normal" else None
        row["speed_for_energy_mps"] = speed
        output.append(row)
    return output


def energy_balance(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Compute mechanical-energy residuals when backend drag/thrust exist."""

    ordered = sorted((dict(row) for row in rows), key=lambda row: float(row["time_s"]))
    output: List[Dict[str, Any]] = []
    if len(ordered) < 2:
        return output
    energies: List[float] = []
    for row in ordered:
        mass = float(row.get("mass_kg", 147.87))
        speed = float(row.get("speed_mps", 0.0))
        altitude = float(row.get("y_m", 0.0))
        energies.append(0.5 * mass * speed * speed + mass * 9.80665 * altitude)
    energy_rate = derivative(energies, [float(row["time_s"]) for row in ordered])
    for row, rate in zip(ordered, energy_rate):
        thrust = row.get("thrust_n")
        drag = row.get("drag_n")
        speed = float(row.get("speed_mps", 0.0))
        if not finite(thrust) or not finite(drag):
            row["energy_balance_residual_w"] = None
            row["energy_balance_status"] = "blocked_missing_drag_or_thrust"
        else:
            thrust_power = float(thrust) * speed
            drag_power = -float(drag) * speed
            fin_power = float(row.get("fin_force_work_w", 0.0))
            row["energy_balance_residual_w"] = float(rate) - (thrust_power + drag_power + fin_power)
            row["energy_balance_status"] = "computed"
        output.append(row)
    return output


def _fit_no_intercept(features: Sequence[Sequence[float]], target: Sequence[float]) -> Dict[str, Any]:
    if not features:
        return {"status": "blocked_no_samples", "coefficients": [], "rmse_n": None, "sample_count": 0}
    matrix, rhs = normal_equations(features, target, ridge=1.0e-12)
    try:
        coefficients = solve_linear_system(matrix, rhs)
    except ValueError:
        return {"status": "blocked_singular", "coefficients": [], "rmse_n": None, "sample_count": len(features)}
    residuals = []
    for row, observed in zip(features, target):
        predicted = sum(float(left) * float(right) for left, right in zip(row, coefficients))
        residuals.append(predicted - float(observed))
    return {
        "status": "fit_complete",
        "coefficients": coefficients,
        "rmse_n": rms(residuals),
        "sample_count": len(features),
        "normal_matrix": matrix,
    }


def fit_fin_drag_candidates(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Compare D0/D1/D2 forms without upgrading on training RMSE alone."""

    usable = [
        row for row in rows
        if finite(row.get("extra_drag_residual_n"))
        and finite(row.get("dynamic_pressure_pa"))
        and float(row.get("dynamic_pressure_pa")) > 0.0
    ]
    targets = [float(row["extra_drag_residual_n"]) for row in usable]
    fin_features = [[float(row.get("fin_force_n", 0.0)) ** 2 / float(row["dynamic_pressure_pa"])] for row in usable]
    body_features = [
        [
            float(row.get("body_force_n", 0.0)) ** 2 / float(row["dynamic_pressure_pa"]),
            2.0 * float(row.get("body_force_n", 0.0)) * float(row.get("fin_force_n", 0.0)) / float(row["dynamic_pressure_pa"]),
            float(row.get("fin_force_n", 0.0)) ** 2 / float(row["dynamic_pressure_pa"]),
        ]
        for row in usable
    ]
    none_residual = rms(targets) if targets else None
    fits = {
        "none": {
            "status": "diagnostic",
            "coefficients": [],
            "rmse_n": none_residual,
            "sample_count": len(usable),
        },
        "fin_load_squared": _fit_no_intercept(fin_features, targets),
        "body_fin_quadratic": _fit_no_intercept(body_features, targets),
    }
    return {
        "candidate_fits": fits,
        "upgrade_gate": {
            "training_and_holdout_required": True,
            "status": "blocked_without_whole_trajectory_holdout" if usable else "blocked_no_energy_residuals",
        },
        "status": "diagnostic_only",
    }


def compare_force_directions(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    comparisons: Dict[str, Any] = {}
    for mode in ("flow_normal", "body_normal"):
        augmented = augment_force_direction(rows, mode)
        projections = [float(row["fin_axial_projection_n"]) for row in augmented]
        powers = [float(row["fin_force_work_w"]) for row in augmented]
        comparisons[mode] = {
            "sample_count": len(augmented),
            "projection_rms_n": rms(projections),
            "power_rms_w": rms(powers),
            "projection_mean_n": sum(projections) / len(projections) if projections else None,
            "status": "computed" if augmented else "blocked_no_samples",
        }
    return {
        "models": comparisons,
        "preferred_by_projection_only": (
            "flow_normal"
            if comparisons["flow_normal"].get("projection_rms_n") is not None
            and comparisons["body_normal"].get("projection_rms_n") is not None
            and comparisons["flow_normal"]["projection_rms_n"] <= comparisons["body_normal"]["projection_rms_n"]
            else None
        ),
        "interpretation": "Projection comparison is not a causal StatShark formula claim.",
    }


__all__ = [
    "augment_force_direction",
    "compare_force_directions",
    "energy_balance",
    "fin_force_vector",
    "fit_fin_drag_candidates",
]
