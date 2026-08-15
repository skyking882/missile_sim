"""Whole-trajectory axial replay and small local optimizer for H5."""

from __future__ import annotations

import math
from statistics import mean
from typing import Any, Iterable, Mapping, Optional, Sequence

from .atmosphere import StandardAtmosphere
from .body_alpha2_drag import (
    BodyAlpha2Parameters,
    H4ShapePrior,
    alpha_rad_from_row,
    body_alpha2_cda,
    body_alpha_power_cda,
)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _speed_from_row(row: Mapping[str, Any]) -> float:
    if _finite(row.get("speed_mps")):
        return float(row["speed_mps"])
    if _finite(row.get("speed_kmh")):
        return float(row["speed_kmh"]) / 3.6
    raise ValueError("trajectory row requires speed_mps or speed_kmh")


def _altitude_from_row(row: Mapping[str, Any]) -> float:
    if _finite(row.get("altitude_m")):
        return float(row["altitude_m"])
    raise ValueError("trajectory row requires altitude_m")


def _gamma_from_row(row: Mapping[str, Any]) -> float:
    if _finite(row.get("flight_path_angle_rad")):
        return float(row["flight_path_angle_rad"])
    if _finite(row.get("flight_path_angle_deg")):
        return math.radians(float(row["flight_path_angle_deg"]))
    return 0.0


def _mass_from_row(row: Mapping[str, Any]) -> float:
    mass = float(row.get("mass_kg", float("nan")))
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError("trajectory row requires positive mass_kg")
    return mass


def _cx_from_row(row: Mapping[str, Any]) -> float:
    value = row.get("cx_aoa", 9.0)
    if not _finite(value):
        raise ValueError("trajectory row requires finite cx_aoa")
    return float(value)


def _interpolate(left: Mapping[str, Any], right: Mapping[str, Any], fraction: float) -> tuple[float, float, float, float, float, float]:
    fraction = max(0.0, min(1.0, float(fraction)))
    return (
        float(left["time_s"]) + fraction * (float(right["time_s"]) - float(left["time_s"])),
        _altitude_from_row(left) + fraction * (_altitude_from_row(right) - _altitude_from_row(left)),
        _gamma_from_row(left) + fraction * (_gamma_from_row(right) - _gamma_from_row(left)),
        _mass_from_row(left) + fraction * (_mass_from_row(right) - _mass_from_row(left)),
        alpha_rad_from_row(left) + fraction * (alpha_rad_from_row(right) - alpha_rad_from_row(left)),
        _cx_from_row(left) + fraction * (_cx_from_row(right) - _cx_from_row(left)),
    )


def _rhs(
    speed_mps: float,
    altitude_m: float,
    gamma_rad: float,
    mass_kg: float,
    alpha_rad: float,
    cx_aoa: float,
    parameters: BodyAlpha2Parameters,
    h4_shape: Optional[H4ShapePrior],
    atmosphere: StandardAtmosphere,
    gravity_mps2: float,
    use_h4_shape: bool,
    alpha_power: Optional[float] = 2.0,
) -> tuple[float, dict[str, float]]:
    if speed_mps <= 0.0 or not math.isfinite(speed_mps):
        raise ValueError("replayed speed must remain positive and finite")
    sample = atmosphere.sample(altitude_m)
    mach = speed_mps / sample.speed_of_sound_mps
    if alpha_power == 2.0:
        cda = body_alpha2_cda(mach, alpha_rad, cx_aoa, parameters, h4_shape, use_h4_shape)
    else:
        cda = body_alpha_power_cda(
            mach,
            alpha_rad,
            cx_aoa,
            parameters,
            alpha_power,
            h4_shape,
            use_h4_shape,
        )
    q = 0.5 * sample.density_kg_m3 * speed_mps * speed_mps
    drag_accel = q * cda / mass_kg
    gravity_axial = gravity_mps2 * math.sin(gamma_rad)
    derivative = -drag_accel - gravity_axial
    return derivative, {
        "mach_pred": mach,
        "dynamic_pressure_pa": q,
        "cda_m2": cda,
        "axial_drag_accel_mps2": drag_accel,
        "gravity_axial_mps2": gravity_axial,
        "dVdt_pred_mps2": derivative,
    }


def replay_body_alpha2(
    rows: Sequence[Mapping[str, Any]],
    parameters: BodyAlpha2Parameters,
    h4_shape: Optional[H4ShapePrior] = None,
    atmosphere: Optional[StandardAtmosphere] = None,
    gravity_mps2: float = 9.80665,
    max_step_s: float = 0.02,
    use_h4_shape: bool = True,
    alpha_power: Optional[float] = 2.0,
) -> list[dict[str, Any]]:
    """Replay speed while using observed altitude, gamma, alpha and CxAoA histories."""

    if len(rows) < 2:
        raise ValueError("at least two trajectory rows are required")
    ordered = sorted((dict(row) for row in rows), key=lambda row: float(row["time_s"]))
    for left, right in zip(ordered, ordered[1:]):
        if float(right["time_s"]) <= float(left["time_s"]):
            raise ValueError("trajectory time must be strictly increasing")
    atmosphere_model = atmosphere or StandardAtmosphere()
    predicted_speed = _speed_from_row(ordered[0])
    output: list[dict[str, Any]] = []

    def diagnostic(row: Mapping[str, Any], speed: float) -> dict[str, Any]:
        altitude = _altitude_from_row(row)
        gamma = _gamma_from_row(row)
        mass = _mass_from_row(row)
        alpha_rad = alpha_rad_from_row(row)
        cx_aoa = _cx_from_row(row)
        _derivative, values = _rhs(
            speed,
            altitude,
            gamma,
            mass,
            alpha_rad,
            cx_aoa,
            parameters,
            h4_shape,
            atmosphere_model,
            gravity_mps2,
            use_h4_shape,
            alpha_power,
        )
        observed_speed = _speed_from_row(row)
        values.update({
            "trajectory_id": str(row.get("trajectory_id", row.get("case_id", "unknown"))),
            "case_id": str(row.get("case_id", row.get("trajectory_id", "unknown"))),
            "source_kind": str(row.get("source_kind", "unknown")),
            "time_s": float(row["time_s"]),
            "observed_speed_mps": observed_speed,
            "predicted_speed_mps": speed,
            "speed_residual_mps": speed - observed_speed,
            "altitude_m": altitude,
            "flight_path_angle_rad": gamma,
            "alpha_rad": alpha_rad,
            "cx_aoa": cx_aoa,
            "mass_kg": mass,
            "gravity_cancellation_ratio": abs(values["gravity_axial_mps2"]) / max(abs(values["axial_drag_accel_mps2"]), 1.0e-9),
        })
        return values

    output.append(diagnostic(ordered[0], predicted_speed))
    for left, right in zip(ordered, ordered[1:]):
        t0 = float(left["time_s"])
        t1 = float(right["time_s"])
        duration = t1 - t0
        substeps = max(1, int(math.ceil(duration / max(float(max_step_s), 1.0e-6))))
        dt = duration / substeps

        def state_at(time_s: float) -> tuple[float, float, float, float, float, float]:
            fraction = (time_s - t0) / duration
            return _interpolate(left, right, fraction)

        def derivative(speed: float, time_s: float) -> float:
            _time, altitude, gamma, mass, alpha_rad, cx_aoa = state_at(time_s)
            return _rhs(
                speed,
                altitude,
                gamma,
                mass,
                alpha_rad,
                cx_aoa,
                parameters,
                h4_shape,
                atmosphere_model,
                    gravity_mps2,
                    use_h4_shape,
                    alpha_power,
            )[0]

        for substep in range(substeps):
            local_t0 = t0 + substep * dt
            local_t1 = local_t0 + dt
            k1 = derivative(predicted_speed, local_t0)
            k2 = derivative(predicted_speed + 0.5 * dt * k1, local_t0 + 0.5 * dt)
            k3 = derivative(predicted_speed + 0.5 * dt * k2, local_t0 + 0.5 * dt)
            k4 = derivative(predicted_speed + dt * k3, local_t1)
            predicted_speed += dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            if predicted_speed <= 0.0 or not math.isfinite(predicted_speed):
                raise ValueError("axial replay produced non-positive or non-finite speed")
        output.append(diagnostic(right, predicted_speed))
    return output


def trajectory_replay_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int | None]:
    residuals = [float(row["speed_residual_mps"]) for row in rows if _finite(row.get("speed_residual_mps"))]
    observed = [float(row["observed_speed_mps"]) for row in rows if _finite(row.get("observed_speed_mps"))]
    if not residuals:
        return {"sample_count": 0, "speed_rmse_mps": None, "speed_relative_rmse": None, "terminal_speed_error_mps": None}
    rmse = math.sqrt(mean(value * value for value in residuals))
    return {
        "sample_count": len(residuals),
        "speed_rmse_mps": rmse,
        "speed_relative_rmse": rmse / max(mean(abs(value) for value in observed), 1.0e-9),
        "terminal_speed_error_mps": residuals[-1],
    }


def _solve_linear_system(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    n = len(vector)
    augmented = [list(float(value) for value in row) + [float(vector[index])] for index, row in enumerate(matrix)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1.0e-15:
            raise ValueError("singular optimizer matrix")
        if pivot != column:
            augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(n):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [left - factor * right for left, right in zip(augmented[row], augmented[column])]
    return [augmented[index][-1] for index in range(n)]


def _parameter_from_vector(values: Sequence[float]) -> BodyAlpha2Parameters:
    return BodyAlpha2Parameters(float(values[0]), float(values[1]), float(values[2]))


def _trajectory_residuals(
    trajectories: Sequence[Sequence[Mapping[str, Any]]],
    vector: Sequence[float],
    h4_shape: Optional[H4ShapePrior],
    use_h4_shape: bool,
    alpha_power: Optional[float],
    max_step_s: float,
) -> list[float]:
    parameters = _parameter_from_vector(vector)
    residuals: list[float] = []
    for trajectory in trajectories:
        replayed = replay_body_alpha2(
            trajectory,
            parameters,
            h4_shape,
            use_h4_shape=use_h4_shape,
            alpha_power=alpha_power,
            max_step_s=max_step_s,
        )
        trajectory_weight = 1.0 / max(len(trajectories), 1)
        sample_weight = math.sqrt(trajectory_weight / max(len(replayed), 1))
        residuals.extend(sample_weight * float(row["speed_residual_mps"]) for row in replayed)
    return residuals


def _loss(residuals: Sequence[float]) -> float:
    return sum(value * value for value in residuals)


def balanced_weighted_rmse(residual_groups: Sequence[Sequence[float]]) -> float:
    """Return equal-trajectory-weighted RMSE for groups of sample residuals."""

    groups = [list(float(value) for value in group) for group in residual_groups if group]
    if not groups:
        return 0.0
    mean_over_trajectories = sum(
        sum(value * value for value in group) / len(group)
        for group in groups
    ) / len(groups)
    return math.sqrt(mean_over_trajectories)


def fit_trajectory_parameters(
    trajectories: Sequence[Sequence[Mapping[str, Any]]],
    initial: BodyAlpha2Parameters,
    h4_shape: Optional[H4ShapePrior] = None,
    use_h4_shape: bool = True,
    max_iterations: int = 12,
    alpha_power: Optional[float] = 2.0,
    max_step_s: float = 0.05,
) -> dict[str, Any]:
    """Fit H5 coefficients by damped finite-difference Gauss-Newton replay."""

    if not trajectories:
        raise ValueError("at least one trajectory is required")
    vector = [initial.cda0_1p5, initial.k_residual_1p5, initial.s_cx_aoa_1p5]
    steps = [1.0e-5, 1.0e-3, 1.0e-4]
    history: list[dict[str, float | int]] = []
    for iteration in range(max_iterations):
        base_residuals = _trajectory_residuals(
            trajectories, vector, h4_shape, use_h4_shape, alpha_power, max_step_s
        )
        base_loss = _loss(base_residuals)
        jacobian: list[list[float]] = []
        for index, step in enumerate(steps):
            perturbed = list(vector)
            perturbed[index] += step
            perturbed_residuals = _trajectory_residuals(
                trajectories, perturbed, h4_shape, use_h4_shape, alpha_power, max_step_s
            )
            jacobian.append([(right - left) / step for left, right in zip(base_residuals, perturbed_residuals)])
        normal = [[0.0, 0.0, 0.0] for _ in range(3)]
        rhs = [0.0, 0.0, 0.0]
        for row_index, residual in enumerate(base_residuals):
            for left in range(3):
                derivative_left = jacobian[left][row_index]
                rhs[left] -= derivative_left * residual
                for right in range(3):
                    normal[left][right] += derivative_left * jacobian[right][row_index]
        for diagonal in range(3):
            normal[diagonal][diagonal] += max(1.0e-12, 1.0e-8 * normal[diagonal][diagonal])
        delta = _solve_linear_system(normal, rhs)
        accepted = False
        best_vector = list(vector)
        best_loss = base_loss
        for scale in (1.0, 0.5, 0.25, 0.1, 0.05):
            candidate = [vector[index] + scale * delta[index] for index in range(3)]
            if candidate[0] <= 0.0:
                continue
            candidate_loss = _loss(
                _trajectory_residuals(
                    trajectories, candidate, h4_shape, use_h4_shape, alpha_power, max_step_s
                )
            )
            if candidate_loss < best_loss:
                best_vector = candidate
                best_loss = candidate_loss
                accepted = True
                break
        history.append({"iteration": iteration, "loss": base_loss, "accepted": int(accepted)})
        vector = best_vector
        if not accepted or max(abs(value) for value in delta) < 1.0e-8:
            break
    final_residuals = _trajectory_residuals(
        trajectories, vector, h4_shape, use_h4_shape, alpha_power, max_step_s
    )
    return {
        "parameters": _parameter_from_vector(vector),
        "loss": _loss(final_residuals),
        # Each residual already carries sqrt(1 / trajectory_count /
        # samples_in_trajectory).  The sum of squares is therefore the
        # balanced mean-over-trajectories / mean-over-samples loss; dividing
        # by the total sample count would weight it a second time.
        "weighted_rmse_mps": math.sqrt(_loss(final_residuals)),
        "iterations": len(history),
        "history": history,
    }


__all__ = [
    "balanced_weighted_rmse",
    "fit_trajectory_parameters",
    "replay_body_alpha2",
    "trajectory_replay_metrics",
]
