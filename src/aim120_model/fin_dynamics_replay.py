"""Whole-trajectory attitude replay and effective-parameter fitting for H6."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .fin_dynamics import FinDynamicsParams, rk4_step
from .h6_utils import derivative, matrix_rank, normal_equations, rms, solve_linear_system, unwrap_angles


def _ordered(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    ordered = [dict(row) for row in sorted(rows, key=lambda item: float(item["time_s"]))]
    if len(ordered) < 2:
        raise ValueError("at least two samples are required")
    for left, right in zip(ordered, ordered[1:]):
        if float(right["time_s"]) <= float(left["time_s"]):
            raise ValueError("trajectory time must be strictly increasing")
    return ordered


def _interp(left: Mapping[str, Any], right: Mapping[str, Any], time_s: float) -> Dict[str, float]:
    t0 = float(left["time_s"])
    t1 = float(right["time_s"])
    fraction = 0.0 if t1 == t0 else max(0.0, min(1.0, (float(time_s) - t0) / (t1 - t0)))
    values: Dict[str, float] = {}
    for key in ("flight_path_yaw_rad", "fin_normal_accel_mps2", "dynamic_pressure_pa"):
        values[key] = float(left.get(key, 0.0)) + fraction * (float(right.get(key, 0.0)) - float(left.get(key, 0.0)))
    return values


def replay_attitude(
    rows: Sequence[Mapping[str, Any]],
    params: FinDynamicsParams,
    distance_cm_to_stabilizer_m: float = 0.175,
    initial_psi_rad: Optional[float] = None,
    initial_yaw_rate_rad_s: Optional[float] = None,
    max_step_s: float = 0.02,
) -> List[Dict[str, Any]]:
    """Replay ``psi/r`` using observed effective fin force as forcing."""

    ordered = _ordered(rows)
    psi = float(initial_psi_rad if initial_psi_rad is not None else ordered[0].get("yaw_rad", 0.0))
    yaw_rate = float(initial_yaw_rate_rad_s if initial_yaw_rate_rad_s is not None else ordered[0].get("yaw_rate_rad_s", 0.0))
    output: List[Dict[str, Any]] = []

    def emit(index: int, predicted_psi: float, predicted_rate: float) -> None:
        source = ordered[index]
        chi = float(source.get("flight_path_yaw_rad", 0.0))
        output.append({
            "case_id": source.get("case_id"),
            "model_id": source.get("model_id"),
            "time_s": float(source["time_s"]),
            "observed_psi_rad": float(source.get("yaw_rad", 0.0)),
            "predicted_psi_rad": predicted_psi,
            "psi_residual_rad": predicted_psi - float(source.get("yaw_rad", 0.0)),
            "observed_yaw_rate_rad_s": float(source.get("yaw_rate_rad_s", 0.0)),
            "predicted_yaw_rate_rad_s": predicted_rate,
            "yaw_rate_residual_rad_s": predicted_rate - float(source.get("yaw_rate_rad_s", 0.0)),
            "observed_beta_yaw_rad": float(source.get("beta_yaw_rad", predicted_psi - chi)),
            "predicted_beta_yaw_rad": predicted_psi - chi,
            "dynamic_pressure_pa": float(source.get("dynamic_pressure_pa", params.q_ref_pa)),
        })

    emit(0, psi, yaw_rate)
    for index, (left, right) in enumerate(zip(ordered, ordered[1:]), start=1):
        duration = float(right["time_s"]) - float(left["time_s"])
        substeps = max(1, int(math.ceil(duration / max(float(max_step_s), 1.0e-6))))
        dt = duration / substeps
        for substep in range(substeps):
            local_time = float(left["time_s"]) + substep * dt
            forcing = _interp(left, right, local_time)
            psi, yaw_rate = rk4_step(
                (psi, yaw_rate),
                forcing,
                dt,
                distance_cm_to_stabilizer_m,
                params,
            )
        emit(index, psi, yaw_rate)
    return output


def _derivative_fit_seed(
    trajectories: Sequence[Sequence[Mapping[str, Any]]],
    distance_cm_to_stabilizer_m: float,
) -> Tuple[FinDynamicsParams, Dict[str, Any]]:
    design: List[List[float]] = []
    target: List[float] = []
    for trajectory in trajectories:
        ordered = _ordered(trajectory)
        rates = [float(row.get("yaw_rate_rad_s", 0.0)) for row in ordered]
        times = [float(row["time_s"]) for row in ordered]
        rate_dot = derivative(rates, times)
        for row, observed_rate_dot in zip(ordered, rate_dot):
            beta = float(row.get("beta_yaw_rad", float(row.get("yaw_rad", 0.0)) - float(row.get("flight_path_yaw_rad", 0.0))))
            accel = float(row.get("fin_normal_accel_mps2", 0.0))
            rate = float(row.get("yaw_rate_rad_s", 0.0))
            design.append([
                float(distance_cm_to_stabilizer_m) * accel,
                -beta,
                -rate,
            ])
            target.append(float(observed_rate_dot))
    if len(design) < 3:
        raise ValueError("at least three dynamic samples are required")
    normal, rhs = normal_equations(design, target, ridge=1.0e-12)
    coefficients = solve_linear_system(normal, rhs)
    params = FinDynamicsParams(coefficients[0], coefficients[1], coefficients[2])
    rank = matrix_rank(design)
    correlations: Dict[str, Optional[float]] = {}
    for left in range(3):
        for right in range(left + 1, 3):
            left_values = [row[left] for row in design]
            right_values = [row[right] for row in design]
            x_mean = sum(left_values) / len(left_values)
            y_mean = sum(right_values) / len(right_values)
            numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(left_values, right_values))
            denominator = math.sqrt(
                sum((x - x_mean) ** 2 for x in left_values)
                * sum((y - y_mean) ** 2 for y in right_values)
            )
            correlations["{}:{}".format(left, right)] = numerator / denominator if denominator > 0.0 else None
    return params, {
        "sample_count": len(design),
        "design_rank": rank,
        "design_column_correlations": correlations,
        "method": "local_derivative_seed_only",
    }


def _replay_residuals(
    trajectories: Sequence[Sequence[Mapping[str, Any]]],
    params: FinDynamicsParams,
    distance_cm_to_stabilizer_m: float,
    max_step_s: float,
) -> List[float]:
    residuals: List[float] = []
    for trajectory in trajectories:
        replayed = replay_attitude(
            trajectory,
            params,
            distance_cm_to_stabilizer_m=distance_cm_to_stabilizer_m,
            max_step_s=max_step_s,
        )
        # Angle and rate are both retained in the objective.  The rate scale
        # keeps a low-rate trajectory from making the angle fit meaningless.
        for row in replayed:
            residuals.append(float(row["psi_residual_rad"]))
            residuals.append(0.25 * float(row["yaw_rate_residual_rad_s"]))
    return residuals


def fit_full_trajectory(
    trajectories: Sequence[Sequence[Mapping[str, Any]]],
    distance_cm_to_stabilizer_m: float = 0.175,
    max_iterations: int = 8,
    max_step_s: float = 0.02,
) -> Dict[str, Any]:
    """Fit effective ``B_f/K_beta/C_r`` then evaluate by full replay."""

    if not trajectories:
        raise ValueError("at least one trajectory is required")
    seed, seed_report = _derivative_fit_seed(trajectories, distance_cm_to_stabilizer_m)
    vector = [seed.b_f_ref, seed.k_beta_ref, seed.c_r_ref]
    finite_steps = [max(abs(value) * 1.0e-4, 1.0e-6) for value in vector]
    history: List[Dict[str, Any]] = []

    def make_params(values: Sequence[float]) -> FinDynamicsParams:
        # Effective stiffness/damping are constrained non-negative; the
        # force gain remains signed so sign errors are visible rather than
        # hidden by an absolute-value operation.
        return FinDynamicsParams(float(values[0]), max(0.0, float(values[1])), max(0.0, float(values[2])))

    for iteration in range(max_iterations):
        base_params = make_params(vector)
        base = _replay_residuals(trajectories, base_params, distance_cm_to_stabilizer_m, max_step_s)
        base_loss = sum(value * value for value in base)
        jacobian: List[List[float]] = []
        for column, step in enumerate(finite_steps):
            perturbed = list(vector)
            perturbed[column] += step
            residuals = _replay_residuals(trajectories, make_params(perturbed), distance_cm_to_stabilizer_m, max_step_s)
            jacobian.append([(right - left) / step for left, right in zip(base, residuals)])
        normal = [[0.0, 0.0, 0.0] for _ in range(3)]
        rhs = [0.0, 0.0, 0.0]
        for row_index, residual in enumerate(base):
            for left in range(3):
                derivative_left = jacobian[left][row_index]
                rhs[left] -= derivative_left * residual
                for right in range(3):
                    normal[left][right] += derivative_left * jacobian[right][row_index]
        for diagonal in range(3):
            normal[diagonal][diagonal] += max(1.0e-12, 1.0e-8 * normal[diagonal][diagonal])
        try:
            delta = solve_linear_system(normal, rhs)
        except ValueError:
            break
        best_vector = list(vector)
        best_loss = base_loss
        accepted = False
        for scale in (1.0, 0.5, 0.25, 0.1, 0.05):
            candidate = [vector[index] + scale * delta[index] for index in range(3)]
            candidate_loss = sum(
                value * value
                for value in _replay_residuals(
                    trajectories,
                    make_params(candidate),
                    distance_cm_to_stabilizer_m,
                    max_step_s,
                )
            )
            if candidate_loss < best_loss:
                best_vector = candidate
                best_loss = candidate_loss
                accepted = True
                break
        history.append({"iteration": iteration, "loss": base_loss, "accepted": accepted})
        vector = best_vector
        if not accepted or max(abs(value) for value in delta) < 1.0e-8:
            break

    params = make_params(vector)
    all_replayed: List[Dict[str, Any]] = []
    angle_residuals: List[float] = []
    rate_residuals: List[float] = []
    for trajectory in trajectories:
        replayed = replay_attitude(
            trajectory,
            params,
            distance_cm_to_stabilizer_m=distance_cm_to_stabilizer_m,
            max_step_s=max_step_s,
        )
        all_replayed.extend(replayed)
        angle_residuals.extend(float(row["psi_residual_rad"]) for row in replayed)
        rate_residuals.extend(float(row["yaw_rate_residual_rad_s"]) for row in replayed)
    max_correlation = max(
        (abs(float(value)) for value in seed_report["design_column_correlations"].values() if value is not None),
        default=0.0,
    )
    return {
        "parameters": params,
        "seed": seed,
        "seed_report": seed_report,
        "fit_method": "full_trajectory_replay_with_derivative_seed",
        "history": history,
        "replayed_rows": all_replayed,
        "angle_rmse_rad": rms(angle_residuals),
        "yaw_rate_rmse_rad_s": rms(rate_residuals),
        "max_abs_design_correlation": max_correlation,
        "identifiability": {
            "design_full_rank": seed_report["design_rank"] >= 3,
            "correlation_gate": max_correlation <= 0.95,
            "status": "pass" if seed_report["design_rank"] >= 3 and max_correlation <= 0.95 else "blocked",
        },
    }


__all__ = ["fit_full_trajectory", "replay_attitude"]
