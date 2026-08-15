"""Forward axial speed replay along observed glide height and path-angle histories."""

from __future__ import annotations

import math
from statistics import mean
from typing import Any, Callable, Mapping, Sequence

from .atmosphere import StandardAtmosphere
from .glide_drag_envelope import LogCdaEnvelope


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _speed_from_row(row: Mapping[str, Any]) -> float:
    if _finite(row.get("speed_mps")):
        return float(row["speed_mps"])
    velocity = row.get("velocity_mps")
    if isinstance(velocity, (list, tuple)) and len(velocity) == 3:
        return math.sqrt(sum(float(value) * float(value) for value in velocity))
    raise ValueError("trajectory row requires speed_mps or velocity_mps")


def _gamma_from_row(row: Mapping[str, Any]) -> float:
    if _finite(row.get("flight_path_angle_rad")):
        return float(row["flight_path_angle_rad"])
    if _finite(row.get("flight_path_angle_deg")):
        return math.radians(float(row["flight_path_angle_deg"]))
    velocity = row.get("velocity_mps")
    if isinstance(velocity, (list, tuple)) and len(velocity) == 3:
        speed = _speed_from_row(row)
        if speed > 1.0e-12:
            return math.asin(max(-1.0, min(1.0, float(velocity[1]) / speed)))
    raise ValueError("trajectory row requires flight-path angle or velocity_mps")


def _altitude_from_row(row: Mapping[str, Any]) -> float:
    if _finite(row.get("altitude_m")):
        return float(row["altitude_m"])
    position = row.get("position_m")
    if isinstance(position, (list, tuple)) and len(position) == 3:
        return float(position[1])
    raise ValueError("trajectory row requires altitude_m or position_m")


def _mass_from_row(row: Mapping[str, Any]) -> float:
    mass = float(row.get("mass_kg", float("nan")))
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError("trajectory row requires positive mass_kg")
    return mass


def _interpolate(left: Mapping[str, Any], right: Mapping[str, Any], fraction: float) -> tuple[float, float, float, float]:
    fraction = max(0.0, min(1.0, float(fraction)))
    altitude = _altitude_from_row(left) + fraction * (_altitude_from_row(right) - _altitude_from_row(left))
    gamma = _gamma_from_row(left) + fraction * (_gamma_from_row(right) - _gamma_from_row(left))
    mass = _mass_from_row(left) + fraction * (_mass_from_row(right) - _mass_from_row(left))
    time_s = float(left["time_s"]) + fraction * (float(right["time_s"]) - float(left["time_s"]))
    return time_s, altitude, gamma, mass


def _rhs(
    speed_mps: float,
    altitude_m: float,
    gamma_rad: float,
    mass_kg: float,
    envelope: LogCdaEnvelope,
    atmosphere: StandardAtmosphere,
    gravity_mps2: float,
) -> tuple[float, dict[str, float]]:
    if speed_mps <= 0.0:
        raise ValueError("replayed speed must remain positive")
    atm = atmosphere.sample(altitude_m)
    mach = speed_mps / atm.speed_of_sound_mps
    cda = envelope.cda_m2(mach)
    q = 0.5 * atm.density_kg_m3 * speed_mps * speed_mps
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


def replay_trajectory(
    rows: Sequence[Mapping[str, Any]],
    envelope: LogCdaEnvelope,
    atmosphere: StandardAtmosphere | None = None,
    gravity_mps2: float = 9.80665,
    max_step_s: float = 0.02,
) -> list[dict[str, Any]]:
    """Replay speed using observed altitude and path angle as exogenous inputs."""

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
        _derivative, values = _rhs(
            speed,
            altitude,
            gamma,
            mass,
            envelope,
            atmosphere_model,
            gravity_mps2,
        )
        observed_speed = _speed_from_row(row)
        drag_accel = values["axial_drag_accel_mps2"]
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
            "mass_kg": mass,
            "gravity_cancellation_ratio": abs(values["gravity_axial_mps2"]) / max(abs(drag_accel), 1.0e-9),
        })
        return values

    output.append(diagnostic(ordered[0], predicted_speed))
    for left, right in zip(ordered, ordered[1:]):
        t0 = float(left["time_s"])
        t1 = float(right["time_s"])
        duration = t1 - t0
        substeps = max(1, int(math.ceil(duration / max(float(max_step_s), 1.0e-6))))
        dt = duration / substeps
        for substep in range(substeps):
            local_t0 = t0 + substep * dt
            local_t1 = local_t0 + dt

            def state_at(time_s: float) -> tuple[float, float, float, float]:
                fraction = (time_s - t0) / duration
                return _interpolate(left, right, fraction)

            def derivative(speed: float, time_s: float) -> float:
                _time, altitude, gamma, mass = state_at(time_s)
                return _rhs(
                    speed,
                    altitude,
                    gamma,
                    mass,
                    envelope,
                    atmosphere_model,
                    gravity_mps2,
                )[0]

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
        return {
            "sample_count": 0,
            "speed_rmse_mps": None,
            "speed_relative_rmse": None,
            "terminal_speed_error_mps": None,
        }
    rmse = math.sqrt(mean(value * value for value in residuals))
    relative_scale = max(mean(abs(value) for value in observed), 1.0e-9)
    return {
        "sample_count": len(residuals),
        "speed_rmse_mps": rmse,
        "speed_relative_rmse": rmse / relative_scale,
        "terminal_speed_error_mps": residuals[-1],
    }


__all__ = ["replay_trajectory", "trajectory_replay_metrics"]
