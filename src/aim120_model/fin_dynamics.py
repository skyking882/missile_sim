"""Reduced effective fin-plant equations used by the isolated H6 model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

from .h6_utils import wrap_angle


@dataclass(frozen=True)
class FinDynamicsParams:
    """Effective parameters; no real inertia tensor is implied."""

    b_f_ref: float
    k_beta_ref: float
    c_r_ref: float
    q_ref_pa: float = 1.0
    b_q_exponent: float = 0.0
    k_q_exponent: float = 0.0
    c_q_exponent: float = 0.0

    def _scale(self, q_pa: float, exponent: float) -> float:
        if exponent == 0.0:
            return 1.0
        q_value = max(float(q_pa), 1.0e-9)
        q_ref = max(float(self.q_ref_pa), 1.0e-9)
        return (q_value / q_ref) ** exponent

    def b_f(self, q_pa: float) -> float:
        return float(self.b_f_ref) * self._scale(q_pa, self.b_q_exponent)

    def k_beta(self, q_pa: float) -> float:
        return float(self.k_beta_ref) * self._scale(q_pa, self.k_q_exponent)

    def c_r(self, q_pa: float) -> float:
        return float(self.c_r_ref) * self._scale(q_pa, self.c_q_exponent)

    def as_dict(self) -> dict[str, float]:
        return {
            "B_f_ref": float(self.b_f_ref),
            "K_beta_ref": float(self.k_beta_ref),
            "C_r_ref": float(self.c_r_ref),
            "q_ref_pa": float(self.q_ref_pa),
            "B_q_exponent": float(self.b_q_exponent),
            "K_beta_q_exponent": float(self.k_q_exponent),
            "C_r_q_exponent": float(self.c_q_exponent),
        }


def angular_acceleration(
    psi_rad: float,
    yaw_rate_rad_s: float,
    flight_path_yaw_rad: float,
    fin_normal_accel_mps2: float,
    distance_cm_to_stabilizer_m: float,
    dynamic_pressure_pa: float,
    params: FinDynamicsParams,
) -> float:
    """Return ``dr/dt`` for the declared effective H6 plant."""

    beta = wrap_angle(float(psi_rad) - float(flight_path_yaw_rad))
    moment_term = params.b_f(dynamic_pressure_pa) * float(distance_cm_to_stabilizer_m) * float(fin_normal_accel_mps2)
    restoring_term = params.k_beta(dynamic_pressure_pa) * beta
    damping_term = params.c_r(dynamic_pressure_pa) * float(yaw_rate_rad_s)
    return moment_term - restoring_term - damping_term


def rhs(
    state: Tuple[float, float],
    forcing: Mapping[str, float],
    distance_cm_to_stabilizer_m: float,
    params: FinDynamicsParams,
) -> Tuple[float, float]:
    psi, yaw_rate = state
    return (
        yaw_rate,
        angular_acceleration(
            psi,
            yaw_rate,
            float(forcing.get("flight_path_yaw_rad", 0.0)),
            float(forcing.get("fin_normal_accel_mps2", 0.0)),
            distance_cm_to_stabilizer_m,
            float(forcing.get("dynamic_pressure_pa", params.q_ref_pa)),
            params,
        ),
    )


def rk4_step(
    state: Tuple[float, float],
    forcing: Mapping[str, float],
    dt_s: float,
    distance_cm_to_stabilizer_m: float,
    params: FinDynamicsParams,
) -> Tuple[float, float]:
    dt = float(dt_s)
    if dt <= 0.0:
        raise ValueError("dt_s must be positive")
    k1 = rhs(state, forcing, distance_cm_to_stabilizer_m, params)
    s2 = (state[0] + 0.5 * dt * k1[0], state[1] + 0.5 * dt * k1[1])
    k2 = rhs(s2, forcing, distance_cm_to_stabilizer_m, params)
    s3 = (state[0] + 0.5 * dt * k2[0], state[1] + 0.5 * dt * k2[1])
    k3 = rhs(s3, forcing, distance_cm_to_stabilizer_m, params)
    s4 = (state[0] + dt * k3[0], state[1] + dt * k3[1])
    k4 = rhs(s4, forcing, distance_cm_to_stabilizer_m, params)
    return (
        state[0] + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
        state[1] + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
    )


__all__ = ["FinDynamicsParams", "angular_acceleration", "rk4_step", "rhs"]
