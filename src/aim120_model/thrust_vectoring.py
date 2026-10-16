"""Stateless propulsion control for the game's shared two-axis PID output."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ThrustVectoringOutput:
    aerodynamic_control: tuple[float, float]
    thrust_body_n: tuple[float, float, float]
    moment_body_nm: tuple[float, float, float]
    steering_angles_rad: tuple[float, float]


def thrust_vectoring(
    control: tuple[float, float],
    thrust_n: float,
    max_angle_rad: float,
    aft_distance_m: float,
) -> ThrustVectoringOutput:
    """Map shared PID (horizontal/yaw, vertical/pitch) to force and moment.

    Body axes are x forward, y up, z right. Each command is in [-1, 1];
    thrust is finite and nonnegative; 0 <= max_angle_rad < pi/2 (radians).
    Each axis independently limits its equivalent steering plane angle,
    so the combined deflection can exceed max_angle_rad. The returned
    steering angles equal control * max_angle_rad; actual force plane
    angles have the opposite signs to turn the nose towards the command.

    aft_distance_m is the finite nonnegative distance from the centre of
    mass to the caller-supplied propulsion station r = (-distance, 0, 0).
    Force is on the body, not the exhaust. Moment is r x force: positive
    yaw gives negative My, positive pitch gives positive Mz. No inertia,
    PID state, actuator delay, or aerodynamic scaling is introduced.
    aerodynamic_control preserves the same PID output for the aero path.
    """
    if len(control) != 2 or any(
        not math.isfinite(u) or not -1.0 <= u <= 1.0 for u in control
    ):
        raise ValueError("control must contain two finite values in [-1, 1]")
    if not math.isfinite(thrust_n) or thrust_n < 0.0:
        raise ValueError("thrust_n must be finite and nonnegative")
    if not math.isfinite(max_angle_rad) or not 0.0 <= max_angle_rad < math.pi / 2:
        raise ValueError("max_angle_rad must be finite and in [0, pi/2)")
    if not math.isfinite(aft_distance_m) or aft_distance_m < 0.0:
        raise ValueError("aft_distance_m must be finite and nonnegative")

    yaw, pitch = (u * max_angle_rad for u in control)
    direction_y, direction_z = -math.tan(pitch), -math.tan(yaw)
    force_x = thrust_n / math.hypot(1.0, direction_y, direction_z)
    force = (force_x, force_x * direction_y, force_x * direction_z)
    moment = (0.0, aft_distance_m * force[2], -aft_distance_m * force[1])
    return ThrustVectoringOutput(control, force, moment, (yaw, pitch))
