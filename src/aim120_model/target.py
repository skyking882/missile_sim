"""Target initialization and simple constant-g/constant-speed motion."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .math3d import Vector, add, scale
from .units import deg_to_rad, g_to_mps2, kmh_to_mps


@dataclass(frozen=True)
class TargetState:
    position: Vector
    velocity: Vector


class TargetModel:
    def __init__(self, initial_conditions: dict[str, Any], gravity_mps2: float):
        self.turn_g = float(initial_conditions.get("target_constant_g_turn", 0.0))
        self.gravity_mps2 = gravity_mps2
        distance = float(initial_conditions["initial_target_distance_m"])
        azimuth = deg_to_rad(initial_conditions["target_azimuth_deg"])
        target_altitude = float(initial_conditions["target_altitude_m"])
        position = (
            distance * math.cos(azimuth),
            target_altitude,
            distance * math.sin(azimuth),
        )
        speed = kmh_to_mps(initial_conditions["target_speed_kmh"])
        course = deg_to_rad(initial_conditions.get("target_course_deg", 0.0))
        course_reference = str(initial_conditions.get("target_course_reference", "absolute_world"))
        if course_reference == "statshark_relative_to_los":
            # StatShark defines TargetCourse relative to the initial line of sight:
            # 0 deg is head-on (toward the launcher) and 180 deg is tail-away.
            course += azimuth + math.pi
        elif course_reference == "sensorwhale_launch_axis":
            # SensorWhale's public calculator uses a fixed launch-axis convention:
            # 0 deg is head-on along world -X, independent of target azimuth.
            # Therefore its native course C is equivalent to our LOS-relative
            # course C - target_azimuth.
            course += math.pi
        elif course_reference != "absolute_world":
            raise ValueError(f"unknown target_course_reference: {course_reference}")
        vertical_course = deg_to_rad(initial_conditions.get("target_vertical_course_deg", 0.0))
        heading = (
            math.cos(vertical_course) * math.cos(course),
            math.sin(vertical_course),
            math.cos(vertical_course) * math.sin(course),
        )
        self.initial_state = TargetState(position, scale(heading, speed))

    def advance(self, state: TargetState, dt_s: float) -> TargetState:
        if abs(self.turn_g) <= 1e-12:
            return TargetState(add(state.position, scale(state.velocity, dt_s)), state.velocity)
        horizontal_speed = math.hypot(state.velocity[0], state.velocity[2])
        if horizontal_speed <= 1e-9:
            return TargetState(add(state.position, scale(state.velocity, dt_s)), state.velocity)
        turn_angle = self.turn_g * g_to_mps2(1.0, self.gravity_mps2) * dt_s / horizontal_speed
        c = math.cos(turn_angle)
        s = math.sin(turn_angle)
        velocity = (
            c * state.velocity[0] - s * state.velocity[2],
            state.velocity[1],
            s * state.velocity[0] + c * state.velocity[2],
        )
        midpoint_velocity = scale(add(state.velocity, velocity), 0.5)
        return TargetState(add(state.position, scale(midpoint_velocity, dt_s)), velocity)


class TabulatedTargetModel:
    """TargetModel-shaped wrapper around an absolute-time trajectory object."""

    def __init__(self, trajectory: Any):
        self.trajectory = trajectory
        self.initial_state = trajectory.state_at(trajectory.start_time_s)

    def state_at(self, time_s: float) -> TargetState:
        return self.trajectory.state_at(time_s)
