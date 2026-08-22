"""Fixed-step RK4 H1 simulator with explicit stage and terminal events."""

from __future__ import annotations

import math
from dataclasses import asdict, replace
from datetime import datetime, timezone
from typing import Any

from .aerodynamics import body_axes
from .control import update_control_feedback
from .dynamics import SimState, clamp_rates, forces_for_state, rk4_step, state_is_finite
from .events import event_candidates
from .guidance import GuidanceOutput, guidance_command
from .math3d import Vector, add, lerp, norm, sub
from .propulsion import PiecewisePropulsion
from .target import TargetModel, TargetState
from .units import deg_to_rad, g_to_mps2, kmh_to_mps, mps_to_kmh


_POWERED_VARIANTS = {"full", "power_only", "guidance_no_control"}
_GUIDED_VARIANTS = {"full", "glide_guided", "guidance_no_control"}
_CONTROLLED_VARIANTS = {"full", "glide_guided"}


def _vector_list(vector: Vector) -> list[float]:
    return [float(value) for value in vector]


def _interpolate_state(a: SimState, b: SimState, fraction: float) -> SimState:
    scalar_names = (
        "pitch", "yaw", "pitch_rate", "yaw_rate", "mass",
        "pitch_pid_integral", "yaw_pid_integral", "previous_pitch_error", "previous_yaw_error",
        "pitch_error_derivative", "yaw_error_derivative", "actual_pitch_acceleration_g",
        "actual_yaw_acceleration_g", "actual_pitch_fin_angle_rad", "actual_yaw_fin_angle_rad",
        "pitch_fin_command", "yaw_fin_command",
    )
    values = {name: getattr(a, name) + fraction * (getattr(b, name) - getattr(a, name)) for name in scalar_names}
    return SimState(
        position=lerp(a.position, b.position, fraction),
        velocity=lerp(a.velocity, b.velocity, fraction),
        **values,
    )


def _interpolate_target(a: TargetState, b: TargetState, fraction: float) -> TargetState:
    return TargetState(lerp(a.position, b.position, fraction), lerp(a.velocity, b.velocity, fraction))


class H1Simulator:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.propulsion = PiecewisePropulsion.from_config(config)

    def initial_state(self, case: dict[str, Any]) -> SimState:
        initial = case["initial_conditions"]
        pitch = deg_to_rad(initial["launch_angle_deg"])
        yaw = deg_to_rad(initial["launch_yaw_deg"])
        axes = body_axes(pitch, yaw)
        velocity = tuple(value * kmh_to_mps(initial["start_speed_kmh"]) for value in axes.forward)
        return SimState(
            position=(0.0, float(initial["launch_altitude_m"]), 0.0),
            velocity=velocity,
            pitch=pitch,
            yaw=yaw,
            pitch_rate=0.0,
            yaw_rate=0.0,
            mass=self.config["geometry"]["initial_mass_kg"],
        )

    def _sample(
        self,
        time_s: float,
        state: SimState,
        target: TargetState,
        case: dict[str, Any],
        powered: bool,
        guided: bool,
        reuse_guidance: Any = None,
    ) -> dict[str, Any]:
        variant = case["model_variant"]
        guidance = (
            reuse_guidance
            if reuse_guidance is not None
            else guidance_command(state, target, time_s, self.config, enabled=guided)
        )
        diagnostics = forces_for_state(state, time_s, self.config, self.propulsion, powered)
        relative = sub(target.position, state.position)
        return {
            "time_s": float(time_s),
            "position_m": _vector_list(state.position),
            "velocity_mps": _vector_list(state.velocity),
            "mass_kg": float(state.mass),
            "thrust_n": float(diagnostics.propulsion.thrust_n),
            "drag_n": float(norm(diagnostics.drag_force_n)),
            "mach": float(diagnostics.aero.mach),
            "angle_of_attack_rad": float(diagnostics.aero.angle_of_attack_rad),
            "commanded_acceleration_mps2": _vector_list(guidance.commanded_acceleration_mps2),
            "commanded_acceleration_g": [
                float(guidance.commanded_body_acceleration_g[0]),
                float(guidance.commanded_body_acceleration_g[1]),
            ],
            "actual_overload_g": float(
                norm(diagnostics.non_gravity_acceleration_mps2)
                / max(self.config["atmosphere"]["gravity_mps2"], 1e-12)
            ),
            "pitch_rad": float(state.pitch),
            "yaw_rad": float(state.yaw),
            "pitch_rate_rad_s": float(state.pitch_rate),
            "yaw_rate_rad_s": float(state.yaw_rate),
            "distance_to_target_m": float(norm(relative)),
            "current_gain": float(guidance.effective_gain),
            "closing_speed_mps": float(guidance.closing_speed_mps),
            "los_rate_vector_rad_s": _vector_list(guidance.los_rate_vector_rad_s),
            "loft_active": bool(guidance.loft_active),
            "within_lock_range": bool(guidance.within_lock_range),
            "pitch_fin_command": float(state.pitch_fin_command),
            "yaw_fin_command": float(state.yaw_fin_command),
            "actual_pitch_fin_angle_rad": float(state.actual_pitch_fin_angle_rad),
            "actual_yaw_fin_angle_rad": float(state.actual_yaw_fin_angle_rad),
            "model_variant": variant,
        }

    def run(self, case: dict[str, Any]) -> dict[str, Any]:
        variant = str(case["model_variant"])
        if variant not in _POWERED_VARIANTS | {"glide_guided"}:
            raise ValueError(f"unknown model variant: {variant}")
        powered = variant in _POWERED_VARIANTS
        guided = variant in _GUIDED_VARIANTS
        controlled = variant in _CONTROLLED_VARIANTS
        initial = case["initial_conditions"]
        gravity = self.config["atmosphere"]["gravity_mps2"]
        target_model = TargetModel(initial, gravity)
        launch_position = (0.0, float(initial["launch_altitude_m"]), 0.0)
        state = self.initial_state(case)
        target = target_model.initial_state
        time_s = 0.0
        dt_nominal = float(self.config["numerics"]["dt_s"])
        lifetime = float(self.config["performance"]["lifetime_s"])
        max_steps = int(self.config["numerics"]["max_steps"])
        samples = [self._sample(time_s, state, target, case, powered, guided)]
        event_type: str | None = None
        event_time = time_s
        numerical_failure = False

        for _step_index in range(max_steps):
            if time_s >= lifetime - 1e-12:
                event_type = "lifetime"
                event_time = lifetime
                break
            step = min(dt_nominal, lifetime - time_s)
            if powered:
                boundary = self.propulsion.next_boundary_after(time_s)
                if boundary is not None:
                    step = min(step, boundary - time_s)
            if step <= 1e-12:
                event_type = "numerical_failure"
                numerical_failure = True
                break
            # Keep the exact piecewise mass at every explicit boundary.
            state = replace(state, mass=self.propulsion.mass_at(time_s, powered=powered))
            guidance = guidance_command(state, target, time_s, self.config, enabled=guided)
            feedback = update_control_feedback(
                state,
                guidance.commanded_body_acceleration_g,
                self.config,
                step,
                enabled=controlled,
            )
            state_for_step = replace(state, **feedback)
            next_state = rk4_step(
                state_for_step,
                time_s,
                step,
                self.config,
                self.propulsion,
                powered,
            )
            if powered:
                next_state = replace(next_state, mass=self.propulsion.mass_at(time_s + step, powered=True))
            else:
                next_state = replace(next_state, mass=self.propulsion.initial_mass_kg)
            next_state = clamp_rates(next_state, self.config)
            next_target = target_model.advance(target, step)
            next_time = time_s + step
            if not state_is_finite(next_state) or not state_is_finite(target_state_as_sim_state(next_target)):
                event_type = "numerical_failure"
                event_time = next_time
                numerical_failure = True
                break
            candidates = event_candidates(
                state_for_step,
                next_state,
                target,
                next_target,
                time_s,
                next_time,
                self.config,
                launch_position,
            )
            if candidates:
                candidate = candidates[0]
                fraction = candidate.fraction
                event_state = _interpolate_state(state_for_step, next_state, fraction)
                event_target = _interpolate_target(target, next_target, fraction)
                event_time = time_s + fraction * step
                samples.append(self._sample(
                    event_time,
                    event_state,
                    event_target,
                    case,
                    powered,
                    guided,
                    reuse_guidance=guidance,
                ))
                event_type = candidate.event_type
                break
            state = next_state
            target = next_target
            time_s = next_time
            samples.append(self._sample(time_s, state, target, case, powered, guided))
        else:
            event_type = "numerical_failure"
            event_time = time_s
            numerical_failure = True

        if event_type is None:
            event_type = "numerical_failure"
            numerical_failure = True
        if numerical_failure and (not samples or samples[-1]["time_s"] < event_time):
            samples.append(self._sample(event_time, state, target, case, powered, guided))
        return {
            "model_label": self.config["model_label"],
            "case_name": case["name"],
            "model_variant": variant,
            "event_type": event_type,
            "terminal_time_s": float(samples[-1]["time_s"]),
            "time_step_s": dt_nominal,
            "integrator": self.config["numerics"]["integrator"],
            "powered": powered,
            "guidance_enabled": guided,
            "control_enabled": controlled,
            "samples": samples,
        }


def target_state_as_sim_state(target: TargetState) -> SimState:
    """Adapter used only for finite-value checking of target position/velocity."""

    return SimState(target.position, target.velocity, 0.0, 0.0, 0.0, 0.0, 1.0)


# Public compatibility export: callers may use either ``h2_simulator`` or
# the original simulator module as the entry point for the staged H2 model.
from .h2_simulator import H2Simulator

__all__ = ["H1Simulator", "H2Simulator", "target_state_as_sim_state"]
