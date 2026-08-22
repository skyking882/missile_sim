"""H2 simulator with explicit geometry, load telemetry, and fin authority."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

from .aerodynamics import (
    body_axes,
    normalize_quaternion,
    pitch_yaw_from_quaternion,
    quaternion_from_pitch_yaw,
)
from .control import base_indicated_speed_schedule, update_control_feedback
from .dynamics import SimState
from .events import event_candidates
from .guidance import guidance_command
from .h2_dynamics import _uses_quaternion_candidate, forces_for_state_h2, rk4_step_h2
from .math3d import Vector, lerp, norm, sub
from .observation import IdealTruthTrackProvider, KinematicTrackProvider, SensorTrackProvider
from .propulsion import PiecewisePropulsion
from .target import TargetModel, TargetState
from .tracking import TrackSolution
from .units import deg_to_rad, kmh_to_mps


_POWERED_VARIANTS = {"full", "power_only", "guidance_no_control"}
_GUIDED_VARIANTS = {"full", "glide_guided", "guidance_no_control"}
_CONTROLLED_VARIANTS = {"full", "glide_guided", "guidance_no_control"}
_FIN_AUTHORITY = {
    "full": 1.0,
    "power_only": 0.0,
    "glide_guided": 1.0,
    "guidance_no_control": 0.0,
}


def _vector_list(vector: Vector) -> list[float]:
    return [float(value) for value in vector]


def _interpolate_state(a: SimState, b: SimState, fraction: float) -> SimState:
    scalar_names = (
        "pitch", "yaw", "pitch_rate", "yaw_rate", "mass",
        "pitch_pid_integral", "yaw_pid_integral", "previous_pitch_error", "previous_yaw_error",
        "pitch_error_derivative", "yaw_error_derivative", "actual_pitch_acceleration_g",
        "actual_yaw_acceleration_g", "actual_pitch_fin_angle_rad", "actual_yaw_fin_angle_rad",
        "pitch_fin_command", "yaw_fin_command",
        "pitch_pid_output", "yaw_pid_output", "pitch_requested_fin_command",
        "yaw_requested_fin_command", "measured_pitch_normal_g", "measured_yaw_normal_g",
        "commanded_pitch_rate_rad_s", "commanded_yaw_rate_rad_s",
        "pitch_rate_error_rad_s", "yaw_rate_error_rad_s",
    )
    values = {
        name: getattr(a, name) + fraction * (getattr(b, name) - getattr(a, name))
        for name in scalar_names
    }
    quaternion = None
    if a.orientation_quaternion is not None and b.orientation_quaternion is not None:
        b_quaternion = b.orientation_quaternion
        if sum(x * y for x, y in zip(a.orientation_quaternion, b_quaternion)) < 0.0:
            b_quaternion = tuple(-value for value in b_quaternion)
        quaternion = normalize_quaternion(
            tuple(
                x + fraction * (y - x)
                for x, y in zip(a.orientation_quaternion, b_quaternion)
            )
        )
        values["pitch"], values["yaw"] = pitch_yaw_from_quaternion(quaternion)
    return SimState(
        position=lerp(a.position, b.position, fraction),
        velocity=lerp(a.velocity, b.velocity, fraction),
        orientation_quaternion=quaternion,
        **values,
    )


def _interpolate_target(a: TargetState, b: TargetState, fraction: float) -> TargetState:
    return TargetState(lerp(a.position, b.position, fraction), lerp(a.velocity, b.velocity, fraction))


def _target_is_finite(target: TargetState) -> bool:
    return all(math.isfinite(value) for value in target.position + target.velocity)


class H2Simulator:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.propulsion = PiecewisePropulsion.from_config(config)

    def initial_state(self, case: dict[str, Any]) -> SimState:
        initial = case["initial_conditions"]
        pitch = deg_to_rad(initial["launch_angle_deg"])
        yaw = deg_to_rad(initial["launch_yaw_deg"])
        axes = body_axes(pitch, yaw)
        velocity = tuple(value * kmh_to_mps(initial["start_speed_kmh"]) for value in axes.forward)
        candidate_orientation = (
            quaternion_from_pitch_yaw(pitch, yaw)
            if _uses_quaternion_candidate(self.config)
            else None
        )
        return SimState(
            position=(0.0, float(initial["launch_altitude_m"]), 0.0),
            velocity=velocity,
            pitch=pitch,
            yaw=yaw,
            pitch_rate=0.0,
            yaw_rate=0.0,
            mass=self.config["geometry"]["initial_mass_kg"],
            orientation_quaternion=candidate_orientation,
        )

    def _sample(
        self,
        time_s: float,
        state: SimState,
        target: TargetState,
        track: TrackSolution,
        provider: Any,
        observation_mode: str,
        case: dict[str, Any],
        powered: bool,
        guided: bool,
        controlled: bool,
        reuse_guidance: Any = None,
    ) -> dict[str, Any]:
        variant = case["model_variant"]
        guidance = (
            reuse_guidance
            if reuse_guidance is not None
            else guidance_command(state, track, time_s, self.config, enabled=guided)
        )
        diagnostics = forces_for_state_h2(state, time_s, self.config, self.propulsion, powered)
        speed_schedule = base_indicated_speed_schedule(diagnostics.aero.dynamic_pressure_pa, self.config)
        relative = sub(target.position, state.position)
        radar_detection = getattr(provider, "radar_detection", None)
        seeker_state = getattr(provider, "seeker_state", None)
        observation_provider = str(getattr(provider, "provider_name", observation_mode))
        observation_reject_reason = str(getattr(provider, "observation_reject_reason", "") or "")
        seeker_display_state = str(
            getattr(
                provider,
                "seeker_display_state",
                seeker_state.value if seeker_state is not None else "not_applicable",
            )
        )
        if radar_detection is None:
            radar_detection_valid = False
            radar_reject_reason = ""
            radar_range_m = None
            radar_ground_radial_speed_mps = None
            radar_off_boresight_deg = None
            radar_los_rate_deg_s = None
            radar_look_down = None
            radar_notch_half_width_mps = None
        else:
            radar_detection_valid = bool(radar_detection.valid)
            radar_reject_reason = str(radar_detection.reason)
            radar_range_m = float(radar_detection.range_m)
            radar_ground_radial_speed_mps = float(radar_detection.ground_radial_speed_mps)
            radar_off_boresight_deg = float(radar_detection.off_boresight_deg)
            radar_los_rate_deg_s = float(radar_detection.los_rate_deg_s)
            radar_look_down = bool(radar_detection.look_down)
            radar_notch_half_width_mps = float(radar_detection.notch_half_width_mps)
        track_error = norm(sub(track.position, target.position))
        generalized_candidate = self.config["aerodynamics"].get(
            "generalized_aero_moment_candidate"
        )
        generalized_candidate_active = (
            self.config["control"].get("plant_semantics") == "generalized_aero_moment"
            and isinstance(generalized_candidate, dict)
        )
        return {
            "time_s": float(time_s),
            "position_m": _vector_list(state.position),
            "velocity_mps": _vector_list(state.velocity),
            "target_position_m": _vector_list(target.position),
            "target_velocity_mps": _vector_list(target.velocity),
            "mass_kg": float(state.mass),
            "thrust_n": float(diagnostics.propulsion.thrust_n),
            "drag_n": float(norm(diagnostics.drag_force_n)),
            "fin_drag_n": float(norm(diagnostics.fin_drag_force_n)),
            "mach": float(diagnostics.aero.mach),
            "dynamic_pressure_pa": float(diagnostics.aero.dynamic_pressure_pa),
            "body_normal_dynamic_pressure_pa": float(
                diagnostics.aero.normal_force_dynamic_pressure_pa
            ),
            "indicated_speed_kmh": float(speed_schedule.indicated_speed_kmh),
            "base_indicated_speed_kmh": speed_schedule.base_indicated_speed_kmh,
            "base_indicated_speed_mode": speed_schedule.mode,
            "base_indicated_speed_q_ratio": float(speed_schedule.dynamic_pressure_ratio),
            "pid_output_speed_scale": float(speed_schedule.pid_output_scale),
            "requested_fin_speed_scale": float(speed_schedule.requested_fin_scale),
            "fin_force_speed_scale": float(speed_schedule.fin_force_scale),
            "angle_of_attack_rad": float(diagnostics.aero.angle_of_attack_rad),
            "pitch_alpha_rad": float(diagnostics.aero.pitch_alpha_rad),
            "yaw_alpha_rad": float(diagnostics.aero.yaw_alpha_rad),
            "cda0_m2": float(diagnostics.aero.cda0_m2),
            "cda_alpha_m2": float(diagnostics.aero.cda_alpha_m2),
            "total_cda_m2": float(diagnostics.aero.total_cda_m2),
            "commanded_acceleration_mps2": _vector_list(guidance.commanded_acceleration_mps2),
            "commanded_acceleration_g": [
                float(guidance.commanded_body_acceleration_g[0]),
                float(guidance.commanded_body_acceleration_g[1]),
            ],
            "commanded_acceleration_g_basis": "compatibility_body_up_right_kinematic_acceleration",
            "controller_command_basis": self.config["control"].get("controller_command_basis"),
            "controller_measurement_basis": self.config["control"].get("feedback_measurement"),
            "controller_specific_force_command_g": list(
                guidance.controller_specific_force_command_g
            ),
            "wind_normal_specific_force_command_g": list(
                guidance.wind_normal_specific_force_command_g
            ),
            "gravity_compensation_wind_normal_g": list(
                guidance.gravity_compensation_wind_normal_g
            ),
            "wind_normal_pitch_axis": _vector_list(guidance.wind_normal_pitch_axis),
            "wind_normal_yaw_axis": _vector_list(guidance.wind_normal_yaw_axis),
            "axial_specific_force_g": float(diagnostics.axial_specific_force_g),
            "pitch_normal_acceleration_g": float(diagnostics.pitch_normal_acceleration_g),
            "yaw_normal_acceleration_g": float(diagnostics.yaw_normal_acceleration_g),
            "body_axis_pitch_specific_force_g": float(diagnostics.pitch_normal_acceleration_g),
            "body_axis_yaw_specific_force_g": float(diagnostics.yaw_normal_acceleration_g),
            "wind_normal_pitch_specific_force_g": float(diagnostics.wind_normal_pitch_acceleration_g),
            "wind_normal_yaw_specific_force_g": float(diagnostics.wind_normal_yaw_acceleration_g),
            "lateral_load_g": float(diagnostics.lateral_load_g),
            "trajectory_pitch_normal_acceleration_g": float(diagnostics.trajectory_pitch_normal_acceleration_g),
            "trajectory_yaw_normal_acceleration_g": float(diagnostics.trajectory_yaw_normal_acceleration_g),
            "trajectory_lateral_load_g": float(diagnostics.trajectory_lateral_load_g),
            "total_specific_force_g": float(diagnostics.total_specific_force_g),
            "actual_overload_g": float(diagnostics.lateral_load_g),
            "drag_power_w": float(diagnostics.drag_power_w),
            "lift_power_w": float(diagnostics.lift_power_w),
            "body_tail_force_power_at_cg_w": float(diagnostics.body_tail_force_power_at_cg_w),
            "pitch_rad": float(state.pitch),
            "yaw_rad": float(state.yaw),
            "pitch_rate_rad_s": float(state.pitch_rate),
            "yaw_rate_rad_s": float(state.yaw_rate),
            "orientation_quaternion_wxyz": (
                None
                if state.orientation_quaternion is None
                else [float(value) for value in state.orientation_quaternion]
            ),
            "distance_to_target_m": float(norm(relative)),
            "current_gain": float(guidance.effective_gain),
            "time_to_go_s": float(guidance.time_to_go_s),
            "closing_speed_mps": float(guidance.closing_speed_mps),
            "los_rate_vector_rad_s": _vector_list(guidance.los_rate_vector_rad_s),
            "loft_active": bool(guidance.loft_active),
            "within_lock_range": bool(guidance.within_lock_range),
            "observation_mode": observation_mode,
            "observation_provider": observation_provider,
            "observation_reject_reason": observation_reject_reason,
            "track_mode": track.mode.value,
            "track_valid": bool(track.valid),
            "track_source": track.source,
            "track_age_s": float(track.age_s),
            "track_position_m": _vector_list(track.position),
            "track_velocity_mps": _vector_list(track.velocity),
            "track_position_error_m": float(track_error),
            "seeker_state": seeker_state.value if seeker_state is not None else "not_applicable",
            "seeker_display_state": seeker_display_state,
            "radar_detection_valid": radar_detection_valid,
            "radar_reject_reason": radar_reject_reason,
            "radar_range_m": radar_range_m,
            "radar_ground_radial_speed_mps": radar_ground_radial_speed_mps,
            "radar_off_boresight_deg": radar_off_boresight_deg,
            "radar_los_rate_deg_s": radar_los_rate_deg_s,
            "radar_look_down": radar_look_down,
            "radar_notch_half_width_mps": radar_notch_half_width_mps,
            "datalink_connected": bool(getattr(provider, "datalink_connected", False)),
            "pitch_fin_command": float(state.pitch_fin_command),
            "yaw_fin_command": float(state.yaw_fin_command),
            "actual_pitch_fin_angle_rad": float(state.actual_pitch_fin_angle_rad),
            "actual_yaw_fin_angle_rad": float(state.actual_yaw_fin_angle_rad),
            "pitch_pid_output": float(state.pitch_pid_output),
            "yaw_pid_output": float(state.yaw_pid_output),
            "pid_output_applied": not (
                str(self.config["control"].get("plant_semantics", "")) == "fin_torque_body_aoa"
                and str(self.config["control"].get("pid_output_semantics", ""))
                == "body_rate_command_rad_s"
            ),
            "pitch_pid_integral": float(state.pitch_pid_integral),
            "yaw_pid_integral": float(state.yaw_pid_integral),
            "pitch_requested_fin_command": float(state.pitch_requested_fin_command),
            "yaw_requested_fin_command": float(state.yaw_requested_fin_command),
            "pitch_requested_fin_angle_rad": float(
                state.pitch_requested_fin_command
                * math.radians(self.config["aerodynamics"]["horizontal_fin_aoa_limit_deg"])
            ),
            "yaw_requested_fin_angle_rad": float(
                state.yaw_requested_fin_command
                * math.radians(self.config["aerodynamics"]["vertical_fin_aoa_limit_deg"])
            ),
            "pid_feedback_pitch_g": float(state.measured_pitch_normal_g),
            "pid_feedback_yaw_g": float(state.measured_yaw_normal_g),
            "commanded_pitch_rate_rad_s": float(state.commanded_pitch_rate_rad_s),
            "commanded_yaw_rate_rad_s": float(state.commanded_yaw_rate_rad_s),
            "pitch_rate_error_rad_s": float(state.pitch_rate_error_rad_s),
            "yaw_rate_error_rad_s": float(state.yaw_rate_error_rad_s),
            "body_reference_area_m2": float(diagnostics.body_reference_area_m2),
            "body_reference_length_m": float(diagnostics.body_reference_length_m),
            "body_cp_cg_arm_over_diameter": float(diagnostics.body_cp_cg_arm_over_diameter),
            "body_cn_alpha_per_rad": float(diagnostics.body_cn_alpha_per_rad),
            "body_cn_q": float(diagnostics.body_cn_q),
            "body_cm_alpha_per_rad": float(diagnostics.body_cm_alpha_per_rad),
            "body_cm_q": float(diagnostics.body_cm_q),
            "generalized_cm_alpha_dot_per_rad": float(
                diagnostics.generalized_cm_alpha_dot_per_rad
            ),
            "generalized_pitch_alpha_dot_hat": float(
                diagnostics.generalized_pitch_alpha_dot_hat
            ),
            "generalized_yaw_alpha_dot_hat": float(
                diagnostics.generalized_yaw_alpha_dot_hat
            ),
            "generalized_cm_alpha_dot_runtime_enabled": bool(
                diagnostics.generalized_cm_alpha_dot_runtime_enabled
            ),
            "generalized_cm_alpha_dot_runtime_status": (
                str(generalized_candidate.get("cm_alpha_dot_runtime_status"))
                if generalized_candidate_active
                else "not_applicable"
            ),
            "generalized_identified_rate_term": (
                "Cm_q only" if generalized_candidate_active else "not_applicable"
            ),
            "generalized_aero_moment_candidate": generalized_candidate,
            "pitch_body_normal_force_n": float(diagnostics.pitch_body_normal_force_n),
            "yaw_body_normal_force_n": float(diagnostics.yaw_body_normal_force_n),
            "pitch_body_static_moment_nm": float(diagnostics.pitch_body_static_moment_nm),
            "yaw_body_static_moment_nm": float(diagnostics.yaw_body_static_moment_nm),
            "pitch_body_rate_moment_nm": float(diagnostics.pitch_body_rate_moment_nm),
            "yaw_body_rate_moment_nm": float(diagnostics.yaw_body_rate_moment_nm),
            "pitch_body_total_moment_nm": float(diagnostics.pitch_body_total_moment_nm),
            "yaw_body_total_moment_nm": float(diagnostics.yaw_body_total_moment_nm),
            "fixed_lifting_surface_multiplier": float(diagnostics.fixed_lifting_surface_multiplier),
            "body_normal_force_area_slope_m2_per_rad": float(diagnostics.body_normal_force_area_slope_m2_per_rad),
            "fixed_lifting_surface_area_slope_m2_per_rad": float(diagnostics.fixed_lifting_surface_area_slope_m2_per_rad),
            "fixed_lifting_surface_station_x_m": float(diagnostics.fixed_lifting_surface_station_x_m),
            "pitch_fixed_lifting_surface_alpha_rad": float(diagnostics.pitch_fixed_lifting_surface_alpha_rad),
            "yaw_fixed_lifting_surface_alpha_rad": float(diagnostics.yaw_fixed_lifting_surface_alpha_rad),
            "pitch_fixed_lifting_surface_force_n": float(diagnostics.pitch_fixed_lifting_surface_force_n),
            "yaw_fixed_lifting_surface_force_n": float(diagnostics.yaw_fixed_lifting_surface_force_n),
            "pitch_fixed_lifting_surface_moment_nm": float(diagnostics.pitch_fixed_lifting_surface_moment_nm),
            "yaw_fixed_lifting_surface_moment_nm": float(diagnostics.yaw_fixed_lifting_surface_moment_nm),
            "pitch_total_body_wing_tail_normal_force_n": float(diagnostics.pitch_total_body_wing_tail_normal_force_n),
            "yaw_total_body_wing_tail_normal_force_n": float(diagnostics.yaw_total_body_wing_tail_normal_force_n),
            "tail_station_x_m": float(diagnostics.tail_station_x_m),
            "tail_force_semantics": (
                "generalized_CN_delta_force"
                if self.config["control"].get("plant_semantics") == "generalized_aero_moment"
                else (
                    "empirical_fin_acceleration_authority"
                    if self.config["control"].get("plant_semantics") == "body_cm_tail_force_moment"
                    else "legacy_equivalent_fin_force"
                )
            ),
            "tail_arm_semantics": self.config["aerodynamics"].get(
                "tail_station_semantics"
            ),
            "empirical_fin_acceleration_authority_g": (
                self.config["aerodynamics"].get("empirical_fin_authority", {}).get(
                    "acceleration_authority_g"
                )
            ),
            "pitch_tail_force_n": float(diagnostics.pitch_tail_force_n),
            "yaw_tail_force_n": float(diagnostics.yaw_tail_force_n),
            "pitch_tail_moment_nm": float(diagnostics.pitch_tail_moment_nm),
            "yaw_tail_moment_nm": float(diagnostics.yaw_tail_moment_nm),
            "pitch_tail_authority_fraction": float(diagnostics.pitch_tail_authority_fraction),
            "yaw_tail_authority_fraction": float(diagnostics.yaw_tail_authority_fraction),
            "tail_alpha_force_multiplier": float(diagnostics.tail_alpha_force_multiplier),
            "tail_delta_force_multiplier": float(diagnostics.tail_delta_force_multiplier),
            "pitch_tail_alpha_force_slope_n_per_rad": float(diagnostics.pitch_tail_alpha_force_slope_n_per_rad),
            "yaw_tail_alpha_force_slope_n_per_rad": float(diagnostics.yaw_tail_alpha_force_slope_n_per_rad),
            "pitch_tail_delta_force_slope_n_per_rad": float(diagnostics.pitch_tail_delta_force_slope_n_per_rad),
            "yaw_tail_delta_force_slope_n_per_rad": float(diagnostics.yaw_tail_delta_force_slope_n_per_rad),
            "pitch_tail_alpha_force_n": float(diagnostics.pitch_tail_alpha_force_n),
            "yaw_tail_alpha_force_n": float(diagnostics.yaw_tail_alpha_force_n),
            "pitch_tail_delta_force_n": float(diagnostics.pitch_tail_delta_force_n),
            "yaw_tail_delta_force_n": float(diagnostics.yaw_tail_delta_force_n),
            "pitch_tail_net_force_pre_cap_n": float(diagnostics.pitch_tail_net_force_pre_cap_n),
            "yaw_tail_net_force_pre_cap_n": float(diagnostics.yaw_tail_net_force_pre_cap_n),
            "tail_force_cap_n": float(diagnostics.tail_force_cap_n),
            "tail_force_cap_scale": float(diagnostics.tail_force_cap_scale),
            "tail_force_cap_active": bool(diagnostics.tail_force_cap_active),
            "pitch_tail_alpha_moment_nm": float(diagnostics.pitch_tail_alpha_moment_nm),
            "yaw_tail_alpha_moment_nm": float(diagnostics.yaw_tail_alpha_moment_nm),
            "pitch_tail_delta_moment_nm": float(diagnostics.pitch_tail_delta_moment_nm),
            "yaw_tail_delta_moment_nm": float(diagnostics.yaw_tail_delta_moment_nm),
            "tail_force_cap_mode": self.config["aerodynamics"].get("split_tail_candidate", {}).get("tail_force_cap_mode"),
            "tail_gain_mass_mode": self.config["aerodynamics"].get("split_tail_candidate", {}).get("tail_gain_mass_mode"),
            "tail_alpha_moment_arm_m": self.config["aerodynamics"].get("split_tail_candidate", {}).get("tail_alpha_moment_arm_m"),
            "tail_delta_moment_arm_m": self.config["aerodynamics"].get("split_tail_candidate", {}).get("tail_delta_moment_arm_m"),
            "fin_mechanical_limit_pitch_rad": self.config["aerodynamics"].get("split_tail_candidate", {}).get("fin_mechanical_limit_pitch_rad"),
            "fin_mechanical_limit_yaw_rad": self.config["aerodynamics"].get("split_tail_candidate", {}).get("fin_mechanical_limit_yaw_rad"),
            "fin_authority_angle_reference_pitch_rad": self.config["aerodynamics"].get("split_tail_candidate", {}).get("fin_authority_angle_reference_pitch_rad"),
            "fin_authority_angle_reference_yaw_rad": self.config["aerodynamics"].get("split_tail_candidate", {}).get("fin_authority_angle_reference_yaw_rad"),
            "pitch_tail_local_alpha_rad": float(
                state.actual_pitch_fin_angle_rad - diagnostics.pitch_tail_effective_incidence_rad
            ),
            "yaw_tail_local_alpha_rad": float(
                state.actual_yaw_fin_angle_rad - diagnostics.yaw_tail_effective_incidence_rad
            ),
            "pitch_residual_damping_moment_nm": float(diagnostics.pitch_residual_damping_moment_nm),
            "yaw_residual_damping_moment_nm": float(diagnostics.yaw_residual_damping_moment_nm),
            "pitch_total_moment_nm": float(diagnostics.pitch_total_moment_nm),
            "yaw_total_moment_nm": float(diagnostics.yaw_total_moment_nm),
            "pitch_body_aoa_force_g": float(diagnostics.pitch_body_aoa_force_g),
            "yaw_body_aoa_force_g": float(diagnostics.yaw_body_aoa_force_g),
            "pitch_fin_moment_equivalent_g": float(diagnostics.pitch_fin_moment_equivalent_g),
            "yaw_fin_moment_equivalent_g": float(diagnostics.yaw_fin_moment_equivalent_g),
            "pitch_tail_rate_incidence_rad": float(diagnostics.pitch_tail_rate_incidence_rad),
            "yaw_tail_rate_incidence_rad": float(diagnostics.yaw_tail_rate_incidence_rad),
            "pitch_tail_effective_incidence_rad": float(diagnostics.pitch_tail_effective_incidence_rad),
            "yaw_tail_effective_incidence_rad": float(diagnostics.yaw_tail_effective_incidence_rad),
            "pitch_natural_frequency_rad_s": float(diagnostics.pitch_natural_frequency_rad_s),
            "yaw_natural_frequency_rad_s": float(diagnostics.yaw_natural_frequency_rad_s),
            "pitch_tail_rate_damping_per_s": float(diagnostics.pitch_tail_rate_damping_per_s),
            "yaw_tail_rate_damping_per_s": float(diagnostics.yaw_tail_rate_damping_per_s),
            "pitch_residual_rate_damping_per_s": float(diagnostics.pitch_residual_rate_damping_per_s),
            "yaw_residual_rate_damping_per_s": float(diagnostics.yaw_residual_rate_damping_per_s),
            "fin_authority_scale": float(_FIN_AUTHORITY[variant]),
            "control_enabled": bool(controlled),
            "model_variant": variant,
            "model_label": self.config["model_label"],
            "aero_model_version": self.config["aero_model_version"],
            "force_geometry_version": self.config["force_geometry_version"],
            "control_model_version": self.config["control_model_version"],
        }

    def run(self, case: dict[str, Any]) -> dict[str, Any]:
        variant = str(case["model_variant"])
        if variant not in _POWERED_VARIANTS | {"glide_guided"}:
            raise ValueError(f"unknown model variant: {variant}")
        powered = variant in _POWERED_VARIANTS
        guided = variant in _GUIDED_VARIANTS
        controlled = variant in _CONTROLLED_VARIANTS
        authority_scale = _FIN_AUTHORITY[variant]
        initial = case["initial_conditions"]
        gravity = self.config["atmosphere"]["gravity_mps2"]
        target_model = case.get("_target_model")
        if target_model is None:
            target_model = TargetModel(initial, gravity)
        launch_position = (0.0, float(initial["launch_altitude_m"]), 0.0)
        state = self.initial_state(case)
        target = target_model.initial_state
        time_s = 0.0
        requested_observation_mode = str(case.get("observation_mode", "ideal_truth"))
        if requested_observation_mode not in {"ideal_truth", "sensor_track"}:
            raise ValueError(f"unknown observation_mode: {requested_observation_mode}")
        observation_mode = requested_observation_mode
        requested_dt = case.get("simulation_dt_s")
        if requested_dt in (None, ""):
            dt_nominal = 1.0 / 256.0 if observation_mode == "sensor_track" else float(self.config["numerics"]["dt_s"])
        else:
            dt_nominal = float(requested_dt)
        if not math.isfinite(dt_nominal) or dt_nominal <= 0.0:
            raise ValueError("simulation_dt_s must be a positive finite number")
        lifetime = float(self.config["performance"]["lifetime_s"])
        # Every propulsion boundary can split one nominal step into an extra
        # partial step.  Budget those splits explicitly; otherwise a two-stage
        # missile at the 1/256 s sensor cadence can exhaust the loop a fraction
        # of a step before lifetime and be mislabeled as numerical_failure.
        boundary_step_budget = len(self.config["propulsion"]["stages"]) if powered else 0
        required_steps = int(math.ceil(lifetime / dt_nominal)) + boundary_step_budget + 1
        max_steps = max(int(self.config["numerics"]["max_steps"]), required_steps)
        if observation_mode == "sensor_track":
            guidance_config = self.config.get("guidance", {})
            sensor_model = guidance_config.get("sensor_model")
            if not isinstance(sensor_model, dict):
                sensor_model = {}
                provider_kind = "profile_kinematic_v1"
            else:
                provider_kind = str(sensor_model.get("provider", "radar_datalink_ins_v1"))
            if provider_kind == "profile_kinematic_v1":
                provider = KinematicTrackProvider(
                    lock_range_m=sensor_model.get("lock_range_m", guidance_config.get("lock_range_m")),
                    maximum_angular_rate_deg_s=sensor_model.get(
                        "maximum_angular_rate_deg_s",
                        guidance_config.get("maximum_angular_rate_deg_s"),
                    ),
                    seeker_type=str(sensor_model.get("seeker_type", guidance_config.get("type", "unknown"))),
                )
            else:
                provider = SensorTrackProvider(
                    sensor_model,
                    lock_range_m=float(guidance_config.get("lock_range_m", 0.0)),
                    datalink_enabled=case.get("datalink_enabled", True),
                    datalink_disconnect_time_s=case.get("datalink_disconnect_time_s"),
                    inertial_drift_direction=tuple(case.get("inertial_drift_direction") or (0.0, 0.0, 1.0)),
                )
        else:
            provider = IdealTruthTrackProvider()
        track = provider.update(time_s, dt_nominal, state, target)
        samples = [self._sample(time_s, state, target, track, provider, observation_mode, case, powered, guided, controlled)]
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
                event_time = time_s
                numerical_failure = True
                break
            state = replace(state, mass=self.propulsion.mass_at(time_s, powered=powered))
            guidance = guidance_command(state, track, time_s, self.config, enabled=guided)
            pre_control_diagnostics = forces_for_state_h2(
                state,
                time_s,
                self.config,
                self.propulsion,
                powered,
            )
            speed_schedule = base_indicated_speed_schedule(
                pre_control_diagnostics.aero.dynamic_pressure_pa,
                self.config,
            )
            feedback = update_control_feedback(
                state,
                guidance.controller_specific_force_command_g,
                self.config,
                step,
                enabled=controlled,
                authority_scale=authority_scale,
                feedback_measurement=self.config["control"].get("feedback_measurement", "physical_normal_g"),
                speed_schedule=speed_schedule,
                plant_diagnostics=pre_control_diagnostics,
            )
            state_for_step = replace(state, **feedback)
            next_state = rk4_step_h2(
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
            from .dynamics import clamp_body_angle_of_attack, state_is_finite

            next_state = clamp_body_angle_of_attack(next_state, self.config)
            next_time = time_s + step
            if hasattr(target_model, "state_at"):
                next_target = target_model.state_at(next_time)
            else:
                next_target = target_model.advance(target, step)
            next_diagnostics = forces_for_state_h2(
                next_state,
                next_time,
                self.config,
                self.propulsion,
                powered,
            )
            feedback_measurement = self.config["control"].get("feedback_measurement", "physical_normal_g")
            if feedback_measurement == "cg_wind_normal_specific_force_g":
                measured_pitch_g = next_diagnostics.wind_normal_pitch_acceleration_g
                measured_yaw_g = next_diagnostics.wind_normal_yaw_acceleration_g
            elif feedback_measurement == "body_specific_force_g":
                measured_pitch_g = next_diagnostics.pitch_normal_acceleration_g
                measured_yaw_g = next_diagnostics.yaw_normal_acceleration_g
            else:
                # Compatibility path for frozen H2 artifacts whose historical
                # "physical_normal_g" telemetry used trajectory-normal axes.
                measured_pitch_g = next_diagnostics.trajectory_pitch_normal_acceleration_g
                measured_yaw_g = next_diagnostics.trajectory_yaw_normal_acceleration_g
            next_state = replace(
                next_state,
                measured_pitch_normal_g=measured_pitch_g,
                measured_yaw_normal_g=measured_yaw_g,
            )
            if not state_is_finite(next_state) or not _target_is_finite(next_target):
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
                    track,
                    provider,
                    observation_mode,
                    case,
                    powered,
                    guided,
                    controlled,
                    reuse_guidance=guidance,
                ))
                event_type = candidate.event_type
                break
            state = next_state
            target = next_target
            time_s = next_time
            track = provider.update(time_s, step, state, target)
            samples.append(self._sample(time_s, state, target, track, provider, observation_mode, case, powered, guided, controlled))
        else:
            event_type = "numerical_failure"
            event_time = time_s
            numerical_failure = True

        if event_type is None:
            event_type = "numerical_failure"
            numerical_failure = True
        if numerical_failure and (not samples or samples[-1]["time_s"] < event_time):
            samples.append(self._sample(event_time, state, target, track, provider, observation_mode, case, powered, guided, controlled))
        return {
            "model_label": self.config["model_label"],
            "aero_model_version": self.config["aero_model_version"],
            "force_geometry_version": self.config["force_geometry_version"],
            "control_model_version": self.config["control_model_version"],
            "case_name": case["name"],
            "model_variant": variant,
            "event_type": event_type,
            "terminal_time_s": float(samples[-1]["time_s"]),
            "time_step_s": dt_nominal,
            "guidance_update_hz": 1.0 / dt_nominal,
            "observation_mode": observation_mode,
            "requested_observation_mode": requested_observation_mode,
            "observation_provider": str(getattr(provider, "provider_name", observation_mode)),
            "datalink_update_count": int(getattr(provider, "datalink_update_count", 0)),
            "integrator": self.config["numerics"]["integrator"],
            "powered": powered,
            "guidance_enabled": guided,
            "control_enabled": controlled,
            "fin_authority_scale": authority_scale,
            "samples": samples,
        }


__all__ = ["H2Simulator"]
