"""Ideal truth, radar/DL/INS, and fixed-point observation providers."""

from __future__ import annotations

import math
from typing import Any

from .aerodynamics import body_axes
from .math3d import Vector, add, as_vector, clamp, cross, dot, norm, normalize, scale, sub
from .radar_seeker import RadarDetection, RadarSeekerObserver, SeekerState
from .target import TargetState
from .tracking import TrackMode, TrackSolution
from .units import rad_to_deg


ZERO_VECTOR: Vector = (0.0, 0.0, 0.0)


def _copy_solution(
    position: Vector,
    velocity: Vector,
    time_s: float,
    mode: TrackMode,
    valid: bool,
    source: str,
    sample_time_s: float | None = None,
) -> TrackSolution:
    return TrackSolution(
        position=tuple(float(value) for value in position),
        velocity=tuple(float(value) for value in velocity),
        sample_time_s=float(time_s if sample_time_s is None else sample_time_s),
        solution_time_s=float(time_s),
        mode=mode,
        valid=valid,
        source=source,
    )


class IdealTruthTrackProvider:
    provider_name = "ideal_truth"

    def update(
        self,
        time_s: float,
        dt_s: float,
        missile_state: Any,
        target_truth: TargetState,
    ) -> TrackSolution:
        del dt_s, missile_state
        return _copy_solution(
            target_truth.position,
            target_truth.velocity,
            time_s,
            TrackMode.IDEAL_TRUTH,
            True,
            "target_truth",
        )


class InertialTrackPropagator:
    def __init__(
        self,
        drift_speed_mps: float = 0.0,
        drift_direction: Vector = (0.0, 0.0, 1.0),
        break_lock_max_time_s: float = math.inf,
        enabled: bool = True,
    ):
        self.drift_speed_mps = float(drift_speed_mps)
        self.drift_direction = normalize(as_vector(tuple(drift_direction)))
        self.break_lock_max_time_s = float(break_lock_max_time_s)
        self.enabled = bool(enabled)
        if not math.isfinite(self.drift_speed_mps) or self.drift_speed_mps < 0.0:
            raise ValueError("inertial drift speed must be a finite non-negative number")
        if not math.isfinite(self.break_lock_max_time_s) and self.break_lock_max_time_s != math.inf:
            raise ValueError("break_lock_max_time_s must be finite or infinity")
        self._initialized = False
        self._position_estimate: Vector = ZERO_VECTOR
        self._velocity_estimate: Vector = ZERO_VECTOR
        self._error: Vector = ZERO_VECTOR
        self._last_observation_time_s = 0.0
        self._last_update_time_s = 0.0
        self._sample_time_s = 0.0

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def last_observation_time_s(self) -> float:
        return self._last_observation_time_s

    def reset(self, solution: TrackSolution) -> None:
        if not solution.valid:
            return
        self._position_estimate = tuple(float(value) for value in solution.position)
        self._velocity_estimate = tuple(float(value) for value in solution.velocity)
        self._error = ZERO_VECTOR
        self._last_observation_time_s = float(solution.solution_time_s)
        self._last_update_time_s = float(solution.solution_time_s)
        self._sample_time_s = float(solution.sample_time_s)
        self._initialized = True

    def update(self, time_s: float, dt_s: float) -> TrackSolution:
        time_s = float(time_s)
        if not self._initialized or not self.enabled:
            return _copy_solution(ZERO_VECTOR, ZERO_VECTOR, time_s, TrackMode.LOST, False, "inertial")
        elapsed = max(0.0, time_s - self._last_update_time_s)
        if elapsed <= 0.0 and math.isfinite(float(dt_s)) and float(dt_s) > 0.0:
            elapsed = float(dt_s) if time_s > self._last_update_time_s else 0.0
        self._position_estimate = add(self._position_estimate, scale(self._velocity_estimate, elapsed))
        self._error = add(self._error, scale(self.drift_direction, self.drift_speed_mps * elapsed))
        self._last_update_time_s = time_s
        age_s = max(0.0, time_s - self._last_observation_time_s)
        valid = age_s <= self.break_lock_max_time_s
        mode = TrackMode.INERTIAL if valid else TrackMode.LOST
        position = add(self._position_estimate, self._error)
        return _copy_solution(
            position,
            self._velocity_estimate,
            time_s,
            mode,
            valid,
            "inertial",
            sample_time_s=self._sample_time_s,
        )


class SnsFixedPointProvider:
    def __init__(self, target_point: Vector, fixed_error_vector: Vector = ZERO_VECTOR):
        self.target_point = as_vector(tuple(target_point))
        self.fixed_error_vector = as_vector(tuple(fixed_error_vector))
        self.fixed_position = add(self.target_point, self.fixed_error_vector)

    def update(self, time_s: float, dt_s: float, *args: Any, **kwargs: Any) -> TrackSolution:
        del dt_s, args, kwargs
        return _copy_solution(
            self.fixed_position,
            ZERO_VECTOR,
            time_s,
            TrackMode.SNS_FIXED_POINT,
            True,
            "sns_fixed_point",
        )


class KinematicTrackProvider:
    """Deterministic fallback track for profiles without sensor datamine.

    This provider deliberately uses only profile-level guidance geometry that
    already exists in every runnable profile.  It does not invent radar,
    Doppler, RCS, noise, or seeker-specific constants.  Profiles with an
    explicit ``sensor_model`` continue to use ``SensorTrackProvider``.
    """

    provider_name = "profile_kinematic_v1"

    def __init__(
        self,
        lock_range_m: float | None = None,
        maximum_angular_rate_deg_s: float | None = None,
        seeker_type: str = "unknown",
    ):
        self.lock_range_m = None if lock_range_m is None else float(lock_range_m)
        self.maximum_angular_rate_deg_s = (
            None if maximum_angular_rate_deg_s is None else float(maximum_angular_rate_deg_s)
        )
        if self.lock_range_m is not None and (
            not math.isfinite(self.lock_range_m) or self.lock_range_m <= 0.0
        ):
            raise ValueError("kinematic lock_range_m must be positive and finite")
        if self.maximum_angular_rate_deg_s is not None and (
            not math.isfinite(self.maximum_angular_rate_deg_s)
            or self.maximum_angular_rate_deg_s < 0.0
        ):
            raise ValueError("kinematic maximum_angular_rate_deg_s must be finite and non-negative")
        self.seeker_type = str(seeker_type)
        self.update_count = 0
        self.datalink_connected = False
        self.datalink_update_count = 0
        self.radar_detection: RadarDetection | None = None
        self._seeker_state = SeekerState.SEARCH
        self.observation_reject_reason = ""
        self.last_solution: TrackSolution | None = None
        self.inertial = InertialTrackPropagator(
            drift_speed_mps=0.0,
            break_lock_max_time_s=math.inf,
            enabled=True,
        )
        self.first_lock_loss_time_s: float | None = None
        self.first_reacquire_time_s: float | None = None

    @property
    def seeker_state(self) -> SeekerState:
        return self._seeker_state

    @property
    def seeker_display_state(self) -> str:
        if self.last_solution is not None and self.last_solution.mode is TrackMode.PROFILE_KINEMATIC:
            return "TRK"
        if self.last_solution is not None and self.last_solution.mode is TrackMode.INS_SEARCH:
            return "INS+SRC"
        if self.last_solution is not None and self.last_solution.mode is TrackMode.RADAR_SEARCH:
            return "SRC"
        if self.last_solution is not None and self.last_solution.mode is TrackMode.LOST:
            return "LOST"
        return "SRC" if self._seeker_state is SeekerState.SEARCH else "TRK"

    def _lost(self, time_s: float, reason: str) -> TrackSolution:
        if self._seeker_state is SeekerState.TRACK and self.first_lock_loss_time_s is None:
            self.first_lock_loss_time_s = float(time_s)
        self._seeker_state = SeekerState.SEARCH
        self.observation_reject_reason = str(reason)
        solution = _copy_solution(
            ZERO_VECTOR,
            ZERO_VECTOR,
            time_s,
            TrackMode.LOST,
            False,
            self.provider_name,
        )
        self.last_solution = solution
        return solution

    def update(
        self,
        time_s: float,
        dt_s: float,
        missile_state: Any,
        target_truth: TargetState,
    ) -> TrackSolution:
        time_s = float(time_s)
        self.update_count += 1
        relative = sub(target_truth.position, missile_state.position)
        range_sq = dot(relative, relative)
        range_m = math.sqrt(max(0.0, range_sq))
        if not math.isfinite(range_m) or range_m <= 1.0e-12:
            return self._lost(time_s, "invalid_geometry")

        if self.lock_range_m is not None and range_m > self.lock_range_m:
            reason = "range_gate"
        else:
            reason = ""

        relative_velocity = sub(target_truth.velocity, missile_state.velocity)
        los_rate_rad_s = norm(cross(relative, relative_velocity)) / max(range_sq, 1.0e-12)
        los_rate_deg_s = rad_to_deg(los_rate_rad_s)
        if not reason and self.maximum_angular_rate_deg_s is not None and los_rate_deg_s > self.maximum_angular_rate_deg_s:
            reason = "rate_gate"

        if reason:
            if self._seeker_state is SeekerState.TRACK and self.first_lock_loss_time_s is None:
                self.first_lock_loss_time_s = time_s
            self._seeker_state = SeekerState.SEARCH
            self.observation_reject_reason = reason
            if self.inertial.initialized:
                inertial_solution = self.inertial.update(time_s, dt_s)
                if inertial_solution.valid:
                    solution = _copy_solution(
                        inertial_solution.position,
                        inertial_solution.velocity,
                        time_s,
                        TrackMode.INS_SEARCH,
                        True,
                        f"{self.provider_name}:inertial",
                        sample_time_s=inertial_solution.sample_time_s,
                    )
                    self.last_solution = solution
                    return solution
            return self._lost(time_s, reason)

        # Keep the geometric boresight calculation explicit for diagnostics,
        # while avoiding a guessed angle gate when the profile did not declare one.
        los_hat = normalize(relative)
        forward = body_axes(missile_state.pitch, missile_state.yaw).forward
        _off_boresight_rad = math.acos(clamp(dot(forward, los_hat), -1.0, 1.0))
        self.observation_reject_reason = ""
        if self.first_lock_loss_time_s is not None and self.first_reacquire_time_s is None:
            self.first_reacquire_time_s = time_s
        self._seeker_state = SeekerState.TRACK
        self.inertial.reset(
            _copy_solution(
                target_truth.position,
                target_truth.velocity,
                time_s,
                TrackMode.PROFILE_KINEMATIC,
                True,
                self.provider_name,
            )
        )
        solution = _copy_solution(
            target_truth.position,
            target_truth.velocity,
            time_s,
            TrackMode.PROFILE_KINEMATIC,
            True,
            self.provider_name,
        )
        self.last_solution = solution
        return solution


class SensorTrackProvider:
    """Compose radar, per-tick datalink, and deterministic INS propagation."""

    provider_name = "radar_datalink_ins_v1"

    def __init__(
        self,
        sensor_model: dict[str, Any],
        lock_range_m: float | None = None,
        datalink_enabled: bool | None = None,
        datalink_disconnect_time_s: float | None = None,
        inertial_drift_direction: Vector = (0.0, 0.0, 1.0),
    ):
        if not isinstance(sensor_model, dict):
            raise ValueError("sensor_track requires a sensor_model object")
        self.sensor_model = sensor_model
        self.radar = RadarSeekerObserver(sensor_model, lock_range_m=lock_range_m)
        inertial_value = sensor_model.get("inertial_navigation", sensor_model.get("inertialNavigation"))
        inertial_navigation = True if inertial_value is None else bool(inertial_value)
        drift_value = sensor_model.get("inertial_drift_speed_mps", sensor_model.get("inertialNavigationDriftSpeed"))
        drift_speed = 0.0 if drift_value is None else float(drift_value)
        break_lock_value = sensor_model.get("break_lock_max_time_s", sensor_model.get("breakLockMaxTime"))
        break_lock = math.inf if break_lock_value is None else float(break_lock_value)
        self.inertial = InertialTrackPropagator(
            drift_speed_mps=drift_speed,
            drift_direction=inertial_drift_direction,
            break_lock_max_time_s=break_lock,
            enabled=inertial_navigation,
        )
        datalink_value = sensor_model.get("datalink")
        configured_datalink = True if datalink_value is None else bool(datalink_value)
        self.datalink_enabled = configured_datalink if datalink_enabled is None else bool(datalink_enabled)
        self.datalink_connected = self.datalink_enabled
        self.datalink_disconnect_time_s = datalink_disconnect_time_s
        reconnect_value = sensor_model.get("reconnect_datalink", sensor_model.get("reconnectDatalink"))
        self.reconnect_datalink = False if reconnect_value is None else bool(reconnect_value)
        self.radar_has_tracked_once = False
        self.update_count = 0
        self.datalink_update_count = 0
        self.last_detection: RadarDetection | None = None
        self.last_solution: TrackSolution | None = None
        self.observation_reject_reason = ""
        self.first_lock_loss_time_s: float | None = None
        self.first_reacquire_time_s: float | None = None

    @property
    def seeker_state(self) -> SeekerState:
        return self.radar.state

    @property
    def seeker_display_state(self) -> str:
        solution = self.last_solution
        if solution is not None and solution.mode is TrackMode.RADAR_TRACK:
            return "TRK"
        if solution is not None and solution.mode is TrackMode.INS_SEARCH:
            return "INS+SRC"
        if solution is not None and solution.mode is TrackMode.RADAR_SEARCH:
            return "SRC"
        if solution is not None and solution.mode is TrackMode.DATALINK:
            return "DL+INS"
        if solution is not None and solution.mode is TrackMode.INERTIAL:
            return "INS"
        return "LOST"

    @property
    def radar_detection(self) -> RadarDetection | None:
        return self.last_detection

    @property
    def radar_notch_half_width_mps(self) -> float:
        return self.radar.notch_half_width_mps

    def _refresh_datalink_state(self, time_s: float) -> None:
        if not self.datalink_enabled:
            self.datalink_connected = False
            return
        if self.datalink_disconnect_time_s is not None and time_s >= float(self.datalink_disconnect_time_s):
            self.datalink_connected = False
        if self.radar_has_tracked_once and not self.reconnect_datalink:
            self.datalink_connected = False

    @staticmethod
    def _datalink_solution(time_s: float, target_truth: TargetState) -> TrackSolution:
        return _copy_solution(
            target_truth.position,
            target_truth.velocity,
            time_s,
            TrackMode.DATALINK,
            True,
            "datalink",
        )

    @staticmethod
    def _lost_solution(time_s: float, position: Vector = ZERO_VECTOR, velocity: Vector = ZERO_VECTOR) -> TrackSolution:
        return _copy_solution(position, velocity, time_s, TrackMode.LOST, False, "lost")

    def update(
        self,
        time_s: float,
        dt_s: float,
        missile_state: Any,
        target_truth: TargetState,
    ) -> TrackSolution:
        time_s = float(time_s)
        self.update_count += 1
        self._refresh_datalink_state(time_s)
        detection, radar_solution = self.radar.update(time_s, dt_s, missile_state, target_truth)
        self.last_detection = detection
        self.observation_reject_reason = "" if detection.valid else detection.reason
        if detection.valid and radar_solution is not None and radar_solution.mode is TrackMode.RADAR_TRACK:
            if self.last_solution is not None and self.last_solution.mode is TrackMode.INS_SEARCH:
                if self.first_reacquire_time_s is None:
                    self.first_reacquire_time_s = time_s
            if self.first_lock_loss_time_s is None and self.last_solution is not None and self.last_solution.valid:
                # The first valid radar solution is not a loss/reacquisition.
                pass
            self.radar_has_tracked_once = True
            self._refresh_datalink_state(time_s)
            self.inertial.reset(radar_solution)
            self.last_solution = radar_solution
            return radar_solution

        self._refresh_datalink_state(time_s)
        if self.datalink_connected:
            self.datalink_update_count += 1
            datalink_solution = self._datalink_solution(time_s, target_truth)
            self.inertial.reset(datalink_solution)
            self.last_solution = datalink_solution
            return datalink_solution

        if self.inertial.enabled and self.inertial.initialized:
            inertial_solution = self.inertial.update(time_s, dt_s)
            if inertial_solution.valid and self.radar.active:
                if self.first_lock_loss_time_s is None:
                    self.first_lock_loss_time_s = time_s
                inertial_solution = _copy_solution(
                    inertial_solution.position,
                    inertial_solution.velocity,
                    time_s,
                    TrackMode.INS_SEARCH,
                    True,
                    "inertial_search",
                    sample_time_s=inertial_solution.sample_time_s,
                )
            self.last_solution = inertial_solution
            return inertial_solution

        if self.radar.active:
            self.last_solution = _copy_solution(
                ZERO_VECTOR,
                ZERO_VECTOR,
                time_s,
                TrackMode.RADAR_SEARCH,
                False,
                "radar_search",
            )
            return self.last_solution

        self.last_solution = self._lost_solution(time_s)
        return self.last_solution


__all__ = [
    "IdealTruthTrackProvider",
    "InertialTrackPropagator",
    "KinematicTrackProvider",
    "SensorTrackProvider",
    "SnsFixedPointProvider",
]
