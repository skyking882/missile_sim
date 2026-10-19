"""Ideal truth, radar/DL/INS, and fixed-point observation providers."""

from __future__ import annotations

import math
from typing import Any
from dataclasses import replace

from .aerodynamics import body_axes_for_state
from .math3d import Vector, add, as_vector, clamp, cross, dot, norm, normalize, scale, sub
from .radar_seeker import RadarDetection, RadarSeekerObserver, SeekerState
from .target import TargetState
from .tracking import (AlphaBetaStateObserver, DatalinkObservation, TrackMode, TrackSolution,
                       TrackIdentity, ObserverStageSnapshot, INTENDED_TARGET, reflector_identity)
from .units import rad_to_deg
from .aerodynamics import body_axes, body_axes_from_quaternion
from .radar_geometry import limited_antenna_direction, within_circular_beam, validate_launcher_radar_geometry


ZERO_VECTOR: Vector = (0.0, 0.0, 0.0)


def _available_datalink_time(sample_time_s: float, time_s: float) -> float:
    """Treat only clock roundoff as simultaneous; preserve the raw packet stamp."""
    if sample_time_s > time_s and not math.isclose(sample_time_s, time_s, rel_tol=0., abs_tol=1e-12):
        raise ValueError("datalink observation timestamp must not be in the future")
    return min(sample_time_s, time_s)


def validate_target_observer(mode: str | None) -> str | None:
    if mode not in (None, "unified_alpha_beta"):
        raise ValueError("target_observer must be None or 'unified_alpha_beta'")
    return mode


def validate_min_altitude(min_altitude_m: float | None) -> float | None:
    if min_altitude_m is None:
        return None
    if isinstance(min_altitude_m, bool) or not isinstance(min_altitude_m, (int, float)) \
            or not math.isfinite(min_altitude_m):
        raise ValueError("min_altitude_m must be a finite number or None")
    return float(min_altitude_m)


def _copy_solution(
    position: Vector,
    velocity: Vector,
    time_s: float,
    mode: TrackMode,
    valid: bool,
    source: str,
    sample_time_s: float | None = None,
    *, template: TrackSolution | None = None, identity: TrackIdentity | None = None,
) -> TrackSolution:
    return TrackSolution(
        position=tuple(float(value) for value in position),
        velocity=tuple(float(value) for value in velocity),
        sample_time_s=float(time_s if sample_time_s is None else sample_time_s),
        solution_time_s=float(time_s),
        mode=mode,
        valid=valid,
        source=source,
        full_rate_rad_s=None if template is None else template.full_rate_rad_s,
        identity=identity if template is None else template.identity,
        estimate_source=source if template is None else (template.estimate_source or template.source),
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
    """Coast the last observation; optional floor applies to estimated y-up altitude.

    ``min_altitude_m=None`` retains unrestricted propagation. A floor clips only
    propagated estimates (including drift), zeroing their vertical velocity;
    reset observations keep their measured position and velocity unchanged.
    """

    def __init__(
        self,
        drift_speed_mps: float = 0.0,
        drift_direction: Vector = (0.0, 0.0, 1.0),
        break_lock_max_time_s: float = math.inf,
        enabled: bool = True,
        min_altitude_m: float | None = None,
    ):
        self.min_altitude_m = validate_min_altitude(min_altitude_m)
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
        self.estimate: TrackSolution | None = None

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def last_observation_time_s(self) -> float:
        return self._last_observation_time_s

    def reset(self, solution: TrackSolution) -> None:
        if not solution.valid:
            return
        self.estimate = solution
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
        if self.min_altitude_m is not None and position[1] < self.min_altitude_m:
            self._position_estimate = (
                self._position_estimate[0], self.min_altitude_m - self._error[1], self._position_estimate[2]
            )
            self._velocity_estimate = (self._velocity_estimate[0], 0.0, self._velocity_estimate[2])
            position = (position[0], self.min_altitude_m, position[2])
        return _copy_solution(
            position,
            self._velocity_estimate,
            time_s,
            mode,
            valid,
            "inertial",
            sample_time_s=self._sample_time_s,
            # Keep the accepted angular state in estimate for handover memory,
            # but Cartesian INS does not publish a frozen former TRK rate.
            template=(replace(self.estimate, full_rate_rad_s=None)
                if self.estimate is not None and self.estimate.mode in
                    (TrackMode.RADAR_TRACK, TrackMode.RADAR_COAST) else self.estimate),
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
        min_altitude_m: float | None = None,
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
            min_altitude_m=min_altitude_m,
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
        forward = body_axes_for_state(missile_state).forward
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
    supports_decoys = True

    def __init__(
        self,
        sensor_model: dict[str, Any],
        lock_range_m: float | None = None,
        datalink_enabled: bool | None = None,
        datalink_disconnect_time_s: float | None = None,
        inertial_drift_direction: Vector = (0.0, 0.0, 1.0),
        min_altitude_m: float | None = None,
        *, strict_autonomous: bool = False, observer_diagnostics: bool = False,
        target_identity: TrackIdentity | None = None, launch_global_tick: int = 0,
    ):
        if not isinstance(sensor_model, dict):
            raise ValueError("sensor_track requires a sensor_model object")
        self.sensor_model = sensor_model
        self.strict_autonomous = bool(strict_autonomous)
        self.observer_diagnostics = bool(observer_diagnostics)
        if target_identity is not None and not isinstance(target_identity, TrackIdentity):
            raise TypeError("target_identity must be TrackIdentity or None")
        self.intended_target_identity = target_identity or INTENDED_TARGET
        self.selected_reflector_identity = None
        self.estimator_identity = None
        self.ins_estimate = None
        self.measurement_cache = None
        self.antenna_axis_world = None
        self.target_los_estimate_world = None
        self.stage_snapshots: tuple[ObserverStageSnapshot, ...] = ()
        self._previous_accepted_time_s = None
        self._current_detection = None
        self._datalink_prior = None
        self._strict_datalink_is_new = False
        self.radar = RadarSeekerObserver(sensor_model, lock_range_m=lock_range_m,
            launch_global_tick=launch_global_tick)
        self.radar.target_identity = self.intended_target_identity
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
            min_altitude_m=min_altitude_m,
        )
        datalink_value = sensor_model.get("datalink")
        configured_datalink = True if datalink_value is None else bool(datalink_value)
        self.datalink_enabled = configured_datalink if datalink_enabled is None else bool(datalink_enabled)
        self.datalink_connected = self.datalink_enabled
        self.datalink_disconnect_time_s = datalink_disconnect_time_s
        reconnect_value = sensor_model.get("reconnect_datalink", sensor_model.get("reconnectDatalink"))
        self.reconnect_datalink = False if reconnect_value is None else bool(reconnect_value)
        self.radar_has_tracked_once = False
        # Opt-in (enable_launcher_radar): the datalink lasts only while the launching
        # aircraft's own radar still holds the target; off by default.
        self.launcher_radar = None
        self.launcher_support = None
        self.datalink_estimate: tuple[Vector, Vector] | None = None
        self.datalink_lost_time_s: float | None = None
        self.datalink_lost_reason = ""
        self.update_count = 0
        self.datalink_update_count = 0
        self.last_detection: RadarDetection | None = None
        self.last_solution: TrackSolution | None = None
        self.observation_reject_reason = ""
        self.first_lock_loss_time_s: float | None = None
        self.first_reacquire_time_s: float | None = None
        self.first_decoy_track_time_s: float | None = None
        self.decoy_track_ticks = 0
        self.alpha_beta_observer: AlphaBetaStateObserver | None = None
        self.unified_target_observer = False
        self.angular_observer = None
        self.datalink_observation: DatalinkObservation | None = None
        self._last_datalink_sample_s: float | None = None
        self.observer_identity = None
        self.observer_events: list[dict[str, Any]] = []

    @property
    def track_direction_world(self):
        return None if self.angular_observer is None else self.angular_observer.direction

    @property
    def full_rate_world_rad_s(self):
        return None if self.angular_observer is None else self.angular_observer.rate

    @property
    def los_axial_rate_rad_s(self):
        n, w = self.track_direction_world, self.full_rate_world_rad_s
        return None if n is None or w is None else dot(n, w)

    @property
    def seeker_state(self) -> SeekerState:
        return self.radar.state

    @property
    def seeker_display_state(self) -> str:
        solution = self.last_solution
        if solution is not None and solution.mode in (TrackMode.RADAR_TRACK, TrackMode.RADAR_COAST):
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

    def enable_launcher_radar(self, position: Vector, velocity: Vector, gimbal_deg: float | None = None, *, geometry=None) -> None:
        """Tie the datalink to the launcher's radar track (user-reported game rule, D).

        The launcher flies straight on at its launch velocity. Its radar loses the
        target, and the datalink drops for good, once the target leaves the
        ``gimbal_deg`` yaw/pitch limits about the launcher's velocity, or sits in the ground
        clutter notch as seen from the launcher: the seeker's clutter rule from the
        launcher's position, and a ground radial speed within the seeker's notch
        half-width (the launcher radar's own width is not in the game files used).
        """
        if self.launcher_support is not None:
            raise ValueError('launcher_support and launcher_radar_gimbal_deg are mutually exclusive')
        geometry = validate_launcher_radar_geometry(geometry)
        if geometry is not None and gimbal_deg is not None:
            raise ValueError('launcher_radar_geometry and launcher_radar_gimbal_deg are mutually exclusive')
        if geometry is None:
            if gimbal_deg is None:
                raise ValueError('launcher radar needs mechanical limits')
            pitch = math.atan2(velocity[1], math.hypot(velocity[0], velocity[2]))
            axes = body_axes(pitch, math.atan2(velocity[2], velocity[0]))
            limit = math.radians(gimbal_deg)
            yaw_limits = pitch_limits = (-limit, limit)
            half = 0.  # Legacy scalar supplies neither aircraft beam nor body roll.
        else:
            x, y, z, w = geometry['orientation_quaternion_xyzw']
            axes = body_axes_from_quaternion((w, x, y, z))
            yaw_limits = tuple(math.radians(x) for x in geometry['yaw_limits_deg'])
            pitch_limits = tuple(math.radians(x) for x in geometry['pitch_limits_deg'])
            half = math.radians(geometry['beam_half_angle_deg'])
        self.launcher_radar = (tuple(position), tuple(velocity), axes, yaw_limits, pitch_limits, half)

    def enable_launcher_support(self, callback) -> None:
        """Use callback(t_since_launch, target_truth) for datalink support.

        Return an empty string for legacy truth support, or opt in by returning
        (reason, native_y_up_position, native_y_up_velocity). Nonempty reason
        drops the datalink permanently; absent estimates must supply a reason. Mutually exclusive with linear-launcher radar.
        """
        if not callable(callback):raise ValueError('launcher_support must be callable')
        if self.launcher_radar is not None:
            raise ValueError('launcher_support and launcher_radar_gimbal_deg are mutually exclusive')
        self.launcher_support = callback

    def enable_q_angular_observer(self) -> None:
        """Enable the empirical Q-at-capture observer before receiving samples."""
        if self.update_count:
            raise RuntimeError("Enable the angular observer before the first update")
        if not self.radar.active:
            raise ValueError("q_capture requires active radar sensor_track")
        if self.alpha_beta_observer is not None:
            raise ValueError("angular and Cartesian alpha-beta observers are mutually exclusive")
        from .q_angular_observer import QCaptureAngularObserver
        self.angular_observer = QCaptureAngularObserver()
        self.radar.position_only_measurements = True

    def enable_alpha_beta_observer(self, alpha: float, beta: float) -> None:
        """Filter accepted own-radar positions before guidance; off by default.

        At first acceptance, seed p/v once from the preceding available DL/INS
        solution. With no prior solution, the first radar track supplies the
        seed. DL outputs retain their original source and cadence; INS coasts
        the last selected (and, for radar, filtered) solution.
        """
        if self.update_count:
            raise RuntimeError("Enable the alpha-beta observer before the first update")
        if self.angular_observer is not None:
            raise ValueError("angular and Cartesian alpha-beta observers are mutually exclusive")
        if self.unified_target_observer:
            raise ValueError("radar-only and unified alpha-beta observers are mutually exclusive")
        self.alpha_beta_observer = AlphaBetaStateObserver(alpha, beta)

    def enable_unified_target_observer(self) -> None:
        """Opt in to one world-state observer for timestamped DL and radar.

        Fixed .8/.05 are experimental 3D coefficients, not identified WT
        semantics. Echoes measure ideal world position only. External estimated
        p/v may seed the first target prior; no such prior means zero velocity.
        Same-object reacquisition retains history; a reflector transfer resets
        position with zero velocity. Untimestamped legacy DL supplies a single
        prior, never assumed 48 Hz measurements. INS/search coast the observer.
        """
        if self.update_count:
            raise RuntimeError("Enable the unified target observer before the first update")
        if self.alpha_beta_observer is not None:
            raise ValueError("radar-only and unified alpha-beta observers are mutually exclusive")
        if self.angular_observer is not None:
            raise ValueError("angular and Cartesian alpha-beta observers are mutually exclusive")
        self.unified_target_observer = True
        self.alpha_beta_observer = AlphaBetaStateObserver(.8, .05)
        self.radar.position_only_measurements = True

    def _consume_unified_datalink(self, time_s: float) -> None:
        observer = self.alpha_beta_observer
        if not self.datalink_connected or self.datalink_estimate is None:
            return  # Connected support without an external estimate is not a measurement.
        packet = self.datalink_observation
        if packet is None:
            if observer.solution is not None:
                return  # Legacy tuple cadence is unknown; do not fabricate packets.
            position, velocity = self.datalink_estimate
            packet = DatalinkObservation(position, velocity, time_s, "unstamped_datalink_prior")
        stamp = packet.sample_time_s
        effective_stamp = _available_datalink_time(stamp, time_s)
        if self._last_datalink_sample_s is not None and stamp <= self._last_datalink_sample_s:
            return
        self._last_datalink_sample_s = stamp
        if self.observer_identity not in (None, ('target',)):
            return  # A target DL packet cannot update an associated chaff state.
        measurement = _copy_solution(packet.position, packet.velocity, effective_stamp,
                                     TrackMode.DATALINK, True, "unified_alpha_beta_datalink", stamp,
                                     identity=packet.identity or self.intended_target_identity)
        if observer.solution is None:
            observer.initialize(measurement)
            self.observer_identity = ('target',)
            self.observer_events.append(dict(time_s=time_s, sample_time_s=stamp,
                event="initialize_external_prior", source=packet.source))
        elif effective_stamp > observer.solution.solution_time_s:
            observer.update(measurement, source="unified_alpha_beta_datalink")
            self.observer_events.append(dict(time_s=time_s, sample_time_s=stamp, event="datalink_correction"))
        else:
            self.observer_events.append(dict(time_s=time_s, sample_time_s=stamp, event="stale_datalink_ignored"))

    def _unified_prediction(self, time_s: float, mode: TrackMode) -> TrackSolution | None:
        observer = self.alpha_beta_observer
        if observer.solution is None:
            return None
        solution = observer.solution
        if time_s - solution.sample_time_s > self.inertial.break_lock_max_time_s:
            return None
        position = observer.predict(time_s)
        velocity = solution.velocity
        # Preserve the opt-in altitude floor on propagated (not measured) states.
        if self.inertial.min_altitude_m is not None and time_s > solution.solution_time_s \
                and position[1] < self.inertial.min_altitude_m:
            position = (position[0], self.inertial.min_altitude_m, position[2])
            velocity = (velocity[0], 0., velocity[2])
        return _copy_solution(position, velocity, time_s, mode, True,
                              "unified_alpha_beta_prediction", solution.sample_time_s, template=solution)

    def _filter_unified_radar(self, measurement: TrackSolution) -> TrackSolution:
        observer = self.alpha_beta_observer
        identity = self.radar.measurement_identity
        if identity is None:
            raise ValueError("unified target observer requires a radar measurement identity")
        if observer.solution is None or identity != self.observer_identity:
            event = "initialize_zero_velocity" if observer.solution is None else "identity_reset_zero_velocity"
            self.alpha_beta_observer = observer = AlphaBetaStateObserver(.8, .05)
            seeded = _copy_solution(measurement.position, ZERO_VECTOR, measurement.solution_time_s,
                measurement.mode, True, "unified_alpha_beta_radar", measurement.sample_time_s, template=measurement)
            observer.initialize(seeded)
            self.observer_events.append(dict(time_s=measurement.solution_time_s, event=event, identity=identity))
        else:
            observer.update(measurement, source="unified_alpha_beta_radar", allow_same_time=True)
            self.observer_events.append(dict(time_s=measurement.solution_time_s,
                sample_time_s=measurement.sample_time_s, event="radar_correction", identity=identity))
        self.observer_identity = identity
        self.radar.set_estimated_track(observer.solution)
        return observer.solution

    def _filter_radar_solution(self, radar_solution: TrackSolution) -> TrackSolution:
        observer = self.alpha_beta_observer
        if observer is None:
            return radar_solution
        if observer.solution is None:
            previous = self.last_solution
            if previous is not None and previous.valid and previous.solution_time_s < radar_solution.solution_time_s:
                observer.initialize(previous)
            else:
                # Only initialization may copy a measured velocity. There is no
                # elapsed prediction interval on this first-ever observation.
                seeded = _copy_solution(radar_solution.position, radar_solution.velocity,
                    radar_solution.solution_time_s, radar_solution.mode, True, "radar_alpha_beta",
                    sample_time_s=radar_solution.sample_time_s, template=radar_solution)
                observer.initialize(seeded)
                return seeded
        return observer.update(radar_solution)

    def _launcher_loses_target(self, time_s: float, target_truth: TargetState) -> str:
        position, velocity, axes, yaw_limits, pitch_limits, half = self.launcher_radar
        launcher = add(position, scale(velocity, time_s))
        relative = sub(target_truth.position, launcher)
        range_m = norm(relative)
        if range_m <= 1.0e-9:
            return ""
        los = scale(relative, 1.0/range_m)
        center = limited_antenna_direction(axes, los, yaw_limits, pitch_limits)
        if not within_circular_beam(los, center, half):
            return "launcher_gimbal"
        look_down = target_truth.position[1] < launcher[1]
        if self.radar._on_clutter(look_down, launcher, range_m, relative) \
                and abs(dot(target_truth.velocity, los)) <= self.radar.notch_half_width_mps:
            return "launcher_clutter_notch"
        return ""

    def _refresh_datalink_state(self, time_s: float, target_truth: TargetState | None = None) -> None:
        if not self.datalink_enabled:
            self.datalink_connected = False
            return
        if self.datalink_disconnect_time_s is not None and time_s >= float(self.datalink_disconnect_time_s):
            self.datalink_connected = False
        keep_after_lock = self.radar.track_gate_model is not None and self.radar.track_gate_model['datalink_after_lock']
        if self.radar_has_tracked_once and not self.reconnect_datalink and not keep_after_lock:
            self.datalink_connected = False
        if self.launcher_radar is not None and self.datalink_connected and target_truth is not None:
            reason = self._launcher_loses_target(time_s, target_truth)
            if reason:
                self.datalink_connected = False
                self.datalink_lost_time_s, self.datalink_lost_reason = time_s, reason

        if self.launcher_support is not None and self.datalink_connected and target_truth is not None:
            reply = self.launcher_support(time_s, target_truth)
            self.datalink_estimate = None
            self.datalink_observation = None
            if isinstance(reply, str):
                reason = reply
            elif isinstance(reply, tuple) and len(reply) == 2 and isinstance(reply[0], str) \
                    and isinstance(reply[1], DatalinkObservation):
                reason, packet = reply
                if not reason:
                    effective_stamp = _available_datalink_time(packet.sample_time_s, time_s)
                    self.datalink_observation = packet
                    self.datalink_estimate = (add(packet.position, scale(packet.velocity, time_s-effective_stamp)), packet.velocity)
                    if self.strict_autonomous:
                        if self._datalink_prior is None or packet.sample_time_s > self._datalink_prior.sample_time_s:
                            self._datalink_prior = TrackSolution(packet.position, packet.velocity,
                                packet.sample_time_s, effective_stamp, TrackMode.DATALINK, True, packet.source,
                                packet.full_rate_rad_s, packet.identity or self.intended_target_identity, packet.source)
                            self._strict_datalink_is_new = True
                        else:
                            self.datalink_observation = None
                            self.datalink_estimate = None
            elif isinstance(reply, tuple) and len(reply) == 3 and isinstance(reply[0], str):
                reason, position, velocity = reply
                if not reason:
                    for value in (position, velocity):
                        if (not isinstance(value, (tuple, list)) or len(value) != 3 or
                                any(isinstance(x, bool) or not isinstance(x, (int, float)) or
                                    not math.isfinite(x) for x in value)):
                            raise ValueError('launcher_support estimate must contain finite xyz vectors')
                    self.datalink_estimate = (tuple(position), tuple(velocity))
                    if self.strict_autonomous:
                        self._datalink_prior = TrackSolution(tuple(position), tuple(velocity), time_s, time_s,
                            TrackMode.DATALINK, True, "unstamped_external_datalink", identity=self.intended_target_identity,
                            estimate_source="unstamped_external_datalink")
                        self._strict_datalink_is_new = True
            else:
                raise TypeError('launcher_support must return a string, (reason, position, velocity), or (reason, DatalinkObservation)')
            if reason:
                self.datalink_connected = False
                self.datalink_lost_time_s, self.datalink_lost_reason = time_s, reason

    def _datalink_solution(self, time_s: float, target_truth: TargetState) -> TrackSolution:
        if self.unified_target_observer:
            solution = self._unified_prediction(time_s, TrackMode.DATALINK)
            return solution if solution is not None else self._lost_solution(time_s)
        if self.strict_autonomous:
            # A gap, repeated packet or legacy support-only callback supplies no
            # new estimate. Coast owned history; never substitute target truth.
            prior = self._datalink_prior
            if self._strict_datalink_is_new and prior is not None:
                if time_s - prior.sample_time_s > self.inertial.break_lock_max_time_s:
                    return self._lost_solution(time_s)
                position = add(prior.position, scale(prior.velocity, time_s-prior.solution_time_s))
                return replace(prior, position=position, solution_time_s=time_s)
            if self.inertial.initialized:
                held = self.inertial.update(time_s, 0.)
                if time_s - held.sample_time_s > self.inertial.break_lock_max_time_s:
                    return replace(held, valid=False, mode=TrackMode.LOST)
                return held
            return self._lost_solution(time_s)
        position, velocity = self.datalink_estimate if self.datalink_estimate is not None else (target_truth.position, target_truth.velocity)
        stamp = self.datalink_observation.sample_time_s if self.datalink_observation is not None else time_s
        packet = self.datalink_observation
        return TrackSolution(position, velocity, stamp, time_s, TrackMode.DATALINK, True, "datalink",
            None if packet is None else packet.full_rate_rad_s,
            self.intended_target_identity if packet is None or packet.identity is None else packet.identity,
            ("legacy_truth_datalink" if self.datalink_estimate is None else
             (packet.source if packet is not None else "unstamped_external_datalink")))

    @staticmethod
    def _lost_solution(time_s: float, position: Vector = ZERO_VECTOR, velocity: Vector = ZERO_VECTOR) -> TrackSolution:
        return _copy_solution(position, velocity, time_s, TrackMode.LOST, False, "lost")

    def _record_stage(self, stage, estimate, reason=""):
        if not self.observer_diagnostics:
            return
        observer = self.angular_observer
        angular = observer is not None and observer.direction is not None
        stamp = (observer.sample_time_s if angular else
                 (self.measurement_cache.sample_time_s if self.measurement_cache is not None else None))
        snapshot = ObserverStageSnapshot(
            self.update_count, float(self._current_time_s), stage,
            self.intended_target_identity, self.selected_reflector_identity,
            (observer.identity if angular else (estimate.identity if estimate is not None else self.estimator_identity)),
            estimate, self.measurement_cache, self._previous_accepted_time_s, stamp,
            observer.time_s if angular else None,
            self._current_time_s if stage == 'publish_guidance' else None,
            self.antenna_axis_world, self.target_los_estimate_world,
            observer.direction if angular else None, observer.rate if angular else None,
            observer.range_m if angular else None, observer.closing_mps if angular else None,
            'instantaneous_sensor_relative_radial_speed' if angular else None, reason,
            observer.seed_source if observer is not None and stage == 'capture_handover' else None,
            observer.capture_rate_seed if observer is not None and stage == 'capture_handover' else None,
            measurement_range_m=(getattr(self._current_detection, 'range_m', None)
                if self._current_detection is not None and self._current_detection.valid else None),
            measurement_relative_radial_mps=(getattr(self._current_detection, 'relative_radial_speed_mps', None)
                if self._current_detection is not None and self._current_detection.valid else None),
            measurement_source=('synthetic_radar_echo_instantaneous' if self._current_detection is not None
                and self._current_detection.valid else None))
        self.stage_snapshots += (snapshot,)

    def update(self, time_s, dt_s, missile_state, target_truth, decoys=None, target_rcs_m2=1.0):
        if self.strict_autonomous and self.last_solution is not None and time_s < self.last_solution.solution_time_s-1e-12:
            raise ValueError('sensor provider time must not move backwards')
        self.stage_snapshots = ()
        # Previous acceptance remains in measurement_cache; current-call echo
        # fields become available only after this call evaluates the radar.
        self._current_detection = None
        self.antenna_axis_world = None
        self._current_time_s = float(time_s)
        self._previous_accepted_time_s = (self.measurement_cache.sample_time_s
            if self.measurement_cache is not None else None)
        self._strict_datalink_is_new = False
        solution = self._update_cartesian(time_s, dt_s, missile_state, target_truth, decoys, target_rcs_m2)
        received = bool(self.last_detection.valid and solution.mode is TrackMode.RADAR_TRACK)
        if received:
            identity = solution.identity or reflector_identity(self.radar.measurement_identity,
                self.intended_target_identity, self.radar.last_reflector)
            solution = replace(solution, identity=identity, estimate_source=solution.estimate_source or solution.source)
        if self.angular_observer is not None:
            self.angular_observer.stage_callback = self._record_stage
            solution = self.angular_observer.update(solution, missile_state, received=received,
                identity=self.radar.measurement_identity,
                closing_mps=-self.last_detection.relative_radial_speed_mps if received else None)
            if received:
                self.radar.set_estimated_track(solution)
                self.inertial.reset(solution)
        elif received:
            self._record_stage('accepted_measurement', solution)
        elif solution.mode is TrackMode.RADAR_COAST:
            self._record_stage('cartesian_coast', solution, 'legacy_nonangular_prediction')
        self.estimator_identity = solution.identity
        solution = replace(solution, publication_time_s=float(time_s))
        self.last_solution = solution
        if received:
            if self.alpha_beta_observer is not None or self.angular_observer is not None:
                self.radar.set_estimated_track(solution)
            else:
                self.radar.last_solution = solution
        self._record_stage('publish_guidance', solution)
        return solution

    def _update_cartesian(
        self,
        time_s: float,
        dt_s: float,
        missile_state: Any,
        target_truth: TargetState,
        decoys: list[Any] | None = None,
        target_rcs_m2: float = 1.0,
    ) -> TrackSolution:
        time_s = float(time_s)
        self.update_count += 1
        self._refresh_datalink_state(time_s, target_truth)
        if self.unified_target_observer:
            self._consume_unified_datalink(time_s)
            self.radar.estimated_track = self._unified_prediction(time_s, TrackMode.INS_SEARCH)
        search = self.radar.search
        search_inertial = None
        gate_enabled = self.radar.track_gate_model is not None
        estimate = None
        # Preserve legacy truth-only low-level support when no external prior
        # has ever been supplied. Once a causal DL/INS state exists, it must
        # drive search activation, including the ordinary radar path.
        estimated_input = self.strict_autonomous or self.datalink_estimate is not None or self.inertial.initialized \
            or self.launcher_support is not None \
            or (self.unified_target_observer and self.alpha_beta_observer.solution is not None)
        if search is not None or gate_enabled or (
            estimated_input and self.radar.active and not self.radar._within_track_hold(time_s)
        ):
            # Build the current available estimate before radar evaluation.
            # It is never replaced by this tick's unobserved target truth.
            if self.unified_target_observer:
                predicted = self._unified_prediction(time_s, TrackMode.INS_SEARCH)
                estimate = predicted.position if predicted is not None else None
                if gate_enabled and predicted is not None and self.radar.track_gate_prediction is not None:
                    self.radar.track_gate_prediction = (time_s, predicted.position, predicted.velocity)
            elif self.datalink_connected and self.datalink_estimate is not None:
                estimate = self.datalink_estimate[0]
            elif self.sensor_model.get('inertial_navigation') is True and self.inertial.enabled and self.inertial.initialized:
                search_inertial = self.inertial.update(time_s, dt_s)
                estimate = search_inertial.position if search_inertial.valid else None
            else:
                estimate = None
            if search is not None:search.estimate = estimate
            if gate_enabled:self.radar.track_gate_estimate = estimate
        self.radar.search_estimate_position = estimate
        self.radar.require_search_estimate = estimated_input
        self.ins_estimate = (self._datalink_prior if self.strict_autonomous and self._strict_datalink_is_new
                             else (search_inertial if search_inertial is not None else self.inertial.estimate))
        if self.datalink_estimate is not None:
            packet = self.datalink_observation
            self.ins_estimate = TrackSolution(*self.datalink_estimate,
                time_s if packet is None else packet.sample_time_s, time_s,
                TrackMode.DATALINK, True, "datalink", None if packet is None else packet.full_rate_rad_s,
                self.intended_target_identity if packet is None or packet.identity is None else packet.identity,
                packet.source if packet is not None else "unstamped_external_datalink")
        los_estimate = estimate
        if los_estimate is None and self.ins_estimate is not None and self.ins_estimate.valid:
            los_estimate = add(self.ins_estimate.position,
                scale(self.ins_estimate.velocity, time_s-self.ins_estimate.solution_time_s))
        relative = None if los_estimate is None else sub(los_estimate, missile_state.position)
        self.target_los_estimate_world = (normalize(relative) if relative is not None and norm(relative)>1e-12 else None)
        self._record_stage('ins_refresh', self.ins_estimate)
        if self.angular_observer is not None:
            self.angular_observer.stage_callback = self._record_stage
            if self.angular_observer.direction is None and self.ins_estimate is not None and self.ins_estimate.valid:
                self.angular_observer.refresh_prior(self.ins_estimate, missile_state)
        if search is not None or gate_enabled:
            detection, radar_solution = self.radar.update(time_s, dt_s, missile_state, target_truth,
                                                          decoys=decoys, target_rcs_m2=target_rcs_m2)
            if decoys is not None and detection.valid and radar_solution is not None and self.radar.last_reflector != 'target':
                self.decoy_track_ticks += 1
                if self.first_decoy_track_time_s is None:
                    self.first_decoy_track_time_s = time_s
        elif decoys is None:
            detection, radar_solution = self.radar.update(time_s, dt_s, missile_state, target_truth,
                                                          target_rcs_m2=target_rcs_m2)
        else:
            detection, radar_solution = self.radar.update(time_s, dt_s, missile_state, target_truth,
                                                          decoys=decoys, target_rcs_m2=target_rcs_m2)
            if detection.valid and radar_solution is not None and self.radar.last_reflector != "target":
                self.decoy_track_ticks += 1
                if self.first_decoy_track_time_s is None:
                    self.first_decoy_track_time_s = time_s
        self.last_detection = detection
        self._current_detection = detection
        self.antenna_axis_world = self.radar.last_antenna_axis_world
        previous_reflector = self.selected_reflector_identity
        if detection.valid and radar_solution is not None:
            identity = radar_solution.identity or reflector_identity(self.radar.measurement_identity,
                self.intended_target_identity, self.radar.last_reflector)
            radar_solution = replace(radar_solution, identity=identity, estimate_source=radar_solution.estimate_source or radar_solution.source)
            self.selected_reflector_identity = identity
            self.measurement_cache = radar_solution
        self._record_stage('radar_association', radar_solution, detection.reason)
        if detection.valid and radar_solution is not None and self.angular_observer is None and (
            not self.radar_has_tracked_once or previous_reflector != self.selected_reflector_identity or
            (self.last_solution is not None and self.last_solution.mode not in (TrackMode.RADAR_TRACK, TrackMode.RADAR_COAST))):
            self._record_stage('capture_handover', radar_solution,
                'reflector_changed' if previous_reflector is not None and previous_reflector != self.selected_reflector_identity
                else 'legacy_radar_observation_handover')
        self.observation_reject_reason = "" if detection.valid else detection.reason
        if detection.valid and radar_solution is not None and radar_solution.mode is TrackMode.RADAR_TRACK:
            radar_solution = (self._filter_unified_radar(radar_solution) if self.unified_target_observer
                              else self._filter_radar_solution(radar_solution))
            if self.alpha_beta_observer is not None and not self.unified_target_observer:
                self.radar.set_estimated_track(radar_solution)
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

        if radar_solution is not None and radar_solution.valid and radar_solution.mode is TrackMode.RADAR_COAST:
            # A missed echo is not a new observation. Radar memory is anchored
            # at the last accepted (possibly filtered) state and timestamp.
            self.last_solution = radar_solution
            return self.last_solution

        self._refresh_datalink_state(time_s)
        if self.unified_target_observer:
            mode = TrackMode.DATALINK if self.datalink_connected and self.observer_identity == ('target',) \
                else TrackMode.INS_SEARCH
            prediction = self._unified_prediction(time_s, mode)
            if prediction is not None and (self.datalink_connected or self.inertial.enabled):
                if self.radar_has_tracked_once and self.first_lock_loss_time_s is None:
                    self.first_lock_loss_time_s = time_s
                self.last_solution = prediction
            else:
                self.last_solution = _copy_solution(ZERO_VECTOR, ZERO_VECTOR, time_s,
                    TrackMode.RADAR_SEARCH if self.radar.active else TrackMode.LOST, False, "unified_observer_uninitialized")
            return self.last_solution
        if self.datalink_connected:
            self.datalink_update_count += 1
            datalink_solution = self._datalink_solution(time_s, target_truth)
            if not self.strict_autonomous or self._strict_datalink_is_new:
                self.inertial.reset(datalink_solution)
            self.last_solution = datalink_solution
            return datalink_solution

        if self.inertial.enabled and self.inertial.initialized:
            inertial_solution = search_inertial if search_inertial is not None else self.inertial.update(time_s, dt_s)
            if inertial_solution.valid and (self.radar.active or search is not None):
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
                    template=inertial_solution,
                )
            self.last_solution = inertial_solution
            return inertial_solution

        if self.radar.active or search is not None:
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
