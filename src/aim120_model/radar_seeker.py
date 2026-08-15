"""Deterministic active-radar observation candidate for PLAN8."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .aerodynamics import body_axes
from .dynamics import SimState
from .math3d import Vector, add, clamp, cross, dot, norm, normalize, scale, sub
from .target import TargetState
from .tracking import AlphaBetaGate, TrackMode, TrackSolution
from .units import deg_to_rad, rad_to_deg


class SeekerState(str, Enum):
    SEARCH = "search"
    TRACK = "track"
    COAST = "coast"
    LOST = "lost"


@dataclass(frozen=True)
class RadarDetection:
    valid: bool
    reason: str
    range_m: float
    relative_radial_speed_mps: float
    ground_radial_speed_mps: float
    off_boresight_deg: float
    los_rate_deg_s: float
    look_down: bool
    notch_half_width_mps: float = 0.0


def _number(mapping: dict[str, Any], *keys: str, default: float) -> float:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, bool) or value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return number
    return float(default)


def _boolean(mapping: dict[str, Any], *keys: str, default: bool) -> bool:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, bool):
            return value
    return bool(default)


class RadarSeekerObserver:
    """Generate a copied radar track from target truth and deterministic gates."""

    def __init__(self, sensor_model: dict[str, Any] | None = None, lock_range_m: float | None = None):
        sensor_model = sensor_model if isinstance(sensor_model, dict) else {}
        radar = sensor_model.get("radar_seeker")
        self.radar_config = radar if isinstance(radar, dict) else {}
        self.active = _boolean(self.radar_config, "active", default=_boolean(sensor_model, "active_radar", default=False))
        self.use_target_velocity = _boolean(sensor_model, "use_target_velocity", "useTargetVel", default=True)
        lock_range_default = math.inf if lock_range_m is None else float(lock_range_m)
        self.lock_range_m = _number(sensor_model, "lock_range_m", "lockDistance", default=lock_range_default)
        self.angle_gate_rate_rad_s = deg_to_rad(_number(self.radar_config, "angle_gate_rate_deg_s", "angleGateRate", default=math.inf))
        self.prolongation_time_max_s = _number(
            self.radar_config,
            "prolongation_time_max_s",
            "prolongationTimeMax",
            default=1.0,
        )
        self.lock_angle_max_rad = deg_to_rad(_number(self.radar_config, "lock_angle_max_deg", "lockAngleMax", default=180.0))
        self.angle_max_rad = deg_to_rad(_number(self.radar_config, "angle_max_deg", "angleMax", default=180.0))
        self.rate_max_rad_s = deg_to_rad(_number(self.radar_config, "rate_max_deg_s", "rateMax", default=math.inf))

        receiver = self.radar_config.get("receiver")
        receiver = receiver if isinstance(receiver, dict) else {}
        self.receiver_range_m = _number(receiver, "range_m", "range", default=math.inf)
        self.receiver_range_max_m = _number(
            receiver,
            "range_max_m",
            "rangeMax",
            default=self.receiver_range_m if math.isfinite(self.receiver_range_m) else self.lock_range_m,
        )

        doppler_gate_cfg = self.radar_config.get("doppler_speed_gate")
        doppler_gate_cfg = doppler_gate_cfg if isinstance(doppler_gate_cfg, dict) else {}
        distance_gate_cfg = self.radar_config.get("dist_gate")
        distance_gate_cfg = distance_gate_cfg if isinstance(distance_gate_cfg, dict) else {}
        self.distance_gate = AlphaBetaGate(
            alpha=_number(distance_gate_cfg, "filter_alpha", "filterAlpha", default=0.8),
            beta=_number(distance_gate_cfg, "filter_beta", "filterBetta", default=0.05),
            search_range=_number(distance_gate_cfg, "search_range_m", "dist_gate_search_range", "distGateSearchRange", default=5000.0),
        )
        self.doppler_gate = AlphaBetaGate(
            alpha=_number(doppler_gate_cfg, "filter_alpha", "filterAlpha", default=0.8),
            beta=_number(doppler_gate_cfg, "filter_beta", "filterBetta", default=0.05),
            search_range=_number(doppler_gate_cfg, "search_range_mps", "doppler_speed_gate_search_range", "dopplerSpeedGateSearchRange", default=300.0),
        )

        doppler_cfg = self.radar_config.get("doppler_speed")
        doppler_cfg = doppler_cfg if isinstance(doppler_cfg, dict) else {}
        self.doppler_min_mps = _number(doppler_cfg, "min_mps", "minValue", default=-math.inf)
        self.doppler_max_mps = _number(doppler_cfg, "max_mps", "maxValue", default=math.inf)
        target_width = max(
            abs(_number(doppler_cfg, "width_mps", "width", default=0.0)),
            abs(_number(doppler_cfg, "signal_width_min_mps", "signalWidthMin", default=0.0)),
        )
        clutter_width = abs(_number(doppler_cfg, "ref_width_mps", "refWidth", default=0.0))
        self.notch_half_width_mps = target_width / 2.0 + clutter_width / 2.0

        self._state = SeekerState.SEARCH
        self._last_detection_time_s: float | None = None
        self._last_accepted_los: Vector | None = None
        self._last_measurement_position: Vector | None = None
        self._last_measurement_time_s: float | None = None
        self._last_position: Vector = (0.0, 0.0, 0.0)
        self._last_velocity: Vector = (0.0, 0.0, 0.0)
        self._search_gates_reset = False
        self.last_detection: RadarDetection | None = None
        self.last_solution: TrackSolution | None = None

    @property
    def state(self) -> SeekerState:
        return self._state

    @property
    def seeker_state(self) -> SeekerState:
        return self._state

    def reset(self) -> None:
        self._state = SeekerState.SEARCH
        self._last_detection_time_s = None
        self._last_accepted_los = None
        self._last_measurement_position = None
        self._last_measurement_time_s = None
        self._last_position = (0.0, 0.0, 0.0)
        self._last_velocity = (0.0, 0.0, 0.0)
        self._search_gates_reset = False
        self.distance_gate = AlphaBetaGate(self.distance_gate.alpha, self.distance_gate.beta, self.distance_gate.search_range)
        self.doppler_gate = AlphaBetaGate(self.doppler_gate.alpha, self.doppler_gate.beta, self.doppler_gate.search_range)
        self.last_detection = None
        self.last_solution = None

    def _effective_dt(self, time_s: float, dt_s: float) -> float:
        if self._last_detection_time_s is not None and time_s > self._last_detection_time_s:
            return time_s - self._last_detection_time_s
        if math.isfinite(float(dt_s)) and float(dt_s) > 0.0:
            return float(dt_s)
        return 1.0 / 256.0

    def _detection(
        self,
        valid: bool,
        reason: str,
        range_m: float,
        range_rate_mps: float,
        ground_radial_mps: float,
        off_boresight_deg: float,
        los_rate_deg_s: float,
        look_down: bool,
    ) -> RadarDetection:
        return RadarDetection(
            valid=bool(valid),
            reason=str(reason),
            range_m=float(range_m),
            relative_radial_speed_mps=float(range_rate_mps),
            ground_radial_speed_mps=float(ground_radial_mps),
            off_boresight_deg=float(off_boresight_deg),
            los_rate_deg_s=float(los_rate_deg_s),
            look_down=bool(look_down),
            notch_half_width_mps=float(self.notch_half_width_mps),
        )

    def _coast_solution(self, time_s: float) -> TrackSolution | None:
        if self._last_detection_time_s is None:
            return None
        age_s = max(0.0, float(time_s) - self._last_detection_time_s)
        if age_s > self.prolongation_time_max_s:
            self._state = SeekerState.LOST
            return None
        position = add(self._last_position, scale(self._last_velocity, age_s))
        return TrackSolution(
            position=position,
            velocity=self._last_velocity,
            sample_time_s=self._last_detection_time_s,
            solution_time_s=float(time_s),
            mode=TrackMode.RADAR_COAST,
            valid=True,
            source="radar_coast",
        )

    def _invalid_result(
        self,
        detection: RadarDetection,
        time_s: float,
    ) -> tuple[RadarDetection, TrackSolution | None]:
        del time_s
        # The composite provider owns INS propagation.  The radar observer
        # reports the failed measurement and returns control to SEARCH so the
        # next tick can use the appropriate search/hold gates.
        self._state = SeekerState.SEARCH
        self.last_solution = None
        return detection, None

    def update(
        self,
        time_s: float,
        dt_s: float,
        missile_state: SimState,
        target_truth: TargetState,
    ) -> tuple[RadarDetection, TrackSolution | None]:
        time_s = float(time_s)
        if not self.active:
            detection = self._detection(False, "inactive", 0.0, 0.0, 0.0, 0.0, 0.0, False)
            self._state = SeekerState.LOST
            self.last_detection = detection
            self.last_solution = None
            return detection, None

        relative = sub(target_truth.position, missile_state.position)
        range_sq = dot(relative, relative)
        range_m = math.sqrt(max(0.0, range_sq))
        if not math.isfinite(range_m) or range_m <= 1.0e-12:
            detection = self._detection(False, "invalid_geometry", range_m, 0.0, 0.0, 180.0, math.inf, False)
            self.last_detection = detection
            return self._invalid_result(detection, time_s)

        los_hat = normalize(relative)
        relative_velocity = sub(target_truth.velocity, missile_state.velocity)
        range_rate_mps = dot(relative_velocity, los_hat)
        ground_radial_mps = dot(target_truth.velocity, los_hat)
        look_down = target_truth.position[1] < missile_state.position[1]
        forward = body_axes(missile_state.pitch, missile_state.yaw).forward
        off_boresight_rad = math.acos(clamp(dot(forward, los_hat), -1.0, 1.0))
        los_rate_rad_s = norm(cross(relative, relative_velocity)) / max(range_sq, 1.0e-12)
        off_boresight_deg = rad_to_deg(off_boresight_rad)
        los_rate_deg_s = rad_to_deg(los_rate_rad_s)
        detection_kwargs = {
            "range_m": range_m,
            "range_rate_mps": range_rate_mps,
            "ground_radial_mps": ground_radial_mps,
            "off_boresight_deg": off_boresight_deg,
            "los_rate_deg_s": los_rate_deg_s,
            "look_down": look_down,
        }
        if not all(math.isfinite(value) for value in (range_rate_mps, ground_radial_mps, off_boresight_deg, los_rate_deg_s)):
            detection = self._detection(False, "invalid_geometry", **detection_kwargs)
            self.last_detection = detection
            return self._invalid_result(detection, time_s)

        track_age_s = math.inf
        if self._last_detection_time_s is not None:
            track_age_s = max(0.0, time_s - self._last_detection_time_s)
        track_hold = (
            self._last_detection_time_s is not None
            and track_age_s <= self.prolongation_time_max_s
        )
        if self._last_detection_time_s is not None and not track_hold and not self._search_gates_reset:
            # Once the old track window expires, restart both scalar gates for
            # a fresh search.  The composite provider may still present INS to
            # guidance while this observer searches.
            self.distance_gate = AlphaBetaGate(self.distance_gate.alpha, self.distance_gate.beta, self.distance_gate.search_range)
            self.doppler_gate = AlphaBetaGate(self.doppler_gate.alpha, self.doppler_gate.beta, self.doppler_gate.search_range)
            self._last_accepted_los = None
            self._last_measurement_position = None
            self._last_measurement_time_s = None
            self._search_gates_reset = True

        # Before the first detection and after the prolongation window use
        # search acquisition gates.  During the window retain track gates to
        # allow deterministic re-acquisition around the previous track.
        searching = not track_hold
        range_limit_m = min(self.lock_range_m, self.receiver_range_max_m) if searching else self.receiver_range_max_m
        angle_limit_rad = self.lock_angle_max_rad if searching else self.angle_max_rad
        if range_m > range_limit_m:
            detection = self._detection(False, "range_gate", **detection_kwargs)
            self.last_detection = detection
            return self._invalid_result(detection, time_s)
        if off_boresight_rad > angle_limit_rad:
            detection = self._detection(False, "angle_gate", **detection_kwargs)
            self.last_detection = detection
            return self._invalid_result(detection, time_s)
        if los_rate_rad_s > self.rate_max_rad_s:
            detection = self._detection(False, "rate_gate", **detection_kwargs)
            self.last_detection = detection
            return self._invalid_result(detection, time_s)

        effective_dt = self._effective_dt(time_s, dt_s)
        if track_hold and self._last_accepted_los is not None and math.isfinite(self.angle_gate_rate_rad_s):
            los_change = math.acos(clamp(dot(self._last_accepted_los, los_hat), -1.0, 1.0)) / effective_dt
            if los_change > self.angle_gate_rate_rad_s:
                detection = self._detection(False, "angle_gate_rate", **detection_kwargs)
                self.last_detection = detection
                return self._invalid_result(detection, time_s)

        if range_rate_mps < self.doppler_min_mps or range_rate_mps > self.doppler_max_mps:
            detection = self._detection(False, "doppler_out_of_bounds", **detection_kwargs)
            self.last_detection = detection
            return self._invalid_result(detection, time_s)

        if look_down and abs(ground_radial_mps) <= self.notch_half_width_mps:
            detection = self._detection(False, "ground_clutter_notch", **detection_kwargs)
            self.last_detection = detection
            return self._invalid_result(detection, time_s)

        if not self.distance_gate.accepts(range_m, effective_dt):
            detection = self._detection(False, "distance_gate", **detection_kwargs)
            self.last_detection = detection
            return self._invalid_result(detection, time_s)
        if not self.doppler_gate.accepts(range_rate_mps, effective_dt):
            detection = self._detection(False, "doppler_gate", **detection_kwargs)
            self.last_detection = detection
            return self._invalid_result(detection, time_s)

        self.distance_gate.update(range_m, effective_dt)
        self.doppler_gate.update(range_rate_mps, effective_dt)
        if self.use_target_velocity:
            velocity = tuple(float(value) for value in target_truth.velocity)
        elif self._last_measurement_position is None or self._last_measurement_time_s is None or time_s <= self._last_measurement_time_s:
            velocity = (0.0, 0.0, 0.0)
        else:
            velocity = scale(
                sub(target_truth.position, self._last_measurement_position),
                1.0 / (time_s - self._last_measurement_time_s),
            )
        position = tuple(float(value) for value in target_truth.position)
        solution = TrackSolution(
            position=position,
            velocity=velocity,
            sample_time_s=time_s,
            solution_time_s=time_s,
            mode=TrackMode.RADAR_TRACK,
            valid=True,
            source="radar",
        )
        detection = self._detection(True, "", **detection_kwargs)
        self._state = SeekerState.TRACK
        self._last_detection_time_s = time_s
        self._last_accepted_los = tuple(float(value) for value in los_hat)
        self._last_measurement_position = position
        self._last_measurement_time_s = time_s
        self._last_position = position
        self._last_velocity = velocity
        self._search_gates_reset = False
        self.last_detection = detection
        self.last_solution = solution
        return detection, solution


__all__ = ["RadarDetection", "RadarSeekerObserver", "SeekerState"]
