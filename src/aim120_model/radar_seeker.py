"""Deterministic active-radar observation candidate for PLAN8."""

from __future__ import annotations

import math
import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .aerodynamics import body_axes_for_state
from .dynamics import SimState
from .math3d import Vector, add, clamp, cross, dot, norm, normalize, scale, sub
from .target import TargetState
from .tracking import AlphaBetaGate, TrackMode, TrackSolution
from .units import deg_to_rad, rad_to_deg
from .radar_geometry import full_cone_half_angle_rad, limited_antenna_direction


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


def _multipath_table(value: Any) -> tuple[tuple[float, float], ...]:
    """``multipathEffect`` flat [x0, y0, x1, y1, ...] -> nodes with strictly increasing x."""
    if not isinstance(value, (list, tuple)) or len(value) < 2 or len(value) % 2:
        raise ValueError("profile has no usable radar_seeker.multipath_effect table")
    numbers = [float(v) for v in value if not isinstance(v, bool)]
    if len(numbers) != len(value) or not all(math.isfinite(v) for v in numbers):
        raise ValueError("radar_seeker.multipath_effect must hold finite numbers")
    nodes = tuple(zip(numbers[::2], numbers[1::2]))
    if any(b[0] <= a[0] for a, b in zip(nodes, nodes[1:])):
        raise ValueError("radar_seeker.multipath_effect heights must increase")
    return nodes


def _interpolate(nodes: tuple[tuple[float, float], ...], x: float) -> float:
    """Piecewise linear, held at the end nodes."""
    if x <= nodes[0][0]:
        return nodes[0][1]
    for (x0, y0), (x1, y1) in zip(nodes, nodes[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return nodes[-1][1]


def validate_seeker_search(options: Any) -> dict[str, Any] | None:
    """Validate opt-in model assumptions without changing the legacy path."""
    if options is None:
        return None
    if not isinstance(options, dict):
        raise ValueError('seeker_search must be an object or None')
    if not all(isinstance(key, str) for key in options):
        raise ValueError('seeker_search keys must be strings')
    unknown = set(options) - {'scan_half_angle_deg', 'frame_time_s', 'pattern', 'dwell_s'}
    if unknown:
        raise ValueError('unknown seeker_search options: ' + ', '.join(sorted(unknown)))
    result = dict(options)
    for key, default, allow_zero in [('scan_half_angle_deg', 20., True), ('frame_time_s', 1.5, False),
                                    ('dwell_s', None, True)]:
        value = result.get(key, default)
        if value is None and key == 'dwell_s' and key not in result:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
                or value < 0 or (not allow_zero and value == 0):
            raise ValueError(key + ' must be finite and ' + ('nonnegative' if allow_zero else 'positive'))
        if key == 'scan_half_angle_deg' and value > 180:
            raise ValueError('scan_half_angle_deg must be at most 180')
        result[key] = float(value)
    result.setdefault('pattern', 'spiral')
    if result['pattern'] not in ('spiral', 'raster'):
        raise ValueError('pattern must be spiral or raster')
    return result


def validate_chaff_capture(options: Any) -> dict[str, Any] | None:
    """C-grade game replay curve; exponential form/delay are D assumptions."""
    if options is None:
        return None
    if not isinstance(options, dict) or any(not isinstance(k, str) for k in options):
        raise ValueError('chaff_capture must be an object or None')
    defaults = dict(model='vg_soft', p_max=.5, v0_mps=25., scale_mps=58., delay_s=1.3, seed=0)
    if set(options) - set(defaults):
        raise ValueError('unknown chaff_capture options')
    result = {**defaults, **options}
    if result['model'] != 'vg_soft':
        raise ValueError('chaff_capture model must be vg_soft')
    for key in ('p_max', 'v0_mps', 'scale_mps', 'delay_s'):
        v = result[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            raise ValueError(key + ' must be finite and nonnegative')
        result[key] = float(v)
    if result['p_max'] > 1 or result['scale_mps'] <= 0:
        raise ValueError('p_max must be <= 1 and scale_mps must be positive')
    if isinstance(result['seed'], bool) or not isinstance(result['seed'], int):
        raise ValueError('chaff_capture seed must be an integer')
    return result


def validate_track_gate_model(options: Any) -> dict[str, Any] | None:
    """Opt-in WT tracking assumptions; widths and return competition are D."""
    if options is None:
        return None
    if not isinstance(options, dict) or any(not isinstance(k, str) for k in options):
        raise ValueError('track_gate_model must be an object or None')
    defaults = dict(range_gate_m=150., speed_gate_mps=10., angle_gate_deg=5.,
                    reacquire_half_deg=5., acquire_half_deg=5., datalink_after_lock=True, compete='stronger', seed=0,
                    fluct_sigma=.25, notch_track_hazard=[[25., .34], [50., .39], [75., .09]])
    if set(options) - set(defaults):
        raise ValueError('unknown track_gate_model options')
    result = {**defaults, **options}
    for key in ('range_gate_m', 'speed_gate_mps', 'angle_gate_deg', 'reacquire_half_deg', 'fluct_sigma'):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(key + ' must be finite and nonnegative')
        result[key] = float(value)
    value = result['acquire_half_deg']
    if value is not None:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 180:
            raise ValueError('acquire_half_deg must be None or a finite angle from 0 to 180')
        result['acquire_half_deg'] = float(value)
    if not isinstance(result['datalink_after_lock'], bool):
        raise ValueError('datalink_after_lock must be boolean')
    if result['angle_gate_deg'] > 180 or result['reacquire_half_deg'] > 180:
        raise ValueError('track_gate_model angles must be at most 180')
    if result['compete'] not in ('stronger', 'stronger_fluct'):
        raise ValueError('compete must be stronger or stronger_fluct')
    if isinstance(result['seed'], bool) or not isinstance(result['seed'], int):
        raise ValueError('track_gate_model seed must be an integer')
    table = result['notch_track_hazard']
    if not isinstance(table, (list, tuple)):
        raise ValueError('notch_track_hazard must be [upper speed, hazard per second] rows')
    checked = []
    for row in table:
        if not isinstance(row, (list, tuple)) or len(row) != 2 or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in row):
            raise ValueError('invalid notch_track_hazard row')
        upper, hazard = map(float, row)
        if upper <= (checked[-1][0] if checked else 0.):
            raise ValueError('notch_track_hazard upper speeds must increase')
        checked.append([upper, hazard])
    result['notch_track_hazard'] = checked
    return result


class SeekerSearch:
    """D-grade scan, antenna and acquisition model; all inputs are per missile."""

    def __init__(self, observer, options):
        self.observer = observer
        self.options = validate_seeker_search(options)
        self.half_rad = deg_to_rad(min(self.options['scan_half_angle_deg'],
                                      rad_to_deg(observer.lock_angle_max_rad)))
        self.frame_s, self.pattern = self.options['frame_time_s'], self.options['pattern']
        self.dwell_s = self.options.get('dwell_s', 0.)
        if isinstance(self.dwell_s, bool) or not isinstance(self.dwell_s, (float, int)) \
                or not math.isfinite(self.dwell_s) or self.dwell_s < 0:
            raise ValueError('seeker_search dwell_s must be finite and nonnegative')
        self.rx = self._antenna('receiver')
        self.tx = self._antenna('transmitter')
        receiver = observer.radar_config.get('receiver') or {}
        self.range_m = self._positive(receiver.get('range_m'), 'receiver.range_m')
        self.rcs_m2 = self._positive(receiver.get('rcs_m2'), 'receiver.rcs_m2')
        self.search_start_s = None
        self.searching = False
        self.candidate = None
        self.candidate_start_s = None
        self.first_lock_time_s = None
        self.lock_loss_count = 0
        self.reacquire_times_s = []
        self.reacquire_delays_s = []
        self.loss_time_s = None
        self.beam = (1., 0., 0.)
        self.scan_offset_deg = None
        self.beam_target_deg = None
        self.estimate = None

    @staticmethod
    def _positive(value, label):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(label + ' is required and must be positive and finite')
        return float(value)

    def _antenna(self, kind):
        antenna = (self.observer.radar_config.get(kind) or {}).get('antenna') or {}
        width = self._positive(antenna.get('angle_half_sens_deg'), kind + '.antenna.angle_half_sens_deg')
        side = antenna.get('side_lobes_sensitivity_db')
        if isinstance(side, bool) or not isinstance(side, (int, float)) or not math.isfinite(side) or side > 0:
            raise ValueError(kind + '.antenna.side_lobes_sensitivity_db is required and must be finite and <= 0')
        return full_cone_half_angle_rad(width), 10. ** (side / 10.)

    @staticmethod
    def gain(theta, antenna):
        half, floor = antenna
        return max(math.exp(-math.log(2.) * (theta / half) ** 2), floor)

    @staticmethod
    def cone(direction, forward, limit):
        angle = math.acos(clamp(dot(direction, forward), -1., 1.))
        if angle <= limit:
            return direction
        tangent = sub(direction, scale(forward, dot(direction, forward)))
        if norm(tangent) <= 1e-12:
            axis = (0., 1., 0.) if abs(forward[1]) < .9 else (0., 0., 1.)
            tangent = cross(forward, axis)
        return add(scale(forward, math.cos(limit)), scale(normalize(tangent), math.sin(limit)))

    def prepare(self, time_s, missile_state, target_truth, searching):
        if searching and not self.searching:
            self.search_start_s = time_s
            self.candidate = self.candidate_start_s = None
            if self.first_lock_time_s is not None:
                self.lock_loss_count += 1
                self.loss_time_s = time_s
        self.searching = searching
        forward = body_axes_for_state(missile_state).forward
        estimate = self.estimate
        if self.observer.position_only_measurements and self.observer.estimated_track is not None:
            track = self.observer.estimated_track
            estimate = add(track.position, scale(track.velocity, time_s-track.solution_time_s))
        elif not searching and self.observer._last_position is not None:
            age = max(0., time_s - self.observer._last_detection_time_s)
            estimate = add(self.observer._last_position, scale(self.observer._last_velocity, age))
        relative = None if estimate is None else sub(estimate, missile_state.position)
        center = forward if relative is None or norm(relative) <= 1e-12 else normalize(relative)
        center = self.observer.pointable_direction(missile_state, center, searching)
        beam = center
        if searching:
            phase = ((time_s - self.search_start_s) / self.frame_s) % 1.
            if self.pattern == 'spiral':
                radius = self.half_rad * math.sqrt(phase)
                azimuth = 4. * math.pi * phase
                x, y = radius * math.cos(azimuth), radius * math.sin(azimuth)
            else:
                points = [(0., 0.), (-1., -1.), (1., -1.), (1., -.5), (-1., -.5),
                          (-1., 0.), (1., 0.), (1., .5), (-1., .5), (-1., 1.), (1., 1.), (0., 0.)]
                coordinate = phase * (len(points) - 1)
                index = min(int(coordinate), len(points) - 2)
                fraction = coordinate - index
                x, y = (self.half_rad * (a + fraction * (b - a)) for a, b in zip(points[index], points[index+1]))
            radius = math.hypot(x, y)
            if radius > 0:
                axis = (0., 1., 0.) if abs(center[1]) < .9 else (0., 0., 1.)
                right = normalize(cross(center, axis)); up = cross(right, center)
                offset = add(scale(right, x / radius), scale(up, y / radius))
                radius = min(radius, self.half_rad)
                beam = add(scale(center, math.cos(radius)), scale(offset, math.sin(radius)))
            beam = self.observer.pointable_direction(missile_state, beam, True)
        self.beam = normalize(beam)
        self.scan_offset_deg = rad_to_deg(math.acos(clamp(dot(self.beam, center), -1., 1.)))
        target_los = sub(target_truth.position, missile_state.position)
        self.beam_target_deg = None if norm(target_los) <= 1e-12 else rad_to_deg(math.acos(clamp(dot(self.beam, normalize(target_los)), -1., 1.)))
    def return_gain(self, los):
        theta = math.acos(clamp(dot(self.beam, los), -1., 1.))
        receiver = self.gain(theta, self.rx)
        return receiver * self.gain(theta, self.tx)

    def detection_range(self, los, rcs_m2):
        gain = self.return_gain(los)
        if isinstance(rcs_m2, bool) or not isinstance(rcs_m2, (int, float)) or not math.isfinite(rcs_m2) or rcs_m2 < 0:
            raise ValueError('target/reflector rcs_m2 must be finite and nonnegative')
        return self.range_m * (rcs_m2 / self.rcs_m2 * gain) ** .25

    def acquisition(self, time_s, identity):
        if not self.searching:
            return True
        if identity is None or self.candidate != identity:
            self.candidate, self.candidate_start_s = identity, time_s
        if time_s - self.candidate_start_s < self.dwell_s - 1e-12:
            return False
        if self.first_lock_time_s is None:
            self.first_lock_time_s = time_s
        elif self.loss_time_s is not None:
            self.reacquire_times_s.append(time_s)
            self.reacquire_delays_s.append(time_s - self.loss_time_s)
            self.loss_time_s = None
        return True

    def miss(self):
        if self.searching:
            self.candidate = self.candidate_start_s = None

    def summary(self):
        return dict(first_lock_time_s=self.first_lock_time_s, lock_loss_count=self.lock_loss_count,
                    reacquire_times_s=list(self.reacquire_times_s), reacquire_delays_s=list(self.reacquire_delays_s))


class RadarSeekerObserver:
    """Generate a copied radar track from target truth and deterministic gates."""

    def __init__(self, sensor_model: dict[str, Any] | None = None, lock_range_m: float | None = None):
        sensor_model = sensor_model if isinstance(sensor_model, dict) else {}
        self.sensor_model = sensor_model
        self.search = None
        self.chaff_capture = None
        self.track_gate_model = None
        self.track_gate_estimate = None
        # Current causal DL/INS position, supplied by the composite provider.
        # A direct low-level caller without an estimate retains its legacy gate.
        self.search_estimate_position: Vector | None = None
        self.require_search_estimate = False
        radar = sensor_model.get("radar_seeker")
        self.radar_config = radar if isinstance(radar, dict) else {}
        self.active = _boolean(self.radar_config, "active", default=_boolean(sensor_model, "active_radar", default=False))
        # time_s throughout the observation API is age since missile launch.
        self.startup_delay_s = max(0.0, _number(sensor_model, "lock_timeout_s", default=0.0))
        self.use_target_velocity = _boolean(sensor_model, "use_target_velocity", "useTargetVel", default=True)
        self.position_only_measurements = False
        self.measurement_identity = None
        self.estimated_track = None
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
        # Resolution cell for the opt-in decoy path only (D: reflectors closer
        # than one width in range and Doppler, inside the half-power beam, are
        # unresolved and the stronger RCS/R^4 return captures the track).
        distance_cfg = self.radar_config.get("distance")
        distance_cfg = distance_cfg if isinstance(distance_cfg, dict) else {}
        receiver_antenna = receiver.get("antenna")
        receiver_antenna = receiver_antenna if isinstance(receiver_antenna, dict) else {}
        self.resolution_range_m = abs(_number(distance_cfg, "width_m", "width", default=300.0))
        self.resolution_doppler_mps = max(target_width, 1.0e-9)
        # The legacy profile key names sensitivity, but its angular value is
        # the full cone width, not a mechanical limit or an off-axis radius.
        beam_width_deg = _number(receiver_antenna, "angle_half_sens_deg", "angleHalfSens", default=15.0)
        self.beam_half_rad = full_cone_half_angle_rad(beam_width_deg)
        transmitter_antenna = (self.radar_config.get("transmitter") or {}).get("antenna") or {}
        self.illumination_half_rad = min(self.beam_half_rad, full_cone_half_angle_rad(_number(
            transmitter_antenna, "angle_half_sens_deg", "angleHalfSens", default=beam_width_deg)))
        # Half-sensitivity widths describe Gaussian antenna power, not a
        # visibility boundary. Keep the receiver width separately for resolution
        # cells and clutter geometry.
        def antenna(kind, fallback_width):
            cfg = (self.radar_config.get(kind) or {}).get('antenna') or {}
            half = full_cone_half_angle_rad(_number(cfg, 'angle_half_sens_deg', 'angleHalfSens', default=fallback_width))
            side_db = min(0., _number(cfg, 'side_lobes_sensitivity_db', 'sideLobesSensitivity', default=-math.inf))
            return half, 10. ** (side_db / 10.)
        self.receiver_antenna = antenna('receiver', beam_width_deg)
        self.transmitter_antenna = antenna('transmitter', beam_width_deg)
        self.reference_rcs_m2 = _number(receiver, 'rcs_m2', 'rcs', default=1.)
        if self.reference_rcs_m2 <= 0:
            raise ValueError('receiver.rcs_m2 must be positive')
        self.last_reflector = "target"
        # "look_down" (default): any reflector below the missile sits on clutter.
        # "geometric_mainlobe" (opt-in, D): only when the beam's lower edge reaches
        # flat ground no farther than the reflector, i.e. ground is behind it.
        # "look_down_angle" (opt-in, C/D): the line of sight must point at least
        # clutter_min_depression_deg below the horizon (player reports: notch
        # needs look-down, not co-altitude; terrain is not modelled).
        self.clutter_model = "look_down"
        self.clutter_min_depression_rad = deg_to_rad(2.0)
        # Opt-in (C, player reports): with no ground clutter behind a beaming
        # target the seeker leaves Doppler (PD) operation, so any chaff bundle
        # unresolved in range and angle captures the track; without chaff the
        # target is still tracked. Decoy path only.
        self.cw_on_clear_beam = False
        self.cw_ticks = 0
        # Opt-in multipath (see enable_multipath); None leaves measurements on truth.
        self.multipath_gain: float | None = None
        self.multipath_nodes: tuple[tuple[float, float], ...] = ()

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
        if self.search is not None:
            self.search = SeekerSearch(self, self.search.options)
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
        self.measurement_identity = None
        self.estimated_track = None
        self.search_estimate_position = None
        if self.chaff_capture is not None:
            self.enable_chaff_capture(self.chaff_capture)
        if self.track_gate_model is not None:
            self.enable_track_gate_model(self.track_gate_model)

    def set_estimated_track(self, solution: TrackSolution) -> None:
        """Commit the provider's estimate after acceptance, before next prediction.

        The detection timestamp remains the real echo timestamp. This method is
        called on accepted radar observations, so the estimate is at that same
        time. Scalar range/Doppler gates still contain actual echo measurements.
        """
        self._last_position, self._last_velocity = solution.position, solution.velocity
        self.estimated_track = solution
        self.last_solution = solution
        if self.track_gate_model is not None:
            self.track_gate_prediction = (solution.solution_time_s, solution.position, solution.velocity)
            self.track_gate_cached_result = (self.last_detection, solution)

    def enable_chaff_capture(self, options):
        options = validate_chaff_capture(options)
        if options is not None and self.track_gate_model is not None:
            raise ValueError('track_gate_model and chaff_capture are mutually exclusive')
        if options is not None and not self.active:
            raise ValueError('chaff_capture requires active radar')
        self.chaff_capture = options
        self.capture_trials = {}
        self.capture_identity = None
        self.capture_previous = None

    def capture_summary(self):
        return dict(options=dict(self.chaff_capture), trials=[dict(v) for v in self.capture_trials.values()],
                    parameter_sources=dict(
                        vg_curve=dict(kind='calibrated', grade='C', source='2026-10-08 WT replay releases'),
                        functional_form=dict(kind='assumed', grade='D', source='saturated exponential'),
                        delay_s=dict(kind='assumed', grade='D', source='fixed 1.3 s proxy for sampled capture median'),
                        association=dict(kind='assumed', grade='D', source='earliest successful release; stable bundle order')))

    def enable_multipath(self, gain: float) -> None:
        """Opt-in surface multipath on the measured position (D, see the evidence ledger).

        ``multipathEffect`` is read as (height above the surface in m, strength)
        nodes: the 2.37 changelog ends the effect 60 m above the surface (B) and
        the AAM tables are [0, 1, 60, 0]. The seeker measures the angle centroid
        of the reflector and its flat-surface image at -height: the point
        ``gain * strength`` of the way to the image. Gain 0.5 at full strength
        aims at the surface below the reflector; gain 1 at the image. Gates and
        range stay on the true reflector. Flat surface at y = 0, no noise.
        """
        if isinstance(gain, bool) or not isinstance(gain, (int, float)) or not 0.0 <= gain <= 1.0:
            raise ValueError("multipath gain must be between 0 and 1")
        self.multipath_nodes = _multipath_table(self.radar_config.get("multipath_effect",
                                                                      self.radar_config.get("multipathEffect")))
        self.multipath_gain = float(gain)

    def _multipath(self, position: Vector, velocity: Vector) -> tuple[Vector, Vector]:
        height = float(position[1])
        shrink = 1.0 - 2.0 * self.multipath_gain * _interpolate(self.multipath_nodes, height)
        return ((float(position[0]), height * shrink, float(position[2])),
                (float(velocity[0]), float(velocity[1]) * shrink, float(velocity[2])))

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

    def _within_track_hold(self, time_s: float) -> bool:
        if self._last_detection_time_s is None:
            return False
        age_s = max(0.0, float(time_s) - self._last_detection_time_s)
        return age_s <= self.prolongation_time_max_s or math.isclose(
            age_s, self.prolongation_time_max_s, rel_tol=0.0, abs_tol=1.0e-12)

    def _coast_solution(self, time_s: float) -> TrackSolution | None:
        if self._last_detection_time_s is None:
            return None
        age_s = max(0.0, float(time_s) - self._last_detection_time_s)
        if not self._within_track_hold(time_s):
            self._state = SeekerState.LOST
            return None
        self._state = SeekerState.COAST
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
        solution = self._coast_solution(time_s)
        if solution is None:
            self._state = SeekerState.SEARCH
        self.last_solution = solution
        return detection, solution

    def return_gain(self, los, antenna_center):
        """Transmit and receive power gain at the circular beam offset."""
        theta = math.acos(clamp(dot(antenna_center, los), -1., 1.))
        return math.prod(SeekerSearch.gain(theta, antenna)
                         for antenna in (self.receiver_antenna, self.transmitter_antenna))

    def detection_range(self, los, antenna_center, rcs_m2):
        if isinstance(rcs_m2, bool) or not isinstance(rcs_m2, (int, float)) or not math.isfinite(rcs_m2) or rcs_m2 < 0:
            raise ValueError('target/reflector rcs_m2 must be finite and nonnegative')
        if rcs_m2 == 0:
            return 0.
        return self.receiver_range_m * (rcs_m2 / self.reference_rcs_m2
                                       * self.return_gain(los, antenna_center)) ** .25

    def _receiver_range_limit(self, searching: bool) -> float:
        # lockDistance enables search from the available target estimate.
        # Once such an estimate is supplied, actual echoes still obey the
        # receiver's physical rangeMax, rather than a second true-16km switch.
        if searching and self.search_estimate_position is None and not self.require_search_estimate:
            return min(self.lock_range_m, self.receiver_range_max_m)
        return self.receiver_range_max_m

    def _search_activation_reason(self, time_s: float, missile_state) -> str | None:
        if not self.active or self._within_track_hold(time_s):
            return None
        estimate = self.search_estimate_position
        if estimate is None:
            return 'search_estimate_unavailable' if self.require_search_estimate else None
        if norm(sub(estimate, missile_state.position)) > self.lock_range_m:
            return 'INS_range_search_disabled'
        return None

    def update(
        self,
        time_s: float,
        dt_s: float,
        missile_state: SimState,
        target_truth: TargetState,
        decoys: list[Any] | None = None,
        target_rcs_m2: float = 1.0,
    ) -> tuple[RadarDetection, TrackSolution | None]:
        time_s = float(time_s)
        if self.active and time_s < self.startup_delay_s - 1e-12:
            detection = self._detection(False, "startup_delay", 0., 0., 0., 0., 0., False)
            self._state = SeekerState.SEARCH
            self.last_detection, self.last_solution = detection, None
            return detection, None
        reason = self._search_activation_reason(time_s, missile_state)
        if reason is not None:
            detection = self._detection(False, reason, 0., 0., 0., 0., 0., False)
            self.last_detection = detection
            return self._invalid_result(detection, time_s)
        if self.track_gate_model is not None:
            return self._update_track_gate(float(time_s), dt_s, missile_state, target_truth, decoys or [], target_rcs_m2)
        if self.chaff_capture is not None:
            return self._update_vg_capture(float(time_s), dt_s, missile_state, target_truth, decoys or [], target_rcs_m2)
        if decoys is not None:
            return self._update_with_decoys(float(time_s), dt_s, missile_state, target_truth, decoys, target_rcs_m2)
        if self.search is not None:
            return self._update_with_decoys(float(time_s), dt_s, missile_state, target_truth, [], target_rcs_m2)
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
        forward = body_axes_for_state(missile_state).forward
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

        track_hold = self._within_track_hold(time_s)
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
        range_limit_m = self._receiver_range_limit(searching)
        if range_m > range_limit_m:
            detection = self._detection(False, "range_gate", **detection_kwargs)
            self.last_detection = detection
            return self._invalid_result(detection, time_s)
        center = self._antenna_center(missile_state, los_hat, searching, time_s)
        if range_m > self.detection_range(los_hat, center, target_rcs_m2):
            detection = self._detection(False, "beam_range_gate", **detection_kwargs)
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

        if self._on_clutter(look_down, missile_state.position, range_m, relative) \
                and abs(ground_radial_mps) <= self.notch_half_width_mps:
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
        measured, measured_velocity = target_truth.position, target_truth.velocity
        if self.multipath_gain is not None:
            measured, measured_velocity = self._multipath(measured, measured_velocity)
        if self.position_only_measurements:
            velocity = (0.0, 0.0, 0.0)
        elif self.use_target_velocity:
            velocity = tuple(float(value) for value in measured_velocity)
        elif self._last_measurement_position is None or self._last_measurement_time_s is None or time_s <= self._last_measurement_time_s:
            velocity = (0.0, 0.0, 0.0)
        else:
            velocity = scale(
                sub(measured, self._last_measurement_position),
                1.0 / (time_s - self._last_measurement_time_s),
            )
        position = tuple(float(value) for value in measured)
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
        self.measurement_identity = ('target',)
        self._search_gates_reset = False
        self.last_detection = detection
        self.last_solution = solution
        return detection, solution

    def _clear_beam(self, missile_state, target_truth, kwargs) -> bool:
        """Target beaming (inside the notch window) with no ground clutter behind it."""
        if abs(kwargs["ground_radial_mps"]) > self.notch_half_width_mps:
            return False
        relative = sub(target_truth.position, missile_state.position)
        return not self._on_clutter(kwargs["look_down"], missile_state.position, kwargs["range_m"], relative)

    def _on_clutter(self, look_down, missile_position, range_m, relative) -> bool:
        if self.clutter_model == "look_down":
            return look_down
        if self.clutter_model == "look_down_angle":
            return -relative[1] / max(range_m, 1.0e-12) >= math.sin(self.clutter_min_depression_rad)
        # Flat earth, y up: the beam's lower edge (LOS depression + half-power
        # half-angle) reaches the ground at slant range h / sin(edge). Ground
        # within one range cell beyond the reflector counts as behind it.
        depression = math.asin(clamp(-relative[1] / max(range_m, 1.0e-12), -1.0, 1.0))
        edge = depression + self.beam_half_rad
        if edge <= 0.0:
            return False
        return max(0.0, float(missile_position[1])) / math.sin(edge) <= range_m + self.resolution_range_m

    def pointable_direction(self, missile_state, direction, searching):
        limit = self.lock_angle_max_rad if searching else self.angle_max_rad
        return limited_antenna_direction(body_axes_for_state(missile_state), direction,
                                         (-limit, limit), (-limit, limit))

    def _antenna_center(self, missile_state, fallback_los, searching, time_s=None):
        if self.search is not None:
            return self.search.beam
        estimate = self.search_estimate_position if searching else self.track_gate_estimate
        if estimate is None:
            estimate = self.track_gate_estimate
        if not searching and self._last_position is not None:
            age = 0. if time_s is None else max(0., time_s-self._last_detection_time_s)
            estimate = add(self._last_position, scale(self._last_velocity, age))
        relative = None if estimate is None else sub(estimate, missile_state.position)
        direction = fallback_los if relative is None or norm(relative) <= 1e-12 else normalize(relative)
        return self.pointable_direction(missile_state, direction, searching)

    def _reflector_check(self, effective_dt, missile_state, position, velocity, track_hold, searching, *, capture=False, antenna_center=None, time_s=None, rcs_m2=1.):
        """The single-target gates applied to one reflector -> (reason, detection kwargs, LOS)."""
        relative = sub(position, missile_state.position)
        range_sq = dot(relative, relative)
        range_m = math.sqrt(max(0.0, range_sq))
        if not math.isfinite(range_m) or range_m <= 1.0e-12:
            return "invalid_geometry", dict(range_m=range_m, range_rate_mps=0.0, ground_radial_mps=0.0,
                                            off_boresight_deg=180.0, los_rate_deg_s=math.inf, look_down=False), None
        los_hat = normalize(relative)
        relative_velocity = sub(velocity, missile_state.velocity)
        range_rate_mps = dot(relative_velocity, los_hat)
        ground_radial_mps = dot(velocity, los_hat)
        look_down = position[1] < missile_state.position[1]
        forward = body_axes_for_state(missile_state).forward
        off_boresight_rad = math.acos(clamp(dot(forward, los_hat), -1.0, 1.0))
        los_rate_rad_s = norm(cross(relative, relative_velocity)) / max(range_sq, 1.0e-12)
        kwargs = dict(range_m=range_m, range_rate_mps=range_rate_mps, ground_radial_mps=ground_radial_mps,
                      off_boresight_deg=rad_to_deg(off_boresight_rad), los_rate_deg_s=rad_to_deg(los_rate_rad_s),
                      look_down=look_down)
        if not all(math.isfinite(kwargs[k]) for k in ("range_rate_mps", "ground_radial_mps", "off_boresight_deg", "los_rate_deg_s")):
            return "invalid_geometry", kwargs, None
        range_limit_m = self._receiver_range_limit(searching)
        if range_m > range_limit_m:
            return "range_gate", kwargs, los_hat
        center = antenna_center if antenna_center is not None else self._antenna_center(missile_state, los_hat, searching, time_s)
        if range_m > self.detection_range(los_hat, center, rcs_m2):
            return "beam_range_gate", kwargs, los_hat
        if los_rate_rad_s > self.rate_max_rad_s:
            return "rate_gate", kwargs, los_hat
        if not capture and track_hold and self._last_accepted_los is not None and math.isfinite(self.angle_gate_rate_rad_s):
            if math.acos(clamp(dot(self._last_accepted_los, los_hat), -1.0, 1.0)) / effective_dt > self.angle_gate_rate_rad_s:
                return "angle_gate_rate", kwargs, los_hat
        if not capture and (range_rate_mps < self.doppler_min_mps or range_rate_mps > self.doppler_max_mps):
            return "doppler_out_of_bounds", kwargs, los_hat
        if not capture and self._on_clutter(look_down, missile_state.position, range_m, relative) \
                and abs(ground_radial_mps) <= self.notch_half_width_mps:
            return "ground_clutter_notch", kwargs, los_hat
        if not capture and not self.distance_gate.accepts(range_m, effective_dt):
            return "distance_gate", kwargs, los_hat
        if not capture and not self.doppler_gate.accepts(range_rate_mps, effective_dt):
            return "doppler_gate", kwargs, los_hat
        return "", kwargs, los_hat

    def enable_track_gate_model(self, options):
        options = validate_track_gate_model(options)
        if options is not None and (not self.active or self.chaff_capture is not None):
            raise ValueError('track_gate_model requires active radar and cannot combine with chaff_capture')
        self.track_gate_model = options
        self.track_gate_identity = None
        self.track_gate_prediction = None
        self.track_gate_ever_locked = False
        self.track_gate_events = []
        self.track_gate_last_time = None
        self.track_gate_cached_result = None
        self.track_gate_estimate = None

    def track_gate_summary(self):
        return dict(options=dict(self.track_gate_model), events=[dict(e) for e in self.track_gate_events],
                    parameter_sources=dict(
                        tracking_notch=dict(kind='assumed', grade='D', source='aircraft PD official tracking rule extended to missile; replay C'),
                        notch_track_hazard=dict(kind='calibrated', grade='C', source='five WT replays 2026-10-09; no-chaff exposure'),
                        datalink_after_lock=dict(kind='assumed', grade='C', source='2026-10-09 replay return after chaff; opt-in overrides reconnect_datalink assumption'),
                        acquisition=dict(kind='assumed', grade='C/D', source='2026-10-09 replay memory-error acquisition; nominal 5 degrees, pending calibration'),
                        reacquisition=dict(kind='assumed', grade='C/D', source='replay 3-8 degree window; nominal 5 degrees'),
                        gate_widths=dict(kind='assumed', grade='D', source='unknown full/half-width mapping; nominal 150 m / 10 m/s'),
                        angle_gate=dict(kind='assumed', grade='D', source='official mechanism B; nominal 5 degrees, blkx angleGateRate'),
                        competition=dict(kind='assumed', grade='D', source='RCS/range^4; optional keyed lognormal fluctuation')))

    def _gate_draw(self, time_s, identity, purpose):
        key = json.dumps([self.track_gate_model['seed'], float(time_s).hex(), identity, purpose],
                         sort_keys=True, separators=(',', ':'), allow_nan=False)
        digest = hashlib.sha256(key.encode()).digest()
        return (int.from_bytes(digest[:8], 'big') + .5) / 2**64

    def _update_track_gate(self, time_s, dt_s, missile_state, target_truth, decoys, target_rcs_m2):
        # One evaluation per timestamp: hazard/fluctuation cannot be redrawn by
        # terminal sampling or repeated observation at the same control instant.
        if self.track_gate_last_time == time_s:
            return self.track_gate_cached_result
        if self.track_gate_last_time is not None and time_s < self.track_gate_last_time:
            raise ValueError('track_gate_model observation times must be nondecreasing')
        elapsed = 0. if self.track_gate_last_time is None else time_s-self.track_gate_last_time
        options = self.track_gate_model
        tracking = self._within_track_hold(time_s) and self.track_gate_prediction is not None
        effective_dt = self._effective_dt(time_s, dt_s)
        if self.search is not None:
            self.search.prepare(time_s, missile_state, target_truth, not tracking)
        reflectors = [('target', ('target',), tuple(target_truth.position), tuple(target_truth.velocity), target_rcs_m2, None)]
        for d in decoys:
            if d.identity is None:
                raise ValueError('track_gate_model requires stable reflector identity')
            # Validate stable identifiers once per echo; never use process hash().
            try:
                json.dumps(d.identity, sort_keys=True, allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise ValueError('track_gate_model identities must be JSON serializable') from exc
            reflectors.append((d.kind, ('decoy', d.identity), tuple(d.position), tuple(d.velocity), d.rcs_m2,
                               getattr(d, 'release_time_s', None)))
        for r in reflectors:
            if isinstance(r[4], bool) or not isinstance(r[4], (int, float)) or not math.isfinite(r[4]) or r[4] < 0:
                raise ValueError('target/reflector rcs_m2 must be finite and nonnegative')
        forward = body_axes_for_state(missile_state).forward
        if tracking:
            stamp, position, velocity = self.track_gate_prediction
            predicted = add(position, scale(velocity, time_s-stamp))
            relative = sub(predicted, missile_state.position)
            predicted_range = norm(relative)
            center = normalize(relative) if predicted_range > 1e-12 else forward
            relative_velocity = sub(velocity, missile_state.velocity)
            predicted_speed = dot(relative_velocity, center)
            predicted_omega = scale(cross(relative, relative_velocity), 1./max(predicted_range**2, 1e-12))
        else:
            estimate = self.search_estimate_position
            if estimate is None:
                estimate = self.track_gate_estimate
            if estimate is None and self.track_gate_prediction is not None:
                stamp, position, velocity = self.track_gate_prediction
                estimate = add(position, scale(velocity, time_s-stamp))
            relative = None if estimate is None else sub(estimate, missile_state.position)
            center = forward if relative is None or norm(relative) <= 1e-12 else normalize(relative)
        # Association uses the designation/predicted LOS before mechanical
        # clipping. The antenna beam independently tests the reachable center.
        antenna_center = (self.search.beam if self.search is not None else
                          self.pointable_direction(missile_state, center, not tracking))
        checked = []
        for r in reflectors:
            kind, identity, position, velocity, rcs, release = r
            reason, kw, los = self._reflector_check(effective_dt, missile_state, position, velocity,
                                                   False, not tracking, capture=tracking, antenna_center=antenna_center, rcs_m2=rcs)
            if not reason and rcs <= 0:
                reason = 'zero_return'
            if not reason and tracking:
                angle = math.acos(clamp(dot(center, los), -1., 1.))
                rel = sub(position, missile_state.position)
                omega = scale(cross(rel, sub(velocity, missile_state.velocity)), 1./max(kw['range_m']**2, 1e-12))
                if kw['range_rate_mps'] < self.doppler_min_mps or kw['range_rate_mps'] > self.doppler_max_mps:
                    reason = 'doppler_out_of_bounds'
                elif abs(kw['range_m']-predicted_range) > options['range_gate_m']:
                    reason = 'track_range_gate'
                elif abs(kw['range_rate_mps']-predicted_speed) > options['speed_gate_mps']:
                    reason = 'track_speed_gate'
                elif angle > deg_to_rad(options['angle_gate_deg']):
                    reason = 'track_angle_gate'
                elif norm(sub(omega, predicted_omega)) > self.angle_gate_rate_rad_s:
                    reason = 'track_angle_rate_gate'
            elif not reason:
                half = options['reacquire_half_deg'] if self.track_gate_ever_locked else options['acquire_half_deg']
                if half is not None and math.acos(clamp(dot(center, los), -1., 1.)) > deg_to_rad(half):
                    reason = 'reacquire_window' if self.track_gate_ever_locked else 'acquire_window'
            checked.append((r, reason, kw, los))
        candidates = [c for c in checked if not c[1]]
        def power(c):
            value = c[0][4] * self.return_gain(c[3], antenna_center) / max(c[2]['range_m'], 1.)**4
            if options['compete'] == 'stronger_fluct':
                u = self._gate_draw(time_s, c[0][1], 'fluct_u')
                v = self._gate_draw(time_s, c[0][1], 'fluct_v')
                z = math.sqrt(-2.*math.log(u))*math.cos(2.*math.pi*v)
                sigma = options['fluct_sigma']
                value *= math.exp(sigma*z-.5*sigma*sigma)
            return value
        chosen = max(candidates, key=power) if candidates else None
        failure = checked[0][1] or 'track_gate_empty'
        failure_kw = checked[0][2]
        # Only the target's own notch hazard is stochastic; there is no release
        # probability and no >=75 m/s background term.
        if chosen is not None and tracking and chosen[0][0] == 'target':
            kw = chosen[2]
            if self._on_clutter(kw['look_down'], missile_state.position, kw['range_m'],
                                sub(chosen[0][2], missile_state.position)) and abs(kw['ground_radial_mps']) <= self.notch_half_width_mps:
                vg = abs(kw['ground_radial_mps'])
                hazard = next((h for upper, h in options['notch_track_hazard'] if vg < upper), 0.)
                probability = -math.expm1(-hazard*elapsed)
                if self._gate_draw(time_s, chosen[0][1], 'notch_loss') < probability:
                    failure, failure_kw, chosen = 'track_notch_hazard', kw, None
        if chosen is not None and self.search is not None and not self.search.acquisition(time_s, chosen[0][1]):
            failure, failure_kw, chosen = 'acquisition_dwell', chosen[2], None
        if chosen is None:
            if self.search is not None:
                self.search.miss()
            detection = self._detection(False, failure, **failure_kw)
            self.last_detection = detection
            if tracking and self._state is SeekerState.TRACK:
                self.track_gate_events.append(dict(time_s=time_s, event='loss', identity=self.track_gate_identity, reason=failure))
            result = self._invalid_result(detection, time_s)
        else:
            r, _, kw, los = chosen
            kind, identity, position, truth_velocity, _, release = r
            if not tracking or identity != self.track_gate_identity:
                self.track_gate_events.append(dict(time_s=time_s, event='acquire' if not tracking else 'transfer',
                    identity=identity, previous_identity=self.track_gate_identity, release_time_s=release,
                    pointing_error_deg=rad_to_deg(math.acos(clamp(dot(center, los), -1., 1.)))))
            self.track_gate_identity = identity
            self.track_gate_prediction = (time_s, position, (0., 0., 0.) if self.position_only_measurements else truth_velocity)
            self.track_gate_ever_locked = True
            if self.multipath_gain is not None:
                position, truth_velocity = self._multipath(position, truth_velocity)
            velocity = truth_velocity
            if self.position_only_measurements:
                velocity = (0., 0., 0.)
            elif not self.use_target_velocity:
                velocity = ((0., 0., 0.) if self._last_measurement_time_s is None or time_s <= self._last_measurement_time_s
                            else scale(sub(position, self._last_measurement_position), 1./(time_s-self._last_measurement_time_s)))
            self.distance_gate.update(kw['range_m'], effective_dt)
            self.doppler_gate.update(kw['range_rate_mps'], effective_dt)
            solution = TrackSolution(position=position, velocity=velocity, sample_time_s=time_s,
                solution_time_s=time_s, mode=TrackMode.RADAR_TRACK, valid=True, source='radar')
            detection = self._detection(True, '', **kw)
            self._state = SeekerState.TRACK
            self._last_detection_time_s = self._last_measurement_time_s = time_s
            self._last_accepted_los = los
            self._last_measurement_position = self._last_position = position
            self._last_velocity = velocity
            self.measurement_identity = identity
            self.last_reflector = kind
            self.last_detection, self.last_solution = detection, solution
            result = detection, solution
        self.track_gate_last_time, self.track_gate_cached_result = time_s, result
        return result

    def _update_vg_capture(self, time_s, dt_s, missile_state, target_truth, decoys, target_rcs_m2):
        """One keyed Bernoulli trial per release, delayed range/angle capture.

        New releases require a target track at release. Earliest successful
        release, then stable bundle identity, wins simultaneous candidates.
        A captured bundle is held until it expires or fails hardware geometry;
        normal target acquisition then resumes, including configured scan/dwell.
        """
        options = self.chaff_capture
        previous = self.capture_previous
        on_target = self._state is SeekerState.TRACK and self.last_reflector == 'target'
        groups = {}
        for d in decoys:
            if d.kind != 'chaff':
                continue
            if any(getattr(d, key, None) is None for key in
                   ('identity', 'release_identity', 'release_time_s', 'release_position', 'release_velocity')):
                raise ValueError('chaff_capture requires stable bundle/release identity and release state/time')
            # JSON avoids process-randomized hash() and carries through snapshots.
            try:
                key = json.dumps(d.release_identity, sort_keys=True, separators=(',', ':'), allow_nan=False)
                bundle_key = json.dumps(d.identity, sort_keys=True, separators=(',', ':'), allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise ValueError('chaff_capture identities must be JSON serializable') from exc
            groups.setdefault(key, []).append((bundle_key, d))
            if key in self.capture_trials:
                continue
            release = float(d.release_time_s)
            if not math.isfinite(release) or release > time_s + 1e-12:
                raise ValueError('invalid chaff release time')
            # Interpolate within this control interval; for a first observation
            # after release use the current velocity over that short gap.
            if previous is not None and previous[0] <= release <= time_s and time_s > previous[0]:
                fraction = (release - previous[0]) / (time_s - previous[0])
                missile_position = add(previous[1], scale(sub(missile_state.position, previous[1]), fraction))
                eligible = previous[2]
            else:
                missile_position = add(missile_state.position, scale(missile_state.velocity, release - time_s))
                eligible = on_target and abs(release - time_s) <= 1e-12
            los = sub(d.release_position, missile_position)
            if norm(los) <= 1e-12:
                raise ValueError('invalid release geometry')
            vg = abs(dot(d.release_velocity, normalize(los)))
            probability = options['p_max'] * math.exp(-max(0., vg - options['v0_mps']) / options['scale_mps'])
            digest = hashlib.sha256((str(options['seed']) + ':' + key).encode()).digest()
            draw = int.from_bytes(digest[:8], 'big') / 2**64
            self.capture_trials[key] = dict(release_identity=d.release_identity, release_time_s=release,
                vg_mps=vg, probability=probability, draw=draw, eligible=bool(eligible),
                selected=bool(eligible and draw < probability), captured_time_s=None,
                release_range_m=norm(los), captured_offtarget_time_s=None)
        self.capture_previous = (time_s, tuple(missile_state.position), on_target)
        effective_dt = self._effective_dt(time_s, dt_s)
        hold = self._within_track_hold(time_s)
        if self.search is not None:
            self.search.prepare(time_s, missile_state, target_truth, not hold)
        chosen = None
        candidates = []
        target_los = sub(target_truth.position, missile_state.position)
        target_range = norm(target_los)
        for key, bundles in groups.items():
            trial = self.capture_trials[key]
            for identity, d in sorted(bundles, key=lambda item: item[0]):
                reason, kwargs, los = self._reflector_check(effective_dt, missile_state, d.position, d.velocity,
                                                          hold, not hold, capture=True, time_s=time_s, rcs_m2=d.rcs_m2)
                if reason or d.rcs_m2 <= 0:
                    continue
                if self.capture_identity == (key, identity):
                    chosen = (key, identity, d, kwargs, los)
                    break
                if not trial['selected'] or trial['captured_time_s'] is not None \
                        or time_s + 1e-12 < trial['release_time_s'] + options['delay_s']:
                    continue
                if target_range <= 1e-12 or abs(kwargs['range_m'] - target_range) > self.resolution_range_m:
                    continue
                if math.acos(clamp(dot(los, normalize(target_los)), -1., 1.)) > self.beam_half_rad:
                    continue
                candidates.append((trial['release_time_s'], key, identity, d, kwargs, los))
            if chosen is not None:
                break
        if chosen is None and candidates:
            _, key, identity, d, kwargs, los = min(candidates, key=lambda item: item[:3])
            chosen = (key, identity, d, kwargs, los)
        if chosen is None:
            self.capture_identity = None
            # Reuse the unmodified target gates, never RCS/R^4 decoy competition.
            self.chaff_capture = None
            try:
                result = self.update(time_s, dt_s, missile_state, target_truth, target_rcs_m2=target_rcs_m2)
            finally:
                self.chaff_capture = options
            if result[0].valid:
                self.last_reflector = 'target'
            self.capture_previous = (time_s, tuple(missile_state.position),
                                     result[0].valid and self.last_reflector == 'target')
            return result
        key, identity, d, kwargs, los = chosen
        if self.search is not None and not self.search.acquisition(time_s, ('decoy', d.identity)):
            detection = self._detection(False, 'acquisition_dwell', **kwargs)
            self.last_detection = detection
            self.capture_previous = (time_s, tuple(missile_state.position), False)
            return self._invalid_result(detection, time_s)
        if self.capture_trials[key]['captured_time_s'] is None:
            self.capture_trials[key]['captured_time_s'] = time_s
        if self.capture_trials[key]['captured_offtarget_time_s'] is None \
                and norm(sub(d.position, target_truth.position)) > 300.:
            self.capture_trials[key]['captured_offtarget_time_s'] = time_s
        self.capture_identity = (key, identity)
        position, velocity = tuple(d.position), tuple(d.velocity)
        if self.multipath_gain is not None:
            position, velocity = self._multipath(position, velocity)
        if self.position_only_measurements:
            velocity = (0., 0., 0.)
        elif not self.use_target_velocity:
            velocity = ((0., 0., 0.) if self._last_measurement_time_s is None or time_s <= self._last_measurement_time_s
                        else scale(sub(position, self._last_measurement_position), 1 / (time_s - self._last_measurement_time_s)))
        self.distance_gate.update(kwargs['range_m'], effective_dt)
        self.doppler_gate.update(kwargs['range_rate_mps'], effective_dt)
        solution = TrackSolution(position=position, velocity=velocity, sample_time_s=time_s,
                                 solution_time_s=time_s, mode=TrackMode.RADAR_TRACK, valid=True, source='radar')
        detection = self._detection(True, '', **kwargs)
        self._state = SeekerState.TRACK
        self._last_detection_time_s = time_s
        self._last_accepted_los = los
        self._last_measurement_position = self._last_position = position
        self._last_measurement_time_s = time_s
        self._last_velocity = velocity
        self.measurement_identity = ('decoy', d.identity)
        self._search_gates_reset = False
        self.last_detection, self.last_solution, self.last_reflector = detection, solution, 'chaff'
        self.capture_previous = (time_s, tuple(missile_state.position), False)
        return detection, solution

    def _update_with_decoys(self, time_s, dt_s, missile_state, target_truth, decoys, target_rcs_m2):
        """Opt-in multi-reflector path (D).

        Every reflector passes the same gates as the single-target path. The
        track associates with the candidate nearest the alpha-beta prediction
        (strongest at first acquisition); reflectors unresolved from it in
        range, Doppler and beam then compete, and the largest RCS/R^4 return
        is measured. The aircraft's own gate failure is reported when nothing
        passes.
        """
        if not self.active and self.search is None:
            return self.update(time_s, dt_s, missile_state, target_truth)
        track_hold = self._within_track_hold(time_s)
        if self.search is not None:
            self.search.prepare(time_s, missile_state, target_truth, not track_hold)
            for rcs in [target_rcs_m2, *(getattr(d, 'rcs_m2', None) for d in decoys)]:
                if isinstance(rcs, bool) or not isinstance(rcs, (int, float)) or not math.isfinite(rcs) or rcs < 0:
                    raise ValueError('target/reflector rcs_m2 must be finite and nonnegative')
        if self._last_detection_time_s is not None and not track_hold and not self._search_gates_reset:
            self.distance_gate = AlphaBetaGate(self.distance_gate.alpha, self.distance_gate.beta, self.distance_gate.search_range)
            self.doppler_gate = AlphaBetaGate(self.doppler_gate.alpha, self.doppler_gate.beta, self.doppler_gate.search_range)
            self._last_accepted_los = None
            self._last_measurement_position = None
            self._last_measurement_time_s = None
            self._search_gates_reset = True
        effective_dt = self._effective_dt(time_s, dt_s)
        reflectors = [("target", tuple(target_truth.position), tuple(target_truth.velocity), float(target_rcs_m2))]
        reflectors += [(d.kind, tuple(d.position), tuple(d.velocity), float(d.rcs_m2)) for d in decoys]
        antenna_center = self._antenna_center(missile_state, normalize(sub(target_truth.position, missile_state.position)),
                                               not track_hold, time_s)
        checked = [(r, *self._reflector_check(effective_dt, missile_state, r[1], r[2], track_hold, not track_hold,
                                              antenna_center=antenna_center, rcs_m2=r[3]))
                   for r in reflectors]
        candidates = [(r, kw, los) for r, reason, kw, los in checked if not reason]
        if not candidates:
            if self.search is not None:
                self.search.miss()
            _, reason, kwargs, _ = checked[0]
            detection = self._detection(False, reason, **kwargs)
            self.last_detection = detection
            return self._invalid_result(detection, time_s)

        def power(item):
            return item[0][3] * self.return_gain(item[2], antenna_center) / max(item[1]['range_m'], 1.) ** 4

        if self.distance_gate.initialized and self.doppler_gate.initialized:
            predicted_range = self.distance_gate.predict(effective_dt)
            predicted_rate = self.doppler_gate.predict(effective_dt)
            anchor = min(candidates, key=lambda c: ((c[1]["range_m"] - predicted_range) / self.resolution_range_m) ** 2
                         + ((c[1]["range_rate_mps"] - predicted_rate) / self.resolution_doppler_mps) ** 2)
        else:
            anchor = max(candidates, key=power)

        def unresolved(c):
            return (abs(c[1]["range_m"] - anchor[1]["range_m"]) <= self.resolution_range_m
                    and abs(c[1]["range_rate_mps"] - anchor[1]["range_rate_mps"]) <= self.resolution_doppler_mps
                    and math.acos(clamp(dot(c[2], anchor[2]), -1.0, 1.0)) <= self.beam_half_rad)

        competing = [c for c in candidates if unresolved(c)]
        target_reason, target_kwargs, _ = checked[0][1], checked[0][2], checked[0][3]
        if self.cw_on_clear_beam and self._clear_beam(missile_state, target_truth, target_kwargs):
            self.cw_ticks += 1
            cw_decoys = [c for c in candidates if c[0][0] != "target"
                      and abs(c[1]["range_m"] - anchor[1]["range_m"]) <= self.resolution_range_m
                      and math.acos(clamp(dot(c[2], anchor[2]), -1.0, 1.0)) <= self.beam_half_rad]
            if cw_decoys:
                competing = cw_decoys  # No Doppler: a bundle in the range/angle cell wins.
        chosen = max(competing, key=power)
        (kind, position, truth_velocity, _), kwargs, los_hat = chosen
        identity = ('target',) if chosen[0] is reflectors[0] else next(
            (('decoy', getattr(decoy, 'identity', None)) for decoy, r in zip(decoys, reflectors[1:]) if r is chosen[0]), None)
        if self.position_only_measurements and identity == ('decoy', None):
            raise ValueError('unified target observer requires stable reflector identity')
        if self.search is not None:
            identity = ('target',) if chosen[0] is reflectors[0] else next(
                (getattr(decoy, 'identity', None) for decoy, r in zip(decoys, reflectors[1:]) if r is chosen[0]), None)
            if identity is not None and chosen[0] is not reflectors[0]:
                identity = ('decoy', identity)
            if not self.search.acquisition(time_s, identity):
                detection = self._detection(False, 'acquisition_identity_missing' if identity is None else 'acquisition_dwell', **kwargs)
                self.last_detection = detection
                return self._invalid_result(detection, time_s)
        if self.multipath_gain is not None:
            position, truth_velocity = self._multipath(position, truth_velocity)
        self.distance_gate.update(kwargs["range_m"], effective_dt)
        self.doppler_gate.update(kwargs["range_rate_mps"], effective_dt)
        if self.position_only_measurements:
            velocity = (0., 0., 0.)
        elif self.use_target_velocity:
            velocity = truth_velocity
        elif self._last_measurement_position is None or self._last_measurement_time_s is None or time_s <= self._last_measurement_time_s:
            velocity = (0.0, 0.0, 0.0)
        else:
            velocity = scale(sub(position, self._last_measurement_position), 1.0 / (time_s - self._last_measurement_time_s))
        solution = TrackSolution(position=position, velocity=velocity, sample_time_s=time_s, solution_time_s=time_s,
                                 mode=TrackMode.RADAR_TRACK, valid=True, source="radar")
        detection = self._detection(True, "", **kwargs)
        self._state = SeekerState.TRACK
        self._last_detection_time_s = time_s
        self._last_accepted_los = tuple(float(value) for value in los_hat)
        self._last_measurement_position = position
        self._last_measurement_time_s = time_s
        self._last_position = position
        self._last_velocity = velocity
        self.measurement_identity = identity
        self._search_gates_reset = False
        self.last_detection = detection
        self.last_solution = solution
        self.last_reflector = kind
        return detection, solution


__all__ = ["RadarDetection", "RadarSeekerObserver", "SeekerState"]
