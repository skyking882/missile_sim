"""Three-dimensional proportional-navigation and minimal loft candidate."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .aerodynamics import body_axes_for_state, cg_wind_normal_basis
from .math3d import Vector, clamp, clamp_norm, cross, dot, limit_unit_disk, norm, normalize, scale, sub
from .target import TargetState
from .tracking import TrackMode, TrackSolution
from .units import deg_to_rad, g_to_mps2, mps2_to_g


@dataclass(frozen=True)
class GuidanceOutput:
    enabled: bool
    range_m: float
    closing_speed_mps: float
    los_rate_vector_rad_s: Vector
    pn_acceleration_mps2: Vector
    loft_acceleration_mps2: Vector
    commanded_acceleration_mps2: Vector
    commanded_body_acceleration_g: tuple[float, float]
    controller_specific_force_command_g: tuple[float, float]
    wind_normal_specific_force_command_g: tuple[float, float]
    gravity_compensation_wind_normal_g: tuple[float, float]
    wind_normal_pitch_axis: Vector
    wind_normal_yaw_axis: Vector
    effective_gain: float
    time_to_go_s: float
    loft_active: bool
    within_lock_range: bool
    midcourse_weight: float
    heading_error_rad: float
    pip_time_to_go_s: float
    midcourse_fin_fraction: tuple[float, float]
    # PCC-alpha telemetry (midcourse_lead_turn.mode == "pcc_alpha" only;
    # "off" / 0.0 whenever the timer_blend path or no midcourse is active).
    capture_mode: str = "off"
    capture_ratio_r: float = 0.0
    capture_envelope_g: float = 0.0
    # v0.1 release washout: the alpha command actually routed to the fins
    # after the asymmetric washout filter (equals the raw capture alpha while
    # release_washout_time_s == 0, i.e. the v0 path).
    capture_routed_alpha_deg: float = 0.0


def interpolate_table(x: float, table: list[list[float]]) -> float:
    if not table:
        return 1.0
    points = sorted((float(row[0]), float(row[1])) for row in table)
    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= x <= x1:
            fraction = (x - x0) / (x1 - x0)
            return y0 + fraction * (y1 - y0)
    return points[-1][1]


# Below these floors the 1/R and 1/R^2 PN terms are leftover numerics, not
# a guidance solution.  Do not substitute the floor into the divisor.
NEAR_RANGE_M = 1.0e-6
NEAR_CLOSING_MPS = 1.0e-6
NEAR_LOS_RATE = 1.0e-8
TIME_TO_GO_CLOSING_SPEED = "closing_speed"
TIME_TO_GO_CLOSING_OR_RELATIVE_SPEED = "closing_or_relative_speed"


def _flush_tiny(value: float, floor: float) -> float:
    return 0.0 if abs(value) < floor else value


def pn_acceleration(
    relative_position: Vector,
    relative_velocity: Vector,
    missile_velocity: Vector,
    pn_gain: float,
    epsilon: float = NEAR_RANGE_M,
) -> tuple[Vector, float, Vector]:
    """Return candidate PN acceleration, closing speed, and LOS rate."""

    range_sq = dot(relative_position, relative_position)
    range_m = math.sqrt(range_sq)
    if range_m <= epsilon:
        return (0.0, 0.0, 0.0), 0.0, (0.0, 0.0, 0.0)
    missile_speed = norm(missile_velocity)
    v_hat = normalize(missile_velocity)
    los_rate = scale(cross(relative_position, relative_velocity), 1.0 / range_sq)
    los_rate = (
        _flush_tiny(los_rate[0], NEAR_LOS_RATE),
        _flush_tiny(los_rate[1], NEAR_LOS_RATE),
        _flush_tiny(los_rate[2], NEAR_LOS_RATE),
    )
    closing = -dot(relative_position, relative_velocity) / range_m
    if closing <= NEAR_CLOSING_MPS or missile_speed <= epsilon:
        return (0.0, 0.0, 0.0), _flush_tiny(closing, NEAR_CLOSING_MPS), los_rate
    acceleration = scale(cross(los_rate, v_hat), pn_gain * closing)
    return acceleration, closing, los_rate


def pn_acceleration_from_los_rate(
    los_rate: Vector,
    missile_velocity: Vector,
    closing_mps: float,
    pn_gain: float,
    epsilon: float = NEAR_RANGE_M,
) -> Vector:
    """Rebuild the PN product for an externally supplied LOS-rate vector.

    ``pn_acceleration`` stays a pure geometric measurement of the true LOS
    rate; a caller that wants to shape that rate first (the PCC-alpha seeker
    trackloop surrogate, ``los_rate_filter_time_constant_s``) filters the
    vector outside and re-forms the same N*Vc*(omega x v_hat) product here,
    with the identical near-range/near-closing guards.
    """

    missile_speed = norm(missile_velocity)
    if closing_mps <= NEAR_CLOSING_MPS or missile_speed <= epsilon:
        return (0.0, 0.0, 0.0)
    return scale(cross(los_rate, normalize(missile_velocity)), pn_gain * float(closing_mps))


def estimate_time_to_go_s(
    range_m: float,
    closing_mps: float,
    relative_velocity: Vector,
    config: dict[str, Any],
) -> float:
    """Return t_go used by timeToHitGain and loft exit.

    Frozen H1/H2 keep R/Vc.  Profile H2 floors the divisor at κ|V_rel| so a
    beam shot cannot inflate t_go and then get chopped by the gain table.
    Head-on is unchanged because |V_rel| ≈ Vc.
    """

    mode = str(config.get("guidance", {}).get("time_to_go_mode", TIME_TO_GO_CLOSING_SPEED))
    if mode == TIME_TO_GO_CLOSING_OR_RELATIVE_SPEED:
        kappa = float(config["guidance"].get("time_to_go_relative_speed_weight", 1.0))
        if not math.isfinite(kappa) or kappa < 0.0:
            raise ValueError("guidance.time_to_go_relative_speed_weight must be finite and >= 0")
        divisor = max(float(closing_mps), kappa * norm(relative_velocity))
    elif mode == TIME_TO_GO_CLOSING_SPEED:
        divisor = float(closing_mps)
    else:
        raise ValueError(f"unknown time_to_go_mode: {mode}")
    if divisor <= NEAR_CLOSING_MPS:
        return 0.0
    return float(range_m) / divisor


def solve_pip_time_to_go_s(
    relative_position: Vector,
    target_velocity: Vector,
    missile_speed_mps: float,
) -> float:
    """Solve the collision triangle |R + V_t t| = V_bar t for the smallest positive root.

    ``missile_speed_mps`` is the caller's already speed-floored V_bar; it sets
    the quadratic's leading coefficient and, via ``max(V_bar, 1)``, the
    fallback divisor used when the triangle has no positive root (the target
    outruns the assumed missile speed along the current geometry).
    """

    v_bar = float(missile_speed_mps)
    fallback = norm(relative_position) / max(v_bar, 1.0)
    a = dot(target_velocity, target_velocity) - v_bar * v_bar
    b = 2.0 * dot(relative_position, target_velocity)
    c = dot(relative_position, relative_position)
    roots: list[float] = []
    if abs(a) <= 1.0e-9:
        if abs(b) > 1.0e-9:
            roots.append(-c / b)
    else:
        discriminant = b * b - 4.0 * a * c
        if discriminant >= 0.0:
            sqrt_discriminant = math.sqrt(discriminant)
            roots.append((-b + sqrt_discriminant) / (2.0 * a))
            roots.append((-b - sqrt_discriminant) / (2.0 * a))
    positive_roots = [root for root in roots if root > 0.0]
    return min(positive_roots) if positive_roots else fallback


def midcourse_blend_weight(time_s: float, midcourse_cfg: dict[str, Any]) -> float:
    """Return w(t): 1 before lock_delay_s+hold_time_s, linear down to 0 across blend_time_s.

    PN stays on the whole flight (it is small early); this weight only keeps
    the lead-turn term from double-counting once PN has taken over.

    ``hold_time_s`` (default 0.0) extends the full-authority (w=1) plateau
    past ``lock_delay_s`` before the linear release begins; it replaces the
    old linear-fade-from-lock_delay_s shape with a hold-then-release shape.
    hold_time_s=0.0 collapses hold_end_s to lock_delay_s and reproduces the
    original ramp exactly (bit-identical for every config that does not set
    hold_time_s).
    """

    lock_delay_s = float(midcourse_cfg.get("lock_delay_s", 0.8))
    hold_time_s = float(midcourse_cfg.get("hold_time_s", 0.0))
    blend_time_s = float(midcourse_cfg.get("blend_time_s", 0.5))
    hold_end_s = lock_delay_s + hold_time_s
    if time_s < hold_end_s:
        return 1.0
    if blend_time_s <= 0.0:
        return 0.0
    fraction = (time_s - hold_end_s) / blend_time_s
    return 0.0 if fraction >= 1.0 else 1.0 - fraction


def midcourse_lead_turn_acceleration(
    relative_position: Vector,
    track_position: Vector,
    track_velocity: Vector,
    missile_position: Vector,
    missile_velocity: Vector,
    v_hat: Vector,
    midcourse_cfg: dict[str, Any],
    loft_active: bool = False,
    loft_elevation_rad: float = 0.0,
) -> tuple[Vector, float, float, Vector]:
    """Return (acceleration, heading_error_rad, pip_time_to_go_s, e_hat) for the launch lead-turn candidate.

    Solves for the predicted intercept point (PIP), then commands a lateral
    acceleration proportional to missile speed and heading error over a turn
    time constant.  Not itself clamped; the caller blends it with PN/loft and
    lets the existing perp-to-v projection and reqAccelMax clamp apply once.
    ``e_hat`` (the unit direction toward the PIP, normal to v_hat) is also
    returned so the caller can derive a direct fin-fraction command from the
    same geometry instead of only the accelerometer-shaped acceleration.

    While the loft program is active the IOG target is not the raw PIP
    bearing: a level target's PIP sits at (near) constant altitude, so once
    loft has climbed the missile above it the raw PIP pitch goes *negative*
    (the target now looks "below"), and the unmodified lead-turn would
    command pitching down to chase it -- fighting the climb.  d_hat's pitch
    is floored at the missile's own current pitch (never asked to point
    below where the missile already is, so there is no downward fight) and
    capped at loft_elevation_rad (so this never *adds* its own independent
    climb demand on top of loft's dedicated pitch_error term once the
    airframe is still short of the loft target -- loft alone owns getting
    there).  The PIP's own horizontal azimuth is kept unchanged either way.
    """

    turn_time_constant_s = float(midcourse_cfg.get("turn_time_constant_s", 0.5))
    speed_floor_mps = float(midcourse_cfg.get("speed_floor_mps", 200.0))
    missile_speed = norm(missile_velocity)
    v_bar = max(missile_speed, speed_floor_mps)
    pip_time_to_go_s = solve_pip_time_to_go_s(relative_position, track_velocity, v_bar)
    pip_position = add_vectors(track_position, scale(track_velocity, pip_time_to_go_s))
    d_hat = normalize(sub(pip_position, missile_position), fallback=v_hat)
    if loft_active:
        horizontal_hat = normalize((d_hat[0], 0.0, d_hat[2]))
        pip_pitch_rad = math.asin(clamp(d_hat[1], -1.0, 1.0))
        current_pitch_rad = math.asin(clamp(v_hat[1], -1.0, 1.0))
        target_pitch_rad = clamp(
            max(pip_pitch_rad, current_pitch_rad),
            -0.5 * math.pi,
            max(loft_elevation_rad, current_pitch_rad),
        )
        d_hat = (
            horizontal_hat[0] * math.cos(target_pitch_rad),
            math.sin(target_pitch_rad),
            horizontal_hat[2] * math.cos(target_pitch_rad),
        )
    heading_error_rad = math.acos(clamp(dot(d_hat, v_hat), -1.0, 1.0))
    perpendicular = sub(d_hat, scale(v_hat, dot(d_hat, v_hat)))
    e_hat = normalize(perpendicular, fallback=(0.0, 0.0, 0.0))
    acceleration = scale(e_hat, missile_speed * heading_error_rad / turn_time_constant_s)
    return acceleration, heading_error_rad, pip_time_to_go_s, e_hat


PCC_CAPTURE = "capture"
PCC_HOMING = "homing"

# --- PCC-alpha v0.1 stateful filters (docs/PCC_ALPHA_V0.md, "v0.1") ---------
#
# The launch-capture alpha command does not vanish at release in the game data;
# it washes out over roughly a second.  Only the DOWNWARD side of that command
# carries that memory -- upward the fin cannot move faster than its actuator
# anyway, so the rise uses a fixed actuator-scale constant and only the decay
# is the calibrated ``release_washout_time_s``.
PCC_WASHOUT_RISE_TIME_S = 0.05
_PCC_FILTER_TIME_KEY = "pcc_filter_time_s"
_PCC_WASHOUT_FRACTION_KEY = "pcc_washout_fin_fraction"
_PCC_WASHOUT_WEIGHT_KEY = "pcc_washout_weight"
_PCC_LOS_RATE_KEY = "pcc_los_rate_filtered"


def pcc_filter_step_s(time_s: float, guidance_state: dict[str, Any]) -> float:
    """Return the elapsed time since the PCC filters last advanced.

    ``guidance_command`` runs TWICE per timestamp in the H2 run loop (once in
    the step loop, once in ``_sample`` at the same t), so every stateful PCC
    filter shares this one guard: the first call at a new timestamp consumes
    the real step, the repeat call at the same timestamp gets 0.0 and
    therefore re-reads an unchanged filter state.  The first call of a run
    also gets 0.0, which seeds each filter at its raw value instead of at
    zero (no spurious start-of-flight transient).
    """

    last = guidance_state.get(_PCC_FILTER_TIME_KEY)
    guidance_state[_PCC_FILTER_TIME_KEY] = float(time_s)
    if last is None:
        return 0.0
    return max(float(time_s) - float(last), 0.0)


def _first_order_blend(dt_s: float, tau_s: float) -> float:
    """Backward-Euler blend coefficient dt/(tau+dt); 0 for dt<=0, 1 for tau<=0."""

    if dt_s <= 0.0:
        return 0.0
    if tau_s <= 0.0:
        return 1.0
    return dt_s / (tau_s + dt_s)


def pcc_washout_fin_fraction(
    raw_fraction: tuple[float, float],
    dt_s: float,
    washout_time_s: float,
    guidance_state: dict[str, Any],
) -> tuple[float, float]:
    """Asymmetric first-order washout of the routed capture fin-fraction VECTOR.

    Growing commands follow ``PCC_WASHOUT_RISE_TIME_S`` (actuator scale, so the
    plateau is entered and steered exactly as in v0); shrinking commands follow
    ``washout_time_s``.  The raw vector is the alpha-inverted demand of
    whichever sub-mode is active (alpha_max*R along the PIP error direction in
    CAPTURE, HOMING's own PN+loft command through the same inversion), so it
    STEPS at the handoff while the lag does not -- that is what makes the
    routed fin command continuous there instead of dropping to zero.

    Filtering the VECTOR rather than a magnitude with a live direction is the
    load-bearing choice: once the velocity vector has swung onto the collision
    course the demand direction flips sign through zero, and a decaying
    magnitude re-aimed along the flipped direction slams the airframe the
    other way (measured: a 20 g -> 7 g -> 20 g limit cycle across the
    shoulder).  Equally, relaxing into HOMING's live demand rather than into a
    frozen vector is what keeps the residual from flying the airframe past the
    collision course (measured: heading error 1.6 deg -> 17 deg in 0.8 s, and a
    recapture that destroys the shot).
    """

    previous = guidance_state.get(_PCC_WASHOUT_FRACTION_KEY)
    raw = (float(raw_fraction[0]), float(raw_fraction[1]))
    if previous is None:
        guidance_state[_PCC_WASHOUT_FRACTION_KEY] = raw
        return raw
    # "Growing" means growing IN THE SAME SENSE.  A demand that has reversed
    # (the airframe has swung past the collision course and PN now wants the
    # other way) must come in on the slow side however large it is, or the lag
    # snaps through zero and rings.
    same_sense = raw[0] * previous[0] + raw[1] * previous[1] >= 0.0
    growing = same_sense and math.hypot(*raw) >= math.hypot(*previous)
    blend = _first_order_blend(dt_s, PCC_WASHOUT_RISE_TIME_S if growing else washout_time_s)
    value = (
        previous[0] + blend * (raw[0] - previous[0]),
        previous[1] + blend * (raw[1] - previous[1]),
    )
    guidance_state[_PCC_WASHOUT_FRACTION_KEY] = value
    return value


def pcc_washout_weight(
    raw_weight: float,
    dt_s: float,
    washout_time_s: float,
    guidance_state: dict[str, Any],
) -> float:
    """Asymmetric first-order washout of the direct-routing blend weight.

    The raw weight is v0's own: 1.0 in CAPTURE (direct routing owns the fin,
    control.py's tracking integrator engaged), 0.0 in HOMING.  Washing it out
    with the same constant as the fin fraction turns v0's step handover into a
    crossfade -- at the handoff instant the weight is still 1.0, so there is no
    jump in the routed fin command, and PID/PN authority (1-w) then grows back
    at exactly the rate the residual capture command fades.
    """

    previous = guidance_state.get(_PCC_WASHOUT_WEIGHT_KEY)
    if previous is None:
        guidance_state[_PCC_WASHOUT_WEIGHT_KEY] = float(raw_weight)
        return float(raw_weight)
    tau_s = PCC_WASHOUT_RISE_TIME_S if raw_weight >= previous else washout_time_s
    value = previous + _first_order_blend(dt_s, tau_s) * (float(raw_weight) - previous)
    guidance_state[_PCC_WASHOUT_WEIGHT_KEY] = value
    return value


PCC_POLAR_DIRECTION_TIME_S = 0.08
_PCC_POLAR_MAG_KEY = "pcc_polar_mag"
_PCC_POLAR_DIR_KEY = "pcc_polar_dir"


def pcc_polar_washout_fin_fraction(
    raw_fraction: tuple[float, float],
    direction_target: tuple[float, float],
    dt_s: float,
    washout_time_s: float,
    guidance_state: dict[str, Any],
) -> tuple[float, float]:
    """'Polar release': magnitude-slow / direction-fast washout (2026-08-26c review).

    The scalar washout (``pcc_washout_fin_fraction``) lags the routed VECTOR,
    so magnitude AND direction both decay at ``washout_time_s`` -- the memory
    stays pointed near the stale capture direction and injects an extra
    heading integral Delta-psi ~= a_r*tau_r/V (~15-20 deg), which PN must then
    unwind (the 4.5-6.5 s G hump the game never shows).  Here the two are
    separated:

      * MAGNITUDE keeps the identified slow release (asymmetric: rise at
        ``PCC_WASHOUT_RISE_TIME_S``, decay at ``washout_time_s``) toward the
        raw demand magnitude;
      * DIRECTION is a unit vector in fin (pitch,yaw) space slewed toward
        ``direction_target`` with the much faster
        ``PCC_POLAR_DIRECTION_TIME_S`` -- the physical statement being that
        aerodynamic load/alpha magnitude decays slowly while control
        allocation can rotate the load vector quickly.

    ``direction_target`` should be the CURRENT total guidance demand
    including the gravity-compensation vertical share (a >= 1 g floor), so
    the target direction never degenerates when PN's kinematic demand
    passes through zero; if it still does, the previous direction is held
    while the magnitude keeps decaying.
    """

    raw_mag = math.hypot(*raw_fraction)
    tgt_mag = math.hypot(*direction_target)
    prev_mag = guidance_state.get(_PCC_POLAR_MAG_KEY)
    prev_dir = guidance_state.get(_PCC_POLAR_DIR_KEY)
    if tgt_mag > 1e-9:
        tgt_unit = (direction_target[0] / tgt_mag, direction_target[1] / tgt_mag)
    elif prev_dir is not None:
        tgt_unit = prev_dir
    elif raw_mag > 1e-9:
        tgt_unit = (raw_fraction[0] / raw_mag, raw_fraction[1] / raw_mag)
    else:
        tgt_unit = (0.0, 0.0)
    if prev_mag is None or prev_dir is None:
        guidance_state[_PCC_POLAR_MAG_KEY] = raw_mag
        guidance_state[_PCC_POLAR_DIR_KEY] = tgt_unit
        return (raw_mag * tgt_unit[0], raw_mag * tgt_unit[1])
    growing = raw_mag >= prev_mag
    mag_blend = _first_order_blend(
        dt_s, PCC_WASHOUT_RISE_TIME_S if growing else washout_time_s
    )
    mag = prev_mag + mag_blend * (raw_mag - prev_mag)
    dir_blend = _first_order_blend(dt_s, PCC_POLAR_DIRECTION_TIME_S)
    d0 = prev_dir[0] + dir_blend * (tgt_unit[0] - prev_dir[0])
    d1 = prev_dir[1] + dir_blend * (tgt_unit[1] - prev_dir[1])
    d_mag = math.hypot(d0, d1)
    direction = (d0 / d_mag, d1 / d_mag) if d_mag > 1e-9 else tgt_unit
    guidance_state[_PCC_POLAR_MAG_KEY] = mag
    guidance_state[_PCC_POLAR_DIR_KEY] = direction
    return (mag * direction[0], mag * direction[1])


def pcc_filtered_los_rate(
    los_rate: Vector,
    dt_s: float,
    tau_s: float,
    guidance_state: dict[str, Any],
) -> Vector:
    """First-order low-pass of the LOS-rate VECTOR (seeker trackloop surrogate).

    ``ideal_truth`` observation hands PN a noiseless per-step LOS rate, which
    a real trackloop cannot follow; the unfiltered rate is what makes the
    modelled G wiggle where the game's is smooth.  Filtering the vector (not
    the finished acceleration) keeps the direction consistent with the
    magnitude, and leaves ``pn_acceleration`` itself a pure measurement.
    """

    previous = guidance_state.get(_PCC_LOS_RATE_KEY)
    if previous is None:
        seeded = (float(los_rate[0]), float(los_rate[1]), float(los_rate[2]))
        guidance_state[_PCC_LOS_RATE_KEY] = seeded
        return seeded
    blend = _first_order_blend(dt_s, tau_s)
    value = (
        previous[0] + blend * (float(los_rate[0]) - previous[0]),
        previous[1] + blend * (float(los_rate[1]) - previous[1]),
        previous[2] + blend * (float(los_rate[2]) - previous[2]),
    )
    guidance_state[_PCC_LOS_RATE_KEY] = value
    return value


def pcc_alpha_mode_update(
    time_s: float,
    epsilon_rad: float,
    capture_ratio: float,
    closing_mps: float,
    midcourse_cfg: dict[str, Any],
    guidance_state: dict[str, Any] | None,
) -> str:
    """CAPTURE/HOMING state machine for the PCC-alpha launch-capture candidate.

    Hysteresis (epsilon_enter < epsilon_exit) plus a short dwell keep the mode
    from chattering; the recapture guard additionally requires the capture law
    to actually WANT a large maneuver (capture_ratio >= recapture_r_min) so
    ordinary PN flight with a drifting collision-course reference (an
    accelerating missile moves the PIP) cannot re-trigger CAPTURE — see
    docs/PCC_ALPHA_V0.md.  ``guidance_state`` is a mutable per-run dict owned
    by the simulator loop; with ``None`` (legacy stateless callers) the mode
    degrades to a memoryless threshold on epsilon_exit.
    """

    epsilon_enter = deg_to_rad(float(midcourse_cfg.get("epsilon_enter_deg", 2.0)))
    epsilon_exit = deg_to_rad(float(midcourse_cfg.get("epsilon_exit_deg", 15.0)))
    handoff_r_max = float(midcourse_cfg.get("handoff_r_max", 0.9))
    recapture_r_min = float(midcourse_cfg.get("recapture_r_min", 1.0))
    dwell_s = float(midcourse_cfg.get("handoff_dwell_s", 0.1))
    if guidance_state is None:
        return PCC_CAPTURE if (epsilon_rad > epsilon_exit or capture_ratio >= 1.0) else PCC_HOMING
    mode = guidance_state.get("pcc_mode", PCC_CAPTURE)
    if mode == PCC_CAPTURE:
        condition = (
            capture_ratio < handoff_r_max
            and epsilon_rad < epsilon_enter
            and closing_mps > 0.0
        )
        key = "pcc_handoff_since_s"
    else:
        condition = epsilon_rad > epsilon_exit and capture_ratio >= recapture_r_min
        key = "pcc_recapture_since_s"
    if condition:
        since = guidance_state.setdefault(key, time_s)
        if time_s - since >= dwell_s:
            mode = PCC_HOMING if mode == PCC_CAPTURE else PCC_CAPTURE
            guidance_state.pop(key, None)
    else:
        guidance_state.pop(key, None)
    guidance_state["pcc_mode"] = mode
    return mode


def guidance_command(
    state: Any,
    target: TrackSolution | TargetState,
    time_s: float,
    config: dict[str, Any],
    enabled: bool,
    guidance_state: dict[str, Any] | None = None,
    plant_envelope_g: float | None = None,
) -> GuidanceOutput:
    """Build the existing PN/loft command from a copied track solution.

    ``TargetState`` remains accepted as a narrow compatibility shim for the
    older standalone geometry tests and legacy callers.  H2 always supplies a
    ``TrackSolution``; an invalid solution is an explicit zero-command state.
    """

    if isinstance(target, TargetState):
        track = TrackSolution(
            position=target.position,
            velocity=target.velocity,
            sample_time_s=time_s,
            solution_time_s=time_s,
            mode=TrackMode.IDEAL_TRUTH,
            valid=True,
            source="legacy_target_state",
        )
    elif isinstance(target, TrackSolution):
        track = target
    else:
        raise TypeError("guidance_command requires TrackSolution or TargetState")
    guidance_cfg = config["guidance"]
    relative_position = sub(track.position, state.position)
    relative_velocity = sub(track.velocity, state.velocity)
    range_m = norm(relative_position)

    if not track.valid:
        return GuidanceOutput(
            enabled=False,
            range_m=range_m,
            closing_speed_mps=0.0,
            los_rate_vector_rad_s=(0.0, 0.0, 0.0),
            pn_acceleration_mps2=(0.0, 0.0, 0.0),
            loft_acceleration_mps2=(0.0, 0.0, 0.0),
            commanded_acceleration_mps2=(0.0, 0.0, 0.0),
            commanded_body_acceleration_g=(0.0, 0.0),
            controller_specific_force_command_g=(0.0, 0.0),
            wind_normal_specific_force_command_g=(0.0, 0.0),
            gravity_compensation_wind_normal_g=(0.0, 0.0),
            wind_normal_pitch_axis=(0.0, 1.0, 0.0),
            wind_normal_yaw_axis=(0.0, 0.0, 1.0),
            effective_gain=0.0,
            time_to_go_s=0.0,
            loft_active=False,
            within_lock_range=False,
            midcourse_weight=0.0,
            heading_error_rad=0.0,
            pip_time_to_go_s=0.0,
            midcourse_fin_fraction=(0.0, 0.0),
        )
    pn_raw, closing, los_rate = pn_acceleration(
        relative_position,
        relative_velocity,
        state.velocity,
        guidance_cfg["pn_gain"],
    )
    flight_gain = interpolate_table(time_s, guidance_cfg["flight_time_gain_table"])
    time_to_go = estimate_time_to_go_s(range_m, closing, relative_velocity, config)
    hit_gain = interpolate_table(time_to_go, guidance_cfg["time_to_hit_gain_table"])
    effective_gain = flight_gain * hit_gain
    pn = scale(pn_raw, effective_gain)
    v_hat = normalize(state.velocity, fallback=body_axes_for_state(state).forward)
    loft_active = False
    loft = (0.0, 0.0, 0.0)
    loft_elevation_rad = 0.0
    if enabled and guidance_cfg.get("lofting_enabled", False):
        exit_distance = float(guidance_cfg.get("loft_exit_distance_m", guidance_cfg["lock_range_m"]))
        exit_tgo = float(guidance_cfg.get("loft_exit_time_to_go_s", 0.0))
        if range_m > exit_distance and time_to_go > exit_tgo:
            loft_active = True
            loft_elevation_rad = deg_to_rad(guidance_cfg["lofting_elevation_deg"])
            pitch_error = loft_elevation_rad - float(state.pitch)
            loft_g = guidance_cfg["angle_to_acceleration_multiplier"] * pitch_error
            gravity_mps2 = float(config["atmosphere"]["gravity_mps2"])
            # Profile loft: cap the implied pitch rate at omega_max.  Frozen
            # H1/H2 configs omit loft_omega_max_deg_s and keep the uncapped
            # vertical-G command.
            omega_max_deg_s = guidance_cfg.get("loft_omega_max_deg_s")
            if omega_max_deg_s is not None:
                speed = max(norm(state.velocity), 50.0)
                omega_cmd = loft_g * gravity_mps2 / speed
                omega_max = deg_to_rad(float(omega_max_deg_s))
                if abs(omega_cmd) > omega_max > 0.0:
                    loft_g = math.copysign(omega_max * speed / gravity_mps2, loft_g)
            loft = (0.0, g_to_mps2(loft_g, gravity_mps2), 0.0)
    midcourse_cfg = guidance_cfg.get("midcourse") or {}
    midcourse_accel, heading_error_rad, pip_time_to_go_s, midcourse_e_hat = midcourse_lead_turn_acceleration(
        relative_position,
        track.position,
        track.velocity,
        state.position,
        state.velocity,
        v_hat,
        midcourse_cfg,
        loft_active=loft_active,
        loft_elevation_rad=loft_elevation_rad,
    )
    midcourse_active = enabled and bool(midcourse_cfg.get("enabled", False))
    pcc_active = midcourse_active and str(midcourse_cfg.get("mode", "timer_blend")) == "pcc_alpha"
    max_accel = g_to_mps2(
        guidance_cfg["maximum_lateral_acceleration_g"],
        config["atmosphere"]["gravity_mps2"],
    )
    capture_mode = "off"
    capture_ratio = 0.0
    capture_routed_alpha_rad = 0.0
    capture_envelope_g = float(plant_envelope_g) if plant_envelope_g is not None else 0.0
    release_washout_time_s = 0.0
    pcc_filter_dt_s = 0.0
    if pcc_active:
        # PCC-alpha (docs/PCC_ALPHA_V0.md): CAPTURE steers the velocity vector
        # onto the predicted collision course with a_cap = V*sin(eps)/tau_c
        # along the same loft-aware e_hat the timer path used, explicitly
        # clipped to the alpha-limited achievable envelope; the alpha plateau
        # and its release (R_cap crossing 1) are emergent, not scheduled.  PN
        # is NOT summed in during CAPTURE; HOMING is the untouched PN(+loft)
        # path.  plant_envelope_g is trajectory-normal and thrust-inclusive,
        # matching the physical_normal_g feedback basis the PID tracks.
        # midcourse_weight starts at 0.0 (HOMING default) and is raised by the
        # alpha-routing block below, where control.py reads it as the
        # direct-fin-routing blend weight ((1-w)*pid): 1.0 on the v0 CAPTURE
        # plateau, and the washed-out alpha fraction once v0.1's
        # release_washout_time_s is active, so PID/PN authority grows back
        # exactly as the capture command fades.
        midcourse_weight = 0.0
        # v0.1 filters.  Both default off (0.0), in which case no state is
        # touched at all and the v0 path is reproduced bit-for-bit.
        release_washout_time_s = float(midcourse_cfg.get("release_washout_time_s", 0.0))
        los_rate_filter_time_constant_s = float(
            midcourse_cfg.get("los_rate_filter_time_constant_s", 0.0)
        )
        pcc_filters_live = guidance_state is not None and (
            release_washout_time_s > 0.0 or los_rate_filter_time_constant_s > 0.0
        )
        # One shared dt per timestamp for every PCC filter (see
        # pcc_filter_step_s: guidance_command runs twice per timestamp).
        pcc_filter_dt_s = (
            pcc_filter_step_s(time_s, guidance_state) if pcc_filters_live else 0.0
        )
        if los_rate_filter_time_constant_s > 0.0 and guidance_state is not None:
            pn = scale(
                pn_acceleration_from_los_rate(
                    pcc_filtered_los_rate(
                        los_rate,
                        pcc_filter_dt_s,
                        los_rate_filter_time_constant_s,
                        guidance_state,
                    ),
                    state.velocity,
                    closing,
                    guidance_cfg["pn_gain"],
                ),
                effective_gain,
            )
        speed = norm(state.velocity)
        tau_capture_s = max(float(midcourse_cfg.get("tau_capture_s", 0.3)), 1e-6)
        capture_accel_mag = speed * math.sin(clamp(heading_error_rad, 0.0, math.pi)) / tau_capture_s
        envelope_mps2 = (
            g_to_mps2(capture_envelope_g, config["atmosphere"]["gravity_mps2"])
            if capture_envelope_g > 0.0
            else None
        )
        capture_ratio = capture_accel_mag / envelope_mps2 if envelope_mps2 else 0.0
        capture_mode = pcc_alpha_mode_update(
            time_s, heading_error_rad, capture_ratio, closing, midcourse_cfg, guidance_state
        )
        if capture_mode == PCC_CAPTURE:
            capture_command = scale(midcourse_e_hat, capture_accel_mag)
            commanded = add_vectors(capture_command, loft)
            commanded = sub(commanded, scale(v_hat, dot(commanded, v_hat)))
            lateral_cap = min(envelope_mps2, max_accel) if envelope_mps2 else max_accel
            commanded = clamp_norm(commanded, lateral_cap)
        else:
            commanded = add_vectors(pn, loft)
            commanded = sub(commanded, scale(v_hat, dot(commanded, v_hat)))
            commanded = clamp_norm(commanded, max_accel)
    else:
        midcourse_weight = midcourse_blend_weight(time_s, midcourse_cfg)
        midcourse_command = scale(midcourse_accel, midcourse_weight) if midcourse_active else (0.0, 0.0, 0.0)
        commanded = add_vectors(add_vectors(pn, loft), midcourse_command)
        commanded = sub(commanded, scale(v_hat, dot(commanded, v_hat)))
        commanded = clamp_norm(commanded, max_accel)
    if not enabled:
        commanded = (0.0, 0.0, 0.0)
        pn = (0.0, 0.0, 0.0)
        loft = (0.0, 0.0, 0.0)
        loft_active = False
        heading_error_rad = 0.0
        pip_time_to_go_s = 0.0
        midcourse_weight = 0.0
    axes = body_axes_for_state(state)
    if pcc_active:
        # a_n^-1 step of PCC-alpha: the packed force law is linear in alpha,
        # so alpha_cmd = alpha_max * clip(R_cap, 0, 1), and this plant trims
        # at alpha = delta (I*omega_dot = K(delta-alpha) - C*omega), so the
        # fin-angle command that realizes alpha_cmd IS alpha_cmd.  Route it
        # through the existing direct fin path (weight=1 engages control.py's
        # tracking-mode integrator for bumpless handover); the accel-PID alone
        # is known to be far too sluggish to reach the launch-capture plateau
        # (the original v1.0.2 Sec.1 finding).  While saturated (R>=1) this
        # holds alpha_max; as R falls below 1 the commanded alpha releases
        # continuously -- the platform release is this fraction shrinking, not
        # a schedule.
        alpha_full_rad = deg_to_rad(float(midcourse_cfg["capture_alpha_max_deg"]))
        pitch_fin_limit_rad = math.radians(
            max(float(config["aerodynamics"]["horizontal_fin_aoa_limit_deg"]), 1e-9)
        )
        capture_now = capture_mode == PCC_CAPTURE
        if capture_now:
            demand_ratio = clamp(capture_ratio, 0.0, 1.0)
            demand_hat = midcourse_e_hat
        else:
            # v0.1: HOMING's own command run through the SAME a_n^-1 inversion,
            # so the washout below has something to relax INTO.  Unused when
            # release_washout_time_s == 0 (the v0 branch routes nothing here).
            demand_ratio = (
                clamp(norm(commanded) / envelope_mps2, 0.0, 1.0) if envelope_mps2 else 0.0
            )
            demand_hat = normalize(commanded, fallback=(0.0, 0.0, 0.0))
        fin_fraction_mag = clamp(
            alpha_full_rad * demand_ratio / pitch_fin_limit_rad, 0.0, 1.0
        )
        raw_fin_fraction = limit_unit_disk(
            dot(demand_hat, axes.up) * fin_fraction_mag,
            dot(demand_hat, axes.right) * fin_fraction_mag,
        )
        release_memory_mode = str(midcourse_cfg.get("release_memory_mode", "scalar"))
        if (
            release_washout_time_s > 0.0
            and guidance_state is not None
            and release_memory_mode == "polar"
        ):
            # 2026-08-26c 'polar release' ablation (docs/PCC_ALPHA_V0.md): the
            # magnitude memory is kept but the direction slews quickly to the
            # CURRENT total demand -- built from the specific-force vector
            # (kinematic command minus gravity), whose >= 1 g vertical share
            # keeps the target direction defined when PN's demand vanishes.
            gravity_ref = (0.0, -float(config["atmosphere"]["gravity_mps2"]), 0.0)
            sf_ref = sub(commanded, gravity_ref)
            direction_target = (dot(sf_ref, axes.up), dot(sf_ref, axes.right))
            lagged_fraction = pcc_polar_washout_fin_fraction(
                raw_fin_fraction,
                direction_target,
                pcc_filter_dt_s,
                release_washout_time_s,
                guidance_state,
            )
            midcourse_weight = pcc_washout_weight(
                1.0 if capture_now else 0.0,
                pcc_filter_dt_s,
                release_washout_time_s,
                guidance_state,
            )
            midcourse_fin_fraction = limit_unit_disk(
                midcourse_weight * lagged_fraction[0],
                midcourse_weight * lagged_fraction[1],
            )
        elif release_washout_time_s > 0.0 and guidance_state is not None:
            # v0.1 release washout (docs/PCC_ALPHA_V0.md).  v0 dropped the
            # routed command to zero the instant the state machine handed off,
            # which collapsed alpha in ~0.6 s; the replay's shoulder decays over
            # ~1.5 s and no PN-derived signal can supply it (the true lambda-dot
            # is ~0 there), so the capture channel keeps the memory.
            #
            # Two states, both with the same time constant:
            #   * the alpha-inverted demand itself is LAGGED, so the routed fin
            #     command is continuous through the handoff (the demand steps,
            #     the lag does not) and then relaxes into what PN is asking for
            #     -- not into a frozen vector.  Relaxing into PN is what keeps
            #     the residual from flying the airframe past the collision
            #     course: once the demand reverses, the lag follows it down.
            #   * the direct-routing weight crossfades 1 -> 0, and it multiplies
            #     the routed fraction.  So CAPTURE is exactly v0 (w=1, routed =
            #     alpha_max*R), the handoff instant is bumpless (w still 1),
            #     PID/PN authority (1-w) grows back at the same rate the routed
            #     command fades, and at w=0 the direct term is gone rather than
            #     double-counting PN's own feed-forward.
            # The handoff / recapture gates themselves are untouched.
            lagged_fraction = pcc_washout_fin_fraction(
                raw_fin_fraction, pcc_filter_dt_s, release_washout_time_s, guidance_state
            )
            midcourse_weight = pcc_washout_weight(
                1.0 if capture_now else 0.0,
                pcc_filter_dt_s,
                release_washout_time_s,
                guidance_state,
            )
            midcourse_fin_fraction = (
                midcourse_weight * lagged_fraction[0],
                midcourse_weight * lagged_fraction[1],
            )
        elif capture_now:
            midcourse_fin_fraction = raw_fin_fraction
            midcourse_weight = 1.0
        else:
            midcourse_fin_fraction = (0.0, 0.0)
        capture_routed_alpha_rad = (
            math.hypot(*midcourse_fin_fraction) * pitch_fin_limit_rad
        )
    elif midcourse_active and midcourse_weight > 0.0:
        # Direct IOG fin routing: same PIP lead-turn geometry as the
        # acceleration term above, projected onto the body pitch/yaw axes and
        # scaled by how far the heading error sits past fin_error_ref_rad.
        # Not itself disk-limited by reqAccelMax; it is a fin-travel fraction,
        # not a specific-force command.
        fin_error_ref_rad = float(midcourse_cfg.get("fin_error_ref_rad", 0.15))
        pitch_fin_direction = dot(midcourse_e_hat, axes.up)
        yaw_fin_direction = dot(midcourse_e_hat, axes.right)
        fin_error_magnitude = clamp(heading_error_rad / max(fin_error_ref_rad, 1e-9), 0.0, 1.0)
        midcourse_fin_fraction = limit_unit_disk(
            pitch_fin_direction * fin_error_magnitude * midcourse_weight,
            yaw_fin_direction * fin_error_magnitude * midcourse_weight,
        )
    else:
        midcourse_fin_fraction = (0.0, 0.0)
    gravity_mps2 = float(config["atmosphere"]["gravity_mps2"])
    gravity_vector = (0.0, -gravity_mps2, 0.0)
    body_pitch_g = mps2_to_g(dot(commanded, axes.up), gravity_mps2)
    body_yaw_g = mps2_to_g(dot(commanded, axes.right), gravity_mps2)
    # Required specific force keeps the gravity term, then is radially
    # clamped to reqAccelMax so a 38 g kinematic command plus 1 g hold
    # does not become a 39 g controller demand.
    raw_required = clamp_norm(sub(commanded, gravity_vector), max_accel)
    body_pitch_sf_g = mps2_to_g(dot(raw_required, axes.up), gravity_mps2)
    body_yaw_sf_g = mps2_to_g(dot(raw_required, axes.right), gravity_mps2)
    wind_basis = cg_wind_normal_basis(state, config)
    required_specific_force = clamp_norm(
        sub(
            raw_required,
            scale(wind_basis.forward, dot(raw_required, wind_basis.forward)),
        ),
        max_accel,
    )
    gravity_compensation = scale(gravity_vector, -1.0)
    gravity_compensation = sub(
        gravity_compensation,
        scale(wind_basis.forward, dot(gravity_compensation, wind_basis.forward)),
    )
    wind_command = (
        mps2_to_g(dot(required_specific_force, wind_basis.up), gravity_mps2),
        mps2_to_g(dot(required_specific_force, wind_basis.right), gravity_mps2),
    )
    gravity_command = (
        mps2_to_g(dot(gravity_compensation, wind_basis.up), gravity_mps2),
        mps2_to_g(dot(gravity_compensation, wind_basis.right), gravity_mps2),
    )
    plant_semantics = str(config["control"].get("plant_semantics", ""))
    candidate = plant_semantics in {
        "body_cm_tail_force_moment",
        "generalized_aero_moment",
    }
    if candidate:
        controller_command = wind_command
    elif plant_semantics == "fin_torque_body_aoa":
        controller_command = (body_pitch_sf_g, body_yaw_sf_g)
    else:
        # Frozen H1/H2 compatibility: kinematic body command, no gravity term.
        controller_command = (body_pitch_g, body_yaw_g)
    return GuidanceOutput(
        enabled=enabled,
        range_m=range_m,
        closing_speed_mps=closing,
        los_rate_vector_rad_s=los_rate,
        pn_acceleration_mps2=pn,
        loft_acceleration_mps2=loft,
        commanded_acceleration_mps2=commanded,
        commanded_body_acceleration_g=(body_pitch_g, body_yaw_g),
        controller_specific_force_command_g=controller_command,
        wind_normal_specific_force_command_g=wind_command,
        gravity_compensation_wind_normal_g=gravity_command,
        wind_normal_pitch_axis=wind_basis.up,
        wind_normal_yaw_axis=wind_basis.right,
        effective_gain=effective_gain,
        time_to_go_s=time_to_go,
        loft_active=loft_active,
        within_lock_range=range_m <= guidance_cfg["lock_range_m"],
        midcourse_weight=midcourse_weight,
        heading_error_rad=heading_error_rad,
        pip_time_to_go_s=pip_time_to_go_s,
        midcourse_fin_fraction=midcourse_fin_fraction,
        capture_mode=capture_mode,
        capture_ratio_r=capture_ratio,
        capture_envelope_g=capture_envelope_g,
        capture_routed_alpha_deg=math.degrees(capture_routed_alpha_rad),
    )


def add_vectors(a: Vector, b: Vector) -> Vector:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])
