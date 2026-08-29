# ==== VERBATIM EXCERPT ====================================================
# source : src/aim120_model/guidance.py
# lines  : 188-541   (1-based, inclusive; original line numbers)
# note   : 制导路由（前半）：lead-turn 几何、PCC 滤波器、CAPTURE/HOMING 状态机
# repo   : working tree at 2026-08-26, unmodified
# ==========================================================================

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


