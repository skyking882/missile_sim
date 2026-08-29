# ==== VERBATIM EXCERPT ====================================================
# source : src/aim120_model/guidance.py
# lines  : 542-937   (1-based, inclusive; original line numbers)
# note   : 制导路由（后半）：guidance_command()，含 a_n^-1 反演与 cos-alpha 投影（770-776 行）
# repo   : working tree at 2026-08-26, unmodified
# ==========================================================================

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


