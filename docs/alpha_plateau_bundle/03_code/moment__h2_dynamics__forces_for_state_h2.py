# ==== VERBATIM EXCERPT ====================================================
# source : src/aim120_model/h2_dynamics.py
# lines  : 374-1324   (1-based, inclusive; original line numbers)
# note   : 力矩路径：完整的 forces_for_state_h2()。fin_torque_body_aoa 走的活分支见 INDEX.md
# repo   : working tree at 2026-08-26, unmodified
# ==========================================================================

def forces_for_state_h2(
    state: SimState,
    time_s: float,
    config: dict[str, Any],
    propulsion: PiecewisePropulsion,
    powered: bool,
) -> H2DynamicsDiagnostics:
    atmosphere = StandardAtmosphere()
    prop = propulsion.sample(time_s, powered=powered)
    mass = max(float(state.mass), 1e-6)
    plant_semantics = str(config["control"].get("plant_semantics", "direct_fin_g"))
    legacy_fin_torque_plant = plant_semantics == "fin_torque_body_aoa"
    body_cm_tail_plant = plant_semantics == "body_cm_tail_force_moment"
    generalized_aero_plant = plant_semantics == "generalized_aero_moment"
    if plant_semantics not in {
        "direct_fin_g",
        "fin_torque_body_aoa",
        "body_cm_tail_force_moment",
        "generalized_aero_moment",
    }:
        raise ValueError(f"unknown plant_semantics: {plant_semantics}")
    axes = body_axes_for_state(state)
    normal_force_velocity = None
    if body_cm_tail_plant:
        body_candidate = config["aerodynamics"]["body_cp_force_candidate"]
        body_station_x_m = (
            float(body_candidate["cp_cg_arm_over_diameter"])
            * float(config["geometry"]["caliber_m"])
        )
        normal_force_velocity = _station_velocity_inertial(state, axes, body_station_x_m)
    aero = compute_aerodynamics_h2(
        state.position[1],
        state.velocity,
        state.pitch,
        state.yaw,
        config,
        atmosphere,
        axes_override=axes,
        normal_force_velocity_mps=normal_force_velocity,
    )
    cx_fin = float(config["aerodynamics"].get("cx_vs_fin_delta", 0.0))
    fin_drag_force = (0.0, 0.0, 0.0)
    if cx_fin > 0.0:
        delta_sq = (
            float(state.actual_pitch_fin_angle_rad) ** 2
            + float(state.actual_yaw_fin_angle_rad) ** 2
        )
        cda_fin = area_basis(config) * cx_fin * delta_sq
        fin_drag_force = scale(aero.air_velocity_hat, -aero.dynamic_pressure_pa * cda_fin)
    # The generalized candidate deliberately resolves normal force and pitch/
    # yaw moment as separate nondimensional outputs.  It has no force station:
    # the shared inputs are CG flow alpha, actual fin angle, body rate, and the
    # local alpha-rate proxy.  This block is inactive for legacy and v1 body-Cm
    # candidates, preserving their existing equations point-for-point.
    generalized_body_force_vector = (0.0, 0.0, 0.0)
    generalized_delta_force_vector = (0.0, 0.0, 0.0)
    generalized_pitch_body_static_moment = 0.0
    generalized_yaw_body_static_moment = 0.0
    generalized_pitch_body_rate_moment = 0.0
    generalized_yaw_body_rate_moment = 0.0
    generalized_pitch_delta_moment = 0.0
    generalized_yaw_delta_moment = 0.0
    generalized_pitch_alpha_force = 0.0
    generalized_yaw_alpha_force = 0.0
    generalized_pitch_delta_force = 0.0
    generalized_yaw_delta_force = 0.0
    generalized_pitch_alpha_force_slope = 0.0
    generalized_yaw_alpha_force_slope = 0.0
    generalized_pitch_delta_force_slope = 0.0
    generalized_yaw_delta_force_slope = 0.0
    generalized_pitch_alpha_dot_hat = 0.0
    generalized_yaw_alpha_dot_hat = 0.0
    generalized_cm_alpha_dot_per_rad = 0.0
    generalized_cm_alpha_dot_runtime_enabled = False
    generalized_pitch_cm_rate = 0.0
    generalized_yaw_cm_rate = 0.0
    generalized_reference_area_m2 = 0.0
    generalized_reference_length_m = 0.0
    if generalized_aero_plant:
        generalized = config["aerodynamics"].get("generalized_aero_moment_candidate")
        if not isinstance(generalized, dict):
            raise ValueError(
                "generalized_aero_moment requires generalized_aero_moment_candidate"
            )
        generalized_reference_area_m2 = caliber_circular_area(config)
        generalized_reference_length_m = float(config["geometry"]["caliber_m"])
        speed_for_generalized = max(float(aero.speed_mps), 1e-6)
        q_dyn = float(aero.normal_force_dynamic_pressure_pa)
        q_hat_pitch = state.pitch_rate * generalized_reference_length_m / (2.0 * speed_for_generalized)
        q_hat_yaw = state.yaw_rate * generalized_reference_length_m / (2.0 * speed_for_generalized)
        # The current reduced state has body rate but no independently
        # integrated alpha-dot measurement.  Do not alias alpha-dot to q:
        # that would make Cm_q and Cm_alpha-dot structurally collinear.
        generalized_pitch_alpha_dot_hat = 0.0
        generalized_yaw_alpha_dot_hat = 0.0
        pitch_alpha = float(aero.pitch_alpha_rad)
        yaw_alpha = float(aero.yaw_alpha_rad)
        pitch_delta = float(state.actual_pitch_fin_angle_rad)
        yaw_delta = float(state.actual_yaw_fin_angle_rad)
        cn_alpha = float(generalized["cn_alpha_per_rad"])
        cn_delta = float(generalized["cn_delta_per_rad"])
        cm_alpha = float(generalized["cm_alpha_per_rad"])
        cm_delta = float(generalized["cm_delta_per_rad"])
        cm_q = float(generalized["cm_q"])
        generalized_pitch_alpha_force_slope = q_dyn * generalized_reference_area_m2 * cn_alpha
        generalized_yaw_alpha_force_slope = generalized_pitch_alpha_force_slope
        generalized_pitch_delta_force_slope = q_dyn * generalized_reference_area_m2 * cn_delta
        generalized_yaw_delta_force_slope = generalized_pitch_delta_force_slope
        generalized_pitch_alpha_force = generalized_pitch_alpha_force_slope * pitch_alpha
        generalized_yaw_alpha_force = generalized_yaw_alpha_force_slope * yaw_alpha
        generalized_pitch_delta_force = generalized_pitch_delta_force_slope * pitch_delta
        generalized_yaw_delta_force = generalized_yaw_delta_force_slope * yaw_delta
        generalized_body_force_vector = _add_many(
            scale(aero.flow_normal_pitch, generalized_pitch_alpha_force),
            scale(aero.flow_normal_yaw, generalized_yaw_alpha_force),
        )
        generalized_delta_force_vector = _add_many(
            scale(aero.flow_normal_pitch, generalized_pitch_delta_force),
            scale(aero.flow_normal_yaw, generalized_yaw_delta_force),
        )
        generalized_pitch_body_static_moment = (
            q_dyn * generalized_reference_area_m2 * generalized_reference_length_m
            * cm_alpha * pitch_alpha
        )
        generalized_yaw_body_static_moment = (
            q_dyn * generalized_reference_area_m2 * generalized_reference_length_m
            * cm_alpha * yaw_alpha
        )
        generalized_pitch_cm_rate = cm_q * q_hat_pitch
        generalized_yaw_cm_rate = cm_q * q_hat_yaw
        generalized_pitch_body_rate_moment = (
            q_dyn * generalized_reference_area_m2 * generalized_reference_length_m
            * generalized_pitch_cm_rate
        )
        generalized_yaw_body_rate_moment = (
            q_dyn * generalized_reference_area_m2 * generalized_reference_length_m
            * generalized_yaw_cm_rate
        )
        generalized_pitch_delta_moment = (
            q_dyn * generalized_reference_area_m2 * generalized_reference_length_m
            * cm_delta * pitch_delta
        )
        generalized_yaw_delta_moment = (
            q_dyn * generalized_reference_area_m2 * generalized_reference_length_m
            * cm_delta * yaw_delta
        )
    body_normal_force_vector = (
        generalized_body_force_vector
        if generalized_aero_plant
        else aero.natural_lift_force_n
    )
    gravity = float(config["atmosphere"]["gravity_mps2"])
    thrust_force = scale(axes.forward, prop.thrust_n)
    arm = float(config["aerodynamics"]["distance_cm_to_stabilizer_m"])
    pitch_body_aoa_force_g = (
        dot(body_normal_force_vector, aero.flow_normal_pitch) / (gravity * mass)
    )
    yaw_body_aoa_force_g = (
        dot(body_normal_force_vector, aero.flow_normal_yaw) / (gravity * mass)
    )
    if legacy_fin_torque_plant and config["aerodynamics"].get("path_g_from_alpha"):
        # L_total is finsLatAccel*(q/q_base)*(alpha/finsAoa)*m*g.  Do not also
        # add the CN_alpha body force or alpha is counted twice.
        body_normal_force_vector = (0.0, 0.0, 0.0)
        pitch_body_aoa_force_g = 0.0
        yaw_body_aoa_force_g = 0.0
    fixed_lifting_surface_multiplier = 0.0
    body_normal_force_area_slope = 0.0
    fixed_lifting_surface_area_slope = 0.0
    fixed_lifting_surface_station_x = 0.0
    pitch_fixed_lifting_surface_alpha = 0.0
    yaw_fixed_lifting_surface_alpha = 0.0
    pitch_fixed_lifting_surface_force = 0.0
    yaw_fixed_lifting_surface_force = 0.0
    pitch_fixed_lifting_surface_moment = 0.0
    yaw_fixed_lifting_surface_moment = 0.0
    fixed_lifting_surface_force_vector = (0.0, 0.0, 0.0)
    cg_flow_normal_pitch = aero.flow_normal_pitch
    cg_flow_normal_yaw = aero.flow_normal_yaw
    if body_cm_tail_plant:
        fixed_lift = config["aerodynamics"].get("fixed_lifting_surface_candidate")
    else:
        fixed_lift = None
    if isinstance(fixed_lift, dict):
        fixed_lifting_surface_multiplier = float(
            fixed_lift["fixed_lifting_surface_multiplier"]
        )
        body_normal_force_area_slope = float(
            fixed_lift["body_normal_force_area_slope_m2_per_rad"]
        )
        fixed_lifting_surface_area_slope = (
            fixed_lifting_surface_multiplier * body_normal_force_area_slope
        )
        fixed_lifting_surface_station_x = float(fixed_lift["station_x_m"])
        if abs(fixed_lifting_surface_station_x) <= 1e-15:
            # Preserve the prior near-CG diagnostic exactly: x_W=0 used the
            # CG station flow and the common CG dynamic pressure.  Only a
            # non-zero station introduces local-q/alpha changes.
            (
                fixed_lifting_surface_speed,
                pitch_fixed_lifting_surface_alpha,
                yaw_fixed_lifting_surface_alpha,
                fixed_flow_normal_pitch,
                fixed_flow_normal_yaw,
            ) = _flow_at_station(state, axes, 0.0, config)
            fixed_lifting_surface_dynamic_pressure = aero.dynamic_pressure_pa
        else:
            (
                fixed_lifting_surface_speed,
                pitch_fixed_lifting_surface_alpha,
                yaw_fixed_lifting_surface_alpha,
                fixed_flow_normal_pitch,
                fixed_flow_normal_yaw,
            ) = _flow_at_station(
                state, axes, fixed_lifting_surface_station_x, config
            )
            fixed_lifting_surface_dynamic_pressure = (
                0.5
                * aero.atmosphere.density_kg_m3
                * fixed_lifting_surface_speed
                * fixed_lifting_surface_speed
            )
        pitch_fixed_lifting_surface_force = (
            fixed_lifting_surface_dynamic_pressure
            * fixed_lifting_surface_area_slope
            * pitch_fixed_lifting_surface_alpha
        )
        yaw_fixed_lifting_surface_force = (
            fixed_lifting_surface_dynamic_pressure
            * fixed_lifting_surface_area_slope
            * yaw_fixed_lifting_surface_alpha
        )
        fixed_lifting_surface_force_vector = _add_many(
            scale(fixed_flow_normal_pitch, pitch_fixed_lifting_surface_force),
            scale(fixed_flow_normal_yaw, yaw_fixed_lifting_surface_force),
        )
        fixed_lifting_surface_arm_vector = scale(
            axes.forward, fixed_lifting_surface_station_x
        )
        fixed_lifting_surface_moment_vector = cross(
            fixed_lifting_surface_arm_vector, fixed_lifting_surface_force_vector
        )
        pitch_fixed_lifting_surface_moment = dot(
            fixed_lifting_surface_moment_vector, axes.right
        )
        yaw_fixed_lifting_surface_moment = -dot(
            fixed_lifting_surface_moment_vector, axes.up
        )
    pitch_fin_moment_equivalent_g = state.actual_pitch_acceleration_g
    yaw_fin_moment_equivalent_g = state.actual_yaw_acceleration_g
    pitch_fin_translation_equivalent_g = state.actual_pitch_acceleration_g
    yaw_fin_translation_equivalent_g = state.actual_yaw_acceleration_g
    pitch_tail_rate_incidence = 0.0
    yaw_tail_rate_incidence = 0.0
    pitch_tail_effective_incidence = 0.0
    yaw_tail_effective_incidence = 0.0
    scheduled_fins_g = 0.0
    pitch_fin_limit = 1.0
    yaw_fin_limit = 1.0
    tail_station_x_m = arm
    pitch_tail_force_n = 0.0
    yaw_tail_force_n = 0.0
    pitch_tail_moment_nm = 0.0
    yaw_tail_moment_nm = 0.0
    pitch_tail_force_vector = (0.0, 0.0, 0.0)
    yaw_tail_force_vector = (0.0, 0.0, 0.0)
    pitch_moment_fraction = 0.0
    yaw_moment_fraction = 0.0
    tail_alpha_force_multiplier = 0.0
    tail_delta_force_multiplier = 0.0
    pitch_tail_alpha_force_slope = 0.0
    yaw_tail_alpha_force_slope = 0.0
    pitch_tail_delta_force_slope = 0.0
    yaw_tail_delta_force_slope = 0.0
    pitch_tail_alpha_force = 0.0
    yaw_tail_alpha_force = 0.0
    pitch_tail_delta_force = 0.0
    yaw_tail_delta_force = 0.0
    pitch_tail_net_force_pre_cap = 0.0
    yaw_tail_net_force_pre_cap = 0.0
    tail_force_cap = 0.0
    tail_force_cap_scale = 1.0
    tail_force_cap_active = False
    pitch_tail_alpha_moment = 0.0
    yaw_tail_alpha_moment = 0.0
    pitch_tail_delta_moment = 0.0
    yaw_tail_delta_moment = 0.0
    speed_for_rate = max(aero.speed_mps, 1e-6)
    tail_flow_normal_pitch = aero.flow_normal_pitch
    tail_flow_normal_yaw = aero.flow_normal_yaw
    if legacy_fin_torque_plant or body_cm_tail_plant:
        fins_g = float(config["aerodynamics"]["fins_lateral_acceleration_g"])
        speed_schedule = base_indicated_speed_schedule(aero.dynamic_pressure_pa, config)
        scheduled_fins_g = fins_g * speed_schedule.fin_force_scale
        scheduled_fins_g_force = scheduled_fins_g
        if legacy_fin_torque_plant:
            # 2026-08 three-flight joint replay fit (PL-12 fast/slow launch +
            # R-77, 59 frames total; T*sin(alpha)/(m*g) removed from the
            # displayed G before fitting): the G/alpha slope is a pure line
            # through the origin in eta_q, R^2=0.97-0.998, with no elbow and
            # no saturation across eta_q in [0.55, 1.7].  k =
            # packed_lift_slope_scale * finsLatAccel/finsAoa, where
            # packed_lift_slope_scale = 0.58 is the fitted 1.08-1.17
            # g/deg/eta slope divided by each missile's A/alpha_max (which
            # cluster at 0.56-0.59).  scheduled_fins_g feeds only the moment
            # channel below (pitch/yaw_fin_moment_equivalent_g and the
            # pitch/yaw_stiffness_s2 spring): the dataset behind
            # packed_lift_force_eta_law identifies the force law only, so the
            # moment keeps this constant fleet-wide slope.
            packed_lift_slope_scale = float(config["aerodynamics"].get("packed_lift_slope_scale", 1.0))
            scheduled_fins_g *= packed_lift_slope_scale
            # 2026-08-24 46-frame R-77-1 level-shot replay fit: unlike the
            # moment slope above, the FORCE slope (alpha,M,q -> G) is not
            # fleet-constant -- per-frame k rose 0.45-0.74 over eta 0.35-2.62.
            # packed_lift_force_eta_law is an optional per-missile override
            # (absent -> force_k=packed_lift_slope_scale, bit-identical to
            # before); eta is the same q/q_base ratio fin_force_scale already
            # carries, and the clamp applies only inside this force_k
            # evaluation, not to the linear eta factor below.
            eta_law = config["aerodynamics"].get("packed_lift_force_eta_law")
            # EXPERIMENTAL (cn20 closed-loop exam, scripts/cn20_closed_loop.py):
            # packed_lift_fixed_cn replaces the FORCE channel's slope outright
            # with a fixed-coefficient standard-aero force law instead of the
            # packed empirical eta-law/slope-scale above.  Per radian of alpha
            # the aero lateral load is cn_alpha_per_rad*q*S_d/(m*g) (S_d =
            # caliber circular area, q = true dynamic pressure, m = current
            # mass); the alpha-disk clamp below is untouched, so the force
            # plateau becomes cn_alpha_per_rad*q*S_d*alpha_max/(m*g).  Absent
            # key -> scheduled_fins_g_force is computed exactly as before
            # (bit-identical).  The moment channel (scheduled_fins_g above) is
            # not affected by this key at all.
            fixed_cn = config["aerodynamics"].get("packed_lift_fixed_cn")
            if fixed_cn is not None:
                if eta_law:
                    warnings.warn(
                        "aerodynamics.packed_lift_fixed_cn and packed_lift_force_eta_law "
                        "are both present; packed_lift_fixed_cn takes precedence for the "
                        "force channel and packed_lift_force_eta_law is ignored there",
                        stacklevel=2,
                    )
                alpha_max_rad_for_cn = math.radians(
                    max(float(config["aerodynamics"]["horizontal_fin_aoa_limit_deg"]), 1e-9)
                )
                cn_alpha_per_rad = float(fixed_cn["cn_alpha_per_rad"])
                s_d = caliber_circular_area(config)
                cn_slope_g_per_rad = (
                    cn_alpha_per_rad * aero.dynamic_pressure_pa * s_d / (mass * gravity)
                )
                scheduled_fins_g_force = cn_slope_g_per_rad * alpha_max_rad_for_cn
            elif eta_law:
                eta_lo, eta_hi = eta_law["eta_clamp"]
                clamped_eta = min(
                    max(speed_schedule.dynamic_pressure_ratio, float(eta_lo)), float(eta_hi)
                )
                force_k = float(eta_law["coefficient"]) * clamped_eta ** float(eta_law["eta_exponent"])
                scheduled_fins_g_force = fins_g * speed_schedule.fin_force_scale * force_k
            else:
                force_k = packed_lift_slope_scale
                scheduled_fins_g_force = fins_g * speed_schedule.fin_force_scale * force_k
        if body_cm_tail_plant:
            pitch_fin_limit = float(config["control"]["fin_actuator_travel"]["pitch_limit_rad"])
            yaw_fin_limit = float(config["control"]["fin_actuator_travel"]["yaw_limit_rad"])
        else:
            pitch_fin_limit = math.radians(
                max(float(config["aerodynamics"]["horizontal_fin_aoa_limit_deg"]), 1e-9)
            )
            yaw_fin_limit = math.radians(
                max(float(config["aerodynamics"]["vertical_fin_aoa_limit_deg"]), 1e-9)
            )

        if body_cm_tail_plant:
            tail_station_x_m = float(config["aerodynamics"]["tail_station_x_m"])
            (
                _tail_speed,
                tail_pitch_alpha,
                tail_yaw_alpha,
                tail_flow_normal_pitch,
                tail_flow_normal_yaw,
            ) = _flow_at_station(
                state,
                axes,
                tail_station_x_m,
                config,
            )
            (
                _cg_speed,
                cg_pitch_alpha,
                cg_yaw_alpha,
                _cg_pitch_normal,
                _cg_yaw_normal,
            ) = _flow_at_station(
                state,
                axes,
                0.0,
                config,
            )
            pitch_tail_rate_incidence = tail_pitch_alpha - cg_pitch_alpha
            yaw_tail_rate_incidence = tail_yaw_alpha - cg_yaw_alpha
            pitch_tail_effective_incidence = state.actual_pitch_fin_angle_rad - tail_pitch_alpha
            yaw_tail_effective_incidence = state.actual_yaw_fin_angle_rad - tail_yaw_alpha
            empirical_authority = config["aerodynamics"]["empirical_fin_authority"]
            pitch_authority_reference = float(empirical_authority["pitch_incidence_reference_rad"])
            yaw_authority_reference = float(empirical_authority["yaw_incidence_reference_rad"])
        else:
            # Spec §5: I ω̇ = K(δ-α)-Cω.  Rate incidence ω·Δ/V is a diagnostic
            # only; it does not enter the weathervane spring.
            pitch_tail_rate_incidence = state.pitch_rate * arm / speed_for_rate
            yaw_tail_rate_incidence = state.yaw_rate * arm / speed_for_rate
            pitch_tail_effective_incidence = (
                state.actual_pitch_fin_angle_rad - aero.pitch_alpha_rad
            )
            yaw_tail_effective_incidence = (
                state.actual_yaw_fin_angle_rad - aero.yaw_alpha_rad
            )
            pitch_authority_reference = pitch_fin_limit
            yaw_authority_reference = yaw_fin_limit
        if body_cm_tail_plant:
            split_config = config["aerodynamics"]["split_tail_candidate"]
            tail_alpha_force_multiplier = float(
                split_config["tail_alpha_force_multiplier"]
            )
            tail_delta_force_multiplier = float(
                split_config["tail_delta_force_multiplier"]
            )
            split_tail = split_empirical_tail_force(
                mass_kg=mass,
                gravity_mps2=gravity,
                dynamic_pressure_pa=aero.dynamic_pressure_pa,
                q_base_pa=float(split_config["q_base_pa"]),
                acceleration_authority_g=fins_g,
                pitch_alpha_rad=tail_pitch_alpha,
                yaw_alpha_rad=tail_yaw_alpha,
                pitch_delta_rad=state.actual_pitch_fin_angle_rad,
                yaw_delta_rad=state.actual_yaw_fin_angle_rad,
                pitch_authority_reference_rad=pitch_authority_reference,
                yaw_authority_reference_rad=yaw_authority_reference,
                tail_alpha_force_multiplier=tail_alpha_force_multiplier,
                tail_delta_force_multiplier=tail_delta_force_multiplier,
            )
            pitch_tail_alpha_force_slope = split_tail.pitch_alpha_slope_n_per_rad
            yaw_tail_alpha_force_slope = split_tail.yaw_alpha_slope_n_per_rad
            pitch_tail_delta_force_slope = split_tail.pitch_delta_slope_n_per_rad
            yaw_tail_delta_force_slope = split_tail.yaw_delta_slope_n_per_rad
            pitch_tail_alpha_force = split_tail.pitch_alpha_force_n
            yaw_tail_alpha_force = split_tail.yaw_alpha_force_n
            pitch_tail_delta_force = split_tail.pitch_delta_force_n
            yaw_tail_delta_force = split_tail.yaw_delta_force_n
            pitch_tail_net_force_pre_cap = split_tail.pitch_pre_cap_force_n
            yaw_tail_net_force_pre_cap = split_tail.yaw_pre_cap_force_n
            tail_force_cap = split_tail.force_cap_n
            tail_force_cap_scale = split_tail.cap_scale
            tail_force_cap_active = split_tail.cap_active
            pitch_tail_force_n = split_tail.pitch_force_n
            yaw_tail_force_n = split_tail.yaw_force_n
            # Compatibility authority fraction retains the old chi=(delta-alpha)
            # sign, while the split force fields expose the physical force sign.
            if tail_force_cap > 0.0:
                pitch_moment_fraction = -pitch_tail_force_n / tail_force_cap
                yaw_moment_fraction = -yaw_tail_force_n / tail_force_cap
            # Positive command must turn the nose positive.  At an aft station
            # that requires a negative physical tail force; the same force then
            # supplies M_tail=N_tail*x_t with no same-sign G shortcut.
            pitch_tail_force_vector = scale(tail_flow_normal_pitch, pitch_tail_force_n)
            yaw_tail_force_vector = scale(tail_flow_normal_yaw, yaw_tail_force_n)
            tail_arm_vector = scale(axes.forward, tail_station_x_m)
            pitch_tail_alpha_moment = dot(
                cross(tail_arm_vector, scale(tail_flow_normal_pitch, pitch_tail_alpha_force)),
                axes.right,
            )
            yaw_tail_alpha_moment = -dot(
                cross(tail_arm_vector, scale(tail_flow_normal_yaw, yaw_tail_alpha_force)),
                axes.up,
            )
            pitch_tail_delta_moment = dot(
                cross(tail_arm_vector, scale(tail_flow_normal_pitch, pitch_tail_delta_force)),
                axes.right,
            )
            yaw_tail_delta_moment = -dot(
                cross(tail_arm_vector, scale(tail_flow_normal_yaw, yaw_tail_delta_force)),
                axes.up,
            )
            pitch_tail_moment_vector = cross(tail_arm_vector, pitch_tail_force_vector)
            yaw_tail_moment_vector = cross(tail_arm_vector, yaw_tail_force_vector)
            pitch_tail_moment_nm = dot(pitch_tail_moment_vector, axes.right)
            yaw_tail_moment_nm = -dot(yaw_tail_moment_vector, axes.up)
            moment_scale = gravity * mass * abs(tail_station_x_m)
            pitch_fin_moment_equivalent_g = pitch_tail_moment_nm / moment_scale
            yaw_fin_moment_equivalent_g = yaw_tail_moment_nm / moment_scale
        else:
            pitch_moment_fraction, yaw_moment_fraction = _limit_unit_disk(
                pitch_tail_effective_incidence / pitch_authority_reference,
                yaw_tail_effective_incidence / yaw_authority_reference,
            )
            pitch_delta_fraction, yaw_delta_fraction = _limit_unit_disk(
                state.actual_pitch_fin_angle_rad / pitch_authority_reference,
                state.actual_yaw_fin_angle_rad / yaw_authority_reference,
            )
            pitch_alpha_fraction, yaw_alpha_fraction = _limit_unit_disk(
                aero.pitch_alpha_rad / pitch_authority_reference,
                aero.yaw_alpha_rad / yaw_authority_reference,
            )
            # Moments stay on K(δ-α) using scheduled_fins_g, the constant
            # fleet-wide slope.  Path G uses scheduled_fins_g_force, which
            # may carry a per-missile packed_lift_force_eta_law instead of
            # the constant slope, so total lift follows angle of attack, not
            # fin deflection or arm*length.  loadFactorMax later radially
            # caps F_N only; this moment channel is left unscaled.
            pitch_fin_moment_equivalent_g = scheduled_fins_g * pitch_moment_fraction
            yaw_fin_moment_equivalent_g = scheduled_fins_g * yaw_moment_fraction
            if config["aerodynamics"].get("path_g_from_alpha"):
                pitch_fin_translation_equivalent_g = (
                    scheduled_fins_g_force * pitch_alpha_fraction
                )
                yaw_fin_translation_equivalent_g = (
                    scheduled_fins_g_force * yaw_alpha_fraction
                )
            else:
                share = float(config["aerodynamics"].get("fin_translation_share", 1.0))
                path_g_scale = share
                if config["aerodynamics"].get("path_g_scales_with_arm_times_length"):
                    path_g_scale *= arm * max(float(config["geometry"]["length_m"]), 1e-9)
                pitch_fin_translation_equivalent_g = (
                    scheduled_fins_g_force * pitch_delta_fraction * path_g_scale
                )
                yaw_fin_translation_equivalent_g = (
                    scheduled_fins_g_force * yaw_delta_fraction * path_g_scale
                )
            pitch_tail_force_n = pitch_fin_translation_equivalent_g * gravity * mass
            yaw_tail_force_n = yaw_fin_translation_equivalent_g * gravity * mass
            pitch_tail_moment_nm = pitch_fin_moment_equivalent_g * gravity * mass * arm
            yaw_tail_moment_nm = yaw_fin_moment_equivalent_g * gravity * mass * arm

    if legacy_fin_torque_plant:
        control_force = _add_many(
            scale(aero.flow_normal_pitch, pitch_fin_translation_equivalent_g * gravity * mass),
            scale(aero.flow_normal_yaw, yaw_fin_translation_equivalent_g * gravity * mass),
        )
    elif body_cm_tail_plant:
        control_force = _add_many(pitch_tail_force_vector, yaw_tail_force_vector)
    elif generalized_aero_plant:
        control_force = generalized_delta_force_vector
    else:
        control_force = _add_many(
            scale(
                aero.flow_normal_pitch,
                state.actual_pitch_acceleration_g * gravity * mass,
            ),
            scale(
                aero.flow_normal_yaw,
                state.actual_yaw_acceleration_g * gravity * mass,
            ),
        )
    if legacy_fin_torque_plant:
        # Spec §4: s_cap = min(1, n_max g / |F_N/m|) applies to packed lift
        # only.  Drag along -v̂ and thrust along f̂ are not scaled.
        load_factor_max_g = config.get("performance", {}).get("load_factor_max_g")
        if load_factor_max_g is None:
            load_factor_max_g = config.get("guidance", {}).get("maximum_lateral_acceleration_g")
        if load_factor_max_g is not None:
            cap = max(float(load_factor_max_g), 0.0)
            if cap > 0.0:
                packed_lift_n = norm(control_force)
                cap_n = cap * gravity * mass
                if packed_lift_n > cap_n:
                    load_scale = cap_n / packed_lift_n
                    control_force = scale(control_force, load_scale)
                    pitch_fin_translation_equivalent_g *= load_scale
                    yaw_fin_translation_equivalent_g *= load_scale
                    pitch_tail_force_n *= load_scale
                    yaw_tail_force_n *= load_scale
    # EXPERIMENTAL (cn20 closed-loop exam, scripts/cn20_closed_loop.py):
    # induced_drag_mode="momentum_tilt" zeroes the shipped cx_vs_aoa alpha^2
    # induced-drag proxy and instead adds an along-velocity retarding force
    # D_i = |L_aero|*tan(alpha), where L_aero is the packed channel's aero
    # lateral force (control_force, taken after the loadFactorMax cap above
    # so drag matches what the missile actually realizes; thrust is
    # excluded).  Absent key -> effective_drag_force_n is aero.drag_force_n
    # unchanged, bit-identical to before.
    induced_drag_mode = str(config["aerodynamics"].get("induced_drag_mode") or "")
    effective_drag_force_n = aero.drag_force_n
    if induced_drag_mode == "momentum_tilt":
        cda0_only_drag_force_n = drag_force_from_cda(
            aero.dynamic_pressure_pa, aero.cda0_m2, aero.air_velocity_hat
        )
        induced_drag_n = norm(control_force) * math.tan(aero.angle_of_attack_rad)
        effective_drag_force_n = add(
            cda0_only_drag_force_n,
            scale(aero.air_velocity_hat, -induced_drag_n),
        )
    elif induced_drag_mode:
        raise ValueError(f"unknown aerodynamics.induced_drag_mode: {induced_drag_mode!r}")
    gravity_force = (0.0, -mass * gravity, 0.0)
    non_gravity = _add_many(
        thrust_force,
        effective_drag_force_n,
        fin_drag_force,
        body_normal_force_vector,
        fixed_lifting_surface_force_vector,
        control_force,
    )
    total_force = _add_many(non_gravity, gravity_force)
    specific_force = scale(non_gravity, 1.0 / mass)
    acceleration = scale(total_force, 1.0 / mass)
    # Acceleration-controller feedback belongs to the missile-mounted body
    # axes, matching guidance.commanded_body_acceleration_g exactly.
    pitch_g = dot(specific_force, axes.up) / gravity
    yaw_g = dot(specific_force, axes.right) / gravity
    wind_basis = cg_wind_normal_basis(state, config)
    wind_pitch_g = dot(specific_force, wind_basis.up) / gravity
    wind_yaw_g = dot(specific_force, wind_basis.right) / gravity
    trajectory_pitch_g = dot(specific_force, aero.flow_normal_pitch) / gravity
    trajectory_yaw_g = dot(specific_force, aero.flow_normal_yaw) / gravity
    axial_g = dot(specific_force, aero.air_velocity_hat) / gravity
    lateral_g = math.hypot(pitch_g, yaw_g)
    trajectory_lateral_g = math.hypot(trajectory_pitch_g, trajectory_yaw_g)
    total_g = norm(specific_force) / gravity

    length = max(float(config["geometry"]["length_m"]), 1e-6)
    inertia_per_mass = max(length * length / 12.0, 1e-6)
    inertia_kg_m2 = mass * inertia_per_mass
    body_reference_area_m2 = 0.0
    body_reference_length_m = 0.0
    body_cp_cg_arm_over_diameter = 0.0
    body_cn_alpha_per_rad = 0.0
    body_cn_q = 0.0
    body_cm_alpha_per_rad = 0.0
    body_cm_q = 0.0
    pitch_body_static_moment_nm = 0.0
    yaw_body_static_moment_nm = 0.0
    pitch_body_rate_moment_nm = 0.0
    yaw_body_rate_moment_nm = 0.0
    pitch_body_total_moment_nm = 0.0
    yaw_body_total_moment_nm = 0.0
    pitch_residual_damping_moment_nm = 0.0
    yaw_residual_damping_moment_nm = 0.0
    pitch_natural_frequency = 0.0
    yaw_natural_frequency = 0.0
    pitch_tail_rate_damping = 0.0
    yaw_tail_rate_damping = 0.0
    pitch_residual_rate_damping = 0.0
    yaw_residual_rate_damping = 0.0
    if generalized_aero_plant:
        body_reference_area_m2 = generalized_reference_area_m2
        body_reference_length_m = generalized_reference_length_m
        generalized = config["aerodynamics"]["generalized_aero_moment_candidate"]
        body_cn_alpha_per_rad = float(generalized["cn_alpha_per_rad"])
        body_cn_q = 0.0
        body_cm_alpha_per_rad = float(generalized["cm_alpha_per_rad"])
        body_cm_q = float(generalized["cm_q"])
        generalized_cm_alpha_dot_per_rad = 0.0
        generalized_cm_alpha_dot_runtime_enabled = False
        pitch_body_static_moment_nm = generalized_pitch_body_static_moment
        yaw_body_static_moment_nm = generalized_yaw_body_static_moment
        pitch_body_rate_moment_nm = generalized_pitch_body_rate_moment
        yaw_body_rate_moment_nm = generalized_yaw_body_rate_moment
        pitch_body_total_moment_nm = (
            pitch_body_static_moment_nm + pitch_body_rate_moment_nm
        )
        yaw_body_total_moment_nm = (
            yaw_body_static_moment_nm + yaw_body_rate_moment_nm
        )
        pitch_tail_force_n = generalized_pitch_delta_force
        yaw_tail_force_n = generalized_yaw_delta_force
        pitch_tail_moment_nm = generalized_pitch_delta_moment
        yaw_tail_moment_nm = generalized_yaw_delta_moment
        pitch_tail_delta_force_n = generalized_pitch_delta_force
        yaw_tail_delta_force_n = generalized_yaw_delta_force
        pitch_tail_delta_force_slope = generalized_pitch_delta_force_slope
        yaw_tail_delta_force_slope = generalized_yaw_delta_force_slope
        # Static CN terms are exported through the body-normal-force fields;
        # the tail fields below remain reserved for the v1 split-tail model.
        pitch_tail_net_force_pre_cap = generalized_pitch_delta_force
        yaw_tail_net_force_pre_cap = generalized_yaw_delta_force
        pitch_tail_delta_moment = generalized_pitch_delta_moment
        yaw_tail_delta_moment = generalized_yaw_delta_moment
        pitch_fin_moment_equivalent_g = (
            pitch_tail_moment_nm / max(gravity * mass * body_reference_length_m, 1e-12)
        )
        yaw_fin_moment_equivalent_g = (
            yaw_tail_moment_nm / max(gravity * mass * body_reference_length_m, 1e-12)
        )
        # This is the resolved generalized Cm_q rate term.  Cm_alpha_dot is
        # frozen at zero because no independent alpha-dot state is available;
        # this is not a synthetic closure or station-derived damping coefficient.
        speed_for_rate = max(float(aero.speed_mps), 1e-6)
        rate_moment_scale = (
            float(aero.normal_force_dynamic_pressure_pa)
            * generalized_reference_area_m2
            * generalized_reference_length_m
            * generalized_reference_length_m
            / (2.0 * speed_for_rate)
        )
        pitch_tail_rate_damping = max(
            0.0,
            -rate_moment_scale
            * float(generalized["cm_q"])
            / max(inertia_kg_m2, 1e-12),
        )
        yaw_tail_rate_damping = pitch_tail_rate_damping
    elif legacy_fin_torque_plant:
        angular_response_scale = 1.0
        arm_magnitude = abs(arm)
        pitch_stiffness_s2 = max(
            scheduled_fins_g * gravity * arm_magnitude / (inertia_per_mass * pitch_fin_limit),
            0.0,
        )
        yaw_stiffness_s2 = max(
            scheduled_fins_g * gravity * arm_magnitude / (inertia_per_mass * yaw_fin_limit),
            0.0,
        )
        pitch_natural_frequency = math.sqrt(pitch_stiffness_s2)
        yaw_natural_frequency = math.sqrt(yaw_stiffness_s2)
        # Spec §5: C = 2 ζ √(K I) with ζ=1, so C/I = 2 ω_n.  The spring
        # no longer includes ω·Δ/V, so this is the full critical-damping
        # term rather than a remainder.
        pitch_tail_rate_damping = 0.0
        yaw_tail_rate_damping = 0.0
        pitch_residual_rate_damping = 2.0 * pitch_natural_frequency
        yaw_residual_rate_damping = 2.0 * yaw_natural_frequency
        pitch_damping = pitch_residual_rate_damping
        yaw_damping = yaw_residual_rate_damping
    elif body_cm_tail_plant:
        body_candidate = config["aerodynamics"]["body_cp_force_candidate"]
        body_reference_area_m2 = caliber_circular_area(config)
        body_reference_length_m = float(config["geometry"]["caliber_m"])
        body_cp_cg_arm_over_diameter = float(
            body_candidate["cp_cg_arm_over_diameter"]
        )
        body_cn_alpha_per_rad = float(body_candidate["cn_alpha_body_per_rad"])
        body_cn_q = float(body_candidate["derived_cn_q"])
        body_cm_alpha_per_rad = float(body_candidate["derived_cm_alpha_per_rad"])
        body_cm_q = float(body_candidate["derived_cm_q"])
        body_arm_inertial = scale(axes.forward, body_station_x_m)
        body_moment_vector = cross(body_arm_inertial, aero.natural_lift_force_n)
        pitch_body_total_moment_nm = dot(body_moment_vector, axes.right)
        yaw_body_total_moment_nm = -dot(body_moment_vector, axes.up)
        static_body_aero = compute_aerodynamics_h2(
            state.position[1],
            state.velocity,
            state.pitch,
            state.yaw,
            config,
            atmosphere,
            axes_override=axes,
            normal_force_velocity_mps=state.velocity,
        )
        static_body_moment_vector = cross(
            body_arm_inertial,
            static_body_aero.natural_lift_force_n,
        )
        pitch_body_static_moment_nm = dot(static_body_moment_vector, axes.right)
        yaw_body_static_moment_nm = -dot(static_body_moment_vector, axes.up)
        pitch_body_rate_moment_nm = (
            pitch_body_total_moment_nm - pitch_body_static_moment_nm
        )
        yaw_body_rate_moment_nm = yaw_body_total_moment_nm - yaw_body_static_moment_nm
        # Body and tail rate damping now arise only from local station velocity.
        # No synthetic critical residual or independent Cm term is executed.
        pitch_tail_rate_damping = (
            scheduled_fins_g
            * gravity
            * tail_station_x_m ** 2
            / (inertia_per_mass * pitch_authority_reference * speed_for_rate)
        )
        yaw_tail_rate_damping = (
            scheduled_fins_g
            * gravity
            * tail_station_x_m ** 2
            / (inertia_per_mass * yaw_authority_reference * speed_for_rate)
        )
        pitch_damping = 0.0
        yaw_damping = 0.0
    else:
        angular_response_scale = float(config["control"].get("angular_response_scale", 1.0))
        pitch_damping = float(config["control"]["angular_damping"])
        yaw_damping = pitch_damping
    pitch_moment_equivalent_g = pitch_fin_moment_equivalent_g
    yaw_moment_equivalent_g = yaw_fin_moment_equivalent_g
    if (
        plant_semantics == "direct_fin_g"
        and config["control"].get("fin_aoa_moment_enabled", False)
    ):
        fins_g = float(config["aerodynamics"]["fins_lateral_acceleration_g"])
        speed_schedule = base_indicated_speed_schedule(aero.dynamic_pressure_pa, config)
        fins_g *= speed_schedule.fin_force_scale
        pitch_fin_limit = math.radians(max(float(config["aerodynamics"]["horizontal_fin_aoa_limit_deg"]), 1e-9))
        yaw_fin_limit = math.radians(max(float(config["aerodynamics"]["vertical_fin_aoa_limit_deg"]), 1e-9))
        # The fin actuator supplies the commanded lateral load, while body AoA
        # supplies the stabilizing part of the tail moment.  This keeps CyK out
        # of the plant and makes fin angle, finsLatAccel and arm explicit.
        pitch_moment_equivalent_g -= fins_g * aero.pitch_alpha_rad / pitch_fin_limit
        yaw_moment_equivalent_g -= fins_g * aero.yaw_alpha_rad / yaw_fin_limit
    if body_cm_tail_plant:
        pitch_total_moment_nm = (
            pitch_body_total_moment_nm
            + pitch_fixed_lifting_surface_moment
            + pitch_tail_moment_nm
        )
        yaw_total_moment_nm = (
            yaw_body_total_moment_nm
            + yaw_fixed_lifting_surface_moment
            + yaw_tail_moment_nm
        )
        pitch_angular_accel = pitch_total_moment_nm / inertia_kg_m2
        yaw_angular_accel = yaw_total_moment_nm / inertia_kg_m2
    elif generalized_aero_plant:
        pitch_total_moment_nm = (
            pitch_body_total_moment_nm
            + pitch_fixed_lifting_surface_moment
            + pitch_tail_moment_nm
        )
        yaw_total_moment_nm = (
            yaw_body_total_moment_nm
            + yaw_fixed_lifting_surface_moment
            + yaw_tail_moment_nm
        )
        pitch_angular_accel = pitch_total_moment_nm / inertia_kg_m2
        yaw_angular_accel = yaw_total_moment_nm / inertia_kg_m2
    else:
        pitch_angular_accel = (
            pitch_moment_equivalent_g * gravity * arm / inertia_per_mass * angular_response_scale
        ) - pitch_damping * state.pitch_rate
        yaw_angular_accel = (
            yaw_moment_equivalent_g * gravity * arm / inertia_per_mass * angular_response_scale
        ) - yaw_damping * state.yaw_rate
        if not legacy_fin_torque_plant:
            pitch_tail_force_n = state.actual_pitch_acceleration_g * gravity * mass
            yaw_tail_force_n = state.actual_yaw_acceleration_g * gravity * mass
            pitch_tail_moment_nm = pitch_moment_equivalent_g * gravity * mass * arm
            yaw_tail_moment_nm = yaw_moment_equivalent_g * gravity * mass * arm
        pitch_residual_damping_moment_nm = -pitch_damping * state.pitch_rate * inertia_kg_m2
        yaw_residual_damping_moment_nm = -yaw_damping * state.yaw_rate * inertia_kg_m2
        pitch_total_moment_nm = pitch_tail_moment_nm + pitch_residual_damping_moment_nm
        yaw_total_moment_nm = yaw_tail_moment_nm + yaw_residual_damping_moment_nm
    lift_force = _add_many(
        body_normal_force_vector,
        fixed_lifting_surface_force_vector,
        control_force,
    )
    total_body_wing_tail_normal_force = lift_force
    return H2DynamicsDiagnostics(
        propulsion=prop,
        aero=aero,
        thrust_force_n=thrust_force,
        drag_force_n=effective_drag_force_n,
        fin_drag_force_n=fin_drag_force,
        natural_lift_force_n=body_normal_force_vector,
        fixed_lifting_surface_force_n=fixed_lifting_surface_force_vector,
        control_force_n=control_force,
        gravity_force_n=gravity_force,
        total_force_n=total_force,
        non_gravity_acceleration_mps2=specific_force,
        specific_force_mps2=specific_force,
        acceleration_mps2=acceleration,
        axial_specific_force_g=axial_g,
        pitch_normal_acceleration_g=pitch_g,
        yaw_normal_acceleration_g=yaw_g,
        wind_normal_pitch_acceleration_g=wind_pitch_g,
        wind_normal_yaw_acceleration_g=wind_yaw_g,
        lateral_load_g=lateral_g,
        trajectory_pitch_normal_acceleration_g=trajectory_pitch_g,
        trajectory_yaw_normal_acceleration_g=trajectory_yaw_g,
        trajectory_lateral_load_g=trajectory_lateral_g,
        total_specific_force_g=total_g,
        drag_power_w=dot(effective_drag_force_n, aero.air_velocity_mps),
        lift_power_w=dot(lift_force, aero.air_velocity_mps),
        body_tail_force_power_at_cg_w=dot(lift_force, aero.air_velocity_mps),
        pitch_angular_acceleration_rad_s2=pitch_angular_accel,
        yaw_angular_acceleration_rad_s2=yaw_angular_accel,
        body_reference_area_m2=body_reference_area_m2,
        body_reference_length_m=body_reference_length_m,
        body_cp_cg_arm_over_diameter=body_cp_cg_arm_over_diameter,
        body_cn_alpha_per_rad=body_cn_alpha_per_rad,
        body_cn_q=body_cn_q,
        body_cm_alpha_per_rad=body_cm_alpha_per_rad,
        body_cm_q=body_cm_q,
        generalized_cm_alpha_dot_per_rad=generalized_cm_alpha_dot_per_rad,
        generalized_pitch_alpha_dot_hat=generalized_pitch_alpha_dot_hat,
        generalized_yaw_alpha_dot_hat=generalized_yaw_alpha_dot_hat,
        generalized_cm_alpha_dot_runtime_enabled=generalized_cm_alpha_dot_runtime_enabled,
        pitch_body_normal_force_n=dot(body_normal_force_vector, aero.flow_normal_pitch),
        yaw_body_normal_force_n=dot(body_normal_force_vector, aero.flow_normal_yaw),
        pitch_body_static_moment_nm=pitch_body_static_moment_nm,
        yaw_body_static_moment_nm=yaw_body_static_moment_nm,
        pitch_body_rate_moment_nm=pitch_body_rate_moment_nm,
        yaw_body_rate_moment_nm=yaw_body_rate_moment_nm,
        pitch_body_total_moment_nm=pitch_body_total_moment_nm,
        yaw_body_total_moment_nm=yaw_body_total_moment_nm,
        fixed_lifting_surface_multiplier=fixed_lifting_surface_multiplier,
        body_normal_force_area_slope_m2_per_rad=body_normal_force_area_slope,
        fixed_lifting_surface_area_slope_m2_per_rad=fixed_lifting_surface_area_slope,
        fixed_lifting_surface_station_x_m=fixed_lifting_surface_station_x,
        pitch_fixed_lifting_surface_alpha_rad=pitch_fixed_lifting_surface_alpha,
        yaw_fixed_lifting_surface_alpha_rad=yaw_fixed_lifting_surface_alpha,
        pitch_fixed_lifting_surface_force_n=pitch_fixed_lifting_surface_force,
        yaw_fixed_lifting_surface_force_n=yaw_fixed_lifting_surface_force,
        pitch_fixed_lifting_surface_moment_nm=pitch_fixed_lifting_surface_moment,
        yaw_fixed_lifting_surface_moment_nm=yaw_fixed_lifting_surface_moment,
        pitch_total_body_wing_tail_normal_force_n=dot(
            total_body_wing_tail_normal_force, cg_flow_normal_pitch
        ),
        yaw_total_body_wing_tail_normal_force_n=dot(
            total_body_wing_tail_normal_force, cg_flow_normal_yaw
        ),
        tail_station_x_m=tail_station_x_m,
        pitch_tail_force_n=pitch_tail_force_n,
        yaw_tail_force_n=yaw_tail_force_n,
        pitch_tail_moment_nm=pitch_tail_moment_nm,
        yaw_tail_moment_nm=yaw_tail_moment_nm,
        pitch_tail_authority_fraction=pitch_moment_fraction,
        yaw_tail_authority_fraction=yaw_moment_fraction,
        tail_alpha_force_multiplier=tail_alpha_force_multiplier,
        tail_delta_force_multiplier=tail_delta_force_multiplier,
        pitch_tail_alpha_force_slope_n_per_rad=pitch_tail_alpha_force_slope,
        yaw_tail_alpha_force_slope_n_per_rad=yaw_tail_alpha_force_slope,
        pitch_tail_delta_force_slope_n_per_rad=pitch_tail_delta_force_slope,
        yaw_tail_delta_force_slope_n_per_rad=yaw_tail_delta_force_slope,
        pitch_tail_alpha_force_n=pitch_tail_alpha_force,
        yaw_tail_alpha_force_n=yaw_tail_alpha_force,
        pitch_tail_delta_force_n=pitch_tail_delta_force,
        yaw_tail_delta_force_n=yaw_tail_delta_force,
        pitch_tail_net_force_pre_cap_n=pitch_tail_net_force_pre_cap,
        yaw_tail_net_force_pre_cap_n=yaw_tail_net_force_pre_cap,
        tail_force_cap_n=tail_force_cap,
        tail_force_cap_scale=tail_force_cap_scale,
        tail_force_cap_active=tail_force_cap_active,
        pitch_tail_alpha_moment_nm=pitch_tail_alpha_moment,
        yaw_tail_alpha_moment_nm=yaw_tail_alpha_moment,
        pitch_tail_delta_moment_nm=pitch_tail_delta_moment,
        yaw_tail_delta_moment_nm=yaw_tail_delta_moment,
        pitch_residual_damping_moment_nm=pitch_residual_damping_moment_nm,
        yaw_residual_damping_moment_nm=yaw_residual_damping_moment_nm,
        pitch_total_moment_nm=pitch_total_moment_nm,
        yaw_total_moment_nm=yaw_total_moment_nm,
        pitch_body_aoa_force_g=pitch_body_aoa_force_g,
        yaw_body_aoa_force_g=yaw_body_aoa_force_g,
        pitch_fin_moment_equivalent_g=pitch_fin_moment_equivalent_g,
        yaw_fin_moment_equivalent_g=yaw_fin_moment_equivalent_g,
        pitch_tail_rate_incidence_rad=pitch_tail_rate_incidence,
        yaw_tail_rate_incidence_rad=yaw_tail_rate_incidence,
        pitch_tail_effective_incidence_rad=pitch_tail_effective_incidence,
        yaw_tail_effective_incidence_rad=yaw_tail_effective_incidence,
        pitch_natural_frequency_rad_s=pitch_natural_frequency,
        yaw_natural_frequency_rad_s=yaw_natural_frequency,
        pitch_tail_rate_damping_per_s=pitch_tail_rate_damping,
        yaw_tail_rate_damping_per_s=yaw_tail_rate_damping,
        pitch_residual_rate_damping_per_s=pitch_residual_rate_damping,
        yaw_residual_rate_damping_per_s=yaw_residual_rate_damping,
    )


