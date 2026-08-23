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
    """Return w(t): 1 before lock_delay_s, linear down to 0 across blend_time_s.

    PN stays on the whole flight (it is small early); this weight only keeps
    the lead-turn term from double-counting once PN has taken over.
    """

    lock_delay_s = float(midcourse_cfg.get("lock_delay_s", 0.8))
    blend_time_s = float(midcourse_cfg.get("blend_time_s", 0.5))
    if time_s < lock_delay_s:
        return 1.0
    if blend_time_s <= 0.0:
        return 0.0
    fraction = (time_s - lock_delay_s) / blend_time_s
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


def guidance_command(
    state: Any,
    target: TrackSolution | TargetState,
    time_s: float,
    config: dict[str, Any],
    enabled: bool,
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
    midcourse_weight = midcourse_blend_weight(time_s, midcourse_cfg)
    midcourse_active = enabled and bool(midcourse_cfg.get("enabled", False))
    midcourse_command = scale(midcourse_accel, midcourse_weight) if midcourse_active else (0.0, 0.0, 0.0)
    commanded = add_vectors(add_vectors(pn, loft), midcourse_command)
    commanded = sub(commanded, scale(v_hat, dot(commanded, v_hat)))
    max_accel = g_to_mps2(
        guidance_cfg["maximum_lateral_acceleration_g"],
        config["atmosphere"]["gravity_mps2"],
    )
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
    if midcourse_active and midcourse_weight > 0.0:
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
    )


def add_vectors(a: Vector, b: Vector) -> Vector:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])
