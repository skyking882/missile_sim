# ==== VERBATIM EXCERPT ====================================================
# source : src/aim120_model/dynamics.py
# lines  : 80-253   (1-based, inclusive; original line numbers)
# note   : 姿态路径：forces_for_state / 状态向量 / RK4 / clamp_rates / clamp_body_angle_of_attack
# repo   : working tree at 2026-08-26, unmodified
# ==========================================================================

def forces_for_state(
    state: SimState,
    time_s: float,
    config: dict[str, Any],
    propulsion: PiecewisePropulsion,
    powered: bool,
) -> DynamicsDiagnostics:
    atmosphere = StandardAtmosphere()
    prop = propulsion.sample(time_s, powered=powered)
    # The mass state is authoritative during integration; use the explicit
    # piecewise value at the start/end of a step in the simulator.
    mass = max(float(state.mass), 1e-6)
    axes = body_axes(state.pitch, state.yaw)
    aero = compute_aerodynamics(
        state.position[1], state.velocity, state.pitch, state.yaw, config, atmosphere
    )
    gravity = config["atmosphere"]["gravity_mps2"]
    thrust_force = scale(axes.forward, prop.thrust_n)
    control_force = _add_many(
        scale(axes.up, state.actual_pitch_acceleration_g * gravity * mass),
        scale(axes.right, state.actual_yaw_acceleration_g * gravity * mass),
    )
    gravity_force = (0.0, -mass * gravity, 0.0)
    total_force = _add_many(
        thrust_force,
        aero.drag_force_n,
        aero.natural_lift_force_n,
        control_force,
        gravity_force,
    )
    non_gravity = _add_many(thrust_force, aero.drag_force_n, aero.natural_lift_force_n, control_force)
    non_gravity_acceleration = scale(non_gravity, 1.0 / mass)
    acceleration = scale(total_force, 1.0 / mass)
    length = max(float(config["geometry"]["length_m"]), 1e-6)
    inertia_per_mass = max(length * length / 12.0, 1e-6)
    arm = float(config["aerodynamics"]["distance_cm_to_stabilizer_m"])
    angular_response_scale = float(config["control"].get("angular_response_scale", 1.0))
    damping = float(config["control"]["angular_damping"])
    pitch_angular_accel = (state.actual_pitch_acceleration_g * gravity * arm / inertia_per_mass * angular_response_scale) - damping * state.pitch_rate
    yaw_angular_accel = (state.actual_yaw_acceleration_g * gravity * arm / inertia_per_mass * angular_response_scale) - damping * state.yaw_rate
    return DynamicsDiagnostics(
        propulsion=prop,
        aero=aero,
        thrust_force_n=thrust_force,
        drag_force_n=aero.drag_force_n,
        natural_lift_force_n=aero.natural_lift_force_n,
        control_force_n=control_force,
        gravity_force_n=gravity_force,
        total_force_n=total_force,
        non_gravity_acceleration_mps2=non_gravity_acceleration,
        acceleration_mps2=acceleration,
        pitch_angular_acceleration_rad_s2=pitch_angular_accel,
        yaw_angular_acceleration_rad_s2=yaw_angular_accel,
    )


def _dynamic_vector(state: SimState) -> tuple[float, ...]:
    return (
        state.position[0], state.position[1], state.position[2],
        state.velocity[0], state.velocity[1], state.velocity[2],
        state.pitch, state.yaw, state.pitch_rate, state.yaw_rate, state.mass,
    )


def _state_from_dynamic(state: SimState, values: tuple[float, ...]) -> SimState:
    return replace(
        state,
        position=(values[0], values[1], values[2]),
        velocity=(values[3], values[4], values[5]),
        pitch=values[6],
        yaw=values[7],
        pitch_rate=values[8],
        yaw_rate=values[9],
        mass=values[10],
    )


def _derivative_vector(
    state: SimState,
    time_s: float,
    config: dict[str, Any],
    propulsion: PiecewisePropulsion,
    powered: bool,
) -> tuple[float, ...]:
    diagnostics = forces_for_state(state, time_s, config, propulsion, powered)
    return (
        state.velocity[0], state.velocity[1], state.velocity[2],
        diagnostics.acceleration_mps2[0], diagnostics.acceleration_mps2[1], diagnostics.acceleration_mps2[2],
        state.pitch_rate, state.yaw_rate,
        diagnostics.pitch_angular_acceleration_rad_s2,
        diagnostics.yaw_angular_acceleration_rad_s2,
        diagnostics.propulsion.mass_flow_kg_s,
    )


def _combine(values: tuple[float, ...], derivative: tuple[float, ...], factor: float) -> tuple[float, ...]:
    return tuple(value + factor * delta for value, delta in zip(values, derivative))


def rk4_step(
    state: SimState,
    time_s: float,
    dt_s: float,
    config: dict[str, Any],
    propulsion: PiecewisePropulsion,
    powered: bool,
) -> SimState:
    values = _dynamic_vector(state)
    k1 = _derivative_vector(state, time_s, config, propulsion, powered)
    s2 = _state_from_dynamic(state, _combine(values, k1, dt_s / 2.0))
    k2 = _derivative_vector(s2, time_s + dt_s / 2.0, config, propulsion, powered)
    s3 = _state_from_dynamic(state, _combine(values, k2, dt_s / 2.0))
    k3 = _derivative_vector(s3, time_s + dt_s / 2.0, config, propulsion, powered)
    s4 = _state_from_dynamic(state, _combine(values, k3, dt_s))
    k4 = _derivative_vector(s4, time_s + dt_s, config, propulsion, powered)
    next_values = tuple(
        value + dt_s * (d1 + 2.0 * d2 + 2.0 * d3 + d4) / 6.0
        for value, d1, d2, d3, d4 in zip(values, k1, k2, k3, k4)
    )
    return _state_from_dynamic(state, next_values)


def clamp_rates(state: SimState, config: dict[str, Any]) -> SimState:
    maximum = deg_to_rad(config["control"]["max_pitch_yaw_rate_deg_s"])
    return replace(
        state,
        pitch_rate=max(-maximum, min(maximum, state.pitch_rate)),
        yaw_rate=max(-maximum, min(maximum, state.yaw_rate)),
    )


def clamp_body_angle_of_attack(state: SimState, config: dict[str, Any]) -> SimState:
    """Project the roll-free body attitude onto an explicit total-AoA bound."""

    control = config["control"]
    if not control.get("limit_angle_of_attack_enabled", False):
        return state
    vx, vy, vz = state.velocity
    horizontal_speed = math.hypot(vx, vz)
    if horizontal_speed <= 1e-9 and abs(vy) <= 1e-9:
        return state
    flight_pitch = math.atan2(vy, horizontal_speed)
    flight_yaw = math.atan2(vz, vx)
    pitch_error = math.atan2(math.sin(state.pitch - flight_pitch), math.cos(state.pitch - flight_pitch))
    yaw_error = math.atan2(math.sin(state.yaw - flight_yaw), math.cos(state.yaw - flight_yaw))
    total_error = math.hypot(pitch_error, yaw_error)
    maximum = deg_to_rad(max(float(control["maximum_body_angle_of_attack_deg"]), 0.0))
    if total_error <= maximum or total_error <= 1e-12:
        return state
    ratio = maximum / total_error
    limited_pitch_error = pitch_error * ratio
    limited_yaw_error = yaw_error * ratio
    pitch_rate = state.pitch_rate
    yaw_rate = state.yaw_rate
    if pitch_error * pitch_rate > 0.0:
        pitch_rate = 0.0
    if yaw_error * yaw_rate > 0.0:
        yaw_rate = 0.0
    new_pitch = flight_pitch + limited_pitch_error
    new_yaw = flight_yaw + limited_yaw_error
    return replace(
        state,
        pitch=new_pitch,
        yaw=new_yaw,
        pitch_rate=pitch_rate,
        yaw_rate=yaw_rate,
        orientation_quaternion=(
            quaternion_from_pitch_yaw(new_pitch, new_yaw)
            if state.orientation_quaternion is not None
            else None
        ),
    )


