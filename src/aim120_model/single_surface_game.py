"""Opt-in freely integrated, single-surface War Thunder candidate.

This promotes the accepted R-77-1 conditional attitude family to a free
three-dimensional translation/attitude model. Its coefficients are empirical
game-replay calibration, not native engine equations. One aerodynamic
resultant and one reference free couple act at the same aft station. The
couple realizes the calibrated angular-acceleration law; the station's lever
moment is not separately identified by that law.

The wind-linear reference and body-linear projection differ only in force
basis. Existing total wind drag is applied once. No observed missile state is
read after creation. Targets, raw propulsion, geometry, PID and gain tables
are external inputs. The controller runs at 48 Hz independently of RK4 plant
substeps, with zero memory before the end-of-interval flight gain activates.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace
from typing import Any

from .aerodynamics import (BodyAxes, body_axes_for_state, normalize_quaternion,
    pitch_yaw_from_quaternion, quaternion_derivative, quaternion_from_pitch_yaw,
    reference_area, wind_normal_basis)
from .atmosphere import StandardAtmosphere
from .control import scheduled_pid_values
from .drag_models import effective_cda0, effective_cda_alpha
from .dynamics import SimState
from .guidance import estimate_time_to_go_s, interpolate_table, pn_acceleration
from .math3d import add, clamp, clamp_norm, dot, norm, normalize, scale, sub
from .propulsion import PiecewisePropulsion
from .target import TargetState
from .units import G0_MPS2


@dataclass(frozen=True)
class SurfaceCoefficients:
    ca: float = 14.606915444766543
    speed_slope: float = 0.17017244388915495
    control_gain: float = 865.9202017326287
    restoring_gain: float = 31.833800106342316
    rate_damping: float = 3.534664411413437

    def __post_init__(self):
        if not all(math.isfinite(x) and x > 0 for x in
                   (self.ca, self.control_gain, self.restoring_gain, self.rate_damping)):
            raise ValueError("force, control, restoring and damping coefficients must be positive")
        if not math.isfinite(self.speed_slope) or self.speed_slope < 0:
            raise ValueError("speed slope must be finite and nonnegative")


@dataclass(frozen=True)
class SurfaceEvaluation:
    acceleration: tuple[float, float, float]
    specific_acceleration: tuple[float, float, float]
    aerodynamic_force: tuple[float, float, float]
    wind_lift_force: tuple[float, float, float]
    wind_drag_force: tuple[float, float, float]
    wind_basis: BodyAxes
    alpha_pitch: float
    alpha_yaw: float
    alpha_total: float
    local_incidence: tuple[float, float]
    feedback_g: tuple[float, float]
    angular_acceleration: tuple[float, float]
    lever_moment: tuple[float, float]
    reference_couple: tuple[float, float]
    inertia: float
    mass: float
    thrust: float
    qbar: float
    drag_n: float


class SingleSurfaceGame:
    """A transparent candidate; not selected by any production runtime."""
    controller_hz = 48

    def __init__(self, config: dict[str, Any], *, coefficients=SurfaceCoefficients(),
                 force_basis="wind_linear", plant_substeps=4, loft_cap_unit="deg_s"):
        if force_basis not in ("wind_linear", "body_linear"):
            raise ValueError("force_basis must be wind_linear or body_linear")
        if int(plant_substeps) != plant_substeps or plant_substeps < 1:
            raise ValueError("plant_substeps must be a positive integer")
        self.config = copy.deepcopy(config)
        self.coefficients = coefficients
        self.force_basis = force_basis
        if loft_cap_unit not in ("deg_s", "rad_s"):
            raise ValueError("loft_cap_unit must be deg_s or rad_s")
        self.loft_cap_unit = loft_cap_unit
        self.plant_substeps = int(plant_substeps)
        self.propulsion = PiecewisePropulsion.from_config(config)
        self.atmosphere = StandardAtmosphere()
        self.area = reference_area(config)
        self.arm = float(config["aerodynamics"]["distance_cm_to_stabilizer_m"])
        self.length = float(config["geometry"]["length_m"])
        self.gravity = float(config["atmosphere"]["gravity_mps2"])
        self.wind = tuple(float(x) for x in config["atmosphere"].get("wind_mps", (0, 0, 0)))

    def initialize(self, state: SimState):
        """Set normalization once from supplied creation state, never observations."""
        state = replace(state, orientation_quaternion=normalize_quaternion(
            state.orientation_quaternion or quaternion_from_pitch_yaw(state.pitch, state.yaw)))
        self.speed0 = norm(sub(state.velocity, self.wind))
        if self.speed0 <= 1e-6:
            raise ValueError("creation airspeed must be positive")
        rho0 = self.atmosphere.sample(state.position[1]).density_kg_m3
        self.q_over_mass0 = .5 * rho0 * self.speed0**2 / self.propulsion.mass_at(0.)
        return state

    def evaluate(self, state: SimState, time_s: float, held: tuple[float, float]) -> SurfaceEvaluation:
        prop = self.propulsion.sample(time_s)
        mass = prop.mass_kg
        atm = self.atmosphere.sample(state.position[1])
        axes = body_axes_for_state(state)
        air_velocity = sub(state.velocity, self.wind)
        speed = norm(air_velocity)
        if speed <= 1e-6:
            raise ValueError("candidate airspeed reached zero")
        wind_basis = wind_normal_basis(axes, air_velocity)
        tangent = wind_basis.forward
        forward = dot(tangent, axes.forward)
        alpha_pitch = math.atan2(-dot(tangent, axes.up), forward)
        alpha_yaw = math.atan2(-dot(tangent, axes.right), forward)
        alpha_total = math.atan2(math.hypot(dot(tangent, axes.up), dot(tangent, axes.right)), forward)
        local = (alpha_pitch + self.arm*state.pitch_rate/speed,
                 alpha_yaw + self.arm*state.yaw_rate/speed)
        qbar = .5*atm.density_kg_m3*speed**2
        c = self.coefficients
        npitch, nyaw = qbar*self.area*c.ca*local[0], qbar*self.area*c.ca*local[1]
        if self.force_basis == "wind_linear":
            lift = add(scale(wind_basis.up, npitch), scale(wind_basis.right, nyaw))
        else:
            # In the planar limit this is L = q*S*Ca*alpha_local*cos(alpha).
            # Its tangent is removed because CxAoA already denotes wind drag.
            normal = add(scale(axes.up, npitch), scale(axes.right, nyaw))
            lift = sub(normal, scale(tangent, dot(normal, tangent)))
        mach = speed/atm.speed_of_sound_mps
        drag_n = qbar*(effective_cda0(mach, self.config) +
                      effective_cda_alpha(alpha_pitch, alpha_yaw, mach, self.config))
        drag = scale(tangent, -drag_n)
        aero = add(lift, drag)
        specific = scale(add(aero, scale(axes.forward, prop.thrust_n)), 1/mass)
        acceleration = add(specific, (0., -self.gravity, 0.))
        feedback = (dot(specific, wind_basis.up)/G0_MPS2,
                    dot(specific, wind_basis.right)/G0_MPS2)
        active = (qbar/mass)/self.q_over_mass0
        damping_scale = active*self.speed0/speed
        gain = c.control_gain*(1+c.speed_slope*(speed/self.speed0-1))
        rates = (state.pitch_rate, state.yaw_rate)
        qdot = tuple(gain*u-active*c.restoring_gain*a-damping_scale*c.rate_damping*w
                     for u, a, w in zip(held, local, rates))
        inertia = mass*self.length**2/12
        # One resultant at x=-arm; both reference couples are at that point.
        lever = (-self.arm*dot(aero, axes.up), -self.arm*dot(aero, axes.right))
        couple = tuple(inertia*q-l for q, l in zip(qdot, lever))
        return SurfaceEvaluation(acceleration, specific, aero, lift, drag, wind_basis,
            alpha_pitch, alpha_yaw, alpha_total, local, feedback, qdot, lever,
            couple, inertia, mass, prop.thrust_n, qbar, drag_n)

    def request(self, state: SimState, target: TargetState, time_s: float, evaluation: SurfaceEvaluation):
        """Existing PN plus raw simple loft hypothesis, without timed direct fin routing.

        Gain placement reproduces the accepted conditional family's lifecycle:
        timeToHitGain scales PN; flightTimeGain scales the entire PID output.
        Current profile interpretation treats the loft cap as deg/s. Raw
        targetElevation remains retained but unused by this unchanged simple
        law; this is an explicit candidate limitation, not identified semantics.
        """
        cfg = self.config["guidance"]
        rel = sub(target.position, state.position)
        rel_v = sub(target.velocity, state.velocity)
        pn, closing, los = pn_acceleration(rel, rel_v, state.velocity, float(cfg["pn_gain"]))
        distance = norm(rel)
        tgo = estimate_time_to_go_s(distance, closing, rel_v, self.config)
        hit_gain = interpolate_table(tgo, cfg["time_to_hit_gain_table"])
        pn = scale(pn, hit_gain)
        loft_g = 0.
        loft_active = bool(cfg.get("lofting_enabled", False) and
            distance > float(cfg.get("loft_exit_distance_m", 0.)) and
            tgo > float(cfg.get("loft_exit_time_to_go_s", 0.)))
        if loft_active:
            loft_g = float(cfg["angle_to_acceleration_multiplier"])*(
                math.radians(float(cfg["lofting_elevation_deg"]))-state.pitch)
            omega_raw = cfg.get("loft_omega_max_deg_s")
            if omega_raw is not None:
                omega = math.radians(float(omega_raw)) if self.loft_cap_unit == "deg_s" else float(omega_raw)
                cap = omega*norm(state.velocity)/self.gravity
                if cap > 0:
                    loft_g = clamp(loft_g, -cap, cap)
        tangent = evaluation.wind_basis.forward
        max_accel = float(cfg["maximum_lateral_acceleration_g"])*self.gravity
        commanded = add(pn, (0., loft_g*self.gravity, 0.))
        commanded = clamp_norm(sub(commanded, scale(tangent, dot(commanded, tangent))), max_accel)
        required = clamp_norm(add(commanded, (0., self.gravity, 0.)), max_accel)
        request = (dot(required, evaluation.wind_basis.up)/G0_MPS2,
                   dot(required, evaluation.wind_basis.right)/G0_MPS2)
        return request, {"range_m": distance, "tgo_s": tgo, "los_rate_rad_s": los,
            "pn_mps2": pn, "loft_g": loft_g, "loft_active": loft_active,
            "hit_gain": hit_gain}

    def _pid(self, error, memory, pid, fin_limit, gain):
        integral, previous_error, previous_derivative = memory
        dt = 1/self.controller_hz
        provisional = clamp(integral+pid["i"]*error*dt, -pid["integral_limit"], pid["integral_limit"])
        raw_d = (error-previous_error)/dt
        derivative = previous_derivative+dt/(1e-9+dt)*(raw_d-previous_derivative)
        output = pid["p"]*error+provisional+pid["d"]*derivative
        u = clamp(output, -fin_limit, fin_limit)
        next_i = integral if abs(u) >= .99*fin_limit and error*u > 0 else provisional
        return gain*u, (next_i, error, derivative)

    def step(self, state: SimState, time_s: float, dt_s: float, held, *, left_end=False):
        quat = state.orientation_quaternion
        values = (*state.position, *state.velocity, *quat, state.pitch_rate, state.yaw_rate)
        def unpack(v):
            q = normalize_quaternion(tuple(v[6:10]))
            p, y = pitch_yaw_from_quaternion(q)
            return replace(state, position=tuple(v[:3]), velocity=tuple(v[3:6]),
                orientation_quaternion=q, pitch=p, yaw=y, pitch_rate=v[10], yaw_rate=v[11])
        def rhs(t, v):
            s = unpack(v)
            f = self.evaluate(s, t, held)
            return (*s.velocity, *f.acceleration,
                    *quaternion_derivative(s.orientation_quaternion, s.pitch_rate, s.yaw_rate),
                    *f.angular_acceleration)
        def plus(v, k, h):
            return tuple(a+h*b for a,b in zip(v,k))
        k1 = rhs(time_s, values)
        k2 = rhs(time_s+dt_s/2, plus(values, k1, dt_s/2))
        k3 = rhs(time_s+dt_s/2, plus(values, k2, dt_s/2))
        end = math.nextafter(time_s+dt_s, time_s) if left_end else time_s+dt_s
        k4 = rhs(end, plus(values, k3, dt_s))
        result = unpack(tuple(v+dt_s*(a+2*b+2*c+d)/6 for v,a,b,c,d in zip(values,k1,k2,k3,k4)))
        return replace(result, mass=self.propulsion.mass_at(time_s+dt_s))

    def run(self, initial_state: SimState, target_at, *, end_time_s: float):
        """Integrate only the creation state and external target function."""
        state = self.initialize(initial_state)
        memory = ((0.,0.,0.), (0.,0.,0.))
        held = (0.,0.)
        samples, ticks = [], []
        first_active = None
        max_alpha = 0.
        minimum_range = math.inf
        terminal = "lifetime"
        nominal_step = 1/(self.controller_hz*self.plant_substeps)
        time_s = 0.
        tick = 0
        fin_limits = (math.radians(self.config["aerodynamics"]["horizontal_fin_aoa_limit_deg"]),
                      math.radians(self.config["aerodynamics"]["vertical_fin_aoa_limit_deg"]))
        while time_s <= end_time_s+1e-10:
            target = target_at(time_s)
            f = self.evaluate(state, time_s, held)
            request, details = self.request(state, target, time_s, f)
            minimum_range = min(minimum_range, details["range_m"])
            max_alpha = max(max_alpha, f.alpha_total)
            is_tick = abs(time_s-tick/self.controller_hz) < 1e-9
            if is_tick:
                gain = interpolate_table((tick+1)/self.controller_hz,
                    self.config["guidance"]["flight_time_gain_table"])
                if gain > 1e-12:
                    pid = scheduled_pid_values(self.config["control"]["pid"], (tick+1)/self.controller_hz)
                    updated = [self._pid(req-fb, mem, pid, limit, gain)
                        for req,fb,mem,limit in zip(request, f.feedback_g, memory, fin_limits)]
                    held, memory = tuple(x[0] for x in updated), tuple(x[1] for x in updated)
                    if first_active is None:
                        first_active = time_s
                else:
                    if memory != ((0.,0.,0.), (0.,0.,0.)):
                        raise ValueError("gain returned inactive after PID activation")
                    held = (0.,0.)
                ticks.append({"time_s": time_s, "gain": gain, "held_fin_rad": held,
                    "feedback_g": f.feedback_g, "request_g": request, "pid_memory": memory,
                    **details})
            samples.append({"time_s": time_s, "position_m": state.position,
                "velocity_mps": state.velocity, "pitch_rad": state.pitch,
                "pitch_rate_rad_s": state.pitch_rate, "yaw_rate_rad_s": state.yaw_rate,
                "orientation_quaternion": state.orientation_quaternion,
                "angle_of_attack_rad": f.alpha_total, "pitch_alpha_rad": f.alpha_pitch,
                "trajectory_lateral_load_g": math.hypot(*f.feedback_g),
                "feedback_pitch_g": f.feedback_g[0], "feedback_yaw_g": f.feedback_g[1],
                "mass_kg": f.mass, "thrust_n": f.thrust,
                "held_fin_rad": held, "lever_moment_nm": f.lever_moment,
                "reference_couple_nm": f.reference_couple,
                "inertia_kg_m2": f.inertia, "angular_acceleration_rad_s2": f.angular_acceleration})
            if terminal == "proximity_fuse":
                break
            if time_s >= end_time_s-1e-10:
                break
            if state.position[1] < 0:
                terminal = "ground"
                break
            if not all(math.isfinite(x) for x in (*state.position, *state.velocity,
                    state.pitch_rate, state.yaw_rate)):
                terminal = "numerical_failure"
                break
            fuse = float(self.config["guidance"].get("proximity_radius_m", 0.))
            if fuse > 0 and details["range_m"] <= fuse:
                terminal = "proximity_fuse"
                break
            next_tick = (tick+1)/self.controller_hz
            finish = min(next_tick, end_time_s)
            boundary = self.propulsion.next_boundary_after(time_s)
            while time_s < finish-1e-12:
                nxt = min(time_s+nominal_step, finish)
                if boundary is not None and boundary < nxt+1e-12:
                    nxt = boundary
                previous_state = state
                previous_time = time_s
                relative_start = sub(target_at(time_s).position, state.position)
                state = self.step(state, time_s, nxt-time_s, held,
                    left_end=boundary is not None and abs(nxt-boundary) < 1e-10)
                relative_end = sub(target_at(nxt).position, state.position)
                relative_delta = sub(relative_end, relative_start)
                rr = dot(relative_delta, relative_delta)
                closest_fraction = clamp(-dot(relative_start, relative_delta)/rr, 0., 1.) if rr else 0.
                closest = norm(add(relative_start, scale(relative_delta, closest_fraction)))
                if fuse > 0 and closest <= fuse and rr > 0:
                    b = 2*dot(relative_start, relative_delta)
                    discriminant = max(0., b*b-4*rr*(dot(relative_start,relative_start)-fuse*fuse))
                    fraction = clamp((-b-math.sqrt(discriminant))/(2*rr), 0., 1.)
                    time_s = previous_time+fraction*(nxt-previous_time)
                    state = self.step(previous_state, previous_time, time_s-previous_time, held)
                    # The detector's segment reaches the sphere at its first
                    # root; the segment after that boundary is not flown.
                    minimum_range = min(minimum_range, fuse)
                    terminal = "proximity_fuse"
                    break
                minimum_range = min(minimum_range, closest)
                time_s = nxt
                if boundary is not None and abs(time_s-boundary) < 1e-10:
                    boundary = self.propulsion.next_boundary_after(time_s)
            if abs(time_s-next_tick) < 1e-9:
                tick += 1
        return {"samples": samples, "summary": {"flight_time_s": time_s,
            "termination_event": terminal, "minimum_range_m": minimum_range,
            "maximum_alpha_deg": math.degrees(max_alpha), "first_active_s": first_active,
            "controller_updates": len(ticks), "creation_state_only": True},
            "ticks": ticks, "model": {"model_label": "single_surface_free_game_candidate",
                "force_basis": self.force_basis, "controller_hz": self.controller_hz,
                "plant_substeps": self.plant_substeps,
                "loft_interpretation": "existing_simple_pitch_error_with_raw_rate_cap_hypothesis",
                "loft_cap_unit_hypothesis": self.loft_cap_unit,
                "flight_gain_placement": "total_PID_output_at_interval_end",
                "reference_couple_identified": False,
                "midcourse_timer_or_direct_fin_routing": False,
                "missile_observation_forcing": False}}


class RateAngleLoftGame(SingleSurfaceGame):
    """Experimental four-field loft calibration on the unchanged free plant.

    Raw endpoints are amplitudes at the G request interface. The rate field's
    deg/s scale, one-way unloading and body-relative angle feedback are D-level
    hypotheses. The two positive coefficients are replay calibrations, not
    native game constants. Only controller ticks commit guidance memory.
    """

    def __init__(self, config, *, amplitude_coefficient=.75,
                 transition_coefficient=1.3, **kwargs):
        super().__init__(config, **kwargs)
        if not all(math.isfinite(x) and x > 0 for x in
                   (amplitude_coefficient, transition_coefficient)):
            raise ValueError("loft calibration coefficients must be positive")
        self.amplitude_coefficient = amplitude_coefficient
        self.transition_coefficient = transition_coefficient
        self.initial_range = None
        self.minimum_weight = 1.
        self.request_records = {}

    def initialize(self, state):
        self.initial_range = None
        self.minimum_weight = 1.
        self.request_records = {}
        return super().initialize(state)

    def request(self, state, target, time_s, evaluation):
        cfg = self.config["guidance"]
        if not cfg.get("lofting_enabled", False):
            request, details = super().request(state, target, time_s, evaluation)
            details = dict(details, wind_drag_n=evaluation.drag_n)
        else:
            rel = sub(target.position, state.position)
            rel_v = sub(target.velocity, state.velocity)
            pn, closing, los_rate = pn_acceleration(rel, rel_v, state.velocity, float(cfg["pn_gain"]))
            distance = norm(rel)
            is_tick = abs(time_s*self.controller_hz-round(time_s*self.controller_hz)) < 1e-8
            initial_range = self.initial_range if self.initial_range is not None else distance
            if is_tick and self.initial_range is None:
                self.initial_range = distance
            tgo = estimate_time_to_go_s(distance, closing, rel_v, self.config)
            hit_gain = interpolate_table(tgo, cfg["time_to_hit_gain_table"])
            pn = scale(pn, hit_gain)
            threshold = self.transition_coefficient*math.radians(abs(float(cfg["loft_omega_max_deg_s"])))
            rate = norm(los_rate)
            multiplier = float(cfg["angle_to_acceleration_multiplier"])
            z = clamp(multiplier*(rate/max(threshold, 1e-12)-1.), -60., 60.)
            current_weight = 1./(1.+math.exp(z))
            weight = min(self.minimum_weight, current_weight)
            if is_tick:
                self.minimum_weight = weight
            xi = clamp(distance/initial_range, 0., 1.)
            axes = body_axes_for_state(state)
            eta = math.atan2(-dot(rel, axes.up), dot(rel, axes.forward))
            positive_endpoint = float(cfg["lofting_elevation_deg"])-multiplier*max(eta, 0.)
            loft_g = self.amplitude_coefficient*xi**2*(positive_endpoint*weight+
                float(cfg["target_elevation_deg"])*(1.-weight))
            tangent = evaluation.wind_basis.forward
            maximum = float(cfg["maximum_lateral_acceleration_g"])*self.gravity
            commanded = add(pn, (0., loft_g*self.gravity, 0.))
            commanded = clamp_norm(sub(commanded, scale(tangent, dot(commanded, tangent))), maximum)
            required = clamp_norm(add(commanded, (0., self.gravity, 0.)), maximum)
            request = (dot(required, evaluation.wind_basis.up)/G0_MPS2,
                       dot(required, evaluation.wind_basis.right)/G0_MPS2)
            details = {"range_m": distance, "tgo_s": tgo, "los_rate_rad_s": los_rate,
                "pn_mps2": pn, "loft_g": loft_g, "loft_active": True, "hit_gain": hit_gain,
                "loft_weight": weight, "loft_current_weight": current_weight,
                "loft_los_rate_magnitude_rad_s": rate, "loft_rate_threshold_rad_s": threshold,
                "loft_range_fraction": xi, "loft_relative_body_angle_rad": eta,
                "wind_drag_n": evaluation.drag_n}
        # Recording output does not feed the plant or commit guidance memory.
        self.request_records[time_s] = {"request_g": request, **details}
        return request, details

    def run(self, initial_state, target_at, *, end_time_s):
        result = super().run(initial_state, target_at, end_time_s=end_time_s)
        result["model"].update(loft_family="rate_angle_latch",
            loft_coefficients=[self.amplitude_coefficient, self.transition_coefficient],
            loft_interpretation="empirical_raw_endpoints_with_target_LOS_rate_transition",
            loft_cap_unit_hypothesis=None, native_game_law_identified=False,
            raw_loft_semantics_evidence_grade="D")
        return result
