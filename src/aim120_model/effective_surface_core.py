"""Opt-in effective model of one commercial-game unit, calibrated to reference.

No historical profile, flight observations or fitted game coefficients enter
this module. Its inputs are launch state, raw game metadata and the separate
reference parameter file. The scalar control is an unidentified exported game
channel, not a physical fin angle. One equivalent surface and a moment couple
at the same station represent the game's effective force and angle response.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .atmosphere import StandardAtmosphere

G = 9.80665
V3 = tuple[float, float, float]
Q4 = tuple[float, float, float, float]


def add(a: V3, b: V3) -> V3:
    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])


def scale(a: V3, s: float) -> V3:
    return (a[0]*s, a[1]*s, a[2]*s)


def sub(a: V3, b: V3) -> V3:
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])


def dot(a: V3, b: V3) -> float:
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]


def cross(a: V3, b: V3) -> V3:
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def norm(a: V3) -> float:
    return math.sqrt(dot(a, a))


def unit(a: V3) -> V3:
    n = norm(a)
    if n <= 1e-12:
        raise ValueError("Direction is undefined for zero vector")
    return scale(a, 1/n)


def clamp(x: float, lower: float, upper: float) -> float:
    return min(max(x, lower), upper)


def axes(q: Q4) -> tuple[V3, V3, V3]:
    x, y, z, w = q
    return ((1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y)),
            (2*(x*y-w*z), 1-2*(x*x+z*z), 2*(y*z+w*x)),
            (2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)))


def world(b: V3, basis: tuple[V3, V3, V3]) -> V3:
    return add(add(scale(basis[0], b[0]), scale(basis[1], b[1])), scale(basis[2], b[2]))


def body(a: V3, basis: tuple[V3, V3, V3]) -> V3:
    return tuple(dot(a, b) for b in basis)


def qdot(q: Q4, w: V3) -> Q4:
    x, y, z, s = q
    a, b, c = w
    return (.5*(s*a+y*c-z*b), .5*(s*b+z*a-x*c),
            .5*(s*c+x*b-y*a), -.5*(x*a+y*b+z*c))


def normalized(q: Q4) -> Q4:
    n = math.sqrt(sum(a*a for a in q))
    if n <= 1e-12:
        raise ValueError("Zero quaternion")
    return tuple(a/n for a in q)


def table(points: list[list[float]], x: float) -> float:
    if x <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x <= x1:
            return y0+(y1-y0)*(x-x0)/(x1-x0)
    return points[-1][1]


def read_raw(path: Path) -> dict[str, Any]:
    """Read rocket metadata without using profile adapters or snapshots."""
    rocket = json.loads(path.read_text(encoding="utf-8"))["rocket"]
    g = rocket["guidance"]["guidanceAutopilot"]
    expected = {"accelControlProp": .002125, "accelControlIntg": .006481,
                "accelControlDiff": .000023, "accelControlIntgLim": 1.,
                "loftElevation": 24., "loftTargetElevation": -3.5,
                "loftAngleToAccelMult": 17.5, "loftTargetOmegaMax": .5}
    if any(g[k] != v for k, v in expected.items()) or rocket["distFromCmToStab"] != .4:
        raise ValueError("Raw metadata differs from the declared calibration boundary")
    return rocket


class EffectiveSurfaceGame:
    """Launch-only forward simulator; no access to observed future values."""

    def __init__(self, raw: dict[str, Any], config: dict[str, Any], creation: dict[str, Any]):
        self.raw, self.config, self.creation = raw, config, creation
        self.loft_pitch_target_convention = config.get("loft_pitch_target_convention", "plus_los")
        if self.loft_pitch_target_convention not in ("plus_los", "absolute"):
            raise ValueError(f"Unknown loft pitch target convention: {self.loft_pitch_target_convention}")
        self.p = config["effective_parameters"]
        self.autopilot = raw["guidance"]["guidanceAutopilot"]
        self.atmosphere = StandardAtmosphere()
        self.area = math.pi*raw["caliber"]**2/4
        self.ell = raw["distFromCmToStab"]
        self.v0 = norm(tuple(creation["velocity_xyz_mps"]))
        self.m0 = creation["mass_source_kg"]
        a0 = self.atmosphere.sample(creation["position_xyz_m"][1])
        self.qm0 = .5*a0.density_kg_m3*self.v0**2/self.m0
        if self.m0 != raw["mass"]:
            raise ValueError("Raw and creation masses disagree")
        # The supplied launch metadata fixes the clock. PID never changes mode.
        self.dt = creation["scenario_exact"]["dt"]
        if abs(self.dt-1/48) > 1e-8:
            raise ValueError("This branch requires the declared 48 Hz controller")
        self.integration_substeps = int(config["integration_substeps"])
        self.integral = [0., 0.]
        self.previous_error = [0., 0.]
        self.control = (0., 0.)
        self.loft_phase = "uninitialized"
        self.previous_acceleration: V3 = (0., -G, 0.)
        self.last_request: V3 = (0., 0., 0.)
        self.last_error = (0., 0.)
        self.target0 = tuple(creation["target_position_xyz_m"])
        self.target_velocity = tuple(creation["target_velocity_xyz_mps"])

    def mass_thrust(self, t: float) -> tuple[float, float]:
        fraction = clamp(t/self.raw["timeFire"], 0., 1.)
        mass = self.raw["mass"]+(self.raw["massEnd"]-self.raw["mass"])*fraction
        return mass, self.raw["force"] if t < self.raw["timeFire"] else 0.

    def coefficients(self, mach: float, alpha: float) -> tuple[float, float, float]:
        c0, c1, c2 = self.p["cd_base_inverse_mach"]
        cd = c0+c1/mach+c2/mach**2
        if cd <= 0:
            raise ValueError("Effective drag law left its positive calibrated domain")
        cd_aoa = self.p["cd_incidence_ratio"]*cd*math.sin(alpha)**2
        stall = max(alpha-self.p["lift_stall_rad"], 0.)/self.p["lift_rolloff_width_rad"]
        cl = self.p["lift_sine_slope"]*math.sin(alpha)*math.exp(-stall*stall)
        return cd, cd_aoa, cl

    def plant(self, t: float, state: tuple[float, ...], control: tuple[float, float]) -> tuple[tuple[float, ...], dict[str, Any]]:
        pos, vel, q, w = state[:3], state[3:6], state[6:10], state[10:13]
        q = normalized(q)
        basis = axes(q)
        f, up, right = basis
        speed = norm(vel)
        tangent = unit(vel) if speed > 0. else f
        mass, thrust = self.mass_thrust(t)
        atmosphere = self.atmosphere.sample(pos[1])
        dynamic = .5*atmosphere.density_kg_m3*speed**2
        mach = speed/atmosphere.speed_of_sound_mps
        transverse = sub(f, scale(tangent, dot(f, tangent)))
        transverse_length = norm(transverse)
        alpha = math.atan2(transverse_length, dot(f, tangent))
        cd, cda, cl = self.coefficients(mach, alpha)
        lift = scale(transverse, dynamic*self.area*cl/transverse_length) if transverse_length > 1e-12 else (0., 0., 0.)
        drag = scale(tangent, -dynamic*self.area*(cd+cda))
        specific = scale(add(add(lift, drag), scale(f, thrust)), 1/mass)
        acceleration = add(specific, (0., -G, 0.))
        # The single station sees local velocity, including its rotation.
        station = scale(f, -self.ell)
        local_velocity = add(vel, cross(world(w, basis), station))
        local_body = body(local_velocity, basis)
        restoring_angles = self.config.get("restoring_angle_convention", "independent_planes")
        if restoring_angles == "independent_planes":
            ap = math.atan2(-local_body[1], local_body[0])
            ay = math.atan2(-local_body[2], local_body[0])
        elif restoring_angles == "total_incidence_vector":
            rho = math.hypot(local_body[1], local_body[2])
            incidence = math.atan2(rho, local_body[0])
            ap = incidence*(-local_body[1])/rho if rho > 0. else 0.
            ay = incidence*(-local_body[2])/rho if rho > 0. else 0.
        else:
            raise ValueError(f"Unknown restoring angle convention: {restoring_angles}")
        pressure_mass = (dynamic/mass)/self.qm0
        damping_scale = pressure_mass*self.v0/speed if speed > 0. else 0.
        ku, ka, kw = (self.p[k] for k in ("control_response_rad_s2", "restoring_response_rad_s2_per_rad", "damping_response_per_s"))
        wd = (-kw*damping_scale*w[0],
              -ku*pressure_mass*control[0]+ka*pressure_mass*ay-kw*damping_scale*w[1],
              ku*pressure_mass*control[1]-ka*pressure_mass*ap-kw*damping_scale*w[2])
        # This is the total effective moment, including a couple at the same
        # station. It is not asserted to be only station x exported lift.
        deriv = tuple(vel)+tuple(acceleration)+qdot(q, w)+tuple(wd)
        diagnostic = {"speed_mps": speed, "body_pitch_deg": math.degrees(math.asin(clamp(f[1], -1., 1.))),
                      "body_yaw_deg": math.degrees(math.atan2(f[2], f[0])),
                      "geometric_incidence_unsigned_deg": math.degrees(alpha),
                      "normal_specific_load_g": norm(sub(specific, scale(tangent, dot(specific, tangent))))/G,
                      "normal_trajectory_load_g": norm(sub(acceleration, scale(tangent, dot(acceleration, tangent))))/G,
                      "acceleration_xyz_mps2": acceleration, "specific_acceleration_xyz_mps2": specific,
                      "mass_kg": mass, "thrust_n": thrust, "drag_base_n": dynamic*self.area*cd,
                      "drag_aoa_n": dynamic*self.area*cda, "lift_n": norm(lift), "mach": mach}
        return deriv, diagnostic

    def guidance_kinematics(self, t: float, state: tuple[float, ...]):
        """Relative motion and PN rotation state, in runtime world axes."""
        pos, vel = state[:3], state[3:6]
        target = add(self.target0, scale(self.target_velocity, t))
        relative = sub(target, pos)
        relative_v = sub(self.target_velocity, vel)
        distance = norm(relative)
        los_omega = scale(cross(relative, relative_v), 1/distance**2)
        return relative, relative_v, los_omega

    def guidance(self, t: float, state: tuple[float, ...]) -> V3:
        vel, q = state[3:6], state[6:10]
        relative, relative_v, los_omega = self.guidance_kinematics(t, state)
        distance = norm(relative)
        rhat = unit(relative)
        closing = -dot(relative_v, rhat)
        nav = self.autopilot["propNavMult"]
        hit_points = self.autopilot.get("_hit_gain") or [self.autopilot[f"timeToHitToGain{i}"] for i in range(3)]
        tgo_convention = self.config.get("time_to_hit_convention", "relative_speed")
        if tgo_convention == "relative_speed":
            tgo = distance/norm(relative_v)
        elif tgo_convention == "linear_closing":
            # No positive closure means no finite linear intercept time. The
            # raw gain table's final value applies without an invented cutoff.
            tgo = distance/closing if closing > 0. else math.inf
        else:
            raise ValueError(f"Unknown time-to-hit convention: {tgo_convention}")
        hit_gain = table(hit_points, tgo)
        mode = self.config["pn_candidate"]
        if mode == "pure_velocity_normal":
            request = scale(cross(los_omega, vel), nav*hit_gain)
        elif mode == "true_los_normal":
            request = scale(cross(los_omega, rhat), nav*max(closing, 0.)*hit_gain)
        elif mode == "ideal_relative_velocity_normal":
            request = scale(cross(los_omega, relative_v), -nav*hit_gain)
        else:
            raise ValueError(f"Unknown complete PN candidate: {mode}")
        if self.autopilot["loftEnabled"] and self.creation["scenario_exact"]["loft_state"] == "config":
            los_elevation = math.degrees(math.atan2(relative[1], math.hypot(relative[0], relative[2])))
            velocity_elevation = math.degrees(math.atan2(vel[1], math.hypot(vel[0], vel[2])))
            # Loft gates on LOS direction motion, not rotation about that LOS.
            # The complete PN state may also contain an independent axial part.
            omega_deg_s = math.degrees(norm(cross(relative, relative_v))/distance**2)
            gate = self.config.get("loft_gate_convention", "relative_flight_path")
            if gate == "relative_flight_path":
                elevation_for_gate = los_elevation-velocity_elevation
            elif gate == "absolute_los":
                elevation_for_gate = los_elevation
            else:
                raise ValueError(f"Unknown loft gate convention: {gate}")
            eligible = (omega_deg_s <= self.autopilot["loftTargetOmegaMax"]
                        and elevation_for_gate >= self.autopilot["loftTargetElevation"])
            if self.loft_phase == "uninitialized":
                self.loft_phase = "climb" if eligible else "complete"
            if self.loft_phase == "climb" and not eligible:
                self.loft_phase = "complete"
            if self.loft_phase == "climb":
                forward = axes(normalized(q))[0]
                pitch = math.degrees(math.asin(clamp(forward[1], -1., 1.)))
                pitch_error = (self.autopilot["loftElevation"]+los_elevation-pitch
                               if self.loft_pitch_target_convention == "plus_los"
                               else self.autopilot["loftElevation"]-pitch)
                bias = self.p["loft_common_scale"]*self.autopilot["loftAngleToAccelMult"]*pitch_error
                velocity_tangent = unit(vel)
                vertical = sub((0., 1., 0.), scale(velocity_tangent, velocity_tangent[1]))
                request = add(request, scale(vertical, bias/norm(vertical)) if norm(vertical) > 0. else (0., 0., 0.))
        return self.limit_acceleration_request(request, state)

    def limit_acceleration_request(self, request: V3, state: tuple[float, ...]) -> V3:
        """Limit the requested acceleration in the selected feedback plane."""
        if self.config.get("pid_error_frame", "body") == "velocity_normal_projected_body":
            # Discard speed-change demand before it consumes the turn budget.
            # update_control compares feedback in this same velocity plane,
            # then resolves the remaining error onto the two body axes.
            tangent = unit(state[3:6])
            request = sub(request, scale(tangent, dot(request, tangent)))
        maximum = self.autopilot["reqAccelMax"]*G
        length = norm(request)
        return scale(request, maximum/length) if length > maximum else request

    def update_control(self, t: float, state: tuple[float, ...]) -> tuple[float, float]:
        points = self.autopilot.get("_age_gain") or [self.autopilot[f"timeToGain{i}"] for i in range(2)]
        age_gain = table(points, t)
        if age_gain == 0.:
            self.last_request = (0., 0., 0.)
            self.last_error = (0., 0.)
            return (0., 0.)
        request = self.guidance(t, state)
        self.last_request = request
        basis = axes(normalized(state[6:10]))
        difference = sub(request, self.previous_acceleration)
        error_frame = self.config.get("pid_error_frame", "body")
        if error_frame == "velocity_normal_projected_body":
            tangent = unit(state[3:6])
            difference = sub(difference, scale(tangent, dot(difference, tangent)))
        elif error_frame != "body":
            raise ValueError(f"Unknown PID error frame: {error_frame}")
        divisor = G if self.config["pid_error_units"] == "G" else 1.
        if self.config["pid_error_units"] not in ("G", "m/s^2"):
            raise ValueError("Unrecognized PID error units")
        error = [dot(difference, basis[2])/divisor, dot(difference, basis[1])/divisor]
        self.last_error = tuple(error)
        speed = norm(state[3:6])
        density = self.atmosphere.sample(state[1]).density_kg_m3
        indicated_speed_kmh = speed*3.6*math.sqrt(density/1.225)
        speed_gain = (self.autopilot["baseIndSpeed"]/indicated_speed_kmh)**2
        pressure_gain = self.config.get("pid_pressure_gain_convention", "inverse_square")
        if pressure_gain == "floor_one":
            speed_gain = max(1., speed_gain)
        elif pressure_gain != "inverse_square":
            raise ValueError(f"Unknown PID pressure gain convention: {pressure_gain}")
        kp, ki, kd, limit = (self.autopilot[k] for k in ("accelControlProp", "accelControlIntg", "accelControlDiff", "accelControlIntgLim"))
        gain_convention = self.config.get("pid_gain_convention", "post_gain")
        output = []
        for j in range(2):
            if gain_convention == "post_gain":
                self.integral[j] = clamp(self.integral[j]+ki*error[j]*self.dt, -limit, limit)
                raw = kp*error[j]+self.integral[j]+kd*(error[j]-self.previous_error[j])/self.dt
                output.append(clamp(age_gain*speed_gain*raw, -1., 1.))
            elif gain_convention == "scheduled_terms":
                self.integral[j] = clamp(self.integral[j]+speed_gain*ki*error[j]*self.dt, -limit, limit)
                pd = kp*error[j]+kd*(error[j]-self.previous_error[j])/self.dt
                output.append(clamp(age_gain*(speed_gain*pd+self.integral[j]), -1., 1.))
            elif gain_convention == "conditional_integration":
                increment = ki*error[j]*self.dt
                trial = clamp(self.integral[j]+increment, -limit, limit)
                pd = kp*error[j]+kd*(error[j]-self.previous_error[j])/self.dt
                attempted = age_gain*speed_gain*(pd+trial)
                pushes_saturation = (attempted > 1. and increment > 0.) or (attempted < -1. and increment < 0.)
                if not pushes_saturation:
                    self.integral[j] = trial
                output.append(clamp(age_gain*speed_gain*(pd+self.integral[j]), -1., 1.))
            else:
                raise ValueError(f"Unknown ordinary PID gain convention: {gain_convention}")
            self.previous_error[j] = error[j]
        return tuple(output)

    def step(self, t: float, state: tuple[float, ...], control: tuple[float, float], dt: float) -> tuple[float, ...]:
        # Midpoint integration keeps the controller's 48 Hz clock fixed.
        k1, _ = self.plant(t, state, control)
        mid = tuple(a+.5*dt*b for a, b in zip(state, k1))
        k2, _ = self.plant(t+.5*dt, mid, control)
        new = tuple(a+dt*b for a, b in zip(state, k2))
        new = new[:6]+normalized(new[6:10])+new[10:]
        if not all(math.isfinite(a) for a in new):
            raise ValueError("Nonfinite forward state")
        return new

    def run(self, duration_s: float | None = None) -> list[dict[str, Any]]:
        duration = self.config["forward_duration_s"] if duration_s is None else duration_s
        state = (tuple(self.creation["position_xyz_m"])+tuple(self.creation["velocity_xyz_mps"])
                 +normalized(tuple(self.creation["quaternion_xyzw_raw"]))
                 +tuple(-a for a in self.creation["omega_export_raw"]))
        rows = []
        count = math.ceil(duration/self.dt)
        for tick in range(count+1):
            t = tick*self.dt
            _, diagnostic = self.plant(t, state, self.control)
            rows.append({"t": t, "position_xyz_m": state[:3], "velocity_xyz_mps": state[3:6],
                         "quaternion_xyzw": state[6:10], "body_omega_xyz_rad_s": state[10:],
                         "controls_raw": self.control, "request_xyz_mps2": self.last_request,
                         "pid_error": self.last_error, "pid_integral_output": tuple(self.integral),
                         "loft_phase": self.loft_phase, **diagnostic})
            if tick == count:
                break
            self.control = self.update_control(t, state)
            h = self.dt/self.integration_substeps
            for j in range(self.integration_substeps):
                state = self.step(t+j*h, state, self.control, h)
            _, completed = self.plant(t+self.dt, state, self.control)
            self.previous_acceleration = completed["acceleration_xyz_mps2"]
        return rows


def forward_from_files(creation_path: Path, config_path: Path, raw_path: Path, case_id: str) -> list[dict[str, Any]]:
    """Three-file runtime entry point. It cannot read the observation archive."""
    inputs = json.loads(creation_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    creation = inputs["cases"][case_id]
    return EffectiveSurfaceGame(read_raw(raw_path), config, creation).run()
