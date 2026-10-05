"""One bounded R-77-1 replay experiment: PID output drives effective pitch moment.

The velocity trace and target trajectory are external inputs. Missile position,
flight path, attitude, pitch rate and PID memory evolve from their release
states. This script does not alter the production model or its PID parameters.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aim120_model.control import update_control_feedback  # noqa: E402
from aim120_model.guidance import interpolate_table, pn_acceleration  # noqa: E402

SOURCE = ROOT / "outputs/r771_replay_20260927_h110/source_payload.json"
OUT = ROOT / "outputs/r771_replay_20260927_h110/effective_moment"
G = 9.80665
CONTROL_DT = .02
PLANT_DT = .005
HORIZON = 3.56
BOUNDS = np.array([[1.0, 10000.0], [0.0, 300.0], [0.0, 100.0]])
FIT_BOUNDS = (.30, 1.35)
HOLD_BOUNDS = (1.45, 2.45)
EVENT_WINDOWS = ((.35, .60, "max"), (.60, .80, "min"),
                 (.80, 1.05, "max"), (1.10, 1.35, "min"))


def signed16(raw):
    return np.where(raw >= 32768, raw - 65536, raw)


def isa_density(height):
    temp = 288.15 - .0065 * height
    pressure = 101325.0 * (temp / 288.15) ** 5.2558797
    return pressure / (287.05 * temp)


def clamp(value, low, high):
    return min(high, max(low, value))


def raw_parameters(payload):
    rocket = payload["optional_existing_parent_parameters"]["rocket"]
    auto = rocket["guidance"]["guidanceAutopilot"]
    return {
        "source_version": payload["optional_existing_parent_parameters"]["source_version"],
        "finsLatAccel_custom": 110.0,
        "finsLatAccel_parent": float(rocket["finsLatAccel"]),
        "mass0_kg": float(rocket["mass"]),
        "mass_end_kg": float(rocket["massEnd"]),
        "time_fire_s": float(rocket["timeFire"]),
        "thrust_n": float(rocket["force"]),
        "caliber_m": float(rocket["caliber"]),
        "length_m": float(rocket["length"]),
        "wing_area_mult": float(rocket["wingAreaMult"]),
        "raw_load_factor_max_g": float(rocket["loadFactorMax"]),
        "fin_aoa_rad": float(rocket["finsAoaHor"]),
        "tail_arm_m": float(rocket["distFromCmToStab"]),
        "kp": float(auto["accelControlProp"]),
        "ki": float(auto["accelControlIntg"]),
        "kd": float(auto["accelControlDiff"]),
        "integral_limit": float(auto["accelControlIntgLim"]),
        "pn_gain": float(auto["propNavMult"]),
        "req_accel_max_g": float(auto["reqAccelMax"]),
        "loft_enabled": bool(auto["loftEnabled"]),
        "loft_elevation_rad": math.radians(float(auto["loftElevation"])),
        "loft_target_elevation_deg": float(auto["loftTargetElevation"]),
        "loft_angle_to_accel_mult": float(auto["loftAngleToAccelMult"]),
        "loft_omega_max_deg_s": float(auto["loftTargetOmegaMax"]),
        "base_ind_speed_kmh": float(auto["baseIndSpeed"]),
        "time_gain": [[float(auto[k][0]), float(auto[k][1])]
                      for k in ("timeToGain0", "timeToGain1")],
        "hit_gain": [[float(auto[k][0]), float(auto[k][1])]
                     for k in ("timeToHitToGain0", "timeToHitToGain1", "timeToHitToGain2")],
    }


def load_replay():
    payload = json.loads(SOURCE.read_text())
    columns = payload["missile_columns"]
    a = np.asarray(payload["missile_rows"], float)
    target = np.asarray(payload["target_rows"], float)
    assert a.shape == (165, 18) and target.shape == (51, 7)
    assert np.isfinite(a).all() and np.isfinite(target).all()
    col = lambda name: a[:, columns.index(name)]
    t = (col("replay_time_ms") - col("replay_time_ms")[0]) / 1000
    t_abs = col("replay_time_ms") / 1000
    assert np.all(np.diff(t) > 0)
    velocity3 = signed16(np.stack([col(f"raw_b7_{axis}_raw_u16") for axis in "xyz"], axis=1)) * (4000/65535)
    speed2 = np.hypot(velocity3[:, 0], velocity3[:, 1])
    gamma = np.arctan2(velocity3[:, 1], velocity3[:, 0])
    theta = col("raw_euler_z")
    alpha = theta-gamma
    omega = -signed16(col("raw_b8_z_raw_u16")) * (30/65535)
    target_t = target[:, 0] / 1000
    target_velocity = np.gradient(target[:, 1:3], target_t, axis=0)
    planar_speed_gap = np.linalg.norm(velocity3, axis=1) - speed2
    out_of_plane = np.degrees(np.arctan2(np.abs(velocity3[:, 2]), speed2))
    return {"payload": payload, "t": t, "t_abs": t_abs,
            "missile_packet0_s": float(t_abs[0]), "x": col("x_m"), "y": col("y_m"),
            "theta": theta, "gamma": gamma, "alpha": alpha, "omega": omega,
            "speed2": speed2, "velocity3": velocity3,
            "target_t": target_t, "target_xy": target[:, 1:3], "target_vxy": target_velocity,
            "max_planar_speed_gap_mps": float(np.max(planar_speed_gap)),
            "max_out_of_plane_velocity_angle_deg": float(np.max(out_of_plane)),
            "max_abs_missile_z_displacement_m": float(np.max(np.abs(col("z_m")-col("z_m")[0])))}


def target_at(replay, t):
    stamp = replay["missile_packet0_s"] + t
    ts = replay["target_t"]
    xy = np.array([np.interp(stamp, ts, replay["target_xy"][:, j]) for j in range(2)])
    vxy = np.array([np.interp(stamp, ts, replay["target_vxy"][:, j]) for j in range(2)])
    return xy, vxy


def mass_thrust(t, p):
    mass = p["mass0_kg"]-(p["mass0_kg"]-p["mass_end_kg"])*clamp(t/p["time_fire_s"],0,1)
    return mass, p["thrust_n"] if t < p["time_fire_s"] else 0.0


def force_terms(t, state, replay, p):
    theta, omega, gamma, x, y = state
    v = float(np.interp(t, replay["t"], replay["speed2"]))
    mass, thrust = mass_thrust(t, p)
    alpha = theta-gamma
    area = math.pi*p["caliber_m"]**2/4*p["wing_area_mult"]
    qbar = .5*isa_density(y)*v*v
    aero_normal_n = qbar*area*14.0*alpha
    thrust_normal_n = thrust*math.sin(alpha)
    feedback_g = (aero_normal_n+thrust_normal_n)/(mass*G)
    return {"v": v, "mass": mass, "thrust": thrust, "alpha": alpha,
            "aero_normal_n": aero_normal_n, "thrust_normal_n": thrust_normal_n,
            "actual_specific_force_g": feedback_g}


def request(t, state, replay, p, loft_armed=True, force_geometric_active=None):
    theta, omega, gamma, x, y = state
    terms = force_terms(t, state, replay, p)
    v = terms["v"]
    target_xy, target_vxy = target_at(replay, t)
    r = target_xy-np.array([x,y])
    missile_v = np.array([v*math.cos(gamma), v*math.sin(gamma)])
    rel_v = target_vxy-missile_v
    pn_vec, closing, los_rate = pn_acceleration(
        (float(r[0]),float(r[1]),0.0),
        (float(rel_v[0]),float(rel_v[1]),0.0),
        (float(missile_v[0]),float(missile_v[1]),0.0),p["pn_gain"])
    distance = float(np.linalg.norm(r))
    los_elevation = math.atan2(float(r[1]),float(r[0]))
    geometric_mode=p.get("geometric_loft_mode")
    geometry_allows=(los_elevation>math.radians(p["loft_target_elevation_deg"])
                     and abs(los_rate[2])<math.radians(p["loft_omega_max_deg_s"]))
    if geometric_mode:
        loft_active=bool(p["loft_enabled"] and loft_armed and geometry_allows)
        if force_geometric_active is not None:
            loft_active=bool(force_geometric_active)
    else:
        loft_active=bool(p["loft_enabled"])
    tgo = distance/max(closing, float(np.linalg.norm(rel_v)), 1e-9)
    flight_gain = interpolate_table(t,p["time_gain"])
    hit_gain = interpolate_table(tgo,p["hit_gain"])
    normal = np.array([-math.sin(gamma), math.cos(gamma)])
    schedule_gate = p.get("schedule_gate") == "total_pid_output"
    pn_normal_mps2 = float(np.dot(pn_vec[:2], normal))*hit_gain
    if not schedule_gate:
        pn_normal_mps2 *= flight_gain  # Preserve the earlier PN-only baseline.
    loft_reference = p.get("loft_reference", "body")
    if loft_reference == "body":
        loft_angle = theta
        loft_target = p["loft_elevation_rad"]
    elif loft_reference == "flight_path":
        loft_angle = gamma
        loft_target = p["loft_elevation_rad"]
    elif loft_reference == "los_relative_flight_path":
        loft_angle = gamma
        loft_target = math.atan2(float(r[1]),float(r[0]))+p["loft_elevation_rad"]
    else:
        raise ValueError(f"unknown loft reference {loft_reference}")
    loft_raw_g = (p["loft_angle_to_accel_mult"]*(loft_target-loft_angle)
                  if loft_active else 0.0)
    if geometric_mode:
        # In this candidate 0.5 is a target-LOS-rate gate in deg/s, not a
        # missile turn-rate / acceleration cap.
        loft_limited_g=loft_raw_g
    else:
        omega_limit = (p["loft_omega_max_deg_s"]
                       if p.get("loft_omega_unit") == "rad_per_s"
                       else math.radians(p["loft_omega_max_deg_s"]))
        loft_cap_g = omega_limit*v/G
        loft_limited_g = clamp(loft_raw_g,-loft_cap_g,loft_cap_g)
    loft_normal_mps2 = loft_limited_g*G*normal[1]
    max_accel = p["req_accel_max_g"]*G
    pn_for_command=(0.0 if geometric_mode=="exclusive_vertical" and loft_active
                    else pn_normal_mps2)
    command_normal = clamp(pn_for_command+loft_normal_mps2,-max_accel,max_accel)
    commanded = command_normal*normal
    required_vector = commanded+np.array([0.0,G])
    req_norm = float(np.linalg.norm(required_vector))
    if req_norm>max_accel:
        required_vector *= max_accel/req_norm
    required_normal = clamp(float(np.dot(required_vector,normal)),-max_accel,max_accel)
    return {"a_req_g": required_normal/G,
            "pn_g": pn_normal_mps2/G,"pn_for_command_g":pn_for_command/G,
            "loft_raw_g": loft_raw_g,
            "flight_gain":flight_gain,"loft_reference_rad":loft_angle,
            "loft_target_rad":loft_target,
            "loft_active":loft_active,"geometry_allows_loft":geometry_allows,
            "los_elevation_deg":math.degrees(los_elevation),
            "los_rate_deg_s":math.degrees(los_rate[2]),
            "loft_limited_g": loft_limited_g,
            "loft_projected_g": loft_normal_mps2/G,
            "gravity_compensation_g": math.cos(gamma),
            "actual_g": terms["actual_specific_force_g"],
            "error_g": required_normal/G-terms["actual_specific_force_g"],
            "range_m": distance, "tgo_s": tgo, "closing_mps": closing,
            "los_rate_z_rad_s": los_rate[2], "flight_gain": flight_gain,
            "hit_gain": hit_gain}


def pid_step(error_g, state, p):
    integral, previous_error, previous_derivative = state
    dt = CONTROL_DT
    provisional = clamp(integral+p["ki"]*error_g*dt,
                        -p["integral_limit"],p["integral_limit"])
    raw_derivative = (error_g-previous_error)/dt
    filter_weight = min(1.0,dt/(1e-9+dt))  # control.py's zero-tau numeric floor
    derivative = previous_derivative+filter_weight*(raw_derivative-previous_derivative)
    output = p["kp"]*error_g+provisional+p["kd"]*derivative
    u = clamp(output,-p["fin_aoa_rad"],p["fin_aoa_rad"])
    if abs(u)>=.99*p["fin_aoa_rad"] and error_g*u>0:
        next_integral = integral  # hold memory for next tick, as control.py
    else:
        next_integral = provisional
    return u,(next_integral,error_g,derivative),{"raw_output_rad":output,
                                      "integral_term_rad":next_integral,
                                      "derivative_g_s":derivative}


def scheduled_pid_step(t, req, state, p, was_active):
    """Candidate: gate the total active fin request, leaving passive physics live."""
    gain=req["flight_gain"]
    if gain<=0:
        # No integration or error history accumulates before arming.
        return 0.0,state,{"raw_output_rad":0.0,"integral_term_rad":state[0],
                          "derivative_g_s":0.0},False
    if not was_active:
        # Initialize previous error at the first active sample to avoid an
        # artificial D kick from a controller that was dormant at launch.
        state=(state[0],req["error_g"],0.0)
    raw_u,next_state,diag=pid_step(req["error_g"],state,p)
    return gain*raw_u,next_state,diag,True


def control_interface_comparison(p):
    """Targeted same-input comparison to the production PID, including clamp."""
    cfg={"control":{"pid":{"p":p["kp"],"i":p["ki"],"d":p["kd"],"integral_limit":p["integral_limit"]},
                    "derivative_filter_time_constant_s":0.0,"feedback_measurement":"cg_wind_normal_specific_force_g",
                    "pid_error_units":"g","pid_error_scale":1.0,"pid_output_semantics":"fin_angle_rad",
                    "integral_limit_semantics":"term","plant_semantics":"rebuild_first_principles",
                    "actuator_time_constant_s":0.0},
         "aerodynamics":{"horizontal_fin_aoa_limit_deg":math.degrees(p["fin_aoa_rad"]),
                         "vertical_fin_aoa_limit_deg":math.degrees(p["fin_aoa_rad"])},
         "atmosphere":{"gravity_mps2":G}}
    state=SimpleNamespace(pitch_pid_integral=0.0,yaw_pid_integral=0.0,
        previous_pitch_error=0.0,previous_yaw_error=0.0,
        pitch_error_derivative=0.0,yaw_error_derivative=0.0,
        actual_pitch_acceleration_g=0.0,actual_yaw_acceleration_g=0.0,
        actual_pitch_fin_angle_rad=0.0,actual_yaw_fin_angle_rad=0.0,
        measured_pitch_normal_g=0.0,measured_yaw_normal_g=0.0)
    local=(0.0,0.0,0.0)
    largest=0.0
    # The large error deliberately exercises output clipping and anti-windup.
    inputs=((1.2,.2),(10.,1.),(-6.,2.),(500.,0.),(500.,0.),(-1.,2.))
    for n,(command,actual) in enumerate(inputs):
        state.measured_pitch_normal_g=actual
        updates=update_control_feedback(state,(command,0.0),cfg,CONTROL_DT,True,time_s=n*CONTROL_DT)
        u,local,_=pid_step(command-actual,local,p)
        largest=max(largest,abs(u-updates["actual_pitch_fin_angle_rad"]),
                    abs(local[0]-updates["pitch_pid_integral"]),
                    abs(local[1]-updates["previous_pitch_error"]),
                    abs(local[2]-updates["pitch_error_derivative"]))
        for key,value in updates.items():
            setattr(state,key,value)
    assert largest<1e-9
    return {"same_input_steps":len(inputs),"max_u_integral_error_difference":largest,
            "includes_saturation_and_antiwindup":True}


def simulate(coeff,replay,p,plant_dt=PLANT_DT):
    ku,ka,kw=map(float,coeff)
    state=np.array([replay["theta"][0],replay["omega"][0],replay["gamma"][0],
                    replay["x"][0],replay["y"][0]],float)
    pid_state=(0.0,0.0,0.0)
    pid_active=False
    loft_armed=True
    ticks=int(round(HORIZON/CONTROL_DT))
    substeps=int(round(CONTROL_DT/plant_dt))
    assert abs(substeps*plant_dt-CONTROL_DT)<1e-12
    rows=[]

    def derivative(t,values,u):
        theta,omega,gamma,x,y=values
        f=force_terms(t,values,replay,p)
        v=f["v"]; mass=f["mass"]
        gamma_dot=((f["aero_normal_n"]+f["thrust_normal_n"])/(mass*v)
                   -G*math.cos(gamma)/v)
        return np.array([omega,ku*u-ka*(theta-gamma)-kw*omega,
                         gamma_dot,v*math.cos(gamma),v*math.sin(gamma)])

    for tick in range(ticks+1):
        t=tick*CONTROL_DT
        if not np.isfinite(state).all() or abs(state[0]-state[2])>1.25 or abs(state[2])>1.0:
            return None
        req=request(t,state,replay,p,loft_armed=loft_armed)
        if p.get("geometric_loft_mode"):
            loft_armed=bool(req["loft_active"])
        if p.get("schedule_gate")=="total_pid_output":
            u,pid_state,pid_diag,pid_active=scheduled_pid_step(t,req,pid_state,p,pid_active)
        else:
            u,pid_state,pid_diag=pid_step(req["error_g"],pid_state,p)
        theta,omega,gamma,x,y=state
        rows.append({"t":t,"theta":theta,"omega":omega,"gamma":gamma,
                     "alpha":theta-gamma,"x":x,"y":y,"u":u,
                     "flight_gain":req["flight_gain"],
                     "loft_active":req["loft_active"],
                     "los_elevation_deg":req["los_elevation_deg"],
                     "los_rate_deg_s":req["los_rate_deg_s"],
                     "pn_for_command_g":req["pn_for_command_g"],
                     "accel_u":ku*u,"accel_alpha":-ka*(theta-gamma),
                     "accel_damp":-kw*omega,"actual_g":req["actual_g"],
                     "a_req_g":req["a_req_g"],"pn_g":req["pn_g"],
                     "loft_raw_g":req["loft_raw_g"],
                     "loft_limited_g":req["loft_limited_g"],
                     "loft_projected_g":req["loft_projected_g"],
                     "gravity_compensation_g":req["gravity_compensation_g"],
                     "range_m":req["range_m"],"pid_error_g":req["error_g"],
                     "integral_term_rad":pid_diag["integral_term_rad"],
                     "raw_pid_output_rad":pid_diag["raw_output_rad"]})
        if tick==ticks:
            break
        h=CONTROL_DT/substeps
        for sub in range(substeps):
            ts=t+sub*h
            k1=derivative(ts,state,u)
            k2=derivative(ts+h/2,state+h*k1/2,u)
            k3=derivative(ts+h/2,state+h*k2/2,u)
            k4=derivative(ts+h,state+h*k3,u)
            state=state+h*(k1+2*k2+2*k3+k4)/6
    return rows


def series(rows,key,t):
    return np.interp(t,[r["t"] for r in rows],[r[key] for r in rows])


def event_features(t,omega,windows=EVENT_WINDOWS):
    out=[]
    for low,high,kind in windows:
        ids=np.flatnonzero((t>=low)&(t<=high))
        index=int(ids[np.argmax(omega[ids]) if kind=="max" else np.argmin(omega[ids])])
        out.append({"kind":kind,"window_s":[low,high],"time_s":float(t[index]),
                    "omega_rad_s":float(omega[index])})
    return out


def objective(coeff,replay,p,return_detail=False,plant_dt=PLANT_DT):
    rows=simulate(coeff,replay,p,plant_dt)
    if rows is None:
        return (1e6,None) if return_detail else 1e6
    t=replay["t"]
    fit=(t>=FIT_BOUNDS[0])&(t<=FIT_BOUNDS[1])
    pred={key:series(rows,key,t) for key in ("theta","alpha","omega","gamma","y")}
    # All three attitude variables enter, with fixed scales set before fitting.
    loss=np.mean(((pred["theta"][fit]-replay["theta"][fit])/math.radians(3))**2
                 +((pred["alpha"][fit]-replay["alpha"][fit])/math.radians(3))**2
                 +((pred["omega"][fit]-replay["omega"][fit])/.5)**2)
    obs_events=event_features(t,replay["omega"])
    pred_events=event_features(t,pred["omega"])
    event_loss=np.mean([((q["time_s"]-o["time_s"])/.08)**2
                        +((q["omega_rad_s"]-o["omega_rad_s"])/.5)**2
                        for o,q in zip(obs_events,pred_events)])
    total=float(loss+event_loss)
    if return_detail:
        return total,{"rows":rows,"pred":pred,"state_loss":float(loss),
                      "event_loss":float(event_loss),
                      "obs_events":obs_events,"pred_events":pred_events}
    return total


def optimize(replay,p):
    # Bounded deterministic pattern search; no guide or PID parameters move.
    seeds=((100.,10.,5.),(500.,50.,10.),(1500.,100.,20.),(4000.,30.,30.))
    ranges=np.array([math.log10(BOUNDS[0,1]),BOUNDS[1,1],BOUNDS[2,1]])
    def decode(z):
        z=np.clip(z,0,1)
        return np.array([10**(4*z[0]),300*z[1],100*z[2]])
    def encode(c):
        return np.array([math.log10(c[0])/4,c[1]/300,c[2]/100])
    cache={}
    def score(z):
        key=tuple(np.round(np.clip(z,0,1),7))
        if key not in cache:
            cache[key]=objective(decode(np.array(key)),replay,p)
        return cache[key]
    candidates=[]
    for seed in seeds:
        z=encode(seed)
        value=score(z)
        step=np.array([.22,.15,.12])
        iterations=0
        while iterations<45 and float(np.max(step))>0.006:
            iterations+=1
            proposals=[]
            for axis in range(3):
                for direction in (-1,1):
                    trial=np.clip(z+direction*step[axis]*np.eye(3)[axis],0,1)
                    proposals.append((score(trial),trial))
            next_value,next_z=min(proposals,key=lambda item:item[0])
            if next_value<value-1e-8:
                z,value=next_z,next_value
            else:
                step*=.5
        candidates.append((value,decode(z),iterations))
    best=min(candidates,key=lambda item:item[0])
    return best[1],{"start_count":len(seeds),"evaluations":len(cache),
                    "candidate_objectives":[float(v[0]) for v in candidates],
                    "best_objective":float(best[0]),"bounds":BOUNDS.tolist(),
                    "boundary_hit":[bool(best[1][j]<=BOUNDS[j,0]*1.001+1e-7 or
                                         best[1][j]>=BOUNDS[j,1]*.999-1e-7) for j in range(3)]}


def rms(actual,pred):
    return float(np.sqrt(np.mean((np.asarray(actual)-np.asarray(pred))**2)))


def crossing_times(t,values,lo,hi):
    ids=np.flatnonzero((t[:-1]>=lo)&(t[1:]<=hi)&(values[:-1]*values[1:]<0))
    return [float(t[i]-values[i]*(t[i+1]-t[i])/(values[i+1]-values[i])) for i in ids]


def diagnostics_replay_command(replay,p):
    pid_state=(0.0,0.0,0.0)
    pid_active=False
    loft_armed=True
    rows=[]
    for n in range(int(round(HORIZON/CONTROL_DT))+1):
        t=n*CONTROL_DT
        state=np.array([np.interp(t,replay["t"],replay[k]) for k in ("theta","omega","gamma","x","y")])
        req=request(t,state,replay,p,loft_armed=loft_armed)
        if p.get("geometric_loft_mode"):
            loft_armed=bool(req["loft_active"])
        if p.get("schedule_gate")=="total_pid_output":
            u,pid_state,_,pid_active=scheduled_pid_step(t,req,pid_state,p,pid_active)
        else:
            u,pid_state,_=pid_step(req["error_g"],pid_state,p)
        rows.append({"t":t,"u":u,"a_req_g":req["a_req_g"],"actual_g":req["actual_g"],
                     "flight_gain":req["flight_gain"],
                     "loft_active":req["loft_active"],
                     "los_elevation_deg":req["los_elevation_deg"],
                     "los_rate_deg_s":req["los_rate_deg_s"],
                     "pn_for_command_g":req["pn_for_command_g"],
                     "error_g":req["error_g"],"loft_raw_g":req["loft_raw_g"],
                     "loft_limited_g":req["loft_limited_g"],
                     "loft_projected_g":req["loft_projected_g"],
                     "pn_g":req["pn_g"],
                     "gravity_compensation_g":req["gravity_compensation_g"]})
    return rows


def sign_conflicts(replay, diagnostic):
    """Native intervals that contradict all nonnegative model coefficients."""
    t=replay["t"]
    diagnostic_time=np.array([r["t"] for r in diagnostic])
    diagnostic_u=np.array([r["u"] for r in diagnostic])
    conflicts=[]
    for i in range(len(t)-1):
        midpoint=(t[i]+t[i+1])/2
        if not (FIT_BOUNDS[0]<=midpoint<=FIT_BOUNDS[1]):
            continue
        u_mid=float(np.interp(midpoint,diagnostic_time,diagnostic_u))
        alpha_mid=float((replay["alpha"][i]+replay["alpha"][i+1])/2)
        omega_mid=float((replay["omega"][i]+replay["omega"][i+1])/2)
        observed_accel=float((replay["omega"][i+1]-replay["omega"][i])/(t[i+1]-t[i]))
        if u_mid<0 and alpha_mid>0 and omega_mid>0 and observed_accel>0:
            conflicts.append({"native_interval_id":i,"start_end_s":[float(t[i]),float(t[i+1])],
                "diagnostic_u_mid_rad":u_mid,"alpha_mid_rad":alpha_mid,
                "omega_mid_rad_s":omega_mid,"observed_omega_dot_rad_s2":observed_accel})
    return conflicts


def main():
    replay=load_replay()
    p=raw_parameters(replay["payload"])
    assert p["source_version"]=="2.59.0.28" and p["finsLatAccel_parent"]==55
    interface_check=control_interface_comparison(p)
    diagnostic=diagnostics_replay_command(replay,p)
    coeff,opt=optimize(replay,p)
    score,detail=objective(coeff,replay,p,True)
    half_score,half=objective(coeff,replay,p,True,plant_dt=PLANT_DT/2)
    assert detail is not None and half is not None
    rows=detail["rows"];t=replay["t"]
    fit=(t>=FIT_BOUNDS[0])&(t<=FIT_BOUNDS[1])
    hold=(t>=HOLD_BOUNDS[0])&(t<=HOLD_BOUNDS[1])
    pred=detail["pred"]
    residuals={block:{**{key+"_rmse_"+("rad_s" if key=="omega" else "rad"):
                         rms(replay[key][mask],pred[key][mask])
                         for key in ("theta","alpha","omega")},
                      "gamma_rmse_rad":rms(replay["gamma"][mask],pred["gamma"][mask]),
                      "height_rmse_m":rms(replay["y"][mask],pred["y"][mask])}
               for block,mask in (("fit",fit),("holdout",hold))}
    measured_events=event_features(t,replay["omega"])
    predicted_events=event_features(t,pred["omega"])
    hold_windows=((1.45,1.60,"max"),(1.60,1.80,"min"),(1.80,2.05,"max"))
    hold_obs_events=event_features(t,replay["omega"],hold_windows)
    hold_pred_events=event_features(t,pred["omega"],hold_windows)
    event_compare=[{"observed":o,"candidate":q,"time_error_s":q["time_s"]-o["time_s"],
                    "amplitude_error_rad_s":q["omega_rad_s"]-o["omega_rad_s"]}
                   for o,q in zip(measured_events+hold_obs_events,predicted_events+hold_pred_events)]
    target0,_=target_at(replay,0.0)
    initial_range=float(np.linalg.norm(target0-np.array([replay["x"][0],replay["y"][0]])))
    cap_exceed=[r["t"] for r in rows if abs(r["actual_g"])>p["raw_load_factor_max_g"]]
    diagnostic_time=np.array([r["t"] for r in diagnostic])
    diagnostic_u=np.array([r["u"] for r in diagnostic])
    sign_conflicts=[]
    for i in range(len(t)-1):
        midpoint=(t[i]+t[i+1])/2
        if not (FIT_BOUNDS[0]<=midpoint<=FIT_BOUNDS[1]):
            continue
        u_mid=float(np.interp(midpoint,diagnostic_time,diagnostic_u))
        alpha_mid=float((replay["alpha"][i]+replay["alpha"][i+1])/2)
        omega_mid=float((replay["omega"][i]+replay["omega"][i+1])/2)
        observed_accel=float((replay["omega"][i+1]-replay["omega"][i])/(t[i+1]-t[i]))
        if u_mid<0 and alpha_mid>0 and omega_mid>0 and observed_accel>0:
            sign_conflicts.append({"native_interval_id":i,"start_end_s":[float(t[i]),float(t[i+1])],
                "diagnostic_u_mid_rad":u_mid,"alpha_mid_rad":alpha_mid,
                "omega_mid_rad_s":omega_mid,"observed_omega_dot_rad_s2":observed_accel})
    result={
        "candidate":"theta_dot=omega; omega_dot=k_u*u-k_alpha*alpha-k_omega*omega; alpha=theta-gamma",
        "coefficient_units":{"k_u":"rad/s^2 per rad of clamped PID output",
                             "k_alpha":"s^-2", "k_omega":"s^-1"},
        "coefficients":{"k_u":float(coeff[0]),"k_alpha":float(coeff[1]),"k_omega":float(coeff[2])},
        "optimization":opt,
        "fixed_source_parameters":p,
        "interface":{
            "status":"Project-code-aligned fixed candidate, not decoded game guidance or command",
            "time_axis":"packet timestamps; 20 ms is project controller update candidate, not inferred game rate",
            "external_inputs":"observed planar speed V(t), actual target x/y trajectory and derived target velocity only",
            "initial_state":"first native missile x/y, theta, gamma and -decoded b8_z omega; PID integral/error zero",
            "guidance":"project guidance.pn_acceleration with raw PN gain, raw flight/time-to-hit gains and closing_or_relative_speed time-to-go; raw loft enabled throughout 0-3.56 s",
            "loft_cap":"Only loft component: omega_loft=loft_g*g/V is clipped to +/-0.5 deg/s; PN is unaffected until PN+loft reqAccelMax=50g clamp",
            "loft_target_elevation":"raw -3.5 deg retained but unused by current project loft expression",
            "loft_gate_difference":"profile_adapter defaults missing exit distance to 16 km lockRange and exit TGO to 18 s, which would disable loft at this shot's ~7.63 km initial range; raw-enabled choice is an explicit experiment boundary",
            "request_feedback":"PN+loft projected normal, 50g clamp, gravity added and vector-clamped/projected for required specific force; feedback=(qbar*S_eff*14*alpha+T*sin(alpha))/(m*g); error=request-feedback in g",
            "pid":"20 ms discrete, raw P/I/D, I term accumulated by Ki*error*dt and limited +/-1; derivative is raw difference /dt (rebuild tau=0); output clamped +/-0.698132 rad; saturated pushing integral held for next tick; no extra delay, no capture, no baseIndSpeed scaling of PID",
            "plant":"RK4 with zero-order held u; V(t) external, N_aero=qbar*S_eff*14*alpha, gamma_dot=(N_aero+T*sin(alpha))/(m*V)-g*cos(gamma)/V, x/y dot=V*(cos gamma,sin gamma)",
            "controller_same_input_check":interface_check},
        "data":{"missile_rows":len(t),"target_rows":len(replay["target_t"]),
                "initial_planar_target_range_m":initial_range,
                "max_planar_speed_gap_mps":replay["max_planar_speed_gap_mps"],
                "max_out_of_plane_velocity_angle_deg":replay["max_out_of_plane_velocity_angle_deg"],
                "max_abs_missile_z_displacement_m":replay["max_abs_missile_z_displacement_m"]},
        "diagnostic_replay_command":{"meaning":"u reconstructed from observed state and fixed request/PID; not decoded, not fed to forward candidate",
            "max_abs_u_rad_first2":float(max(abs(r["u"]) for r in diagnostic if r["t"]<=2)),
            "loft_raw_g_first2_max":float(max(r["loft_raw_g"] for r in diagnostic if r["t"]<=2)),
            "loft_limited_g_first2_max":float(max(r["loft_limited_g"] for r in diagnostic if r["t"]<=2)),
            "nonnegative_coefficient_sign_conflict_intervals":sign_conflicts,
            "sign_conflict_logic":"With k_u>0, k_alpha>=0, k_omega>=0, u<0, alpha>0 and omega>0 force predicted omega_dot<0; observed omega_dot>0 in these intervals",
            "rows":diagnostic},
        "fit":{"bounds_s":FIT_BOUNDS,"native_points":int(fit.sum()),"objective":score,
               "state_part":detail["state_loss"],"event_part":detail["event_loss"],
               "errors":residuals["fit"]},
        "holdout":{"bounds_s":HOLD_BOUNDS,"native_points":int(hold.sum()),
                   "excluded_from_objective":True,"no_reset":True,"errors":residuals["holdout"]},
        "late_scope_limit":{"first_predicted_feedback_over_raw_50g_s":min(cap_exceed) if cap_exceed else None,
            "raw_50g_field_not_enforced_in_specified_force_equation":True},
        "event_comparison":event_compare,
        "zero_crossings_rad_s":{"observed_train":crossing_times(t,replay["omega"],*FIT_BOUNDS),
              "candidate_train":crossing_times(t,pred["omega"],*FIT_BOUNDS),
              "observed_hold":crossing_times(t,replay["omega"],*HOLD_BOUNDS),
              "candidate_hold":crossing_times(t,pred["omega"],*HOLD_BOUNDS)},
        "numeric_step_check":{"plant_step_s":PLANT_DT,"half_step_s":PLANT_DT/2,
            "theta_max_abs_difference_rad":float(max(abs(series(rows,"theta",t)-series(half["rows"],"theta",t)))),
            "omega_max_abs_difference_rad_s":float(max(abs(series(rows,"omega",t)-series(half["rows"],"omega",t)))),
            "objective_half_step":half_score},
        "scope":"Conditional speed-forced closed-loop pitch/trajectory candidate; no speed equation, fin deflection or game-command inference"
    }
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"effective_moment_results.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    # A compact native-sample comparison keeps the three distinct observables inspectable.
    comparison=[]
    for i,tm in enumerate(t):
        comparison.append({"t_packet_s":float(tm),
            **{key+"_observed":float(replay[key][i]) for key in ("theta","alpha","omega","gamma","y")},
            **{key+"_candidate":float(pred[key][i]) for key in ("theta","alpha","omega","gamma","y")}})
    (OUT/"native_comparison.json").write_text(json.dumps(comparison,indent=2)+"\n")

    rt=np.array([r["t"] for r in rows]); dt=np.array([r["t"] for r in diagnostic])
    fig,axs=plt.subplots(2,3,figsize=(17,9.5),constrained_layout=True)
    fig.get_layout_engine().set(rect=(0,.06,1,.94))
    def blocks(ax):
        ax.axvspan(*FIT_BOUNDS,color="tab:blue",alpha=.07,label="fit")
        ax.axvspan(*HOLD_BOUNDS,color="tab:orange",alpha=.10,label="holdout")
        ax.grid(alpha=.25)
    ax=axs[0,0]
    ax.plot(t,np.degrees(replay["theta"]),"o",ms=2.5,color="tab:blue",label="θ observed")
    ax.plot(rt,np.degrees([r["theta"] for r in rows]),color="tab:blue",label="θ candidate")
    ax.plot(t,np.degrees(replay["alpha"]),"o",ms=2.5,color="tab:green",label="α observed")
    ax.plot(rt,np.degrees([r["alpha"] for r in rows]),color="tab:green",label="α candidate")
    blocks(ax);ax.set(xlim=(0,3.56),ylabel="degrees",title="Attitude and incidence")
    ax.legend(fontsize=7,ncol=2)
    ax=axs[0,1]
    ax.plot(t,replay["omega"],"o",ms=2.5,label="−b8_z observed")
    ax.plot(rt,[r["omega"] for r in rows],label="candidate")
    blocks(ax);ax.set(xlim=(0,3.56),ylabel="rad/s",title="Body pitch rate")
    ax.legend(fontsize=7)
    ax=axs[0,2]
    ax.plot(t,np.degrees(replay["gamma"]),"o",ms=2.5,color="tab:purple",label="γ observed")
    ax.plot(rt,np.degrees([r["gamma"] for r in rows]),color="tab:purple",label="γ candidate")
    ax2=ax.twinx()
    ax2.plot(t,replay["y"]-replay["y"][0],"o",ms=1.8,color="tab:gray",label="height observed")
    ax2.plot(rt,[r["y"]-replay["y"][0] for r in rows],color="tab:gray",label="height candidate")
    blocks(ax);ax.set(xlim=(0,3.56),ylabel="γ (deg)",title="Flight path and height")
    ax2.set_ylabel("height gain (m)");ax.legend(loc="upper left",fontsize=7);ax2.legend(loc="lower right",fontsize=7)
    ax=axs[1,0]
    ax.plot(rt,[r["a_req_g"] for r in rows],label="required specific force")
    ax.plot(rt,[r["actual_g"] for r in rows],label="candidate feedback")
    ax.plot(rt,[r["pn_g"] for r in rows],label="PN component",alpha=.75)
    ax.plot(rt,[r["loft_raw_g"] for r in rows],label="loft before cap",ls="--")
    ax.plot(rt,[r["loft_limited_g"] for r in rows],label="loft after cap",ls="--")
    ax.plot(rt,[r["gravity_compensation_g"] for r in rows],label="gravity comp.",ls=":")
    blocks(ax);ax.set(xlim=(0,3.56),xlabel="packet time from release (s)",ylabel="g",title="Fixed request and candidate feedback")
    ax.legend(fontsize=6.5,ncol=2)
    ax=axs[1,1]
    ax.plot(rt,[r["u"] for r in rows],label="candidate PID u")
    ax.plot(dt,[r["u"] for r in diagnostic],label="replay-state diagnostic u",alpha=.7)
    ax.axhline(p["fin_aoa_rad"],color="gray",ls=":",lw=.8)
    ax.axhline(-p["fin_aoa_rad"],color="gray",ls=":",lw=.8)
    blocks(ax);ax.set(xlim=(0,3.56),xlabel="packet time from release (s)",ylabel="clamped PID output (rad)",title="PID output is calculated, not decoded")
    ax.legend(fontsize=7,loc="lower left")
    inset=ax.inset_axes([.55,.54,.42,.36])
    inset.plot(rt,[r["u"] for r in rows],lw=1.2)
    inset.plot(dt,[r["u"] for r in diagnostic],lw=1.2)
    inset.axhline(0,color="black",lw=.6)
    inset.set(xlim=(.30,.65),ylim=(-.07,.04),title="early u")
    inset.tick_params(labelsize=7)
    inset.grid(alpha=.2)
    ax=axs[1,2]
    for key,label in (("accel_u","k_u u"),("accel_alpha","−k_α α"),("accel_damp","−k_ω ω")):
        ax.plot(rt,[r[key] for r in rows],label=label)
    blocks(ax);ax.set(xlim=(0,3.56),xlabel="packet time from release (s)",ylabel="rad/s²",title="Three direct angular-acceleration terms")
    ax.legend(fontsize=7)
    fig.suptitle("R-77-1 h110: fixed guidance/PID and three-parameter direct-moment candidate",fontsize=14)
    fig.text(.5,.013,"V(t) is measured input; target motion is external. States and PID memory evolve continuously. Loft cap applies only to loft, using 0.5°/s project interpretation.",ha="center",fontsize=9)
    fig.savefig(OUT/"effective_moment.png",dpi=170)
    plt.close(fig)
    print(json.dumps({"coefficients":result["coefficients"],"fit":result["fit"],
                      "holdout":result["holdout"],"optimization":opt,
                      "step_check":result["numeric_step_check"],
                      "interface_check":interface_check},indent=2))


def unit_sensitivity():
    """Only vary the unit assigned to the raw loftTargetOmegaMax=0.5."""
    replay=load_replay()
    p=raw_parameters(replay["payload"])
    assert p["loft_omega_max_deg_s"]==.5
    baseline_result=OUT/"effective_moment_results.json"
    baseline_figure=OUT/"effective_moment.png"
    assert baseline_result.exists() and baseline_figure.exists()
    comparison={}
    requested_times=(.32,.40,.44,.84)
    for mode in ("deg_per_s","rad_per_s"):
        params=dict(p)
        params["loft_omega_unit"]=mode
        diagnostic=diagnostics_replay_command(replay,params)
        conflicts=sign_conflicts(replay,diagnostic)
        snapshots=[]
        for time in requested_times:
            row=next(r for r in diagnostic if abs(r["t"]-time)<1e-9)
            snapshots.append({key:float(row[key]) for key in
                ("t","pn_g","loft_raw_g","loft_limited_g","loft_projected_g",
                 "gravity_compensation_g","a_req_g","actual_g","error_g","u")})
        comparison[mode]={"loft_omega_limit_rad_s":(.5 if mode=="rad_per_s" else math.radians(.5)),
            "request_snapshots":snapshots,
            "native_sign_conflict_count":len(conflicts),
            "native_sign_conflicts":conflicts,
            "diagnostic_control_rows":diagnostic}
    refit_allowed=comparison["rad_per_s"]["native_sign_conflict_count"]==0
    result={"source":"source_payload.json; original numeric cells preserved",
        "raw_loftTargetOmegaMax":.5,
        "only_intervention":"Interpret the same raw 0.5 as degrees/second or radians/second in the existing loft-only cap omega_loft=loft_g*g/V; all raw values, PN+loft formulas, gain placement, raw-enabled loft gate, initial state and PID are fixed",
        "interpretation":"Both are project-interface unit candidates, not verified War Thunder semantics",
        "diagnostic_u_semantics":"Computed from measured replay state and each fixed guidance/PID candidate; it is not a decoded control input and did not enter the previous forward run",
        "positive_coefficient_necessary_condition":"For k_u>0, k_alpha>=0 and k_omega>=0, u<0, alpha>0 and omega>0 imply omega_dot<0",
        "comparison":comparison,
        "rad_refit_performed":refit_allowed,
        "stop_reason":None if refit_allowed else "Two native positive-omega-dot sign conflicts remain under 0.5 rad/s; the instructed gate forbids a second fit",
        "baseline_files_preserved":[str(baseline_result),str(baseline_figure)]}
    if refit_allowed:
        params=dict(p)
        params["loft_omega_unit"]="rad_per_s"
        coeff,opt=optimize(replay,params)
        score,detail=objective(coeff,replay,params,True)
        assert detail is not None
        t=replay["t"]
        fit=(t>=FIT_BOUNDS[0])&(t<=FIT_BOUNDS[1])
        hold=(t>=HOLD_BOUNDS[0])&(t<=HOLD_BOUNDS[1])
        result["rad_refit"]={"coefficients":{"k_u":float(coeff[0]),"k_alpha":float(coeff[1]),"k_omega":float(coeff[2])},
            "optimization":opt,"training_objective":score,
            "fit_rmse":{"theta_rad":rms(replay["theta"][fit],detail["pred"]["theta"][fit]),
                        "alpha_rad":rms(replay["alpha"][fit],detail["pred"]["alpha"][fit]),
                        "omega_rad_s":rms(replay["omega"][fit],detail["pred"]["omega"][fit])},
            "holdout_rmse":{"theta_rad":rms(replay["theta"][hold],detail["pred"]["theta"][hold]),
                            "alpha_rad":rms(replay["alpha"][hold],detail["pred"]["alpha"][hold]),
                            "omega_rad_s":rms(replay["omega"][hold],detail["pred"]["omega"][hold])}}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"unit_sensitivity.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")

    fig,axs=plt.subplots(2,1,figsize=(11,6.7),sharex=True,constrained_layout=True)
    fig.get_layout_engine().set(rect=(0,.075,1,.90))
    fig.suptitle("Raw loftTargetOmegaMax=0.5: unit-sensitive diagnostic input",fontsize=14)
    t=replay["t"]
    axs[0].plot(t,replay["omega"],"o-",markersize=3,lw=1.2,color="black",label="observed −b8_z pitch rate")
    axs[0].set(ylabel="observed ω (rad/s)")
    colors={"deg_per_s":"tab:blue","rad_per_s":"tab:red"}
    for mode,label in (("deg_per_s","0.5°/s diagnostic u"),("rad_per_s","0.5 rad/s diagnostic u")):
        rows=comparison[mode]["diagnostic_control_rows"]
        axs[1].plot([r["t"] for r in rows],[r["u"] for r in rows],color=colors[mode],lw=1.6,label=label)
    axs[1].axhline(0,color="black",lw=.8)
    for ax in axs:
        ax.axvspan(*FIT_BOUNDS,color="gray",alpha=.045,label="training bounds")
        # Blue spans show all degree interpretation conflicts; red spans show
        # those that persist under the radian interpretation.
        for i,interval in enumerate(comparison["deg_per_s"]["native_sign_conflicts"]):
            ax.axvspan(*interval["start_end_s"],color="tab:blue",alpha=.12,
                       label="°/s sign conflict" if i==0 else None)
        for i,interval in enumerate(comparison["rad_per_s"]["native_sign_conflicts"]):
            ax.axvspan(*interval["start_end_s"],color="tab:red",alpha=.18,
                       label="rad/s sign conflict" if i==0 else None)
        ax.grid(alpha=.2)
    axs[0].legend(fontsize=8,ncol=2,loc="upper right")
    axs[1].legend(fontsize=8,ncol=2,loc="lower left")
    axs[1].set(xlim=(.25,1.05),xlabel="packet time from release (s)",ylabel="diagnostic PID u (rad)")
    fig.text(.5,.018,"u is reconstructed from measured states, not decoded or used in the original forward integration. Shaded intervals: observed ω̇>0 while u<0, α>0, ω>0.",ha="center",fontsize=8.5)
    fig.savefig(OUT/"unit_sensitivity.png",dpi=180)
    plt.close(fig)
    print(json.dumps({"rad_refit_performed":result["rad_refit_performed"],
        "conflict_counts":{mode:value["native_sign_conflict_count"] for mode,value in comparison.items()},
        "stop_reason":result["stop_reason"]},indent=2,ensure_ascii=False))


def full_interval_sign_conflicts(replay, diagnostic):
    """Check the full native interval under the candidate's interpolation/ZOH."""
    times=replay["t"]
    d_times=np.array([row["t"] for row in diagnostic])
    d_u=np.array([row["u"] for row in diagnostic])
    found=[]
    for i in range(len(times)-1):
        lo,hi=map(float,(times[i],times[i+1]))
        if not (FIT_BOUNDS[0]<=lo and hi<=FIT_BOUNDS[1]):
            continue
        dq=float(replay["omega"][i+1]-replay["omega"][i])
        if dq<=0:
            continue
        # The diagnostic PID is evaluated at 20 ms ticks and held until the
        # next tick. Include each tick in the native interval plus both ends.
        control_indices=np.flatnonzero((d_times>=lo-1e-12)&(d_times<=hi+1e-12))
        u_values=[float(d_u[max(0,np.searchsorted(d_times,lo,side="right")-1)])]
        u_values.extend(float(d_u[j]) for j in control_indices)
        if min(float(replay["alpha"][i]),float(replay["alpha"][i+1]))<=0:
            continue
        if min(float(replay["omega"][i]),float(replay["omega"][i+1]))<=0:
            continue
        if max(u_values)>=0:
            continue
        found.append({"interval_s":[lo,hi],"native_interval_id":i,
                      "alpha_endpoints_rad":[float(replay["alpha"][i]),float(replay["alpha"][i+1])],
                      "omega_endpoints_rad_s":[float(replay["omega"][i]),float(replay["omega"][i+1])],
                      "u_values_on_held_control_ticks_rad":u_values,
                      "observed_mean_omega_dot_rad_s2":dq/(hi-lo)})
    return found


def passive_launch_estimate(replay):
    """Early zero-input integral relation; a diagnostic, not game coefficients."""
    t=replay["t"]
    ids=np.flatnonzero((t[:-1]>=0)&(t[1:]<=.291+1e-12))
    x=np.array([[-.5*(replay["alpha"][i]+replay["alpha"][i+1])*(t[i+1]-t[i]),
                 -.5*(replay["omega"][i]+replay["omega"][i+1])*(t[i+1]-t[i])]
                for i in ids])
    y=np.array([replay["omega"][i+1]-replay["omega"][i] for i in ids])
    beta,residuals,rank,singular=np.linalg.lstsq(x,y,rcond=None)
    return {"interval_s":[float(t[ids[0]]),float(t[ids[-1]+1])],
            "interval_count":len(ids),"columns":["minus_integral_alpha_dt","minus_integral_omega_dt"],
            "rank":int(rank),"condition_number":float(singular[0]/singular[-1]),
            "unconstrained_k_alpha_k_omega":beta.tolist(),
            "delta_omega_rmse_rad_s":rms(y,x@beta),
            "warning":"Short pre-arm segment with native sampling; no unique physical calibration."}


def target_elevation_geometry(replay,threshold_deg):
    """Bound two plausible angle meanings without installing either as a gate."""
    indices=np.flatnonzero(replay["t"]<=2.5)
    absolute=[];relative=[]
    for i in indices:
        target,_=target_at(replay,float(replay["t"][i]))
        los=math.atan2(float(target[1]-replay["y"][i]),
                       float(target[0]-replay["x"][i]))
        absolute.append(math.degrees(los))
        relative.append(math.degrees(los-replay["gamma"][i]))
    crossings=crossing_times(replay["t"][indices],
                              np.asarray(relative)-threshold_deg,0,2.5)
    return {"raw_threshold_deg":threshold_deg,
            "absolute_los_elevation_range_deg":[min(absolute),max(absolute)],
            "los_minus_flight_path_range_deg":[min(relative),max(relative)],
            "los_minus_flight_path_threshold_crossings_s":crossings,
            "meaning":"Geometry only; no evidence that raw loftTargetElevation is either angle or an exit gate."}


def schedule_loft_experiment():
    """One bounded total-output gate and loft-reference comparison."""
    replay=load_replay()
    raw=raw_parameters(replay["payload"])
    p=dict(raw,schedule_gate="total_pid_output",loft_omega_unit="rad_per_s")
    references=("body","flight_path","los_relative_flight_path")
    diagnostic={}
    for reference in references:
        q=dict(p,loft_reference=reference)
        rows=diagnostics_replay_command(replay,q)
        diagnostic[reference]={"rows":rows,"full_interval_conflicts":full_interval_sign_conflicts(replay,rows)}
    # The body reference fails already during the first positive-rate peak.
    # LOS-relative and absolute flight-path references have almost identical
    # geometry for this near-level target; fit only the simpler flight path.
    selected="flight_path"
    p["loft_reference"]=selected
    coeff,opt=optimize(replay,p)
    score,detail=objective(coeff,replay,p,True)
    assert detail is not None
    half_score,half=objective(coeff,replay,p,True,plant_dt=PLANT_DT/2)
    assert half is not None
    rows=detail["rows"]
    t=replay["t"]
    fit=(t>=FIT_BOUNDS[0])&(t<=FIT_BOUNDS[1])
    hold=(t>=HOLD_BOUNDS[0])&(t<=HOLD_BOUNDS[1])
    pred=detail["pred"]
    errors={label:{key:rms(replay[key][mask],pred[key][mask])
                   for key in ("theta","alpha","omega","gamma","y")}
            for label,mask in (("train",fit),("holdout",hold))}
    first_peak=(t>=.35)&(t<=.75)
    attitude_peaks={key:{"observed_deg":float(np.degrees(max(replay[key][first_peak]))),
                         "forward_deg":float(np.degrees(max(pred[key][first_peak]))),
                         "observed_time_s":float(t[np.flatnonzero(first_peak)[np.argmax(replay[key][first_peak])]]),
                         "forward_time_s":float(t[np.flatnonzero(first_peak)[np.argmax(pred[key][first_peak])]])}
                    for key in ("theta","alpha")}
    events=[{"observed":o,"forward":q,"time_error_s":q["time_s"]-o["time_s"],
             "amplitude_error_rad_s":q["omega_rad_s"]-o["omega_rad_s"]}
            for o,q in zip(detail["obs_events"],detail["pred_events"])]
    zero_gain_rows=[row for row in rows if row["t"]<=.30+1e-12]
    assert zero_gain_rows and all(row["u"]==0.0 for row in zero_gain_rows)
    assert any(abs(row["omega"]-rows[0]["omega"])>1e-5 for row in zero_gain_rows[1:])
    assert all(row["flight_gain"]==0.0 for row in zero_gain_rows)
    d=diagnostic[selected]["rows"]
    # Native observations are read only for scoring in the stated training
    # window and for diagnostics; the forward run never resets at the holdout.
    result={
        "candidate":"theta_dot=omega; omega_dot=k_u*u-k_alpha*alpha-k_omega*omega; alpha=theta-gamma",
        "schedule_equation":"u=g(t)*PID(a_req-a_feedback); g=timeToGain; g=0 freezes PID memory; first active previous_error=current error; no D kick",
        "passive_equations":"N=qbar*S_eff*14*alpha; gamma_dot=(N+T*sin(alpha))/(m*V)-g_gravity*cos(gamma)/V; gravity and aero remain active for g=0",
        "gain_placement":"timeToGain occurs once at total clamped PID output, not in PN; timeToHit still multiplies PN only",
        "gate_samples":[{"t":row["t"],"gain":row["flight_gain"],"u":row["u"]}
                        for row in rows if .28-1e-12<=row["t"]<=.34+1e-12],
        "raw_fixed_parameters":raw,
        "raw_warmUpTime_s":replay["payload"]["optional_existing_parent_parameters"]["rocket"]["guidance"]["warmUpTime"],
        "loft_structures":{
            "body":"loft_g=17.5*(24deg-theta); cap only loft via |loft_g*g_gravity/V|<=0.5 rad/s",
            "flight_path":"loft_g=17.5*(24deg-gamma); same cap",
            "los_relative_flight_path":"loft_g=17.5*(LOS_elevation+24deg-gamma); same cap"},
        "loft_unit_status":"0.5 rad/s is a candidate interpretation, not decoded game semantics",
        "loft_target_elevation_status":"raw -3.5 deg retained but unused; its geometric role is unknown and no exit threshold is invented",
        "loft_target_elevation_geometry":target_elevation_geometry(replay,raw["loft_target_elevation_deg"]),
        "diagnostic_meaning":"replay-state u uses measured states and fixed controller memory; it is not decoded and is not fed into the forward run",
        "diagnostic":{ref:{"full_interval_conflicts":value["full_interval_conflicts"],
                            "u_at_s":{str(time):float(series(value["rows"],"u",time))
                                      for time in (.28,.30,.32,.40,.44,.80,.837,.86,.874)}}
                      for ref,value in diagnostic.items()},
        "pre_arm_passive_estimate":passive_launch_estimate(replay),
        "fitted_reference":selected,"coefficients":{"k_u_rad_s2_per_rad":float(coeff[0]),
                                                   "k_alpha_s_minus2":float(coeff[1]),
                                                   "k_omega_s_minus1":float(coeff[2])},
        "optimization":opt,"training":{"bounds_s":FIT_BOUNDS,"native_points":int(fit.sum()),
                                           "objective":score,"errors":errors["train"]},
        "holdout":{"bounds_s":HOLD_BOUNDS,"native_points":int(hold.sum()),
                   "excluded_from_objective":True,"no_state_reset":True,"errors":errors["holdout"]},
        "peak_valley_comparison":events,
        "first_attitude_peaks":attitude_peaks,
        "omega_zero_crossings_s":{
            "observed_train":crossing_times(t,replay["omega"],*FIT_BOUNDS),
            "forward_train":crossing_times(t,pred["omega"],*FIT_BOUNDS),
            "observed_holdout":crossing_times(t,replay["omega"],*HOLD_BOUNDS),
            "forward_holdout":crossing_times(t,pred["omega"],*HOLD_BOUNDS)},
        "raw_50g_field_scope":{"first_forward_feedback_over_50g_s":next((row["t"] for row in rows if abs(row["actual_g"])>raw["raw_load_factor_max_g"]),None),
                               "meaning":"Source field only; no confirmed game response clamp is applied."},
        "step_check":{"plant_dt_s":PLANT_DT,"half_dt_s":PLANT_DT/2,
                      "max_theta_difference_rad":float(max(abs(series(rows,"theta",t)-series(half["rows"],"theta",t)))),
                      "max_omega_difference_rad_s":float(max(abs(series(rows,"omega",t)-series(half["rows"],"omega",t)))),
                      "objective_half_step":half_score},
        "scope":"Conditional forward trajectory with measured V(t) and measured target trajectory; target/velocity are external, theta/omega/gamma/position/PID memory evolve from release",
        "source_unchanged":True,"production_unchanged":True}
    out=OUT/"schedule_loft"
    out.mkdir(parents=True,exist_ok=True)
    (out/"results.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    figure_schedule_loft(out,replay,p,rows,d,diagnostic,coeff)
    print(json.dumps({"selected":selected,"coefficients":result["coefficients"],
                      "training":result["training"],"holdout":result["holdout"],
                      "conflicts":{ref:len(value["full_interval_conflicts"]) for ref,value in diagnostic.items()},
                      "step_check":result["step_check"]},indent=2))


def figure_schedule_loft(out,replay,p,rows,diagnostic_rows,all_diagnostic,coeff):
    rt=np.array([row["t"] for row in rows]);t=replay["t"]
    fig,axs=plt.subplots(2,3,figsize=(17,9),constrained_layout=True)
    fig.get_layout_engine().set(rect=(0,.045,1,.94))
    def decorate(ax):
        ax.axvspan(*FIT_BOUNDS,color="tab:blue",alpha=.07,label="fit")
        ax.axvspan(*HOLD_BOUNDS,color="tab:orange",alpha=.09,label="holdout")
        ax.axvline(.30,color="black",ls=":",lw=.8)
        ax.set_xlim(0,2.5);ax.grid(alpha=.22)
    ax=axs[0,0]
    for name,color in (("theta","tab:blue"),("alpha","tab:green")):
        ax.plot(t,np.degrees(replay[name]),"o",ms=2.5,color=color,label=f"{name} observed")
        ax.plot(rt,np.degrees([row[name] for row in rows]),color=color,label=f"{name} forward")
    decorate(ax);ax.set(ylabel="deg",title="Pitch and angle of attack");ax.legend(fontsize=7,ncol=2)
    ax=axs[0,1]
    ax.plot(t,replay["omega"],"o",ms=2.5,label="−b8z observed")
    ax.plot(rt,[row["omega"] for row in rows],label="forward")
    decorate(ax);ax.set(ylabel="rad/s",title="Pitch rate");ax.legend(fontsize=8)
    ax=axs[0,2]
    ax.plot(t,np.degrees(replay["gamma"]),"o",ms=2.5,label="γ observed")
    ax.plot(rt,np.degrees([row["gamma"] for row in rows]),label="γ forward")
    decorate(ax);ax.set(ylabel="deg",title="Flight path");ax.legend(fontsize=8)
    ax=axs[1,0]
    for key,label in (("pn_g","PN × timeToHit"),("loft_limited_g","loft after cap"),("a_req_g","request + gravity")):
        ax.plot(rt,[row[key] for row in rows],label=label)
    decorate(ax);ax.set(ylabel="g",xlabel="packet age (s)",title="Forward request before output gate");ax.legend(fontsize=7)
    ax=axs[1,1]
    ax.plot(rt,[row["u"] for row in rows],label="forward total u")
    ax.plot([row["t"] for row in diagnostic_rows],[row["u"] for row in diagnostic_rows],ls="--",label="replay-state diagnostic u")
    ax.axhline(0,color="black",lw=.7)
    ax2=ax.twinx();ax2.plot(rt,[row["flight_gain"] for row in rows],color="tab:gray",lw=1.2,label="timeToGain")
    decorate(ax);ax.set(ylabel="u (rad)",xlabel="packet age (s)",title="Total active command and gain")
    ax2.set_ylabel("gain");ax.legend(loc="lower left",fontsize=7);ax2.legend(loc="upper right",fontsize=7)
    ax=axs[1,2]
    for key,label in (("accel_u","kᵤu"),("accel_alpha","−kα α"),("accel_damp","−kω ω")):
        ax.plot(rt,[row[key] for row in rows],label=label)
    decorate(ax);ax.set(ylabel="rad/s²",xlabel="packet age (s)",title="Forward angular acceleration terms")
    ax.legend(fontsize=7)
    fig.suptitle("R-77-1 h110: total timeToGain gate; flight-path loft reference (candidate)",fontsize=14)
    fig.text(.5,.014,"Measured V(t) and target track are external; states/PID integrate from release. Blue = fit; orange = continuous holdout. Raw 0.5 is a rad/s candidate; loftTargetElevation=-3.5° is unmapped.",ha="center",fontsize=8)
    fig.savefig(out/"forward_comparison.png",dpi=175)
    plt.close(fig)


def diagnostic_interval_residuals(replay,diagnostic,coeff,bounds):
    """Native Δω closure with 20 ms held diagnostic u and measured α, ω."""
    ku,ka,kw=map(float,coeff)
    dtimes=np.array([row["t"] for row in diagnostic])
    dus=np.array([row["u"] for row in diagnostic])
    t=replay["t"]
    rows=[]
    for i in range(len(t)-1):
        lo,hi=map(float,(t[i],t[i+1]))
        if lo<bounds[0]-1e-12 or hi>bounds[1]+1e-12:
            continue
        cuts=[lo]+[float(v) for v in dtimes if lo<v<hi]+[hi]
        integral_u=sum((right-left)*float(dus[max(0,np.searchsorted(dtimes,left,side="right")-1)])
                       for left,right in zip(cuts,cuts[1:]))
        integral_alpha=.5*(replay["alpha"][i]+replay["alpha"][i+1])*(hi-lo)
        integral_omega=.5*(replay["omega"][i]+replay["omega"][i+1])*(hi-lo)
        pred=ku*integral_u-ka*integral_alpha-kw*integral_omega
        actual=float(replay["omega"][i+1]-replay["omega"][i])
        rows.append({"interval_s":[lo,hi],"observed_delta_omega_rad_s":actual,
                     "conditional_delta_omega_rad_s":float(pred),
                     "residual_rad_s":float(actual-pred)})
    residual=np.array([row["residual_rad_s"] for row in rows])
    return {"interval_count":len(rows),"delta_omega_rmse_rad_s":float(np.sqrt(np.mean(residual**2))),
            "delta_omega_max_abs_residual_rad_s":float(np.max(np.abs(residual))),
            "mean_residual_rad_s":float(np.mean(residual)),"rows":rows}


def geometric_transition(replay,p,rows):
    """First one-way exit and the instantaneous request discontinuity."""
    index=next((i for i in range(1,len(rows)) if rows[i-1]["loft_active"]
                and not rows[i]["loft_active"]),None)
    if index is None:
        return None
    row=rows[index]
    t=float(row["t"])
    state=np.array([row[key] for key in ("theta","omega","gamma","x","y")],float)
    active=request(t,state,replay,p,force_geometric_active=True)
    inactive=request(t,state,replay,p,force_geometric_active=False)
    return {"last_active_tick_s":float(rows[index-1]["t"]),"first_inactive_tick_s":t,
            "los_elevation_deg":row["los_elevation_deg"],"los_rate_deg_s":row["los_rate_deg_s"],
            "same_state_request_active_g":active["a_req_g"],
            "same_state_request_inactive_g":inactive["a_req_g"],
            "instantaneous_request_jump_g":inactive["a_req_g"]-active["a_req_g"],
            "adjacent_tick_request_change_g":row["a_req_g"]-rows[index-1]["a_req_g"]}


def geometric_loft_experiment():
    replay=load_replay()
    raw=raw_parameters(replay["payload"])
    prior=json.loads((OUT/"schedule_loft/results.json").read_text())
    cc=prior["coefficients"]
    coeff=(cc["k_u_rad_s2_per_rad"],cc["k_alpha_s_minus2"],cc["k_omega_s_minus1"])
    common=dict(raw,schedule_gate="total_pid_output",loft_reference="flight_path")
    previous=dict(common,loft_omega_unit="rad_per_s")
    prior_rows=simulate(coeff,replay,previous)
    assert prior_rows is not None
    modes=("additive","exclusive_vertical")
    mode_rows={};diagnostics={};mode_results={}
    for mode in modes:
        p=dict(common,geometric_loft_mode=mode)
        diag=diagnostics_replay_command(replay,p)
        rows=simulate(coeff,replay,p)
        assert rows is not None and len(rows)==len(prior_rows)
        half_rows=simulate(coeff,replay,p,plant_dt=PLANT_DT/2)
        assert half_rows is not None and len(half_rows)==len(rows)
        assert [row["loft_active"] for row in half_rows]==[row["loft_active"] for row in rows]
        assert all(row["loft_limited_g"]==row["loft_raw_g"] for row in rows)
        if mode=="exclusive_vertical":
            assert all((row["pn_for_command_g"]==0.0) if row["loft_active"]
                       else math.isclose(row["pn_for_command_g"],row["pn_g"],abs_tol=1e-12)
                       for row in rows)
        assert not any(rows[i]["loft_active"] and not rows[i-1]["loft_active"]
                       for i in range(1,len(rows)))
        diagnostics[mode]=diag;mode_rows[mode]=rows
        t=replay["t"]
        pred={key:series(rows,key,t) for key in ("theta","alpha","omega","gamma","y")}
        errors={label:{key:rms(replay[key][mask],pred[key][mask])
                       for key in pred}
                for label,mask in (("train",(t>=FIT_BOUNDS[0])&(t<=FIT_BOUNDS[1])),
                                   ("holdout",(t>=HOLD_BOUNDS[0])&(t<=HOLD_BOUNDS[1])))}
        mode_results[mode]={
            "formula":("a_req=PN*timeToHit+loft+gravity" if mode=="additive"
                       else "a_req=loft+gravity while active; PN*timeToHit+gravity after exit"),
            "diagnostic_full_interval_sign_conflicts":full_interval_sign_conflicts(replay,diag),
            "diagnostic_transition":geometric_transition(
                replay,p,[{"t":row["t"],"loft_active":row["loft_active"],
                           "los_elevation_deg":row["los_elevation_deg"],
                           "los_rate_deg_s":row["los_rate_deg_s"],
                           "a_req_g":row["a_req_g"],
                           **{key:float(np.interp(row["t"],t,replay[key]))
                              for key in ("theta","omega","gamma","x","y")}}
                          for row in diag]),
            "forward_transition":geometric_transition(replay,p,rows),
            "diagnostic_u_at_s":{str(v):float(series(diag,"u",v))
                                  for v in (.32,.40,.44,.80,.837,.86,.874,1.04)},
            "conditional_native_interval_residuals":{
                "train":diagnostic_interval_residuals(replay,diag,coeff,FIT_BOUNDS),
                "holdout":diagnostic_interval_residuals(replay,diag,coeff,HOLD_BOUNDS)},
            "forward_errors":errors,
            "forward_omega_events":[{"observed":o,"forward":q,
                                      "time_error_s":q["time_s"]-o["time_s"],
                                      "amplitude_error_rad_s":q["omega_rad_s"]-o["omega_rad_s"]}
                                    for o,q in zip(event_features(t,replay["omega"]),
                                                   event_features(t,pred["omega"]))],
            "forward_omega_zero_crossings_s":{
                "train":crossing_times(t,pred["omega"],*FIT_BOUNDS),
                "holdout":crossing_times(t,pred["omega"],*HOLD_BOUNDS)},
            "numeric_half_step_check":{
                "plant_steps_s":[PLANT_DT,PLANT_DT/2],
                "transition_tick_unchanged":True,
                "max_theta_difference_rad":max(abs(a["theta"]-b["theta"])
                                                for a,b in zip(rows,half_rows)),
                "max_omega_difference_rad_s":max(abs(a["omega"]-b["omega"])
                                                    for a,b in zip(rows,half_rows))}}
    # Exact native packet inputs show which original geometric threshold can
    # affect this particular trajectory; no threshold is fitted to the peaks.
    observed_los=[]
    for i,tm in enumerate(replay["t"]):
        target,tv=target_at(replay,float(tm))
        rel=target-np.array([replay["x"][i],replay["y"][i]])
        rel_v=tv-replay["velocity3"][i,:2]
        observed_los.append({"t":float(tm),"elevation_deg":math.degrees(math.atan2(rel[1],rel[0])),
                             "rate_deg_s":math.degrees((rel[0]*rel_v[1]-rel[1]*rel_v[0])/(rel@rel))})
    result={
        "hypothesis_origin":"Community first-person explanation supplied by main review, not game source: https://forum.warthunder.com/t/guided-weaponry-spreadsheet-variables-help/293812/7",
        "hypothesis_status":"One explicit geometry candidate, not verified War Thunder semantics",
        "active_condition":"At each 20 ms controller tick, loft active iff it has never exited, LOS elevation > raw -3.5 deg, and |LOS elevation rate| < raw 0.5 deg/s; first failure exits for the remaining run",
        "latch_reason":"Pure selector would re-enter when observed |LOS rate| falls below 0.5 deg/s near 2.75 s; one-way loft-phase exit avoids re-entry without fitted timing or smoothing. The latch is an unverified structural choice.",
        "loft_law":"When active loft_g=17.5*(24 deg - gamma) in g units; no a_loft*g/V cap. The 0.5 field gates target LOS rate only.",
        "fixed_controller_plant":"Raw P/I/D unchanged; total timeToGain output gate unchanged; timeToHit still on PN; N_aero=qbar*S_eff*14*alpha; measured V and target external; 3 moment coefficients frozen from schedule_loft/results.json",
        "raw_fields":{"loftElevation_deg":24,"loftTargetElevation_deg":raw["loft_target_elevation_deg"],
                      "loftTargetOmegaMax_candidate_deg_s":raw["loft_omega_max_deg_s"],
                      "loftAngleToAccelMult":raw["loft_angle_to_accel_mult"]},
        "fixed_coefficients":{"k_u":coeff[0],"k_alpha":coeff[1],"k_omega":coeff[2]},
        "refit_performed":False,
        "observed_geometry":{"first_2p5s_los_elevation_range_deg":[
            min(row["elevation_deg"] for row in observed_los if row["t"]<=2.5),
            max(row["elevation_deg"] for row in observed_los if row["t"]<=2.5)],
            "first_native_abs_rate_at_or_above_0p5":next((row for row in observed_los if abs(row["rate_deg_s"])>=.5),None),
            "first_native_elevation_at_or_below_minus_3p5":next((row for row in observed_los if row["elevation_deg"]<=-3.5),None),
            "los_native_rows":observed_los},
        "models":mode_results,
        "previous_candidate":"schedule_loft flight_path with 0.5 rad/s interpreted as loft cap; identical frozen coefficients and total-output gate",
        "interpretation_limit":"This shot never reaches -3.5 deg LOS elevation, so that raw threshold is unidentifiable here even though wired into the candidate. Diagnostic u is reconstructed, not decoded. One-shot forward comparison is not cross-shot validation."}
    out=OUT/"schedule_loft/geometric_loft"
    out.mkdir(parents=True,exist_ok=True)
    (out/"results.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    figure_geometric_loft(out,replay,prior_rows,mode_rows,diagnostics,result)
    print(json.dumps({"observed_first_rate_crossing":result["observed_geometry"]["first_native_abs_rate_at_or_above_0p5"],
                      "summary":{mode:{"diagnostic_exit":info["diagnostic_transition"],
                                       "forward_exit":info["forward_transition"],
                                       "conflicts":len(info["diagnostic_full_interval_sign_conflicts"]),
                                       "holdout":info["forward_errors"]["holdout"]}
                                 for mode,info in mode_results.items()}},indent=2))


def figure_geometric_loft(out,replay,prior_rows,mode_rows,diagnostics,result):
    t=replay["t"]
    colors={"previous":"tab:gray","additive":"tab:blue","exclusive_vertical":"tab:orange"}
    fig,axs=plt.subplots(2,3,figsize=(17,9.5),constrained_layout=True)
    fig.get_layout_engine().set(rect=(0,.045,1,.94))
    def decorate(ax):
        ax.axvspan(*FIT_BOUNDS,color="tab:blue",alpha=.045)
        ax.axvspan(*HOLD_BOUNDS,color="tab:orange",alpha=.06)
        ax.axvline(.30,color="black",ls=":",lw=.8)
        ax.set_xlim(0,2.5);ax.grid(alpha=.22)
    for ax,key,scale,title,unit in ((axs[0,0],"theta",180/math.pi,"Pitch","deg"),
                                    (axs[0,1],"alpha",180/math.pi,"Angle of attack","deg"),
                                    (axs[0,2],"omega",1,"Body pitch rate","rad/s")):
        ax.plot(t,replay[key]*scale,"o",ms=2.4,color="black",label="observed")
        for mode,rows in (("previous",prior_rows),*mode_rows.items()):
            ax.plot([row["t"] for row in rows],[row[key]*scale for row in rows],
                    color=colors[mode],label=mode.replace("_"," "))
        decorate(ax);ax.set(ylabel=unit,title=title);ax.legend(fontsize=7,ncol=2)
    ax=axs[1,0]
    geo=[row for row in result["observed_geometry"]["los_native_rows"] if row["t"]<=2.5]
    ax.plot([row["t"] for row in geo],[row["elevation_deg"] for row in geo],"o",ms=2,
            color="tab:green",label="observed LOS elevation")
    ax.axhline(-3.5,color="tab:green",ls="--",label="raw −3.5°")
    ax2=ax.twinx()
    ax2.plot([row["t"] for row in geo],[abs(row["rate_deg_s"]) for row in geo],
             color="tab:red",label="|observed LOS rate|")
    ax2.axhline(.5,color="tab:red",ls="--",label="raw 0.5°/s")
    ax.axvline(1.04,color="tab:red",ls=":",lw=1)
    decorate(ax);ax.set(ylabel="LOS elevation (deg)",xlabel="packet age (s)",title="Observed geometry; diagnostic exit 1.04 s")
    ax.set_ylim(-3.8,.3);ax2.set_ylim(0,.85)
    ax2.set_ylabel("|LOS rate| (deg/s)");ax.legend(loc="upper left",fontsize=7);ax2.legend(loc="center left",fontsize=7)
    ax=axs[1,1]
    for mode in mode_rows:
        rows=mode_rows[mode];diag=diagnostics[mode]
        ax.plot([row["t"] for row in rows],[row["u"] for row in rows],color=colors[mode],
                label=f"{mode} forward")
        ax.plot([row["t"] for row in diag],[row["u"] for row in diag],color=colors[mode],
                ls="--",label=f"{mode} replay-state")
    ax.axhline(0,color="black",lw=.8)
    decorate(ax);ax.set(ylabel="total u (rad)",xlabel="packet age (s)",title="Command: forward vs diagnostic")
    ax.legend(fontsize=7,ncol=2)
    ax=axs[1,2]
    rows=mode_rows["exclusive_vertical"]
    for key,label in (("pn_g","PN raw"),("pn_for_command_g","PN used"),
                      ("loft_limited_g","loft"),("a_req_g","total + gravity")):
        ax.plot([row["t"] for row in rows],[row[key] for row in rows],label=label)
    decorate(ax);ax.set(ylabel="g",xlabel="packet age (s)",title="B forward request and hard switch")
    ax.legend(fontsize=7,ncol=2)
    fig.suptitle("R-77-1 h110: target-geometry loft gate (fixed-coefficient candidates)",fontsize=14)
    fig.text(.5,.015,"TimeToGain gates total u; measured V(t) and target are external. A = PN + loft; B = loft owns vertical until exit. Pale blue/orange = train/holdout; dashed u = replay-state diagnostic, not decoded.",ha="center",fontsize=8)
    fig.savefig(out/"forward_comparison.png",dpi=175)
    plt.close(fig)


if __name__=="__main__":
    if sys.argv[1:]==["--unit-sensitivity-only"]:
        unit_sensitivity()
    elif sys.argv[1:]==["--schedule-loft"]:
        schedule_loft_experiment()
    elif sys.argv[1:]==["--geometric-loft"]:
        geometric_loft_experiment()
    elif not sys.argv[1:]:
        main()
    else:
        raise SystemExit("usage: try_r771_effective_moment.py [--unit-sensitivity-only|--schedule-loft|--geometric-loft]")
