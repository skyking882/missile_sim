"""Fit one no-loft R-77-1 closed-loop attitude candidate to two L shots.

L55 is opened only after model/lifecycle selection. Speed and target motion are
measured external inputs; all other missile states and PID memory evolve from
the decoded creation state. Nothing here is a production model change.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze_r771_batch_loft as batch
import try_r771_effective_moment as model

OUT=batch.BASE/"no_loft_control"
TRAIN_NAMES=("L110","L27.5")
HOLD_NAME="L55"
TRAIN=(0.0,1.35)
TIME_HOLD=(1.45,2.45)
LATE=(2.45,3.50)
HORIZON=3.50
PARAM_BOUNDS=np.array([[1.,5000.],[0.,150.],[0.,30.]])
STATE_SCALES={"theta":math.radians(1.0),"alpha":math.radians(1.0),
              "omega":.15,"gamma":math.radians(.25)}
EVENTS=(("q_peak1","omega",.35,.58,"max"),
        ("q_valley1","omega",.60,.84,"min"),
        ("q_peak2","omega",.85,1.18,"max"),
        ("alpha_peak1","alpha",.38,.76,"max"))


def creation_pitch(quat):
    """The source quaternion is xyzw; R10=sin(Dagor Euler Z pitch)."""
    x,y,z,w=map(float,quat)
    assert abs(x*x+y*y+z*z+w*w-1)<2e-4
    return math.asin(max(-1.,min(1.,2*(w*z+x*y))))


def clock_mapping(packet_age,b5_age,target_packet_age):
    """Target remains tied to packet stamps while flight age uses b5 ticks."""
    xp=np.r_[0.,packet_age]
    fp=np.r_[0.,b5_age]
    assert np.all(np.diff(xp)>0) and np.all(np.diff(fp)>0)
    mapped=np.interp(target_packet_age,xp,fp)
    left=target_packet_age<xp[0]
    right=target_packet_age>xp[-1]
    mapped[left]=target_packet_age[left]
    mapped[right]=target_packet_age[right]+fp[-1]-xp[-1]
    return mapped


def make_shot(raw,clock="b5"):
    assert raw["p"]["loft_enabled"] is False
    source=raw["r"];creation=raw["case"]["creation"]
    quat=creation["u16_1"]
    initial_velocity=np.array(creation["u12_2"],float)
    initial_position=np.array(creation["starting_pos"],float)
    theta0=creation_pitch(quat)
    omega0=-float(creation["u12_3"][2])
    gamma0=math.atan2(initial_velocity[1],initial_velocity[0])
    speed0=float(np.hypot(initial_velocity[0],initial_velocity[1]))
    init=np.array([theta0,omega0,gamma0,initial_position[0],initial_position[1]],float)
    packet_age=np.array(source["t"],float)
    b5_age=np.array(raw["b5"],float)
    assert np.all(np.diff(packet_age)>0) and np.all(np.diff(b5_age)>0),"L cases must have unambiguous native b5 ticks"
    native_t=b5_age if clock=="b5" else packet_age
    assert native_t[0]>0
    # The release velocity is a decoded creation field, not an extrapolated
    # first sample or a second copy of carrier speed.
    replay=dict(source)
    replay["t"]=np.r_[0.,native_t]
    replay["speed2"]=np.r_[speed0,source["speed2"]]
    if clock=="b5":
        target_packet_age=source["target_t"]-source["missile_packet0_s"]
        replay["target_t"]=clock_mapping(packet_age,b5_age,target_packet_age)
        replay["missile_packet0_s"]=0.0
    else:
        replay["target_t"]=source["target_t"]-source["missile_packet0_s"]
        replay["missile_packet0_s"]=0.0
    # request/force_terms read x/y/theta/etc. only through the evolving state;
    # maintain consistent array lengths for any external inspection.
    for key,value in (("x",initial_position[0]),("y",initial_position[1]),
                      ("theta",theta0),("gamma",gamma0),("alpha",theta0-gamma0),
                      ("omega",omega0)):
        replay[key]=np.r_[value,source[key]]
    observed={key:np.array(source[key],float) for key in ("theta","gamma","alpha","omega","x","y","speed2")}
    observed["t"]=native_t
    return {"name":batch.NAMES[raw["case"]["case"]],"p":dict(raw["p"]),"replay":replay,
            "observed":observed,"initial":init,"initial_speed_mps":speed0,
            "clock":clock,"first_packet_age_s":float(packet_age[0]),
            "first_b5_age_s":float(b5_age[0]),"sample_count":len(native_t),
            "creation_quat":quat,"creation_omega_z_raw":creation["u12_3"][2]}


def passive_interval_estimate(shots):
    rows=[]
    for shot in shots.values():
        o=shot["observed"];init=shot["initial"]
        tt=np.r_[0.,o["t"]]
        alpha=np.r_[init[0]-init[2],o["alpha"]]
        omega=np.r_[init[1],o["omega"]]
        for i in range(len(tt)-1):
            if tt[i+1]>.30+1e-12:break
            dt=tt[i+1]-tt[i]
            rows.append(([-.5*(alpha[i]+alpha[i+1])*dt,-.5*(omega[i]+omega[i+1])*dt],
                         omega[i+1]-omega[i],shot["name"],[float(tt[i]),float(tt[i+1])]))
    x=np.array([r[0] for r in rows]);y=np.array([r[1] for r in rows])
    coeff,_,rank,sv=np.linalg.lstsq(x,y,rcond=None)
    return {"ka_s_minus2":float(coeff[0]),"kw_s_minus1":float(coeff[1]),
            "rank":int(rank),"condition_number":float(sv[0]/sv[-1]),
            "native_interval_count":len(rows),"delta_q_rmse_rad_s":float(np.sqrt(np.mean((x@coeff-y)**2))),
            "meaning":"Linear native-interval diagnostic before gain activation; not a fixed game coefficient."}


def pid_step_with_dt(error_g,state,p,dt):
    """Same fixed raw PID arithmetic as pid_step, with an explicit tick period."""
    integral,previous_error,previous_derivative=state
    provisional=model.clamp(integral+p["ki"]*error_g*dt,-p["integral_limit"],p["integral_limit"])
    raw_derivative=(error_g-previous_error)/dt
    weight=min(1.,dt/(1e-9+dt))
    derivative=previous_derivative+weight*(raw_derivative-previous_derivative)
    output=p["kp"]*error_g+provisional+p["kd"]*derivative
    u=model.clamp(output,-p["fin_aoa_rad"],p["fin_aoa_rad"])
    next_integral=(integral if abs(u)>=.99*p["fin_aoa_rad"] and error_g*u>0 else provisional)
    return u,(next_integral,error_g,derivative)


def simulate(shot,coeff,lifecycle,scaling,plant_dt=None,horizon=HORIZON,
             schedule="20ms_continuous"):
    """RK4 conditional closed loop from creation with one fixed schedule."""
    assert lifecycle in ("updating","frozen","zero_memory_standard_D") and scaling in ("constant","qbar_over_I","constant_control")
    assert schedule in ("20ms_continuous","48hz_end_hold")
    control_dt=(model.CONTROL_DT if schedule=="20ms_continuous" else 1/48)
    if schedule=="48hz_end_hold":assert lifecycle in ("updating","zero_memory_standard_D")
    if lifecycle=="zero_memory_standard_D":assert schedule=="48hz_end_hold"
    if plant_dt is None:plant_dt=control_dt/(2 if schedule=="20ms_continuous" else 4)
    assert abs(round(control_dt/plant_dt)*plant_dt-control_dt)<1e-12
    p=shot["p"];r=shot["replay"]
    state=shot["initial"].copy()
    pid=(0.,0.,0.);armed=False
    rows=[]
    reference_mass,_=model.mass_thrust(0.,p)
    reference_q=.5*model.isa_density(state[4])*shot["initial_speed_mps"]**2
    reference_scale=reference_q/reference_mass
    reference_speed=shot["initial_speed_mps"]
    ku,ka,kw=map(float,coeff)

    def rhs(t,s,raw_held,held_u):
        th,om,ga,x,y=s
        f=model.force_terms(t,s,r,p)
        speed=max(f["v"],1e-6)
        # Current baseline: gain acts continuously on a 20 ms held PID value.
        # Fixed new candidate: gain at the 1/48 s interval END is held over it.
        u=(model.interpolate_table(t,p["time_gain"])*raw_held
           if schedule=="20ms_continuous" else held_u)
        if scaling=="constant":
            active_scale=1.;damping_scale=1.
        else:
            q=.5*model.isa_density(y)*speed*speed
            active_scale=(q/f["mass"])/reference_scale
            damping_scale=active_scale*reference_speed/speed
        control_scale=1. if scaling=="constant_control" else active_scale
        qdot=control_scale*ku*u-active_scale*ka*(th-ga)-damping_scale*kw*om
        gamma_dot=model.G*(f["actual_specific_force_g"]-math.cos(ga))/speed
        return np.array([om,qdot,gamma_dot,speed*math.cos(ga),speed*math.sin(ga)])

    steps=round(control_dt/plant_dt)
    tick_count=round(horizon/control_dt)
    for tick in range(tick_count+1):
        t=tick*control_dt
        if not np.isfinite(state).all() or abs(state[0]-state[2])>1.2 or abs(state[2])>1.0:
            return None
        req=model.request(t,state,r,p)
        if lifecycle=="updating":
            if tick==0:pid=(0.,req["error_g"],0.)  # no dormant D kick
            if schedule=="20ms_continuous":
                raw_held,pid,_=model.pid_step(req["error_g"],pid,p)
            else:
                raw_held,pid=pid_step_with_dt(req["error_g"],pid,p,control_dt)
        elif lifecycle=="zero_memory_standard_D":
            gain_at_end=model.interpolate_table(t+control_dt,p["time_gain"])
            if gain_at_end<=0:
                raw_held=0.  # Preserve I, previous error, D memory at zero.
            else:
                raw_held,pid=pid_step_with_dt(req["error_g"],pid,p,control_dt)
        else:
            u_at_tick,pid,_,armed=model.scheduled_pid_step(t,req,pid,p,armed)
            raw_held=u_at_tick/req["flight_gain"] if req["flight_gain"]>0 else 0.
        gain_time=t if schedule=="20ms_continuous" else t+control_dt
        u=model.interpolate_table(gain_time,p["time_gain"])*raw_held
        th,om,ga,x,y=state
        rows.append({"t":t,"theta":th,"omega":om,"gamma":ga,"alpha":th-ga,
                     "x":x,"y":y,"u":u,"raw_pid_held":raw_held,"I":pid[0],"a_req_g":req["a_req_g"],
                     "a_feedback_g":req["actual_g"],"pn_g":req["pn_g"],
                     "gain":req["flight_gain"]})
        if tick==tick_count:break
        for k in range(steps):
            ts=t+k*plant_dt
            d1=rhs(ts,state,raw_held,u)
            d2=rhs(ts+plant_dt/2,state+d1*plant_dt/2,raw_held,u)
            d3=rhs(ts+plant_dt/2,state+d2*plant_dt/2,raw_held,u)
            d4=rhs(ts+plant_dt,state+d3*plant_dt,raw_held,u)
            state=state+plant_dt*(d1+2*d2+2*d3+d4)/6
    return rows


def event(t,y,lo,hi,kind):
    ids=np.flatnonzero((t>=lo-1e-10)&(t<=hi+1e-10))
    i=int(ids[np.argmax(y[ids]) if kind=="max" else np.argmin(y[ids])])
    return {"time_s":float(t[i]),"value":float(y[i])}


def event_table(shot,rows):
    t=np.array([v["t"] for v in rows]);o=shot["observed"]
    out={}
    for label,key,lo,hi,kind in EVENTS:
        out[label]={"observed":event(o["t"],o[key],lo,hi,kind),
                    "forward":event(t,np.array([v[key] for v in rows]),lo,hi,kind)}
        out[label]["time_error_s"]=out[label]["forward"]["time_s"]-out[label]["observed"]["time_s"]
        out[label]["relative_amplitude_error"]=((out[label]["forward"]["value"]-
            out[label]["observed"]["value"])/abs(out[label]["observed"]["value"]))
    return out


def crossings(t,y,lo,hi):
    found=[]
    for i in range(len(t)-1):
        if y[i]*y[i+1]>=0:continue
        crossing=float(t[i]-y[i]*(t[i+1]-t[i])/(y[i+1]-y[i]))
        if lo<=crossing<=hi:found.append(crossing)
    return found


def compare(shot,rows):
    o=shot["observed"];pt=np.array([v["t"] for v in rows])
    result={"events":event_table(shot,rows),"windows":{}}
    for label,(lo,hi) in (("passive",(0.,.30)),("train",TRAIN),
                           ("time_hold",TIME_HOLD),("late",LATE)):
        mask=(o["t"]>=lo-1e-10)&(o["t"]<=hi+1e-10)
        rec={}
        for key in ("theta","alpha","omega","gamma","y"):
            pred=np.interp(o["t"][mask],pt,[v[key] for v in rows])
            target=o[key][mask]
            rec[key+"_rmse"] = float(np.sqrt(np.mean((pred-target)**2))) if len(target) else None
            rec[key+"_max_abs"] = float(np.max(np.abs(pred-target))) if len(target) else None
        rec["native_sample_count"]=int(np.sum(mask))
        result["windows"][label]=rec
    oq=crossings(o["t"],o["omega"],.30,2.45)
    pq=crossings(pt,np.array([v["omega"] for v in rows]),.30,2.45)
    result["omega_zero_crossings_0p3_2p45"]={"observed":oq,"forward":pq}
    last_passive=int(np.flatnonzero(o["t"]<=.30)[-1])
    passive_time=float(o["t"][last_passive])
    result["last_native_passive_q"]={"time_s":passive_time,
        "observed_q":float(o["omega"][last_passive]),
        "forward_q":float(np.interp(passive_time,pt,[v["omega"] for v in rows]))}
    return result


def activation_interval(shot,rows,coeff,scaling):
    """Check the native interval straddling the raw .30-.31 gain ramp."""
    o=shot["observed"];t=o["t"]
    i=int(np.flatnonzero((t[:-1]<.30)&(t[1:]>.31))[0])
    lo,hi=map(float,(t[i],t[i+1]))
    observed_dq=float(o["omega"][i+1]-o["omega"][i])
    ft=np.array([v["t"] for v in rows]);fq=np.array([v["omega"] for v in rows])
    forward_dq=float(np.interp(hi,ft,fq)-np.interp(lo,ft,fq))
    alpha=.5*(o["alpha"][i]+o["alpha"][i+1])
    omega=.5*(o["omega"][i]+o["omega"][i+1])
    speed=.5*(o["speed2"][i]+o["speed2"][i+1])
    height=.5*(o["y"][i]+o["y"][i+1])
    mass,_=model.mass_thrust(.5*(lo+hi),shot["p"])
    mass0,_=model.mass_thrust(0.,shot["p"])
    qref=.5*model.isa_density(shot["initial"][4])*shot["initial_speed_mps"]**2
    active=((.5*model.isa_density(height)*speed*speed/mass)/(qref/mass0)
            if scaling=="qbar_over_I" else 1.)
    damping=active*shot["initial_speed_mps"]/speed if scaling=="qbar_over_I" else 1.
    passive_qdot=-active*coeff[1]*alpha-damping*coeff[2]*omega
    # Integral of the unchanged raw timeToGain table over this native interval.
    gain_integral=.5*(.31-.30)+(hi-.31)
    required_raw=(observed_dq-passive_qdot*(hi-lo))/(coeff[0]*active*gain_integral)
    raw_at_030=next(v["raw_pid_held"] for v in rows if abs(v["t"]-.30)<1e-9)
    return {"native_interval_s":[lo,hi],"observed_delta_q_rad_s":observed_dq,
            "forward_delta_q_rad_s":forward_dq,
            "observed_mean_qdot_rad_s2":observed_dq/(hi-lo),
            "forward_mean_qdot_rad_s2":forward_dq/(hi-lo),
            "assumed_passive_mean_qdot_rad_s2":float(passive_qdot),
            "required_mean_control_qdot_rad_s2":float(observed_dq/(hi-lo)-passive_qdot),
            "gain_integral_s":gain_integral,
            "equivalent_required_held_raw_pid_rad":float(required_raw),
            "forward_held_raw_pid_at_0p30_rad":float(raw_at_030),
            "interpretation":"Required raw PID is conditional on selected coefficients, measured midpoint state, zero active input before .30, and continuous raw timeToGain ramp; not decoded game command."}


def state_loss(shot,rows,bounds):
    o=shot["observed"];pt=np.array([v["t"] for v in rows])
    mask=(o["t"]>=bounds[0]-1e-10)&(o["t"]<=bounds[1]+1e-10)
    if not np.any(mask):return math.inf
    terms=[]
    for key,scale in STATE_SCALES.items():
        pred=np.interp(o["t"][mask],pt,[v[key] for v in rows])
        terms.append(np.mean(((pred-o[key][mask])/scale)**2))
    return float(np.mean(terms))


def fit_objective(shots,coeff,lifecycle,scaling,schedule="20ms_continuous"):
    scores=[]
    for shot in shots.values():
        rows=simulate(shot,coeff,lifecycle,scaling,horizon=2.5,schedule=schedule)
        if rows is None:return 1e8
        passive=state_loss(shot,rows,(0.,.30))
        active=state_loss(shot,rows,(.30,1.35))
        events=event_table(shot,rows)
        event_penalty=0.
        for label,record in events.items():
            amplitude_scale=.04 if label.startswith("q_") else math.radians(.4)
            event_penalty+=(record["time_error_s"]/.042)**2
            event_penalty+=((record["forward"]["value"]-record["observed"]["value"])/amplitude_scale)**2
        event_penalty/=len(events)*2
        scores.append(.7*passive+active+.20*event_penalty)
    return float(np.mean(scores))


def optimize(shots,lifecycle,scaling,passive,schedule="20ms_continuous",seed_override=None):
    # Fixed bounded coordinate search; no timing, PID or guide parameters move.
    seeds=((550.,passive["ka_s_minus2"],passive["kw_s_minus1"]),
           (850.,68.,3.5)) if seed_override is None else (tuple(seed_override),
                (550.,passive["ka_s_minus2"],passive["kw_s_minus1"]))
    best=None;eval_count=0
    cache={}
    def cost(z):
        nonlocal eval_count
        z=np.clip(z,PARAM_BOUNDS[:,0],PARAM_BOUNDS[:,1])
        key=tuple(np.round(z,5))
        if key not in cache:
            cache[key]=fit_objective(shots,np.array(key),lifecycle,scaling,schedule)
            eval_count+=1
        return cache[key]
    for seed in seeds:
        x=np.array(seed,float)
        value=cost(x)
        step=np.array([260.,14.,2.])
        for _ in range(32):
            proposals=[]
            for axis in range(3):
                for direction in (-1.,1.):
                    trial=x.copy();trial[axis]+=direction*step[axis]
                    trial=np.clip(trial,PARAM_BOUNDS[:,0],PARAM_BOUNDS[:,1])
                    proposals.append((cost(trial),trial))
            next_value,next_x=min(proposals,key=lambda z:z[0])
            if next_value<value-1e-8:x,value=next_x,next_value
            else:step*=.55
            if max(step/[15.,1.,.12])<1:break
        if best is None or value<best[0]:best=(value,x)
    assert best is not None
    coeff=best[1]
    return coeff,{"training_objective":best[0],"evaluations":eval_count,
                  "bounds":PARAM_BOUNDS.tolist(),"seeds":[list(x) for x in seeds],
                  "weights":{"passive_state":.7,"active_state":1.,"event_penalty":.20},
                  "state_scales":STATE_SCALES,
                  "event_time_scale_s":.042,"event_omega_amplitude_scale_rad_s":.04,
                  "event_alpha_amplitude_scale_deg":.4,
                  "boundary_hit":[bool(coeff[j]<=PARAM_BOUNDS[j,0]+1e-6 or coeff[j]>=PARAM_BOUNDS[j,1]-1e-6)
                                  for j in range(3)]}


def candidate_scores(shots,coeff,lifecycle,scaling,schedule="20ms_continuous"):
    out={}
    for name,shot in shots.items():
        rows=simulate(shot,coeff,lifecycle,scaling,schedule=schedule)
        out[name]=compare(shot,rows)
    train=np.mean([state_loss(shots[n],simulate(shots[n],coeff,lifecycle,scaling,schedule=schedule),TRAIN) for n in shots])
    hold=np.mean([state_loss(shots[n],simulate(shots[n],coeff,lifecycle,scaling,schedule=schedule),TIME_HOLD) for n in shots])
    return out,{"train_scaled_state_mse":float(train),"time_hold_scaled_state_mse":float(hold)}


def plot_result(shots,forward,selected):
    fig,axs=plt.subplots(3,3,figsize=(17,12),sharex=True,layout="constrained")
    keys=(("theta","θ (deg)",180/math.pi),("alpha","signed α (deg)",180/math.pi),
          ("omega","body pitch rate ω (rad/s)",1),("gamma","γ (deg)",180/math.pi),
          ("y","height gain (m)",1),("u","PID output u (rad)",1),
          ("a_req_g","PN + gravity request (g)",1),("a_feedback_g","modeled wind-normal feedback (g)",1),
          ("I","PID integral term (rad)",1))
    colors={"L110":"#2369a8","L27.5":"#d27821","L55":"#8d4bb0"}
    for ax,(key,title,scale) in zip(axs.flat,keys):
        for name in (*TRAIN_NAMES,HOLD_NAME):
            shot=shots[name];r=forward[name];o=shot["observed"]
            ct=np.array([v["t"] for v in r]);cv=np.array([v[key] for v in r])
            if key=="y":cv=cv-shot["initial"][4]
            ax.plot(ct,cv*scale,color=colors[name],lw=1.35,label=name+" forward")
            if key in o:
                ov=o[key]-(shot["initial"][4] if key=="y" else 0.)
                ax.plot(o["t"],ov*scale,color=colors[name],lw=0,marker=".",ms=4,
                        label=name+" native")
        ax.axvspan(*TRAIN,color="#dcebf6",alpha=.32)
        ax.axvspan(*TIME_HOLD,color="#f6e5cf",alpha=.42)
        ax.set(title=title,xlim=(0,2.5),xlabel="b5 physical age from creation (s)")
        ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=8,ncol=2)
    fig.suptitle("No-loft conditional closed-loop: "+selected+"\n"
                 "L55 excluded from current numerical fit/selection; earlier L55 mismatch motivated this timing check",fontsize=14)
    fig.savefig(OUT/"no_loft_forward.png",dpi=180)
    plt.close(fig)


def plot_activation(shots,forward):
    fig,axs=plt.subplots(2,2,figsize=(13,8),sharex=True,layout="constrained")
    colors={"L110":"#2369a8","L27.5":"#d27821","L55":"#8d4bb0"}
    for name in (*TRAIN_NAMES,HOLD_NAME):
        o=shots[name]["observed"];rows=forward[name];t=np.array([v["t"] for v in rows])
        axs[0,0].plot(o["t"],o["omega"],".",color=colors[name],ms=7,label=name+" native")
        axs[0,0].plot(t,[v["omega"] for v in rows],color=colors[name],lw=1.4,label=name+" forward")
        axs[0,1].plot(t,[v["raw_pid_held"] for v in rows],color=colors[name],lw=1.4,label=name)
        axs[1,0].plot(t,[v["u"] for v in rows],color=colors[name],lw=1.4,label=name)
        axs[1,1].plot(t,[v["a_req_g"]-v["a_feedback_g"] for v in rows],color=colors[name],lw=1.4,label=name)
    axs[0,0].set_title("Pitch-rate onset: source dots versus continuous forward")
    axs[0,1].set_title("Raw PID output held at 20 ms updates (rad)")
    axs[1,0].set_title("u = timeToGain(t) × held PID output at ticks (rad)")
    axs[1,1].set_title("Forward normal specific-force error (g)")
    for ax in axs.flat:
        ax.axvspan(.30,.31,color="#f4d48e",alpha=.5)
        ax.axvline(.3125,color="0.3",ls=":",lw=.8)
        ax.set(xlim=(.24,.48),xlabel="b5 physical age from creation (s)")
        ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=8,ncol=2)
    fig.suptitle("No-loft activation detail; shaded: raw timeToGain ramp 0→1",fontsize=14)
    fig.savefig(OUT/"activation_detail.png",dpi=180)
    plt.close(fig)


def onset_changes(shot,rows):
    observed=shot["observed"];t=observed["t"]
    i=int(np.flatnonzero((t[:-1]<.30)&(t[1:]>.31))[0])
    lo,hi=map(float,(t[i],t[i+1]))
    rt=np.array([row["t"] for row in rows])
    result={"native_interval_s":[lo,hi]}
    for key in ("theta","omega"):
        actual=float(observed[key][i+1]-observed[key][i])
        predicted=float(np.interp(hi,rt,[row[key] for row in rows])-
                        np.interp(lo,rt,[row[key] for row in rows]))
        result[key]={"observed_delta":actual,"forward_delta":predicted,
                     "absolute_delta_error":abs(predicted-actual)}
    return result


def onset_control_requirement(shot,rows,coeff):
    """Conditional force/moment balance across the native gain-onset interval."""
    o=shot["observed"];t=o["t"]
    i=int(np.flatnonzero((t[:-1]<.30)&(t[1:]>.31))[0])
    lo,hi=map(float,(t[i],t[i+1]))
    dt=hi-lo
    omega=.5*(o["omega"][i]+o["omega"][i+1])
    alpha=.5*(o["alpha"][i]+o["alpha"][i+1])
    speed=.5*(o["speed2"][i]+o["speed2"][i+1])
    height=.5*(o["y"][i]+o["y"][i+1])
    mass,_=model.mass_thrust(.5*(lo+hi),shot["p"])
    mass0,_=model.mass_thrust(0.,shot["p"])
    qref=.5*model.isa_density(shot["initial"][4])*shot["initial_speed_mps"]**2
    active_scale=(.5*model.isa_density(height)*speed*speed/mass)/(qref/mass0)
    damping_scale=active_scale*shot["initial_speed_mps"]/speed
    passive=-active_scale*coeff[1]*alpha-damping_scale*coeff[2]*omega
    obs_delta=float(o["omega"][i+1]-o["omega"][i])
    required_mean_control=obs_delta/dt-passive
    active_duration=1/48
    required_active_qdot=required_mean_control*dt/active_duration
    raw_at_14=next(row["raw_pid_held"] for row in rows if abs(row["t"]-14/48)<1e-9)
    modeled_active_qdot=active_scale*coeff[0]*raw_at_14
    return {"native_interval_s":[lo,hi],"observed_delta_omega_rad_s":obs_delta,
            "observed_mean_qdot_rad_s2":obs_delta/dt,
            "assumed_passive_mean_qdot_rad_s2":float(passive),
            "required_mean_control_qdot_over_full_interval_rad_s2":float(required_mean_control),
            "required_mean_control_qdot_if_only_last_1_over_48_active_rad_s2":float(required_active_qdot),
            "modeled_active_control_qdot_at_14_over_48_rad_s2":float(modeled_active_qdot),
            "forward_raw_pid_at_14_over_48_rad":float(raw_at_14),
            "equivalent_required_raw_pid_rad":float(required_active_qdot/(active_scale*coeff[0])),
            "meaning":"Uses measured midpoint alpha/omega/V and the selected three coefficients; conditional diagnostic, not a decoded game control signal."}


def plot_schedule_comparison(shots,baseline,new_fixed,new_fit,out,scenario="1/48 s END gain"):
    fig,axs=plt.subplots(3,3,figsize=(17,12),sharex=True,layout="constrained")
    for col,name in enumerate((*TRAIN_NAMES,HOLD_NAME)):
        shot=shots[name];obs=shot["observed"]
        for row,(key,scale) in enumerate((("theta",180/math.pi),("omega",1),("alpha",180/math.pi))):
            ax=axs[row,col]
            visible=obs["t"]<=2.5
            ax.plot(obs["t"][visible],obs[key][visible]*scale,"k.",ms=4,
                    label="native" if row==0 and col==0 else None)
            for rows,color,style,label in ((baseline[name],"#9a7440",":","20 ms continuous gain"),
                                            (new_fit[name],"#2467a4","-",scenario+"; refit")):
                tt=np.array([v["t"] for v in rows]);keep=tt<=2.5
                vv=np.array([v[key] for v in rows])
                ax.plot(tt[keep],vv[keep]*scale,color=color,ls=style,lw=1.4,
                        label=label if row==0 and col==0 else None)
            if name in new_fixed:
                rows=new_fixed[name];tt=np.array([v["t"] for v in rows]);keep=tt<=2.5
                vv=np.array([v[key] for v in rows])
                ax.plot(tt[keep],vv[keep]*scale,color="#b5476a",ls="--",lw=1.0,
                        label=scenario+"; same coefficients" if row==0 and col==0 else None)
            ax.axvspan(*TRAIN,color="#dcebf6",alpha=.30)
            ax.axvspan(*TIME_HOLD,color="#f6e5cf",alpha=.35)
            ax.set(xlim=(0,2.5),title=(name+" — "+key),xlabel="b5 age from creation (s)")
            ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=8,loc="best")
    fig.suptitle("No-loft fixed scheduling check ("+scenario+"): source and three-parameter conditional forward\n"
                 "L55 excluded from numeric refit; its earlier mismatch had already been inspected",fontsize=14)
    fig.savefig(out/"schedule_forward.png",dpi=180)
    plt.close(fig)

    fig,axs=plt.subplots(2,2,figsize=(13,8),sharex=True,layout="constrained")
    for col,name in enumerate(TRAIN_NAMES):
        shot=shots[name];obs=shot["observed"]
        for row,(key,scale) in enumerate((("theta",180/math.pi),("omega",1))):
            ax=axs[row,col]
            visible=(obs["t"]>=.24)&(obs["t"]<=.50)
            ax.plot(obs["t"][visible],obs[key][visible]*scale,"ko",ms=5,label="native")
            for series,color,style,label in ((baseline[name],"#9a7440",":","20 ms continuous"),
                                              (new_fixed[name],"#b5476a","--",scenario+" fixed coeff"),
                                              (new_fit[name],"#2467a4","-",scenario+" refit")):
                tt=np.array([v["t"] for v in series]);keep=(tt>=.24)&(tt<=.50)
                ax.plot(tt[keep],np.array([v[key] for v in series])[keep]*scale,
                        color=color,ls=style,lw=1.5,label=label)
            ax.axvspan(14/48,15/48,color="#f4d48e",alpha=.4)
            ax.set(xlim=(.24,.50),title=name+" — "+key,xlabel="b5 age from creation (s)")
            ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=8)
    fig.suptitle("Native onset interval: 0.2708–0.3125 s; shaded active 1/48 s interval",fontsize=14)
    fig.savefig(out/"schedule_onset.png",dpi=180)
    plt.close(fig)


def schedule_48hz_experiment():
    out=OUT/"schedule_48hz"
    out.mkdir(parents=True,exist_ok=True)
    raw,rawp=batch.load_all()
    assert rawp["source_version"]=="2.59.0.28"
    assert [rawp[k] for k in ("kp","ki","kd","integral_limit")]==[.002125,.006481,.000023,1.]
    assert rawp["pn_gain"]==6 and rawp["time_gain"]==[[.30,0.],[.31,1.]]
    state=(.001,.4,.02)
    test_new=pid_step_with_dt(2.3,state,raw["L110"]["p"],.02)
    test_old=model.pid_step(2.3,state,raw["L110"]["p"])
    assert abs(test_new[0]-test_old[0])<1e-12 and np.max(np.abs(np.array(test_new[1])-np.array(test_old[1])))<1e-12
    prior=json.loads((OUT/"results.json").read_text())
    coeff0=np.array(prior["selected_before_L55"]["coefficients"],float)
    assert prior["selected_before_L55"]["name"]=="updating_qbar_over_I"
    train={name:make_shot(raw[name],"b5") for name in TRAIN_NAMES}
    assert all(not shot["p"]["loft_enabled"] for shot in train.values())
    baseline={name:simulate(shot,coeff0,"updating","qbar_over_I") for name,shot in train.items()}
    fixed={name:simulate(shot,coeff0,"updating","qbar_over_I",schedule="48hz_end_hold")
           for name,shot in train.items()}
    onset_before={name:onset_changes(train[name],baseline[name]) for name in TRAIN_NAMES}
    onset_fixed={name:onset_changes(train[name],fixed[name]) for name in TRAIN_NAMES}
    # A constant q until 14/48 and a linear ramp to measured endpoint over
    # 14/48--15/48 predicts the measured negative Δtheta without changing the
    # angle integrator. This is evidence for a late response, not its source.
    native_theta_check={}
    for name,shot in train.items():
        o=shot["observed"];tt=o["t"]
        i=int(np.flatnonzero((tt[:-1]<.30)&(tt[1:]>.31))[0]);h=1/48
        piecewise_delta=h*(1.5*o["omega"][i]+.5*o["omega"][i+1])
        native_theta_check[name]={"observed_delta_theta_rad":float(o["theta"][i+1]-o["theta"][i]),
                                  "late_linear_rate_piecewise_delta_theta_rad":float(piecewise_delta),
                                  "difference_rad":float(piecewise_delta-(o["theta"][i+1]-o["theta"][i]))}
    improved=all(onset_fixed[name][key]["absolute_delta_error"]<onset_before[name][key]["absolute_delta_error"]
                 for name in TRAIN_NAMES for key in ("theta","omega"))
    result={"candidate":"1/48 s controller/physics update; PID evaluates interval-beginning state with dt=1/48, original timeToGain evaluated at interval END and held over that interval; RK4 theta_dot=omega, no fitted delay/phase.",
            "fixed_raw":{"kp":rawp["kp"],"ki":rawp["ki"],"kd":rawp["kd"],
                         "integral_limit":rawp["integral_limit"],"pn_gain":rawp["pn_gain"],
                         "time_gain":rawp["time_gain"],"loft_enabled":False,"lift_slope_per_rad":14.},
            "external_inputs":"Measured V and packet-associated target trajectory; all missile states and PID memory integrate from creation.",
            "prior_coefficients":coeff0.tolist(),"onset_same_coeff_current_20ms":onset_before,
            "onset_same_coeff_48hz_end_hold":onset_fixed,
            "native_theta_integral_check":native_theta_check,
            "same_coefficient_onset_improved_both_theta_and_omega_on_two_train_shots":improved,
            "pid_arithmetic_same_input_20ms_error_below_1e_minus12":True,
            "train_cases":list(TRAIN_NAMES),"time_hold_window_s":list(TIME_HOLD),
            "L55_provenance":"No L55 numeric fit or schedule choice in this run; its earlier mismatch was inspected before this scheduling hypothesis, so it is not a pristine blind holdout."}
    if not improved:
        result["stop_reason"]="Same-coefficient 48 Hz endpoint gate failed the predeclared two-shot onset improvement; no refit or L55 evaluation."
        (out/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
        print(json.dumps({"improved":False,"onset":onset_fixed},indent=2))
        return
    passive=passive_interval_estimate(train)
    coeff,opt=optimize(train,"updating","qbar_over_I",passive,
                       schedule="48hz_end_hold",seed_override=coeff0)
    train_scores,aggregate=candidate_scores(train,coeff,"updating","qbar_over_I",schedule="48hz_end_hold")
    # The coefficients are now fixed. Only here is the L55 case constructed.
    hold=make_shot(raw[HOLD_NAME],"b5")
    assert not hold["p"]["loft_enabled"]
    shots=dict(train,**{HOLD_NAME:hold})
    refit={name:simulate(shot,coeff,"updating","qbar_over_I",schedule="48hz_end_hold")
           for name,shot in shots.items()}
    baseline[HOLD_NAME]=simulate(hold,coeff0,"updating","qbar_over_I")
    fixed[HOLD_NAME]=simulate(hold,coeff0,"updating","qbar_over_I",schedule="48hz_end_hold")
    assert all(rows is not None and all(np.isfinite([row[k] for k in ("theta","omega","gamma","x","y","u","I")]).all()
                                     for row in rows) for rows in refit.values())
    hold_score=compare(hold,refit[HOLD_NAME])
    step={}
    for name,shot in shots.items():
        finer=simulate(shot,coeff,"updating","qbar_over_I",plant_dt=1/384,
                       schedule="48hz_end_hold")
        step[name]={key:float(max(abs(a[key]-b[key]) for a,b in zip(refit[name],finer)))
                    for key in ("theta","omega","gamma","y")}
    plot_schedule_comparison(shots,baseline,fixed,refit,out)
    result.update({"passive_native_interval_fit":passive,"refit_coefficients":coeff.tolist(),
                   "refit_optimization":opt,"train_case_checks":train_scores,
                   "train_aggregate":aggregate,
                   "onset_refit_48hz_end_hold":{name:onset_changes(shots[name],refit[name]) for name in shots},
                   "onset_required_control_acceleration":{
                       name:onset_control_requirement(shots[name],refit[name],coeff) for name in shots},
                   "L55_evaluation":hold_score,"step_halving_same_48hz_control_period":step,
                   "native_sample_counts":{name:shot["sample_count"] for name,shot in shots.items()},
                   "figures":["schedule_forward.png","schedule_onset.png"]})
    (out/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"improved":improved,"same_coeff_onset":onset_fixed,
                      "refit_coeff":coeff.tolist(),"refit_onset":result["onset_refit_48hz_end_hold"],
                      "L55_events":hold_score["events"]},indent=2))


def standard_D_experiment(scaling="qbar_over_I"):
    """One fixed zero-memory PID lifecycle under the existing 1/48 end gate."""
    assert scaling in ("constant","qbar_over_I","constant_control")
    out=OUT/"schedule_48hz"/("zero_memory_standard_D" if scaling=="qbar_over_I"
                             else "zero_memory_standard_D_"+scaling)
    out.mkdir(parents=True,exist_ok=True)
    raw,rawp=batch.load_all()
    assert rawp["source_version"]=="2.59.0.28"
    assert [rawp[k] for k in ("kp","ki","kd","integral_limit")]==[.002125,.006481,.000023,1.]
    assert rawp["pn_gain"]==6 and rawp["time_gain"]==[[.30,0.],[.31,1.]]
    train={name:make_shot(raw[name],"b5") for name in TRAIN_NAMES}
    assert all(not shot["p"]["loft_enabled"] for shot in train.values())
    prior=json.loads((OUT/"results.json").read_text())
    frozen=prior["candidates"]["frozen_"+("qbar_over_I" if scaling=="constant_control" else scaling)]
    coeff0=np.array(frozen["coefficients"],float)
    old={name:simulate(shot,coeff0,"frozen",scaling) for name,shot in train.items()}
    warm={name:simulate(shot,coeff0,"updating",scaling,schedule="48hz_end_hold")
          for name,shot in train.items()}
    standard={name:simulate(shot,coeff0,"zero_memory_standard_D",scaling,schedule="48hz_end_hold")
              for name,shot in train.items()}
    onset={label:{name:onset_changes(train[name],series[name]) for name in TRAIN_NAMES}
           for label,series in (("old_20ms_frozen",old),("48hz_warm_memory",warm),
                                ("48hz_zero_memory_standard_D",standard))}
    improved=all(onset["48hz_zero_memory_standard_D"][name][key]["absolute_delta_error"]
                 <onset["old_20ms_frozen"][name][key]["absolute_delta_error"]
                 for name in TRAIN_NAMES for key in ("theta","omega"))
    first_active={}
    for name,shot in train.items():
        row=next(v for v in standard[name] if abs(v["t"]-14/48)<1e-10)
        req=model.request(14/48,np.array([row[k] for k in ("theta","omega","gamma","x","y")]),
                          shot["replay"],shot["p"])
        p=shot["p"];e=req["error_g"]
        pieces={"error_g":e,"P_rad":p["kp"]*e,
                "I_rad":p["ki"]*e/48,"D_rad":p["kd"]*e*48/(1.+48e-9),
                "raw_PID_rad":row["raw_pid_held"],"gain_at_interval_end":row["u"]/row["raw_pid_held"],
                "PID_memory_at_first_active":row["I"]}
        assert abs(sum(pieces[k] for k in ("P_rad","I_rad","D_rad"))-row["raw_pid_held"])<1e-10
        assert all(abs(v["I"])<1e-15 and abs(v["raw_pid_held"])<1e-15
                   for v in standard[name] if v["t"]<14/48-1e-10)
        first_active[name]=pieces
    result={"candidate":"1/48 s interval-beginning PID; raw timeToGain at interval END held over interval. Memory (I, previous error, derivative) remains zero until first positive gain, then ordinary raw PID arithmetic including first D difference from zero. RK4 theta_dot=omega; no timing or PID fit.",
            "moment_scaling":scaling,
            "fixed_raw":{"kp":rawp["kp"],"ki":rawp["ki"],"kd":rawp["kd"],
                         "integral_limit":rawp["integral_limit"],"pn_gain":rawp["pn_gain"],
                         "time_gain":rawp["time_gain"],"loft_enabled":False,"lift_slope_per_rad":14.},
            "external_inputs":"Measured speed and packet-associated target trajectory; missile states and PID integrate continuously from creation.",
            "prior_frozen_coefficients":coeff0.tolist(),"same_coeff_onset":onset,
            "first_active_PID_terms":first_active,
            "same_coeff_onset_improved_vs_old_frozen_on_both_train_shots":improved,
            "train_cases":list(TRAIN_NAMES),"time_hold_window_s":list(TIME_HOLD),
            "L55_provenance":"No L55 numerical fit or selection in this run. Its previous mismatch was already inspected, so this is not a pristine blind holdout."}
    if not improved:
        result["stop_reason"]="Same-coefficient native onset did not improve in both theta and omega over frozen 20ms; no refit or L55 evaluation."
        (out/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
        print(json.dumps({"improved":False,"onset":onset,"PID":first_active},indent=2))
        return
    passive=passive_interval_estimate(train)
    coeff,opt=optimize(train,"zero_memory_standard_D",scaling,passive,
                       schedule="48hz_end_hold",seed_override=coeff0)
    train_scores,aggregate=candidate_scores(train,coeff,"zero_memory_standard_D",scaling,
                                            schedule="48hz_end_hold")
    # From here the shared coefficients are locked; L55 is constructed once.
    hold=make_shot(raw[HOLD_NAME],"b5")
    assert not hold["p"]["loft_enabled"]
    shots=dict(train,**{HOLD_NAME:hold})
    refit={name:simulate(shot,coeff,"zero_memory_standard_D",scaling,schedule="48hz_end_hold")
           for name,shot in shots.items()}
    old[HOLD_NAME]=simulate(hold,coeff0,"frozen",scaling)
    standard[HOLD_NAME]=simulate(hold,coeff0,"zero_memory_standard_D",scaling,schedule="48hz_end_hold")
    assert all(rows is not None and all(np.isfinite([row[k] for k in ("theta","omega","gamma","x","y","u","I")]).all()
                                     for row in rows) for rows in refit.values())
    step={}
    for name,shot in shots.items():
        fine=simulate(shot,coeff,"zero_memory_standard_D",scaling,
                      plant_dt=1/384,schedule="48hz_end_hold")
        step[name]={key:float(max(abs(a[key]-b[key]) for a,b in zip(refit[name],fine)))
                    for key in ("theta","omega","gamma","y")}
    plot_schedule_comparison(shots,old,standard,refit,out,
                             scenario="1/48 END + zero memory + natural D; "+scaling)
    result.update({"passive_native_interval_fit":passive,"refit_coefficients":coeff.tolist(),
                   "refit_optimization":opt,"train_case_checks":train_scores,
                   "train_aggregate":aggregate,
                   "onset_refit":{name:onset_changes(shots[name],refit[name]) for name in shots},
                   "L55_evaluation":compare(hold,refit[HOLD_NAME]),
                   "step_halving_same_48hz_control_period":step,
                   "native_sample_counts":{name:shot["sample_count"] for name,shot in shots.items()},
                   "figures":["schedule_forward.png","schedule_onset.png"]})
    (out/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"same_coeff_onset":onset,"first_active_PID":first_active,
                      "refit_coeff":coeff.tolist(),"refit_onset":result["onset_refit"],
                      "train_events":{name:train_scores[name]["events"] for name in TRAIN_NAMES},
                      "L55_events":result["L55_evaluation"]["events"]},indent=2))


def plot_review():
    """Recreate a clean cross-shot check from the saved candidate; no fitting."""
    saved=json.loads((OUT/"schedule_48hz"/"zero_memory_standard_D"/"results.json").read_text())
    raw,_=batch.load_all()
    coeff=np.array(saved["refit_coefficients"])
    shots={name:make_shot(raw[name],"b5") for name in (*TRAIN_NAMES,HOLD_NAME)}
    rows={name:simulate(shot,coeff,"zero_memory_standard_D","qbar_over_I",
                        schedule="48hz_end_hold") for name,shot in shots.items()}
    # Verify the saved result still describes the current implementation.
    for name,shot in shots.items():
        expected=(saved["L55_evaluation"] if name==HOLD_NAME else saved["train_case_checks"][name])
        current=compare(shot,rows[name])
        for label in EVENTS:
            key=label[0]
            for field in ("time_s","value"):
                assert abs(current["events"][key]["forward"][field]-expected["events"][key]["forward"][field])<1e-10
    fig,axs=plt.subplots(2,3,figsize=(14,7),sharex=True,layout="constrained")
    detail={}
    for col,name in enumerate((*TRAIN_NAMES,HOLD_NAME)):
        o=shots[name]["observed"];rr=rows[name]
        pt=np.array([r["t"] for r in rr])
        for row,(key,scale,ylabel) in enumerate((("alpha",180/math.pi,"Angle of attack (deg)"),
                                                ("omega",1.,"Pitch rate (rad/s)"))):
            ax=axs[row,col];mask=o["t"]<=2.5;pred=pt<=2.5
            ax.plot(o["t"][mask],o[key][mask]*scale,"ko",ms=3.5,label="Game replay")
            ax.plot(pt[pred],np.array([r[key] for r in rr])[pred]*scale,
                    color="#2473b4",lw=1.8,label="Fixed-PID candidate")
            ax.set(xlim=(0.,2.5),ylabel=ylabel,xlabel="Time from release (s)")
            ax.grid(alpha=.2)
        axs[0,col].set_title(name+(" | not fitted" if name==HOLD_NAME else " | calibration"))
        for key in ("alpha","theta"):
            obs=event(o["t"],o[key],.8,1.,"min")
            pred=event(pt,np.array([r[key] for r in rr]),.8,1.,"min")
            detail[name+"_"+key+"_first_trough_deg"]={
                "observed":math.degrees(obs["value"]),"candidate":math.degrees(pred["value"]),
                "observed_time_s":obs["time_s"],"candidate_time_s":pred["time_s"]}
    axs[0,0].legend(fontsize=9)
    fig.suptitle("Loft OFF | one shared three-coefficient rotational model | raw PID gains unchanged\n"
                 "Measured speed and target motion supplied; attitude and PID evolve from release",fontsize=13)
    fig.savefig(OUT/"no_loft_review.png",dpi=180)
    plt.close(fig)
    print(json.dumps({"saved_result_reproduced":True,"first_troughs":detail},indent=2))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    raw,rawp=batch.load_all()
    assert rawp["source_version"]=="2.59.0.28"
    assert [rawp[k] for k in ("kp","ki","kd","integral_limit")]==[.002125,.006481,.000023,1.]
    assert rawp["pn_gain"]==6 and rawp["time_gain"]==[[.30,0.],[.31,1.]]
    # The fit/selection references only L110 and L27.5. L55 is constructed
    # and evaluated after `chosen` is fixed below.
    train={n:make_shot(raw[n],"b5") for n in TRAIN_NAMES}
    assert all(not sh["p"]["loft_enabled"] for sh in train.values())
    passive=passive_interval_estimate(train)
    variants={}
    for lifecycle in ("frozen","updating"):
        name=lifecycle+"_constant"
        coeff,opt=optimize(train,lifecycle,"constant",passive)
        case_scores,aggregate=candidate_scores(train,coeff,lifecycle,"constant")
        variants[name]={"coefficients":coeff.tolist(),"optimization":opt,
                        "training_case_checks":case_scores,"aggregate":aggregate,
                        "lifecycle":lifecycle,"scaling":"constant"}
    # Predeclared topology rule: check first/second q extrema and time-hold q
    # waveform on both training cases. Only a failure opens one physical
    # three-coefficient scaling comparison; no L55 residual is read here.
    def fails_topology(v):
        return (any(abs(v["training_case_checks"][n]["events"][key]["time_error_s"])>.042
                    or abs(v["training_case_checks"][n]["events"][key]["relative_amplitude_error"])>.10
                    for n in TRAIN_NAMES for key in ("q_peak1","q_valley1","q_peak2"))
                or v["aggregate"]["time_hold_scaled_state_mse"]>.10)
    constant_failure=all(fails_topology(v) for v in variants.values())
    if constant_failure:
        for lifecycle in ("frozen","updating"):
            name=lifecycle+"_qbar_over_I"
            coeff,opt=optimize(train,lifecycle,"qbar_over_I",passive)
            case_scores,aggregate=candidate_scores(train,coeff,lifecycle,"qbar_over_I")
            variants[name]={"coefficients":coeff.tolist(),"optimization":opt,
                            "training_case_checks":case_scores,"aggregate":aggregate,
                            "lifecycle":lifecycle,"scaling":"qbar_over_I"}
    # Selection uses training objective and continuous time hold only. The
    # chosen name/coefficients are fixed before the L55 object is constructed.
    def selection_score(v):
        return v["optimization"]["training_objective"]+v["aggregate"]["time_hold_scaled_state_mse"]
    chosen=min(variants,key=lambda n:selection_score(variants[n]))
    selected=variants[chosen]
    selected["selection_score"]=selection_score(selected)
    hold=make_shot(raw[HOLD_NAME],"b5")
    assert not hold["p"]["loft_enabled"]
    shots=dict(train,**{HOLD_NAME:hold})
    coeff=np.array(selected["coefficients"])
    forwards={n:simulate(s,coeff,selected["lifecycle"],selected["scaling"]) for n,s in shots.items()}
    assert all(rows is not None and all(np.isfinite([v[k] for k in ("theta","omega","gamma","x","y","u","I")]).all()
                                     for v in rows) for rows in forwards.values())
    hold_eval=compare(hold,forwards[HOLD_NAME])
    onset={n:activation_interval(s,forwards[n],coeff,selected["scaling"]) for n,s in shots.items()}
    # One targeted plant integration step-halving check. Controller still .02s.
    step={}
    for n,s in shots.items():
        finer=simulate(s,coeff,selected["lifecycle"],selected["scaling"],plant_dt=.005)
        step[n]={key:float(max(abs(a[key]-b[key]) for a,b in zip(forwards[n],finer)))
                 for key in ("theta","omega","gamma","y")}
    # Packet-age sensitivity is a same-parameter replay; no re-fit/selection.
    packet={n:make_shot(raw[n],"packet") for n in (*TRAIN_NAMES,HOLD_NAME)}
    packet_sensitivity={n:compare(s,simulate(s,coeff,selected["lifecycle"],selected["scaling"]))
                        for n,s in packet.items()}
    plot_result(shots,forwards,chosen);plot_activation(shots,forwards)
    result={"source":"source_payload.json 2.59.0.28 + batch_compact.json; only three loftEnabled=false L cases",
            "model":"theta_dot=omega; omega_dot=S(t)*(ku*u-ka*(theta-gamma))-D(t)*kw*omega; gamma_dot=[qbar*S_eff*14*alpha+T*sin(alpha)]/(m*V)-g*cos(gamma)/V; xdot=V*cos(gamma), ydot=V*sin(gamma). Constant: S=D=1. qbar_over_I: S=(qbar/m)/(qbar0/m0); D=S*(V0/V), same L and reference convention for all cases.",
            "control":"20 ms raw PID updates and hold; original timeToGain table continuously multiplies the held clipped PID output within each interval, including .30-.31 s. Wind-normal feedback in g from fixed 14/rad force; PN6*timeToHitGain + gravity compensation; raw P/I/D and I cap fixed. Updating integrates I while gain=0; frozen does not. First updating previous_error is release error to avoid D kick.",
            "time_and_initial":"Primary physical age=b5/48-creation.creation_time. Native targets retain packet association via monotone packet-to-b5 interpolation; target velocity remains packet-derived. Creation quaternion xyzw sets theta, creation.u12_2 sets complete initial missile velocity, creation.u12_3 z with negative sign sets omega, starting_pos sets x/y. First native sample is never treated as t=0 or a state reset. Packet-age rerun uses same selected coefficients as sensitivity only.",
            "external_inputs":"Measured speed curve and measured target position/velocity; not free-speed flight.",
            "fixed_raw":{"source_version":rawp["source_version"],"kp":rawp["kp"],"ki":rawp["ki"],
                         "kd":rawp["kd"],"integral_limit":rawp["integral_limit"],
                         "pn_gain":rawp["pn_gain"],"time_gain":rawp["time_gain"],
                         "loft_enabled":False,"lift_slope_per_rad":14.0},
            "training_cases":list(TRAIN_NAMES),"whole_case_holdout":HOLD_NAME,
            "holdout_provenance":"L55 did not enter any numerical fit or current candidate selection, but an earlier L55 forward mismatch motivated the continuous-gain timing correction; therefore this is not a fully blind first inspection.",
            "train_window_s":list(TRAIN),"time_hold_window_s":list(TIME_HOLD),
            "passive_interval_diagnostic":passive,"constant_topology_failure_triggered_scaling":constant_failure,
            "candidates":variants,"selected_before_L55":{"name":chosen,"coefficients":selected["coefficients"],
                        "selection_score":selected["selection_score"]},
            "L55_whole_case_validation":hold_eval,"selected_forward_training_cases":
                {n:variants[chosen]["training_case_checks"][n] for n in TRAIN_NAMES},
            "activation_native_intervals":onset,
            "packet_clock_same_coefficients_sensitivity":packet_sensitivity,
            "step_halving_control_period_unchanged":step,
            "initial_conditions":{n:{"theta_rad":float(s["initial"][0]),"omega_rad_s":float(s["initial"][1]),
                                     "gamma_rad":float(s["initial"][2]),"x_m":float(s["initial"][3]),
                                     "y_m":float(s["initial"][4]),"speed_mps":s["initial_speed_mps"],
                                     "first_packet_age_s":s["first_packet_age_s"],"first_b5_age_s":s["first_b5_age_s"]}
                                  for n,s in shots.items()},
            "native_sample_counts":{n:s["sample_count"] for n,s in shots.items()},
            "figures":["no_loft_forward.png","activation_detail.png"]}
    (OUT/"results.json").write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    print(json.dumps({"passive":passive,"selected":result["selected_before_L55"],
                      "candidate_scores":{n:{"train":v["optimization"]["training_objective"],
                                              "hold":v["aggregate"]["time_hold_scaled_state_mse"]}
                                          for n,v in variants.items()},
                      "L55_events":hold_eval["events"],"step":step},indent=2))


if __name__=="__main__":
    if "--plot-review" in sys.argv:plot_review()
    elif "--standard-D-constant-control" in sys.argv:standard_D_experiment("constant_control")
    elif "--standard-D-constant" in sys.argv:standard_D_experiment("constant")
    elif "--standard-D" in sys.argv:standard_D_experiment()
    elif "--schedule-48hz" in sys.argv:schedule_48hz_experiment()
    else:main()
