"""Fixed-coefficient no-loft PID update-rate sensitivity, using replay inputs.

Compare discretizing the raw gain table at each rate with holding the previous
48 Hz candidate's activation time fixed. No PID or moment coefficient is fit.
"""
from __future__ import annotations

import json
import math

import numpy as np

import fit_r771_no_loft_control as fit

model=fit.model
OUT=fit.OUT/"control_rate"
RATES=(24,30,48,50,60,96,120,240,480)
NAMES=(*fit.TRAIN_NAMES,fit.HOLD_NAME)
HORIZON=2.5
STEP=1/2400
COMMON_ONSET=14/48
KEYS=("t","theta","omega","gamma","x","y","u","I")


def simulate(shot,coeff,hz,mode,step=STEP,damping_law="rho_v_over_m",damping_function=None,
             control_tau=0.):
    assert mode in ("end_gain","common_onset")
    assert damping_law in ("rho_v_over_m","constant","none")
    assert control_tau>=0
    dt=1/hz
    state=shot["initial"].copy()
    p,r=shot["p"],shot["replay"]
    pid=(0.,0.,0.)
    q0=.5*model.isa_density(state[4])*shot["initial_speed_mps"]**2
    reference=q0/p["mass0_kg"]
    v0=shot["initial_speed_mps"]
    ku,ka,kw=map(float,coeff)
    starts=(np.arange(math.ceil(HORIZON*hz)+1)/hz if mode=="end_gain" else
            np.r_[0.,COMMON_ONSET+np.arange(math.ceil((HORIZON-COMMON_ONSET)*hz)+1)/hz])
    starts=starts[starts<HORIZON-1e-12]
    ends=np.r_[starts[1:],HORIZON]
    rows=[];first_active=None
    effective_rows=[];effective_u=0.

    def rhs(t,s,u):
        th,om,ga,x,y=s
        f=model.force_terms(t,s,r,p)
        v=f["v"]
        a=(.5*model.isa_density(y)*v*v/f["mass"])/reference
        b=(a*v0/v if damping_law=="rho_v_over_m" else
           (1. if damping_law=="constant" else 0.))
        gamma_dot=model.G*(f["actual_specific_force_g"]-math.cos(ga))/v
        damping=(-b*kw*om if damping_function is None else
                 damping_function(om,gamma_dot,a,a*v0/v))
        return np.array([om,a*(ku*u-ka*(th-ga))+damping,
                         gamma_dot,
                         v*math.cos(ga),v*math.sin(ga)])

    for i,(start,end) in enumerate(zip(starts,ends)):
        # Use the exact rational tick endpoint, avoiding an epsilon-positive
        # gain at nominal t=.30 that would spuriously initialize PID memory.
        gate_time=(i+1)/hz if mode=="end_gain" else (15/48 if i else 0.)
        gain=model.interpolate_table(gate_time,p["time_gain"])
        if gain<=1e-12:
            u=0.
            assert pid==(0.,0.,0.)
        else:
            req=model.request(float(start),state,r,p)
            raw,pid=fit.pid_step_with_dt(req["error_g"],pid,p,dt)
            u=gain*raw
            if first_active is None:
                first_active={"start_s":float(start),"gain":gain,"raw_output_rad":raw,
                              "error_g":req["error_g"],"derivative_term_rad":p["kd"]*pid[2]}
        # The gated, clipped PID command is held. Only its contribution to the
        # control moment has a first-order response; restoring/damping terms
        # and the force law are unchanged. Exact actuator evolution avoids an
        # additional stiff RK4 state as tau tends to zero.
        initial_effective=effective_u
        def applied(elapsed):
            return (u if control_tau==0. else
                    u+(initial_effective-u)*math.exp(-elapsed/control_tau))
        rows.append([start,*state,u,pid[0]])
        effective_rows.append(applied(0.))
        steps=math.ceil((end-start)/step-1e-9)
        h=(end-start)/steps
        for k in range(steps):
            t=start+k*h
            u1=applied(k*h);u2=applied((k+.5)*h);u4=applied((k+1)*h)
            d1=rhs(t,state,u1)
            d2=rhs(t+h/2,state+h*d1/2,u2)
            d3=rhs(t+h/2,state+h*d2/2,u2)
            d4=rhs(t+h,state+h*d3,u4)
            state+=h*(d1+2*d2+2*d3+d4)/6
            if k<steps-1:
                rows.append([t+h,*state,u,pid[0]])
                effective_rows.append(u4)
        effective_u=applied(end-start)
        assert np.isfinite(state).all()
    rows.append([HORIZON,*state,u,pid[0]])
    effective_rows.append(effective_u)
    a=np.array(rows)
    result={key:a[:,j] for j,key in enumerate(KEYS)}
    result["alpha"]=result["theta"]-result["gamma"]
    result["u_effective"]=np.array(effective_rows)
    result["first_active"]=first_active
    return result


def metrics(shot,pred):
    obs=shot["observed"]
    out={}
    events=(*fit.EVENTS,("alpha_valley1","alpha",.80,1.,"min"),
            ("q_valley2","omega",1.25,1.45,"min"))
    for label,key,lo,hi,kind in events:
        real=fit.event(obs["t"],obs[key],lo,hi,kind)
        calc=fit.event(pred["t"],pred[key],lo,hi,kind)
        out[label]={"observed":real,"predicted":calc,
                    "time_error_s":calc["time_s"]-real["time_s"],
                    "value_error":calc["value"]-real["value"],
                    "relative_amplitude_error":(calc["value"]-real["value"])/abs(real["value"])}
    for label,lo,hi in (("first_recovery",.6,1.15),("later",1.45,2.45)):
        mask=(obs["t"]>=lo)&(obs["t"]<=hi)
        out[label]={key+"_rmse":float(np.sqrt(np.mean((np.interp(obs["t"][mask],pred["t"],pred[key])-obs[key][mask])**2)))
                    for key in ("theta","alpha","omega","gamma")}
    out["first_active"]=pred["first_active"]
    return out


def recovery_score(result):
    # Same scales at every rate. Selection uses L110/L27.5, not L55.
    return sum((result[k]["time_error_s"]/.042)**2+
               (result[k]["value_error"]/scale)**2
               for k,scale in (("q_valley1",.02),("alpha_valley1",math.radians(.2)),
                               ("q_peak2",.02)))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    saved=json.loads((fit.OUT/"schedule_48hz"/"zero_memory_standard_D"/"results.json").read_text())
    coeff=np.array(saved["refit_coefficients"])
    raw,p=fit.batch.load_all()
    assert [p[k] for k in ("kp","ki","kd")]==[.002125,.006481,.000023]
    shots={name:fit.make_shot(raw[name],"b5") for name in NAMES}
    assert all(not s["p"]["loft_enabled"] for s in shots.values())
    curves={};records={};best={}
    for mode in ("end_gain","common_onset"):
        curves[mode]={};records[mode]={}
        for hz in RATES:
            curves[mode][hz]={name:simulate(s,coeff,hz,mode) for name,s in shots.items()}
            checks={name:metrics(s,curves[mode][hz][name]) for name,s in shots.items()}
            records[mode][hz]={"checks":checks,"recovery_score_training_cases":
                float(np.mean([recovery_score(checks[n]) for n in fit.TRAIN_NAMES]))}
        best[mode]=min(RATES,key=lambda h:records[mode][h]["recovery_score_training_cases"])

    # Direct equivalence check against the previous 48 Hz implementation.
    equivalence={}
    for name,shot in shots.items():
        original=fit.simulate(shot,coeff,"zero_memory_standard_D","qbar_over_I",schedule="48hz_end_hold",horizon=HORIZON)
        pred=curves["end_gain"][48][name]
        equivalence[name]={key:float(np.max(np.abs(np.interp([r["t"] for r in original],pred["t"],pred[key])-
                                                np.array([r[key] for r in original]))))
                           for key in ("theta","omega","gamma","y")}
        assert equivalence[name]["omega"]<1e-6

    # One integration-accuracy check at the two frequency extremes. All scan
    # trajectories already use the same 1/2400 s maximum physical step.
    convergence={}
    for hz in (24,480):
        base=curves["end_gain"][hz]["L110"]
        fine=simulate(shots["L110"],coeff,hz,"end_gain",step=STEP/2)
        convergence[hz]=float(np.max(np.abs(base["omega"]-np.interp(base["t"],fine["t"],fine["omega"]))))
        assert convergence[hz]<1e-6

    # Numerical damping coefficient in the original 48 Hz candidate.
    damping=[]
    s=shots["L55"];pred=curves["end_gain"][48]["L55"]
    q0=.5*model.isa_density(s["initial"][4])*s["initial_speed_mps"]**2
    for t in (0.,.30,.75,1.5,2.5):
        state=np.array([np.interp(t,pred["t"],pred[k]) for k in ("theta","omega","gamma","x","y")])
        f=model.force_terms(t,state,s["replay"],s["p"])
        a=(.5*model.isa_density(state[4])*f["v"]**2/f["mass"])/(q0/s["p"]["mass0_kg"])
        b=a*s["initial_speed_mps"]/f["v"]
        damping.append({"t_s":t,"speed_mps":f["v"],"damping_rate_s_minus1":float(b*coeff[2]),
                        "omega_rad_s":float(state[1]),"damping_qdot_rad_s2":float(-b*coeff[2]*state[1])})

    result={"source_version":p["source_version"],"fixed_coefficients":coeff.tolist(),
            "fixed_raw_PID":{k:p[k] for k in ("kp","ki","kd","integral_limit")},
            "frequencies_hz":list(RATES),"physical_step_s":STEP,"records":records,"best_by_recovery_training_only":best,
            "modes":{"end_gain":"Each rate samples the raw timeToGain at interval end; first actuation time can change.",
                     "common_onset":"All rates start control at previous 48 Hz candidate onset 14/48 s with gain=1; PID ticks restart there. Isolates post-onset update interval."},
            "scope":"Conditional forward with replay speed and target motion; no fitting; all loft off; L55 not used for frequency selection and not a pristine blind holdout.",
            "baseline_equivalence":equivalence,"physical_step_halving_max_omega_difference":convergence,
            "baseline_damping_L55":damping}
    (OUT/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")

    plt=fit.plt
    fig,axs=plt.subplots(2,2,figsize=(13,8),sharex=True,layout="constrained")
    selected=sorted({24,48,120,480,best["end_gain"],best["common_onset"]})
    colors={24:"#e28a25",30:"#a15a22",48:"#2674ad",50:"#299caa",60:"#6a6ebb",96:"#884ebb",120:"#39954e",240:"#8a608a",480:"#ad4b57"}
    obs=shots["L55"]["observed"]
    for col,mode in enumerate(("end_gain","common_onset")):
        for row,(key,scale,ylabel) in enumerate((("omega",1.,"Pitch rate (rad/s)"),("alpha",180/math.pi,"Angle of attack (deg)"))):
            ax=axs[row,col];mask=(obs["t"]>=.2)&(obs["t"]<=1.65)
            ax.plot(obs["t"][mask],obs[key][mask]*scale,"ko",ms=4,label="Game replay")
            for hz in selected:
                pr=curves[mode][hz]["L55"];keep=(pr["t"]>=.2)&(pr["t"]<=1.65)
                ax.plot(pr["t"][keep],pr[key][keep]*scale,color=colors[hz],lw=1.5 if hz==48 else 1.1,
                        ls="-" if hz==48 else "--",label=f"{hz} Hz")
            ax.axvspan(.65,.95,color="0.8",alpha=.16)
            ax.set(xlim=(.2,1.65),xlabel="Time from release (s)",ylabel=ylabel)
            ax.grid(alpha=.2)
        axs[0,col].set_title("Raw gate sampled at each rate" if mode=="end_gain" else "Same activation time; only PID interval changes")
    axs[0,0].legend(fontsize=9,ncol=2)
    fig.suptitle("Loft OFF | L55 | identical PID gains and moment coefficients at every rate",fontsize=14)
    fig.savefig(OUT/"frequency_comparison.png",dpi=180)
    plt.close(fig)
    print(json.dumps({"best":best,"equivalence":equivalence,"step_halving":convergence,"damping":damping},indent=2))


if __name__=="__main__":main()
