"""L110 measured-state restarts with fixed plant/PID and absolute flight age.

Reconstruct PID history through each cut using measured states under the
existing controller interpretation. A second run keeps the full-forward I
term but the same measured-state derivative history. Neither is game telemetry.
"""
from __future__ import annotations

import json
import math

import numpy as np

import fit_r771_single_shot_damping as previous

fit=previous.fit
rate=previous.rate
model=rate.model
OUT=fit.OUT/"l110_midflight"
HZ=48
HORIZON=2.5
KEYS=("t","theta","omega","gamma","x","y","u","I")
CUTS=((21,"Pitch-rate peak"),(29,"AoA peak"),(44,"AoA valley"))
STATE_KEYS=("theta","omega","gamma","x","y")


def measured_state(shot,t):
    replay=shot["replay"]
    # Native ages differ from ideal ticks only by float32 creation-time error.
    j=int(np.argmin(abs(replay["t"]-t)))
    if abs(replay["t"][j]-t)<1e-7:
        return np.array([replay[k][j] for k in STATE_KEYS],float)
    return np.array([np.interp(t,replay["t"],replay[k]) for k in STATE_KEYS])


def update_pid(tick,state,pid,shot):
    gain=model.interpolate_table((tick+1)/HZ,shot["p"]["time_gain"])
    if gain<=1e-12:
        assert pid==(0.,0.,0.)
        return 0.,pid
    req=model.request(tick/HZ,state,shot["replay"],shot["p"])
    raw,pid=fit.pid_step_with_dt(req["error_g"],pid,shot["p"],1/HZ)
    return gain*raw,pid


def replay_pid_before(shot,tick_cut):
    pid=(0.,0.,0.)
    for tick in range(tick_cut):
        _,pid=update_pid(tick,measured_state(shot,tick/HZ),pid,shot)
    return pid


def simulate(shot,coeff,start_tick=0,initial=None,pid_initial=(0.,0.,0.),step=rate.STEP,
             damping_variable="omega"):
    assert damping_variable in ("omega","alpha_dot")
    state=(shot["initial"] if initial is None else initial).copy()
    p,r=shot["p"],shot["replay"]
    pid=tuple(pid_initial)
    # The aerodynamic normalization belongs to the original launch. It must
    # not be recalculated using the restart height/speed or remaining mass.
    v0=shot["initial_speed_mps"]
    reference=.5*model.isa_density(shot["initial"][4])*v0*v0/p["mass0_kg"]
    ku,ka,kw=map(float,coeff)
    rows=[];snapshots={}
    def rhs(t,s,u):
        th,om,ga,x,y=s
        f=model.force_terms(t,s,r,p)
        v=f["v"]
        a=(.5*model.isa_density(y)*v*v/f["mass"])/reference
        b=a*v0/v
        gamma_dot=model.G*(f["actual_specific_force_g"]-math.cos(ga))/v
        damping_rate=om if damping_variable=="omega" else om-gamma_dot
        return np.array([om,a*(ku*u-ka*(th-ga))-b*kw*damping_rate,
                         gamma_dot,
                         v*math.cos(ga),v*math.sin(ga)])
    for tick in range(start_tick,round(HORIZON*HZ)):
        start,end=tick/HZ,(tick+1)/HZ
        snapshots[tick]={"state":state.copy(),"pid":pid}
        u,pid=update_pid(tick,state,pid,shot)
        rows.append([start,*state,u,pid[0]])
        steps=math.ceil((end-start)/step-1e-9);h=(end-start)/steps
        for k in range(steps):
            t=start+k*h
            d1=rhs(t,state,u);d2=rhs(t+h/2,state+h*d1/2,u)
            d3=rhs(t+h/2,state+h*d2/2,u);d4=rhs(t+h,state+h*d3,u)
            state+=h*(d1+2*d2+2*d3+d4)/6
            if k<steps-1:rows.append([t+h,*state,u,pid[0]])
    rows.append([HORIZON,*state,u,pid[0]])
    a=np.array(rows)
    result={key:a[:,j] for j,key in enumerate(KEYS)}
    result["alpha"]=result["theta"]-result["gamma"]
    assert np.isfinite(a).all()
    return result,snapshots


def event_comparison(shot,pred,start):
    obs=shot["observed"];result={}
    for label,key,lo,hi,kind,_ in previous.EVENTS:
        real=fit.event(obs["t"],obs[key],lo,hi,kind)
        if real["time_s"]<=start+1e-6:continue
        if hi<=start:continue
        calc=fit.event(pred["t"],pred[key],max(lo,start),hi,kind)
        mask=(pred["t"]>=max(lo,start))&(pred["t"]<=hi)
        times=pred["t"][mask]
        result[label]={"observed":real,"predicted":calc,
            "time_error_s":calc["time_s"]-real["time_s"],
            "value_error":calc["value"]-real["value"],
            "predicted_extremum_at_window_edge":bool(calc["time_s"] in (times[0],times[-1]))}
    return result


def metrics(shot,pred,start):
    obs=shot["observed"]
    ids=np.flatnonzero((obs["t"]>start+1e-6)&(obs["t"]<HORIZON))
    native=[]
    for j in ids[:3]:
        stamp=float(obs["t"][j])
        native.append({"time_s":stamp,"elapsed_s":stamp-start,
            "omega_error_rad_s":float(np.interp(stamp,pred["t"],pred["omega"])-obs["omega"][j]),
            "alpha_error_deg":math.degrees(float(np.interp(stamp,pred["t"],pred["alpha"])-obs["alpha"][j]))})
    return {"events":event_comparison(shot,pred,start),"first_three_native_samples":native}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    coeff,raw,p=previous.load();shot=fit.make_shot(raw["L110"])
    assert not shot["p"]["loft_enabled"]
    baseline,snapshots=simulate(shot,coeff)
    curves={};records={};warm={}
    for tick,label in CUTS:
        start=tick/HZ;state=measured_state(shot,start)
        reconstructed=replay_pid_before(shot,tick);warm[tick]=reconstructed
        carry_integral=(snapshots[tick]["pid"][0],*reconstructed[1:])
        curves[tick]={};checks={}
        for name,pid in (("replay_PID",reconstructed),("carry_I",carry_integral)):
            pred,_=simulate(shot,coeff,tick,state,pid)
            curves[tick][name]=pred
            checks[name]={"pid_before_restart":list(pid),**metrics(shot,pred,start)}
        records[tick]={"start_s":start,"label":label,"initial_state":dict(zip(STATE_KEYS,state.tolist())),
            "baseline_pid_before_restart":list(snapshots[tick]["pid"]),
            "baseline":metrics(shot,baseline,start),"runs":checks}

    # Focused final verification: unchanged original run; restarting from its
    # own complete state must reproduce its remaining trajectory exactly.
    reference=rate.simulate(shot,coeff,HZ,"end_gain",control_tau=0.)
    baseline_error=max(float(np.max(abs(baseline[k]-reference[k]))) for k in (*STATE_KEYS,"u","I"))
    assert baseline_error<1e-10
    restart_errors={}
    for tick,_ in CUTS:
        pr,_=simulate(shot,coeff,tick,snapshots[tick]["state"],snapshots[tick]["pid"])
        tail=baseline["t"]>=tick/HZ
        error=max(float(np.max(abs(pr[k]-baseline[k][tail]))) for k in (*STATE_KEYS,"u","I"))
        restart_errors[tick]=error
        assert error<1e-10
        initial=measured_state(shot,tick/HZ)
        for pr in curves[tick].values():
            assert np.array_equal(np.array([pr[k][0] for k in STATE_KEYS]),initial)
            assert abs(pr["t"][0]-tick/HZ)<1e-12
    # The AoA-peak release directly covers the questioned first downstroke.
    fine,_=simulate(shot,coeff,29,measured_state(shot,29/HZ),warm[29],step=rate.STEP/2)
    coarse=curves[29]["replay_PID"]
    integration_error=float(np.max(abs(coarse["omega"]-np.interp(coarse["t"],fine["t"],fine["omega"]))))
    assert integration_error<1e-6
    result={"case":"L110","source_version":p["source_version"],"coefficients":coeff.tolist(),
        "control_hz":HZ,"control_tau_s":0.,"damping":"Original -B*kw*omega",
        "fixed_PID":{k:p[k] for k in ("kp","ki","kd","integral_limit")},
        "records":records,"baseline_events":event_comparison(shot,baseline,0.),
        "restart_definition":"One injection of measured theta/omega/gamma/x/y at each original-clock tick. Preserve absolute age for mass/thrust/gains/target, original release normalization, measured speed and target inputs. No later measured attitude/position injection. No coefficient fitting.",
        "PID_history":"replay_PID uses fixed PID driven by linearly interpolated measured states at all prior original 48Hz ticks. carry_I changes only the initial integral to the full-forward value; previous error and derivative retain measured-history consistency to avoid an artificial derivative kick. Both are conditional reconstructed histories, not decoded game PID.",
        "verification":{"baseline_max_difference":baseline_error,
            "restart_own_complete_state_max_differences":restart_errors,
            "measured_initial_states_exact":True,"step_halving_max_omega_difference":integration_error}}
    (OUT/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    export={"baseline__"+k:baseline[k] for k in (*KEYS,"alpha")}
    for tick,both in curves.items():
        for tag,pr in both.items():
            export.update({str(tick)+"__"+tag+"__"+k:pr[k] for k in (*KEYS,"alpha")})
    np.savez_compressed(OUT/"trajectories.npz",**export)
    plot(shot,baseline,curves)
    print(json.dumps({"verification":result["verification"],"records":records},indent=2))


def plot(shot,baseline,curves):
    plt=fit.plt;obs=shot["observed"]
    fig,axs=plt.subplots(2,3,figsize=(15,8),sharex=True,sharey="row",layout="constrained")
    for col,(tick,label) in enumerate(CUTS):
        start=tick/HZ
        for row,(key,scale,ylabel) in enumerate((("omega",1.,"Pitch rate (rad/s)"),("alpha",180/math.pi,"Angle of attack (deg)"))):
            ax=axs[row,col];mask=(obs["t"]>=.3)&(obs["t"]<=2.1)
            ax.plot(obs["t"][mask],obs[key][mask]*scale,"ko",ms=3.6,label="Game replay",zorder=5)
            mask=(baseline["t"]>=.3)&(baseline["t"]<=2.1)
            ax.plot(baseline["t"][mask],baseline[key][mask]*scale,color=".6",ls="--",lw=1.5,label="Original full forward")
            for tag,color,style,legend in (("replay_PID","#287fb3","-","Measured-state restart + replay PID"),
                    ("carry_I","#ce7927",":","Same restart + original I term")):
                pr=curves[tick][tag];keep=pr["t"]<=2.1
                ax.plot(pr["t"][keep],pr[key][keep]*scale,color=color,ls=style,lw=1.8,label=legend)
            ax.axvline(start,color=".4",lw=1.)
            ax.axvspan(.3,start,color=".9",alpha=.35)
            ax.set(xlim=(.3,2.1),xlabel="Absolute time from release (s)")
            if col==0:ax.set_ylabel(ylabel)
            ax.grid(alpha=.18)
        axs[0,col].set_title(f"Restart at {start:.4f} s | {label}")
    axs[0,0].legend(fontsize=8)
    fig.suptitle("L110 | measured-state midflight restarts; fixed original plant and PID\n"
                 "No added control lag | reconstructed controller memory | no fitting",fontsize=13)
    fig.savefig(OUT/"comparison.png",dpi=180)
    plt.close(fig)


if __name__=="__main__":main()
