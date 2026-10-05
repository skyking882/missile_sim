"""One control-response time fit on L110, frozen before cross-shot checks."""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

import fit_r771_single_shot_damping as previous

rate=previous.rate
fit=previous.fit
OUT=fit.OUT/"control_response"
TRAIN="L110"
VALIDATION=("L27.5","L55")
SCAN_MS=(0.,.5,1.,2.,3.,5.,7.5,10.,15.,20.,30.,40.,60.,80.)
FIT_STEP=1/960
ILLUSTRATIVE_TAU=1/48


def simulate(shot,coeff,tau,step=FIT_STEP):
    # Resolve rapid command changes without using tau to change PID timing.
    physical_step=min(step,tau/4) if tau>0. else step
    return rate.simulate(shot,coeff,48,"end_gain",step=physical_step,control_tau=tau)


def summarize(shot,pred):
    return {"objective":previous.objective(shot,pred),
            "checks":previous.checks.evaluate(shot,pred),
            "negative_lobe":previous.negative_lobe(pred),
            "max_abs_control_rad":float(np.max(np.abs(pred["u"]))),
            "max_abs_effective_control_rad":float(np.max(np.abs(pred["u_effective"])))}


def train():
    OUT.mkdir(parents=True,exist_ok=True)
    coeff,raw,p=previous.load();shot=fit.make_shot(raw[TRAIN])
    assert not shot["p"]["loft_enabled"]
    trials={}
    def trial(tau):
        tau=float(tau)
        if tau not in trials:
            pred=simulate(shot,coeff,tau)
            trials[tau]=summarize(shot,pred)
        return trials[tau]["objective"]
    grid=np.array(SCAN_MS)/1000
    for tau in grid:
        value=trial(tau)
        print(json.dumps({"tau_ms":tau*1000,"training_score":value}),flush=True)
    best_index=int(np.argmin([trial(t) for t in grid]))
    # Refine an interior grid minimum without excluding the exact tau=0 model.
    # Values <0.5 ms are not separately identified in this experiment.
    if 0<best_index<len(grid)-1:
        lo=max(.0005,float(grid[best_index-1]));hi=float(grid[best_index+1])
        ratio=(math.sqrt(5)-1)/2
        c=hi-ratio*(hi-lo);d=lo+ratio*(hi-lo)
        for _ in range(16):
            if hi-lo<.00005:break
            if trial(c)<trial(d):hi,d=d,c;c=hi-ratio*(hi-lo)
            else:lo,c=c,d;d=lo+ratio*(hi-lo)
        trial((lo+hi)/2)
    chosen=min(trials,key=lambda t:trials[t]["objective"])
    # Chosen a priori as one control interval, for visible sensitivity, not as
    # a parameter selected using either validation case.
    trial(ILLUSTRATIVE_TAU)
    record={"training_case":TRAIN,"validation_cases":list(VALIDATION),
            "source_version":p["source_version"],"selected_tau_s":chosen,
            "illustrative_tau_s":ILLUSTRATIVE_TAU,"scan_tau_ms":list(SCAN_MS),
            "positive_tau_resolution_floor_s":.0005,
            "trials":[{"tau_s":t,**trials[t]} for t in sorted(trials)],
            "fixed_coefficients":coeff.tolist(),"control_hz":48,
            "fixed_PID":{k:p[k] for k in ("kp","ki","kd","integral_limit")},
            "law":"tau*du_effective/dt + u_effective = gated_PID_output; u_effective(0)=0; tau=0 bypass. omega_dot=A*(ku*u_effective-ka*alpha)-B*kw*omega; A=(qbar/m)/(qbar0/m0), B=A*V0/V.",
            "objective":"Same native-waveform and peak/valley/lobe objective as single_shot_damping; no event time shift. Fit only 0.30-2.45s L110.",
            "scope":"Only control-response time is new. Old damping and all ku/ka/kw, raw PID, 48Hz schedule, lift and decoded release states fixed. Measured speed/target inputs retained. ku/ka/kw have historical L110/L27.5 calibration, and L55 has previously been inspected, so validation is of the new tau, not a blind whole-model test."}
    (OUT/"fit.json").write_text(json.dumps(record,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"selected_tau_ms":chosen*1000,"baseline_score":trial(0.),
                      "selected_score":trial(chosen)}),flush=True)


def validate():
    frozen=json.loads((OUT/"fit.json").read_text())
    coeff,raw,p=previous.load();tau=frozen["selected_tau_s"]
    assert np.array_equal(coeff,frozen["fixed_coefficients"])
    shots={};curves={};records={}
    for name in (TRAIN,*VALIDATION):
        shot=fit.make_shot(raw[name]);shots[name]=shot
        assert not shot["p"]["loft_enabled"]
        curves[name]={tag:simulate(shot,coeff,t,step=rate.STEP)
                      for tag,t in (("baseline",0.),("selected",tau),("one_interval",ILLUSTRATIVE_TAU))}
        records[name]={tag:summarize(shot,pr) for tag,pr in curves[name].items()}
        records[name]["observed_negative_lobe"]=previous.negative_lobe(shot["observed"])
    # One final verification pass covering the unchanged path, response law,
    # initial condition, fixed PID activation, physical-step convergence.
    old=json.loads((previous.OUT/"results.json").read_text())
    default_error=0.
    for name in shots:
        for event in ("q_peak1","q_valley1","q_peak2","q_valley2","q_peak3","alpha_valley1"):
            for key in ("time_s","value"):
                default_error=max(default_error,abs(records[name]["baseline"]["checks"][event]["predicted"][key]
                          -old["records"][name]["baseline"]["checks"][event]["predicted"][key]))
        assert np.array_equal(curves[name]["baseline"]["u"],curves[name]["baseline"]["u_effective"])
        for tag,pr in curves[name].items():
            assert abs(pr["first_active"]["start_s"]-14/48)<1e-12
            assert np.max(np.abs(pr["u_effective"][pr["t"]<14/48]))==0.
    assert default_error<1e-10
    response=curves[TRAIN]["one_interval"]
    onset=14/48
    mask=(response["t"]>=onset)&(response["t"]<15/48)
    expected=response["u"][mask]*(1-np.exp(-(response["t"][mask]-onset)/ILLUSTRATIVE_TAU))
    response_error=float(np.max(np.abs(response["u_effective"][mask]-expected)))
    assert response_error<1e-12
    convergence={}
    for tag,t in (("selected",tau),("one_interval",ILLUSTRATIVE_TAU)):
        fine=simulate(shots[TRAIN],coeff,t,step=rate.STEP/2)
        base=curves[TRAIN][tag]
        error=float(np.max(np.abs(base["omega"]-np.interp(base["t"],fine["t"],fine["omega"]))))
        convergence[tag]=error
        assert error<1e-6
    fit_resolution=simulate(shots[TRAIN],coeff,tau)
    base=curves[TRAIN]["selected"]
    fit_error=float(np.max(np.abs(fit_resolution["omega"]-np.interp(fit_resolution["t"],base["t"],base["omega"]))))
    assert fit_error<1e-4
    result={**frozen,"records":records,"verification":{"baseline_event_max_difference":default_error,
            "first_interval_response_law_max_error_rad":response_error,
            "step_halving_max_omega_difference_rad_s":convergence,
            "fit_vs_fine_step_max_omega_difference_rad_s":fit_error,
            "PID_activation_and_zero_initial_effective_control_preserved":True}}
    (OUT/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    np.savez_compressed(OUT/"trajectories.npz",**{name+"__"+tag+"__"+key:pr[key]
        for name,both in curves.items() for tag,pr in both.items()
        for key in ("t","omega","alpha","theta","gamma","u","u_effective")})
    plot(shots,curves,frozen)
    print(json.dumps({"selected_tau_ms":tau*1000,"verification":result["verification"],
                      "scores":{name:{tag:records[name][tag]["objective"] for tag in curves[name]} for name in shots}},indent=2))


def plot(shots,curves,frozen):
    plt=fit.plt;tau_ms=frozen["selected_tau_s"]*1000
    fig,axs=plt.subplots(2,3,figsize=(15,8),sharex=True,sharey="row",layout="constrained")
    styles=[("baseline","#7b7b7b","--","Instantaneous control")]
    if tau_ms>0:styles.append(("selected","#d37321","-",f"Fitted response: {tau_ms:.2f} ms"))
    styles.append(("one_interval","#29809d",":","Response: 20.83 ms"))
    for col,name in enumerate((TRAIN,*VALIDATION)):
        obs=shots[name]["observed"]
        for row,(key,scale,label) in enumerate((("omega",1.,"Pitch rate (rad/s)"),("alpha",180/math.pi,"Angle of attack (deg)"))):
            ax=axs[row,col];mask=(obs["t"]>=.2)&(obs["t"]<=2.3)
            ax.plot(obs["t"][mask],obs[key][mask]*scale,"ko",ms=3.6,label="Game replay",zorder=4)
            for tag,color,style,label_curve in styles:
                pr=curves[name][tag];mask=(pr["t"]>=.2)&(pr["t"]<=2.3)
                ax.plot(pr["t"][mask],pr[key][mask]*scale,color=color,ls=style,lw=1.8,label=label_curve)
            ax.axvspan(.65,.95,color="0.85",alpha=.16)
            ax.set(xlim=(.2,2.3),xlabel="Time from release (s)")
            if col==0:ax.set_ylabel(label)
            ax.grid(alpha=.2)
        axs[0,col].set_title(name+(" | FIT" if name==TRAIN else " | VALIDATION"))
    axs[0,0].legend(fontsize=8.5)
    fig.suptitle(f"First-order control response | fit L110, then freeze tau = {tau_ms:.2f} ms\n"
                 "Original damping, fixed PID, 48 Hz control, control/restoring gains and lift",fontsize=13)
    fig.savefig(OUT/"comparison.png",dpi=180)
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("phase",choices=("fit","validate"))
    args=parser.parse_args()
    train() if args.phase=="fit" else validate()
