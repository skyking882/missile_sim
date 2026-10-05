"""L110 midflight fit of one alpha-rate damping coefficient, with controls."""
from __future__ import annotations

import json
import math

import numpy as np

import replay_r771_l110_from_midflight as mid

previous=mid.previous
fit=mid.fit
OUT=fit.OUT/"l110_alpha_rate_damping"
TRAIN_TICK=29
FIT_END=1.15
FIT_STEP=1/192
GRID=(0.,.5,1.,2.,3.,4.,5.,6.,8.,10.,12.)


def run(shot,base,tick,law,k,step=FIT_STEP):
    coeff=base.copy();coeff[2]=k
    initial=mid.measured_state(shot,tick/mid.HZ)
    pid=mid.replay_pid_before(shot,tick)
    return mid.simulate(shot,coeff,tick,initial,pid,step=step,damping_variable=law)[0]


def score(shot,pred):
    obs=shot["observed"];residual=[]
    for name,key,lo,hi,kind,scale in previous.EVENTS:
        if name not in ("q_valley1","q_peak2","alpha_valley1"):continue
        hi=min(hi,FIT_END)
        ot,ov=previous.extremum(obs["t"],obs[key],lo,hi,kind)
        pt,pv=previous.extremum(pred["t"],pred[key],lo,hi,kind)
        residual.extend([(pt-ot)/.03,(pv-ov)/scale])
    real,calc=previous.negative_lobe(obs),previous.negative_lobe(pred)
    if calc is None:return 1e6
    residual.extend([(calc["down_s"]-real["down_s"])/.03,
                     (calc["up_s"]-real["up_s"])/.03])
    mask=(obs["t"]>TRAIN_TICK/mid.HZ+1e-6)&(obs["t"]<=FIT_END)
    waveform=[]
    for key,scale in (("omega",.03),("alpha",math.radians(.2))):
        delta=np.interp(obs["t"][mask],pred["t"],pred[key])-obs[key][mask]
        waveform.append(float(np.mean((delta/scale)**2)))
    return float(np.mean(np.square(residual))+.5*np.mean(waveform))


def optimize(shot,base,law):
    trials={}
    def evaluate(k):
        k=float(k)
        if k not in trials:trials[k]=score(shot,run(shot,base,TRAIN_TICK,law,k))
        return trials[k]
    for k in GRID:evaluate(k)
    best=min(trials,key=trials.get);index=GRID.index(best)
    # The zero-strength case remains eligible, as does the upper boundary.
    lo=GRID[max(0,index-1)];hi=GRID[min(len(GRID)-1,index+1)]
    golden=(math.sqrt(5)-1)/2
    c=hi-golden*(hi-lo);d=lo+golden*(hi-lo)
    for _ in range(18):
        if hi-lo<.002:break
        if evaluate(c)<evaluate(d):hi,d=d,c;c=hi-golden*(hi-lo)
        else:lo,c=c,d;d=lo+golden*(hi-lo)
    evaluate((lo+hi)/2)
    selected=min(trials,key=trials.get)
    return {"law":law,"k_s_inverse":selected,"objective":trials[selected],
            "trials":[{"k":k,"score":trials[k]} for k in sorted(trials)]}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    base,raw,p=previous.load();shot=fit.make_shot(raw["L110"])
    assert not shot["p"]["loft_enabled"]
    fits={}
    for law in ("omega","alpha_dot"):
        fits[law]=optimize(shot,base,law)
        print(json.dumps({law:{k:v for k,v in fits[law].items() if k!="trials"}}),flush=True)
    frozen={"case":"L110","source_version":p["source_version"],"fit_start_s":TRAIN_TICK/mid.HZ,
            "fit_end_s":FIT_END,"fits":fits,"original_coefficients":base.tolist(),
            "fixed_PID":{k:p[k] for k in ("kp","ki","kd","integral_limit")},
            "control_hz":mid.HZ,"control_tau_s":0.,
            "alpha_dot_definition":"omega - gamma_dot, with gamma_dot from current simulated normal-force balance; no future measured alpha or omega is supplied after restart.",
            "damping_laws":{"omega":"-k*B*omega","alpha_dot":"-k*B*(omega-gamma_dot)"},
            "B_definition":"(rho/rho0)*(V/V0)*(m0/m); original launch normalization retained.",
            "fit_definition":"One nonnegative damping strength per law, 0..12/s. Same objective: first omega valley, next omega peak, first AoA valley, downward/upward omega zero crossings, and native omega/AoA shape through 1.15s only. Same reconstructed initial PID memory.",
            "checks_definition":"After freezing both fits, inspect 1.15..2.45s continuation and other restart times of this same L110 shot. These are within-shot checks, not independent launch validation. Measured speed/target inputs remain external. Other coefficients were previously calibrated and are fixed."}
    (OUT/"fit.json").write_text(json.dumps(frozen,indent=2,allow_nan=False)+"\n")
    variants={"original":("omega",float(base[2])),"fitted_omega":("omega",fits["omega"]["k_s_inverse"]),
              "fitted_alpha":("alpha_dot",fits["alpha_dot"]["k_s_inverse"]),"no_damping":("alpha_dot",0.)}
    curves={};records={}
    for tick,label in mid.CUTS:
        curves[tick]={};records[tick]={"label":label,"start_s":tick/mid.HZ,"runs":{}}
        for tag,(law,k) in variants.items():
            pred=run(shot,base,tick,law,k,step=mid.rate.STEP)
            curves[tick][tag]=pred
            rec={"law":law,"k_s_inverse":k,**mid.metrics(shot,pred,tick/mid.HZ),
                 "negative_lobe":previous.negative_lobe(pred)}
            if tick==TRAIN_TICK:rec["training_score"]=score(shot,pred)
            records[tick]["runs"][tag]=rec
    # Final checks preserve the old simulation and exact restart states, test
    # the zero-strength limit, and verify the new rate identity and integration.
    old=np.load(mid.OUT/"trajectories.npz")
    old_error=0.
    for tick,_ in mid.CUTS:
        pred=curves[tick]["original"]
        for key in (*mid.STATE_KEYS,"u","I"):
            old_error=max(old_error,float(np.max(abs(pred[key]-old[f"{tick}__replay_PID__{key}"]))))
        initial=mid.measured_state(shot,tick/mid.HZ)
        for pr in curves[tick].values():
            assert np.array_equal(np.array([pr[k][0] for k in mid.STATE_KEYS]),initial)
    assert old_error<1e-10
    zero=run(shot,base,TRAIN_TICK,"omega",0.,step=mid.rate.STEP)
    zero_error=float(np.max(abs(zero["omega"]-curves[TRAIN_TICK]["no_damping"]["omega"])))
    assert zero_error==0.
    coefficient=fits["alpha_dot"]["k_s_inverse"]
    coarse=curves[TRAIN_TICK]["fitted_alpha"]
    fine=run(shot,base,TRAIN_TICK,"alpha_dot",coefficient,step=mid.rate.STEP/2)
    integration_error=float(np.max(abs(coarse["omega"]-np.interp(coarse["t"],fine["t"],fine["omega"]))))
    assert integration_error<1e-6
    fitted_step=run(shot,base,TRAIN_TICK,"alpha_dot",coefficient)
    step_difference=float(np.max(abs(fitted_step["omega"]-np.interp(fitted_step["t"],coarse["t"],coarse["omega"]))))
    assert step_difference<1e-4
    # Evaluate the rhs identity in the same state, not a finite-difference
    # derivative across discontinuous controller ticks.
    rate_identity=[]
    for index in np.linspace(0,len(coarse["t"])-1,9,dtype=int):
        t=float(coarse["t"][index]);state=np.array([coarse[k][index] for k in mid.STATE_KEYS])
        force=mid.model.force_terms(t,state,shot["replay"],shot["p"])
        gd=mid.model.G*(force["actual_specific_force_g"]-math.cos(state[2]))/force["v"]
        rate_identity.append({"t_s":t,"omega_rad_s":state[1],"gamma_dot_rad_s":gd,"alpha_dot_rad_s":state[1]-gd})
    result={**frozen,"records":records,"observed_negative_lobe":previous.negative_lobe(shot["observed"]),
            "sampled_simulated_rates":rate_identity,"verification":{"old_path_max_difference":old_error,
            "zero_strength_laws_identical":zero_error==0.,"initial_states_preserved":True,
            "step_halving_max_omega_difference":integration_error,"fit_vs_fine_step_max_omega_difference":step_difference}}
    (OUT/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    np.savez_compressed(OUT/"trajectories.npz",**{str(tick)+"__"+tag+"__"+key:pr[key]
        for tick,both in curves.items() for tag,pr in both.items() for key in (*mid.KEYS,"alpha")})
    plot(shot,curves,fits)
    print(json.dumps({"verification":result["verification"],"scores":{tag:rec["training_score"] for tag,rec in records[TRAIN_TICK]["runs"].items()}},indent=2))


def plot(shot,curves,fits):
    plt=fit.plt;obs=shot["observed"]
    fig,axs=plt.subplots(2,3,figsize=(15,8),sharex=True,sharey="row",layout="constrained")
    labels={"original":"Original omega damping (k=3.774)",
            "fitted_omega":f"Fitted omega damping (k={fits['omega']['k_s_inverse']:.3f})",
            "fitted_alpha":f"Fitted alpha-rate damping (k={fits['alpha_dot']['k_s_inverse']:.3f})"}
    for col,(tick,label) in enumerate(mid.CUTS):
        start=tick/mid.HZ
        for row,(key,scale,ylabel) in enumerate((("omega",1.,"Pitch rate (rad/s)"),("alpha",180/math.pi,"Angle of attack (deg)"))):
            ax=axs[row,col];keep=(obs["t"]>=.35)&(obs["t"]<=2.1)
            ax.plot(obs["t"][keep],obs[key][keep]*scale,"ko",ms=3.7,label="Game replay",zorder=5)
            for tag,color,style in (("original",".65","--"),("fitted_omega","#3479aa",":"),("fitted_alpha","#db7824","-")):
                pr=curves[tick][tag];mask=pr["t"]<=2.1
                ax.plot(pr["t"][mask],pr[key][mask]*scale,color=color,ls=style,lw=1.8,label=labels[tag])
            ax.axvline(start,color=".4",lw=1)
            if tick==TRAIN_TICK:ax.axvspan(start,FIT_END,color="#d9e9d5",alpha=.4)
            ax.set(xlim=(.35,2.1),xlabel="Absolute time from release (s)")
            if col==0:ax.set_ylabel(ylabel)
            ax.grid(alpha=.18)
        axs[0,col].set_title(f"Restart {start:.4f} s"+(" | FIT shaded interval" if tick==TRAIN_TICK else " | same-shot check"))
    axs[0,0].legend(fontsize=8)
    fig.suptitle("L110 | alpha-rate versus body-rate damping; one fitted strength per law\n"
                 "Fit 0.6042-1.15 s only; fixed PID, control/restoring gains and speed scaling",fontsize=13)
    fig.savefig(OUT/"comparison.png",dpi=180)
    plt.close(fig)


if __name__=="__main__":main()
