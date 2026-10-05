"""Change only pitch-rate damping strength in the no-loft 48 Hz candidate."""
from __future__ import annotations

import json
import math

import numpy as np

import check_r771_no_loft_control_rate as rate

fit=rate.fit
OUT=fit.OUT/"damping_strength"
FACTORS=(.50,.70,.85,1.,1.15,1.30,1.50)
NAMES=(*fit.TRAIN_NAMES,fit.HOLD_NAME)


def evaluate(shot,pred):
    result=rate.metrics(shot,pred)
    obs=shot["observed"]
    for name,lo,hi,kind in (("q_valley2",1.18,1.45,"min"),("q_peak3",1.38,1.72,"max")):
        actual=fit.event(obs["t"],obs["omega"],lo,hi,kind)
        calc=fit.event(pred["t"],pred["omega"],lo,hi,kind)
        result[name]={"observed":actual,"predicted":calc,
                      "time_error_s":calc["time_s"]-actual["time_s"],
                      "value_error":calc["value"]-actual["value"],
                      "relative_amplitude_error":(calc["value"]-actual["value"])/abs(actual["value"])}
    result["periods"]={}
    for start,end in (("q_peak1","q_peak2"),("q_peak2","q_peak3"),("q_valley1","q_valley2")):
        observed=result[end]["observed"]["time_s"]-result[start]["observed"]["time_s"]
        predicted=result[end]["predicted"]["time_s"]-result[start]["predicted"]["time_s"]
        result["periods"][start+"_to_"+end]={"observed_s":observed,"predicted_s":predicted,
            "frequency_ratio_predicted_over_observed":observed/predicted}
    mask=obs["t"]<.30
    result["passive_max_omega_error_rad_s"]=float(np.max(np.abs(
        np.interp(obs["t"][mask],pred["t"],pred["omega"])-obs["omega"][mask])))
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    saved=json.loads((fit.OUT/"schedule_48hz"/"zero_memory_standard_D"/"results.json").read_text())
    base=np.array(saved["refit_coefficients"])
    raw,p=fit.batch.load_all()
    assert [p[k] for k in ("kp","ki","kd")]==[.002125,.006481,.000023]
    assert p["pn_gain"]==6
    train={name:fit.make_shot(raw[name]) for name in fit.TRAIN_NAMES}
    assert all(not s["p"]["loft_enabled"] for s in train.values())
    curves={};records={}

    def run_training(factor):
        coeff=base.copy();coeff[2]*=factor
        assert np.array_equal(coeff[:2],base[:2])
        predictions={name:rate.simulate(shot,coeff,48,"end_gain") for name,shot in train.items()}
        checks={name:evaluate(train[name],pr) for name,pr in predictions.items()}
        curves[factor]=predictions
        records[factor]={"damping_factor":factor,"k_omega_s_minus1":float(coeff[2]),
            "checks":checks,"first_recovery_score":float(np.mean([rate.recovery_score(c) for c in checks.values()]))}
        return float(np.mean([c["q_valley1"]["value_error"] for c in checks.values()]))

    for factor in FACTORS:run_training(factor)
    # A single scalar diagnostic: match the signed first rate-valley amplitude
    # on L110/L27.5. No phase, other coefficients or L55 quantities enter.
    bracket=None
    for lo,hi in zip(FACTORS,FACTORS[1:]):
        elo=np.mean([c["q_valley1"]["value_error"] for c in records[lo]["checks"].values()])
        ehi=np.mean([c["q_valley1"]["value_error"] for c in records[hi]["checks"].values()])
        if elo*ehi<=0:
            bracket=(lo,hi);break
    match_factor=None
    if bracket is not None:
        lo,hi=bracket
        for _ in range(7):
            mid=(lo+hi)/2
            err=run_training(mid)
            if abs(err)<.00025:break  # about half one decoded rate quantum
            if err<0:lo=mid
            else:hi=mid
        match_factor=mid
    best_grid=min(FACTORS,key=lambda m:records[m]["first_recovery_score"])

    # The choices above are fixed before this cross-shot check is evaluated.
    hold=fit.make_shot(raw[fit.HOLD_NAME]);assert not hold["p"]["loft_enabled"]
    shown=sorted({.70,1.,1.30,match_factor if match_factor is not None else best_grid})
    for factor in sorted(set(FACTORS)|set(shown)):
        coeff=base.copy();coeff[2]*=factor
        pred=rate.simulate(hold,coeff,48,"end_gain")
        curves[factor][fit.HOLD_NAME]=pred
        records[factor]["checks"][fit.HOLD_NAME]=evaluate(hold,pred)

    # One focused final pass: baseline matches the previous frequency scan,
    # release/control timing is unchanged, and selected-result integration
    # converges while retaining the 48 Hz control interval.
    prior=json.loads((fit.OUT/"control_rate"/"results.json").read_text())["records"]["end_gain"]["48"]["checks"]
    for name in NAMES:
        for key in ("q_peak1","q_valley1","q_peak2","alpha_valley1"):
            for field in ("value","time_s"):
                assert abs(records[1.]["checks"][name][key]["predicted"][field]-prior[name][key]["predicted"][field])<1e-10
    for factor,record in records.items():
        for name,check in record["checks"].items():
            assert abs(check["first_active"]["start_s"]-14/48)<1e-12
    selected=match_factor if match_factor is not None else best_grid
    coeff=base.copy();coeff[2]*=selected
    fine=rate.simulate(hold,coeff,48,"end_gain",step=rate.STEP/2)
    coarse=curves[selected][fit.HOLD_NAME]
    convergence=float(np.max(np.abs(coarse["omega"]-np.interp(coarse["t"],fine["t"],fine["omega"]))))
    assert convergence<1e-6
    result={"source_version":p["source_version"],"baseline_coefficients":base.tolist(),
            "fixed_PID":{k:p[k] for k in ("kp","ki","kd","integral_limit")},
            "control_hz":48,"control_schedule":"1/48 interval end gain; zero PID memory before activation; natural D startup",
            "damping_law":"omega_dot_damp = -factor * k_omega * [(qbar/m)/(qbar0/m0)] * (V0/V) * omega",
            "varied":"Only one time-independent multiplier of the existing damping law; ku, ka, PID, force slope14, scheduling, initial states and replay inputs unchanged.",
            "grid_factors":list(FACTORS),"first_valley_match_factor":match_factor,
            "first_recovery_best_grid_factor_training_only":best_grid,"records":records,
            "training_cases":list(fit.TRAIN_NAMES),"cross_shot_check":"L55 excluded from choices; previously viewed, not pristine blind",
            "external_inputs":"Measured speed and target trajectory; remaining states and PID propagate from decoded release",
            "verification":{"baseline_reproduced":True,"activation_time_unchanged":True,
                            "step_halving_max_omega_difference_rad_s":convergence}}
    (OUT/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")

    plt=fit.plt
    fig,axs=plt.subplots(2,1,figsize=(12,8),sharex=True,layout="constrained")
    colors={.70:"#e28a25",1.:"#2475b2",1.30:"#985ca4",selected:"#239260"}
    obs=hold["observed"]
    for ax,key,scale,label in ((axs[0],"omega",1.,"Pitch rate (rad/s)"),
                               (axs[1],"alpha",180/math.pi,"Angle of attack (deg)")):
        mask=obs["t"]<=2.3
        ax.plot(obs["t"][mask],obs[key][mask]*scale,"ko",ms=4,label="Game replay")
        for factor in shown:
            pred=curves[factor][fit.HOLD_NAME];keep=pred["t"]<=2.3
            text=(f"{factor:.3f}x damping | matches first rate valley" if factor==selected else
                  ("1.000x damping | baseline" if factor==1. else f"{factor:.2f}x damping"))
            ax.plot(pred["t"][keep],pred[key][keep]*scale,color=colors[factor],
                    lw=1.8 if factor in (1.,selected) else 1.3,
                    ls="-" if factor in (1.,selected) else "--",label=text)
        ax.axvspan(.65,.95,color="0.8",alpha=.16)
        ax.set(xlim=(0.,2.3),ylabel=label)
        ax.grid(alpha=.2)
    axs[0].legend(fontsize=9,ncol=2)
    axs[1].set_xlabel("Time from release (s)")
    fig.suptitle("Loft OFF | L55 | only damping strength changes\n48 Hz, PID, control gain and restoring coefficient fixed",fontsize=14)
    fig.savefig(OUT/"damping_comparison.png",dpi=180)
    plt.close(fig)
    print(json.dumps({"match_factor":match_factor,"best_grid":best_grid,"verification":result["verification"],
                      "L55":{factor:records[factor]["checks"][fit.HOLD_NAME] for factor in shown}},indent=2))


if __name__=="__main__":main()
