"""Remove or replace only the extra plant pitch-rate damping term."""
from __future__ import annotations

import json
import math

import numpy as np

import check_r771_no_loft_damping as strength

rate=strength.rate
fit=strength.fit
OUT=fit.OUT/"damping_form"


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    previous=json.loads((strength.OUT/"results.json").read_text())
    base=np.array(previous["baseline_coefficients"])
    raw,p=fit.batch.load_all()
    assert [p[k] for k in ("kp","ki","kd")]==[.002125,.006481,.000023]
    train={name:fit.make_shot(raw[name]) for name in fit.TRAIN_NAMES}
    assert all(not s["p"]["loft_enabled"] for s in train.values())
    curves={};records={}

    def run(name,law,kw,shots):
        coeff=base.copy();coeff[2]=kw
        assert np.array_equal(coeff[:2],base[:2])
        predictions={n:rate.simulate(s,coeff,48,"end_gain",damping_law=law) for n,s in shots.items()}
        checks={n:strength.evaluate(shots[n],r) for n,r in predictions.items()}
        for n,r in predictions.items():
            checks[n]["maximum_abs_alpha_deg"]=float(np.degrees(np.max(abs(r["alpha"]))))
            checks[n]["maximum_abs_omega_rad_s"]=float(np.max(abs(r["omega"])))
            checks[n]["maximum_abs_pid_output_rad"]=float(np.max(abs(r["u"])))
        curves[name]=predictions
        records[name]={"law":law,"kw_s_minus1":float(kw),"checks":checks}
        return float(np.mean([v["q_valley1"]["value_error"] for v in checks.values()]))

    run("scaled_baseline","rho_v_over_m",base[2],train)
    run("no_extra_damping","none",0.,train)
    run("constant_same_release_value","constant",base[2],train)
    # Fair comparison of time dependence: allow the constant law its one
    # amplitude coefficient, fitted only to the same first valley in L110/L27.5.
    lo,hi=1.,8.
    elo=run("constant_search_lo","constant",lo,train)
    ehi=run("constant_search_hi","constant",hi,train)
    assert elo<0<ehi
    search=[]
    for _ in range(10):
        kw=.5*(lo+hi)
        err=run("constant_search","constant",kw,train)
        search.append({"kw_s_minus1":kw,"mean_valley_error_rad_s":err})
        if abs(err)<.00025:break
        if err<0:lo=kw
        else:hi=kw
    run("constant_valley_match","constant",kw,train)
    for name in ("constant_search_lo","constant_search_hi","constant_search"):
        del records[name];del curves[name]
    hold=fit.make_shot(raw[fit.HOLD_NAME]);assert not hold["p"]["loft_enabled"]
    for name,rec in list(records.items()):
        prior_curves=curves[name];prior_checks=rec["checks"]
        run(name,rec["law"],rec["kw_s_minus1"],{fit.HOLD_NAME:hold})
        curves[name]={**prior_curves,**curves[name]}
        records[name]["checks"]={**prior_checks,**records[name]["checks"]}

    # One final check of the default path and both changed damping branches.
    for name in (*fit.TRAIN_NAMES,fit.HOLD_NAME):
        now=records["scaled_baseline"]["checks"][name]
        old=previous["records"]["1.0"]["checks"][name]
        for key in ("q_peak1","q_valley1","q_peak2","alpha_valley1"):
            for field in ("value","time_s"):
                assert abs(now[key]["predicted"][field]-old[key]["predicted"][field])<1e-10
    assert p["kd"]==.000023  # Removing plant damping never removes PID D.
    convergence={}
    for name in ("no_extra_damping","constant_valley_match"):
        rec=records[name];coeff=base.copy();coeff[2]=rec["kw_s_minus1"]
        fine=rate.simulate(hold,coeff,48,"end_gain",step=rate.STEP/2,damping_law=rec["law"])
        coarse=curves[name][fit.HOLD_NAME]
        error=float(np.max(np.abs(coarse["omega"]-np.interp(coarse["t"],fine["t"],fine["omega"]))));convergence[name]=error
        assert error<1e-6
    for record in records.values():
        for check in record["checks"].values():
            assert abs(check["first_active"]["start_s"]-14/48)<1e-12

    result={"source_version":p["source_version"],"fixed_ku_ka":base[:2].tolist(),
            "fixed_raw_PID":{k:p[k] for k in ("kp","ki","kd","integral_limit")},
            "control_hz":48,"activation_time_s":14/48,"lift_slope_per_rad":14.,
            "laws":{"none":"No explicit plant omega damping; original PID D and every other term retained.",
                    "constant":"omega_dot_damp=-kw*omega, independent of speed, density and mass.",
                    "rho_v_over_m":"omega_dot_damp=-kw*[(rho*V/m)/(rho0*V0/m0)]*omega."},
            "records":records,"constant_fit_search":search,
            "selection":"Only the constant law amplitude is matched to the first rate valley of L110/L27.5; ku/ka/PID are never refitted.",
            "scope":"Conditional forward from decoded creation state with measured speed/target; all loft off; L55 excluded from fit but previously viewed.",
            "interpretation_limit":"This is a removal/replacement test conditional on the current control and moment structure, not a proof that every possible game model needs an explicit damping term.",
            "verification":{"baseline_reproduced":True,"original_PID_D_retained":True,
                            "activation_unchanged":True,"step_halving_max_omega_difference_rad_s":convergence}}
    (OUT/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")

    plt=fit.plt;obs=hold["observed"]
    for filename,names,title in (
        ("no_damping.png",("scaled_baseline","no_extra_damping"),"Removing extra plant damping; original PID D retained"),
        ("damping_forms.png",("scaled_baseline","constant_same_release_value","constant_valley_match"),"Time-dependent versus constant plant damping")):
        fig,axs=plt.subplots(2,1,figsize=(12,8),sharex=True,layout="constrained")
        labels={"scaled_baseline":"Current rho V / m damping","no_extra_damping":"No extra plant damping",
                "constant_same_release_value":f"Constant b = {base[2]:.3f} /s",
                "constant_valley_match":f"Constant b = {kw:.3f} /s | first-valley fit"}
        colors={"scaled_baseline":"#2475b2","no_extra_damping":"#c34c3b",
                "constant_same_release_value":"#e28a25","constant_valley_match":"#239260"}
        for ax,key,scale,ylabel in ((axs[0],"omega",1.,"Pitch rate (rad/s)"),
                                   (axs[1],"alpha",180/math.pi,"Angle of attack (deg)")):
            mask=obs["t"]<=2.5
            ax.plot(obs["t"][mask],obs[key][mask]*scale,"ko",ms=4,label="Game replay")
            for name in names:
                r=curves[name][fit.HOLD_NAME]
                ax.plot(r["t"],r[key]*scale,color=colors[name],lw=1.7,label=labels[name])
            ax.axvspan(.65,.95,color="0.8",alpha=.15)
            ax.set(xlim=(0.,2.5),ylabel=ylabel)
            ax.grid(alpha=.2)
        axs[0].legend(fontsize=9,ncol=2)
        axs[1].set_xlabel("Time from release (s)")
        fig.suptitle("Loft OFF | L55 | "+title+"\n48 Hz, PID gains, control gain and restoring coefficient fixed",fontsize=13)
        fig.savefig(OUT/filename,dpi=180);plt.close(fig)
    print(json.dumps({"constant_match_b":kw,"verification":result["verification"],
                      "L55":{n:r["checks"][fit.HOLD_NAME] for n,r in records.items()}},indent=2))


if __name__=="__main__":main()
