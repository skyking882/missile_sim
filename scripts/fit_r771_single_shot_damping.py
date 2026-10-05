"""Fit damping on L110 only; freeze the choice before cross-shot evaluation.

This is a conditional replay experiment. Only damping changes; the existing
control, restoring gain, 48 Hz PID, force law and launch states remain fixed.
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

import check_r771_no_loft_damping as checks

rate=checks.rate
fit=rate.fit
OUT=fit.OUT/"single_shot_damping"
TRAIN="L110"
VALIDATION=("L27.5","L55")
FIT_STEP=1/192
EVENTS=(("q_peak1","omega",.35,.58,"max",.025),
        ("q_valley1","omega",.60,.84,"min",.025),
        ("q_peak2","omega",.85,1.18,"max",.025),
        ("q_valley2","omega",1.18,1.45,"min",.02),
        ("q_peak3","omega",1.38,1.72,"max",.02),
        ("alpha_peak1","alpha",.38,.76,"max",math.radians(.2)),
        ("alpha_valley1","alpha",.80,1.02,"min",math.radians(.2)))
FAMILIES={
    "scaled_linear":{"bounds":[[.1,12.]],"seeds":[[3.77],[5.]],
        "formula":"D=-B*k*omega"},
    "speed_power":{"bounds":[[.1,12.],[-4.,6.]],"seeds":[[3.77,1.],[3.,3.]],
        "formula":"D=-B*k*(V/V0)^(p-1)*omega"},
    "quadratic_rate":{"bounds":[[0.,12.],[0.,40.]],"seeds":[[3.77,0.],[2.,8.]],
        "formula":"D=-B*(k+c*abs(omega))*omega"},
    "relative_rate":{"bounds":[[.1,12.],[0.,1.]],"seeds":[[3.77,0.],[3.77,1.]],
        "formula":"D=-B*k*(omega-c*gamma_dot); c=1 gives alpha_dot"},
    "speed_power_relative":{"bounds":[[.1,12.],[-4.,6.],[0.,1.]],
        "seeds":[[3.,2.,1.],[3.77,1.,0.]],
        "formula":"D=-B*k*(V/V0)^(p-1)*(omega-c*gamma_dot)"},
}


def damping_function(name,x):
    k=float(x[0])
    if name=="scaled_linear":return lambda om,gd,a,b:-b*k*om
    c=float(x[1])
    if name=="speed_power":return lambda om,gd,a,b:-b*k*(a/b)**(c-1)*om
    if name=="quadratic_rate":return lambda om,gd,a,b:-b*(k+c*abs(om))*om
    if name=="relative_rate":return lambda om,gd,a,b:-b*k*(om-c*gd)
    if name=="speed_power_relative":return lambda om,gd,a,b:-b*k*(a/b)**(c-1)*(om-float(x[2])*gd)
    raise ValueError(name)


def simulate(shot,coeff,name,x,step=FIT_STEP):
    return rate.simulate(shot,coeff,48,"end_gain",step=step,
                         damping_function=damping_function(name,x))


def extremum(t,y,lo,hi,kind):
    """Continuous parabolic location for fitting; native extrema also reported."""
    ids=np.flatnonzero((t>=lo)&(t<=hi))
    j=ids[np.argmax(y[ids]) if kind=="max" else np.argmin(y[ids])]
    if 0<j<len(t)-1 and ids[0]<j<ids[-1]:
        c=np.polyfit(t[j-1:j+2]-t[j],y[j-1:j+2],2)
        offset=-c[1]/(2*c[0]) if abs(c[0])>1e-12 else 0.
        if t[j-1]-t[j]<=offset<=t[j+1]-t[j]:
            return float(t[j]+offset),float(np.polyval(c,offset))
    return float(t[j]),float(y[j])


def negative_lobe(curve):
    t,y=curve["t"],curve["omega"]
    crossing=[]
    for j in range(len(t)-1):
        if .55<=t[j]<=1.12 and y[j]*y[j+1]<0:
            z=t[j]-y[j]*(t[j+1]-t[j])/(y[j+1]-y[j])
            crossing.append((float(z),bool(y[j]>0)))
    down=next((z for z,d in crossing if d),None)
    up=next((z for z,d in crossing if not d and down is not None and z>down),None)
    if down is None or up is None:return None
    mask=(t>down)&(t<up)
    integral=np.trapezoid(np.r_[0.,y[mask],0.],np.r_[down,t[mask],up])
    return {"down_s":down,"up_s":up,"width_s":up-down,
            "integral_deg":math.degrees(float(integral))}


def objective(shot,pred):
    obs=shot["observed"]
    residual=[]
    for label,key,lo,hi,kind,scale in EVENTS:
        ot,ov=extremum(obs["t"],obs[key],lo,hi,kind)
        pt,pv=extremum(pred["t"],pred[key],lo,hi,kind)
        residual.extend([(pt-ot)/.03,(pv-ov)/scale])
    actual,calc=negative_lobe(obs),negative_lobe(pred)
    if calc is None:return 1e6
    residual.extend([(calc["width_s"]-actual["width_s"])/.03,
                     (calc["integral_deg"]-actual["integral_deg"])/.2])
    event_loss=float(np.mean(np.square(residual)))
    # Continuous shape at native sample times catches spurious features between
    # the named peaks; no time shift or per-trajectory normalization is fitted.
    mask=(obs["t"]>=.30)&(obs["t"]<=2.45)
    wave=[]
    for key,scale in (("omega",.03),("alpha",math.radians(.2))):
        d=np.interp(obs["t"][mask],pred["t"],pred[key])-obs[key][mask]
        wave.append(float(np.mean((d/scale)**2)))
    return event_loss+.5*float(np.mean(wave))


def optimize(fun,bounds,seeds):
    """Small bounded Nelder-Mead search; no optional optimizer dependency."""
    bounds=np.array(bounds,float);lo=bounds[:,0];span=bounds[:,1]-lo
    cache={};evaluations=0
    def calc(z):
        nonlocal evaluations
        z=np.clip(z,0,1);key=tuple(np.round(z,9))
        if key not in cache:
            cache[key]=fun(lo+span*z);evaluations+=1
        return z,cache[key]
    best=None
    for seed in seeds:
        z=(np.array(seed)-lo)/span
        simplex=[calc(z)]
        for j in range(len(z)):
            zz=z.copy();zz[j]+=(-.07 if z[j]>.85 else .07)
            simplex.append(calc(zz))
        for iteration in range(90):
            simplex.sort(key=lambda zf:zf[1])
            if np.max(np.ptp([q[0] for q in simplex],axis=0))<2e-4:break
            center=np.mean([q[0] for q in simplex[:-1]],axis=0)
            reflected=calc(center+(center-simplex[-1][0]))
            if simplex[0][1]<=reflected[1]<simplex[-2][1]:simplex[-1]=reflected
            elif reflected[1]<simplex[0][1]:
                expanded=calc(center+2*(reflected[0]-center))
                simplex[-1]=min((expanded,reflected),key=lambda q:q[1])
            else:
                target=reflected if reflected[1]<simplex[-1][1] else simplex[-1]
                contracted=calc(center+.5*(target[0]-center))
                if contracted[1]<target[1]:simplex[-1]=contracted
                else:simplex=[simplex[0]]+[calc(simplex[0][0]+.5*(q[0]-simplex[0][0])) for q in simplex[1:]]
        local=min(simplex,key=lambda q:q[1])
        if best is None or local[1]<best[1]:best=local
    return {"parameters":(lo+span*best[0]).tolist(),"objective":best[1],
            "evaluations":evaluations,"bounds":bounds.tolist()}


def load():
    old=json.loads((fit.OUT/"schedule_48hz"/"zero_memory_standard_D"/"results.json").read_text())
    coeff=np.array(old["refit_coefficients"],float)
    raw,p=fit.batch.load_all()
    assert [p[k] for k in ("kp","ki","kd")]==[.002125,.006481,.000023]
    return coeff,raw,p


def train(combine_only=False):
    OUT.mkdir(parents=True,exist_ok=True)
    coeff,raw,p=load();shot=fit.make_shot(raw[TRAIN])
    assert not shot["p"]["loft_enabled"]
    baseline=rate.simulate(shot,coeff,48,"end_gain",step=FIT_STEP)
    results=(json.loads((OUT/"fit.json").read_text())["candidates"] if combine_only else {})
    for name,spec in FAMILIES.items():
        if combine_only and name!="speed_power_relative":continue
        answer=optimize(lambda x:objective(shot,simulate(shot,coeff,name,x)),spec["bounds"],spec["seeds"])
        answer["formula"]=spec["formula"]
        pred=simulate(shot,coeff,name,answer["parameters"])
        answer["training_checks"]=checks.evaluate(shot,pred)
        answer["negative_lobe"]=negative_lobe(pred)
        results[name]=answer
        print(json.dumps({"family":name,**{k:answer[k] for k in ("parameters","objective","evaluations")}}),flush=True)
    chosen=min(results,key=lambda key:results[key]["objective"])
    record={"training_case":TRAIN,"fit_window_s":[.3,2.45],
            "validation_cases_excluded_from_new_fitting":list(VALIDATION),
            "fixed_previous_coefficients":coeff.tolist(),"source_version":p["source_version"],
            "fixed_PID":{k:p[k] for k in ("kp","ki","kd","integral_limit")},
            "baseline_objective":objective(shot,baseline),"candidates":results,
            "selected_on_L110_only":chosen,"fit_step_s":FIT_STEP,
            "objective":"Mean scaled squared peak/valley amplitude and timing, first negative-rate lobe width and area, plus 0.5 times native omega/AoA waveform residual. Event time scale .03s; amplitude .02-.025rad/s or .2deg. No time shift fitted.",
            "scope":"New damping parameters and family selected using L110 only. Existing ku/ka were previously calibrated using L110 and L27.5, so cross-shot evaluation is not fully independent whole-model validation. L55 was previously viewed. Speed and target motion are measured inputs, not predicted."}
    (OUT/"fit.json").write_text(json.dumps(record,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"selected":chosen,"baseline_objective":record["baseline_objective"]}),flush=True)


def validate():
    frozen=json.loads((OUT/"fit.json").read_text())
    coeff,raw,p=load()
    chosen=frozen["selected_on_L110_only"]
    params=frozen["candidates"][chosen]["parameters"]
    curves={};records={};shots={}
    for name in (TRAIN,*VALIDATION):
        shot=fit.make_shot(raw[name]);shots[name]=shot
        assert not shot["p"]["loft_enabled"]
        curves[name]={"baseline":rate.simulate(shot,coeff,48,"end_gain"),
                      "selected":simulate(shot,coeff,chosen,params,step=rate.STEP)}
        records[name]={}
        for tag,pred in curves[name].items():
            records[name][tag]={"checks":checks.evaluate(shot,pred),"objective":objective(shot,pred),
                                "negative_lobe":negative_lobe(pred)}
        records[name]["observed_negative_lobe"]=negative_lobe(shot["observed"])
    # Final check: preserved default, fixed activation, no force/PID changes,
    # and selected numerical result converges without changing control rate.
    equivalence=0.
    # Compare callback and unchanged default directly rather than assuming the
    # previous result JSON uses a particular report schema.
    test=simulate(shots[TRAIN],coeff,"scaled_linear",[coeff[2]],step=rate.STEP)
    for key in ("omega","alpha","gamma","y"):
        equivalence=max(equivalence,float(np.max(np.abs(test[key]-curves[TRAIN]["baseline"][key]))))
    assert equivalence<1e-10
    for both in curves.values():
        for pred in both.values():assert abs(pred["first_active"]["start_s"]-14/48)<1e-12
    fine=simulate(shots[TRAIN],coeff,chosen,params,step=rate.STEP/2)
    base=curves[TRAIN]["selected"]
    integration_error=float(np.max(np.abs(base["omega"]-np.interp(base["t"],fine["t"],fine["omega"]))))
    fit_resolution=simulate(shots[TRAIN],coeff,chosen,params,step=FIT_STEP)
    fit_difference=float(np.max(np.abs(fit_resolution["omega"]-np.interp(fit_resolution["t"],base["t"],base["omega"]))))
    assert integration_error<1e-6 and fit_difference<1e-4
    result={**frozen,"records":records,"verification":{"unchanged_default_max_difference":equivalence,
            "activation_unchanged":True,"step_halving_max_omega_difference":integration_error,
            "fit_vs_fine_step_max_omega_difference":fit_difference}}
    (OUT/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    plot(shots,curves,chosen)
    print(json.dumps({"selected":chosen,"parameters":params,"verification":result["verification"],
                      "scores":{name:{tag:records[name][tag]["objective"] for tag in ("baseline","selected")} for name in shots}},indent=2))


def plot(shots,curves,chosen):
    plt=fit.plt
    fig,axs=plt.subplots(2,3,figsize=(15,8),sharex=True,sharey="row",layout="constrained")
    for col,name in enumerate((TRAIN,*VALIDATION)):
        obs=shots[name]["observed"]
        for row,(key,scale,label) in enumerate((("omega",1.,"Pitch rate (rad/s)"),("alpha",180/math.pi,"Angle of attack (deg)"))):
            ax=axs[row,col];mask=obs["t"]<=2.3
            ax.plot(obs["t"][mask],obs[key][mask]*scale,"ko",ms=3.7,label="Game replay")
            for tag,color,legend in (("baseline","#888888","Previous damping"),("selected","#d77324","Selected damping")):
                pred=curves[name][tag];mask=pred["t"]<=2.3
                ax.plot(pred["t"][mask],pred[key][mask]*scale,color=color,lw=1.8,
                        ls="--" if tag=="baseline" else "-",label=legend)
            ax.axvspan(.65,.95,color="0.8",alpha=.15)
            ax.set(xlim=(.2,2.3),xlabel="Time from release (s)")
            if col==0:ax.set_ylabel(label)
            ax.grid(alpha=.18)
        axs[0,col].set_title(name+(" | FIT" if name==TRAIN else " | VALIDATION"))
    axs[0,0].legend(fontsize=9)
    fig.suptitle("Loft OFF | one damping law fitted on L110, then frozen for both other launches\n"+
                 chosen+" | PID, 48 Hz control, control/restoring gains and lift unchanged",fontsize=13)
    fig.savefig(OUT/"comparison.png",dpi=180)
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("phase",choices=("fit","combine","validate"))
    args=parser.parse_args()
    if args.phase=="validate":validate()
    else:train(combine_only=args.phase=="combine")
