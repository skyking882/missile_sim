#!/usr/bin/env python3
"""Bounded loft-only game-replay calibration on the freely integrated surface.

Two candidate families are declared before calibration. Four original raw
loft fields drive amplitude endpoints and a LOS-rate transition. Their chosen
semantics are hypotheses, not recovered game units or native equations.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aim120_model.guidance import estimate_time_to_go_s, interpolate_table, pn_acceleration
from aim120_model.aerodynamics import body_axes_for_state
from aim120_model.math3d import add, clamp, clamp_norm, dot, norm, scale, sub
from aim120_model.single_surface_game import SingleSurfaceGame, SurfaceCoefficients
from aim120_model.units import G0_MPS2
import compare_unified_game_replays as harness
import run_single_surface_game_candidate as candidate

OUT = ROOT / "outputs/unified_game_replays/free_loft"
TRAIN = ("H110", "H27.5")
PREDICTION = ("H55native", "H55clone")
FAMILIES = ("rate_latch", "rate_current")
FOLLOWUP_FAMILY = "rate_angle_latch"
BOUNDS = ((.25, 2.), (.25, 3.))
MAX_EVALUATIONS = 100
TIME_BUDGET_S = 720.
FIT_WINDOWS = ((0., 1.25), (1.25, 3.5))
SCALES = {"alpha_deg": 5., "omega_rad_s": .5, "gamma_deg": 2.,
          "y_m": 30., "speed_kmh": 30.}


class FreeLoft(SingleSurfaceGame):
    """Replace only the additive loft request, preserving every inner input."""
    def __init__(self, *args, loft_family="rate_latch", loft_coeff=(.65, 1.), **kwargs):
        super().__init__(*args, **kwargs)
        self.loft_family = loft_family
        self.loft_coeff = tuple(map(float, loft_coeff))
        self.initial_range = None
        self.minimum_weight = 1.

    def request(self, state, target, time_s, evaluation):
        cfg = self.config["guidance"]
        if not cfg.get("lofting_enabled", False):
            request, details = super().request(state, target, time_s, evaluation)
            return request, dict(details, wind_drag_n=evaluation.drag_n)
        rel = sub(target.position, state.position)
        rel_v = sub(target.velocity, state.velocity)
        pn, closing, los_rate = pn_acceleration(rel, rel_v, state.velocity, float(cfg["pn_gain"]))
        distance = norm(rel)
        if self.initial_range is None:
            self.initial_range = distance
        tgo = estimate_time_to_go_s(distance, closing, rel_v, self.config)
        hit_gain = interpolate_table(tgo, cfg["time_to_hit_gain_table"])
        pn = scale(pn, hit_gain)
        amp, transition = self.loft_coeff
        raw_rate = float(cfg["loft_omega_max_deg_s"])
        threshold = transition * math.radians(abs(raw_rate))
        # This deg/s rate interpretation is an unverified D-level hypothesis.
        rate = norm(los_rate)
        z = clamp(float(cfg["angle_to_acceleration_multiplier"]) * (rate/max(threshold, 1e-12)-1.), -60., 60.)
        current_weight = 1./(1.+math.exp(z))
        self.minimum_weight = min(self.minimum_weight, current_weight)
        weight = self.minimum_weight if self.loft_family in ("rate_latch",FOLLOWUP_FAMILY) else current_weight
        xi = clamp(distance / self.initial_range, 0., 1.)
        positive_endpoint = float(cfg["lofting_elevation_deg"])
        eta = 0.
        if self.loft_family == FOLLOWUP_FAMILY:
            axes = body_axes_for_state(state)
            eta = math.atan2(-dot(rel,axes.up),dot(rel,axes.forward))
            positive_endpoint -= float(cfg["angle_to_acceleration_multiplier"])*max(eta,0.)
        loft_g = amp * xi**2 * (positive_endpoint*weight + float(cfg["target_elevation_deg"])*(1.-weight))
        tangent = evaluation.wind_basis.forward
        maximum = float(cfg["maximum_lateral_acceleration_g"])*self.gravity
        commanded = add(pn, (0., loft_g*self.gravity, 0.))
        commanded = clamp_norm(sub(commanded, scale(tangent, dot(commanded, tangent))), maximum)
        required = clamp_norm(add(commanded, (0., self.gravity, 0.)), maximum)
        request = (dot(required, evaluation.wind_basis.up)/G0_MPS2,
                   dot(required, evaluation.wind_basis.right)/G0_MPS2)
        return request, {"range_m": distance, "tgo_s": tgo, "los_rate_rad_s": los_rate,
            "pn_mps2": pn, "loft_g": loft_g, "loft_active": True, "hit_gain": hit_gain,
            "loft_weight": weight, "loft_current_weight": current_weight,
            "loft_los_rate_magnitude_rad_s": rate, "loft_rate_threshold_rad_s": threshold,
            "loft_range_fraction": xi, "loft_relative_body_angle_rad":eta,
            "wind_drag_n":evaluation.drag_n}


def runner_for(family, coeff, substeps=4, perturbation=None):
    original_class = candidate.SingleSurfaceGame
    class ConfiguredFreeLoft(FreeLoft):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, loft_family=family, loft_coeff=coeff, **kwargs)
    def run(case, profile):
        changed = copy.deepcopy(profile)
        if perturbation:
            key, factor = perturbation
            changed["_model_config"]["guidance"][key] *= factor
        candidate.SingleSurfaceGame = ConfiguredFreeLoft
        try:
            result = candidate.runner_for("wind_linear", substeps, "rad_s")(case, changed)
        finally:
            candidate.SingleSurfaceGame = original_class
        result["model"].update(loft_interpretation="empirical_raw_endpoints_with_target_LOS_rate_transition",
            loft_family=family, loft_coefficients=list(coeff), loft_cap_unit_hypothesis=None,
            native_game_law_identified=False, raw_loft_semantics_evidence_grade="D")
        return result
    return run


def get_prediction(case, run):
    result = run(case, case["profile"])
    return harness.sample_series(result["samples"], case), result


def score_prediction(case, prediction):
    obs = case["observed"]
    scores = []
    for lo, hi in FIT_WINDOWS:
        mask = (obs["t"] >= lo) & (obs["t"] <= hi)
        if prediction["t"][-1] < hi-1e-6:
            return 1e6
        for key, unit in SCALES.items():
            err = np.interp(obs["t"][mask], prediction["t"], prediction[key])-obs[key][mask]
            scores.append(float(np.mean((err/unit)**2)))
    return float(np.mean(scores))


def key_events(case, prediction):
    obs = case["observed"]
    events = (("q_peak1", "omega_rad_s", "max", .35, .60),
              ("q_valley1", "omega_rad_s", "min", .60, .84),
              ("q_peak2", "omega_rad_s", "max", .85, 1.18),
              ("alpha_peak1", "alpha_deg", "max", .38, .76),
              ("alpha_recovery_min", "alpha_deg", "min", 1.2, 3.5),
              ("height_max", "y_m", "max", 0., math.inf))
    out = {}
    for name, key, mode, lo, hi in events:
        out[name] = {"observed": harness.extremum(obs["t"],obs[key],mode,lo,hi),
                     "model": harness.extremum(prediction["t"],prediction[key],mode,lo,hi)}
    return out


def plot_delta(cases, predictions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h, l = cases["H110"], cases["L110"]
    fields = ("omega_rad_s", "alpha_deg", "gamma_deg", "y_m", "speed_kmh")
    fig, axs = plt.subplots(len(fields), 2, figsize=(14, 15))
    for row, key in enumerate(fields):
        ax, delta = axs[row]
        for case, col, lab in ((h,"#c65037","Loft"),(l,"#227d90","No loft")):
            o,p = case["observed"],predictions[case["id"]]
            ax.scatter(o["t"],o[key],facecolors="none",edgecolors=col,s=15,label=lab+" game",linewidths=.7)
            ax.plot(p["t"],p[key],color=col,label=lab+" free model",lw=1.3)
        ot = h["observed"]["t"]
        mask = (ot >= l["observed"]["t"][0]) & (ot <= l["observed"]["t"][-1])
        delta.scatter(ot[mask], h["observed"][key][mask]-np.interp(ot[mask], l["observed"]["t"], l["observed"][key]),
            facecolors="none",edgecolors="#284855",s=15,label="Game loft minus no loft",linewidths=.7)
        hp, lp = predictions["H110"], predictions["L110"]
        t = hp["t"][(hp["t"] >= lp["t"][0]) & (hp["t"] <= lp["t"][-1])]
        delta.plot(t,np.interp(t,hp["t"],hp[key])-np.interp(t,lp["t"],lp[key]),color="#8054aa",label="Free model loft minus no loft")
        for a in (ax,delta):
            a.set_ylabel(key+" ("+harness.UNITS[key]+")");a.set_xlabel("Creation age (s)");a.grid(alpha=.2);a.set_xlim(0.,7.)
        if row == 0: ax.legend(fontsize=9);delta.legend(fontsize=9)
    fig.suptitle("110 game units: loft/no-loft overlay and difference; free speed, fixed frozen inner model",fontsize=16)
    fig.tight_layout(rect=(0,0,1,.97));fig.savefig(OUT/"110_overlay_delta.png",dpi=150);plt.close(fig)


def plot_request_drag(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h, l = results["H110"]["ticks"], results["L110"]["ticks"]
    ht=np.array([r["time_s"] for r in h]);lt=np.array([r["time_s"] for r in l])
    fig,axs=plt.subplots(3,1,figsize=(13,10))
    for key,label,color in (("loft_g","Loft bias (g)","#b54b35"),("loft_weight","Transition weight","#6b55a1")):
        axs[0].plot(ht,[r[key] for r in h],label=label,color=color)
    axs[0].set_ylabel("Bias / transition weight")
    axs[1].plot(ht,[r["request_g"][0] for r in h],label="Required pitch load",color="#b54b35")
    axs[1].plot(ht,[r["feedback_g"][0] for r in h],label="Free model feedback",color="#277c91")
    axs[1].set_ylabel("Specific normal load (g)")
    for rows,t,col,label in ((h,ht,"#b54b35","Loft wind drag"),(l,lt,"#277c91","No-loft wind drag")):
        axs[2].plot(t,np.array([r["wind_drag_n"] for r in rows])/1000,label=label,color=col)
    axs[2].set_ylabel("Candidate wind drag (kN)")
    for ax in axs:ax.grid(alpha=.2);ax.legend();ax.set_xlabel("Creation age (s)");ax.set_xlim(0.,7.)
    fig.suptitle("H110 request, unloading and free drag; drag is a model output, not native telemetry",fontsize=16)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(OUT/"loft_request_drag.png",dpi=150);plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    frozen_before = candidate.FROZEN.read_bytes()
    cases = {c["id"]: c for c in harness.load_cases(candidate.SNAPSHOT)}
    evaluations, cache = [], {}
    baselines = {}
    for name in TRAIN:
        pred, _ = get_prediction(cases[name],candidate.runner_for("wind_linear",4,"rad_s"))
        baselines[name] = score_prediction(cases[name], pred)
    def evaluate(family, point):
        point = tuple(round(clamp(float(x),lo,hi), 8) for x,(lo,hi) in zip(point,BOUNDS))
        key=(family,*point)
        if key in cache: return cache[key]
        if len(evaluations) >= MAX_EVALUATIONS or time.monotonic()-start > TIME_BUDGET_S:
            return math.inf
        scores = []
        for name in TRAIN:
            case = dict(cases[name], scenario=dict(cases[name]["scenario"], max_simulation_time_s=3.5))
            prediction, _ = get_prediction(case,runner_for(family,point))
            scores.append(score_prediction(case,prediction))
        score = float(np.mean(scores))
        cache[key] = score
        evaluations.append({"family":family,"coefficients":list(point),"score":score,"training_scores":scores})
        print(len(evaluations),family,point,round(score,6),flush=True)
        return score
    # Predeclared grid followed by bounded coordinate refinement; 100 total.
    best = {}
    for family in FAMILIES:
        for a in (.35,.5,.65,.8,1.):
            for rate in (.6,.9,1.2,1.5): evaluate(family,(a,rate))
        choice=min((e for e in evaluations if e["family"]==family),key=lambda e:e["score"])
        point = np.array(choice["coefficients"]);value=choice["score"]
        for da,dr in ((.1,.25),(.05,.125),(.025,.0625)):
            for _ in range(2):
                options=[(value,point.copy())]
                for offset in ((da,0),(-da,0),(0,dr),(0,-dr)):
                    trial=point+offset;options.append((evaluate(family,trial),trial))
                value,point=min(options,key=lambda p:p[0])
        best[family]={"score":value,"coefficients":point.tolist()}
    initial_selected=min(best,key=lambda k:best[k]["score"])
    initial_coeff=best[initial_selected]["coefficients"]
    # Authorized follow-up after the initial two-family fit revealed a late
    # second positive rate peak and excessive early negative incidence.
    # Replace the positive endpoint by 24-17.5*max(body-relative LOS angle,0).
    # The same two coefficients remain. Twenty grid points plus at most eight
    # refinement points keep the whole experiment below 100 evaluations.
    for a in (.6,.65,.7,.75,.8):
        for rate in (1.,1.15,1.3,1.45): evaluate(FOLLOWUP_FAMILY,(a,rate))
    choice=min((e for e in evaluations if e["family"]==FOLLOWUP_FAMILY),key=lambda e:e["score"])
    point=np.array(choice["coefficients"]);value=choice["score"]
    for _ in range(2):
        options=[(value,point.copy())]
        for offset in ((.025,0),(-.025,0),(0,.075),(0,-.075)):
            trial=point+offset;options.append((evaluate(FOLLOWUP_FAMILY,trial),trial))
        value,point=min(options,key=lambda p:p[0])
    best[FOLLOWUP_FAMILY]={"score":value,"coefficients":point.tolist()}
    followup_gate={"training_features":{},"prediction_metrics":{},"passed":True}
    initial_predictions, initial_records = {}, []
    for name in (*TRAIN,*PREDICTION):
        case=cases[name]
        initial,_=get_prediction(case,runner_for(initial_selected,initial_coeff))
        followup,_=get_prediction(case,runner_for(FOLLOWUP_FAMILY,point))
        initial_predictions[name]=initial
        initial_record,_=harness.compare_series(case,initial,{})
        initial_record["loft_events"]=key_events(case,initial)
        initial_records.append(initial_record)
        if name in TRAIN:
            initial_events=key_events(case,initial);followup_events=key_events(case,followup)
            errors={}
            for event,field in (("q_peak2","time_s"),("alpha_recovery_min","value")):
                before=abs(initial_events[event]["model"][field]-initial_events[event]["observed"][field])
                after=abs(followup_events[event]["model"][field]-followup_events[event]["observed"][field])
                errors[event+"_"+field]={"initial_abs_error":before,"followup_abs_error":after}
                followup_gate["passed"] &= after < before
            followup_gate["training_features"][name]=errors
        else:
            b,_=harness.compare_series(case,initial,{})
            f,_=harness.compare_series(case,followup,{})
            errors={}
            for k in ("alpha_deg","omega_rad_s","gamma_deg","y_m"):
                before=b["windows"]["full"]["metrics"][k]["rmse"]
                after=f["windows"]["full"]["metrics"][k]["rmse"]
                errors[k]={"initial_rmse":before,"followup_rmse":after}
                followup_gate["passed"] &= after <= before
            followup_gate["prediction_metrics"][name]=errors
    selected=FOLLOWUP_FAMILY if followup_gate["passed"] and best[FOLLOWUP_FAMILY]["score"]<best[initial_selected]["score"] else initial_selected
    coeff=best[selected]["coefficients"]
    print("SELECTED",selected,coeff,best[selected]["score"],flush=True)
    run=runner_for(selected,coeff)
    predictions,results,arrays,records={},{},{},[]
    for name,case in cases.items():
        prediction,result=get_prediction(case,run)
        predictions[name],results[name]=prediction,result
        record,aligned=harness.compare_series(case,prediction,result["summary"])
        record["model_runtime_metadata"]=result["model"]
        if case["dataset"]=="native_2.59":record["loft_events"]=key_events(case,prediction)
        records.append(record)
        arrays.update({name+"__model__"+k:v for k,v in prediction.items()})
        arrays.update({name+"__aligned__"+k:v for k,v in aligned.items()})
        print("FINAL",name,record.get("compared_rows"),flush=True)
    invariance={}
    for name,case in cases.items():
        if not case["scenario"]["loft_enabled"]:
            base,_=get_prediction(case,candidate.runner_for("wind_linear",4,"rad_s"))
            invariance[name]={k:float(np.max(np.abs(predictions[name][k]-base[k]))) for k in base}
            assert max(invariance[name].values())==0.
    convergence={}
    for name in ("H110","L110","r77-1"):
        fine,_=get_prediction(cases[name],runner_for(selected,coeff,8))
        base=predictions[name]
        common=base["t"]<=fine["t"][-1]
        convergence[name]={k:float(np.max(np.abs(base[k][common]-np.interp(base["t"][common],fine["t"],fine[k])))) for k in base if k!="t"}
    sensitivity={}
    for rawkey in ("lofting_elevation_deg","target_elevation_deg","angle_to_acceleration_multiplier","loft_omega_max_deg_s"):
        pred,_=get_prediction(cases["H110"],runner_for(selected,coeff,perturbation=(rawkey,1.01)))
        base=predictions["H110"];m=base["t"]<=pred["t"][-1]
        sensitivity[rawkey]={k:float(np.max(np.abs(base[k][m]-np.interp(base["t"][m],pred["t"],pred[k])))) for k in ("alpha_deg","omega_rad_s","y_m")}
        assert max(sensitivity[rawkey].values())>1e-10
        # Repeat on no-loft to verify every raw field remains inactive.
        off,_=get_prediction(cases["L110"],runner_for(selected,coeff,perturbation=(rawkey,1.01)))
        assert all(np.array_equal(off[k],predictions["L110"][k]) for k in off)
    assert candidate.FROZEN.read_bytes()==frozen_before
    np.savez_compressed(OUT/"trajectories.npz",**arrays)
    np.savez_compressed(OUT/"initial_candidate_trajectories.npz",
        **{n+"__model__"+k:v for n,p in initial_predictions.items() for k,v in p.items()})
    (OUT/"ticks.json").write_text(json.dumps({n:results[n]["ticks"] for n in ("H110","L110")},indent=2,allow_nan=False))
    harness.plot_cases(list(cases.values()),arrays,OUT,"Free LOS-rate loft")
    plot_delta(cases,predictions)
    plot_request_drag(results)
    manifest={"status":"bounded_candidate_pending_root_review","production_changed":False,
        "goal_complete":False,"selected_family":selected,"coefficients":coeff,
        "formula":("b_g=c_amp*clip(R/R0,0,1)^2*{[24-17.5*max(eta,0)]*w-3.5*(1-w)}; eta=atan2(-R.body_up,R.body_forward); " if selected==FOLLOWUP_FAMILY else
            "b_g=c_amp*clip(R/R0,0,1)^2*[24*w-3.5*(1-w)]; ")+"w_current=1/(1+exp(17.5*(|LOS_rate|/(c_rate*radians(0.5))-1))); w=min(previous w,w_current) for rate_latch/rate_angle_latch, w=w_current for rate_current",
        "coefficient_search_bounds":BOUNDS,"nominal_coefficient_evidence":"bounded replay calibration, not derivation; search bounds are not confidence intervals",
        "raw_semantics_evidence_grade":"D","raw_fields_unchanged":True,
        "four_field_roles":{"24":"positive bias endpoint in g-interface amplitude","-3.5":"recovery bias endpoint","17.5":"smooth transition steepness and follow-up relative body-angle feedback","0.5":"hypothesized target LOS-rate scale in deg/s"},
        "predeclared_families":FAMILIES,"selection":"minimum equal training MSE over two windows/five fields with fixed scales; initial two families never selected on H55",
        "authorized_followup":{"family":FOLLOWUP_FAMILY,"formula":"replace 24*w by [24-17.5*max(eta,0)]*w; eta=atan2(-R.body_up,R.body_forward); same two coefficients",
            "trigger":"initial candidate's second positive omega peak was late and the negative incidence valley early/deep",
            "gate":"both training second-peak time and negative-valley magnitude improve, with no degradation of four H55 prediction RMSE metrics; prediction veto is disclosed, hence H55 is not blind validation",
            "initial_selected":initial_selected,"initial_coefficients":initial_coeff,"gate_results":followup_gate},
        "initial_candidate_cases":initial_records,
        "training_cases":TRAIN,"fit_windows_s":FIT_WINDOWS,"fit_scales":SCALES,
        "prediction_cases":PREDICTION,"prediction_provenance":"excluded from coefficient fit; historically inspected and used for follow-up rejection gate, not blind",
        "cross_version":"r77-1 screenshot 2.57 diagnostic with no refit; native calibration 2.59 uses matching raw inputs",
        "baseline_training_scores":baselines,"family_optima":best,"candidate_evaluations":evaluations,
        "evaluation_count":len(evaluations),"elapsed_s":time.monotonic()-start,
        "frozen_sha256":hashlib.sha256(frozen_before).hexdigest(),"surface_coefficients":dataclasses.asdict(SurfaceCoefficients()),
        "source_profile_snapshot":str(candidate.SNAPSHOT),"source_versions_separated":True,
        "free_state_inputs":{"missile":"creation position/velocity/orientation/body rates only","target":"external decoded target position/derivative on source clock","observed_missile_after_creation":False,"time_shift_s":0.,"resets":False},
        "fixed_controller_hz":48,"plant_substeps":4,"raw_PID_preserved":True,
        "verification":{"all_series_finite":True,"no_loft_max_abs_differences":invariance,
            "plant_4_to_8_max_abs_differences":convergence,"raw_field_1pct_sensitivity":sensitivity,
            "all_raw_perturbations_no_loft_exactly_inactive":True,"frozen_bytes_unchanged":True},
        "limitations":["Native b5 timing and tied states are retained; observation operator remains unresolved.",
            "Free lift/moment family is empirical; native force couple and fin mapping are not identified.",
            "Historical screenshot target/scenario/unsigned G limitations remain in records.",
            "One-way unloading is candidate guidance memory, not evidence of native engine strategy."],
        "cases":records}
    (OUT/"results.json").write_text(json.dumps(harness.jsonable(manifest),indent=2,allow_nan=False))


if __name__=="__main__":main()
