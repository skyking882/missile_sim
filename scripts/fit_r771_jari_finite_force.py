#!/usr/bin/env python3
"""Game-only fixed finite-angle force experiment: six free runs, no refit.

The two positive angles were calibrated on Jari60 same-state lift force only.
All moment, controller, outer request, drag and loft parameters are frozen.
Current production code and previous artifacts are read-only inputs.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/missle_sim_mpl")
import numpy as np

import compare_r771_jari_offaxis as comparison
import compare_unified_game_replays as harness
import fit_r771_jari_capture as fields
import fit_r771_jari_drag as drag
import fit_r771_jari_request_slew as slew
from aim120_model.aerodynamics import body_axes_for_state, pitch_yaw_from_quaternion
from aim120_model.dynamics import SimState
from aim120_model.math3d import add, dot, norm, scale
from aim120_model.single_surface_game import SurfaceCoefficients
from aim120_model.units import G0_MPS2

OUTPUT = ROOT / "outputs/jari_offaxis/finite_force"
ALPHA_S = 0.5520275850474751
WIDTH = 0.32122305878546353
SLEW = 110.
UNITS = {"speed_mps": "m/s", "alpha_deg": "deg", "normal_specific_load_g": "G",
         "body_yaw_deg": "deg", "body_yaw_rate_deg_s": "deg/s"}


class FiniteForceGame(slew.RequestSlewGame):
    """One resultant and a same-station free couple; frozen angular law."""
    def evaluate(self, state, time_s, held):
        original = super().evaluate(state, time_s, held)
        incidence = math.hypot(*original.local_incidence)
        if incidence <= ALPHA_S:
            return original
        ratio = 1./(1.+((incidence-ALPHA_S)/WIDTH)**2)
        lift = scale(original.wind_lift_force, ratio)
        aerodynamic = add(lift, original.wind_drag_force)
        axes = body_axes_for_state(state)
        specific = scale(add(aerodynamic, scale(axes.forward, original.thrust)), 1./original.mass)
        acceleration = add(specific, (0., -self.gravity, 0.))
        feedback = (dot(specific, original.wind_basis.up)/G0_MPS2,
                    dot(specific, original.wind_basis.right)/G0_MPS2)
        lever = (-self.arm*dot(aerodynamic, axes.up), -self.arm*dot(aerodynamic, axes.right))
        couple = tuple(original.inertia*q-l for q, l in zip(original.angular_acceleration, lever))
        return dataclasses.replace(original, acceleration=acceleration, specific_acceleration=specific,
            aerodynamic_force=aerodynamic, wind_lift_force=lift, feedback_g=feedback,
            lever_moment=lever, reference_couple=couple)


def make_model(config, coefficients):
    return FiniteForceGame(config, request_slew_g_s=SLEW, drag_coefficients=coefficients,
        coefficients=SurfaceCoefficients(), force_basis="wind_linear", plant_substeps=4,
        loft_cap_unit="rad_s")


def sample_state(sample):
    q = tuple(sample["orientation_quaternion"])
    p, y = pitch_yaw_from_quaternion(q)
    return SimState(tuple(sample["position_m"]), tuple(sample["velocity_mps"]), p, y,
        sample["pitch_rate_rad_s"], sample["yaw_rate_rad_s"], sample["mass_kg"],
        orientation_quaternion=q)


def calibration_record(inputs):
    x = inputs["jari_259_60"]
    ref = x["reference"]
    mask = ((ref["geometric_incidence_deg"] >= 35.) & (ref["geometric_incidence_deg"] <= 52.) &
            (ref["t"] > .25) & (ref["t"] < 7.))
    if mask.sum() != 48:
        raise AssertionError("The previously calibrated 48 force rows changed")
    return {"family": "C1 rational-squared finite-angle rolloff of original full wind-lift vector",
        "formula": "s=hypot(local_pitch,local_yaw); R=1/(1+(max(s-alpha_s,0)/width)^2); Fa=R*L_linear+D_fixed",
        "parameters_rad": {"alpha_s": ALPHA_S, "width": WIDTH},
        "parameters_deg": {"alpha_s": math.degrees(ALPHA_S), "width": math.degrees(WIDTH)},
        "force_coefficient_peak_local_angle_deg": math.degrees(math.hypot(ALPHA_S, WIDTH)),
        "calibration": {"case": "jari_259_60", "export_sha256": comparison.sha256(x["export_path"]),
            "method": "prior phase same-state least squares of lift-force residual N; no free-flight refit",
            "force_target": "source acc_lift * source mass; magnitude only; not total normal load",
            "original_row_indices": np.flatnonzero(mask), "number_rows": int(mask.sum()),
            "geometric_angle_support_deg": [float(ref["geometric_incidence_deg"][mask].min()), float(ref["geometric_incidence_deg"][mask].max())],
            "speed_support_mps": [float(ref["speed_mps"][mask].min()), float(ref["speed_mps"][mask].max())],
            "time_support_s": [float(ref["t"][mask].min()), float(ref["t"][mask].max())],
            "local_angle_method": "unchanged 2-component local incidence, using source quaternion 3-point body rates",
            "lift_RMS_residual_N": 319.34522277895996,
            "force_only_conditional_curvature_two_standard_errors_deg": {"alpha_s": .35261780762431655, "width": .5809397829837809}},
        "parameters_changed_after_free_flight": False, "source_native_CyMaxAoA_mapped": False,
        "evidence_grade": "D interim Jari game replication hypothesis",
        "low_angle_branch": "exact original evaluation for local angle <= alpha_s",
        "fixed": {"drag": dataclasses.asdict(x["drag_coefficients"]), "surface": dataclasses.asdict(SurfaceCoefficients()),
            "request_slew_g_s": SLEW, "relative_PN_blend": .5, "loft_coefficients": [.75, 1.3],
            "raw_PID": x["config"]["control"]["pid"], "clock_hz": 48, "plant_substeps": 4},
        "new_free_runs": 6, "baseline_free_runs": 0, "composite_score": False}


def force_diagnostics(model, result):
    names = ("local_angle_rad", "rolloff_factor", "lift_n", "drag_n", "feedback_pitch_g", "feedback_yaw_g")
    values = {name: [] for name in names}
    maxima = {"aerodynamic_vector_closure_n": 0., "specific_vector_closure_n": 0.,
        "same_station_moment_closure_nm": 0., "drag_change_n": 0., "same_state_qdot_change_rad_s2": 0.}
    low_exact = 0
    for sample in result["samples"]:
        state = sample_state(sample)
        t, held = sample["time_s"], tuple(sample["held_fin_rad"])
        f = model.evaluate(state, t, held)
        old = slew.RequestSlewGame.evaluate(model, state, t, held)
        local = math.hypot(*old.local_incidence)
        r = 1. if local <= ALPHA_S else 1./(1.+((local-ALPHA_S)/WIDTH)**2)
        if r == 1.:
            if f != old: raise AssertionError("low-angle original evaluation not exact")
            low_exact += 1
        axes = body_axes_for_state(state)
        errors = {"aerodynamic_vector_closure_n": norm(tuple(a-b-c for a,b,c in zip(f.aerodynamic_force,f.wind_lift_force,f.wind_drag_force))),
            "specific_vector_closure_n": norm(tuple(f.mass*s-a-f.thrust*b for s,a,b in zip(f.specific_acceleration,f.aerodynamic_force,axes.forward))),
            "same_station_moment_closure_nm": max(abs(f.inertia*q-l-c) for q,l,c in zip(f.angular_acceleration,f.lever_moment,f.reference_couple)),
            "drag_change_n": norm(tuple(a-b for a,b in zip(f.wind_drag_force,old.wind_drag_force))),
            "same_state_qdot_change_rad_s2": max(abs(a-b) for a,b in zip(f.angular_acceleration,old.angular_acceleration))}
        for name,value in errors.items(): maxima[name] = max(maxima[name],value)
        for name,value in zip(names,(local,r,norm(f.wind_lift_force),f.drag_n,*f.feedback_g)): values[name].append(value)
    if max(maxima.values()) > 1e-8: raise AssertionError(maxima)
    return {name:np.asarray(value) for name,value in values.items()}, {
        "saved_free_states_checked":len(result["samples"]), "low_angle_exact_evaluation_checks":low_exact,
        "maximum_absolute_residuals":maxima, "equivalent_surface_station_m":-model.arm,
        "same_state_held_angular_law_unchanged":True, "moment_law_independently_verified":False}


def intervals(t, mask):
    ids = np.flatnonzero(mask)
    if not len(ids): return []
    groups = np.split(ids,np.flatnonzero(np.diff(ids)>1)+1)
    return [[float(t[g[0]]),float(t[g[-1]])] for g in groups]


def support_report(t, state, fit):
    alo, ahi = fit["calibration"]["geometric_angle_support_deg"]
    vlo, vhi = fit["calibration"]["speed_support_mps"]
    # Calibrated rows are selected by geometric incidence. Local norm is shown separately.
    a = state["alpha_deg"]; v = state["speed_mps"]
    masks = {"geometric_angle_below_training":a<alo, "geometric_angle_above_training":a>ahi,
        "speed_below_training":v<vlo, "speed_above_training":v>vhi,
        "force_rolloff_active":state["rolloff_factor"]<1.,
        "rolloff_active_outside_angle_or_speed_training":(state["rolloff_factor"]<1.)&((a<alo)|(a>ahi)|(v<vlo)|(v>vhi))}
    return {"saved_free_clock_hz":48, "local_angle_range_deg":np.degrees([state["local_angle_rad"].min(),state["local_angle_rad"].max()]),
        "geometric_angle_range_deg":[float(a.min()),float(a.max())], "speed_range_mps":[float(v.min()),float(v.max())],
        "minimum_rolloff_factor":float(state["rolloff_factor"].min()), "masks":{name:{
            "sample_fraction":float(mask.mean()), "duration_fraction_left_sample":float(np.sum(np.diff(t)*mask[:-1])/(t[-1]-t[0])),
            "time_ranges_s":intervals(t,mask)} for name,mask in masks.items()}}


def rate_events(p):
    t,r=p["t"],p["body_yaw_rate_deg_s"]
    maxima=[i for i in range(1,len(r)-1) if r[i]>r[i-1] and r[i]>=r[i+1] and r[i]>2. and .3<=t[i]<=3.5]
    minima=[i for i in range(1,len(r)-1) if r[i]<r[i-1] and r[i]<=r[i+1] and r[i]<-2. and .3<=t[i]<=3.5]
    zero=[]
    for i in range(len(r)-1):
        if r[i]*r[i+1]<0. and .3<=t[i]<=3.5:
            zero.append(float(t[i]-r[i]*(t[i+1]-t[i])/(r[i+1]-r[i])))
    return {"positive_peaks":[{"time_s":float(t[i]),"rate_deg_s":float(r[i])} for i in maxima],
        "negative_peaks":[{"time_s":float(t[i]),"rate_deg_s":float(r[i])} for i in minima],
        "zero_crossing_times_s_linear_between_original_samples":zero,
        "rule":"unfiltered adjacent-sample extrema above 2deg/s; 0.3-3.5s; source 3-point quaternion rate"}


def control_diagnostics(result, baseline):
    fields_list=("feedback_g","request_g","held_fin_rad","pid_memory")
    output={}
    for label,r in (("baseline",baseline),("finite_force",result)):
        t=np.asarray([x["time_s"] for x in r["ticks"]]); out={}
        for ts in (.5,.8333333333333334,1.,1.5,2.,2.5,3.):
            if ts>t[-1]:continue
            i=int(np.argmin(abs(t-ts))); tick=r["ticks"][i]
            out[str(ts)]={"actual_tick_time_s":tick["time_s"], **{f:tick[f] for f in fields_list}}
        out["horizontal_pid_integral_range"]=[min(x["pid_memory"][1][0] for x in r["ticks"]),max(x["pid_memory"][1][0] for x in r["ticks"])]
        out["horizontal_held_peak_abs_deg"]=math.degrees(max(abs(x["held_fin_rad"][1]) for x in r["ticks"]))
        output[label]=out
    return output


def common_endpoint(observed, baseline, candidate):
    t=observed["t"]; mask=(t>=max(baseline["t"][0],candidate["t"][0]))&(t<=min(baseline["t"][-1],candidate["t"][-1]))
    i=int(np.flatnonzero(mask)[-1]); ts=float(t[i]); out={"original_reference_time_s":ts,"excluded_reference_rows":np.flatnonzero(~mask)}
    for name,p in (("reference",observed),("baseline",baseline),("finite_force",candidate)):
        out[name]={f:float(np.interp(ts,p["t"],p[f])) for f in ("speed_mps","alpha_deg","normal_specific_load_g","body_yaw_rate_deg_s")}
        out[name]["y_m"]=float(np.interp(ts,p["t"],p["position_xyz_m"][:,1]))
    return out


def first_support_exit(result, fit, model):
    upper=fit["calibration"]["geometric_angle_support_deg"][1]
    ids=[i for i,s in enumerate(result["samples"]) if math.degrees(s["angle_of_attack_rad"])>upper]
    if not ids:return {"crossed_upper_angle_support":False}
    i=ids[0];ticks={x["time_s"]:x for x in result["ticks"]};points=[]
    for j in range(max(i-2,0),min(i+1,len(result["samples"]))):
        s=result["samples"][j];tick=ticks[s["time_s"]];state=sample_state(s)
        f=model.evaluate(state,s["time_s"],tuple(s["held_fin_rad"]))
        v=norm(state.velocity);offset=model.arm*state.yaw_rate/v;c=model.coefficients.ca
        def derivative(a):
            local=a+offset;z=max(local-ALPHA_S,0)/WIDTH;r=1/(1+z*z)
            return f.qbar*model.area*c*(r-2*local*z*r*r/WIDTH)+f.thrust*math.cos(a)
        left,right=map(math.radians,fit["calibration"]["geometric_angle_support_deg"])
        for _ in range(60):
            mid=(left+right)/2
            if derivative(mid)>0:left=mid
            else:right=mid
        angle=(left+right)/2;local=angle+offset;ratio=1/(1+(max(local-ALPHA_S,0)/WIDTH)**2)
        capacity=(f.qbar*model.area*c*local*ratio+f.thrust*math.sin(angle))/(f.mass*G0_MPS2)
        points.append({"time_s":s["time_s"],"geometric_angle_deg":math.degrees(s["angle_of_attack_rad"]),
            "speed_mps":v,"body_yaw_rate_deg_s":math.degrees(state.yaw_rate),
            "body_yaw_acceleration_deg_s2":math.degrees(s["angular_acceleration_rad_s2"][1]),
            "feedback_g":tick["feedback_g"],"request_g":tick["request_g"],
            "held_fin_deg":np.degrees(tick["held_fin_rad"]),"horizontal_PID_memory":tick["pid_memory"][1],
            "conditional_planar_maximum_normal_g_in_angle_support":capacity,
            "conditional_planar_capacity_angle_deg":math.degrees(angle)})
    return {"crossed_upper_angle_support":True,"first_saved_time_s":result["samples"][i]["time_s"],
        "prior_two_ticks_and_first_exit":points,
        "capacity_method":"one-dimensional analytic derivative root of fixed force law at saved V/m/T/rate; not a parameter scan or free simulation",
        "capacity_limitation":"planar conditional approximation to this candidate only; no native envelope or control-unit identification",
        "post_exit_limitation":"unsupported static extension; 90deg component-coordinate branch cannot identify real game moment dynamics"}


def plot_case(cid, observed, baseline, candidate):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    items=(("speed_mps","Speed (m/s)"),("alpha_deg","Incidence (deg)"),("normal_specific_load_g","Trajectory-normal specific load (G)"),
        ("body_yaw_rate_deg_s","Body yaw rate (deg/s)"),("body_yaw_deg","Body yaw (deg)"))
    for ax,(field,title) in zip(axes.flat,items):
        for name,p,col in (("Jari interim",observed,"#24566a"),("cached linear-force S110",baseline,"#777777"),("fixed finite-angle force",candidate,"#168c72")):
            ax.plot(p["t"],p[field],color=col,lw=1.3,marker="o" if name=="Jari interim" else None,ms=2,mfc="none",label=name)
        ax.set(xlabel="Original time (s)",ylabel=title,xlim=(0,3.5 if field in ("alpha_deg","body_yaw_rate_deg_s","body_yaw_deg") else observed["t"][-1]));ax.grid(alpha=.2)
    ax=axes[1,2];offset=observed["position_xyz_m"][0,1]
    for p,col in ((observed,"#24566a"),(baseline,"#777777"),(candidate,"#168c72")):ax.plot(p["t"],p["position_xyz_m"][:,1]-offset,color=col,lw=1.3)
    ax.set(xlabel="Original time (s)",ylabel="Y displacement (m)");ax.grid(alpha=.2);axes[0,0].legend(fontsize=7)
    fig.suptitle(cid+": force-only high-angle rolloff; angular law / PID / request / drag / loft fixed")
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(OUTPUT/f"{cid}_overlay.png",dpi=150);plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True); full=OUTPUT/"full_runs";full.mkdir(exist_ok=True)
    protected={p for folder in (ROOT/"outputs/jari_offaxis/reference",slew.OUTPUT,drag.OUTPUT) for p in folder.rglob("*") if p.is_file()}
    protected.update((comparison.FROZEN,comparison.SNAPSHOT,Path(slew.__file__),Path(drag.__file__),Path(fields.__file__),Path(comparison.__file__),
        ROOT/"src/aim120_model/single_surface_game.py",ROOT/"src/aim120_model/drag_models.py",harness.NATIVE_BASE/"source_payload.json"))
    hashes={str(p):comparison.sha256(p) for p in protected}
    coefficients=drag.DragCoefficients(**json.loads((drag.OUTPUT/"fit.json").read_text())["coefficients"])
    profiles=json.loads(comparison.SNAPSHOT.read_text());manifest=json.loads(comparison.DEFAULT_MANIFEST.read_text())
    cache=np.load(slew.OUTPUT/"trajectories.npz");inputs={};arrays={};runs={};records={};near={};free_count=0;recovered=0
    for entry in manifest["references"]:
        cid=entry["id"];run=comparison.split_runs(entry["export_path"])[0];ref,diagnostic=comparison.parse_reference(run)
        profile,source=comparison.profile_for(run,entry,profiles);config=copy.deepcopy(profile["_model_config"])
        inputs[cid]={"run":run,"reference":ref,"observed":fields.observed_fields(run,ref),"diagnostic":diagnostic,"config":config,"source":source,
            "export_path":entry["export_path"],"drag_coefficients":coefficients}
    fit=calibration_record(inputs);slew.write_json(OUTPUT/"fit.json",fit)
    cases={x["id"]:x for x in harness.load_cases(comparison.SNAPSHOT)}
    for cid in ("jari_259_0","jari_259_30","jari_259_60","native_H110","native_H27.5","native_H55native"):
        native=cid.startswith("native_")
        if native:
            case=cases[cid.removeprefix("native_")];config=copy.deepcopy(case["profile"]["_model_config"]);state=dataclasses.replace(case["_creation_state"])
            qx,qy,qz,qw=case["creation"]["u16_1"];state.orientation_quaternion=(qw,qx,qy,qz);target_at=case["_target_model"].state_at
            end=case["scenario"]["max_simulation_time_s"]
        else:
            x=inputs[cid];config=copy.deepcopy(x["config"]);state,target_at=comparison.free_inputs(x["run"],x["reference"],config);end=config["performance"]["lifetime_s"]
        baseline=drag.selected_arrays(cache,cid+"__selected__");cached_full=json.loads((slew.OUTPUT/"full_runs"/f"{cid}_selected.json").read_text())
        model=make_model(config,coefficients);path=full/f"{cid}.json"
        if path.exists():
            result=json.loads(path.read_text());model.initialize(dataclasses.replace(state));recovered+=1
        else:
            result=model.run(dataclasses.replace(state),target_at,end_time_s=end);free_count+=1
            slew.record_model(result,model);result["model"].update(finite_force_parameters_rad=fit["parameters_rad"],finite_force_low_angle_exact=True)
            slew.write_json(path,result,True)
        prediction=harness.sample_series(result["samples"],case) if native else fields.prediction_fields(result,model,x["run"]["scenario"])
        diag,force_check=force_diagnostics(model,result)
        # Diagnostics add fields to the candidate cache without modifying its saved clocks.
        for name,value in diag.items():prediction[name]=value
        exact={key:comparison.jsonable(result[key])==cached_full[key] for key in ("samples","ticks","summary")}
        if native:
            item={"incremental_relative_cached_S110":fields.native_pair(case,baseline,prediction),
                "overall_relative_old_accepted":fields.native_pair(case,drag.selected_arrays(cache,cid+"__baseline_old_accepted__"),prediction),
                "exact_cached_S110":exact,"summary":result["summary"],"events":drag.near_events(case["observed"],baseline,prediction)}
            near[cid.removeprefix("native_")]=item;observed=case["observed"]
        else:
            observed=x["observed"]
            windows=(("early_0_0p7",0.,.7),("turn_0p7_3p5",.7,3.5),("full_common",0.,math.inf))
            metric=drag.pair_metrics(observed,baseline,prediction,tuple(UNITS),windows)
            for ww in metric.values():
                for f,mm in ww["metrics"].items():mm["units"]=UNITS[f]
            item={"metrics_separate_units":metric,"exact_cached_S110":exact,"summary":result["summary"],
                "events":drag.event_summary(x["reference"],prediction,result,x["run"]["summary"]),
                "same_time_end":common_endpoint(observed,baseline,prediction),
                "body_rate_events":{name:rate_events(p) for name,p in (("reference",observed),("baseline",baseline),("finite_force",prediction))},
                "source_rows":len(observed["t"]),"scenario":x["run"]["scenario"]}
            records[cid]=item
        item.update(raw_PID=config["control"]["pid"],frozen_surface=dataclasses.asdict(model.coefficients),
            raw_loft_fields={k:config["guidance"][k] for k in ("lofting_elevation_deg","target_elevation_deg","angle_to_acceleration_multiplier","loft_omega_max_deg_s")},
            accepted_loft_coefficients=[model.amplitude_coefficient,model.transition_coefficient],
            control=control_diagnostics(result,cached_full),force_bookkeeping=force_check,full_result_path=str(path))
        item["first_upper_angle_support_exit"]=first_support_exit(result,fit,model)
        if native:
            state_fields={"alpha_deg":np.degrees([s["angle_of_attack_rad"] for s in result["samples"]]),
                "speed_mps":np.linalg.norm([s["velocity_mps"] for s in result["samples"]],axis=1),**diag}
        else:state_fields=prediction
        item["calibration_support"]=support_report(np.asarray([s["time_s"] for s in result["samples"]]),state_fields,fit)
        fields.save_arrays(arrays,cid+"__reference",observed);fields.save_arrays(arrays,cid+"__baseline",baseline);fields.save_arrays(arrays,cid+"__finite_force",prediction)
        runs[cid]=(result,prediction,model,config,state,target_at)
        print(json.dumps({"completed":cid,"summary":result["summary"],"local_angle_peak_deg":float(np.degrees(diag["local_angle_rad"].max())),"exact_baseline":exact}),flush=True)
    # One concentrated verification pass: no additional free integration.
    preserved=hashes=={str(p):comparison.sha256(p) for p in protected};fuse={};clocks={};fixed=[]
    for cid,(result,prediction,model,config,state,target_at) in runs.items():
        ticks=np.asarray([x["time_s"] for x in result["ticks"]]);clocks[cid]=bool(np.max(abs(ticks-np.round(ticks*48)/48))<1e-9)
        if not cid.startswith("native_"):
            distance=norm(tuple(a-b for a,b in zip(target_at(result["samples"][-1]["time_s"]).position,result["samples"][-1]["position_m"])))
            fuse[cid]={"termination":result["summary"]["termination_event"],"end_separation_m":distance,"boundary_error_m":abs(distance-10.)}
        expected=cases[cid.removeprefix("native_")]["profile"]["_model_config"] if cid.startswith("native_") else inputs[cid]["config"]
        fixed.append(config==expected and model.coefficients==SurfaceCoefficients() and model.relative_blend==.5 and model.request_slew_g_s==110. and [model.amplitude_coefficient,model.transition_coefficient]==[.75,1.3])
        # Existing request validator only reads saved states and geometry; force feedback does not enter request().
        # It therefore validates the inherited S110 clock without any baseline integration.
        if not recovered: (near[cid.removeprefix("native_")] if cid.startswith("native_") else records[cid])["request_state_verification"] = slew.request_checks(config,state,target_at,SLEW,coefficients,result,model)
    finite=all(np.isfinite(a).all() for a in arrays.values());rows=sum(len(x["reference"]["t"]) for x in inputs.values())
    ref_clock=all(np.array_equal(arrays[cid+"__reference__t"],x["reference"]["t"]) for cid,x in inputs.items())
    unchanged_near=all(all(x["exact_cached_S110"].values()) for x in near.values())
    zero_exact=all(records["jari_259_0"]["exact_cached_S110"].values())
    verification={"protected_inputs_previous_outputs_unchanged":preserved,"original_reference_rows":rows,"reference_clock_unchanged":ref_clock,
        "all_arrays_finite":finite,"all_raw_PID_loft_drag_frozen_parameters_equal":all(fixed),"clock_grid_48Hz":clocks,
        "plant_substeps":4,"first_fuse_boundary":fuse,"fuse_tolerance_m":.001,"new_integrations_this_invocation":free_count,
        "new_runs_recovered_without_integrating":recovered,"unique_free_runs_delivered":len(runs),"baseline_integrations":0,
        "native_three_exact_cached_S110":unchanged_near,"Jari_zero_exact_cached_S110":zero_exact,"observed_missile_after_creation":False,
        "single_surface_same_station_couple":True,"same_state_held_qdot_exact_unchanged":True,"moment_law_not_independently_validated":True}
    if not(preserved and ref_clock and finite and all(fixed) and all(clocks.values()) and rows==631 and len(runs)==6 and free_count+recovered==6 and zero_exact and unchanged_near):raise AssertionError(verification)
    for x in fuse.values():
        if x["termination"]=="proximity_fuse" and x["boundary_error_m"]>=.001:raise AssertionError(x)
    np.savez_compressed(OUTPUT/"trajectories.npz",**arrays)
    for cid in ("jari_259_30","jari_259_60"):
        plot_case(cid,inputs[cid]["observed"],drag.selected_arrays(cache,cid+"__selected__"),runs[cid][1])
    document={"status":"fixed_finite_angle_force_six_case_AB_complete","fit":fit,"cases":records,"native_near_axis":near,"verification":verification,
        "production_or_GUI_changed":False,"composite_score":False,"baseline_cache_path":str(slew.OUTPUT),"preserved_input_hashes":hashes,
        "limitations":["Interim Jari .28 reference; .27 rocket equality, .28 global unverified; actual game validation pending",
            "Static lift fit only on 60; 30/0 transfer references already seen, not blind",
            "Source normal G is a kinematic display-cadence derivative; source controls and omega units are unknown",
            "Frozen angular law is held at identical state/held, while control feedback and achieved held change in free flight",
            "The two-angle empirical force envelope is not a mapped native CyMaxAoA or identified mechanism outside its calibration domain",
            "No free-flight tuning, alternate family, outer change or frozen moment coefficient change"]}
    slew.write_json(OUTPUT/"results.json",document)
    print(json.dumps({"output":str(OUTPUT),"verification":verification}),flush=True)


if __name__=="__main__":main()
