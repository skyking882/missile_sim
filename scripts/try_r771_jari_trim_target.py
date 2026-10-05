#!/usr/bin/env python3
"""One fixed War Thunder/Jari trim-target interface experiment; no fitting."""
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
sys.dont_write_bytecode = True
os.environ.setdefault("MPLCONFIGDIR", "/tmp/missle_sim_mpl")
import numpy as np

import fit_r771_jari_finite_force as finite
from fit_r771_jari_finite_force import comparison, harness, fields, drag, slew
import analyze_r771_jari_capture_gap as gap
from aim120_model.aerodynamics import body_axes_for_state, pitch_yaw_from_quaternion
from aim120_model.dynamics import SimState
from aim120_model.math3d import dot, norm, sub
from aim120_model.single_surface_game import SurfaceCoefficients
from aim120_model.units import G0_MPS2

OUTPUT = ROOT / "outputs/jari_offaxis/trim_target"
CASES = ("jari_259_0", "jari_259_30", "jari_259_60", "native_H110", "native_H27.5", "native_H55native")
WINDOWS = (("early_0_0p7", 0., .7), ("turn_0p7_3p5", .7, 3.5), ("full_common", 0., math.inf))
SEMANTICS = "held_fin_rad is held target local incidence, not actual fin angle; axis order pitch,yaw"


class TrimTargetGame(finite.FiniteForceGame):
    """Reuse the plant and raw PID algorithm, reinterpret their interface only."""
    def _pid(self, error, memory, pid, fin_limit, gain):
        normalized, updated = super()._pid(error*G0_MPS2, memory, pid, 1., gain)
        return fin_limit*normalized, updated

    def evaluate(self, state, time_s, held):
        f = super().evaluate(state, time_s, held)
        speed = norm(sub(state.velocity, self.wind))
        active = (f.qbar/f.mass)/self.q_over_mass0
        damping = active*self.speed0/speed
        c = self.coefficients
        qdot = tuple(active*c.restoring_gain*(target-local)-damping*c.rate_damping*rate
                     for target, local, rate in zip(held, f.local_incidence, (state.pitch_rate, state.yaw_rate)))
        couple = tuple(f.inertia*q-l for q,l in zip(qdot,f.lever_moment))
        return dataclasses.replace(f, angular_acceleration=qdot, reference_couple=couple)

    def run(self, initial_state, target_at, *, end_time_s):
        result = super().run(initial_state, target_at, end_time_s=end_time_s)
        limits = tuple(math.radians(self.config["aerodynamics"][key]) for key in
            ("horizontal_fin_aoa_limit_deg", "vertical_fin_aoa_limit_deg"))
        by_time = {s["time_s"]: s for s in result["samples"]}
        previous = (0., 0.)
        for tick in result["ticks"]:
            tick.update(held_fin_rad_semantics=SEMANTICS, target_alpha_rad=tick["held_fin_rad"],
                evaluated_target_alpha_rad=previous, normalized_control=tuple(x/l for x,l in zip(tick["held_fin_rad"],limits)),
                PID_input_error_mps2=tuple((r-f)*G0_MPS2 for r,f in zip(tick["request_g"],tick["feedback_g"])),
                PID_memory_error_and_derivative_units="m/s2, m/s3; integral is normalized output contribution")
            by_time[tick["time_s"]]["evaluated_target_alpha_rad"] = previous
            previous = tuple(tick["held_fin_rad"])
        for s in result["samples"]:
            s.update(held_fin_rad_semantics=SEMANTICS, target_alpha_rad=s["held_fin_rad"],
                normalized_control=tuple(x/l for x,l in zip(s["held_fin_rad"],limits)))
            s.setdefault("evaluated_target_alpha_rad", previous)
        slew.record_model(result,self)
        result["model"].update(held_fin_rad_semantics=SEMANTICS,
            PID_error_input_units="m/s2", PID_output="normalized [-1,1] before unchanged flight gain",
            angular_law="A*Ka*(target_alpha-local_incidence)-A*(V0/V)*Kw*rate",
            force_rolloff_parameters_rad={"alpha_s":finite.ALPHA_S,"width":finite.WIDTH},
            control_gain_and_speed_slope="original values recorded but eliminated by target-angle reparameterization",
            native_interface_identified=False, coefficients=dataclasses.asdict(self.coefficients))
        return result


def make_model(config, coefficients):
    return TrimTargetGame(config,request_slew_g_s=110.,drag_coefficients=coefficients,
        coefficients=SurfaceCoefficients(),force_basis="wind_linear",plant_substeps=4,loft_cap_unit="rad_s")


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x)**2)))


def same_state_check(inputs, coefficients):
    records = {}; arrays = {}
    for cid in ("jari_259_30","jari_259_60"):
        x=inputs[cid];ref=x["reference"];t=ref["t"];cfg=x["config"]
        initial,_=comparison.free_inputs(x["run"],ref,cfg);model=make_model(cfg,coefficients);model.initialize(initial)
        omega=ref["omega_xyz_export_units"];rates=np.column_stack((-omega[:,2],omega[:,1]))
        controls=np.asarray([s["controls"] for s in x["run"]["samples"]])[:,[1,0]]
        limits=np.radians([cfg["aerodynamics"][k] for k in ("horizontal_fin_aoa_limit_deg","vertical_fin_aoa_limit_deg")])
        targets=controls*limits;pred=[];local=[];active=[];damping=[]
        for i,ts in enumerate(t):
            qxy=ref["quaternion_xyzw"][i];q=(qxy[3],*qxy[:3]);p,y=pitch_yaw_from_quaternion(q)
            state=SimState(tuple(ref["position_xyz_m"][i]),tuple(ref["velocity_xyz_mps"][i]),p,y,*rates[i],float(ref["mass"][i]),orientation_quaternion=q)
            f=model.evaluate(state,float(ts),tuple(targets[i]));a=(f.qbar/f.mass)/model.q_over_mass0
            pred.append(f.angular_acceleration);local.append(f.local_incidence);active.append(a);damping.append(a*model.speed0/ref["speed_mps"][i])
        pred=np.asarray(pred);local=np.asarray(local);active=np.asarray(active);damping=np.asarray(damping)
        # The stored original clock has a 1.04e-7 s float32 boundary offset.
        # Include its original 3.5 s row without altering any time values.
        mask=(t>=.5-1e-6)&(t<=3.5+1e-6);windows={}
        if mask.sum()!=73:raise AssertionError((cid,"expected 73 original source rows",mask.sum()))
        for width in (3,5):
            observed=gap.derivative(rates,t,width)
            inferred=local+(observed+damping[:,None]*model.coefficients.rate_damping*rates)/(active[:,None]*model.coefficients.restoring_gain)
            windows[str(width)+"_point"]={axis:{"angular_acceleration_RMS_residual_rad_s2":rms((pred-observed)[mask,j]),
                "angular_acceleration_RMS_residual_deg_s2":math.degrees(rms((pred-observed)[mask,j])),
                "inverse_target_angle_RMS_residual_deg":math.degrees(rms((inferred-targets)[mask,j])),
                "inverse_target_angle_range_deg":np.degrees([inferred[mask,j].min(),inferred[mask,j].max()]).tolist()}
                for j,axis in enumerate(("pitch","yaw"))}
            arrays[cid+f"__observed_qdot_{width}_rad_s2"]=observed;arrays[cid+f"__inverse_target_{width}_rad"]=inferred
        records[cid]={"window_s":[.5,3.5],"original_rows":int(mask.sum()),"derivatives":windows,
            "window_boundary_inclusion_tolerance_s":1e-6,"actual_first_last_original_time_s":[float(t[mask][0]),float(t[mask][-1])],
            "source_omega_mapping":"model pitch=-omega.z, yaw=omega.y; rad/s; no scaling or clock shift",
            "source_control_mapping":"model pitch=controls[1], yaw=controls[0]; normalized-target hypothesis",
            "source_data_used_only_for_conditional_check":True,"fits":0}
        arrays.update({cid+"__t":t,cid+"__target_alpha_rad":targets,cid+"__predicted_qdot_rad_s2":pred})
    return records,arrays


def saved_diagnostics(model,result):
    diag={k:[] for k in ("local_angle_rad","rolloff_factor","lift_n","drag_n")}
    errors={k:0. for k in ("force_vector_n","specific_vector_n","same_station_moment_nm","force_change_vs_parent_n","recorded_qdot_rad_s2","recorded_couple_nm")}
    low=0;max_norm=0.;max_target=0.
    for s in result["samples"]:
        state=finite.sample_state(s);held=tuple(s["evaluated_target_alpha_rad"]);f=model.evaluate(state,s["time_s"],held)
        parent=finite.FiniteForceGame.evaluate(model,state,s["time_s"],held);axes=body_axes_for_state(state)
        local=math.hypot(*f.local_incidence);ratio=1/(1+(max(local-finite.ALPHA_S,0)/finite.WIDTH)**2)
        values={"force_vector_n":norm(tuple(a-b-c for a,b,c in zip(f.aerodynamic_force,f.wind_lift_force,f.wind_drag_force))),
            "specific_vector_n":norm(tuple(f.mass*a-b-f.thrust*c for a,b,c in zip(f.specific_acceleration,f.aerodynamic_force,axes.forward))),
            "same_station_moment_nm":max(abs(f.inertia*a-b-c) for a,b,c in zip(f.angular_acceleration,f.lever_moment,f.reference_couple)),
            "force_change_vs_parent_n":norm(tuple(a-b for a,b in zip(f.aerodynamic_force,parent.aerodynamic_force))),
            "recorded_qdot_rad_s2":max(abs(a-b) for a,b in zip(f.angular_acceleration,s["angular_acceleration_rad_s2"])),
            "recorded_couple_nm":max(abs(a-b) for a,b in zip(f.reference_couple,s["reference_couple_nm"]))}
        for k,v in values.items():errors[k]=max(errors[k],v)
        for k,v in zip(diag,(local,ratio,norm(f.wind_lift_force),f.drag_n)):diag[k].append(v)
        low+=int(local<=finite.ALPHA_S);max_norm=max(max_norm,max(abs(a) for a in s["normalized_control"]));max_target=max(max_target,max(abs(a) for a in s["target_alpha_rad"]))
    if max(errors.values())>1e-8 or max_norm>1+1e-12:raise AssertionError({"closure":errors,"normalized_peak":max_norm})
    return {k:np.asarray(v) for k,v in diag.items()},{"maximum_absolute_residuals":errors,"normalized_control_max_abs":max_norm,
        "target_alpha_max_abs_deg":math.degrees(max_target),"unchanged_low_angle_force_samples":low,
        "low_angle_force_equality_does_not_imply_trajectory_equality":True,"surface_station_m":-model.arm}


def endpoint(observed,baselines,candidate):
    t=observed["t"];last=min([candidate["t"][-1],*[p["t"][-1] for p in baselines.values()]])
    i=int(np.flatnonzero((t>=0)&(t<=last))[-1]);ts=float(t[i]);out={"original_reference_time_s":ts,"excluded_reference_rows":len(t)-i-1}
    for label,p in {"reference":observed,**baselines,"trim_target":candidate}.items():
        out[label]={f:float(np.interp(ts,p["t"],p[f])) for f in ("speed_mps","alpha_deg","normal_specific_load_g","body_yaw_rate_deg_s")}
        out[label]["y_m"]=float(np.interp(ts,p["t"],p["position_xyz_m"][:,1]))
    return out


def response_extrema(prediction):
    t=prediction["t"];out={}
    for key in ("alpha_deg","normal_specific_load_g"):
        y=prediction[key]
        ids=[i for i in range(1,len(y)-1) if .3<=t[i]<=3.5 and y[i]>y[i-1] and y[i]>=y[i+1]]
        out[key]=[{"time_s":float(t[i]),"value":float(y[i])} for i in ids]
    return {"positive_local_maxima":out,"rule":"unfiltered adjacent-sample local maxima; original clock; 0.3-3.5 s"}


def plot_case(cid,observed,baselines,candidate):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(14,8));allp=[("Jari interim",observed,"#24566a"),("cached linear force S110",baselines["request_slew"],"#999"),("cached finite force",baselines["finite_force"],"#bb762c"),("trim-target interface",candidate,"#168c72")]
    for ax,(key,label) in zip(axes.flat,(("speed_mps","Speed (m/s)"),("alpha_deg","AoA (deg)"),("normal_specific_load_g","Normal specific load (G)"),("body_yaw_rate_deg_s","Body yaw rate (deg/s)"),("body_yaw_deg","Body yaw (deg)"))):
        for name,p,col in allp:ax.plot(p["t"],p[key],lw=1.3,color=col,label=name)
        ax.set(xlabel="Original time (s)",ylabel=label,xlim=(0,3.5 if key in ("alpha_deg","body_yaw_rate_deg_s","body_yaw_deg") else observed["t"][-1]));ax.grid(alpha=.2)
    for _,p,col in allp:axes[1,2].plot(p["t"],p["position_xyz_m"][:,1]-observed["position_xyz_m"][0,1],color=col,lw=1.3)
    axes[1,2].set(xlabel="Original time (s)",ylabel="Y displacement (m)");axes[1,2].grid(alpha=.2);axes[0,0].legend(fontsize=7)
    fig.suptitle(cid+": fixed SI-error / normalized-PID / trim-target interface; outer and force unchanged");fig.tight_layout(rect=(0,0,1,.96));fig.savefig(OUTPUT/(cid+"_overlay.png"),dpi=150);plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True);full=OUTPUT/"full_runs";full.mkdir(exist_ok=True)
    protected=[ROOT/"src/aim120_model/single_surface_game.py",Path(finite.__file__),Path(slew.__file__),Path(drag.__file__),comparison.SNAPSHOT,comparison.FROZEN,comparison.DEFAULT_MANIFEST,
        finite.OUTPUT/"fit.json",finite.OUTPUT/"trajectories.npz",slew.OUTPUT/"trajectories.npz",drag.OUTPUT/"fit.json"]
    protected.extend(p for folder in (ROOT/"outputs/jari_offaxis/reference",finite.OUTPUT/"full_runs",slew.OUTPUT/"full_runs") for p in folder.iterdir() if p.is_file())
    before={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in protected}
    coefficients=drag.DragCoefficients(**json.loads((drag.OUTPUT/"fit.json").read_text())["coefficients"])
    profiles=json.loads(comparison.SNAPSHOT.read_text());manifest=json.loads(comparison.DEFAULT_MANIFEST.read_text());inputs={}
    for e in manifest["references"]:
        run=comparison.split_runs(e["export_path"])[0];ref,diagnostic=comparison.parse_reference(run);profile,source=comparison.profile_for(run,e,profiles)
        inputs[e["id"]]={"run":run,"reference":ref,"observed":fields.observed_fields(run,ref),"config":copy.deepcopy(profile["_model_config"])}
    native_cases={c["id"]:c for c in harness.load_cases(comparison.SNAPSHOT)}
    allcfg=[x["config"] for x in inputs.values()]+[native_cases[c.removeprefix("native_")]["profile"]["_model_config"] for c in CASES if c.startswith("native_")]
    gain_range=[min(row[1] for cfg in allcfg for row in cfg["guidance"]["flight_time_gain_table"]),max(row[1] for cfg in allcfg for row in cfg["guidance"]["flight_time_gain_table"])]
    if gain_range[0]<0 or gain_range[1]>1:raise ValueError("Flight gain invalidates normalized target cap; no extra cap added: "+str(gain_range))
    conditional,arrays=same_state_check(inputs,coefficients);slew.write_json(OUTPUT/"same_state.json",conditional)
    print(json.dumps({"same_state":conditional,"flight_gain_range":gain_range}),flush=True)
    fit=json.loads((finite.OUTPUT/"fit.json").read_text());fcache=np.load(finite.OUTPUT/"trajectories.npz");scache=np.load(slew.OUTPUT/"trajectories.npz")
    saved_review=json.loads((OUTPUT/"results.json").read_text()) if (OUTPUT/"results.json").exists() else None
    records={};near={};runs={};new_count=0;recovered=0
    for cid in CASES:
        native=cid.startswith("native_")
        if native:
            case=native_cases[cid.removeprefix("native_")];cfg=copy.deepcopy(case["profile"]["_model_config"]);initial=dataclasses.replace(case["_creation_state"])
            qx,qy,qz,qw=case["creation"]["u16_1"];initial.orientation_quaternion=(qw,qx,qy,qz);target_at=case["_target_model"].state_at;end=case["scenario"]["max_simulation_time_s"];observed=case["observed"]
        else:
            x=inputs[cid];cfg=copy.deepcopy(x["config"]);initial,target_at=comparison.free_inputs(x["run"],x["reference"],cfg);end=cfg["performance"]["lifetime_s"];observed=x["observed"]
        baselines={"request_slew":drag.selected_arrays(scache,cid+"__selected__"),"finite_force":drag.selected_arrays(fcache,cid+"__finite_force__")}
        model=make_model(cfg,coefficients);path=full/(cid+".json")
        if path.exists():result=json.loads(path.read_text());model.initialize(dataclasses.replace(initial));recovered+=1
        else:
            result=model.run(dataclasses.replace(initial),target_at,end_time_s=end);new_count+=1;slew.write_json(path,result,True)
        predicted=harness.sample_series(result["samples"],case) if native else fields.prediction_fields(result,model,x["run"]["scenario"])
        diag,closure=saved_diagnostics(model,result);predicted.update(diag)
        item={"summary":result["summary"],"force_and_moment_bookkeeping":closure,"raw_PID":cfg["control"]["pid"],
            "frozen_surface":dataclasses.asdict(model.coefficients),"full_result_path":str(path),
            "raw_finsAoa_limit_pitch_yaw_deg":[cfg["aerodynamics"][k] for k in ("horizontal_fin_aoa_limit_deg","vertical_fin_aoa_limit_deg")],
            "target_interface_hypothesis":SEMANTICS,"accepted_loft_coefficients":[model.amplitude_coefficient,model.transition_coefficient]}
        if native:
            item.update(incremental_vs_cached_finite_force=fields.native_pair(case,baselines["finite_force"],predicted),
                incremental_vs_cached_request_slew=fields.native_pair(case,baselines["request_slew"],predicted),
                overall_vs_old_accepted=fields.native_pair(case,drag.selected_arrays(scache,cid+"__baseline_old_accepted__"),predicted),
                events=drag.near_events(observed,baselines["finite_force"],predicted));near[cid.removeprefix("native_")]=item
            state_fields={"alpha_deg":np.degrees([s["angle_of_attack_rad"] for s in result["samples"]]),"speed_mps":np.linalg.norm([s["velocity_mps"] for s in result["samples"]],axis=1),**diag}
        else:
            item.update(metrics_separate_units={name:drag.pair_metrics(observed,p,predicted,tuple(finite.UNITS),WINDOWS) for name,p in baselines.items()},
                events=drag.event_summary(x["reference"],predicted,result,x["run"]["summary"]),same_time_end=endpoint(observed,baselines,predicted),
                response_extrema={name:response_extrema(p) for name,p in {"reference":observed,**baselines,"trim_target":predicted}.items()},
                body_rate_events={name:finite.rate_events(p) for name,p in {"reference":observed,**baselines,"trim_target":predicted}.items()},source_rows=len(observed["t"]))
            for wins in item["metrics_separate_units"].values():
                for w in wins.values():
                    for name,m in w["metrics"].items():m["units"]=finite.UNITS[name]
            records[cid]=item;state_fields=predicted
        item["calibration_support"]=finite.support_report(np.asarray([s["time_s"] for s in result["samples"]]),state_fields,fit)
        upper=fit["calibration"]["geometric_angle_support_deg"][1];ids=[i for i,s in enumerate(result["samples"]) if math.degrees(s["angle_of_attack_rad"])>upper]
        first=ids[0] if ids else None
        item["first_upper_angle_support_exit"]={"first_time_s":result["samples"][first]["time_s"] if first is not None else None,
            "prior_two_ticks_and_first_exit":[] if first is None else [{**{k:result["samples"][i][k] for k in ("time_s","angle_of_attack_rad","pitch_rate_rad_s","yaw_rate_rad_s","target_alpha_rad","normalized_control")},
                "tick":next((tick for tick in result["ticks"] if tick["time_s"]==result["samples"][i]["time_s"]),None)} for i in range(max(0,first-2),first+1)]}
        fields.save_arrays(arrays,cid+"__reference",observed)
        for name,p in baselines.items():fields.save_arrays(arrays,cid+"__"+name,p)
        fields.save_arrays(arrays,cid+"__trim_target",predicted);runs[cid]=(result,predicted,model,cfg,initial,target_at)
        print(json.dumps({"completed":cid,"summary":result["summary"],"normalized_max":closure["normalized_control_max_abs"],"target_peak_deg":closure["target_alpha_max_abs_deg"],"first_support_exit_s":item["first_upper_angle_support_exit"]["first_time_s"]}),flush=True)
    # One final saved-state verification pass; no additional forward runs.
    protected_ok=before=={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in protected};clocks={};fuses={};fixed=[]
    for cid,(result,predicted,model,cfg,initial,target_at) in runs.items():
        times=np.array([r["time_s"] for r in result["ticks"]]);clocks[cid]=bool(np.max(abs(times*48-np.round(times*48)))<1e-8)
        expected=native_cases[cid.removeprefix("native_")]["profile"]["_model_config"] if cid.startswith("native_") else inputs[cid]["config"]
        fixed.append(cfg==expected and model.coefficients==SurfaceCoefficients() and model.relative_blend==.5 and model.request_slew_g_s==110 and [model.amplitude_coefficient,model.transition_coefficient]==[.75,1.3] and model.drag_coefficients==coefficients)
        if not cid.startswith("native_"):
            last=result["samples"][-1];separation=norm(sub(target_at(last["time_s"]).position,tuple(last["position_m"])))
            fuses[cid]={"termination":result["summary"]["termination_event"],"separation_m":separation,"fuse_boundary_error_m":abs(separation-10)}
            if fuses[cid]["termination"]=="proximity_fuse" and abs(separation-10)>=.001:raise AssertionError(fuses[cid])
        if not recovered:
            (near[cid.removeprefix("native_")] if cid.startswith("native_") else records[cid])["request_state_verification"]=slew.request_checks(cfg,initial,target_at,110.,coefficients,result,model)
        elif saved_review:
            current=near[cid.removeprefix("native_")] if cid.startswith("native_") else records[cid]
            previous=saved_review["native_near_axis"][cid.removeprefix("native_")] if cid.startswith("native_") else saved_review["cases"][cid]
            if "request_state_verification" in previous:current["request_state_verification"]=previous["request_state_verification"]
    finite_arrays=all(np.isfinite(a).all() for a in arrays.values());rows=sum(len(x["reference"]["t"]) for x in inputs.values())
    source_clocks=all(np.array_equal(arrays[cid+"__reference__t"],x["reference"]["t"]) for cid,x in inputs.items())
    verification={"protected_files_size_and_mtime_unchanged":protected_ok,"core_config_raw_GUI_changes":0,"all_frozen_inputs_equal":all(fixed),
        "original_reference_rows":rows,"original_reference_clock_unchanged":source_clocks,"all_arrays_finite":finite_arrays,"controller_clock_48Hz":clocks,
        "plant_substeps":4,"flight_gain_range":gain_range,"normalized_target_cap_verified":True,"single_station_force_and_couple":True,
        "creation_state_only":True,"observed_missile_after_creation":False,"first_fuse_boundary":fuses,"fuse_tolerance_m":.001,
        "new_integrations_this_invocation":new_count,"recovered_without_integrating":recovered,"unique_free_runs":len(runs),"baseline_integrations":0,"Jari_API_calls":0,"fits_or_parameter_scans":0}
    if not(protected_ok and all(fixed) and source_clocks and finite_arrays and all(clocks.values()) and rows==631 and len(runs)==6 and new_count+recovered==6):raise AssertionError(verification)
    np.savez_compressed(OUTPUT/"trajectories.npz",**arrays)
    for cid in ("jari_259_30","jari_259_60"):
        plot_case(cid,inputs[cid]["observed"],{"request_slew":drag.selected_arrays(scache,cid+"__selected__"),"finite_force":drag.selected_arrays(fcache,cid+"__finite_force__")},runs[cid][1])
    slew.write_json(OUTPUT/"results.json",{"status":"fixed_trim_target_interface_six_case_test_complete","same_state":conditional,"cases":records,
        "native_near_axis":near,"verification":verification,"composite_score":False,"production_or_GUI_changed":False,
        "hypothesis":{"new_fitted_coefficients":0,"PID_input":"G error times G0, in m/s2","PID_algorithm_and_raw_values":"unchanged; normalized limit1; unchanged antiwindup and end-interval flight gain",
            "angular_law":"A*Ka*(raw_finsAoa*normalized_PID_output-alpha_local)-B*Kw*omega","retained_but_eliminated_coefficients":["control_gain","speed_slope"],"held_fin_rad_semantics":SEMANTICS},
        "limitations":["Fixed outer S110 and half-PN may compensate the old interface; no outer correction performed",
            "Conditional source-control agreement does not identify free PID request or native implementation",
            "Same high-angle force rolloff has limited calibration support; low-force branch equality is not trajectory equality",
            "Jari game reference only; actual game acceptance pending"]})
    print(json.dumps({"output":str(OUTPUT),"verification":verification}),flush=True)


if __name__=="__main__":main()
