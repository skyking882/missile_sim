#!/usr/bin/env python3
"""One causal high-angle control-memory experiment on the frozen game plant.

Predeclared zero-coefficient mechanism: retain the legacy low-angle PID and
S110; on first entry into the existing finite-force rolloff, transport only
integral coordinates and test a normalized target authority of one. One 30deg
conditional diagnostic precedes six free runs. No oracle input in runtime.
"""
from __future__ import annotations
import copy
import dataclasses
import json
import math
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"scripts"),str(ROOT/"src")]
sys.dont_write_bytecode=True
os.environ.setdefault("MPLCONFIGDIR","/tmp/missle_sim_mpl")
import numpy as np
import fit_r771_jari_finite_force as finite
from fit_r771_jari_finite_force import comparison,fields,drag,slew,harness
import invert_r771_jari_control as inverse
import try_r771_jari_trim_target as trim
from aim120_model.aerodynamics import normalize_quaternion,pitch_yaw_from_quaternion
from aim120_model.control import scheduled_pid_values
from aim120_model.dynamics import SimState
from aim120_model.guidance import interpolate_table
from aim120_model.math3d import norm,sub
from aim120_model.r771_game_control import R771GameControl
from aim120_model.single_surface_game import SurfaceCoefficients

OUTPUT=ROOT/"outputs/jari_offaxis/control_fix"
CASES=trim.CASES


class ControlFixGame(finite.FiniteForceGame):
    """Harness adapter: evaluate/step unchanged, runtime controller has no IO."""
    def initialize(self,state):
        result=super().initialize(state)
        self.game_control=R771GameControl();self.control_history={}
        return result
    def request(self,state,target,time_s,evaluation):
        request,details=super().request(state,target,time_s,evaluation)
        speed=norm(sub(state.velocity,self.wind));a=(evaluation.qbar/evaluation.mass)/self.q_over_mass0
        c=self.coefficients;ku=c.control_gain*(1+c.speed_slope*(speed/self.speed0-1))
        self.current_control_k=ku/(a*c.restoring_gain)
        self.current_force_nonlinear=math.hypot(*evaluation.local_incidence)>finite.ALPHA_S
        self.current_control_time=time_s;self.current_control_axis=0
        return request,details
    def _pid(self,error,memory,pid,fin_limit,gain):
        axis=self.current_control_axis;self.current_control_axis+=1
        result=self.game_control.update_axis(axis,self.current_control_time,error,memory,pid,fin_limit,gain,
            self.current_control_k,self.current_force_nonlinear)
        if axis==1:self.control_history[self.current_control_time]=copy.deepcopy(self.game_control.diagnostics)
        return result
    def run(self,initial_state,target_at,*,end_time_s):
        result=super().run(initial_state,target_at,end_time_s=end_time_s)
        limits=np.radians([self.config["aerodynamics"][k] for k in ("horizontal_fin_aoa_limit_deg","vertical_fin_aoa_limit_deg")])
        previous=(0.,0.);bytime={s["time_s"]:s for s in result["samples"]};mode=False
        for tick in result["ticks"]:
            diagnostics=self.control_history.get(tick["time_s"])
            if diagnostics:mode=diagnostics[0]["nonlinear_latched"]
            tick.update(evaluated_old_held_rad=previous,control_runtime_diagnostics=diagnostics,
                nonlinear_control_latched=mode,held_fin_rad_semantics="old effective actuator input in radians; original frozen Ku(V) interface")
            bytime[tick["time_s"]]["evaluated_old_held_rad"]=previous;previous=tuple(tick["held_fin_rad"])
        for s in result["samples"]:
            s.setdefault("evaluated_old_held_rad",previous)
            state=finite.sample_state(s);f=self.evaluate(state,s["time_s"],tuple(s["evaluated_old_held_rad"]))
            speed=norm(sub(state.velocity,self.wind));a=(f.qbar/f.mass)/self.q_over_mass0;c=self.coefficients
            k=c.control_gain*(1+c.speed_slope*(speed/self.speed0-1))/(a*c.restoring_gain)
            s.update(effective_target_rad=tuple(k*x for x in s["held_fin_rad"]),
                normalized_target_diagnostic=tuple(k*x/l for x,l in zip(s["held_fin_rad"],limits)),
                old_held_semantics="old effective actuator input, not observed fin angle")
        slew.record_model(result,self)
        result["model"].update(runtime_controller="aim120_model.r771_game_control.R771GameControl",
            controller_mechanism="causal nonlinear-regime latch; integral coordinate transport and candidate target-authority antiwindup",
            new_fitted_coefficients=0,old_PID_input_units="G error",memory_integral_units="old actuator radians; K*I has target-angle radians",
            P_D_target_terms="K*P*error_g, K*D*derivative_g_s",integral_transport="I_old <- K_previous/K_current*I_old, clipped by raw integral_limit",
            first_entry_transport="K_previous=K_current",saturation_memory="freeze transported integral",
            nonlinear_trigger="current hypot(local_incidence)>fixed finite-force alpha_s; one-way candidate control latch",
            experimental_target_cap="normalized target1 only after latch; NOT native physical old-held cap",
            original_old_held_cap="raw axis finsAoa in radians remains enforced",reference_data_in_runtime=False,
            outer="unchanged S110, half horizontal relativePN/gatedloft",raw_loft_coefficients=[.75,1.3],native_game_controller_identified=False)
        return result


def make_model(config,coefficients):
    return ControlFixGame(config,request_slew_g_s=110.,drag_coefficients=coefficients,
        coefficients=SurfaceCoefficients(),force_basis="wind_linear",plant_substeps=4,loft_cap_unit="rad_s")


def source_state_at(ref,time_s):
    def row(key):return np.array([np.interp(time_s,ref["t"],ref[key][:,j]) for j in range(ref[key].shape[1])])
    position=row("position_xyz_m");velocity=row("velocity_xyz_mps");omega=row("omega_xyz_export_units")
    qxy=row("quaternion_xyzw");q=normalize_quaternion((qxy[3],*qxy[:3]));p,y=pitch_yaw_from_quaternion(q)
    return SimState(tuple(position),tuple(velocity),p,y,-omega[2],omega[1],float(np.interp(time_s,ref["t"],ref["mass"])),orientation_quaternion=q)


def conditional_30(inputs,coefficients):
    x=inputs["jari_259_30"];cfg=x["config"];ref=x["ref"]
    models={"legacy":finite.make_model(cfg,coefficients),"control_fix":make_model(cfg,coefficients)}
    memories={k:((0.,0.,0.),(0.,0.,0.)) for k in models};series={k:[] for k in models}
    for m in models.values():m.initialize(dataclasses.replace(x["initial"]))
    limits=np.radians([cfg["aerodynamics"][k] for k in ("horizontal_fin_aoa_limit_deg","vertical_fin_aoa_limit_deg")])
    end_tick=math.ceil((3.5+1e-6)*48);t=np.arange(end_tick+1)/48.
    for ts in t:
        state=source_state_at(ref,float(ts));target=x["target_at"](float(ts))
        gain=interpolate_table(float(ts)+1/48.,cfg["guidance"]["flight_time_gain_table"])
        pid=scheduled_pid_values(cfg["control"]["pid"],float(ts)+1/48.)
        for label,m in models.items():
            f=m.evaluate(state,float(ts),(0.,0.));request,_=m.request(state,target,float(ts),f)
            if gain>1e-12:
                updated=[m._pid(r-fb,mem,pid,limit,gain) for r,fb,mem,limit in zip(request,f.feedback_g,memories[label],limits)]
                held=tuple(v[0] for v in updated);memories[label]=tuple(v[1] for v in updated)
            else:held=(0.,0.)
            speed=norm(state.velocity);a=(f.qbar/f.mass)/m.q_over_mass0;c=m.coefficients
            k=c.control_gain*(1+c.speed_slope*(speed/m.speed0-1))/(a*c.restoring_gain)
            series[label].append((held,tuple(k*u/l for u,l in zip(held,limits))))
    z=np.load(inverse.OUTPUT/"inverse_inputs.npz");source_t=z["jari_259_30__t"];mask=inverse.window_mask(source_t)
    record={"case":"jari_259_30","original_interior_rows":int(mask.sum()),"controller_hz":48,"fits":0,
        "derivative_target":"previous saved 3/5-point inverse; conditional only, not runtime input", "series":{},
        "decision_rule":"retain legacy low-regime exactly; report 30 conditional target error/cap conflicts before any free runs; no coefficient selection from60/0/native"}
    arrays={"conditional_tick_t":t}
    for label,values in series.items():
        held=np.asarray([v[0] for v in values]);u=np.asarray([v[1] for v in values]);sampled=np.column_stack([np.interp(source_t[mask],t,u[:,j]) for j in range(2)])
        record["series"][label]={"source_controls_RMS_pitch_yaw":[inverse.rms(sampled[:,j]-z["jari_259_30__source_controls_pitch_yaw"][mask,j]) for j in range(2)],
            "inverse3_RMS_pitch_yaw":[inverse.rms(sampled[:,j]-z["jari_259_30__3__u_norm_req"][mask,j]) for j in range(2)],
            "inverse5_RMS_pitch_yaw":[inverse.rms(sampled[:,j]-z["jari_259_30__5__u_norm_req"][mask,j]) for j in range(2)],
            "normalized_target_range_pitch_yaw":[[float(u[:,j].min()),float(u[:,j].max())] for j in range(2)]}
        arrays[label+"_held_old_rad"]=held;arrays[label+"_normalized_target"]=u
    history=models["control_fix"].control_history
    entry=next((ts for ts,d in history.items() if d[0]["nonlinear_latched"]),None)
    indices=t<entry if entry is not None else np.ones(len(t),bool)
    low_equal=np.array_equal(arrays["legacy_held_old_rad"][indices],arrays["control_fix_held_old_rad"][indices])
    record.update(first_source_state_regime_entry_s=entry,legacy_low_regime_equal=low_equal,
        original_PID_entire_dict=cfg["control"]["pid"],new_shared_constants=[],evidence_grade="D opt-in game controller hypothesis")
    if not low_equal:raise AssertionError("Low regime changed in conditional check")
    OUTPUT.mkdir(parents=True,exist_ok=True);np.savez_compressed(OUTPUT/"conditional_30.npz",**arrays)
    slew.write_json(OUTPUT/"conditional_30.json",record);print(json.dumps(record),flush=True)


def verify_controls(model,result):
    limits=np.radians([model.config["aerodynamics"][k] for k in ("horizontal_fin_aoa_limit_deg","vertical_fin_aoa_limit_deg")])
    maxima=np.max(np.abs([s["held_fin_rad"] for s in result["samples"]]),axis=0)
    ticks=[r for r in result["ticks"] if r["control_runtime_diagnostics"]]
    high=[r for r in ticks if r["nonlinear_control_latched"]]
    peak=max((abs(d["normalized_target_at_tick"]) for r in high for d in r["control_runtime_diagnostics"]),default=0.)
    ratios=[r["control_runtime_diagnostics"][0]["integral_transport_ratio"]-r["control_runtime_diagnostics"][1]["integral_transport_ratio"] for r in ticks]
    if np.any(maxima>limits+1e-12) or peak>1+1e-12 or max(abs(v) for v in ratios)>1e-12:raise AssertionError((maxima,peak,ratios))
    return {"old_held_max_abs_pitch_yaw_deg":np.degrees(maxima).tolist(),"original_old_held_limit_pitch_yaw_deg":np.degrees(limits).tolist(),
        "high_mode_normalized_target_peak":peak,"high_mode_ticks":len(high),
        "first_runtime_regime_entry_s":high[0]["time_s"] if high else None,
        "two_axis_transport_ratio_difference_max_abs":max(abs(v) for v in ratios),
        "target_limit_applies_at_committed_ticks_only":True,"low_mode_target_diagnostic_is_not_capped":True}


def plot_case(cid,observed,base,candidate,arrays):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(14,8));curves=(("Jari reference",observed,"#24566a"),
        ("cached finite force",base,"#bb762c"),("causal control-memory fix",candidate,"#168c72"))
    for ax,(key,label) in zip(axes.flat,(("speed_mps","Speed (m/s)"),("alpha_deg","AoA (deg)"),
        ("normal_specific_load_g","Normal specific load (G)"),("body_yaw_rate_deg_s","Body yaw rate (deg/s)"),("body_yaw_deg","Body yaw (deg)"))):
        for name,p,col in curves:ax.plot(p["t"],p[key],lw=1.4,color=col,label=name)
        ax.set(xlabel="Original time (s)",ylabel=label,xlim=(0,3.5 if key in ("alpha_deg","body_yaw_rate_deg_s","body_yaw_deg") else observed["t"][-1]));ax.grid(alpha=.2)
    for _,p,col in curves:axes[1,2].plot(p["t"],p["position_xyz_m"][:,1]-observed["position_xyz_m"][0,1],color=col,lw=1.4)
    axes[1,2].set(xlabel="Original time (s)",ylabel="Y displacement (m)");axes[1,2].grid(alpha=.2);axes[0,0].legend(fontsize=8)
    fig.suptitle(cid+": causal high-angle integral transport / target-authority test; frozen plant");fig.tight_layout(rect=(0,0,1,.96));fig.savefig(OUTPUT/(cid+"_overlay.png"),dpi=150);plt.close(fig)


def run_six(inputs,coefficients):
    if not (OUTPUT/"conditional_30.json").exists():raise ValueError("Run the single30 conditional diagnostic first")
    full=OUTPUT/"full_runs";full.mkdir(exist_ok=True)
    protected=[ROOT/"src/aim120_model/single_surface_game.py",Path(finite.__file__),Path(slew.__file__),Path(drag.__file__),Path(inverse.__file__),Path(trim.__file__),
        comparison.SNAPSHOT,comparison.FROZEN,comparison.DEFAULT_MANIFEST,finite.OUTPUT/"fit.json",finite.OUTPUT/"trajectories.npz",slew.OUTPUT/"trajectories.npz",drag.OUTPUT/"fit.json",
        trim.OUTPUT/"results.json",trim.OUTPUT/"trajectories.npz",inverse.OUTPUT/"results.json",inverse.OUTPUT/"inverse_inputs.npz"]
    protected.extend(p for folder in (ROOT/"outputs/jari_offaxis/reference",finite.OUTPUT/"full_runs",slew.OUTPUT/"full_runs",trim.OUTPUT/"full_runs",inverse.OUTPUT/"full_runs") for p in folder.iterdir() if p.is_file())
    before={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in protected}
    fcache=np.load(finite.OUTPUT/"trajectories.npz");scache=np.load(slew.OUTPUT/"trajectories.npz");tcache=np.load(trim.OUTPUT/"trajectories.npz")
    native={c["id"]:c for c in harness.load_cases(comparison.SNAPSHOT)}
    records={};near={};runs={};arrays={};new=0;recovered=0
    for cid in CASES:
        is_native=cid.startswith("native_")
        if is_native:
            case=native[cid.removeprefix("native_")];cfg=copy.deepcopy(case["profile"]["_model_config"]);initial=dataclasses.replace(case["_creation_state"])
            qx,qy,qz,qw=case["creation"]["u16_1"];initial.orientation_quaternion=(qw,qx,qy,qz)
            target_at=case["_target_model"].state_at;end=case["scenario"]["max_simulation_time_s"];observed=case["observed"]
        else:
            x=inputs[cid];cfg=copy.deepcopy(x["config"]);initial=x["initial"];target_at=x["target_at"];end=cfg["performance"]["lifetime_s"];observed=x["observed"]
        model=make_model(cfg,coefficients);path=full/(cid+".json")
        if path.exists():result=json.loads(path.read_text());model.initialize(dataclasses.replace(initial));recovered+=1
        else:result=model.run(dataclasses.replace(initial),target_at,end_time_s=end);new+=1;slew.write_json(path,result,True)
        prediction=harness.sample_series(result["samples"],case) if is_native else fields.prediction_fields(result,model,x["run"]["scenario"])
        baseline=drag.selected_arrays(fcache,cid+"__finite_force__");sbaseline=drag.selected_arrays(scache,cid+"__selected__")
        tbaseline=drag.selected_arrays(tcache,cid+"__trim_target__")
        item={"summary":result["summary"],"full_result_path":str(path),"control_output_checks":verify_controls(model,result),
            "force_torque_closure":inverse.saved_closure(model,result),"original_raw_PID":cfg["control"]["pid"],"surface_coefficients":dataclasses.asdict(model.coefficients),
            "new_shared_constants":[],"plant_and_outer_frozen":True}
        if is_native:
            item.update(incremental_vs_cached_finite_force=fields.native_pair(case,baseline,prediction),incremental_vs_cached_request_slew=fields.native_pair(case,sbaseline,prediction),
                overall_vs_old_accepted=fields.native_pair(case,drag.selected_arrays(scache,cid+"__baseline_old_accepted__"),prediction),events=drag.near_events(observed,baseline,prediction))
            near[cid.removeprefix("native_")]=item
        else:
            item.update(metrics_separate_units={name:drag.pair_metrics(observed,p,prediction,tuple(finite.UNITS),trim.WINDOWS) for name,p in
                {"finite_force":baseline,"request_slew":sbaseline,"cached_trim_target":tbaseline}.items()},
                events=drag.event_summary(x["ref"],prediction,result,x["run"]["summary"]),same_time_end=trim.endpoint(observed,{"finite_force":baseline,"request_slew":sbaseline},prediction),
                response_extrema={name:trim.response_extrema(p) for name,p in {"reference":observed,"control_fix":prediction}.items()},
                body_rate_events={name:finite.rate_events(p) for name,p in {"reference":observed,"control_fix":prediction}.items()})
            item["same_time_end"]["control_fix"]=item["same_time_end"].pop("trim_target")
            for w in item["metrics_separate_units"].values():
                for m in w.values():
                    for key,v in m["metrics"].items():v["units"]=finite.UNITS[key]
            records[cid]=item
        # Confirm the low-regime claim on actual saved states, including native.
        local=[]
        for s in result["samples"]:local.append(math.hypot(*model.evaluate(finite.sample_state(s),s["time_s"],tuple(s["evaluated_old_held_rad"])).local_incidence))
        item["local_incidence_max_deg"]=math.degrees(max(local));item["unchanged_force_rolloff_threshold_deg"]=math.degrees(finite.ALPHA_S)
        fields.save_arrays(arrays,cid+"__reference",observed);fields.save_arrays(arrays,cid+"__control_fix",prediction)
        runs[cid]=(model,result,cfg,initial,target_at,baseline,prediction)
        print(json.dumps({"completed":cid,"summary":result["summary"],"control":item["control_output_checks"],
            "native_incremental_gate":item.get("incremental_vs_cached_finite_force",{}).get("full_gate_passed")}),flush=True)
    protected_ok=before=={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in protected};checks={}
    for cid,(model,result,cfg,initial,target_at,baseline,prediction) in runs.items():
        times=np.asarray([s["time_s"] for s in result["ticks"]]);last=result["samples"][-1]
        expected=native[cid.removeprefix("native_")]["profile"]["_model_config"] if cid.startswith("native_") else inputs[cid]["config"]
        fixed=cfg==expected and model.coefficients==SurfaceCoefficients() and model.drag_coefficients==coefficients and model.relative_blend==.5 and model.request_slew_g_s==110. and [model.amplitude_coefficient,model.transition_coefficient]==[.75,1.3]
        sep=norm(sub(target_at(last["time_s"]).position,tuple(last["position_m"])));boundary=abs(sep-10.) if result["summary"]["termination_event"]=="proximity_fuse" else None
        exact=all(np.array_equal(prediction[k],baseline[k]) for k in prediction if k in baseline) if not any(t["nonlinear_control_latched"] for t in result["ticks"]) else None
        checks[cid]={"all_raw_PID_and_config_equal":fixed,"controller_clock48":bool(np.max(abs(times*48-np.round(times*48)))<1e-8),
            "fuse_boundary_error_m":boundary,"never_triggered_prediction_arrays_equal_cached_finite_force":exact,
            "plant_substeps":model.plant_substeps,"single_station_m":-model.arm}
        if not(fixed and checks[cid]["controller_clock48"] and (boundary is None or boundary<.001) and (exact is None or exact)):raise AssertionError(checks[cid])
    runtime_path=ROOT/"src/aim120_model/r771_game_control.py";runtime_text=runtime_path.read_text()
    forbidden=any(token in runtime_text for token in ("open(","read_text(","load(","numpy","requests","reference[","inverse_inputs","__file__"))
    verification={"protected_files_size_mtime_unchanged":protected_ok,"existing_core_config_raw_docs_GUI_changes":0,"source_rows":sum(len(x["ref"]["t"]) for x in inputs.values()),
        "new_integrations_this_invocation":new,"recovered_without_integrating":recovered,"unique_free_runs":len(runs),"baseline_integrations":0,"Jari_API_calls":0,"fits_or_scans":0,
        "new_shared_constants":[],"runtime_has_no_data_or_IO_dependencies":not forbidden,"runtime_source":str(runtime_path),"all_arrays_finite":all(np.isfinite(v).all() for v in arrays.values()),
        "reference_clock_unchanged":all(np.array_equal(arrays[cid+"__reference__t"],x["ref"]["t"]) for cid,x in inputs.items()),"source_state_injected_after_creation":False,
        "checks":checks,"composite_score":False}
    if not(protected_ok and not forbidden and verification["source_rows"]==631 and verification["all_arrays_finite"] and verification["reference_clock_unchanged"] and new+recovered==6):raise AssertionError(verification)
    np.savez_compressed(OUTPUT/"trajectories.npz",**arrays)
    for cid in ("jari_259_30","jari_259_60"):plot_case(cid,inputs[cid]["observed"],runs[cid][5],runs[cid][6],arrays)
    slew.write_json(OUTPUT/"results.json",{"status":"causal_controller_six_case_test_complete","conditional_30":json.loads((OUTPUT/"conditional_30.json").read_text()),
        "cases":records,"native_near_axis":near,"verification":verification,"hypothesis":"zero-new-coefficient high-angle integral-coordinate transport with explicit normalized-target1 authority",
        "production_GUI_defaults_changed":False,"limitations":["Source60 inverse asks normalized target about1.10; tested authority1 leaves a known possible upper-bound gap",
            "Nonlinear force trigger and one-way control regime are opt-in hypotheses, not native-game identification",
            "Fixed plant force-vector bias and oldaccepted overall gate may persist independently of this control-only scope",
            "No oracle control/input is used in free runtime;30 condition is not independent prediction; true-game acceptance pending"]})
    print(json.dumps({"output":str(OUTPUT),"verification":verification}),flush=True)


def main():
    inputs=inverse.source_inputs();coefficients=drag.DragCoefficients(**json.loads((drag.OUTPUT/"fit.json").read_text())["coefficients"])
    if len(sys.argv)!=2 or sys.argv[1] not in ("--conditional30","--run-six"):raise SystemExit("Use --conditional30, then --run-six")
    if sys.argv[1]=="--conditional30":conditional_30(inputs,coefficients)
    else:run_six(inputs,coefficients)


if __name__=="__main__":main()
