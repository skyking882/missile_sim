#!/usr/bin/env python3
"""Two fixed high-angle moment extensions for a War Thunder game plant.

M_all scales the complete original angular law by the existing lift rolloff.
M_restore scales only restoring incidence and uses its corresponding K/R
normalization in the unchanged controller. Neither is native-game recovery.
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
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/"scripts"),str(ROOT/"src")]
os.environ.setdefault("MPLCONFIGDIR","/tmp/missle_sim_mpl")
import numpy as np
import check_r771_lift_direction as geometry
import check_r771_envelope_control as previous
from aim120_model.control import scheduled_pid_values
from aim120_model.math3d import norm,sub
from aim120_model.r771_game_control import R771GameControl
from aim120_model.single_surface_game import SurfaceCoefficients

dec=geometry.dec;fix=dec.audit.fix;finite=fix.finite
fields,drag,slew,harness=finite.fields,dec.audit.drag,dec.audit.slew,finite.harness
OUT=ROOT/"outputs/jari_offaxis/nonlinear_moment"
KINDS=("M_all","M_restore")
CASES=geometry.CASES
NATIVE=("native_H110","native_H27.5","native_H55native")
WINDOWS=previous.WINDOWS
UNITS={**finite.UNITS,"body_pitch_deg":"deg","body_pitch_rate_deg_s":"deg/s"}


def rolloff(local):
    s=math.hypot(*local)
    return 1. if s<=finite.ALPHA_S else 1./(1.+((s-finite.ALPHA_S)/finite.WIDTH)**2)


class NonlinearMomentGame(geometry.LiftDirectionGame):
    def __init__(self,config,*,moment_kind,**kwargs):
        if moment_kind not in KINDS:raise ValueError(moment_kind)
        self.moment_kind=moment_kind
        super().__init__(config,**kwargs)

    def initialize(self,state):
        result=super().initialize(state)
        self.failure_data=None;self.last_safe_evaluation=None
        return result

    def evaluate(self,state,time_s,held):
        try:original=super().evaluate(state,time_s,held)
        except ValueError as error:
            # Diagnose the rejected trial state; never substitute this force
            # evaluation into integration or rescue the geometric branch.
            raw=fix.ControlFixGame.evaluate(self,state,time_s,held)
            if any(abs(v)>=math.pi/2 for v in raw.local_incidence):
                self.failure_data={"type":"geometry_forward_branch_left","time_s":time_s,
                    "local_incidence_rad":raw.local_incidence,"state_repr":repr(state),"held_old_rad":held,
                    "last_safe_evaluation":self.last_safe_evaluation,"error":str(error)}
            raise
        r=rolloff(original.local_incidence)
        if r==1.:changed=original
        else:
            if self.moment_kind=="M_all":qdot=tuple(r*q for q in original.angular_acceleration)
            else:
                a=(original.qbar/original.mass)/self.q_over_mass0
                qdot=tuple(q+(1.-r)*a*self.coefficients.restoring_gain*local for q,local in zip(original.angular_acceleration,original.local_incidence))
            couple=tuple(original.inertia*q-lever for q,lever in zip(qdot,original.lever_moment))
            changed=dataclasses.replace(original,angular_acceleration=qdot,reference_couple=couple)
        if not all(math.isfinite(v) for v in (*changed.local_incidence,*changed.angular_acceleration,*changed.acceleration)):
            self.failure_data={"type":"nonfinite_plant","time_s":time_s,"state_repr":repr(state),"last_safe_evaluation":self.last_safe_evaluation}
            raise ValueError("Nonfinite candidate plant")
        self.last_safe_evaluation={"time_s":time_s,"local_incidence_rad":changed.local_incidence,"rolloff":r,
            "angular_acceleration_rad_s2":changed.angular_acceleration,"held_old_rad":held}
        return changed

    def request(self,state,target,time_s,evaluation):
        request,details=super().request(state,target,time_s,evaluation)
        old_k=self.current_control_k;r=rolloff(evaluation.local_incidence)
        if self.moment_kind=="M_restore" and r!=1.:self.current_control_k=old_k/r
        details.update(moment_kind=self.moment_kind,moment_rolloff=r,moment_local_norm_rad=math.hypot(*evaluation.local_incidence),
            moment_original_K=old_k,moment_controller_K=self.current_control_k)
        return request,details

    def run(self,initial_state,target_at,*,end_time_s):
        result=super().run(initial_state,target_at,end_time_s=end_time_s)
        result["model"].update(moment_kind=self.moment_kind,moment_extension_evidence_grade="D game hypothesis",
            moment_formula="R*original qdot" if self.moment_kind=="M_all" else "original qdot+(1-R)*A*Ka*local",
            control_normalization="originalK" if self.moment_kind=="M_all" else "originalK/R",
            reference_data_in_runtime=False,new_shared_coefficients=[],native_game_formula_recovered=False)
        return result


def make_model(cfg,coefficients,kind):
    return NonlinearMomentGame(cfg,moment_kind=kind,request_slew_g_s=110.,drag_coefficients=coefficients,
        coefficients=SurfaceCoefficients(),force_basis="wind_linear",plant_substeps=4,loft_cap_unit="rad_s")


def snapshot():
    other_script=ROOT/"scripts/check_r771_force_allocation.py"
    other_output=ROOT/"outputs/jari_offaxis/force_allocation"
    return {str(p):(p.stat().st_size,p.stat().st_mtime_ns) for folder in ("src","config","configs","schemas","docs","scripts","data","outputs/jari_offaxis")
        for p in (ROOT/folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts and ".git" not in p.parts
        and OUT not in p.parents and p!=Path(__file__).resolve() and p!=other_script and other_output not in p.parents}


def original_keys_equal(new,old):
    if isinstance(old,dict):return all(k in new and original_keys_equal(new[k],v) for k,v in old.items())
    if isinstance(old,(list,tuple)):return len(new)==len(old) and all(original_keys_equal(a,b) for a,b in zip(new,old))
    return new==old


def run_once(model,initial,target_at,end,path):
    failure_path=path.with_name(path.stem+"_failure.json")
    if failure_path.exists():return None,json.loads(failure_path.read_text()),0,1
    if path.exists():model.initialize(dataclasses.replace(initial));return json.loads(path.read_text()),None,0,1
    try:result=model.run(dataclasses.replace(initial),target_at,end_time_s=end)
    except (ValueError,OverflowError) as error:
        if model.failure_data is None:raise
        failure={"actual_free_failure":True,"moment_kind":model.moment_kind,"diagnostic":model.failure_data,
            "error":str(error),"creation_state_only":True,"fallback_used":False}
        slew.write_json(failure_path,failure);return None,failure,1,0
    if result["summary"]["termination_event"]=="numerical_failure":
        failure={"actual_free_failure":True,"moment_kind":model.moment_kind,"summary":result["summary"],"fallback_used":False}
        slew.write_json(failure_path,failure);return None,failure,1,0
    slew.write_json(path,result,True);return json.loads(path.read_text()),None,1,0


def verify(model,result,cfg,coefficients,initial,target_at,geometry_result):
    old=geometry.make_model(cfg,coefficients);old.initialize(dataclasses.replace(initial))
    maxima={"angular_law_rad_s2":0.,"translational_acceleration_mps2":0.,"aero_force_N":0.,"lever_Nm":0.,
        "K_ratio":0.,"PID_output_rad":0.,"PID_memory":0.,"request_same_state_G":0.}
    local_max=0.;norm_max=0.;r_min=1.;nonlinear_samples=0
    bytime={s["time_s"]:s for s in result["samples"]}
    for s in result["samples"]:
        state=finite.sample_state(s);held=tuple(s["evaluated_old_held_rad"])
        a=old.evaluate(state,s["time_s"],held);b=model.evaluate(state,s["time_s"],held);r=rolloff(a.local_incidence)
        local_max=max(local_max,max(abs(v) for v in a.local_incidence));norm_max=max(norm_max,math.hypot(*a.local_incidence));r_min=min(r_min,r);nonlinear_samples+=int(r<1.)
        expected=a.angular_acceleration if r==1. else (tuple(r*q for q in a.angular_acceleration) if model.moment_kind=="M_all" else tuple(q+(1-r)*(a.qbar/a.mass)/model.q_over_mass0*model.coefficients.restoring_gain*v for q,v in zip(a.angular_acceleration,a.local_incidence)))
        for key,values in (("angular_law_rad_s2",zip(expected,b.angular_acceleration)),("translational_acceleration_mps2",zip(a.acceleration,b.acceleration)),("aero_force_N",zip(a.aerodynamic_force,b.aerodynamic_force)),("lever_Nm",zip(a.lever_moment,b.lever_moment))):
            maxima[key]=max(maxima[key],max(abs(x-y) for x,y in values))
    clone=R771GameControl();memory=[(0.,0.,0.),(0.,0.,0.)]
    limits=np.radians([cfg["aerodynamics"][k] for k in ("horizontal_fin_aoa_limit_deg","vertical_fin_aoa_limit_deg")])
    for tick in result["ticks"]:
        ts=tick["time_s"];state=finite.sample_state(bytime[ts]);f=old.evaluate(state,ts,tuple(tick["evaluated_old_held_rad"]))
        raw,_=old.request(state,target_at(ts),ts,f)
        maxima["request_same_state_G"]=max(maxima["request_same_state_G"],max(abs(a-b) for a,b in zip(raw,tick["request_g"])))
        expected_k=old.current_control_k/(rolloff(f.local_incidence) if model.moment_kind=="M_restore" else 1.)
        maxima["K_ratio"]=max(maxima["K_ratio"],abs(tick["moment_controller_K"]-expected_k))
        if tick["gain"]<=1e-12:continue
        pid=scheduled_pid_values(cfg["control"]["pid"],(round(ts*48)+1)/48)
        for axis in range(2):
            held,memory[axis]=clone.update_axis(axis,ts,tick["request_g"][axis]-tick["feedback_g"][axis],memory[axis],pid,limits[axis],tick["gain"],expected_k,old.current_force_nonlinear)
            maxima["PID_output_rad"]=max(maxima["PID_output_rad"],abs(held-tick["held_fin_rad"][axis]))
            maxima["PID_memory"]=max(maxima["PID_memory"],max(abs(x-y) for x,y in zip(memory[axis],tick["pid_memory"][axis])))
    low_only=nonlinear_samples==0
    exact=all(original_keys_equal(result[k],geometry_result[k]) for k in ("samples","ticks","summary")) if low_only and geometry_result else None
    checks={"maxima":maxima,"max_local_axis_deg":math.degrees(local_max),"max_local_norm_deg":math.degrees(norm_max),"minimum_R":r_min,
        "nonlinear_saved_samples":nonlinear_samples,"low_only_geometry_original_keys_exact":exact,"support_below90deg":local_max<math.pi/2,
        "raw_config_equal":model.config==cfg,"five_coefficients_equal":model.coefficients==SurfaceCoefficients(),"drag_equal":model.drag_coefficients==coefficients,
        "PN_loft_S110_equal":model.relative_blend==.5 and model.request_slew_g_s==110. and [model.amplitude_coefficient,model.transition_coefficient]==[.75,1.3],
        "single_arm_m":model.arm,"controller_hz":model.controller_hz,"plant_substeps":model.plant_substeps,
        "clock48_max_tick_error":max(abs(t["time_s"]*48-i) for i,t in enumerate(result["ticks"])),
        "force_moment_closure":dec.audit.inverse.saved_closure(model,result),"original_control_limits":fix.verify_controls(model,result)}
    if max(maxima.values())>1e-10 or exact is False or not all(checks[k] for k in ("support_below90deg","raw_config_equal","five_coefficients_equal","drag_equal","PN_loft_S110_equal")) or checks["clock48_max_tick_error"]>1e-8:
        raise AssertionError(checks)
    return checks


def plot(cid,curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for ax,(key,unit) in zip(axes.flat,(("speed_mps","Speed (m/s)"),("alpha_deg","AoA (deg)"),("normal_specific_load_g","Normal specific load (G)"),
        ("body_pitch_deg","Body pitch (deg)"),("body_yaw_deg","Body yaw (deg)"),("body_yaw_rate_deg_s","Yaw rate (deg/s)"))):
        for name,p,color in curves:ax.plot(p["t"],p[key],color=color,lw=1.2,label=name)
        ax.set(xlabel="Original clock (s)",ylabel=unit,xlim=(0,curves[0][1]["t"][-1] if key=="speed_mps" else 3.5));ax.grid(alpha=.18)
    axes[0,0].legend(fontsize=7);fig.suptitle(cid+": existing lift R applied to fixed moment channels; original PID / no source feedback")
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(OUT/(cid+"_free.png"),dpi=150);plt.close(fig)


def run_jari(inputs,coefficients):
    reports={k:{} for k in KINDS};arrays={};new=0;recovered=0;plot_data={}
    current=np.load(fix.OUTPUT/"trajectories.npz");geometric=np.load(geometry.OUT/"free_trajectories.npz")
    full=OUT/"full_runs";full.mkdir(parents=True,exist_ok=True)
    for cid in CASES:
        x=inputs[cid];observed=previous.enriched(x["observed"],x)
        baseline={"control_fix":drag.selected_arrays(current,cid+"__control_fix__"),"geometry":drag.selected_arrays(geometric,cid+"__candidate__")}
        geometry_result=json.loads((geometry.OUT/"full_runs"/(cid+".json")).read_text())
        for name,b in baseline.items():
            receipt=json.loads(((fix.OUTPUT if name=="control_fix" else geometry.OUT)/"full_runs"/(cid+".json")).read_text())
            b["body_pitch_rate_deg_s"]=np.degrees([s["pitch_rate_rad_s"] for s in receipt["samples"]])
        curves=[("Jari reference",observed,"#243f56"),("Cached ControlFix",baseline["control_fix"],"#929699"),("Cached geometry",baseline["geometry"],"#ceac62")]
        for kind,color in zip(KINDS,("#168477","#a4508b")):
            model=make_model(x["config"],coefficients,kind);path=full/(kind+"_"+cid+".json")
            result,failure,n,r=run_once(model,x["initial"],x["target_at"],x["config"]["performance"]["lifetime_s"],path);new+=n;recovered+=r
            if failure is not None:reports[kind][cid]={"failure":failure};print(json.dumps({"failed":kind,"case":cid,"failure":failure}),flush=True);continue
            p=fields.prediction_fields(result,model,x["run"]["scenario"]);p["body_pitch_rate_deg_s"]=np.degrees([s["pitch_rate_rad_s"] for s in result["samples"]])
            metrics={name:drag.pair_metrics(observed,b,p,tuple(UNITS),WINDOWS) for name,b in baseline.items()}
            for m in metrics.values():
                for w in m.values():
                    for key,v in w["metrics"].items():v["units"]=UNITS[key]
            checks=verify(model,result,x["config"],coefficients,x["initial"],x["target_at"],geometry_result)
            reports[kind][cid]={"summary":result["summary"],"full_result_path":str(path),"metrics_separate_units":metrics,"checks":checks,
                "first_and_later_peaks":{name:fix.trim.response_extrema(v) for name,v in {"reference":observed,**baseline,"candidate":p}.items()},
                "Omega_events":{name:finite.rate_events(v) for name,v in {"reference":observed,**baseline,"candidate":p}.items()}}
            fields.save_arrays(arrays,cid+"__"+kind,p);curves.append((kind,p,color))
            print(json.dumps({"completed":kind,"case":cid,"summary":result["summary"],"minR":checks["minimum_R"],"low_exact":checks["low_only_geometry_original_keys_exact"]}),flush=True)
        for name,p in {"reference":observed,**baseline}.items():fields.save_arrays(arrays,cid+"__"+name,p)
        plot(cid,curves)
    return reports,arrays,new,recovered


def native_decisions(reports):
    decisions={};tolerance={"speed_mps":1.,"alpha_deg":.25,"normal_specific_load_g":.15,"body_pitch_deg":.3,"body_yaw_deg":.3,"body_yaw_rate_deg_s":2.,"body_pitch_rate_deg_s":1.}
    for kind,cases in reports.items():
        checks={};improvement={};failed=any("failure" in r for r in cases.values())
        for cid,r in cases.items():
            if "failure" in r:continue
            for name in ("control_fix","geometry"):
                for w in ("startup","platform","unloading","early_0_3p5"):
                    for key,v in r["metrics_separate_units"][name][w]["metrics"].items():checks[cid+"__"+name+"__"+w+"__"+key]=v["new"]["rmse"]<=1.25*v["old"]["rmse"]+tolerance[key]
            if cid in CASES[1:]:
                m=r["metrics_separate_units"]["geometry"]["early_0_3p5"]["metrics"]
                improvement[cid]={key:m[key]["new"]["rmse"]<.85*m[key]["old"]["rmse"] for key in ("speed_mps","alpha_deg","normal_specific_load_g")}
        decisions[kind]={"actual_failure":failed,"separate_no_major_regression":checks,"early_material_improvement":improvement,
            "expand_native":not failed and all(checks.values()) and all(any(v.values()) for v in improvement.values()) and len(improvement)==2,"aggregate_score":False}
    return decisions


def run_native(kind,coefficients,arrays):
    cases={c["id"]:c for c in harness.load_cases(finite.comparison.SNAPSHOT)}
    current=np.load(fix.OUTPUT/"trajectories.npz");geometric=np.load(geometry.OUT/"native_trajectories.npz")
    reports={};new=0;recovered=0
    for cid in NATIVE:
        case=cases[cid.removeprefix("native_")];cfg=copy.deepcopy(case["profile"]["_model_config"]);initial=dataclasses.replace(case["_creation_state"])
        qx,qy,qz,qw=case["creation"]["u16_1"];initial.orientation_quaternion=(qw,qx,qy,qz)
        target=case["_target_model"].state_at;end=case["scenario"]["max_simulation_time_s"]
        model=make_model(cfg,coefficients,kind);path=OUT/"full_runs"/(kind+"_"+cid+".json")
        result,failure,n,r=run_once(model,initial,target,end,path);new+=n;recovered+=r
        if failure:reports[cid]={"failure":failure};continue
        p=harness.sample_series(result["samples"],case);observed=case["observed"]
        baseline={"control_fix":drag.selected_arrays(current,cid+"__control_fix__"),"geometry":drag.selected_arrays(geometric,cid+"__candidate__")}
        geometry_result=json.loads((geometry.OUT/(cid+"_run.json")).read_text())
        reports[cid]={"summary":result["summary"],"checks":verify(model,result,cfg,coefficients,initial,target,geometry_result),
            "separate_metrics":{name:drag.pair_metrics(observed,b,p,("speed_kmh","alpha_deg","pitch_deg","omega_rad_s","gamma_deg","y_m"),(*dec.WINDOWS,*harness.WINDOWS)) for name,b in baseline.items()},
            "existing_near_axis_gate":{name:fields.native_pair(case,b,p) for name,b in baseline.items()},
            "load_observed_available":False,"full_result_path":str(path)}
        fields.save_arrays(arrays,cid+"__"+kind,p)
        print(json.dumps({"native_completed":cid,"kind":kind,"gates":{k:v["full_gate_passed"] for k,v in reports[cid]["existing_near_axis_gate"].items()}}),flush=True)
    return reports,new,recovered


def main():
    before=snapshot();OUT.mkdir(parents=True,exist_ok=True)
    coefficients=drag.DragCoefficients(**json.loads((drag.OUTPUT/"fit.json").read_text())["coefficients"])
    reports,arrays,new,recovered=run_jari(dec.audit.inverse.source_inputs(),coefficients)
    decisions=native_decisions(reports);native={}
    for kind,decision in decisions.items():
        if decision["expand_native"]:
            native[kind],n,r=run_native(kind,coefficients,arrays);new+=n;recovered+=r
    verification={"protected_prior_size_mtime_unchanged":before==snapshot(),"all_saved_arrays_finite":all(np.isfinite(v).all() for v in arrays.values()),
        "fixed_candidates":2,"unique_Jari_free_attempts":6,"new_attempts_this_invocation":new,"recovered_this_invocation":recovered,
        "native_free_attempts":sum(len(c) for c in native.values()),"baseline_integrations":0,"fits_scans_new_coefficients_API_calls":0,
        "source_state_after_creation":False,"controller_hz":48,"plant_substeps":4,
        "parallel_authorized_exclusions":["scripts/check_r771_force_allocation.py","outputs/jari_offaxis/force_allocation/"]}
    if not verification["protected_prior_size_mtime_unchanged"] or not verification["all_saved_arrays_finite"]:raise AssertionError(verification)
    np.savez_compressed(OUT/"trajectories.npz",**arrays)
    slew.write_json(OUT/"results.json",{"task":"Frozen force-rolloff high-angle moment formulas; game-only script experiments","cases":reports,"native_decisions":decisions,"native_cases":native,
        "verification":verification,"formulas":{"M_all":"qdot_new=R(s)*qdot_original; K unchanged","M_restore":"qdot_new=qdot_original+(1-R)*A*Ka*local; K_new=K_original/R"},
        "limits":["Both formulas are D-level finite-angle extensions, not recovered game code.","Same single distFromCmToStab station and frozen five numbers remain; moment formula is intentionally changed.",
            "Old linear-law needed-input inverse is not a truth target for these changed moment laws and was not used.","Original G-error PID, transport latch, target1 cap/AW, schedules, halfPN/loft/S110 and clocks are unchanged.","Source24Hz interpolation/derivatives enter comparisons only, never runtime."]})
    print(json.dumps({"complete":True,"native_expand":{k:v["expand_native"] for k,v in decisions.items()},"verification":verification}),flush=True)


if __name__=="__main__":main()
