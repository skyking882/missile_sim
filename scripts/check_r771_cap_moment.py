#!/usr/bin/env python3
"""Combine two existing fixed moment laws with the existing all-active cap.

War Thunder game-only experiments. No new runtime thresholds, units, gains,
fits or schedules. Default code and all prior experiment receipts are inputs.
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
import check_r771_nonlinear_moment as moment
import check_r771_all_active_cap as cap
from aim120_model.control import scheduled_pid_values
from aim120_model.math3d import norm,sub
from aim120_model.single_surface_game import SurfaceCoefficients

geometry=moment.geometry;dec=moment.dec;fix=moment.fix;finite=moment.finite
fields,drag,slew,harness=moment.fields,moment.drag,moment.slew,moment.harness
OUT=ROOT/"outputs/jari_offaxis/cap_moment"
KINDS=moment.KINDS;CASES=moment.CASES;NATIVE=moment.NATIVE
WINDOWS=cap.WINDOWS
UNITS=moment.UNITS
PRIMARY=cap.PRIMARY


class CapMomentGame(moment.NonlinearMomentGame):
    def initialize(self,state):
        initial=super().initialize(state)
        self.game_control=cap.AllActiveCapControl()
        return initial

    def run(self,initial_state,target_at,*,end_time_s):
        result=super().run(initial_state,target_at,end_time_s=end_time_s)
        result["model"].update(controller_mechanism="existing AllActiveCapControl + existing NonlinearMomentGame",
            experimental_target_cap="min(raw old-held limit,raw/K) at every active tick; D hypothesis",
            integral_transport="original high-angle latch only; same existing memory coordinates",
            combination_new_constants=[],source_data_in_runtime=False,default_model_changed=False)
        return result


def make_model(cfg,coefficients,kind):
    return CapMomentGame(cfg,moment_kind=kind,request_slew_g_s=110.,drag_coefficients=coefficients,
        coefficients=SurfaceCoefficients(),force_basis="wind_linear",plant_substeps=4,loft_cap_unit="rad_s")


def snapshot():
    return {str(p):(p.stat().st_size,p.stat().st_mtime_ns) for folder in ("src","config","configs","schemas","docs","scripts","data","outputs/jari_offaxis")
        for p in (ROOT/folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts and ".git" not in p.parts
        and OUT not in p.parents and p!=Path(__file__).resolve()}


def verify(model,result,cfg,coefficients,initial,target_at,cap_full):
    original=geometry.make_model(cfg,coefficients);original.initialize(dataclasses.replace(initial))
    direct_moment=moment.make_model(cfg,coefficients,model.moment_kind);direct_moment.initialize(dataclasses.replace(initial))
    maxima={"qdot_existing_moment_rad_s2":0.,"moment_K":0.,"request_same_state_G":0.,"force_same_state_N":0.,
        "drag_same_state_N":0.,"acceleration_same_state_mps2":0.,"lever_same_state_Nm":0.,
        "PID_held_rad":0.,"PID_memory":0.,"P_I_D_rad":0.,"I_transport_ratio":0.,"effective_limit_rad":0.}
    local_max=0.;min_r=1.;bytime={s["time_s"]:s for s in result["samples"]}
    for s in result["samples"]:
        state=finite.sample_state(s);held=tuple(s["evaluated_old_held_rad"])
        f=original.evaluate(state,s["time_s"],held);m=direct_moment.evaluate(state,s["time_s"],held);c=model.evaluate(state,s["time_s"],held)
        local_max=max(local_max,max(abs(v) for v in c.local_incidence));min_r=min(min_r,moment.rolloff(c.local_incidence))
        for key,expected,actual in (("qdot_existing_moment_rad_s2",m.angular_acceleration,c.angular_acceleration),
            ("force_same_state_N",f.aerodynamic_force,c.aerodynamic_force),("drag_same_state_N",f.wind_drag_force,c.wind_drag_force),
            ("acceleration_same_state_mps2",f.acceleration,c.acceleration),("lever_same_state_Nm",f.lever_moment,c.lever_moment)):
            maxima[key]=max(maxima[key],max(abs(a-b) for a,b in zip(expected,actual)))
    clone=cap.AllActiveCapControl();memory=[(0.,0.,0.),(0.,0.,0.)]
    limits=np.radians([cfg["aerodynamics"][k] for k in ("horizontal_fin_aoa_limit_deg","vertical_fin_aoa_limit_deg")])
    arrays={k:[] for k in ("t","held","memory","P","I_provisional","D","error_G","K","transport_ratio","AW","limit","high_latch")}
    for tick in result["ticks"]:
        ts=tick["time_s"];s=bytime[ts];state=finite.sample_state(s);held=tuple(s["evaluated_old_held_rad"])
        f=direct_moment.evaluate(state,ts,held);request,_=direct_moment.request(state,target_at(ts),ts,f)
        maxima["request_same_state_G"]=max(maxima["request_same_state_G"],max(abs(a-b) for a,b in zip(request,tick["request_g"])))
        k=direct_moment.current_control_k
        maxima["moment_K"]=max(maxima["moment_K"],abs(k-tick["moment_controller_K"]))
        if tick["gain"]<=1e-12:
            if np.any(tick["pid_memory"]) or np.any(tick["held_fin_rad"]):raise AssertionError("inactive memory")
            continue
        pid=scheduled_pid_values(cfg["control"]["pid"],(round(ts*48)+1)/48)
        values={key:[] for key in arrays if key!="t"}
        for axis,d in enumerate(tick["control_runtime_diagnostics"]):
            error=tick["request_g"][axis]-tick["feedback_g"][axis]
            out=clone.update_axis(axis,ts,error,memory[axis],pid,limits[axis],tick["gain"],k,direct_moment.current_force_nonlinear)
            expected_d=clone.diagnostics[axis]
            maxima["PID_held_rad"]=max(maxima["PID_held_rad"],abs(out[0]-tick["held_fin_rad"][axis]))
            maxima["PID_memory"]=max(maxima["PID_memory"],max(abs(a-b) for a,b in zip(out[1],tick["pid_memory"][axis])))
            maxima["P_I_D_rad"]=max(maxima["P_I_D_rad"],max(abs(d[key]-expected_d[key]) for key in ("P_old_rad","I_provisional_old_rad","D_old_rad","natural_old_rad")))
            maxima["I_transport_ratio"]=max(maxima["I_transport_ratio"],abs(d["integral_transport_ratio"]-expected_d["integral_transport_ratio"]))
            maxima["effective_limit_rad"]=max(maxima["effective_limit_rad"],abs(d["effective_old_held_limit_rad"]-min(limits[axis],limits[axis]/k)))
            if d["antiwindup_blocked"]!=expected_d["antiwindup_blocked"] or d["nonlinear_latched"]!=expected_d["nonlinear_latched"]:
                raise AssertionError("AW/high-only latch differs")
            memory[axis]=out[1]
            for key,value in (("held",out[0]),("memory",out[1]),("P",d["P_old_rad"]),("I_provisional",d["I_provisional_old_rad"]),
                ("D",d["D_old_rad"]),("error_G",error),("K",k),("transport_ratio",d["integral_transport_ratio"]),
                ("AW",d["antiwindup_blocked"]),("limit",d["effective_old_held_limit_rad"]),("high_latch",d["nonlinear_latched"])):values[key].append(value)
        arrays["t"].append(ts)
        for key,value in values.items():arrays[key].append(value)
    low_cap_exact=cap.physical_exact(result,cap_full) if min_r==1. and cap_full is not None else None
    checks={"max_residuals":maxima,"max_local_axis_deg":math.degrees(local_max),"minimum_R":min_r,
        "geometry90_support":local_max<math.pi/2,"never_R_physical_all_active_cache_exact":low_cap_exact,
        "raw_config_PID_equal":model.config==cfg,"five_coefficients_equal":model.coefficients==SurfaceCoefficients(),
        "drag_equal":model.drag_coefficients==coefficients,"PN_loft_S110_equal":model.relative_blend==.5 and model.request_slew_g_s==110. and [model.amplitude_coefficient,model.transition_coefficient]==[.75,1.3],
        "one_station0p4":model.arm==.4,"controller_hz":model.controller_hz,"plant_substeps":model.plant_substeps,
        "clock48_error":max(abs(t["time_s"]*48-i) for i,t in enumerate(result["ticks"])),
        "force_moment_record_closure":dec.audit.inverse.saved_closure(model,result),"output_limits":fix.verify_controls(model,result)}
    if max(maxima.values())>1e-10 or low_cap_exact is False or not all(checks[k] for k in ("geometry90_support","raw_config_PID_equal","five_coefficients_equal","drag_equal","PN_loft_S110_equal","one_station0p4")) or checks["clock48_error"]>1e-8:
        raise AssertionError(checks)
    return checks,{k:np.asarray(v) for k,v in arrays.items()}


def plot(cid,curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for ax,(key,unit) in zip(axes.flat,(("speed_mps","Speed (m/s)"),("alpha_deg","AoA (deg)"),("normal_specific_load_g","Normal specific load (G)"),
        ("body_pitch_deg","Body pitch (deg)"),("body_yaw_deg","Body yaw (deg)"),("body_yaw_rate_deg_s","Yaw rate (deg/s)"))):
        for name,p,color in curves:ax.plot(p["t"],p[key],label=name,color=color,lw=1.2)
        ax.set(xlabel="Original clock (s)",ylabel=unit,xlim=(0,curves[0][1]["t"][-1] if key=="speed_mps" else 3.5));ax.grid(alpha=.18)
    axes[0,0].legend(fontsize=7);fig.suptitle(cid+": existing all-active cap + existing moment formulas; no new runtime coefficient")
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(OUT/(cid+"_free.png"),dpi=150);plt.close(fig)


def run_jari(inputs,coefficients):
    reports={kind:{} for kind in KINDS};arrays={};new=recovered=0
    caches={"geometry":np.load(geometry.OUT/"free_trajectories.npz"),"all_active_cap":np.load(cap.OUT/"trajectories.npz"),"moment_only":np.load(moment.OUT/"trajectories.npz")}
    full=OUT/"full_runs";full.mkdir(parents=True,exist_ok=True)
    for cid in CASES:
        x=inputs[cid];observed=drag.selected_arrays(caches["geometry"],cid+"__reference__")
        common={name:drag.selected_arrays(caches[name],cid+"__candidate__") for name in ("geometry","all_active_cap")}
        cap_full=json.loads((cap.OUT/"full_runs"/(cid+".json")).read_text())
        curves=[("Jari reference",observed,"#243f56"),("Cached geometry",common["geometry"],"#a4a4a4"),("Cached all-active cap",common["all_active_cap"],"#c79d49")]
        for kind,color in zip(KINDS,("#168477","#a4508b")):
            model=make_model(x["config"],coefficients,kind);path=full/(kind+"_"+cid+".json")
            result,failure,n,r=moment.run_once(model,x["initial"],x["target_at"],x["config"]["performance"]["lifetime_s"],path);new+=n;recovered+=r
            if failure:reports[kind][cid]={"actual_failure":failure};print(json.dumps({"failed":kind,"case":cid,"failure":failure}),flush=True);continue
            prediction=fields.prediction_fields(result,model,x["run"]["scenario"])
            baselines={**common,"moment_only":drag.selected_arrays(caches["moment_only"],cid+"__"+kind+"__")}
            checks,tick_arrays=verify(model,result,x["config"],coefficients,x["initial"],x["target_at"],cap_full)
            metrics={name:drag.pair_metrics(observed,b,prediction,tuple(cap.UNITS),WINDOWS) for name,b in baselines.items()}
            for m in metrics.values():
                for w in m.values():
                    for key,v in w["metrics"].items():v["units"]=cap.UNITS[key]
            peaks={name:{key:cap.first_peak(p,key) for key in ("body_yaw_rate_deg_s","alpha_deg","normal_specific_load_g")}
                for name,p in {"reference":observed,**baselines,"candidate":prediction}.items()}
            reports[kind][cid]={"summary":result["summary"],"metrics":metrics,"first_positive_peaks":peaks,"focused_checks":checks,
                "later_peaks":{name:fix.trim.response_extrema(p) for name,p in {"reference":observed,**baselines,"candidate":prediction}.items()},
                "Omega_events":{name:finite.rate_events(p) for name,p in {"reference":observed,**baselines,"candidate":prediction}.items()},"full_result_path":str(path)}
            fields.save_arrays(arrays,cid+"__"+kind,prediction);arrays.update({cid+"__"+kind+"__tick__"+key:value for key,value in tick_arrays.items()})
            for name,p in baselines.items():fields.save_arrays(arrays,cid+"__"+kind+"__baseline__"+name,p)
            curves.append(("Cap + "+kind,prediction,color))
            print(json.dumps({"completed":kind,"case":cid,"first_peaks":peaks["candidate"],"early_primary_RMS":{k:metrics["geometry"]["early_0_3p5"]["metrics"][k]["new"]["rmse"] for k in PRIMARY}}),flush=True)
        fields.save_arrays(arrays,cid+"__reference",observed);plot(cid,curves)
    return reports,arrays,new,recovered


def gates(reports):
    decisions={}
    for kind,cases in reports.items():
        rule={};failed=any("actual_failure" in r for r in cases.values())
        for cid,r in cases.items():
            if "actual_failure" in r:continue
            for window in ("early_0_3p5","full"):
                m=r["metrics"]["geometry"][window]["metrics"]
                for key in PRIMARY:rule[cid+"__"+window+"__no10percent_worse__"+key]=m[key]["new"]["rmse"]<=1.10*m[key]["old"]["rmse"]+1e-12
                if cid=="jari_259_60":
                    for key in ("speed_mps","alpha_deg"):rule[cid+"__"+window+"__improve10percent__"+key]=m[key]["new"]["rmse"]<=.90*m[key]["old"]["rmse"]
            if cid=="jari_259_60":
                for key in ("body_yaw_rate_deg_s","alpha_deg"):
                    ref,base,new=(r["first_positive_peaks"][name][key] for name in ("reference","geometry","candidate"))
                    rule[cid+"__first_peak_amplitude_closer__"+key]=bool(ref and base and new and abs(new["value"]-ref["value"])<abs(base["value"]-ref["value"]))
        decisions[kind]={"rules":rule,"actual_failure":failed,"expand_native":not failed and len(cases)==3 and all(rule.values()),"composite_score":False}
    return decisions


def run_native(kind,coefficients,arrays):
    cases={c["id"]:c for c in harness.load_cases(finite.comparison.SNAPSHOT)}
    cached=np.load(geometry.OUT/"native_trajectories.npz");reports={};new=recovered=0
    for cid in NATIVE:
        x=cases[cid.removeprefix("native_")];cfg=copy.deepcopy(x["profile"]["_model_config"]);initial=dataclasses.replace(x["_creation_state"])
        qx,qy,qz,qw=x["creation"]["u16_1"];initial.orientation_quaternion=(qw,qx,qy,qz)
        target=x["_target_model"].state_at;model=make_model(cfg,coefficients,kind);path=OUT/"full_runs"/(kind+"_"+cid+".json")
        result,failure,n,r=moment.run_once(model,initial,target,x["scenario"]["max_simulation_time_s"],path);new+=n;recovered+=r
        if failure:reports[cid]={"actual_failure":failure,"passes":False};continue
        p=harness.sample_series(result["samples"],x);observed=drag.selected_arrays(cached,cid+"__reference__");base=drag.selected_arrays(cached,cid+"__candidate__")
        geometry_full=json.loads((geometry.OUT/(cid+"_run.json")).read_text())
        restored=harness.sample_series(geometry_full["samples"],x)
        if not all(np.array_equal(v,base[k]) for k,v in restored.items()):raise AssertionError("native baseline receipt")
        cap_path=cap.OUT/(cid+"_run.json")
        cap_full=json.loads(cap_path.read_text()) if cap_path.exists() else None
        # An exact-low comparison is optional when a single-cap native receipt
        # does not exist; it never supplies candidate missile states.
        checks,_=verify(model,result,cfg,coefficients,initial,target,cap_full)
        windows=(*dec.WINDOWS,*harness.WINDOWS)
        metrics=drag.pair_metrics(observed,base,p,("speed_kmh","alpha_deg","pitch_deg","omega_rad_s","gamma_deg","y_m"),windows)
        gate={w+"__"+k:v["new"]["rmse"]<=1.10*v["old"]["rmse"]+1e-12 for w,m in metrics.items() for k,v in m["metrics"].items() if k in ("speed_kmh","alpha_deg")}
        load={}
        for name,lo,hi in windows:
            t=observed["t"];mask=(t>=lo-1e-9)&(t<=hi+1e-9)&(t>=max(p["t"][0],base["t"][0]))&(t<=min(p["t"][-1],base["t"][-1]));ts=t[mask]
            load[name]={"independent_observation_available":False,"prediction_delta_G":drag.stats(np.interp(ts,p["t"],p["load_g"])-np.interp(ts,base["t"],base["load_g"]))}
        reports[cid]={"metrics":metrics,"load_prediction_only":load,"gate":gate,"passes":all(gate.values()),"focused_checks":checks,"summary":result["summary"],"full_result_path":str(path)}
        fields.save_arrays(arrays,cid+"__"+kind,p)
        print(json.dumps({"native_completed":cid,"kind":kind,"passes":reports[cid]["passes"]}),flush=True)
    return reports,new,recovered


def main():
    before=snapshot();OUT.mkdir(parents=True,exist_ok=True)
    coefficients=drag.DragCoefficients(**json.loads((drag.OUTPUT/"fit.json").read_text())["coefficients"])
    reports,arrays,new,recovered=run_jari(dec.audit.inverse.source_inputs(),coefficients)
    decisions=gates(reports);native={}
    for kind,d in decisions.items():
        if d["expand_native"]:native[kind],n,r=run_native(kind,coefficients,arrays);new+=n;recovered+=r
    verification={"protected_prior_size_mtime_unchanged":before==snapshot(),"all_arrays_finite":all(np.isfinite(v).all() for v in arrays.values()),
        "fixed_combinations":2,"unique_Jari_free_attempts":6,"native_free_attempts":sum(len(x) for x in native.values()),
        "new_free_attempts_this_invocation":new,"recovered_this_invocation":recovered,"baseline_integrations":0,"fits_scans_new_constants_API_calls":0,
        "runtime_creation_state_only":True,"controller_hz":48,"plant_substeps":4,"default_model_changed":False}
    if not verification["protected_prior_size_mtime_unchanged"] or not verification["all_arrays_finite"]:raise AssertionError(verification)
    np.savez_compressed(OUT/"trajectories.npz",**arrays)
    slew.write_json(OUT/"results.json",{"task":"Fixed all-active-cap + existing nonlinear moment combinations; actual game-unit free runs",
        "cases":reports,"Jari_decisions":decisions,"native_cases":native,"verification":verification,
        "accepted":{kind:d["expand_native"] and bool(native.get(kind)) and all(r["passes"] for r in native[kind].values()) for kind,d in decisions.items()},
        "acceptance":"0/30/60 primary early/full each <=110% geometry;60 speed+alpha early/full <=90%,first yaw/alpha amplitude closer;native speed/alpha each window <=110%;load prediction only",
        "limits":["Two existing D-grade mechanisms are composed; no native game field/formula identification.","No source missile state/feedback/controls enters runtime after fixed creation.","Old needed-input inverse targets are not used for changed moment laws.","First peak timing and every adverse window are reported separately without a combined score."]})
    print(json.dumps({"complete":True,"native_expand":{k:v["expand_native"] for k,v in decisions.items()},"verification":verification}),flush=True)


if __name__=="__main__":main()
