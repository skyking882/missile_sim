#!/usr/bin/env python3
"""One causal current-state envelope/control-cap hypothesis for game units.

Keep the original G-error PID, old-held integral, K transport, clocks and
norm-preserving single-site lift geometry. Above the frozen force peak only,
project unreachable total requests onto the current thrust-centred force disk
and replace the unverified additional target1 bound with the original raw
old-held output bound. No trim, SI interface, latch or fitted coefficient.
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
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/missle_sim_mpl")
import numpy as np
import check_r771_lift_direction as geometry
import check_r771_control_authority as authority
from aim120_model.aerodynamics import body_axes_for_state
from aim120_model.control import scheduled_pid_values
from aim120_model.math3d import dot, norm, sub
from aim120_model.r771_game_control import R771GameControl
from aim120_model.single_surface_game import SurfaceCoefficients
from aim120_model.units import G0_MPS2

dec = geometry.dec
fix = dec.audit.fix
finite, fields, drag, slew, harness = fix.finite, fix.finite.fields, dec.audit.drag, dec.audit.slew, fix.finite.harness
OUT = ROOT / "outputs/jari_offaxis/envelope_control"
CASES = geometry.CASES
PEAK_S = math.hypot(finite.ALPHA_S, finite.WIDTH)
PEAK_FORCE = PEAK_S / (1 + ((PEAK_S - finite.ALPHA_S) / finite.WIDTH) ** 2)
WINDOWS = (*dec.WINDOWS, ("early_0_3p5", 0., 3.5), ("full_common", 0., math.inf))
UNITS = {**finite.UNITS, "body_pitch_deg": "deg", "body_pitch_rate_deg_s": "deg/s",
         "feedback_pitch_G": "G", "feedback_yaw_G": "G"}


class EnvelopeControl(R771GameControl):
    def __init__(self):
        super().__init__()
        self.raw_output_cap = False
        # AuthorityControl's existing transport/raw cell has exactly the same
        # memory coordinates and raw update, changing only effective limit/AW.
        self.transport = True
        self.target_cap = False
        self.last_terms = [None, None]

    def update_axis(self, *args):
        if self.raw_output_cap:
            result = authority.AuthorityControl.update_axis(self, *args)
            self.diagnostics[args[0]]["original_old_held_limit_rad"] = args[5]
            self.diagnostics[args[0]]["raw_cap_current_peak_guard"] = True
            return result
        result = super().update_axis(*args)
        self.diagnostics[args[0]]["raw_cap_current_peak_guard"] = False
        return result


class EnvelopeGame(geometry.LiftDirectionGame):
    def initialize(self, state):
        result = super().initialize(state)
        self.game_control = EnvelopeControl()
        return result

    def request(self, state, target, time_s, evaluation):
        raw, details = super().request(state, target, time_s, evaluation)
        local_s = math.hypot(*evaluation.local_incidence)
        above = local_s > PEAK_S
        axes = body_axes_for_state(state)
        wb = evaluation.wind_basis
        thrust = np.array([dot(axes.forward, b) for b in (wb.up, wb.right)]) * evaluation.thrust / evaluation.mass / G0_MPS2
        radius = evaluation.qbar * self.area * self.coefficients.ca * PEAK_FORCE / evaluation.mass / G0_MPS2
        vector = np.asarray(raw) - thrust
        desired = float(np.linalg.norm(vector))
        projection = above and desired > radius
        ratio = radius / desired if projection else 1.
        effective = tuple(float(x) for x in (thrust + vector * ratio)) if projection else raw
        change = np.asarray(effective) - raw
        gravity = np.array([dot((0., self.gravity, 0.), b) for b in (wb.up, wb.right)]) / G0_MPS2
        gravity_change = (ratio - 1.) * gravity
        self.game_control.raw_output_cap = above
        details.update(envelope_above_peak=above, envelope_projected=projection,
            envelope_local_norm_rad=local_s, envelope_peak_rad=PEAK_S,
            envelope_raw_request_G=raw, envelope_effective_request_G=effective,
            envelope_thrust_center_G=thrust.tolist(), envelope_radius_G=radius,
            envelope_desired_aero_norm_G=desired, envelope_projection_ratio=ratio,
            envelope_request_change_G=change.tolist(), envelope_gravity_change_G=gravity_change.tolist(),
            envelope_remaining_change_G=(change - gravity_change).tolist())
        return effective, details

    def run(self, initial_state, target_at, *, end_time_s):
        result = super().run(initial_state, target_at, end_time_s=end_time_s)
        result["model"].update(runtime_controller="script-local EnvelopeControl; original G-error memory",
            envelope_input="post-original-S110 thrust-centred finite-force disk only above current local force peak",
            envelope_output="raw old-held cap above current peak; original target1 cap elsewhere",
            envelope_guards_latched=False, original_integral_transport_latch_preserved=True,
            lift_direction_rule=geometry.COM_GEOMETRY_FORMULA,
            new_shared_coefficients=[], reference_data_in_runtime=False, native_game_formula_recovered=False)
        return result


def make_model(config, coefficients):
    return EnvelopeGame(config, request_slew_g_s=110., drag_coefficients=coefficients,
        coefficients=SurfaceCoefficients(), force_basis="wind_linear", plant_substeps=4, loft_cap_unit="rad_s")


def snapshot():
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns)
        for folder in ("src", "config", "configs", "schemas", "docs", "scripts", "data", "outputs/jari_offaxis")
        for p in (ROOT / folder).rglob("*") if p.is_file()
        and "__pycache__" not in p.parts and ".git" not in p.parts and OUT not in p.parents
        and p != Path(__file__).resolve()}


def enriched(observed, item):
    p = dict(observed)
    q = fields.gap.continuous_quaternions(item["ref"]["quaternion_xyzw"][:, [3, 0, 1, 2]])
    d = fields.gap.trajectory_fields(item["ref"]["t"], item["ref"]["position_xyz_m"], item["ref"]["velocity_xyz_mps"],
        item["run"]["scenario"]["target_position"], item["run"]["scenario"]["target_velocity"], q)
    p["body_pitch_deg"] = d["body_pitch_deg"]
    p["body_pitch_rate_deg_s"] = -np.degrees(item["ref"]["omega_xyz_export_units"][:, 2])
    return p


def tick_arrays(result):
    rows = result["ticks"]
    return {"t": np.array([t["time_s"] for t in rows]),
        **{k: np.asarray([t[k] for t in rows]) for k in ("request_g", "feedback_g", "held_fin_rad", "pid_memory",
            "envelope_above_peak", "envelope_projected", "envelope_raw_request_G", "envelope_request_change_G",
            "envelope_gravity_change_G", "envelope_remaining_change_G", "envelope_projection_ratio")}}


def verify_result(model, result, config, coefficients, geometry_result):
    limits = np.radians([config["aerodynamics"][k] for k in ("horizontal_fin_aoa_limit_deg", "vertical_fin_aoa_limit_deg")])
    memory = [(0., 0., 0.), (0., 0., 0.)]
    previous_k = [None, None]
    high = False
    maximum = {"output_rad": 0., "memory": 0., "transport_ratio": 0., "projection_G": 0.,
        "geometry_scalar_force_relative": 0., "geometry_angular_law_rad_s2": 0.}
    authority_effect = {"rawcap_axis_ticks":0,"old_target1_would_clip_axis_ticks":0,
        "old_target1_AW_differs_axis_ticks":0,"held_change_max_abs_rad":0.}
    local_max = 0.
    for tick in result["ticks"]:
        raw = np.asarray(tick["envelope_raw_request_G"])
        vector = raw - tick["envelope_thrust_center_G"]
        should_project = tick["envelope_above_peak"] and np.linalg.norm(vector) > tick["envelope_radius_G"]
        ratio = tick["envelope_radius_G"] / np.linalg.norm(vector) if should_project else 1.
        expected = np.asarray(tick["envelope_thrust_center_G"]) + vector * ratio if should_project else raw
        maximum["projection_G"] = max(maximum["projection_G"], float(np.max(abs(expected - tick["request_g"]))))
        if bool(tick["envelope_projected"]) != bool(should_project): raise AssertionError("projection guard")
        if tick["gain"] <= 1e-12: continue
        diagnostics = tick["control_runtime_diagnostics"]
        high = high or tick["envelope_local_norm_rad"] > finite.ALPHA_S
        pid = scheduled_pid_values(config["control"]["pid"], (round(tick["time_s"] * 48) + 1) / 48)
        for axis, d in enumerate(diagnostics):
            k = d["K_target_per_old_held"]
            i, e, derivative = memory[axis]
            transport = previous_k[axis] / k if high and previous_k[axis] is not None else 1.
            i = authority.audit.clip(i * transport, pid["integral_limit"]) if high else i
            error = tick["request_g"][axis] - tick["feedback_g"][axis]
            provisional = authority.audit.clip(i + pid["i"] * error / 48, pid["integral_limit"])
            derivative += (1 / 48) / (1e-9 + 1 / 48) * ((error - e) * 48 - derivative)
            cap = limits[axis] if tick["envelope_above_peak"] or not high else min(limits[axis], limits[axis] / k)
            post = authority.audit.clip(pid["p"] * error + provisional + pid["d"] * derivative, cap)
            blocked = abs(post) >= .99 * cap and error * post > 0
            if tick["envelope_above_peak"]:
                authority_effect["rawcap_axis_ticks"] += 1
                old_cap = min(limits[axis], limits[axis] / k) if high else limits[axis]
                old_post = authority.audit.clip(pid["p"] * error + provisional + pid["d"] * derivative, old_cap)
                old_blocked = abs(old_post) >= .99 * old_cap and error * old_post > 0
                authority_effect["old_target1_would_clip_axis_ticks"] += int(abs(old_post-post)>1e-12)
                authority_effect["old_target1_AW_differs_axis_ticks"] += int(old_blocked!=blocked)
                authority_effect["held_change_max_abs_rad"] = max(authority_effect["held_change_max_abs_rad"],abs(tick["gain"]*(post-old_post)))
            next_memory = (i if blocked else provisional, error, derivative)
            maximum["output_rad"] = max(maximum["output_rad"], abs(tick["gain"] * post - tick["held_fin_rad"][axis]))
            maximum["memory"] = max(maximum["memory"], max(abs(a-b) for a,b in zip(next_memory,tick["pid_memory"][axis])))
            maximum["transport_ratio"] = max(maximum["transport_ratio"], abs(transport-d["integral_transport_ratio"]))
            if d["nonlinear_latched"] != high or d["antiwindup_blocked"] != blocked or abs(d["effective_old_held_limit_rad"]-cap)>1e-12:
                raise AssertionError("memory/cap/AW guard inconsistent")
            if abs(tick["held_fin_rad"][axis]) > limits[axis]+1e-12 or abs(next_memory[0]) > pid["integral_limit"]+1e-12:
                raise AssertionError("raw bounds")
            memory[axis] = next_memory
            if high: previous_k[axis] = k
    original = fix.make_model(config, coefficients)
    original.initialize(finite.sample_state(result["samples"][0]))
    for s in result["samples"]:
        state = finite.sample_state(s)
        held = tuple(s["evaluated_old_held_rad"])
        f = model.evaluate(state, s["time_s"], held)
        old = original.evaluate(state, s["time_s"], held)
        local_max = max(local_max, max(abs(v) for v in f.local_incidence))
        maximum["geometry_scalar_force_relative"] = max(maximum["geometry_scalar_force_relative"], abs(norm(f.wind_lift_force)-norm(old.wind_lift_force))/max(norm(old.wind_lift_force),1.))
        maximum["geometry_angular_law_rad_s2"] = max(maximum["geometry_angular_law_rad_s2"],max(abs(a-b) for a,b in zip(f.angular_acceleration,old.angular_acceleration)))
    never = not any(t["envelope_above_peak"] for t in result["ticks"])
    exact = None
    if never:
        def original_keys_equal(new, old):
            if isinstance(old,dict):return all(k in new and original_keys_equal(new[k],v) for k,v in old.items())
            if isinstance(old,(list,tuple)):return len(new)==len(old) and all(original_keys_equal(a,b) for a,b in zip(new,old))
            return new==old
        exact = all(original_keys_equal(result[key],geometry_result[key]) for key in ("samples","ticks","summary"))
        if not exact: raise AssertionError("never-trigger geometry baseline changed")
    check = {"max_closure_residuals": maximum, "local_axis_max_abs_deg": math.degrees(local_max),
        "rawcap_same_error_memory_one_step_counterfactual":authority_effect,
        "geometry_below90deg": local_max < math.pi / 2, "never_trigger_original_geometry_exact": exact,
        "config_raw_PID_equal": model.config == config, "surface_equal": model.coefficients == SurfaceCoefficients(),
        "drag_equal": model.drag_coefficients == coefficients, "PN_loft_S110_equal": model.relative_blend == .5 and model.request_slew_g_s == 110 and [model.amplitude_coefficient,model.transition_coefficient] == [.75,1.3],
        "controller_hz": model.controller_hz, "plant_substeps": model.plant_substeps,
        "force_moment_closure": dec.audit.inverse.saved_closure(model,result)}
    if max(maximum.values()) > 1e-10 or not all(check[k] for k in ("geometry_below90deg","config_raw_PID_equal","surface_equal","drag_equal","PN_loft_S110_equal")):
        raise AssertionError(check)
    return check


def projection_report(a):
    report = {}
    for name, lo, hi in WINDOWS:
        mask = (a["t"] >= lo-1e-9) & (a["t"] <= hi+1e-9)
        report[name] = {"ticks": int(mask.sum()), "rawcap_ticks": int(np.sum(mask & a["envelope_above_peak"])),
            "range_capped_ticks": int(np.sum(mask & a["envelope_projected"])), "axes": {}}
        for j, axis in enumerate(("pitch","yaw")):
            report[name]["axes"][axis] = {key: {"RMS_G": float(np.sqrt(np.mean(a[key][mask,j]**2))), "bias_G": float(np.mean(a[key][mask,j]))}
                for key in ("envelope_request_change_G","envelope_gravity_change_G","envelope_remaining_change_G")} if mask.any() else {}
    return report


def plot(cid, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,4,figsize=(18,8))
    specs = (("speed_mps","Speed (m/s)"),("alpha_deg","AoA (deg)"),("normal_specific_load_g","Normal specific load (G)"),
        ("body_yaw_rate_deg_s","Yaw rate (deg/s)"),("body_pitch_deg","Body pitch (deg)"),("body_yaw_deg","Body yaw (deg)"),
        ("feedback_pitch_G","Pitch feedback (G)"),("feedback_yaw_G","Yaw feedback (G)"))
    for ax,(key,ylabel) in zip(axes.flat,specs):
        for name,p,color in curves:
            if key in p: ax.plot(p["t"],p[key],label=name,color=color,lw=1.2)
        ax.set(xlabel="Original clock (s)",ylabel=ylabel,xlim=(0,curves[0][1]["t"][-1] if key=="speed_mps" else 3.5))
        ax.grid(alpha=.17)
    axes[0,0].legend(fontsize=7)
    fig.suptitle(cid+": current peak guard; original G-error / I transport; geometry + reachable request / raw cap")
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(OUT/(cid+"_free.png"),dpi=150);plt.close(fig)


def run_jari(inputs,coefficients):
    arrays={};reports={};new=0;recovered=0
    cache=np.load(fix.OUTPUT/"trajectories.npz")
    geometry_cache=np.load(geometry.OUT/"free_trajectories.npz")
    feedback=np.load(ROOT/"outputs/jari_offaxis/feedback_interfaces/axial_completed_interfaces.npz")
    full=OUT/"full_runs";full.mkdir(parents=True,exist_ok=True)
    for cid in CASES:
        x=inputs[cid];cfg=copy.deepcopy(x["config"]);model=make_model(cfg,coefficients);path=full/(cid+".json")
        if path.exists(): result=json.loads(path.read_text());model.initialize(dataclasses.replace(x["initial"]));recovered+=1
        else:
            result=model.run(dataclasses.replace(x["initial"]),x["target_at"],end_time_s=cfg["performance"]["lifetime_s"])
            slew.write_json(path,result,True);new+=1
            result=json.loads(path.read_text())
        p=fields.prediction_fields(result,model,x["run"]["scenario"])
        observed=enriched(x["observed"],x)
        bases={"control_fix":drag.selected_arrays(cache,cid+"__control_fix__"),"geometry":drag.selected_arrays(geometry_cache,cid+"__candidate__")}
        for name,base in bases.items():
            source_dir=fix.OUTPUT if name=="control_fix" else geometry.OUT
            baseline_result=json.loads((source_dir/"full_runs"/(cid+".json")).read_text())
            base["feedback_pitch_G"]=np.array([s["feedback_pitch_g"] for s in baseline_result["samples"]])
            base["feedback_yaw_G"]=np.array([s["feedback_yaw_g"] for s in baseline_result["samples"]])
            base["body_pitch_rate_deg_s"]=np.degrees([s["pitch_rate_rad_s"] for s in baseline_result["samples"]])
            if name=="geometry":geometry_result=baseline_result
        p["feedback_pitch_G"]=np.array([s["feedback_pitch_g"] for s in result["samples"]]);p["feedback_yaw_G"]=np.array([s["feedback_yaw_g"] for s in result["samples"]])
        p["body_pitch_rate_deg_s"]=np.degrees([s["pitch_rate_rad_s"] for s in result["samples"]])
        # Reference signed total-specific-force feedback is cached diagnostic
        # data only; it is never supplied to the runtime model.
        signed={};reference_feedback=None
        for width in (3,5):
            t=feedback[cid+"__t"];source=feedback[cid+f"__{width}__source_wind_specific_feedback_G_diagnostic"]
            edge=width//2;source_t=x["ref"]["t"]
            valid=(t>=source_t[edge])&(t<=source_t[-1-edge])&~feedback[cid+"__gain_transition_derivative_halo"].astype(bool)&np.isfinite(source).all(axis=1)
            signed[str(width)+"point"]={}
            for w,lo,hi in WINDOWS:
                mask=valid&(t>=lo-1e-9)&(t<=hi+1e-9)
                signed[str(width)+"point"][w]={"ticks":int(mask.sum()),"axes":{}}
                for j,axis in enumerate(("pitch","yaw")):
                    key="feedback_"+axis+"_G"
                    signed[str(width)+"point"][w]["axes"][axis]={name:drag.stats(np.interp(t[mask],v["t"],v[key])-source[mask,j]) for name,v in {**bases,"candidate":p}.items()} if mask.any() else {}
            if width==5:reference_feedback={"t":t[valid],"feedback_pitch_G":source[valid,0],"feedback_yaw_G":source[valid,1]}
        compared=tuple(k for k in UNITS if k in observed and all(k in b for b in bases.values()) and k in p)
        metrics={name:drag.pair_metrics(observed,base,p,compared,WINDOWS) for name,base in bases.items()}
        for m in metrics.values():
            for w in m.values():
                for key,v in w["metrics"].items():v["units"]=UNITS[key]
        a=tick_arrays(result)
        checks=verify_result(model,result,cfg,coefficients,geometry_result)
        reports[cid]={"summary":result["summary"],"metrics_separate_units":metrics,"signed_feedback_diagnostic":signed,"projection":projection_report(a),"checks":checks,
            "response_extrema":{name:fix.trim.response_extrema(v) for name,v in {"reference":observed,**bases,"candidate":p}.items()},
            "first_above_peak_s":next((t["time_s"] for t in result["ticks"] if t["envelope_above_peak"]),None),
            "current_peak_guard_transitions":[{"time_s":t["time_s"],"above_peak":t["envelope_above_peak"]} for i,t in enumerate(result["ticks"]) if i==0 or t["envelope_above_peak"]!=result["ticks"][i-1]["envelope_above_peak"]],"full_result_path":str(path)}
        for name,v in {"reference":observed,**bases,"candidate":p}.items():fields.save_arrays(arrays,cid+"__"+name,v)
        arrays.update({cid+"__ticks__"+k:v for k,v in a.items()})
        plot(cid,(("Jari reference",observed,"#243f56"),("Cached ControlFix",bases["control_fix"],"#96999c"),("Cached geometry",bases["geometry"],"#c4933f"),("EnvelopeControl",p,"#168477"),("5-point signed diagnostic",reference_feedback,"#243f56")))
        print(json.dumps({"completed":cid,"summary":result["summary"],"first_peak_s":reports[cid]["first_above_peak_s"],"checks":checks}),flush=True)
    return reports,arrays,new,recovered


def native_decision(reports):
    tolerances={"speed_mps":1.,"alpha_deg":.25,"normal_specific_load_g":.15,"body_pitch_deg":.3,"body_yaw_deg":.3,"body_yaw_rate_deg_s":2.,"body_pitch_rate_deg_s":1.,"feedback_pitch_G":.15,"feedback_yaw_G":.15}
    channels={}
    for cid,r in reports.items():
        for name in ("control_fix","geometry"):
            for w in ("platform","unloading","early_0_3p5"):
                for k,v in r["metrics_separate_units"][name][w]["metrics"].items():
                    channels[cid+"__"+name+"__"+w+"__"+k]=v["new"]["rmse"]<=1.25*v["old"]["rmse"]+tolerances[k]
    r=reports["jari_259_60"]["metrics_separate_units"]["geometry"]["early_0_3p5"]["metrics"]
    improvement={k:v["new"]["rmse"]<.85*v["old"]["rmse"] for k,v in r.items() if k in ("speed_mps","alpha_deg","normal_specific_load_g")}
    return {"separate_no_major_worsening":channels,"60_material_improvement_vs_geometry":improvement,
        "expand_native":all(channels.values()) and any(improvement.values()),"aggregate_score":False}


def main():
    before=snapshot();OUT.mkdir(parents=True,exist_ok=True)
    inputs=dec.audit.inverse.source_inputs()
    coefficients=drag.DragCoefficients(**json.loads((drag.OUTPUT/"fit.json").read_text())["coefficients"])
    reports,arrays,new,recovered=run_jari(inputs,coefficients)
    decision=native_decision(reports)
    if decision["expand_native"]:raise NotImplementedError("Native expansion required after favourable three-case outcome")
    verification={"protected_prior_size_mtime_unchanged":before==snapshot(),"all_arrays_finite":all(np.isfinite(v).all() for v in arrays.values()),
        "unique_candidate":1,"unique_free_runs":len(reports),"new_free_runs_this_invocation":new,"recovered_this_invocation":recovered,
        "baseline_integrations":0,"fits_scans_API_calls":0,"source_state_after_creation":False,"controller_hz":48,"plant_substeps":4}
    if not verification["protected_prior_size_mtime_unchanged"] or not verification["all_arrays_finite"]:raise AssertionError(verification)
    np.savez_compressed(OUT/"trajectories.npz",**arrays)
    slew.write_json(OUT/"results.json",{"candidate":"EnvelopeControl: current peak guard, original G PID/memory, geometry direction",
        "peak_local_rad":PEAK_S,"peak_linear_force_rad":PEAK_FORCE,"cases":reports,"native_decision":decision,"verification":verification,
        "limits":["Game-only D-level request/cap hypotheses, not native-game identification.","Input projection is after originalS110 and does not replace its committed state.",
            "Current cap guard can leave/reenter; original integral-transport latch remains distinct.","Signed reference feedback is noncausal diagnostic only; no reference state/control enters free runtime."]})
    print(json.dumps({"complete":True,"native":decision,"verification":verification}),flush=True)


if __name__=="__main__":main()
